"""FastAPI serving layer — Kindle Recommender.

v0 (deployable now): popularity from Postgres with temperature sampling,
similar-items lookup, latency tracking. Personalized endpoints return 501
until notebooks 03-05 export their artifacts to ARTIFACTS_DIR.

Full online flow (after 04-05): Postgres history -> user tower forward pass
-> FAISS top-100 -> LightGBM re-rank -> top-10.

Swagger docs at /docs (project requirement).
"""
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://recsys:recsys@localhost:5432/kindle")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
TEMPERATURE = float(os.environ.get("POP_TEMPERATURE", "0.5"))  # higher = more variety on reload
POOL_SIZE = 100  # popularity candidates to sample top-k from

STATE: dict = {"models_loaded": False}
# rolling window so memory stays bounded under sustained load
LATENCIES: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=8)
    try:
        from model_service import RecommenderService
        if RecommenderService.available(ARTIFACTS_DIR):
            STATE["service"] = RecommenderService(ARTIFACTS_DIR)
            STATE["models_loaded"] = True
            # item_idx -> category map for the x_cat_match feature (one query, kept in memory)
            with app.state.pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT item_idx, category FROM items")
                STATE["cat_of"] = dict(cur.fetchall())
            print("Personalized pipeline loaded: two-tower + FAISS + LightGBM")
        else:
            print("Artifacts incomplete -> serving v0 (popularity only)")
    except Exception as e:  # missing deps or corrupt artifacts -> degrade gracefully
        print(f"Model loading failed ({e}) -> serving v0 (popularity only)")
    yield
    app.state.pool.close()


app = FastAPI(title="Kindle Recommender API", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def track_latency(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = 1000 * (time.perf_counter() - t0)
    LATENCIES[request.url.path.split("/")[1] or "root"].append(ms)
    response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
    return response


def item_rows_to_json(rows) -> list[dict]:
    return [
        {"item_idx": r[0], "parent_asin": r[1], "title": r[2],
         "image_url": r[3], "category": r[4], "avg_rating": r[5]}
        for r in rows
    ]


def fetch_items(conn, item_idxs: list[int]) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT item_idx, parent_asin, title, image_url, category, avg_rating "
            "FROM items WHERE item_idx = ANY(%s)", (item_idxs,))
        by_idx = {r[0]: r for r in cur.fetchall()}
    return item_rows_to_json([by_idx[i] for i in item_idxs if i in by_idx])


def popularity_sample(conn, k: int, exclude: set[int] | None = None) -> list[int]:
    """Temperature sampling over the popularity head -> recs vary on reload."""
    with conn.cursor() as cur:
        cur.execute("SELECT item_idx, score FROM popularity ORDER BY score DESC LIMIT %s",
                    (POOL_SIZE + len(exclude or ()),))
        cand = [(i, s) for i, s in cur.fetchall() if not exclude or i not in exclude][:POOL_SIZE]
    if not cand:
        raise HTTPException(503, "popularity table is empty — run scripts/populate_db.py")
    rng = np.random.default_rng()  # per-request: numpy Generators are not thread-safe
    idxs = np.array([c[0] for c in cand])
    scores = np.array([c[1] for c in cand], dtype=float)
    p = scores ** (1.0 / TEMPERATURE)
    p /= p.sum()
    chosen = rng.choice(idxs, size=min(k, len(idxs)), replace=False, p=p)
    return [int(i) for i in chosen]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": STATE["models_loaded"]}


@app.get("/metrics")
def metrics():
    """Average response time per endpoint group (reported in README/ANALYSIS)."""
    return {ep: {"avg_ms": round(float(np.mean(v)), 1), "n": len(v)}
            for ep, v in LATENCIES.items() if v}


@app.get("/recommend/popular")
def recommend_popular(k: int = 10):
    """Logged-out homepage: training-set popularity, temperature-sampled."""
    with app.state.pool.connection() as conn:
        items = fetch_items(conn, popularity_sample(conn, k))
    return {"model": "Popular right now", "items": items}


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: str, k: int = 10, mmr_lambda: float | None = None):
    """Logged-in homepage: two-tower retrieval -> LightGBM re-rank.

    Unknown users and v0 (models not exported yet) fall back to popularity,
    personalized by excluding the user's already-read items.
    """
    with app.state.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_idx FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        if row is None:
            items = fetch_items(conn, popularity_sample(conn, k))
            return {"model": "Popular right now (unknown user fallback)", "items": items}
        user_idx = row[0]
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute("SELECT item_idx, is_positive, period FROM interactions "
                        "WHERE user_idx = %s ORDER BY ts", (user_idx,))
            rows = cur.fetchall()
        db_ms = 1000 * (time.perf_counter() - t0)
        seen = {r[0] for r in rows}  # ALL periods: never re-recommend an already-read book
        # user-tower history = test-period positives only (recent activity the model never
        # trained on — the deployment simulation required by the spec)
        history = [r[0] for r in rows if r[1] and r[2] == "test"]

        if not STATE["models_loaded"] or not history:
            items = fetch_items(conn, popularity_sample(conn, k, exclude=seen))
            label = ("Popularity (v0 — models not deployed yet)" if not STATE["models_loaded"]
                     else "Popular right now (no positive history)")
            return {"model": label, "items": items}

        # Real-time flow: history -> user tower -> FAISS top-100 -> LightGBM -> top-k
        rec_idx, timings = STATE["service"].recommend(
            user_idx, history, seen, STATE.get("cat_of", {}), k=k, mmr_lambda=mmr_lambda)
        timings["db_history"] = db_ms
        items = fetch_items(conn, rec_idx)
        label = "Two-tower + LightGBM" + (f" + MMR (λ={mmr_lambda})" if mmr_lambda is not None else "")
        return {"model": label, "items": items,
                "timings_ms": {c: round(v, 1) for c, v in timings.items()}}


@app.get("/similar/{parent_asin}")
def similar_items(parent_asin: str, k: int = 10):
    """Item page: precomputed cosine neighbors on learned item embeddings."""
    with app.state.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT item_idx FROM items WHERE parent_asin = %s", (parent_asin,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(404, f"unknown item {parent_asin}")
            cur.execute(
                "SELECT neighbor_idx FROM similar_items WHERE item_idx = %s "
                "ORDER BY score DESC LIMIT %s", (row[0], k))
            neigh = [r[0] for r in cur.fetchall()]
        if not neigh:
            raise HTTPException(501, "similarities not computed yet (exported by notebook 04)")
        items = fetch_items(conn, neigh)
    return {"model": "Item-item cosine (two-tower embeddings)", "items": items}


@app.get("/because-you-liked/{user_id}")
def because_you_liked(user_id: str, k: int = 10):
    """Homepage row: pick one liked item from history, show its neighbors."""
    with app.state.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT i.item_idx, it.title FROM interactions i JOIN items it USING (item_idx) "
                "JOIN users u USING (user_idx) WHERE u.user_id = %s AND i.is_positive "
                "ORDER BY i.ts DESC LIMIT 20", (user_id,))
            history = cur.fetchall()
        if not history:
            raise HTTPException(404, f"no positive history for user {user_id}")
        anchor_idx, anchor_title = history[np.random.default_rng().integers(len(history))]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT neighbor_idx FROM similar_items WHERE item_idx = %s "
                "ORDER BY score DESC LIMIT %s", (anchor_idx, k))
            neigh = [r[0] for r in cur.fetchall()]
        if not neigh:
            raise HTTPException(501, "similarities not computed yet (exported by notebook 04)")
        items = fetch_items(conn, neigh)
    return {"model": "Item-item cosine (two-tower embeddings)",
            "anchor": {"item_idx": anchor_idx, "title": anchor_title}, "items": items}
