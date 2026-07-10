"""Populate PostgreSQL from notebook artifacts (offline batch step).

Loads: users, items, test-period interactions (= recent history the models
never trained on), popularity scores. Idempotent: truncates then reloads.

Usage (with `docker compose up postgres` running):
    pip install psycopg[binary] pandas pyarrow
    python populate_db.py --data-dir ./data
"""
import argparse
import io
import json
import os

import pandas as pd
import psycopg

DB_URL = os.environ.get("DATABASE_URL", "postgresql://recsys:recsys@localhost:5432/kindle")


def copy_df(cur, df: pd.DataFrame, table: str, cols: list[str]) -> None:
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False)
    buf.seek(0)
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')") as cp:
        cp.write(buf.read())


def main(data_dir: str) -> None:
    with open(os.path.join(data_dir, "id_mappings.json")) as f:
        m = json.load(f)
    users = pd.DataFrame(m["user2idx"].items(), columns=["user_id", "user_idx"])
    items = pd.read_parquet(os.path.join(data_dir, "items.parquet"))
    pop = pd.read_parquet(os.path.join(data_dir, "popularity.parquet"))
    # ALL periods: test rows drive the user tower (recent, unseen-by-model history);
    # train/val rows are required for the seen-filter (never re-recommend a read book).
    parts = []
    for period in ("train", "val", "test"):
        p = pd.read_parquet(os.path.join(data_dir, f"{period}.parquet"))
        p = p.dropna(subset=["user_idx", "item_idx"]).astype({"user_idx": int, "item_idx": int})
        p["period"] = period
        parts.append(p)
    inter = pd.concat(parts, ignore_index=True)

    # Optional: similar_items from notebook 04 (two-tower embeddings)
    sim_path = os.path.join(data_dir, "artifacts_two_tower", "similar_items.parquet")
    sim = pd.read_parquet(sim_path) if os.path.exists(sim_path) else None

    # FK safety: references to items that lacked metadata
    valid = set(items.item_idx)
    pop = pop[pop.item_idx.isin(valid)]
    inter = inter[inter.item_idx.isin(valid)]
    if sim is not None:
        sim = sim[sim.item_idx.isin(valid) & sim.neighbor_idx.isin(valid)]

    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        # migrate pre-existing volumes created before the period column existed
        cur.execute("ALTER TABLE interactions ADD COLUMN IF NOT EXISTS period TEXT")
        cur.execute("TRUNCATE interactions, popularity, similar_items, items, users CASCADE")
        copy_df(cur, users, "users", ["user_idx", "user_id"])
        copy_df(cur, items, "items",
                ["item_idx", "parent_asin", "title", "image_url", "category", "avg_rating", "rating_count"])
        copy_df(cur, pop, "popularity", ["item_idx", "score"])
        inter_out = inter.rename(columns={"timestamp": "ts"})
        copy_df(cur, inter_out, "interactions",
                ["user_idx", "item_idx", "rating", "ts", "is_positive", "period"])
        if sim is not None:
            copy_df(cur, sim, "similar_items", ["item_idx", "neighbor_idx", "score"])
            print("similar_items loaded (notebook 04 artifacts found)")
        else:
            print("similar_items skipped (run notebook 04 first, then rerun this script)")
        conn.commit()

        for t in ("users", "items", "popularity", "interactions", "similar_items"):
            cur.execute(f"SELECT count(*) FROM {t}")
            print(f"{t:13s}: {cur.fetchone()[0]:,} rows")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    main(ap.parse_args().data_dir)
