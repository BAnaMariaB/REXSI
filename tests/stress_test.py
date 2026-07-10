"""Stress test for the deployed stack (run against docker compose on localhost).

Fires a realistic mixed workload with concurrent clients and reports
throughput, error rate and latency percentiles per endpoint.

Usage (stack must be up and populated):
    uv run python tests/stress_test.py                          # default: 8 workers, 30s
    uv run python tests/stress_test.py --workers 20 --seconds 60
"""
import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
import requests

API = "http://localhost:8000"
DB = "postgresql://recsys:recsys@localhost:5432/kindle"


def fetch_ids(n_users: int = 200, n_items: int = 200):
    """Real IDs from the DB: active users (personalized path) and popular items (similar path)."""
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("""SELECT u.user_id FROM users u JOIN interactions i USING (user_idx)
                       GROUP BY u.user_id ORDER BY count(*) DESC LIMIT %s""", (n_users,))
        users = [r[0] for r in cur.fetchall()]
        cur.execute("""SELECT it.parent_asin FROM popularity p JOIN items it USING (item_idx)
                       ORDER BY p.score DESC LIMIT %s""", (n_items,))
        items = [r[0] for r in cur.fetchall()]
    return users, items


def worker(deadline: float, users, items, results, errors):
    rng = random.Random()
    s = requests.Session()
    while time.time() < deadline:
        r = rng.random()
        if r < 0.45:      # personalized homepage (the heavy path)
            name, url = "personalized", f"{API}/recommend/user/{rng.choice(users)}"
        elif r < 0.65:    # logged-out homepage
            name, url = "popular", f"{API}/recommend/popular"
        elif r < 0.80:    # item page
            name, url = "similar", f"{API}/similar/{rng.choice(items)}"
        elif r < 0.90:    # 'because you liked' row
            name, url = "because", f"{API}/because-you-liked/{rng.choice(users)}"
        else:             # unknown user -> fallback path
            name, url = "fallback", f"{API}/recommend/user/stress_test_ghost_{rng.randint(0, 10**9)}"
        t0 = time.perf_counter()
        try:
            resp = s.get(url, timeout=30)
            ms = 1000 * (time.perf_counter() - t0)
            if resp.status_code == 200:
                results.setdefault(name, []).append(ms)
            else:
                errors.setdefault(f"{name}:{resp.status_code}", []).append(url)
        except Exception as e:
            errors.setdefault(f"{name}:{type(e).__name__}", []).append(url)


def pct(v, p):
    return statistics.quantiles(v, n=100)[p - 1] if len(v) >= 100 else sorted(v)[int(len(v) * p / 100)]


def main(workers: int, seconds: int):
    users, items = fetch_ids()
    print(f"Loaded {len(users)} user ids, {len(items)} item asins from DB")
    print(f"Stress: {workers} concurrent clients for {seconds}s against {API}\n")

    # wait for the API to finish loading models (up to 90s), then verify
    h = None
    for attempt in range(45):
        try:
            h = requests.get(f"{API}/health", timeout=5).json()
            break
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                print("API not up yet (models loading at startup) — waiting...")
            time.sleep(2)
    if h is None:
        raise SystemExit("API never became reachable — check: docker compose -f app/docker-compose.yml logs api")
    print(f"health: {h}")
    if not h.get("models_loaded"):
        print("WARNING: models_loaded=false — you are stress-testing the popularity fallback only!")

    results: dict = {}
    errors: dict = {}
    deadline = time.time() + seconds
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, deadline, users, items, results, errors)
                   for _ in range(workers)]
    for f in futures:  # surface worker crashes instead of swallowing them
        if f.exception() is not None:
            print(f"WORKER CRASHED: {f.exception()!r}")

    total = sum(len(v) for v in results.values())
    n_err = sum(len(v) for v in errors.values())
    print(f"\n{'endpoint':14s} {'reqs':>6s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}")
    for name, v in sorted(results.items()):
        print(f"{name:14s} {len(v):6d} {pct(v,50):7.1f}ms {pct(v,95):7.1f}ms {pct(v,99):7.1f}ms {max(v):7.1f}ms")
    print(f"\nThroughput: {total/seconds:.1f} req/s | errors: {n_err} "
          f"({100*n_err/max(total+n_err,1):.2f}%)")
    for k, v in errors.items():
        print(f"  ERROR {k}: {len(v)}x  e.g. {v[0]}")
    if n_err == 0:
        print("No errors — system stable under load.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=30)
    a = ap.parse_args()
    main(a.workers, a.seconds)
