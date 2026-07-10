"""Shared ranking metrics — used by notebooks 02-05 and reported everywhere.

All functions take:
  recs:    dict {user_idx: list of recommended item_idx, ranked}
  truth:   dict {user_idx: set of relevant (positive) item_idx}
Users present in `truth` but absent from `recs` count as zero.
"""
import numpy as np


def recall_at_k(recs: dict, truth: dict, k: int) -> float:
    scores = []
    for u, rel in truth.items():
        if not rel:
            continue
        top = recs.get(u, [])[:k]
        scores.append(len(set(top) & rel) / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(recs: dict, truth: dict, k: int) -> float:
    scores = []
    for u, rel in truth.items():
        if not rel:
            continue
        top = recs.get(u, [])[:k]
        dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(top) if it in rel)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(recs: dict, n_items: int, k: int) -> float:
    """Fraction of the catalog recommended at least once across all users."""
    seen = set()
    for r in recs.values():
        seen.update(r[:k])
    return len(seen) / n_items if n_items else 0.0


def evaluate(recs: dict, truth: dict, n_items: int, ks=(10, 20, 50)) -> dict:
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(recs, truth, k)
    out["NDCG@10"] = ndcg_at_k(recs, truth, 10)
    out["Coverage@10"] = catalog_coverage(recs, n_items, 10)
    return out
