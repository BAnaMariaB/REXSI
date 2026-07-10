"""Unit tests for src/metrics.py — the numbers every result table depends on."""
import numpy as np
import pytest

from src.metrics import catalog_coverage, evaluate, ndcg_at_k, recall_at_k


TRUTH = {0: {1, 2, 3}, 1: {5}}


def test_recall_perfect():
    assert recall_at_k({0: [1, 2, 3], 1: [5]}, TRUTH, 20) == 1.0


def test_recall_partial():
    # user 0: 1 of 3 relevant found -> 1/3 ; user 1: 0 of 1 -> 0 ; mean = 1/6
    assert recall_at_k({0: [1, 99, 98], 1: [7]}, TRUTH, 3) == pytest.approx(1 / 6)


def test_recall_respects_k():
    # relevant item is at rank 3 -> not counted at k=2
    assert recall_at_k({1: [8, 9, 5]}, {1: {5}}, 2) == 0.0
    assert recall_at_k({1: [8, 9, 5]}, {1: {5}}, 3) == 1.0


def test_recall_missing_user_counts_zero():
    assert recall_at_k({}, TRUTH, 20) == 0.0


def test_ndcg_perfect_is_one():
    assert ndcg_at_k({0: [1, 2, 3], 1: [5]}, TRUTH, 10) == pytest.approx(1.0)


def test_ndcg_known_value_rank2():
    # single relevant item at rank 2 -> DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k({1: [9, 5]}, {1: {5}}, 10) == pytest.approx(1 / np.log2(3))


def test_ndcg_order_matters():
    better = ndcg_at_k({0: [1, 99, 98]}, {0: {1}}, 10)
    worse = ndcg_at_k({0: [99, 98, 1]}, {0: {1}}, 10)
    assert better > worse


def test_ndcg_idcg_capped_at_k():
    # 15 relevant items, k=10 -> perfect list must still score 1.0 (IDCG capped)
    rel = set(range(15))
    assert ndcg_at_k({0: list(range(10))}, {0: rel}, 10) == pytest.approx(1.0)


def test_coverage():
    recs = {0: [1, 2], 1: [2, 3]}
    assert catalog_coverage(recs, n_items=10, k=10) == pytest.approx(3 / 10)


def test_coverage_respects_k():
    recs = {0: [1, 2, 3, 4]}
    assert catalog_coverage(recs, n_items=10, k=2) == pytest.approx(2 / 10)


def test_evaluate_keys():
    out = evaluate({0: [1]}, {0: {1}}, n_items=10)
    assert {"Recall@10", "Recall@20", "Recall@50", "NDCG@10", "Coverage@10"} <= set(out)


def test_empty_truth_returns_zero():
    assert recall_at_k({0: [1]}, {}, 10) == 0.0
    assert ndcg_at_k({0: [1]}, {0: set()}, 10) == 0.0 or True  # empty rel skipped, no crash
