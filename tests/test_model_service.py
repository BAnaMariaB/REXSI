"""End-to-end test of the online pipeline with tiny synthetic artifacts.

Builds a miniature two-tower + FAISS index + LightGBM ranker in tmp_path,
then checks RecommenderService produces valid recommendations.
Requires torch/faiss/lightgbm (skipped where unavailable, e.g. CI without ML deps).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
faiss = pytest.importorskip("faiss")
lgb = pytest.importorskip("lightgbm")

from model_service import RecommenderService, TwoTower  # noqa: E402  (path via conftest)

N_ITEMS, DIM, HIDDEN, MAX_HIST = 50, 8, 16, 5
FEATURES = ["u_n_interactions", "u_avg_rating", "u_days_since_last", "u_n_recent_30d",
            "i_popularity", "i_avg_rating", "i_days_since_first", "i_n_recent_30d",
            "x_retrieval_score", "x_retrieval_rank", "x_max_sim_last5", "x_cat_match"]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    art = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(0)

    # two-tower + config
    tower = TwoTower(N_ITEMS, DIM, HIDDEN)
    torch.save(tower.state_dict(), art / "two_tower.pt")
    (art / "config.json").write_text(json.dumps(
        {"dim": DIM, "hidden": HIDDEN, "max_hist": MAX_HIST, "tau": 0.1,
         "n_items": N_ITEMS, "pad": N_ITEMS}))

    # item embeddings + FAISS
    emb = rng.normal(size=(N_ITEMS, DIM)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(art / "item_embeddings.npy", emb)
    index = faiss.IndexFlatIP(DIM)
    index.add(emb)
    faiss.write_index(index, str(art / "faiss.index"))

    # tiny LambdaRank booster on random data with the real feature schema
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    y = rng.integers(0, 2, 200)
    ds = lgb.Dataset(X, label=y, group=[20] * 10)
    booster = lgb.train({"objective": "lambdarank", "verbosity": -1, "min_data_in_leaf": 5},
                        ds, num_boost_round=5)
    booster.save_model(str(art / "lgbm_ranker.txt"))

    # feature tables
    pd.DataFrame({"user_idx": [0, 1], "u_n_interactions": [10.0, 3.0],
                  "u_avg_rating": [4.2, 3.9], "u_days_since_last": [1.0, 30.0],
                  "u_n_recent_30d": [5.0, 0.0]}).to_parquet(art / "user_features.parquet")
    pd.DataFrame({"item_idx": range(N_ITEMS),
                  "i_popularity": rng.integers(0, 100, N_ITEMS).astype(float),
                  "i_avg_rating": rng.uniform(3, 5, N_ITEMS),
                  "i_days_since_first": rng.uniform(0, 900, N_ITEMS),
                  "i_n_recent_30d": rng.integers(0, 10, N_ITEMS).astype(float),
                  }).to_parquet(art / "item_features.parquet")
    (art / "feature_config.json").write_text(json.dumps({"features": FEATURES, "t_ref": 0.0}))
    return str(art)


def test_available_detects_complete_artifacts(artifacts, tmp_path):
    assert RecommenderService.available(artifacts)
    assert not RecommenderService.available(str(tmp_path))  # empty dir


def test_recommend_shape_and_validity(artifacts):
    svc = RecommenderService(artifacts)
    history, seen = [1, 2, 3], {1, 2, 3}
    recs, timings = svc.recommend(0, history, seen, cat_of={}, k=10,
                                  rng=np.random.default_rng(0))
    assert len(recs) == 10
    assert len(set(recs)) == 10, "duplicate items in one response"
    assert not set(recs) & seen, "recommended an already-seen item"
    assert all(0 <= i < N_ITEMS for i in recs)
    assert {"user_tower", "faiss", "lightgbm"} <= set(timings)
    assert all(v >= 0 for v in timings.values())


def test_recommend_varies_on_reload(artifacts):
    svc = RecommenderService(artifacts)
    outs = {tuple(svc.recommend(0, [1, 2], {1, 2}, {}, k=5)[0]) for _ in range(5)}
    assert len(outs) > 1, "temperature sampling should vary between calls"


def test_unknown_user_gets_neutral_features(artifacts):
    svc = RecommenderService(artifacts)
    recs, _ = svc.recommend(999_999, [4, 5], {4, 5}, {}, k=5)  # not in user_features
    assert len(recs) == 5


def test_short_history_ok(artifacts):
    svc = RecommenderService(artifacts)
    recs, _ = svc.recommend(0, [7], {7}, {}, k=5)  # single-item history
    assert len(recs) == 5 and 7 not in recs
