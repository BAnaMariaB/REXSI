"""Online recommendation service: two-tower user embedding -> FAISS -> LightGBM.

Loaded once at API startup. Expects in ARTIFACTS_DIR (copied from the
notebook exports artifacts_two_tower/ + artifacts_ranker/):
    two_tower.pt, config.json, faiss.index, item_embeddings.npy,
    lgbm_ranker.txt, user_features.parquet, item_features.parquet, feature_config.json

Returns per-component timings so /metrics and ANALYSIS.md can report the
latency breakdown (DB fetch is timed by the caller).
"""
import json
import os
import time

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DAY_MS = 86_400_000


class TwoTower(nn.Module):
    """Must match notebook 04 exactly (state dict keys)."""

    def __init__(self, n_items: int, dim: int, hidden: int, gamma: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.user_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.item_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.gamma = gamma  # recency decay; 1.0 = plain masked mean (v1 artifacts)

    def user_tower(self, hist, mask):
        # Recency-weighted pooling — must match notebook 04's forward pass exactly.
        pos = mask.cumsum(1)
        n = mask.sum(1, keepdim=True)
        dist = (n - pos).clamp(min=0)
        w = (self.gamma ** dist) * mask
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        pooled = (self.emb(hist) * w.unsqueeze(-1)).sum(1)
        return F.normalize(self.user_mlp(pooled), dim=-1)


class RecommenderService:
    REQUIRED = ["two_tower.pt", "config.json", "faiss.index", "item_embeddings.npy",
                "lgbm_ranker.txt", "user_features.parquet", "item_features.parquet",
                "feature_config.json"]

    @classmethod
    def available(cls, art_dir: str) -> bool:
        return all(os.path.exists(os.path.join(art_dir, f)) for f in cls.REQUIRED)

    def __init__(self, art_dir: str):
        with open(os.path.join(art_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.tower = TwoTower(self.cfg["n_items"], self.cfg["dim"], self.cfg["hidden"],
                              gamma=self.cfg.get("gamma", 1.0))
        self.tower.load_state_dict(
            torch.load(os.path.join(art_dir, "two_tower.pt"), map_location="cpu"))
        self.tower.eval()

        self.index = faiss.read_index(os.path.join(art_dir, "faiss.index"))
        self.item_emb = np.load(os.path.join(art_dir, "item_embeddings.npy"))
        self.booster = lgb.Booster(model_file=os.path.join(art_dir, "lgbm_ranker.txt"))

        with open(os.path.join(art_dir, "feature_config.json")) as f:
            fc = json.load(f)
        self.features: list[str] = fc["features"]
        self.user_feat = (pd.read_parquet(os.path.join(art_dir, "user_features.parquet"))
                          .set_index("user_idx"))
        self.item_feat = (pd.read_parquet(os.path.join(art_dir, "item_features.parquet"))
                          .set_index("item_idx").reindex(range(self.cfg["n_items"])))

    # ---------------------------------------------------------------- online
    @torch.no_grad()
    def _embed_user(self, history_items: list[int]) -> np.ndarray:
        max_hist, pad = self.cfg["max_hist"], self.cfg["pad"]
        h = history_items[-max_hist:]
        hist = torch.full((1, max_hist), pad, dtype=torch.long)
        hist[0, :len(h)] = torch.tensor(h, dtype=torch.long)
        mask = (hist != pad).float()
        return self.tower.user_tower(hist, mask).numpy().astype("float32")

    def _feature_matrix(self, user_idx: int, history: list[int],
                        cand: np.ndarray, scores: np.ndarray,
                        cat_of: dict[int, str]) -> pd.DataFrame:
        if user_idx in self.user_feat.index:
            uf = self.user_feat.loc[user_idx]
        else:  # user unseen at feature-build time: neutral user features
            uf = pd.Series(0.0, index=self.user_feat.columns)
        itf = self.item_feat.loc[cand]
        last5 = history[-5:]
        max_sim = (self.item_emb[cand] @ self.item_emb[last5].T).max(axis=1) \
            if last5 else np.zeros(len(cand))
        ucats = {cat_of.get(i) for i in history[-20:]} - {None}
        X = pd.DataFrame({
            "u_n_interactions": uf["u_n_interactions"],
            "u_avg_rating": uf["u_avg_rating"],
            "u_days_since_last": uf["u_days_since_last"],
            "u_n_recent_30d": uf["u_n_recent_30d"],
            "i_popularity": itf["i_popularity"].to_numpy(),
            "i_avg_rating": itf["i_avg_rating"].to_numpy(),
            "i_days_since_first": itf["i_days_since_first"].to_numpy(),
            "i_n_recent_30d": itf["i_n_recent_30d"].to_numpy(),
            "x_retrieval_score": scores,
            "x_retrieval_rank": np.arange(len(cand), dtype="float32"),
            "x_max_sim_last5": max_sim,
            "x_cat_match": [float(cat_of.get(int(i)) in ucats) for i in cand],
        })
        return X[self.features].astype("float32")

    def _mmr_order(self, cand: np.ndarray, preds: np.ndarray,
                   mmr_lambda: float, k: int) -> list[int]:
        """Greedy Maximal Marginal Relevance over the ranked candidates (see notebook 06).

        Returns indices into `cand`. lambda=1 -> pure relevance (ranker order);
        lambda=0 -> pure diversity. Similarity is cosine on the stored item
        embeddings; relevance is the ranker score min-max normalized per request
        so it shares the [0, 1] scale with similarity.
        """
        rel = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
        emb = self.item_emb[cand].astype("float32")
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sim = emb @ emb.T
        remaining = list(range(len(cand)))
        selected: list[int] = []
        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda c: rel[c])
            else:
                best = max(remaining, key=lambda c: mmr_lambda * rel[c]
                           - (1 - mmr_lambda) * sim[c, selected].max())
            selected.append(best)
            remaining.remove(best)
        return selected

    def recommend(self, user_idx: int, history: list[int], seen: set[int],
                  cat_of: dict[int, str], k: int = 10, k_retrieve: int = 100,
                  temperature: float = 0.3, rng=None,
                  mmr_lambda: float | None = None) -> tuple[list[int], dict]:
        """history: item_idx chronological (positives). Returns (items, timings_ms).

        mmr_lambda: if set, re-rank the LightGBM top candidates with MMR diversity
        (notebook 06) instead of temperature sampling — deterministic, trades a little
        relevance for a more varied list. None (default) keeps the original behavior.
        """
        rng = rng or np.random.default_rng()
        timings = {}

        t0 = time.perf_counter()
        u = self._embed_user(history)
        timings["user_tower"] = 1000 * (time.perf_counter() - t0)

        t0 = time.perf_counter()
        scores, idx = self.index.search(u, k_retrieve + len(seen) + 10)
        keep = [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1 and i not in seen][:k_retrieve]
        cand = np.array([c[0] for c in keep])
        cscores = np.array([c[1] for c in keep], dtype="float32")
        timings["faiss"] = 1000 * (time.perf_counter() - t0)

        t0 = time.perf_counter()
        X = self._feature_matrix(user_idx, history, cand, cscores, cat_of)
        preds = self.booster.predict(X)
        timings["lightgbm"] = 1000 * (time.perf_counter() - t0)

        if mmr_lambda is not None:
            t0 = time.perf_counter()
            chosen = self._mmr_order(cand, preds, mmr_lambda, k)
            timings["mmr"] = 1000 * (time.perf_counter() - t0)
            return [int(cand[c]) for c in chosen], timings

        # Temperature sampling from the top-30 ranked candidates -> recs vary on reload
        order = np.argsort(-preds)[:max(30, k)]
        p = np.exp((preds[order] - preds[order].max()) / max(temperature, 1e-6))
        p /= p.sum()
        chosen = rng.choice(order, size=min(k, len(order)), replace=False, p=p)
        chosen = sorted(chosen, key=lambda c: -preds[c])  # display in ranker order
        return [int(cand[c]) for c in chosen], timings
