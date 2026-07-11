# ANALYSIS

Results and discussion only - code lives in `notebooks/`, serving in `app/`. All metrics are computed
on the test period of a global time-based split (train < t1 <= val < t2 <= test), positives = rating
>= 4, seen-train items excluded, evaluation protocol identical for every model.

## 1. Ablation: retrieval-only vs + ranking

Same top-100 candidates from the two-tower; the only difference is the ordering of the list.

| Pipeline | Recall@20 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| Two-tower (retrieval only) | 0.0572 | 0.0215 | 0.2996 |
| Two-tower + LightGBM (LambdaRank) | **0.0667** | **0.0278** | 0.2885 |
| change | +17% | **+29%** | -4% |

The ranker cannot add candidates (Recall@100 is fixed by retrieval), so its entire contribution is
reordering, which is why the gain concentrates in NDCG@10: items the retriever placed at ranks 20-100
get promoted into the served top-10 when their features (popularity prior, rating quality, similarity
to the user's recent reads) support it. Coverage dips a little (-4%) as the popularity-family features
pull the served lists toward the head; the effect is small and the ranker is not collapsing onto
bestsellers.

## 2. Cold-start analysis

Performance by user activity level (interactions in the training period; k-core minimum is 5).

| User group | #Users | Recall@20 (retrieval) | Recall@20 (+ranker) | NDCG@10 (+ranker) |
|---|---|---|---|---|
| cold (5-10) | 9,272 | 0.0653 | 0.0775 | 0.0299 |
| medium (11-50) | 10,796 | 0.0632 | 0.0711 | 0.0311 |
| heavy (>50) | 5,116 | 0.0364 | 0.0402 | 0.0187 |

Cold users score best and heavy users worst, and the ranker's lift is largest for cold users. Heavy
users have more test positives, so Recall@20 has a bigger denominator to fill from the same 20 slots
(the fairness section below generalizes this across every model). Because the user tower consumes
history rather than a user-ID embedding, users who appear after training still receive personalized
retrieval as soon as they have interactions - a structural cold-start advantage over MF-BPR, which can
only serve users present in its training matrix. Users with no positive history at all fall back to
popularity in the API.

## 2b. Fairness analysis across activity groups (bonus)

Extends the cold-start breakdown above from "retrieval vs +ranker" to **every model in the pipeline**
(Random, Popularity, MF-BPR, Two-tower, Two-tower + LightGBM), same three activity buckets, same
evaluation protocol (`src/metrics.py`, test-period positives, seen-train items excluded). Deterministic
top-k by score for every model (no temperature sampling for the full pipeline, unlike the serving API).

| User group | #Users | Recall@20 (Random) | Recall@20 (Popularity) | Recall@20 (MF-BPR) | Recall@20 (Two-tower) | Recall@20 (Two-tower + LightGBM) | NDCG@10 (Two-tower + LightGBM) |
|---|---|---|---|---|---|---|---|
| cold (5-10) | 9,272 | 0.0000 | 0.0189 | 0.0265 | 0.0623 | 0.0788 | 0.0310 |
| medium (11-50) | 10,796 | 0.0002 | 0.0105 | 0.0279 | 0.0622 | 0.0725 | 0.0315 |
| heavy (>50) | 5,116 | 0.0001 | 0.0055 | 0.0210 | 0.0353 | 0.0391 | 0.0174 |

(25,184 of the 34,191 evaluable test users fall into a bucket; the remaining ~9,000 have fewer than 5
train-period interactions despite passing the global 5-core filter - see the k-core-before-split note
in notebook 01 section 3 - and are excluded from this breakdown for the same reason cold-start tables
usually start at the k-core floor.)

**Every personalization tier beats its predecessor within every bucket** - the ranking Random <
Popularity < MF-BPR < Two-tower < Two-tower+LightGBM from the headline results table holds
group-by-group, so no bucket is quietly propping up the aggregate numbers.

**The counter-intuitive part: recall is not monotonically increasing in activity - heavy users score
lowest**, on every model. This looks at first like the system is failing its most engaged users, but
the average number of positive test items per user tells a different story:

| Group | Avg. positive test items / user |
|---|---|
| cold (5-10) | 1.68 |
| medium (11-50) | 2.50 |
| heavy (>50) | 5.10 |

Recall@20 is (hits)/(relevant items), and heavy users simply have ~3x more relevant items to find
inside the same 20 slots - a harder target by construction, not evidence the model under-serves them.
A heavy user's absolute hit count is typically similar to or higher than a cold user's; it is just a
smaller fraction of a bigger denominator. This is the same mechanical effect the cold-start section
above is designed to surface, generalized across the full model family rather than just retrieval vs.
+ranker.

**Fairness takeaway:** the model does not add extra unfairness beyond what the metric's own
denominator already implies - the ranking of models is stable across activity levels, and the one
group-level gap that exists (heavy users' lower Recall@20) is explained by a measurement artifact
rather than the personalization pipeline itself favoring light users. The structural cold-start
advantage of the two-tower's history-based user representation (section 2) is the more actionable
fairness lever: it is what lets the cold bucket score competitively at all, rather than being routed to
the flat popularity fallback the way a user-ID-embedding model (MF-BPR) would require for unseen users.

## 3. Feature importance (LightGBM, gain)

| Rank | Feature | Gain |
|---|---|---|
| 1 | i_days_since_first | 75,126 |
| 2 | x_max_sim_last5 | 32,524 |
| 3 | x_retrieval_score | 31,804 |
| 4 | x_retrieval_rank | 25,345 |
| 5 | i_n_recent_30d | 25,282 |
| 6 | u_days_since_last | 23,784 |
| 7 | i_popularity | 16,456 |
| 8 | i_avg_rating | 11,037 |
| 9 | u_n_recent_30d | 9,956 |
| 10 | u_n_interactions | 7,001 |
| 11 | u_avg_rating | 4,680 |
| 12 | x_cat_match | 1,904 |

We expected the retrieval score and popularity to lead. They do not. Item age (`i_days_since_first`)
dominates by a wide margin, then similarity to the user's last five reads (`x_max_sim_last5`) and the
retriever's own score, roughly tied. Popularity is only 7th. The ranker leans on item maturity,
recent-read similarity, and the retriever's ordering more than on raw popularity. `x_cat_match` is
close to dead weight (gain ~1.9k) and is the obvious feature to drop; we kept all 12 for the ablation's
interpretability.

## 4. Latency breakdown (measured on the deployed system)

Single request, `/recommend/user/{id}`, Docker on a MacBook Air (CPU-only serving):

| Component | Time (ms) |
|---|---|
| PostgreSQL history fetch | 0.3 |
| User tower forward pass (PyTorch, dim 128) | 3.0 |
| FAISS top-100 (IndexFlatIP, 120K items) | 5.6 |
| LightGBM re-rank (100 candidates x 12 features) | 5.9 |
| **Total model path** | **~15** |

That is well under interactive thresholds, on CPU with an exact index. The Flat index is affordable at
this catalog size; IVF only becomes necessary at millions of items. The two heaviest stages, FAISS and
LightGBM, are batchable if throughput ever mattered.

## 5. Limitations

**Temporal drift between tuning and serving.** All hyperparameters and early stopping are selected on
the validation window, which immediately follows training; the test window lies further in the future.
We consistently observed val->test decay (e.g., two-tower val Recall@20 ~0.09 vs test 0.057), and a
1-epoch model briefly matched a fully-trained one on test - evidence that "validation-optimal is not
test-optimal" under drift. With more time we would (a) retrain on train+val before final deployment, as
production systems do, (b) evaluate with a rolling-origin scheme (multiple chronological folds) instead
of a single split, and (c) add trend features so the ranker can react to drift directly.

Secondary limitations worth noting: k-core filtering biases the dataset toward active users (mitigated
but not removed by the cold-start breakdown above), and our positives threshold (rating >= 4) discards
disliked-but-informative interactions from retrieval training.

## 6. Final comparison table

| Model | Recall@20 | Recall@50 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|
| Random | 0.0001 | 0.0004 | 0.0000 | 0.9415 |
| Popularity | 0.0141 | 0.0242 | 0.0053 | 0.0002 |
| MF-BPR (from scratch, SGD, popularity-sampled negatives) | 0.0263 | 0.0472 | 0.0098 | 0.2219 |
| Two-tower (history tower, in-batch CE + log-Q, recency pooling) | 0.0572 | 0.0967 | 0.0215 | 0.2996 |
| **Two-tower + LightGBM (full pipeline)** | **0.0667** | **0.1072** | **0.0278** | 0.2885 |

Reading: every personalization stage beats its predecessor on accuracy while recommending orders of
magnitude more of the catalog than the popularity baseline (0.02% -> ~29%). The full pipeline is
**4.7x popularity on Recall@20** and **5.2x on NDCG@10**.

### Appendix: two-tower v1 -> v2 (what the improvements bought)

| Version | Recall@20 | Recall@50 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|
| v1 - mean pooling, dim 64 | 0.0531 | 0.0915 | 0.0207 | 0.383 |
| v2 - recency pooling (gamma=0.85), dim 128 | 0.0572 | 0.0967 | 0.0215 | 0.2996 |

Recency weighting concentrates recommendations on the user's current taste. Log-Q sampling-bias
correction was enabled in both versions (bonus item).

### Appendix: training-dynamics findings (MF-BPR)

Two issues diagnosed from a frozen loss curve (~ ln 2), both now documented in notebook 03: tiny init
(std 0.01) leaves the factor product gradient-starved under plain SGD, and a mean-reduced loss makes
SGD's effective per-triplet step lr/batch_size (~6e-6) - Adam's per-parameter rescaling would have
masked both. Fixes: std 0.1 init + sum-reduced loss. The subsequent lr grid showed the full textbook
spectrum: undertrained (0.05), optimal (0.1), and overfitting with val decay after epoch 12 (0.2).
