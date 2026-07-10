# Kindle Recommender — Real-Time Recommendation System

Real-time recommendation system on **Amazon Reviews 2023 — Kindle Store** (subsampled to the 5M most recent interactions). From raw data to a serving webapp: MF-BPR and two-tower retrieval (PyTorch, from scratch), FAISS candidate retrieval, LightGBM (LambdaRank) re-ranking, served via FastAPI + Streamlit + PostgreSQL with Docker Compose.

![demo](assets/demo.gif)
<!-- TODO: record GIF of the webapp before defense -->

## Results (test period, time-based split)

| Model | Recall@20 | Recall@50 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|
| Random | 0.0001 | 0.0004 | 0.0000 | 0.942 |
| Popularity | 0.0141 | 0.0242 | 0.0053 | 0.0002 |
| MF-BPR | 0.0246 | 0.0461 | 0.0092 | 0.221 |
| Two-tower | 0.0554 | 0.0974 | 0.0215 | 0.305 |
| **Two-tower + LightGBM** | **0.0668** | **0.1078** | **0.0282** | 0.298 |

Full pipeline = **4.7× popularity on Recall@20, 5.3× on NDCG@10**, recommending ~30% of the catalog vs 0.02%.

**API response time:** ~15 ms model path (DB 0.3 + user tower 3.0 + FAISS 5.6 + LightGBM 5.9 — measured on the deployed stack; breakdown in [ANALYSIS.md](ANALYSIS.md), live averages at `/metrics`)

## Architecture

- **Offline (Colab notebooks):** data prep → train MF-BPR & two-tower → export item embeddings + FAISS index + LightGBM model → populate PostgreSQL.
- **Online (docker compose):** request → FastAPI fetches user history from Postgres → user-tower forward pass → FAISS top-100 → LightGBM re-rank → top-10.

```
notebooks/   01_data_preparation_eda → 02_baselines → 03_mf_bpr → 04_two_tower → 05_ranking
src/         shared code (metrics, data utils, models)
app/         api (FastAPI) · frontend (Streamlit) · db (Postgres init)
```

## Setup

1. Run notebooks 01–02 (CPU) then 03–05 (GPU). Artifacts land in the data dir (Drive on Colab, `./data` locally).
2. Build item metadata: `uv run python app/scripts/build_items.py --data-dir <data>` (one-time ~2.5GB download).
3. Start the stack: `cd app && docker compose up --build`.
4. Load the database: `uv run python app/scripts/populate_db.py --data-dir <data>`.
5. Streamlit at http://localhost:8501 · API docs at http://localhost:8000/docs · avg latency at http://localhost:8000/metrics.

After step 4 the app is fully deployable in **v0 mode** (popularity everywhere). Notebooks 04–05 export `app/artifacts/` (item embeddings, `faiss.index`, LightGBM model, `similar_items`) which unlock the personalized homepage and both similarity sections.

## Two-machine workflow (laptop → GPU PC)

Notebooks 03–05 have a `SMOKE = True/False` flag in their setup cell. With `SMOKE = True` they train tiny-but-valid models in minutes (auto-selects CUDA / Apple MPS / CPU), exporting real artifacts so the entire chain — notebook 05, `populate_db.py`, the personalized API, the frontend — can be validated on a laptop. On the GPU machine, set `SMOKE = False`, rerun 03–05, and swap the artifacts. Never report SMOKE numbers.

## Environment (uv)

```bash
uv sync                 # creates .venv with all training + dev dependencies (uv.lock pins versions)
uv run jupyter lab      # run the notebooks locally
```

## Tests

```bash
uv run pytest                                   # unit: metrics, notebooks, model service
API_TEST_URL=http://localhost:8000 uv run pytest tests/test_integration.py -v   # smoke: running stack
```

Covers: metric correctness (known NDCG/Recall values, k-cutoffs, edge cases), notebook integrity (valid JSON, compilable cells, narration present, **no random split anywhere**), the full online pipeline on synthetic artifacts (shape, no seen-item leaks, sampling variety, cold-user path), and live API smoke tests (health, Swagger, fallbacks, latency tracking).

## Dataset

Kindle Store, Amazon Reviews 2023 (McAuley Lab). 25.6M ratings → most recent 5M → 5-core filtered. Chosen for density (~4.6 ratings/user — readers are repeat users), which favors collaborative filtering and enables a real cold-start analysis. Subsampling and filtering decisions are documented in notebook 01.

## Who did what

Each member owns one core component (reproduction, analysis, extension, defense) — see [HANDOFF.md](HANDOFF.md).

| Member | Component owned | Contributions |
|---|---|---|
| Kruthi Shandilya Maramraju | MF-BPR retrieval (03) | TBD |
| Chaithanya Anugu | Two-tower retrieval (04) | TBD |
| Karthik Reddy Changal | Ranking (05) | TBD |
| Nithin Sujith Nair | Data pipeline & baselines (01–02) | Full rerun of notebooks 01 & 02 end-to-end; k-core-before-split acknowledgment added to nb01 §3; `src/metrics.py` + `tests/test_metrics.py` in place; fairness analysis across activity groups, Colab verification pass, slides assembly + demo GIF *(in progress)* |
| Ana-Maria Borduselu | Serving system (app/) + initial build | TBD |
