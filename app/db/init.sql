-- Schema for the Kindle recommender webapp.
-- Populated offline from notebook exports (items.parquet, interactions parquet).

CREATE TABLE IF NOT EXISTS items (
    item_idx     INTEGER PRIMARY KEY,          -- contiguous index used by models/FAISS
    parent_asin  TEXT UNIQUE NOT NULL,
    title        TEXT,
    image_url    TEXT,
    category     TEXT,
    avg_rating   REAL,
    rating_count INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    user_idx  INTEGER PRIMARY KEY,
    user_id   TEXT UNIQUE NOT NULL
);

-- ALL interactions, tagged by split period.
-- period='test' rows = "recent history" the model was NOT trained on (drives the user tower);
-- train/val rows are needed too so already-read books are never re-recommended.
CREATE TABLE IF NOT EXISTS interactions (
    user_idx   INTEGER REFERENCES users(user_idx),
    item_idx   INTEGER REFERENCES items(item_idx),
    rating     REAL,
    ts         BIGINT,
    is_positive BOOLEAN,
    period     TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_idx, ts DESC);

-- Precomputed popularity (training set) for the fallback / logged-out homepage
CREATE TABLE IF NOT EXISTS popularity (
    item_idx INTEGER PRIMARY KEY REFERENCES items(item_idx),
    score    REAL
);

-- Precomputed item-item similarities (static, cosine on learned embeddings)
CREATE TABLE IF NOT EXISTS similar_items (
    item_idx    INTEGER,
    neighbor_idx INTEGER,
    score       REAL,
    PRIMARY KEY (item_idx, neighbor_idx)
);
