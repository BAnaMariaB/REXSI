"""Build items.parquet: item metadata (title, image, category) for the webapp.

Streams the official Kindle Store metadata file and keeps only items that
survived preprocessing (i.e., are in the training vocabulary from notebook 01).

Usage:
    python build_items.py --data-dir ./data
Requires: id_mappings.json in --data-dir (produced by notebook 01).
Downloads meta_Kindle_Store.jsonl.gz (~2.5GB) on first run.
"""
import argparse
import gzip
import json
import os
import urllib.request

import pandas as pd

META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Kindle_Store.jsonl.gz"


def pick_image(images: list) -> str | None:
    for img in images or []:
        for key in ("large", "hi_res", "thumb"):
            if img.get(key):
                return img[key]
    return None


def pick_category(obj: dict) -> str | None:
    cats = obj.get("categories") or []
    # e.g. ["Kindle Store", "Kindle eBooks", "Mystery, Thriller & Suspense"] -> keep the informative tail
    if len(cats) > 1:
        return " > ".join(cats[1:3])
    return obj.get("main_category")


def main(data_dir: str) -> None:
    with open(os.path.join(data_dir, "id_mappings.json")) as f:
        item2idx = json.load(f)["item2idx"]
    print(f"Vocabulary: {len(item2idx):,} items")

    meta_path = os.path.join(data_dir, "meta_Kindle_Store.jsonl.gz")
    if not os.path.exists(meta_path):
        print("Downloading metadata (~2.5GB, one-time)...")
        urllib.request.urlretrieve(META_URL, meta_path)

    rows, scanned = [], 0
    with gzip.open(meta_path, "rt") as fp:
        for line in fp:
            scanned += 1
            if scanned % 200_000 == 0:
                print(f"  scanned {scanned:,} | matched {len(rows):,}")
            obj = json.loads(line)
            asin = obj.get("parent_asin")
            if asin not in item2idx:
                continue
            rows.append({
                "item_idx": item2idx[asin],
                "parent_asin": asin,
                "title": (obj.get("title") or "")[:500],
                "image_url": pick_image(obj.get("images")),
                "category": pick_category(obj),
                "avg_rating": obj.get("average_rating"),
                "rating_count": obj.get("rating_number"),
            })

    items = pd.DataFrame(rows).drop_duplicates("item_idx")
    out = os.path.join(data_dir, "items.parquet")
    items.to_parquet(out, index=False)
    missing = len(item2idx) - len(items)
    print(f"Saved {len(items):,} items -> {out} ({missing:,} vocab items had no metadata)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    main(ap.parse_args().data_dir)
