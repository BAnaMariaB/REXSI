"""Smoke tests against the RUNNING stack (docker compose up + populate_db done).

Skipped unless API_TEST_URL is set:
    API_TEST_URL=http://localhost:8000 pytest tests/test_integration.py -v
"""
import os

import pytest

requests = pytest.importorskip("requests")

API = os.environ.get("API_TEST_URL")
pytestmark = pytest.mark.skipif(not API, reason="API_TEST_URL not set (stack not running)")


def test_health():
    r = requests.get(f"{API}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_swagger_docs_served():
    assert requests.get(f"{API}/docs", timeout=5).status_code == 200  # project requirement


def test_popular_returns_items_with_metadata():
    r = requests.get(f"{API}/recommend/popular?k=10", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 10
    first = body["items"][0]
    assert {"item_idx", "parent_asin", "title"} <= set(first)
    assert "model" in body  # every response must say which model produced it


def test_popular_varies_on_reload():
    a = [i["item_idx"] for i in requests.get(f"{API}/recommend/popular", timeout=10).json()["items"]]
    b = [i["item_idx"] for i in requests.get(f"{API}/recommend/popular", timeout=10).json()["items"]]
    assert a != b or len(set(a)) < 10  # temperature sampling -> near-certain difference


def test_unknown_user_falls_back_to_popularity():
    r = requests.get(f"{API}/recommend/user/definitely_not_a_user_xyz", timeout=10)
    assert r.status_code == 200
    assert "Popular" in r.json()["model"]


def test_unknown_item_404():
    assert requests.get(f"{API}/similar/NOT_AN_ASIN", timeout=10).status_code == 404


def test_metrics_endpoint_tracks_latency():
    requests.get(f"{API}/recommend/popular", timeout=10)
    m = requests.get(f"{API}/metrics", timeout=5).json()
    assert any("recommend" in k for k in m), m
    entry = next(v for k, v in m.items() if "recommend" in k)
    assert entry["avg_ms"] > 0 and entry["n"] >= 1


def test_response_time_header_present():
    r = requests.get(f"{API}/recommend/popular", timeout=10)
    assert "X-Response-Time-Ms" in r.headers
