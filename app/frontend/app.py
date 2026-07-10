"""Streamlit frontend — calls the FastAPI backend, never touches models directly.

Pages (per project spec, each backed by a different model):
  - Home (logged out)  -> popularity
  - Home (logged in)   -> two-tower + LightGBM  + "Because you liked X" row
  - Item page          -> similar items (cosine on learned embeddings), opened by
                          clicking "Similar items" under any recommendation card
Every section shows WHICH model produced it.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Kindle Recommender", layout="wide")
st.title("📚 Kindle Recommender")

user_id = st.sidebar.text_input("Log in as user_id (blank = anonymous)")


def show_items(items: list, model_label: str, allow_similar: bool = True, key_prefix: str = "") -> None:
    st.caption(f"Model: **{model_label}**")
    cols = st.columns(5)
    for i, it in enumerate(items):
        with cols[i % 5]:
            if it.get("image_url"):
                st.image(it["image_url"], use_container_width=True)
            st.markdown(f"**{it.get('title', it.get('item_idx'))}**")
            st.caption(it.get("category") or "")
            if allow_similar and it.get("parent_asin"):
                if st.button("Similar items", key=f"sim-{key_prefix}-{i}-{it['parent_asin']}"):
                    st.session_state["selected_item"] = (it["parent_asin"], it.get("title", ""))


try:
    if not user_id:
        st.subheader("Trending now")
        r = requests.get(f"{API_URL}/recommend/popular", timeout=10).json()
        show_items(r["items"], r.get("model", "Popular right now"), key_prefix="pop")
    else:
        st.subheader(f"Recommended for {user_id}")
        r = requests.get(f"{API_URL}/recommend/user/{user_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            show_items(data["items"], data.get("model", "Two-tower + LightGBM"), key_prefix="rec")
        else:
            st.info("Personalized model not deployed yet — showing popular items.")
            r = requests.get(f"{API_URL}/recommend/popular", timeout=10).json()
            show_items(r["items"], "Popular right now (fallback)", key_prefix="fb")
        r = requests.get(f"{API_URL}/because-you-liked/{user_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.subheader(f"Because you liked *{data['anchor']['title']}*")
            show_items(data["items"], data["model"], key_prefix="byl")

    # --- Item page: similar items for the last clicked item (required section) ---
    sel = st.session_state.get("selected_item")
    if sel:
        asin, title = sel
        st.divider()
        st.subheader(f"📖 Item page — similar to *{title}*")
        r = requests.get(f"{API_URL}/similar/{asin}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            show_items(data["items"], data["model"], allow_similar=False, key_prefix="item")
        else:
            st.info("Similar items not available for this title yet.")
except requests.exceptions.RequestException as e:
    st.error(f"API not reachable at {API_URL}: {e}")
