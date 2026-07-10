"""Integrity checks for the five notebooks: valid JSON, compilable code, narrated."""
import glob
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb")))

EXPECTED = ["01_data_preparation_eda", "02_baselines", "03_mf_bpr",
            "04_two_tower", "05_ranking"]


def test_all_expected_notebooks_exist():
    names = [os.path.basename(p) for p in NOTEBOOKS]
    for e in EXPECTED:
        assert any(e in n for n in names), f"missing notebook: {e}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_valid_and_compiles(path):
    with open(path) as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            compile(src, f"{os.path.basename(path)}:cell{i}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_is_narrated(path):
    """Project rule: 'a notebook with only code and no narrative is not acceptable'."""
    with open(path) as f:
        nb = json.load(f)
    kinds = [c["cell_type"] for c in nb["cells"]]
    n_md, n_code = kinds.count("markdown"), kinds.count("code")
    assert n_md >= max(3, n_code // 3), f"{path}: too little narration ({n_md} md / {n_code} code)"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_no_random_split_anywhere(path):
    """Random split = 0 on the grading criterion. train_test_split must never appear."""
    with open(path) as f:
        content = f.read()
    assert "train_test_split" not in content, f"{path} uses sklearn train_test_split!"
