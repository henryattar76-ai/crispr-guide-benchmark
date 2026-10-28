import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines import PositionLogit, gc_content
from harmonize import harmonize_dataset, is_valid_context, min_hamming_within, seen_mask
from metrics import all_metrics, bootstrap_ci, paired_bootstrap_diff, spearman
from synthetic import make_dataset


def test_context_validation():
    good = "ACGT" + "ACGTACGTACGTACGTACGT" + "AGG" + "ACG"
    assert len(good) == 30 and is_valid_context(good)
    assert not is_valid_context(good[:-1])                       # wrong length
    assert not is_valid_context(good[:25] + "AA" + good[27:])    # no NGG PAM
    assert not is_valid_context(good.replace("A", "N", 1))       # ambiguous base


def test_min_hamming_matches_brute_force():
    rng = np.random.default_rng(0)
    refs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(300)]
    queries = []
    for r in refs[:50]:  # near-duplicates with 0-5 mutations
        q = list(r)
        for pos in rng.choice(20, rng.integers(0, 6), replace=False):
            q[pos] = rng.choice([c for c in "ACGT" if c != q[pos]])
        queries.append("".join(q))
    queries += ["".join(rng.choice(list("ACGT"), 20)) for _ in range(50)]
    fast = min_hamming_within(queries, refs, max_mm=3)
    brute = np.array([min(min(sum(a != b for a, b in zip(q, r)) for r in refs), 4) for q in queries])
    assert (fast == brute).all()


def test_harmonize_and_rank_labels():
    raw = make_dataset(400, seed=1)
    raw.loc[0, "seq"] = "NNNN"  # invalid row should be dropped
    df, rep = harmonize_dataset(raw, "A", "seq", "activity")
    assert rep["n_invalid_context"] == 1 and rep["n_final"] == len(df)
    assert df["label_rank"].between(0, 1).all()
    flipped, _ = harmonize_dataset(raw, "A", "seq", "activity", higher_is_better=False)
    assert spearman(df["label_rank"], flipped["label_rank"]) < -0.99


def test_baseline_beats_chance_and_gc_is_valid():
    a, _ = harmonize_dataset(make_dataset(1500, 1), "A", "seq", "activity")
    b, _ = harmonize_dataset(make_dataset(800, 2), "B", "seq", "activity")
    model = PositionLogit().fit(a["context30"], a["label_rank"])
    m = all_metrics(b["label_rank"], model.predict(b["context30"]))
    assert m["spearman"] > 0.3 and m["auroc_top20"] > 0.65
    gc = gc_content(b["context30"])
    assert 0 <= gc.min() and gc.max() <= 1


def test_seen_mask_detects_overlap():
    a_raw = make_dataset(500, 1)
    a, _ = harmonize_dataset(a_raw, "A", "seq", "activity")
    b_raw = make_dataset(300, 2, overlap_with=a, n_overlap=100)
    b, _ = harmonize_dataset(b_raw, "B", "seq", "activity")
    mask = seen_mask(b["guide20"], a["guide20"], max_mm=3)
    assert mask.sum() >= 100  # the planted overlaps must all be flagged
    assert mask.sum() < 150   # random 20-mers should essentially never fall within 3 mismatches


def test_bootstrap_functions():
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    good, bad = y + rng.normal(0, 0.5, 400), rng.normal(size=400)
    point, lo, hi = bootstrap_ci(y, good, spearman, n_boot=200)
    assert lo <= point <= hi
    diff, dlo, dhi, p = paired_bootstrap_diff(y, good, bad, spearman, n_boot=200)
    assert diff > 0 and dlo > 0 and p < 0.05
