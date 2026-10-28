"""Harmonize SpCas9 on-target datasets into a common format.

Every guide is represented by its 30-nt context: 4 nt upstream + 20 nt protospacer
+ 3 nt PAM (NGG) + 3 nt downstream. Activity labels differ between studies
(indel %, log-fold change, ...), so each dataset gets a within-dataset rank label
in (0, 1] and raw labels are never compared across datasets.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

CONTEXT_LEN = 30
GUIDE_START, GUIDE_END = 4, 24  # 20-nt protospacer is context[4:24]
NUCS = set("ACGT")


def clean_seq(s: str) -> str:
    return str(s).strip().upper().replace("U", "T")


def is_valid_context(s: str) -> bool:
    """30 nt, only ACGT, and an NGG PAM at positions 24-26 (0-indexed)."""
    return len(s) == CONTEXT_LEN and set(s) <= NUCS and s[25:27] == "GG"


def guide20(context: str) -> str:
    return context[GUIDE_START:GUIDE_END]


def harmonize_dataset(
    df: pd.DataFrame,
    dataset: str,
    seq_col: str,
    label_col: str,
    higher_is_better: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Return (tidy dataframe, report) for one raw dataset.

    Invalid contexts are dropped; duplicate contexts are collapsed by averaging
    their raw labels. Every drop is counted in the report so it can be reported
    in the paper's methods.
    """
    n_input = len(df)
    out = pd.DataFrame(
        {
            "context30": df[seq_col].map(clean_seq),
            "label_raw": pd.to_numeric(df[label_col], errors="coerce"),
        }
    )
    n_missing_label = int(out["label_raw"].isna().sum())
    out = out.dropna(subset=["label_raw"])

    valid = out["context30"].map(is_valid_context)
    n_invalid = int((~valid).sum())
    out = out[valid]

    n_before_dedup = len(out)
    out = out.groupby("context30", as_index=False)["label_raw"].mean()
    n_duplicates = n_before_dedup - len(out)

    signed = out["label_raw"] if higher_is_better else -out["label_raw"]
    out["label_rank"] = signed.rank(pct=True, method="average")
    out["guide20"] = out["context30"].map(guide20)
    out["dataset"] = dataset
    out["guide_id"] = [f"{dataset}:{i}" for i in range(len(out))]
    out = out[["guide_id", "dataset", "context30", "guide20", "label_raw", "label_rank"]]

    report = {
        "dataset": dataset,
        "n_input": n_input,
        "n_missing_label": n_missing_label,
        "n_invalid_context": n_invalid,
        "n_duplicate_contexts_merged": n_duplicates,
        "n_final": len(out),
    }
    return out.reset_index(drop=True), report


def _chunk_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(length, parts)
    bounds, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def min_hamming_within(queries, refs, max_mm: int = 3) -> np.ndarray:
    """Minimum Hamming distance from each query to any reference guide.

    Exact for distances <= max_mm; queries with no reference within max_mm get
    max_mm + 1. Uses the pigeonhole principle: two sequences within k mismatches
    must share at least one exactly matching chunk when split into k+1 chunks.
    """
    queries, refs = list(queries), list(refs)
    if not refs:
        return np.full(len(queries), max_mm + 1)
    length = len(refs[0])
    bounds = _chunk_bounds(length, max_mm + 1)
    index = [defaultdict(list) for _ in bounds]
    for r_idx, ref in enumerate(refs):
        for b, (s, e) in enumerate(bounds):
            index[b][ref[s:e]].append(r_idx)

    result = np.full(len(queries), max_mm + 1)
    for q_idx, q in enumerate(queries):
        candidates = set()
        for b, (s, e) in enumerate(bounds):
            candidates.update(index[b].get(q[s:e], ()))
        best = max_mm + 1
        for r_idx in candidates:
            d = _hamming(q, refs[r_idx])
            if d < best:
                best = d
                if best == 0:
                    break
        result[q_idx] = best
    return result


def seen_mask(test_guides, train_guides, max_mm: int = 3) -> np.ndarray:
    """True where a test guide is identical or near-identical (<= max_mm
    mismatches in the 20-mer) to some guide in the training set."""
    return min_hamming_within(test_guides, train_guides, max_mm) <= max_mm
