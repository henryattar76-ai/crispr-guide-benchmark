"""Rank-based evaluation metrics with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score


def spearman(y_true, y_score) -> float:
    if np.ptp(y_score) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_score)[0])


def pearson(y_true, y_score) -> float:
    if np.ptp(y_score) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_score)[0])


def _k(n: int, frac: float) -> int:
    return max(1, int(round(frac * n)))


def top_frac_precision(y_true, y_score, frac: float = 0.2) -> float:
    """Overlap between the true top-`frac` guides and the predicted top-`frac`."""
    k = _k(len(y_true), frac)
    top_true = set(np.argsort(-np.asarray(y_true), kind="stable")[:k])
    top_pred = set(np.argsort(-np.asarray(y_score), kind="stable")[:k])
    return len(top_true & top_pred) / k


def auroc_high_vs_rest(y_true, y_score, frac: float = 0.2) -> float:
    """AUROC for separating the true top-`frac` guides from the rest."""
    y_true = np.asarray(y_true)
    cutoff = np.quantile(y_true, 1 - frac)
    high = y_true >= cutoff
    if high.all() or (~high).all():
        return float("nan")
    return float(roc_auc_score(high, y_score))


def ndcg_top(y_true, y_score, frac: float = 0.2) -> float:
    k = _k(len(y_true), frac)
    return float(ndcg_score([np.asarray(y_true)], [np.asarray(y_score)], k=k))


METRICS = {
    "spearman": spearman,
    "pearson": pearson,
    "top20_precision": top_frac_precision,
    "ndcg20": ndcg_top,
    "auroc_top20": auroc_high_vs_rest,
}


def all_metrics(y_true, y_score) -> dict:
    y_true, y_score = np.asarray(y_true, float), np.asarray(y_score, float)
    return {name: fn(y_true, y_score) for name, fn in METRICS.items()}


def bootstrap_ci(y_true, y_score, metric_fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over guides. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true, float), np.asarray(y_score, float)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(y_true[idx], y_score[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(y_true, score_a, score_b, metric_fn, n_boot: int = 1000, seed: int = 0):
    """Paired bootstrap of metric(A) - metric(B) on the same guides.

    Returns (diff, lo, hi, p_two_sided). The p-value is the usual bootstrap
    approximation: twice the smaller tail mass on either side of zero.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    n = len(y_true)
    diff = metric_fn(y_true, a) - metric_fn(y_true, b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = metric_fn(y_true[idx], a[idx]) - metric_fn(y_true[idx], b[idx])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diff), float(lo), float(hi), float(min(p, 1.0))
