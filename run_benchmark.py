"""Cross-dataset benchmark with seen/unseen leakage control.

Usage:  python src/run_benchmark.py config.yaml

For every (train dataset -> test dataset) pair, test guides are split into
"seen" (within `max_mismatch` mismatches of any training guide) and "unseen".
Baselines are trained on the train dataset. External tools are scored from
precomputed score files; their own training guides (if provided) define seen/unseen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from baselines import PositionLogit, gc_content  # noqa: E402
from harmonize import harmonize_dataset, seen_mask  # noqa: E402
from metrics import METRICS, all_metrics, bootstrap_ci  # noqa: E402


def load_datasets(cfg, root: Path):
    frames, reports = [], []
    for name, spec in cfg["datasets"].items():
        raw = pd.read_csv(root / spec["file"])
        df, rep = harmonize_dataset(
            raw, name, spec["seq_col"], spec["label_col"], spec.get("higher_is_better", True)
        )
        frames.append(df)
        reports.append(rep)
    return frames, pd.DataFrame(reports)


def score_rows(model_name, train_name, test_name, subset, test_df, scores, mask, n_boot):
    rows = []
    for label, sel in (("all", np.ones(len(test_df), bool)), ("seen", mask), ("unseen", ~mask)):
        if sel.sum() < 30:  # too few guides for stable rank metrics
            continue
        y, s = test_df.loc[sel, "label_rank"].to_numpy(), scores[sel]
        row = {"model": model_name, "train": train_name, "test": test_name, "subset": label, "n": int(sel.sum())}
        row.update(all_metrics(y, s))
        _, lo, hi = bootstrap_ci(y, s, METRICS["spearman"], n_boot=n_boot)
        row["spearman_lo"], row["spearman_hi"] = lo, hi
        rows.append(row)
    return rows


def main(config_path: str):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    cfg = yaml.safe_load(config_path.read_text())
    max_mm, n_boot = cfg.get("max_mismatch", 3), cfg.get("n_boot", 1000)

    frames, report = load_datasets(cfg, root)
    out_dir = root / cfg.get("results_dir", "results")
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "harmonization_report.csv", index=False)
    by_name = {f["dataset"].iloc[0]: f for f in frames}
    for name, df in by_name.items():
        df.to_csv(root / "data/processed" / f"{name}.csv", index=False)

    rows = []
    for train_name, train_df in by_name.items():
        model = PositionLogit().fit(train_df["context30"], train_df["label_rank"])
        for test_name, test_df in by_name.items():
            if test_name == train_name:
                continue
            mask = seen_mask(test_df["guide20"], train_df["guide20"], max_mm)
            ctx = test_df["context30"]
            rows += score_rows("gc_content", train_name, test_name, "", test_df, gc_content(ctx), mask, n_boot)
            rows += score_rows("position_logit", train_name, test_name, "", test_df, model.predict(ctx), mask, n_boot)

    # External tools: precomputed scores (guide context30 + score) and optional training-guide list.
    for tool, spec in (cfg.get("tools") or {}).items():
        scores_df = pd.read_csv(root / spec["scores"])  # columns: context30, score
        train_guides = []
        if spec.get("training_guides"):
            train_guides = pd.read_csv(root / spec["training_guides"])["guide20"].tolist()
        for test_name, test_df in by_name.items():
            merged = test_df.merge(scores_df, on="context30", how="inner")
            if merged.empty:
                continue
            mask = seen_mask(merged["guide20"], train_guides, max_mm)
            rows += score_rows(tool, "published", test_name, "", merged, merged["score"].to_numpy(), mask, n_boot)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    print(report.to_string(index=False))
    print()
    print(results.round(3).to_string(index=False) if len(results) else "No results (too few guides?)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
