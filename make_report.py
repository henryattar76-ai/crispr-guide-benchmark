"""Build a self-contained HTML report from the Doench 2016 data.

Usage:  python src/make_report.py [path/to/doench2016.csv]

Runs harmonization plus a gene-held-out 5-fold evaluation of the two baselines
(no gene is ever in both train and test) and writes results/report.html and
results/fig_gene_heldout.png.
"""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from baselines import PositionLogit, gc_content  # noqa: E402
from harmonize import harmonize_dataset  # noqa: E402
from metrics import METRICS, all_metrics, bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GENE_COL = "Target gene"


def gene_folds(genes, k=5, seed=0):
    genes = sorted(genes)
    perm = np.random.default_rng(seed).permutation(len(genes))
    folds = [[genes[j] for j in chunk] for chunk in np.array_split(perm, k)]
    flat = [g for f in folds for g in f]
    assert sorted(flat) == genes and len(set(flat)) == len(genes), "folds must partition genes"
    return folds


def main(csv_path: str):
    raw = pd.read_csv(csv_path).rename(columns={"30mer": "context30"})
    df, rep = harmonize_dataset(raw, "doench2016", "context30", "score_drug_gene_rank")
    df = df.join(raw.drop_duplicates("context30").set_index("context30")[[GENE_COL]], on="context30")

    folds = gene_folds(df[GENE_COL].unique())
    rows, oof = [], pd.DataFrame(index=df.index, columns=["position_logit", "gc_content"], dtype=float)
    for i, held in enumerate(folds):
        te = df[GENE_COL].isin(held)
        model = PositionLogit().fit(df.loc[~te, "context30"], df.loc[~te, "label_rank"])
        oof.loc[te, "position_logit"] = model.predict(df.loc[te, "context30"])
        oof.loc[te, "gc_content"] = gc_content(df.loc[te, "context30"])
        for name in oof.columns:
            y, s = df.loc[te, "label_rank"].to_numpy(), oof.loc[te, name].to_numpy()
            rows.append({"fold": i + 1, "model": name, "n_test": int(te.sum()), "genes_held_out": len(held),
                         "spearman": METRICS["spearman"](y, s)})
    per_fold = pd.DataFrame(rows)

    # pooled out-of-fold metrics with bootstrap CIs
    pooled = []
    for name in oof.columns:
        y, s = df["label_rank"].to_numpy(), oof[name].to_numpy()
        m = all_metrics(y, s)
        _, lo, hi = bootstrap_ci(y, s, METRICS["spearman"], n_boot=1000)
        pooled.append({"model": name, **m, "spearman_ci": f"{lo:.3f} to {hi:.3f}"})
    pooled = pd.DataFrame(pooled)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    per_fold.to_csv(out / "gene_heldout_per_fold.csv", index=False)
    pooled.to_csv(out / "gene_heldout_pooled.csv", index=False)

    # ---- figure: per-fold Spearman, plus top-20% precision (pooled)
    colors = {"position_logit": "#1f6f8b", "gc_content": "#b5651d"}
    labels = {"position_logit": "Position-specific logistic", "gc_content": "GC content"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    w = 0.38
    for j, name in enumerate(["position_logit", "gc_content"]):
        sub = per_fold[per_fold.model == name]
        ax.bar(sub["fold"] + (j - 0.5) * w, sub["spearman"], w, label=labels[name], color=colors[name])
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("Held-out gene fold")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Per fold: genes never seen in training", fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    vals = [float(pooled.loc[pooled.model == n, "top20_precision"].iloc[0]) for n in ["position_logit", "gc_content"]]
    ax.bar([0, 1], vals, color=[colors["position_logit"], colors["gc_content"]], width=0.55)
    ax.axhline(0.2, color="#444", lw=0.9, ls="--")
    ax.text(1.45, 0.205, "chance = 0.20", ha="right", va="bottom", fontsize=8, color="#444")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Position\nlogistic", "GC\ncontent"])
    ax.set_ylabel("Top-20% precision (pooled)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Finding the best 20% of guides", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = out / "fig_gene_heldout.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    b64 = base64.b64encode(png.read_bytes()).decode()

    # ---- HTML
    def table(frame, floatfmt="{:.3f}"):
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
        body = ""
        for _, r in frame.iterrows():
            cells = "".join(
                f"<td>{floatfmt.format(v) if isinstance(v, float) else html.escape(str(v))}</td>" for v in r
            )
            body += f"<tr>{cells}</tr>"
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    pos = pooled[pooled.model == "position_logit"].iloc[0]
    gc = pooled[pooled.model == "gc_content"].iloc[0]
    rep_df = pd.DataFrame([rep]).T.reset_index()
    rep_df.columns = ["item", "value"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guide Benchmark Report</title>
<style>
:root{{--bg:#fff;--fg:#1c2530;--mut:#5b6773;--line:#dfe5ea;--acc:#1f6f8b;--card:#f6f9fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12181e;--fg:#e6ecf1;--mut:#9aa7b3;--line:#2b3540;--acc:#6bbad6;--card:#1a222a}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0}}
main{{max-width:860px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:1.6rem;margin:0 0 4px}} h2{{font-size:1.15rem;margin:30px 0 8px;border-top:1px solid var(--line);padding-top:18px}}
.sub{{color:var(--mut);margin:0 0 18px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.card b{{display:block;font-size:1.5rem;color:var(--acc)}} .card span{{color:var(--mut);font-size:.88rem}}
table{{border-collapse:collapse;width:100%;font-size:.92rem;margin:8px 0}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}} th{{color:var(--mut);font-weight:600}}
img{{max-width:100%;height:auto;border-radius:8px;background:#fff}}
.note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:6px;margin:12px 0}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Leakage-aware benchmark of SpCas9 guide-efficiency predictors</h1>
<p class="sub">Status report: pipeline built and tested; first real-data check on Doench 2016</p>

<div class="cards">
<div class="card"><b>{rep['n_final']:,}</b><span>unique guides after harmonization (from {rep['n_input']:,} rows)</span></div>
<div class="card"><b>{df[GENE_COL].nunique()}</b><span>genes, split into 5 held-out folds</span></div>
<div class="card"><b>{pos['spearman']:.2f}</b><span>baseline Spearman on unseen genes (95% CI {pos['spearman_ci']})</span></div>
<div class="card"><b>{gc['spearman']:.2f}</b><span>GC-content Spearman (95% CI {gc['spearman_ci']})</span></div>
</div>

<h2>What this shows</h2>
<p>A simple model that only looks at which nucleotide sits at each position of the 30-nt context ranks guides on genes it never trained on with Spearman &asymp; {pos['spearman']:.2f}. GC content alone is essentially uninformative here ({gc['spearman']:.2f}). Any published deep model has to clearly beat the position baseline to justify its complexity, and that comparison is what the full benchmark will make.</p>
<div class="note"><b>Scope:</b> this is one dataset and two baselines. No published tool (Rule Set 2, DeepSpCas9, DeepHF, CRISPRon) has been scored yet, so nothing here ranks tools.</div>

<h2>Figure 1. Gene-held-out performance</h2>
<img alt="Per-fold Spearman and pooled top-20% precision for the two baselines" src="data:image/png;base64,{b64}">

<h2>Pooled out-of-fold metrics</h2>
<div class="wrap">{table(pooled.drop(columns=['pearson']))}</div>
<p class="sub">Top-20% precision is the overlap between the true and predicted top 20% of guides (chance = 0.20). AUROC separates the true top 20% from the rest.</p>

<h2>Per-fold results</h2>
<div class="wrap">{table(per_fold)}</div>

<h2>Harmonization report</h2>
<div class="wrap">{table(rep_df)}</div>

<h2>Decisions still open</h2>
<ul>
<li><b>Duplicate guides:</b> {rep['n_duplicate_contexts_merged']} repeated contexts (same guide in several drug screens) were collapsed by averaging raw scores. This choice must be justified in the methods.</li>
<li><b>In-sample predictions:</b> the Rule Set 2 predictions bundled with this file were trained on the same data, so they cannot be reported as unseen performance.</li>
<li><b>Missing datasets:</b> Kim 2019 (DeepSpCas9), Wang 2019 (DeepHF) and Xiang 2021 (CRISPRon) training data are not yet in hand, so the cross-dataset seen/unseen comparison has not been run on real data.</li>
</ul>
</main></body></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")
    print(pooled.round(3).to_string(index=False))
    print(per_fold.round(3).to_string(index=False))
    print("wrote", out / "report.html")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/raw/doench2016.csv"))
