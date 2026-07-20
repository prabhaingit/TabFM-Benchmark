"""Task 1: Coverage audit and fair leaderboard recomputation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = Path(__file__).parent.parent / "aggregated/results.csv"
df = pd.read_csv(CSV)

print("=" * 70)
print("TASK 1 — Coverage audit and fair leaderboard recomputation")
print("=" * 70)

# ── 1. Raw coverage per model ─────────────────────────────────────────────
print("\n## 1. Raw row count per model")
print(df.groupby("model").size().rename("rows").to_string())

print("\n## 2. (dataset, seed) coverage per model  (expect 75 each = 15 × 5)")
coverage = df.groupby(["model", "dataset", "seed"]).size().unstack(["dataset", "seed"])
per_model = df.groupby("model")[["dataset", "seed"]].apply(
    lambda g: len(g.drop_duplicates())
).rename("n_dataset_seed_pairs")
print(per_model.to_string())

# ── 2. Which (dataset, seed) pairs are missing for SAP-RPT-1 vs TabPFN ───
print("\n## 3. Missing (dataset, seed) pairs for SAP-RPT-1 vs TabPFN")
all_ds_seed = set(
    zip(df["dataset"], df["seed"])
)
for model in ["SAP-RPT-1", "TabPFN"]:
    sub = df[df["model"] == model][["dataset", "seed"]].drop_duplicates()
    model_ds_seed = set(zip(sub["dataset"], sub["seed"]))
    missing = sorted(all_ds_seed - model_ds_seed)
    print(f"\n  {model}: {len(model_ds_seed)} pairs present, {len(missing)} missing")
    if missing:
        for ds, s in missing[:20]:
            print(f"    missing: dataset={ds}, seed={s}")

# ── 3. Complete cases: inner join across all 6 models ────────────────────
print("\n## 4. Complete-case (inner join) recomputation")
pivot = df.pivot_table(
    index=["dataset", "seed"],
    columns="model",
    values="roc_auc",
    aggfunc="first",
)
complete = pivot.dropna()
print(f"\n  Total (dataset, seed) pairs: {len(pivot)}")
print(f"  Complete cases (all models non-null): {len(complete)}")
print(f"  Dropped: {len(pivot) - len(complete)} pairs")

# Identify which datasets/seeds have any null
nulls = pivot[pivot.isnull().any(axis=1)]
if not nulls.empty:
    print(f"\n  Rows with at least one null model:")
    print(nulls.to_string())

# ── 4. Mean AUC on complete cases ────────────────────────────────────────
print("\n## 5. Mean AUC — complete cases only")
mean_auc = complete.mean().sort_values(ascending=False).rename("mean_AUC")
print(mean_auc.round(4).to_string())

# ── 5. Mean rank on complete cases ───────────────────────────────────────
print("\n## 6. Mean rank — complete cases only (1=best)")
ranks = complete.rank(axis=1, ascending=False, method="average")
mean_rank = ranks.mean().sort_values().rename("mean_rank")
print(mean_rank.round(3).to_string())

# ── 6. Combined leaderboard ───────────────────────────────────────────────
print("\n## 7. Combined leaderboard (AUC + rank, complete cases)")
leaderboard = pd.concat([mean_auc, mean_rank], axis=1).sort_values("mean_rank")
leaderboard["AUC_rank"] = leaderboard["mean_AUC"].rank(ascending=False).astype(int)
leaderboard["rank_rank"] = leaderboard["mean_rank"].rank(ascending=True).astype(int)
print(leaderboard.round(4).to_string())

if mean_auc.idxmax() != mean_rank.idxmin():
    print(f"\n  ⚠️  AUC leader ({mean_auc.idxmax()}) != rank leader ({mean_rank.idxmin()})")
else:
    print(f"\n  ✓  AUC leader and rank leader agree: {mean_auc.idxmax()}")

# ── 7. Wilcoxon: SAP-RPT-1 vs TabPFN ────────────────────────────────────
print("\n## 8. Wilcoxon signed-rank: SAP-RPT-1 vs TabPFN (complete cases)")
if "SAP-RPT-1" in complete.columns and "TabPFN" in complete.columns:
    sap = complete["SAP-RPT-1"].values
    pfn = complete["TabPFN"].values
    diff = sap - pfn
    wins = (diff > 0).sum()
    losses = (diff < 0).sum()
    ties = (diff == 0).sum()
    stat, pval = wilcoxon(sap, pfn, alternative="two-sided", zero_method="wilcox")
    print(f"  SAP-RPT-1 wins: {wins}/{len(diff)}, losses: {losses}, ties: {ties}")
    print(f"  Win rate: {wins/len(diff):.1%}")
    print(f"  Wilcoxon statistic: {stat:.2f}")
    print(f"  p-value: {pval:.4f}")
    print(f"  Significant at p<0.05: {pval < 0.05}")
    print(f"  Mean AUC SAP-RPT-1: {sap.mean():.4f}  |  TabPFN: {pfn.mean():.4f}")
    print(f"  Mean diff (SAP - TabPFN): {diff.mean():+.4f}")
else:
    print("  One or both models not in complete cases.")

# ── 8. Full Wilcoxon matrix ───────────────────────────────────────────────
print("\n## 9. Full pairwise Wilcoxon p-values (complete cases)")
models = list(complete.columns)
pmat = pd.DataFrame(index=models, columns=models, dtype=float)
for a in models:
    for b in models:
        if a == b:
            pmat.loc[a, b] = np.nan
        else:
            try:
                _, p = wilcoxon(complete[a], complete[b], zero_method="wilcox")
                pmat.loc[a, b] = round(p, 4)
            except Exception:
                pmat.loc[a, b] = np.nan
print(pmat.to_string())
print("\n  (values < 0.05 indicate significant difference)")
sig = pmat < 0.05
print("\n  Significant pairs (p<0.05):")
for a in models:
    for b in models:
        if a < b and sig.loc[a, b]:
            print(f"    {a} vs {b}: p={pmat.loc[a,b]:.4f}")

print("\n" + "=" * 70)
print("Task 1 complete. See audit report for interpretation.")
print("=" * 70)
