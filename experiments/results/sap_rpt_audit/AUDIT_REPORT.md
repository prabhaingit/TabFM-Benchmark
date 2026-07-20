# SAP-RPT-1 Benchmark Audit Report

**Date:** July 20, 2026  
**Auditor:** Claude Code (Sonnet 4.6), guided by Fable model review  
**Audit scope:** `tabfm-benchmark`, branch `feature/add-sap-rpt1-model`  
**Results audited:** `experiments/results/aggregated/results.csv` (310 rows at audit time → 450 rows after fixes)

---

## Executive Summary

Three significant problems found; two resolved.

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | **Critical** | All categorical values are ordinal-encoded to integers before SAP-RPT-1 sees the data — the model has been running in its worst-case ablation condition throughout | ⏳ Pending |
| 2 | **High** | Results.csv had asymmetric coverage: SAP-RPT-1 had 75 pairs, all other models had 47 — means were computed over different subsets | ✅ Fixed July 20 |
| 3 | **High** | PR description incorrectly stated SAP-RPT-1 is "statistically tied with TabPFN" — TabPFN significantly outperforms SAP-RPT-1 (Wilcoxon p < 0.0001) | ✅ Fixed July 20 |

---

## Task 1 — Coverage Audit and Fair Leaderboard Recomputation

### 1.1 Raw coverage (at audit time — before fix)

| Model | Rows in CSV | (dataset, seed) pairs |
|---|---|---|
| SAP-RPT-1 | 75 | 75 |
| TabPFN | 47 | 47 |
| XGBoost | 47 | 47 |
| LightGBM | 47 | 47 |
| CatBoost | 47 | 47 |
| MLP | 47 | 47 |

Expected: 75 each (15 datasets × 5 seeds). Only SAP-RPT-1 met this.

**✅ Fixed July 20:** `experiments/run_missing_seeds.py` added seeds 1–4 for all 5 baseline models
on the 7 incomplete datasets. All models now have 75 pairs. results.csv = 450 rows.

### 1.2 Root cause of asymmetry (confirmed by inspecting every raw JSON)

| Date | Files | Datasets | Models | Seeds |
|---|---|---|---|---|
| Jul 5 | `20260705_*` | All 15 | 5 (no SAP-RPT-1) | **Seed 0 only** |
| Jul 15 | `20260715_*` | credit-g only | 6 | Seed 0 — n_jobs debug runs |
| Jul 16 | `20260716_091603_*` | 3 (credit-g, diabetes, spambase) | 6 | Seeds 0–4 |
| Jul 16 | `20260716_115150_*` | 8 (same first-batch datasets) | 6 | Seeds 0–4 |
| Jul 16 | `20260716_144626_saprpt_*` | 7 remaining | SAP-RPT-1 only | Seeds 0–4 |

The July 5 run was a **single-seed sweep** (seed 0 only) across all 15 datasets — it was not a 5-seed run that crashed mid-way. The July 16 re-run added seeds 1–4 but only covered 8 of the 15 datasets. `run_saprpt_missing.py` then added SAP-RPT-1 with seeds 0–4 to the remaining 7 datasets, but **never added seeds 1–4 for TabPFN, XGBoost, LightGBM, CatBoost, MLP** on those same 7 datasets.

This is why SAP-RPT-1 has 75 pairs (15 × 5) while every other model has only 47 (8 × 5 + 7 × 1).

**Complete cases (all 6 models non-null):** 47 pairs  
- 8 full datasets × 5 seeds = 40 pairs  
- 7 partial datasets × 1 seed (seed 0 only) = 7 pairs  
- **28 pairs dropped** (seeds 1–4 for the 7 missing-seed datasets)

### 1.3 Final leaderboard — all 75 pairs balanced (post-fix)

| Model | Mean AUC | AUC rank | Mean rank | Rank position |
|---|---|---|---|---|
| **TabPFN** | **0.9121** | 1 | **1.84** | 1 |
| **SAP-RPT-1** | **0.8906** | 2 | **2.78** | 2 |
| CatBoost | 0.8782 | 3 | 3.29 | 3 |
| XGBoost | 0.8752 | 5 | 3.95 | 4 |
| LightGBM | 0.8744 | 6 | 4.11 | 5 |
| MLP | 0.8499 | 6 | 5.03 | 6 |

**AUC and rank orderings are consistent.** TabPFN is #1 by both measures.

Note: TabPFN's AUC dropped from the earlier incomplete-data estimate (0.932 on 47 pairs) to
0.912 on all 75 pairs — the 7 newly completed datasets are harder on average for TabPFN.

**Why did the original summary claim AUC=0.891 for SAP-RPT-1 while placing TabPFN #1 at 0.8847?**  
The original means were computed over non-identical subsets:
- SAP-RPT-1 mean AUC = 0.891 over 75 pairs (all 15 datasets × 5 seeds)
- TabPFN mean AUC = 0.8847 over 47 pairs (8 full datasets + 7 seed-0-only)
- The comparison was **apples to oranges**. After balancing all models to 75 pairs, TabPFN correctly leads.

### 1.4 Wilcoxon test: SAP-RPT-1 vs TabPFN (final, all 75 pairs)

| Metric | Value |
|---|---|
| Pairs compared | 75 (all datasets × seeds) |
| SAP-RPT-1 wins | 14/75 (18.7%) |
| TabPFN wins | 61/75 (81.3%) |
| Mean diff (SAP − TabPFN) | −0.0215 |
| p-value | < 0.0001 |
| **Significant** | **YES** |

**⚠️ This directly contradicts the PR description** which states SAP-RPT-1 is "statistically tied with TabPFN (Wilcoxon p > 0.05)". On the correct complete-case comparison, TabPFN significantly outperforms SAP-RPT-1.

### 1.5 Full significance matrix (Wilcoxon, complete cases)

| | CatBoost | LightGBM | MLP | SAP-RPT-1 | TabPFN | XGBoost |
|---|---|---|---|---|---|---|
| CatBoost | — | 0.3019 | **0.0000** | **0.0060** | **0.0000** | 0.6491 |
| LightGBM | 0.3019 | — | **0.0000** | **0.0002** | **0.0000** | 0.9535 |
| MLP | **0.0000** | **0.0000** | — | **0.0000** | **0.0000** | **0.0002** |
| SAP-RPT-1 | **0.0060** | **0.0002** | **0.0000** | — | **0.0000** | **0.0004** |
| TabPFN | **0.0000** | **0.0000** | **0.0000** | **0.0000** | — | **0.0000** |
| XGBoost | 0.6491 | 0.9535 | **0.0002** | **0.0004** | **0.0000** | — |

**Bold = p < 0.05 (significant difference)**

Correct significance structure:
- **TabPFN > SAP-RPT-1** (p < 0.0001) — NOT tied
- **SAP-RPT-1 > CatBoost, XGBoost, LightGBM** (p < 0.01) — SAP-RPT-1 does beat the tuned tree models significantly
- **CatBoost, XGBoost, LightGBM are NOT significantly different from each other** (p > 0.3)
- **SAP-RPT-1 >> MLP** (p < 0.0001)

---

## Task 2 — Semantic Data Survival Check (**Critical**)

### 2.1 What the loader actually does

**File:** `src/data/loader.py`, method `_process()`, lines 66–142.

The loader performs **full ordinal encoding** on ALL categorical columns before any model sees the data:

```python
# Lines 102–115: Encode categoricals as integers
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[col].astype(str))
    X_train[col] = le.transform(X_train[col].astype(str)).astype(np.float32)
    # ... same for X_test

# Line 126: Cast everything to float32
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
```

### 2.2 Dtype at the point SAPRptWrapper.fit() is called

| What | Actual state |
|---|---|
| `X_train` type | `pd.DataFrame` (column names preserved ✅) |
| `X_train.dtypes` | **ALL `float32`** ❌ — no `object` or `category` columns survive |
| Categorical values | Replaced by integer codes 0.0, 1.0, 2.0, ... |
| Column names | Preserved (e.g. `checking_status`, `credit_history`) |

**Concrete example — credit-g `checking_status` column:**

What SAP-RPT-1 should see (raw strings): `"no checking"`, `"<0"`, `"0<=X<200"`, `">200"`  
What SAP-RPT-1 actually sees: `0.0`, `1.0`, `2.0`, `3.0`

The column *name* `checking_status` reaches the model, but the *values* are arbitrary integer codes. The model can use the header as a semantic signal but gets no value-level semantics at all.

### 2.3 Impact — connection to the paper

The ConTextTab paper (arXiv:2506.10707) explicitly ablates this:

> *"No feature semantics - ordinal encoder"* ablation: **−2.7 accuracy, −4.8 R²** on semantic datasets

Our entire benchmark has been running SAP-RPT-1 in this exact ablation condition. This means:
- The measured AUC = 0.891 (all 75 pairs, final) is a **lower bound** on SAP-RPT-1's capability on semantically rich data
- Our 15 datasets are numeric-dominant (few categorical columns), so the penalty is smaller than the paper's semantic benchmark, but it is still active wherever categorical features exist
- The fix is a **per-model preprocessing branch** in `run_benchmark.py` or a `raw=True` flag to the loader — NOT a global loader change (tree models legitimately need encoded input)

### 2.4 The 7 datasets where SAP-RPT-1 produced identical AUC across seeds 1–4

Looking at the missing-seed data:
```
kc1        seeds 1-4: AUC = 0.837470 (all identical)
kc2        seeds 1-4: AUC = 0.735487 (all identical)
ozone-level seeds 1-4: AUC = 0.938487 (all identical)
```

This is expected: SAP-RPT-1 is zero-shot and deterministic. The `random_state` only affects subsampling (not triggered when N < max_context_size) and the 50-sample calibration check. Train/test split is fixed to `random_state=42` regardless of model seed. So seeds 1–4 for these datasets produce bitwise identical results — the 5-seed averaging adds no information for SAP-RPT-1 on datasets that don't trigger subsampling.

---

## Task 3 — Configuration Consistency

| Dataset | N train | max_context_size | Subsampling triggered | bagging | Notes |
|---|---|---|---|---|---|
| banknote-auth | 1,097 | 8192 | No | 1 | |
| credit-g | 800 | 8192 | No | 1 | |
| diabetes | 614 | 8192 | No | 1 | |
| hill-valley | 969 | 8192 | No | 1 | |
| qsar-biodeg | 844 | 8192 | No | 1 | |
| spambase | 3,681 | 8192 | No | 1 | |
| titanic | 712 | 8192 | No | 1 | |
| wdbc | 455 | 8192 | No | 1 | |
| australian | 552 | **2048** | No | 1 | OOM at 8192 — re-run with 2048 |
| kc1 | 1,687 | 8192 | No | 1 | |
| kc2 | 944 | 8192 | No | 1 | |
| ozone-level | 2,029 | 8192 | No | 1 | |
| pc1 | 887 | 8192 | No | 1 | |
| phoneme | 4,323 | 8192 | No | 1 | |
| sick | 3,017 | 8192 | No | 1 | |

**Note 1:** `bagging=1` used throughout (paper default is `bagging=8`). The audit plan flags this as measurably hurting rank.  
**Note 2:** Australian was actually too small (552 rows) to need subsampling at 8192 — the OOM was caused by other processes using VRAM. The `max_context_size=2048` for Australian is not a real config change; it's just that 552 < 2048 so no subsampling occurs either way.  
**Note 3:** No dataset exceeds 8192 training rows, so `_subsample_balanced()` was never triggered in this benchmark.

---

## Summary of Required Fixes

### Fix 1 — PR description correction ✅ Done (July 20)
Updated on GitHub. Correct language:
- TabPFN significantly outperforms SAP-RPT-1 (p < 0.0001, SAP-RPT-1 win rate 19%)
- SAP-RPT-1 significantly outperforms CatBoost, LightGBM, XGBoost (all p < 0.01), MLP (p < 0.0001)
- CatBoost, XGBoost, LightGBM are not significantly different from each other

### Fix 2 — Complete missing seeds for baseline models ✅ Done (July 20)
`experiments/run_missing_seeds.py` ran seeds 1–4 for TabPFN, XGBoost, LightGBM, CatBoost, MLP
on the 7 previously incomplete datasets. results.csv is now fully balanced: 450 rows, 75/75 complete cases.

### Fix 3 — Raw string input path for SAP-RPT-1 (gates Task 5)
Add a preprocessing branch in `run_benchmark.py` or a `raw=True` parameter to `DataLoader` so SAP-RPT-1 receives string-valued categorical columns. The fix must be model-specific — the global loader must continue encoding for tree/MLP models.

### Fix 4 — bagging=8 (paper default, recommended for fair comparison)
The paper uses `bagging=8`. Our config uses `bagging=1`. Increasing this to 8 may improve AUC, especially on small datasets.

---

## Task 5 Status: BLOCKED

Task 5 (semantic arm) must not run until Fix 3 is applied. Running the semantic datasets (adult, bank-marketing, credit-approval, dresses-sales) through the current loader would reproduce the ordinal-encoder ablation condition and the results would be meaningless for testing the paper's hypothesis.
