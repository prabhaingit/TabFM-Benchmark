# SAP-RPT-1 Benchmark Contribution — Summary

**Date:** July 2026  
**Author:** Prabhakaran Kuppusamy  
**PR:** https://github.com/Siri1702/TabFM-Benchmark/pull/1  
**Fork:** https://github.com/prabhaingit/TabFM-Benchmark  
**Backup:** https://github.com/prabhaingit/tabfm-benchmark-backup

---

## What Was Done

### 1. Diagnosed and Fixed Slow Benchmark Seeds

**Root cause:** Nested `n_jobs=-1` oversubscription.  
`LGBMClassifier(n_jobs=-1)` inside `cross_val_score(n_jobs=-1)` spawned 3 joblib
subprocesses each launching all-core OpenMP thread pools → thread oversubscription →
near-zero CPU utilisation → pathological wall times (e.g. 499 s on a 4-feature dataset).

**Fix applied in `src/models/lightgbm_model.py` and `src/models/xgboost_model.py`:**
```python
# Inside Optuna objective — set estimator n_jobs=1
params = { "n_jobs": 1, ... }
# cross_val_score(n_jobs=-1) kept — parallelises 3 folds cleanly
```
**Result:** credit-g seed time dropped from 248 s → 5.2 s (48× speedup).

---

### 2. Ran the Full Benchmark

- **15 OpenML binary classification datasets**, 5 random seeds each
- **6 models:** TabPFN, SAP-RPT-1, XGBoost, LightGBM, CatBoost, MLP
- **Hardware:** NVIDIA T4 GPU (Google Cloud), 16 GB VRAM
- **Total rows in results.csv:** 450 (15 × 6 × 5 = 450, fully balanced after audit fix)

SAP-RPT-1 was added after the baseline run. The 7 datasets missing SAP-RPT-1 results
(australian, kc1, kc2, ozone-level, pc1, phoneme, sick) were run separately using
`experiments/run_saprpt_missing.py` (one-time utility, not in PR) and merged into the
aggregated CSV.

**Australian dataset OOM fix:** `max_context_size=8192` required 4.68 GB but only
4.07 GB was free on T4. Reran with `max_context_size=2048` which triggers stratified
subsampling inside the wrapper.

---

### 3. SAP-RPT-1 Benchmark Results

Numbers below are from the **final balanced analysis** — all 75 dataset×seed pairs (15 datasets × 5 seeds), all 6 models present. Seeds 1–4 for the 5 baseline models on the 7 previously incomplete datasets were added on July 20, 2026 via `experiments/run_missing_seeds.py`.

| Model | Mean AUC | Mean Rank | Speed (avg/dataset-seed) | ECE |
|---|---|---|---|---|
| **TabPFN** 🥇 | **0.9121** | 1.84 | 12.5 s | 0.048 |
| **SAP-RPT-1** 🥈 | **0.8906** | 2.78 | 9.4 s | — |
| CatBoost | 0.8782 | 3.29 | 289.2 s | 0.040 |
| XGBoost | 0.8752 | 3.95 | 84.4 s | 0.034 |
| LightGBM | 0.8744 | 4.11 | 78.5 s | 0.048 |
| MLP | 0.8499 | 5.03 | 18.9 s | 0.052 |

**Key findings (Wilcoxon signed-rank, all 75 pairs):**
- SAP-RPT-1 ranks **#2 by both AUC and mean rank**
- **Significantly outperformed by TabPFN** (p < 0.0001, SAP-RPT-1 win rate 19%)
- **Significantly beats all tuned tree models**: CatBoost, LightGBM, XGBoost (all p < 0.01)
- **Significantly beats MLP** (p < 0.0001)
- **9× faster** than XGBoost with zero tuning overhead

---

### 4. SAP-RPT-1 Wrapper (`src/models/sap_rpt_model.py`)

Three non-obvious implementation details:

**y-index alignment fix:**  
`sap_rpt_oss` aligns X and y by pandas index internally. The DataLoader returns
`y_train` as a numpy array (0-based) while `X_train` is a DataFrame with the
shuffled original row indices. Without the fix, labels are misaligned → near-random predictions.
```python
if hasattr(X_train, "index") and not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train, index=X_train.index)
```
Validated: credit-g AUC reached 0.804, matching the published Kaggle result of 0.8043.

**Positive-column calibration (`_calibrate_pos_col`):**  
Runs a 50-sample in-sample AUC check after `fit()` to determine which output column
is the positive class. Falls back to column 1 (sklearn default) when the margin is < 0.1 AUC.

**Stratified subsampling (`_subsample_balanced`):**  
When `len(X_train) > max_context_size`, subsamples per class preserving class balance.
Use `max_context_size=2048` on GPUs with < 8 GB VRAM free.

---

### 5. Files Changed in the PR

| File | Change |
|---|---|
| `src/models/sap_rpt_model.py` | New — full wrapper with Black formatting + Google docstrings |
| `configs/models.yaml` | Added `sap_rpt` entry |
| `experiments/run_benchmark.py` | Added import + `SAPRptWrapper(random_state=seed)` in `build_models()` |
| `README.md` | Added SAP-RPT-1 row to "What's Being Compared" table; inserted SAP-RPT-1 row in Benchmark Results; added Key Finding #3 |
| `experiments/results/raw/20260716_*` | 22 new raw JSON files from the July 2026 run |
| `experiments/results/aggregated/results.csv` | Updated to 450 rows (all 6 models) |

**Not included in PR** (one-time utilities, on `main` branch only):
- `experiments/run_saprpt_missing.py` — script that ran SAP-RPT-1 on 7 missing datasets

---

### 6. CONTRIBUTING.md Compliance Checklist

| Requirement | Status |
|---|---|
| Black formatting (`--line-length 100`) | ✅ |
| Google-style docstrings on all public methods | ✅ |
| Extends `ModelWrapper` base class | ✅ |
| `configs/models.yaml` entry | ✅ |
| Import + entry in `build_models()` | ✅ |
| README model table updated | ✅ |
| Conventional commit PR title (`feat(models): ...`) | ✅ |
| PR description: What / Why / How / Testing | ✅ |
| Benchmark run date, seeds, datasets documented in PR | ✅ |
| Raw JSON results included | ✅ |
| Aggregated CSV included | ✅ |
| Baseline numbers NOT overwritten | ✅ |

---

### 7. Install Requirements for SAP-RPT-1

```bash
pip install git+https://github.com/SAP-samples/sap-rpt-1-oss

# HuggingFace auth required (model checkpoints download automatically)
huggingface-cli login
```

GPU recommended. Reduce `max_context_size` to `2048` in `configs/models.yaml` if VRAM < 8 GB free.

---

### 8. Repository Layout (local)

```
/home/prabhakaran_samy/claude-projects/tabfm-benchmark/
├── main branch       — full results (450 rows), all figures, run_saprpt_missing.py
└── feature/add-sap-rpt1-model — PR branch (pushed to prabhaingit/TabFM-Benchmark)
```

Remote branches:
- `origin` → `https://github.com/prabhaingit/TabFM-Benchmark.git` (fork, PR source)
- `upstream` → `https://github.com/Siri1702/TabFM-Benchmark.git` (original repo)

---

## Audit Findings (July 20, 2026)

Post-contribution audit guided by Fable model review. Full report at
`experiments/results/sap_rpt_audit/AUDIT_REPORT.md`.

### Finding 1 — Critical: SAP-RPT-1 ran without semantic information

**File:** `src/data/loader.py`, method `_process()` (lines 102–126)

The data loader `LabelEncoder`-encodes all categorical columns to integers, then casts
everything to `float32`, before any model receives the data:

```python
# All categorical columns → integer codes → float32
le = LabelEncoder()
X_train[col] = le.transform(X_train[col].astype(str)).astype(np.float32)
# ...
X_train = X_train.astype(np.float32)   # final cast — no strings survive
```

**What SAP-RPT-1 received throughout the benchmark:**
- Column names: ✅ preserved (e.g. `checking_status`, `credit_history`)
- Column values: ❌ destroyed — `0.0, 1.0, 2.0` instead of `"no checking"`, `"<0"`, `">200"`

The ConTextTab paper (arXiv:2506.10707) explicitly ablates this condition:
*"No feature semantics — ordinal encoder"* → **−2.7 accuracy / −4.8 R²** on semantic datasets.

Every result in our benchmark was produced under this worst-case ablation.

**Required fix (not yet applied):** A per-model preprocessing branch in `run_benchmark.py`
so SAP-RPT-1 receives raw string columns. Tree/MLP models must keep their encoded input —
this cannot be a global loader change.

**Impact on Task 5 (semantic arm):** Blocked until the fix is applied. Running the
planned semantic datasets (adult, bank-marketing, credit-approval, dresses-sales) through
the current loader would produce meaningless results for testing the paper's hypothesis.

---

### Finding 2 — High: Coverage asymmetry made AUC means non-comparable

**Root cause — confirmed by inspecting every raw JSON file:**

| Date | Files | Datasets covered | Models | Seeds run |
|---|---|---|---|---|
| Jul 5 | `20260705_*` | All 15 | 5 (no SAP-RPT-1) | **Seed 0 only** |
| Jul 15 | `20260715_*` | credit-g only | 6 | Seed 0 — n_jobs debug runs |
| Jul 16 | `20260716_091603_*` | 3 (credit-g, diabetes, spambase) | 6 | Seeds 0–4 |
| Jul 16 | `20260716_115150_*` | 8 (same first-batch datasets) | 6 | Seeds 0–4 |
| Jul 16 | `20260716_144626_saprpt_*` | 7 remaining | SAP-RPT-1 only | Seeds 0–4 |

The July 5 run was always a **single-seed sweep** (seed 0 only) across all 15 datasets —
it was not a 5-seed run that crashed. The July 16 re-run (which added seeds 1–4) only
covered 8 datasets and then stopped. `run_saprpt_missing.py` added SAP-RPT-1 seeds 0–4
to the 7 remaining datasets, but **never added seeds 1–4 for the 5 baseline models** on
those same 7 datasets.

Result: SAP-RPT-1 had 75 pairs (15 × 5), every other model had only 47 pairs (8 × 5 + 7 × 1).

Computing mean AUC over different dataset×seed subsets produces numbers that cannot be
compared — the original "AUC=0.891 but ranked #2" headline was this artifact.

**✅ Fixed July 20, 2026:** `experiments/run_missing_seeds.py` ran seeds 1–4 for all 5 baseline
models on the 7 previously incomplete datasets. results.csv is now fully balanced at 450 rows.

**Final leaderboard (all 75 dataset×seed pairs, all 6 models):**

| Model | Mean AUC | Mean Rank |
|---|---|---|
| TabPFN | **0.9121** | **1.84** |
| SAP-RPT-1 | 0.8906 | 2.78 |
| CatBoost | 0.8782 | 3.29 |
| XGBoost | 0.8752 | 3.95 |
| LightGBM | 0.8744 | 4.11 |
| MLP | 0.8499 | 5.03 |

AUC and rank orderings are consistent. TabPFN is #1 by both measures.
Note: TabPFN's AUC dropped from the earlier incomplete-data estimate of 0.932 to 0.912 because
the 7 newly completed datasets are harder on average for TabPFN.

---

### Finding 3 — High: PR description had incorrect significance claims

**Original PR description stated:**
- "Statistically tied with tuned CatBoost, XGBoost, LightGBM (Wilcoxon p > 0.05)"
- "Only significantly outperformed by TabPFN (p = 0.005)"

**Correct numbers from complete-case Wilcoxon analysis:**

| Comparison | p-value | Correct claim |
|---|---|---|
| TabPFN vs SAP-RPT-1 | < 0.0001 | TabPFN **significantly** better (66% win rate) |
| SAP-RPT-1 vs CatBoost | 0.0060 | SAP-RPT-1 significantly better |
| SAP-RPT-1 vs LightGBM | 0.0002 | SAP-RPT-1 significantly better |
| SAP-RPT-1 vs XGBoost | 0.0004 | SAP-RPT-1 significantly better |
| SAP-RPT-1 vs MLP | < 0.0001 | SAP-RPT-1 significantly better |
| CatBoost vs XGBoost | 0.6491 | Not significantly different |
| CatBoost vs LightGBM | 0.3019 | Not significantly different |
| XGBoost vs LightGBM | 0.9535 | Not significantly different |

PR description updated on GitHub to reflect these numbers (July 20, 2026).

---

### Audit: Open Tasks

| Task | Status | Notes |
|---|---|---|
| Fix per-model raw-string preprocessing for SAP-RPT-1 | ⏳ Pending | Needed before semantic arm |
| Run missing seeds 1–4 for 5 baseline models on 7 datasets | ✅ Done (July 20) | results.csv now 450 rows, 75/75 complete |
| Task 4: switch README to mean rank + significance framing | ⏳ Pending | |
| Task 5: add semantic dataset arm (adult, bank-marketing, credit-approval, dresses-sales) | 🔴 Blocked | Blocked on raw-string fix |
