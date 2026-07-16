# SAP-RPT-1 OSS: Investigation and Learnings

A complete record of every experiment run, what was discovered, and the current state of the wrapper.

---

## 1. What SAP-RPT-1 OSS Actually Is

SAP-RPT-1 OSS (also called ConTextTab) is a **zero-shot tabular in-context learner**. It works like this:

1. `fit(X_train, y_train)` — stores the training rows as "examples" inside the model. No gradient-based learning happens. This is identical in spirit to TabPFN.
2. `predict_proba(X_test)` — for each test row, the model constructs a prompt that combines all training examples plus the test row, runs it through an LLM, and produces a probability estimate.

The critical design property: **column names and cell values are embedded via an LLM**. A feature named `checking_status` with value `"no checking"` contributes semantic meaning that a feature named `V1` with value `2.0` does not. This is fundamentally different from tree models or TabPFN, which treat all inputs as opaque numbers.

---

## 2. What the Benchmark's DataLoader Does (and Why It Matters)

The DataLoader (`src/data/loader.py`) pre-processes every dataset before passing it to any model:

- **Categoricals → ordinal integers**: e.g. `"male"` → `0.0`, `"female"` → `1.0`
- **Missing values → median/mode fill**
- **All columns cast to float32**
- The output is a **pandas DataFrame with float32 values and the original column names**

This preprocessing is correct and necessary for tree models and TabPFN. For SAP-RPT-1 it creates a problem: the model's LLM never sees strings like `"no checking"` or `"critical/other existing credit"` — it sees `3.0` and `0.0`. The column names survive (e.g. `checking_status`), but the cell-level semantics are gone.

---

## 3. Experiment 1 — First Run (AUC 0.457, below random)

**Setup:** Passed encoded DataLoader output directly. Returned `proba[:, 1]` (sklearn default).

**Result:** AUC = 0.457 on credit-g. Below random.

**Diagnosis attempted at the time:**
- Checked `classes_` attribute → `[0, 1]`. Correct.
- Tried using `classes_.index(1)` to pick column → same result, still 0.457.
- Tried correlation of training probabilities with y_train → misleading because the model has training data in context ("memorized"), so col[1] looked correct on training data but was wrong on test data.
- Tried small held-out calibration (15% split) → too noisy, ~120 samples, difference was 0.54 vs 0.46.

**First fix applied (wrong):** Hard-coded `return proba[:, 0]`. Got AUC 0.543, which is above random. Committed this under the belief that col[0] is consistently P(class=1).

---

## 4. Experiment 2 — The breast_cancer Falsification Test

**Setup:** Ran the exact README example — `sklearn.datasets.load_breast_cancer()` as a numpy array, 50/50 split, `max_context_size=512`.

**Result:**
```
AUC col[:,0] = 0.0008   (catastrophically wrong)
AUC col[:,1] = 0.9992   (correct)
predict() accuracy = 0.979
```

The `predict()` function returns `classes_[argmax(proba)]`, and `argmax(proba)` = col[1] for positive predictions. This confirms **col[1] = P(class=1) by standard sklearn convention**.

**Conclusion:** The hard-coded `col[0]` fix would have been catastrophically wrong on any dataset where the column convention holds normally. The previous 0.543 result on credit-g was a coincidence — 1 − 0.457 = 0.543 (symmetric around random).

---

## 5. Experiment 3 — Input Format Effect

**Question:** Does the convention change based on whether the input is a numpy array vs a pandas DataFrame?

| Format | Dataset | AUC col[0] | AUC col[1] |
|---|---|---|---|
| numpy array | breast_cancer | 0.0008 | **0.9992** |
| float64 DataFrame | breast_cancer | 0.22 | **0.78** |
| float32 DataFrame | breast_cancer | 0.22 | **0.78** |
| raw DataFrame (strings) | credit-g | 0.38 | **0.62** ← wait |
| float32 DataFrame | credit-g | 0.38 | **0.62** |

Wait — this changes the picture. When input is a DataFrame (any variant), col[1] is still the better column for breast_cancer (0.78 vs 0.22). But there's a big performance drop: 0.9992 → 0.78 when switching from numpy to DataFrame.

For credit-g with a DataFrame, col[1] = 0.35 and col[0] = 0.65 — so col[0] wins here.

**The column convention is dataset-specific, not format-specific.** For breast_cancer, col[1] always wins. For credit-g, col[0] wins. No single hard-coded choice works.

---

## 6. Experiment 4 — The Column Calibration Solution

**Problem:** We cannot hard-code col[0] or col[1] globally.

**Approach tested:** After `fit()`, run `predict_proba()` on 50 training samples and compute AUC for both columns against their true labels. Pick the column with higher AUC.

**Why this works:** The model has the training data stored in its context. Predicting on training samples is an in-sample correlation check — the column that is positively correlated with y on the training set is the one to use for test predictions.

**Results:**

| Dataset | Cal AUC col[0] | Cal AUC col[1] | Chosen | Test AUC |
|---|---|---|---|---|
| credit-g (encoded DF) | 0.575 | 0.425 | col[0] | 0.543 |
| wdbc (encoded DF) | 0.395 | 0.605 | col[1] | 0.594 |
| breast_cancer (float32 DF) | 0.395 | 0.605 | col[1] | 0.785 |

Calibration correctly identifies the right column in all three cases.

**Tie-guard added:** When `|AUC_col1 - AUC_col0| < 0.1`, the signal is statistically unreliable (standard error on 50 samples is ~0.14). Fall back to col[1] (sklearn default).

---

## 7. Experiment 5 — Raw Data vs Encoded Data

**Question:** How much does encoding hurt?

**Controlled test on credit-g (same split, same context size):**

| Data format | AUC col[0] (best) |
|---|---|
| Raw DataFrame (string categoricals) | 0.650 |
| Encoded DataFrame (ordinal float32) | 0.617 |

**Difference: ~0.03.** Smaller than expected.

**Key insight:** The encoding effect is real but not the dominant factor. The bigger performance driver is **column names**, not cell values.

---

## 8. The Column Name Effect — Revised After Bug Fix

**RETRACTION:** The 40-point gap we attributed to column names was caused by the y_train index bug (Section 12), not column names. After fixing the wrapper:

| Source | Column names | y type | AUC |
|---|---|---|---|
| sklearn breast_cancer (numpy) | `"mean radius"`, etc. | numpy (0-based index → accidentally correct) | 0.9992 |
| OpenML wdbc (DataLoader, pre-fix) | `"V1"`, `"V2"`, etc. | numpy misaligned | 0.594 |
| OpenML wdbc (DataLoader, **post-fix**) | `"V1"`, `"V2"`, etc. | pandas Series (correct index) | **0.9983** |

The "40-point gap" shrinks to ~0.001 once y is correctly aligned. The model achieves near-identical AUC (≥0.998) on the same data regardless of whether columns are named "mean radius" or "V1".

**Why sklearn numpy worked:** `sklearn.datasets.load_breast_cancer()` returns numpy arrays. After `train_test_split`, both X and y have 0-based indices. When sap_rpt_oss converts them to pandas internally, indices align automatically — a lucky coincidence.

**The column name hypothesis is not refuted** — it is plausible that semantics helps for harder tasks — but the evidence we had for it was confounded. The full 15-dataset results should be used to re-assess this hypothesis.

---

## 9. Context Size Effect — Invalidated by Bug Fix

The context size ablation in this section was run with y_train as a numpy array (misaligned). All AUC values below were near-random and cannot be interpreted as a context-size effect.

**Prior (invalid) ablation on wdbc (455 rows, y misaligned):**

| max_context_size | AUC col[1] |
|---|---|
| 128 | 0.42 |
| 512 | 0.62 |

**Post-fix result with ctx=8192:** AUC=0.9983. The "internal effect" of ctx parameter was not real — the apparent variation was noise from random near-random predictions. The benchmark config (ctx=8192) is fine.

---

## 10. Current Wrapper Implementation

File: `src/models/sap_rpt_model.py`

**Key design decisions:**

| Decision | Choice | Reason |
|---|---|---|
| y_train index fix | Wrap numpy y as pd.Series with X.index | Root cause of all prior failures — sap_rpt_oss aligns by pandas index |
| Context size | 8192 (config) | Max quality; model truncates internally if needed |
| Bagging | 1 | No measurable AUC effect (tested on credit-g); bagging=8 not needed |
| Column calibration | Auto-detect via 50-sample in-sample check | Safeguard; correctly picks col[1] once y is aligned |
| Tie-guard threshold | 0.1 AUC margin | Below this, 50 samples give ~±0.14 SE — statistically unreliable |
| Subsampling | Balanced stratified (50/50 classes) | Only triggered when train > 8192 rows |

**Flow:**
```
fit(X_train, y_train):
  → subsample if len > 8192
  → fit SAP_RPT_OSS_Classifier
  → run _calibrate_pos_col (50-sample AUC check)
  → store _pos_col_

predict_proba(X_test):
  → model.predict_proba(X_test)
  → return proba[:, _pos_col_]
```

---

## 11. Known Limitations in This Benchmark

1. **Ordinal encoding effect is negligible.** Post-fix, ordinal float32 encoding produces AUC within ±0.03 of raw strings. The "encoding hurts" conclusion from earlier experiments was confounded by the y-index bug. The DataLoader pipeline is fine for this model.

2. **Generic OpenML column names.** Effect is unknown until the full 15-dataset run with the fixed wrapper. The wdbc result (0.998 with generic V1...V30) suggests column names may matter less than hypothesized. The hypothesis stands but needs fresh evidence.

3. **Calibration relies on 50 samples.** With very imbalanced datasets or very small training sets, the column check may be noisy. The tie-guard mitigates this but doesn't eliminate it.

4. **Context size interaction.** `max_context_size` affects model internals beyond truncation. The current value (8192) was not exhaustively tuned.

5. **Model comparison is not fully apples-to-apples (minor).** Tree models benefit from the DataLoader's ordinal encoding; SAP-RPT-1 shows a small (~0.03 AUC) penalty from encoding vs raw string data. This is measurable but small compared to the y-index issue.

---

## 12. Root Cause Discovery — y_train Index Misalignment (Section Added After Kaggle Comparison)

**Symptom:** Kaggle notebook achieved AUC=0.8043 on credit-g with `bagging=8, max_context_size=8192`. Our benchmark wrapper achieved only 0.543 — a 26-point gap.

**Initial theories (all wrong or secondary):**
- Encoding effect: ~0.03 AUC — too small to explain 26 points
- bagging=8 vs bagging=1: zero AUC effect once y is correct
- Model checkpoint difference: same file (`2025-11-04_sap-rpt-one-oss.pt`, 64.6 MB) in both environments

**Actual root cause:** `sap_rpt_oss.fit()` does **index-based alignment** of X and y internally. When y is a numpy array (DataLoader's output), pandas creates a default Series with index `[0, 1, 2, ...]`. But `X_train` after `train_test_split` has a **shuffled original pandas index** (e.g., `[675, 703, 12, 845, ...]`). The model then misaligns every label with a different row, producing near-random predictions.

**Proof (one-variable test):**

| y_train type | AUC col[0] | AUC col[1] |
|---|---|---|
| pandas Series (Kaggle exact) | 0.1957 | **0.8043** |
| numpy array (DataLoader output) | 0.4312 | 0.5688 |
| numpy rewrapped as Series with X_train.index | 0.1968 | **0.8032** |

**Fix:** In `fit()`, convert numpy y to `pd.Series(y, index=X_train.index)` before calling the model.

**Encoding effect with correct y alignment:**

| Data format | y type | AUC col[1] |
|---|---|---|
| raw Categorical strings | pandas Series | 0.8043 |
| ordinal float32 (DataLoader encoded) | pandas Series (correct index) | **0.8109** |
| ordinal float32 (DataLoader encoded) | numpy array (broken, pre-fix) | 0.4852 |

The encoding effect is negligible and slightly positive for credit-g. The 26-point gap was entirely from y-index misalignment.

**Benchmark result after fix:** AUC = **0.776** (benchmark DataLoader, fixed wrapper) vs 0.543 before.

---

## 13. Summary of AUC Results Observed

| Dataset | Format | y type | col chosen | AUC | Notes |
|---|---|---|---|---|---|
| credit-g | encoded DF, ctx=8192 | numpy (broken) | col[0] | 0.543 | **Pre-fix** |
| credit-g | encoded DF, ctx=8192 | pandas Series | col[1] | **0.776** | **Post-fix via benchmark** |
| credit-g | raw DF (strings), ctx=8192 | pandas Series | col[1] | **0.804** | Kaggle result, reproduced |
| wdbc | encoded DF, ctx=8192 | numpy (broken) | col[1] | 0.594 | Pre-fix — needs re-run |
| breast_cancer | numpy, ctx=512 | numpy | col[1] | 0.9992 | numpy arrays have 0-based index matching — works by coincidence |
| breast_cancer | float32 DF, ctx=512 | pandas Series | col[1] | 0.785 | Needs re-run with correct y |
