# SAP-RPT-1 Benchmark Audit — Instructions for Claude Code

**Repo:** `tabfm-benchmark` (also check `tabfm-benchmark-backup` if `main` is ahead)
**Goal:** Audit and extend the SAP-RPT-1 benchmark contribution to determine why SAP-RPT-1 underperforms (or only ties) TabPFN on our 15-dataset benchmark, cross-checking against the ConTextTab paper (arXiv:2506.10707, the model's own paper).

## Context you need before starting

The ConTextTab paper's central finding is that SAP-RPT-1 (= ConTextTab) achieves SOTA **only on the semantically rich CARTE benchmark** (string/text-heavy tables). On the OpenML-CC18-style benchmarks — which is exactly where our 15 datasets come from — the paper's own results show **TabPFN ranks ahead of ConTextTab** (mean rank 2.89 vs 3.56 on OpenML-CC18). So our "TabPFN slightly ahead" result may be a *correct replication of the paper*, not a bug. However, there are several ways our harness could be unfairly penalizing SAP-RPT-1 on top of that, and those are what this audit exists to find.

Key numbers from the paper to keep in mind:
- Below ~1000 training rows, TabPFN is strongest; ConTextTab overtakes it only above ~1000 rows. Half our datasets are near or below that line.
- The column-name semantic effect is real but **modest** (~1% accuracy / ~2% R²), NOT the "30-40 point" effect assumed in our internal eval plan.
- Replacing categorical/string features with ordinal encoding costs ConTextTab **-2.7 accuracy / -4.8 R²** on semantic data — this is the single biggest risk in our pipeline.
- The paper's default config is `max_context_size=8192, bagging=8`. Reducing either measurably hurts rank.
- The paper reports **mean rank + Wilcoxon significance + critical-difference diagrams**, not naive mean-AUC leaderboards.

---

## GROUND RULES (apply to every task)

1. **Read before writing.** Before making ANY code changes, do a read-only pass over `src/data/loader.py`, `src/models/sap_rpt_model.py`, and `experiments/run_benchmark.py`, and summarize the current behavior back to me.
2. **Do not overwrite baselines.** Do NOT modify the existing 15-dataset results or overwrite `experiments/results/aggregated/results.csv` in place. Write all new/corrected outputs to `experiments/results/sap_rpt_audit/`. This is required by CONTRIBUTING.md's "baseline numbers NOT overwritten" rule.
3. **Document as you go.** Produce a written findings report at `experiments/results/sap_rpt_audit/AUDIT_REPORT.md`. Don't fix things silently — record what you found at each step, with the actual numbers/dtypes you observed.
4. **Respect task ordering.** Tasks 1-4 can inform each other, but **Task 5 must not run until Task 2 has reported back** on the encoding question (explained in Task 5).

---

## TASK 1 — Audit results coverage and recompute fair comparisons

1. Load `experiments/results/aggregated/results.csv`.
2. Build a pivot: rows = `(dataset, seed)`, columns = `model`, values = `roc_auc`.
3. Report:
   - How many `(dataset, seed)` cells exist per model (expect 15 × 5 = 75 each).
   - Which specific `(dataset, seed)` pairs are missing for SAP-RPT-1 vs TabPFN.
   - Confirm whether the 7 datasets run via `run_saprpt_missing.py` (australian, kc1, kc2, ozone-level, pc1, phoneme, sick) used the **same seeds (0-4)** and **same train/test split logic** as the main run.
4. Recompute mean AUC for every model using **ONLY** `(dataset, seed)` pairs where ALL 6 models have a non-null result (inner join / complete cases).
5. Recompute using **mean RANK** instead of mean AUC: for each `(dataset, seed)`, rank the 6 models 1 (best) to 6 (worst) by AUC, then average ranks per model. Report both the mean-AUC and mean-rank leaderboards side by side, and flag if the ordering changes.
6. Run a paired **Wilcoxon signed-rank test** between SAP-RPT-1 and TabPFN on the complete-case AUC pairs. Report the p-value and win rate (fraction of dataset-seed pairs where SAP-RPT-1 > TabPFN).

**Why this matters:** Our summary claims SAP-RPT-1 mean AUC = 0.891 (higher than TabPFN's 0.8847) yet labels TabPFN #1. Those are inconsistent. Likely cause: the means are computed over non-identical dataset×seed sets (results.csv has 310 rows, not the expected 450, with aborted runs excluded and 7 datasets run separately). Complete-case recomputation resolves this.

**Output:** A corrected leaderboard table (AUC + rank side by side) + a written note on whether the original "AUC=0.891 but #2" claim holds up.

---

## TASK 2 — Check whether categorical/semantic data survives to the model

**This is the most important task in the audit.**

1. Open `src/data/loader.py` (`DataLoader` class) and trace exactly what dtype `X_train` has by the time it's returned. Is `fetch_openml`'s `as_frame=True` output preserved, or is there an encoding step (`LabelEncoder`, `OrdinalEncoder`, `get_dummies`, `pd.factorize`, `.cat.codes`, etc.) applied before or inside `DataLoader`?
2. Open `src/models/sap_rpt_model.py` (`SAPRptWrapper`) and check the same thing at the point `.fit()` is called — print/log `X_train.dtypes` and a few sample rows for **credit-g** and **titanic** right before the call into `sap_rpt_oss`.
3. Specifically check: does `checking_status` in credit-g arrive as strings like `"no checking account"` / `"<0"`, or as integer codes `0/1/2/3`? Same check for titanic's `sex`, `embarked`, `pclass`.
4. **If encoding IS happening before the model sees the data:** this reproduces the paper's "no feature semantics - ordinal encoder" ablation condition (costs ConTextTab -2.7 accuracy / -4.8 R² on semantic data). Flag this as a likely root cause and note where in the pipeline to bypass encoding **specifically for the SAP-RPT-1 model path**. Other models in the benchmark may legitimately need encoded input, so this is probably a per-model branch in `run_benchmark.py` or a preprocessing flag, NOT a global loader change.
5. Also check whether column headers reach the model unchanged (e.g. not renamed to `col_0, col_1...` anywhere in the pipeline).

**Output:** A dtype/sample dump for 2-3 datasets, and a clear **yes/no** on whether semantic information is being destroyed before SAP-RPT-1 sees it. This yes/no determines whether Task 5 can proceed as-is or needs a pipeline fix first.

---

## TASK 3 — Confirm configuration consistency across all 15 datasets

1. Grep the codebase for every place `max_context_size` and `bagging` are set for SAP-RPT-1 (`configs/models.yaml`, `src/models/sap_rpt_model.py`, `experiments/run_saprpt_missing.py`, `experiments/run_benchmark.py`).
2. Build a table: `dataset | max_context_size used | bagging used | subsampling triggered (Y/N) | reason`.
3. Confirm this matches what's documented for **australian** (`max_context_size=2048` due to OOM). Check if any OTHER dataset silently fell back to reduced settings without being noted — especially the 7 datasets run via the separate `run_saprpt_missing.py` script, which may have different defaults than the main run.
4. Cross-check against `sap_rpt_model.py`'s `_subsample_balanced`: for any dataset where `len(X_train) > max_context_size`, report how many rows were actually used as context vs available.

**Output:** The config table above, as a section in the audit report. Going forward, `effective_context_size` and `bagging_factor` should be required columns in the results CSV if they aren't already.

---

## TASK 4 — Reframe the headline comparison around rank + significance

1. Add/extend a script `experiments/analyze_results.py` to compute, for the full 6-model × 15-dataset × 5-seed grid:
   - Mean AUC (existing)
   - Mean rank per model (new, per Task 1 method)
   - Wilcoxon p-value and win-rate matrix between all model pairs (like the paper's Figure 5)
2. Update `README.md`'s Benchmark Results section: keep the AUC table but add a rank column, and change the "SAP-RPT-1 ranks #2 by AUC" language to state whichever framing (rank or AUC) is actually consistent after Task 1, plus the significance caveat (e.g. "statistically tied with TabPFN, XGBoost, LightGBM, CatBoost per Wilcoxon p>0.05").

**Output:** Updated results table + a short paragraph matching the paper's own framing style (rank + significance caveat, not just point AUC).

---

## TASK 5 — Add a semantic-dataset arm to actually test the paper's claim

**Do not start this task until Task 2 has reported the encoding yes/no.**

**Why:** Our current 15 datasets are all numeric, small-N, generic-or-header-only-semantics tables — the profile where the paper says TabPFN ties or wins. They cannot surface SAP-RPT-1's actual differentiator (string/text cell semantics). This arm adds the missing condition.

### Datasets to add (all binary classification, all OpenML-hosted, verified)

| name            | data_id | target col | rows   | cat feat | num feat | why chosen |
|-----------------|---------|------------|--------|----------|----------|------------|
| adult           | 1590    | class      | 48,842 | 9        | 6        | occupation/workclass/education strings; also tests >8k-row regime |
| bank-marketing  | 1461    | Class      | 45,211 | 10       | 7        | job/marital/contact/month text fields; also >8k rows |
| credit-approval | 29      | class      | 690    | 10       | 6        | high cat ratio, small N — semantic + low-data (paper's ConTextTab sweet spot) |
| dresses-sales   | 23381   | Class      | 500    | 12       | 1        | 12 categorical / 1 numeric — most string-dominated; near-pure semantic test |

Optional 5th (mid-size semantic point, only if you want a data point between the extremes):

| name  | data_id | target col | rows  | cat feat | num feat | why |
|-------|---------|------------|-------|----------|----------|-----|
| churn | 40701   | class      | 5,000 | 5        | 16       | ~5k rows, sits right at/above the 1k crossover zone |

### Execution notes

1. Register these in the `DATASETS` list following the existing pattern in `experiments/run_benchmark.py` / `run_sap_rpt_standalone.py`.
2. Run all 6 models on this new arm, 5 seeds each, SAME pipeline.
3. **CRITICAL — the whole point of the arm:** SAP-RPT-1 must receive the **raw string/categorical columns** (e.g. adult's `occupation` as `"Exec-managerial"`, not an integer code). Confirm via the Task 2 dtype dump that these columns arrive as strings/objects at `.fit()`. **If Task 2 found that the loader label-encodes them first, FIX that for the SAP-RPT-1 path BEFORE running this arm** — otherwise this arm measures nothing and just reproduces the non-semantic condition on different data. Other models keep their normal encoded input.
4. `adult` and `bank-marketing` exceed the 8192 context cap → subsampling WILL trigger. That's expected. Log `effective_context_size` (per Task 3), keep `bagging=8`, and do NOT silently drop to `context=2048`.
5. `dresses-sales` has ~15% missing values and 500 rows → strong test of the README's "missing values handled internally" claim in the low-data + high-semantic regime. Do NOT impute for the SAP-RPT-1 path.
6. `credit-approval` (id 29) has anonymized headers (`A1`..`A16`) but real categorical STRING VALUES → it isolates *cell-value* semantics from *header* semantics. Expect it to behave differently from adult/bank-marketing; it's a deliberate contrast, not a duplicate.
7. **Verify each at load:** print `bunch.details['name']` and assert the target is binary before running. Handle target mapping explicitly — don't assume: `adult` loads as `<=50K`/`>50K`; `bank-marketing` as `1`/`2` or `yes`/`no` depending on version; `dresses-sales` may load as `{0,1}` already.

### Reporting

Report the semantic arm split **two ways**, mirroring the paper's CARTE-vs-CC18 split:
- **(a)** all-4 semantic combined vs the original 15 non-semantic
- **(b)** small-semantic (credit-approval, dresses-sales) vs large-semantic (adult, bank-marketing) — because the paper predicts SAP-RPT-1 wins the small-semantic pair most decisively, while the large-semantic pair is confounded by the >8k-row scaling weakness.

Produce a combined table/chart showing SAP-RPT-1's relative rank in the semantic arm vs the non-semantic arm. **This is the actual test of the paper's core hypothesis.**

---

## Expected outcome

If the pipeline is fair (or made fair after Task 2), the prediction from the paper is precise:
- On the original 15 (non-semantic): TabPFN ties or slightly edges SAP-RPT-1 — **this is correct and expected, not a bug.**
- On the semantic arm (especially the small-semantic pair): SAP-RPT-1 should pull clearly ahead. On CARTE, ConTextTab beats TabPFN with a 96% win rate and a vanishingly small p-value.

If SAP-RPT-1 does NOT pull ahead on the semantic arm even after confirming raw strings reach the model, that's a genuine and reportable finding worth escalating.

---

## Priority order summary

1. Task 1 — recompute on complete cases (resolves the 0.891-but-#2 contradiction)
2. Task 2 — inspect what the loader actually feeds the model (strings vs codes) ← **gates Task 5**
3. Task 3 — confirm bagging=8 / context=8192 everywhere; document deviations
4. Task 4 — switch headline claims to mean rank + Wilcoxon significance
5. Task 5 — add the semantic arm to give the hypothesis a fair test
