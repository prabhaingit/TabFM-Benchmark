# SAP-RPT-1 OSS — Standalone Evaluation Plan

This document defines the 15 datasets to evaluate and the metrics to collect for a standalone SAP-RPT-1 project (outside the full benchmark comparison).

---

## Setup

```bash
cd tabfm-benchmark
source .env                    # loads HF token and TabPFN token
source venv/bin/activate       # activate virtualenv
```

Minimal standalone test script to validate environment:
```bash
python - << 'EOF'
import sys; sys.path.insert(0, '.')
from src.data.loader import DataLoader
from src.models.sap_rpt_model import SAPRptWrapper
from src.evaluation.metrics import compute_metrics

ds = DataLoader().load_openml(31, "credit-g")
model = SAPRptWrapper()
result = model.run(ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                   dataset_name="credit-g", seed=0)
print(compute_metrics(result))
EOF
```

---

## The 15 Datasets

| # | Name | OpenML ID | Rows | Features | Domain | Column Names | Notes |
|---|---|---|---|---|---|---|---|
| 1 | credit-g | 31 | 1,000 | 20 | Finance | Meaningful | Mixed categorical/numeric |
| 2 | diabetes | 37 | 768 | 8 | Healthcare | Meaningful | Pima Indians — small |
| 3 | spambase | 44 | 4,601 | 57 | NLP-derived | Generic (word freq.) | Large, many features |
| 4 | banknote-auth | 1462 | 1,372 | 4 | Security | Generic (V1–V4) | Very few features |
| 5 | hill-valley | 1479 | 1,212 | 100 | Synthetic | Generic (V1–V100) | Near feature limit |
| 6 | wdbc | 1510 | 569 | 30 | Healthcare | Generic (V1–V30) | Same data as breast_cancer |
| 7 | qsar-biodeg | 1494 | 1,055 | 41 | Chemistry | Mixed | Molecular features |
| 8 | titanic | 40945 | 891 | 11 | Historical | Meaningful | Very small N |
| 9 | ozone-level | 1487 | 2,536 | 72 | Environment | Meaningful | High feature/row ratio |
| 10 | kc1 | 1067 | 2,109 | 21 | Software | Meaningful | Highly imbalanced |
| 11 | pc1 | 1068 | 1,109 | 21 | Software | Meaningful | Similar domain to kc1 |
| 12 | sick | 1050 | 3,772 | 37 | Healthcare | Meaningful | Thyroid disease detection |
| 13 | phoneme | 1489 | 5,404 | 5 | Audio-derived | Generic | Very few features |
| 14 | australian | 40536 | 690 | 14 | Finance | Generic | Tiny N |
| 15 | kc2 | 1063 | 1,180 | 21 | Software | Meaningful | Same domain as kc1/pc1 |

**Column name quality matters most for SAP-RPT-1.** Datasets with meaningful names (credit-g, titanic, sick, diabetes) are expected to score higher than datasets with generic V1...Vn names (wdbc, banknote-auth, hill-valley).

---

## Metrics to Collect

For each dataset × seed combination, record all of the following.

### Primary Metric

| Metric | Field name | Description |
|---|---|---|
| **ROC-AUC** | `roc_auc` | Main ranking metric. Higher is better. Compare to random baseline of 0.5. |

### Secondary Metrics

| Metric | Field name | Description |
|---|---|---|
| Accuracy | `accuracy` | Raw correct predictions / total |
| F1 Score | `f1` | Harmonic mean of precision/recall. Useful for imbalanced datasets |
| Log-loss | `log_loss` | Measures calibration quality. Lower is better |
| Precision | `precision` | TP / (TP + FP) |
| Recall | `recall` | TP / (TP + FN) |

### Timing Metrics

| Metric | Field name | Description |
|---|---|---|
| Fit time (sec) | `fit_time_sec` | Time to store training context (includes calibration step) |
| Predict time (sec) | `predict_time_sec` | Time to run predict_proba on test set |
| Total wall time (sec) | `total_wall_time_sec` | fit + predict |

### Diagnostic Fields (SAP-RPT-1 specific)

These are not in the standard benchmark output. You need to capture them manually from the wrapper.

| Field | Source | Description |
|---|---|---|
| `pos_col` | `model._pos_col_` | Which column (0 or 1) the calibration chose as P(class=1) |
| `context_rows` | `len(model._X_fit_)` | Actual number of training rows used as context |
| `n_train` | `len(ds.y_train)` | Total training rows available |
| `col_names_type` | Manual | "meaningful" or "generic" — check OpenML column names |
| `calibration_margin` | Manual | AUC difference between col[0] and col[1] during calibration |

### Dataset Metadata (record once per dataset, not per seed)

| Field | Source | Description |
|---|---|---|
| `dataset_name` | config | Name string |
| `openml_id` | config | OpenML dataset ID |
| `n_rows` | `ds.n_rows` | Total rows (train + test) |
| `n_features` | `ds.n_features` | Number of features |
| `class_balance` | `ds.class_balance` | Fraction of positive class (0.5 = balanced) |
| `n_cat_features` | Computed | Number of categorical columns (SAP-RPT-1 sensitive to this) |

---

## Recommended Evaluation Script

Save as `experiments/run_sap_rpt_standalone.py`:

```python
"""
Standalone SAP-RPT-1 evaluation across all 15 benchmark datasets.
Captures standard metrics plus SAP-RPT-1-specific diagnostics.
"""
import sys, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.models.sap_rpt_model import SAPRptWrapper
from src.evaluation.metrics import compute_metrics

DATASETS = [
    (31,    "credit-g"),
    (37,    "diabetes"),
    (44,    "spambase"),
    (1462,  "banknote-auth"),
    (1479,  "hill-valley"),
    (1510,  "wdbc"),
    (1494,  "qsar-biodeg"),
    (40945, "titanic"),
    (1487,  "ozone-level"),
    (1067,  "kc1"),
    (1068,  "pc1"),
    (1050,  "sick"),
    (1489,  "phoneme"),
    (40536, "australian"),
    (1063,  "kc2"),
]

N_SEEDS = 5
SEEDS   = list(range(N_SEEDS))

loader = DataLoader(test_size=0.2, random_state=42)
all_records = []

for ds_id, ds_name in DATASETS:
    print(f"\n{'='*50}")
    print(f"Dataset: {ds_name} (OpenML {ds_id})")
    ds = loader.load_openml(ds_id, ds_name)

    # Count categorical columns in raw OpenML data (approximate via n_features check)
    n_cat = sum(1 for c in ds.X_train.columns
                if ds.X_train[c].nunique() < 20 and ds.X_train[c].max() < 20)

    for seed in SEEDS:
        model = SAPRptWrapper(random_state=seed)
        result = model.run(
            ds.X_train, ds.y_train,
            ds.X_test,  ds.y_test,
            dataset_name=ds_name,
            seed=seed,
        )
        metrics = compute_metrics(result)

        record = {
            # Dataset metadata
            "dataset":       ds_name,
            "openml_id":     ds_id,
            "n_rows":        ds.n_rows,
            "n_features":    ds.n_features,
            "class_balance": round(ds.class_balance, 4),
            "n_cat_approx":  n_cat,
            # Run info
            "seed":          seed,
            "model":         "SAP-RPT-1",
            # Primary metric
            "roc_auc":       metrics.get("roc_auc"),
            # Secondary metrics
            "accuracy":      metrics.get("accuracy"),
            "f1":            metrics.get("f1"),
            "log_loss":      metrics.get("log_loss"),
            "precision":     metrics.get("precision"),
            "recall":        metrics.get("recall"),
            # Timing
            "fit_time_sec":       result.fit_time_sec,
            "predict_time_sec":   result.predict_time_sec,
            "total_wall_time_sec":result.total_wall_time_sec,
            # SAP-RPT-1 diagnostics
            "pos_col":       getattr(model, "_pos_col_", None),
            "context_rows":  len(model._X_fit_) if hasattr(model, "_X_fit_") else None,
            "n_train":       len(ds.y_train),
            "error":         result.error,
        }
        all_records.append(record)

        status = f"AUC={record['roc_auc']:.4f}" if record['error'] is None else f"ERROR"
        print(f"  seed={seed}  {status}  pos_col={record['pos_col']}  "
              f"ctx={record['context_rows']}  {result.fit_time_sec:.1f}s")

# Save results
Path("experiments/results/sap_rpt").mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv  = f"experiments/results/sap_rpt/sap_rpt_results_{ts}.csv"
out_json = f"experiments/results/sap_rpt/sap_rpt_results_{ts}.json"

df = pd.DataFrame(all_records)
df.to_csv(out_csv, index=False)
with open(out_json, "w") as f:
    json.dump(all_records, f, indent=2, default=str)

print(f"\nResults saved: {out_csv}")

# Quick summary
print("\n--- AUC Summary (mean ± std across seeds) ---")
summary = (df[df["error"].isna()]
           .groupby("dataset")["roc_auc"]
           .agg(["mean","std","min","max"])
           .round(4)
           .sort_values("mean", ascending=False))
print(summary.to_string())
```

---

## Key Questions to Answer From the Results

After running, look for answers to these:

1. **Does calibration always pick the right column?**  
   Check `pos_col` across seeds. If it flips (0 on seed 0, 1 on seed 1) for the same dataset, calibration is unreliable for that dataset.

2. **Do meaningful column names consistently outperform generic names?**  
   Group by column name type. credit-g/titanic/sick (meaningful) should beat wdbc/banknote-auth (generic V1...Vn).

3. **Does the model beat random (AUC > 0.5) on all datasets?**  
   Any dataset below 0.5 means calibration failed — the wrong column was chosen.

4. **How does context size interact with dataset size?**  
   Datasets with > 8192 rows trigger subsampling (none in this benchmark). Datasets < 200 rows may be too small for reliable calibration.

5. **Which datasets does SAP-RPT-1 handle well?**  
   Look for datasets where AUC > 0.70 — these are likely to have meaningful column names AND a tabular structure the LLM understands.

---

## Expected Performance Ranges (Updated After y-Index Bug Fix)

**Note:** A critical bug was found and fixed (see `SAP_RPT1_INVESTIGATION.md` Section 12). Prior estimates were wrong because `y_train` numpy array indices were misaligned with `X_train`'s pandas index inside the model. All prior AUC numbers below 0.70 on datasets with meaningful column names are suspect — they were near-random predictions. The ranges below are revised estimates.

| Dataset | Expected AUC Range | Reason |
|---|---|---|
| credit-g | **0.75–0.82** | Validated: 0.776 via benchmark DataLoader, 0.804 raw |
| diabetes | 0.65–0.80 | Small, meaningful medical names — should be strong |
| wdbc | **0.97–1.00** | Validated post-fix: 0.9983; generic names do not hurt as much as prior theory predicted |
| titanic | 0.75–0.88 | Meaningful names, rich semantic content |
| sick | 0.70–0.85 | Healthcare domain, meaningful names |
| spambase | 0.55–0.70 | Feature names are word frequencies — semi-meaningful |
| banknote-auth | 0.50–0.65 | Only 4 generic features |
| hill-valley | 0.45–0.60 | 100 generic features, synthetic |

These are rough estimates — the actual results may differ. The column name effect can shift AUC by 20–40 points. The key improvement from the bug fix is most visible on datasets with meaningful column names.

---

## What Good Results Look Like

| Scenario | What it means |
|---|---|
| AUC > 0.70 on semantic datasets | Model is leveraging column/cell semantics well |
| AUC 0.55–0.70 on encoded data | Encoding hurts but model still discriminates |
| AUC ≈ 0.50 | Calibration may have failed or model has no signal |
| AUC < 0.50 | Wrong column was selected — calibration margin was too small |
| `pos_col` flips across seeds | Calibration is unreliable for this dataset |
