# TabFM Benchmark — How It Works

A reference guide for understanding the benchmark and contributing new models.

---

## The Big Picture

The benchmark answers: **how do tabular ML models compare across many real-world datasets?**

It runs every model on every dataset multiple times (5 random seeds), records performance + timing + memory, then does statistical analysis to find significant differences.

---

## Step 1: Data Loading (`src/data/loader.py`)

Every dataset comes from **OpenML** (a public dataset repository). The loader:

1. Downloads the dataset by its OpenML ID
2. Does **minimal preprocessing** (intentional — tests models on messy real data):
   - Encodes categorical columns as integers (ordinal)
   - Fills missing values with median (numeric) or mode (categorical)
   - Label-encodes the target to 0/1
3. Does a **stratified 80/20 train/test split** with a **fixed seed (42)** — every model sees the identical train and test data

---

## Step 2: The Base Contract (`src/models/base.py`)

Every model **must** implement two methods:
- `fit(X_train, y_train)` — train the model
- `predict_proba(X_test)` — return probability scores (1D array, positive class)

The `run()` method in the base class wraps these calls and automatically records:

| Field | What it captures |
|---|---|
| `fit_time_sec` | How long training took |
| `predict_time_sec` | How long inference took |
| `peak_memory_mb` | RAM used at peak |
| `tuning_time_sec_` | Time spent searching for hyperparameters |
| `total_wall_time_sec` | tuning + fit + predict (the "real" practitioner cost) |

---

## Step 3: How Each Model Works

There are two fundamentally different types:

### Zero-shot / Foundation Models — no tuning, instant

| Model | File | How it works |
|---|---|---|
| **TabPFN** | `tabpfn_model.py` | `fit()` just *stores* training data as context. No gradient training. At predict time, a pretrained transformer does in-context learning. `tuning_time = 0`. |
| **TabFM (Google)** | `tabfm_model.py` | Same idea — zero-shot in-context learning. Currently commented out in the runner (not working yet). |
| **SAP-RPT-1** | `sap_rpt_model.py` | ConTextTab — semantics-aware in-context learner. Uses column names and cell values via an LLM embedding layer. `fit()` stores context, no gradient training. `tuning_time = 0`. |

### Tuned ML Models — hyperparameter search first, then train

| Model | File | How it works |
|---|---|---|
| **XGBoost** | `xgboost_model.py` | Runs **Optuna** (Bayesian search) with 15–50 trials. Each trial does 3-fold CV on training set to score parameters. After finding best params, trains the final model. |
| **LightGBM** | `lightgbm_model.py` | Same Optuna + 3-fold CV pattern. |
| **CatBoost** | `catboost_model.py` | Same Optuna + 3-fold CV pattern. |
| **MLP (sklearn)** | `mlp_model.py` | Same Optuna + 3-fold CV, fewer trials (10). |

> **Key insight**: `tuning_time_sec_` is non-zero for all tuned models. That cost is counted in comparisons. TabPFN's advantage is zero tuning cost — it's ready instantly.

---

## Step 4: The Benchmark Loop (`experiments/run_benchmark.py`)

```
For each dataset:
    For each seed (0..4):         # 5 seeds for statistical stability
        For each model:
            model.run(X_train, y_train, X_test, y_test)
            → compute_metrics(result)
            → save to JSON
    → save aggregated CSV
    → regenerate all charts
```

- **Incremental** — already-run datasets are skipped automatically (checks for JSON in `experiments/results/raw/`)
- **Crash-safe** — saves raw JSON after each dataset so no work is lost
- Use `--force` flag to re-run datasets that already have results

---

## Step 5: Metrics (`src/evaluation/metrics.py`)

After each model run, `compute_metrics()` computes:

| Category | Metrics |
|---|---|
| **Core** | ROC-AUC, Average Precision, F1-macro, Brier Score, Log Loss |
| **Calibration** | ECE (10 bins), ECE (15 bins), Max Calibration Error |
| **Per-class** | Precision, Recall, F1 for each class |
| **Fairness** | Balanced accuracy, Matthews Correlation Coefficient, demographic parity |
| **Efficiency** | fit time, predict time, tuning time, total wall time, rows/sec, MB RAM |

---

## Step 6: Statistical Analysis & Reports

After all datasets complete:

1. **Average rank table** — ranks each model per dataset per metric
2. **Friedman test** — is there a statistically significant overall difference?
3. **Nemenyi post-hoc** — pairwise significance testing with Critical Difference diagrams
4. **Wilcoxon tests** — pairwise signed-rank tests
5. **Bootstrap CIs** — 95% confidence intervals on mean AUC per model
6. **Visualizations** saved to `reports/figures/`

---

## Adding a New Model (e.g., `sap-rpt-1-oss`)

Three files to touch:

### 1. Create `src/models/sap_rpt_model.py`

```python
from .base import ModelWrapper
import numpy as np

class SAPRptWrapper(ModelWrapper):
    def __init__(self, random_state: int = 42):
        super().__init__(name="SAP-RPT-1", random_state=random_state)
        self.tuning_time_sec_ = 0.0  # set to 0 if zero-shot

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # For zero-shot: load the pretrained model and store context
        # For fine-tuned: run gradient training here
        ...

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        # Must return a 1D array of positive class probabilities
        # shape: (n_samples,)
        ...
```

### 2. Add to `configs/models.yaml`

```yaml
sap_rpt:
  class: src.models.sap_rpt_model.SAPRptWrapper
  description: "SAP tabular foundation model (sap-rpt-1-oss)"
  tunable: false  # set true if it uses Optuna tuning
```

### 3. Wire into `experiments/run_benchmark.py`

In the `build_models()` function, add the import and include the wrapper:

```python
from src.models.sap_rpt_model import SAPRptWrapper

def build_models(exp_cfg, seed):
    return [
        TabPFNWrapper(...),
        XGBoostWrapper(...),
        LightGBMWrapper(...),
        CatBoostWrapper(...),
        MLPWrapper(...),
        SAPRptWrapper(random_state=seed),  # <-- add this
    ]
```

The harness handles everything else — timing, metrics, visualization, statistical tests — automatically.

---

## Key Question Before Writing the Wrapper

**Is `sap-rpt-1-oss` zero-shot or does it fine-tune?**

- **Zero-shot** (like TabPFN): `fit()` stores context, `tuning_time_sec_ = 0.0`
- **Fine-tuned** (like XGBoost): `fit()` runs gradient descent, set `tuning_time_sec_` appropriately

This determines how the timing comparison is reported and how fair the comparison is against other models.

---

## File Structure Reference

```
tabfm-benchmark/
├── configs/
│   ├── datasets.yaml        # 15 OpenML datasets with IDs
│   ├── models.yaml          # model configs + hyperparameter search spaces
│   └── experiment.yaml      # seeds, splits, timeouts, metrics list
├── src/
│   ├── data/loader.py       # OpenML download + preprocessing
│   ├── models/
│   │   ├── base.py          # ModelWrapper + RunResult (start here)
│   │   ├── tabpfn_model.py  # zero-shot example
│   │   ├── xgboost_model.py # tuned model example
│   │   ├── tabfm_model.py   # Google TabFM (disabled)
│   │   └── ...
│   └── evaluation/
│       ├── metrics.py       # all metric computation
│       └── statistical.py   # Friedman, Nemenyi, Wilcoxon, bootstrap CI
├── experiments/
│   └── run_benchmark.py     # main entry point
└── reports/figures/         # generated charts
```
