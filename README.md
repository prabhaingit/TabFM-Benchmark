# TabFM Benchmark

A fair, reproducible benchmark comparing **tabular foundation models** (TabPFN) against traditional gradient boosted trees (XGBoost, LightGBM, CatBoost) and neural networks (MLP) on binary classification tasks.

## 📊 Overview

This project investigates the practical trade-offs between **zero-shot tabular foundation models** and **traditionally-tuned** machine learning approaches. The key question:

> Can a single zero-shot model (TabPFN) compete with extensively-tuned gradient boosted trees — without the computational cost of hyperparameter optimization?

### What's Being Compared

| Model | Type | Tuning Required | Notes |
|-------|------|-----------------|-------|
| **TabPFN** | Tabular Foundation Model | ❌ None (zero-shot) | Stores training data as context; no gradient-based training |
| **XGBoost** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Industry-standard gradient boosting |
| **LightGBM** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Fast histogram-based gradient boosting |
| **CatBoost** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Ordered boosting with symmetric trees |
| **MLP** | Neural Network | ✅ Optuna (30 trials) | Scikit-learn MLPClassifier |

### Key Metrics Tracked

#### Performance Metrics
- **ROC AUC**: Area under the ROC curve
- **Average Precision (PR-AUC)**: Area under the precision-recall curve
- **F1 Macro**: Harmonic mean of precision and recall (averaged across classes)
- **Brier Score**: Mean squared error of probability predictions
- **Log Loss**: Negative log-likelihood of predictions

#### Calibration Metrics (NEW in Phase 1)
- **ECE (Expected Calibration Error)**: Measures prediction confidence vs actual accuracy
  - Computed with 10-bin and 15-bin discretizations
- **MCE (Maximum Calibration Error)**: Worst-case calibration error across bins

#### Per-Class Metrics (NEW in Phase 1)
- **Precision/Recall per class**: Class 0 and Class 1 separately
- **F1 per class**: Class 0 and Class 1 separately
- **Sensitivity (TPR)**: True positive rate for positive class
- **Specificity (TNR)**: True negative rate for negative class
- **PPV (Positive Predictive Value)**: Precision for positive class

#### Fairness & Quality Metrics (NEW in Phase 1)
- **Balanced Accuracy**: Accuracy accounting for class imbalance
- **Matthews Correlation Coefficient (MCC)**: Correlation between predicted and actual labels
- **Demographic Parity**: Overall positive prediction rate
- **TPR/FPR Overall**: True/false positive rates

#### Timing & Resource Metrics
- **Tuning Time**: Hyperparameter optimization time
- **Fit Time**: Model training time
- **Predict Time**: Inference time
- **Total Wall Time**: End-to-end execution time
- **Peak Memory (MB)**: Maximum memory usage during execution
- **GPU Memory (MB)**: GPU memory allocated (if CUDA available)

#### Statistical Testing (Enhanced in Phase 1)
- **Wilcoxon signed-rank tests**: Pairwise significance testing
- **Friedman test**: Overall model comparison across datasets
- **Nemenyi post-hoc test**: Pairwise comparisons after Friedman
- **Bootstrap Confidence Intervals**: 95% CI for each metric
- **Critical Difference (CD) Diagrams**: Visual statistical significance

## 📁 Project Structure

```
tabfm-benchmark/
├── configs/               # Configuration files
│   ├── datasets.yaml     # 15 OpenML datasets
│   ├── models.yaml       # Model configs & hyperparameter search spaces
│   └── experiment.yaml   # Experiment settings
├── src/
│   ├── data/             # Data loading (OpenML)
│   ├── models/           # Model wrappers
│   │   ├── tabpfn_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   └── mlp_model.py
│   ├── evaluation/       # Metrics & statistical tests
│   │   ├── metrics.py    # Performance, calibration, fairness metrics
│   │   ├── statistical.py # Friedman, Nemenyi, bootstrap CI
│   │   └── profiling.py  # Memory & timing profiling
│   └── viz/             # Visualizations
│       ├── leaderboard.py # Heatmaps & ranking plots
│       ├── statistical.py # CD diagrams, confidence intervals
│       └── curves.py     # ROC/PR curves
├── experiments/
│   ├── run_benchmark.py # Main benchmark runner
│   └── results/         # Raw & aggregated results
├── reports/
│   └── figures/         # Generated visualizations
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Siri1702/tabfm-benchmark.git
cd tabfm-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Benchmark

```bash
# Run on specific datasets
python experiments/run_benchmark.py --datasets credit-g diabetes

# Run on all 15 datasets
python experiments/run_benchmark.py --all

# Quick test with fewer seeds
python experiments/run_benchmark.py --dataset credit-g --n_seeds 3
```

### Output

Results are saved to:
- `experiments/results/raw/` — Individual JSON results per dataset
- `experiments/results/aggregated/` — Aggregated CSV with all metrics
- `reports/figures/` — Visualizations including:
  - `auc_heatmap.png` — ROC AUC comparison across models/datasets
  - `average_ranks.png` — Model ranking visualization
  - `calibration_comparison.png` — ECE/MCE comparison plots
  - `confidence_intervals.png` — Bootstrap 95% CI plots
  - `memory_comparison.png` — Memory usage comparison
  - `timing_comparison.png` — Runtime comparison charts
  - `win_loss_matrix.png` — Pairwise win/loss/tie matrix

## 📈 Sample Results

> ⚠️ **Preliminary Results** — Based on 1 random seed (seed=0), 3 datasets. Full benchmark with 10 seeds on all 15 datasets coming soon.

### Performance: ROC AUC (mean across 3 datasets)

| Model | ROC AUC | vs XGBoost |
|-------|---------|------------|
| **TabPFN** | **0.8679** | -0.03% |
| XGBoost | 0.8682 | baseline |
| LightGBM | 0.8638 | -0.51% |
| CatBoost | 0.8598 | -0.97% |
| MLP | 0.8437 | -2.82% |

### Timing: Total Wall Time (mean, seconds)

| Model | Time (s) | Speedup vs XGBoost |
|-------|----------|-------------------|
| **TabPFN** | **6.28** | **16.6×** |
| LightGBM | 54.15 | 1.9× |
| MLP | 34.00 | 3.1× |
| XGBoost | 104.32 | 1.0× |
| CatBoost | 266.53 | 0.4× |

### Breakdown by Dataset (ROC AUC)

| Model | credit-g | diabetes | banknote-auth |
|-------|----------|----------|---------------|
| TabPFN | 0.7794 | 0.8243 | 1.0000 |
| XGBoost | 0.7738 | 0.8309 | 1.0000 |
| LightGBM | 0.7608 | 0.8305 | 1.0000 |
| CatBoost | 0.7714 | 0.8080 | 0.9999 |
| MLP | 0.7035 | 0.8278 | 1.0000 |

> **Key Insight**: TabPFN achieves **comparable performance** to XGBoost (0.8679 vs 0.8682 ROC AUC) with **16.6× less time**. The main advantage is **zero hyperparameter tuning** — making it attractive for rapid prototyping and exploration.

## 📋 Phase 1 Implementation Summary

The Phase 1 enhancements add comprehensive evaluation capabilities:

### ✅ Completed in Phase 1

| Category | Feature | Description |
|----------|---------|-------------|
| **Calibration** | ECE/MCE | Expected and Maximum Calibration Error with 10 & 15 bins |
| **Curves** | ROC/PR Analysis | Optimal threshold (Youden's J), per-curve AUC |
| **Per-Class** | Precision/Recall/F1 | Class-specific metrics for binary classification |
| **Fairness** | Demographic Parity | TPR/FPR disparities, balanced accuracy, MCC |
| **Statistical** | Friedman Test | Non-parametric overall model comparison |
| **Statistical** | Nemenyi Post-hoc | Pairwise significance with Critical Difference |
| **Statistical** | Bootstrap CI | 95% confidence intervals via resampling |
| **Profiling** | Memory Tracking | Peak memory, delta, and GPU memory usage |
| **Visualization** | CD Diagrams | Critical Difference visualization |

---

## 🧪 Datasets

15 OpenML binary classification datasets, curated for TabPFN's sweet spot (N < 10,000, p < 100):

- Credit-g (German Credit)
- Diabetes (Pima Indians)
- Spambase
- Banknote Authentication
- Hill-Valley
- WDBC (Breast Cancer)
- QSAR-Biodeg
- Titanic
- Churn
- Ozone Level
- KC1 / PC1 (Software Defect Prediction)
- First-Order-Theorem
- Phoneme
- Australian Credit

## 🔬 Methodology

1. **Data Split**: 80% train / 20% test (stratified)
2. **Cross-Validation**: 5-fold stratified for hyperparameter tuning
3. **Seeds**: 10 random seeds per dataset for robust comparison (currently: 1 seed for initial results)
4. **TabPFN**: Subsamples training data if >3000 rows (practical CPU limit)
5. **Statistical Tests**: Wilcoxon signed-rank for pairwise significance (p < 0.05)

### Run Details

- **Current Results**: 1 seed (seed=0), 3 datasets (credit-g, diabetes, banknote-auth)
- **Full Benchmark**: 10 seeds, 15 datasets (in progress)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ideas for Contributions

- Add more datasets
- Implement additional models (FT-Transformer, SAINT, NODE)
- Add more statistical tests (Friedman test, Nemenyi post-hoc)
- Improve visualizations
- Add confidence intervals to results
- Support regression tasks

## 📖 Detailed Metrics Documentation

For comprehensive documentation of all metrics implemented in Phase 1, including:
- What each metric means (intuitive definitions)
- How results are aggregated across multiple seeds/datasets
- What each visualization shows and how to interpret it
- Statistical test explanations with formulas
- Quick analysis workflow examples
- TabPFN vs GBDT interpretation guide

See **[METRICS_DOCUMENTATION.md](METRICS_DOCUMENTATION.md)** — The all-in-one guide for understanding benchmark results.

## 📝 License

MIT License — feel free to use this for your own research or projects.

## 📚 Citations

If you use this benchmark in your research, please cite:

```bibtex
@misc{tabfm-benchmark,
  author = {Siri1702},
  title = {TabFM Benchmark: Tabular Foundation Models vs Gradient Boosted Trees},
  year = {2026},
  url = {https://github.com/Siri1702/tabfm-benchmark}
}
```

---

⭐ **Star this repo** if you find it useful for your research!