# TabFM Benchmark 🏆

<p align="center">
  <a href="https://github.com/Siri1702/tabfm-benchmark/stargazers">
    <img src="https://img.shields.io/github/stars/Siri1702/tabfm-benchmark?style=flat-square" alt="Stars">
  </a>
  <a href="https://github.com/Siri1702/tabfm-benchmark/issues">
    <img src="https://img.shields.io/github/issues/Siri1702/tabfm-benchmark?style=flat-square" alt="Issues">
  </a>
  <a href="https://github.com/Siri1702/tabfm-benchmark/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Siri1702/tabfm-benchmark?style=flat-square" alt="License">
  </a>
</p>

A fair, reproducible benchmark comparing **tabular foundation models** (TabPFN) against traditional gradient boosted trees (XGBoost, LightGBM, CatBoost) and neural networks (MLP) on binary classification tasks.

---

## 📑 Table of Contents

1. [What is this benchmark?](#what-is-this-benchmark)
2. [Quick Navigation Guide](#-quick-navigation-guide)
3. [What's Being Compared](#whats-being-compared)
4. [Key Metrics Tracked](#key-metrics-tracked)
5. [Project Structure](#project-structure)
6. [Quick Start](#-quick-start)
7. [Understanding Your Results](#-understanding-your-results)
8. [Benchmark Results](#-benchmark-results)
9. [Datasets](#-datasets)
10. [Methodology](#-methodology)
11. [Contributing](#-contributing)
12. [License](#-license)

---

## 🔍 What is this benchmark?

This project investigates the practical trade-offs between **zero-shot tabular foundation models** and **traditionally-tuned** machine learning approaches. The key question:

> **Can a single zero-shot model (TabPFN) compete with extensively-tuned gradient boosted trees — without the computational cost of hyperparameter optimization?**

### The Answer (Preview)

| Metric | TabPFN | XGBoost | Winner |
|--------|--------|---------|--------|
| **ROC-AUC** | 0.8847 | 0.8698 | TabPFN (+1.5%) |
| **Speed** | 12.5s | 84.4s | TabPFN (15.2× faster) |
| **Calibration** | 0.048 ECE | 0.034 ECE | XGBoost (better) |

*See the [Benchmark Results](#-benchmark-results) section for complete analysis.*

---

## 🧭 Quick Navigation Guide

**I'm a...** | **Go to...** | **Description**
-------------|--------------|------------------
Data Scientist wanting to use a model | [When to Use Which Model](#-when-to-use-which-model) | Decision framework for model selection
Researcher wanting detailed metrics | [Key Metrics Tracked](#key-metrics-tracked) | Complete list of all metrics with definitions
Developer wanting to run the benchmark | [Quick Start](#-quick-start) | Installation and running instructions
Analyst looking at results | [Understanding Your Results](#-understanding-your-results) | Where to find and how to interpret results
Contributor | [Contributing](#-contributing) | How to add models, datasets, or tests

---

## ⚖️ What's Being Compared

| Model | Type | Tuning Required | Best For |
|-------|------|-----------------|----------|
| **TabPFN** | Tabular Foundation Model | ❌ None (zero-shot) | Rapid prototyping, small datasets |
| **SAP-RPT-1** | Tabular Foundation Model | ❌ None (zero-shot) | Semantics-rich datasets, fast deployment |
| **XGBoost** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Production, well-calibrated probabilities |
| **LightGBM** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Large datasets, speed-critical |
| **CatBoost** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Categorical features |
| **MLP** | Neural Network | ✅ Optuna (30 trials) | Real-time inference |

---

## 📊 Key Metrics Tracked

> 💡 **Tip**: See [METRICS_DOCUMENTATION.md](METRICS_DOCUMENTATION.md) for detailed explanations of each metric.

### Performance Metrics
| Metric | Description | Interpretation |
|--------|-------------|-----------------|
| **ROC AUC** | Area under the ROC curve | Higher = better discrimination ability |
| **Average Precision (PR-AUC)** | Area under the precision-recall curve | Higher = better on imbalanced data |
| **F1 Macro** | Harmonic mean of precision/recall (both classes) | Higher = balanced class performance |
| **Brier Score** | Mean squared error of probability predictions | Lower = better calibrated |
| **Log Loss** | Negative log-likelihood | Lower = better probabilistic predictions |

### Calibration Metrics
| Metric | Description | Interpretation |
|--------|-------------|-----------------|
| **ECE-10** | Expected Calibration Error (10 bins) | Lower = better calibrated |
| **ECE-15** | Expected Calibration Error (15 bins) | Lower = better calibrated |
| **MCE** | Maximum Calibration Error | Lower = better worst-case calibration |

### Per-Class Metrics
| Metric | Description |
|--------|-------------|
| **Precision/Recall per class** | Class 0 and Class 1 separately |
| **F1 per class** | Class 0 and Class 1 separately |
| **Sensitivity (TPR)** | True positive rate |
| **Specificity (TNR)** | True negative rate |
| **PPV** | Positive Predictive Value |

### Fairness & Quality Metrics
| Metric | Description | Interpretation |
|--------|-------------|-----------------|
| **Balanced Accuracy** | Accuracy accounting for class imbalance | Higher = better on imbalanced data |
| **MCC** | Matthews Correlation Coefficient | Higher = better (+1 to -1 scale) |
| **Demographic Parity** | Overall positive prediction rate | Lower = less bias |

### Timing & Resource Metrics
| Metric | Description |
|--------|-------------|
| **Tuning Time** | Hyperparameter optimization time |
| **Fit Time** | Model training time |
| **Predict Time** | Inference time |
| **Total Wall Time** | End-to-end execution time |
| **Peak Memory (MB)** | Maximum memory usage |

### Statistical Testing
| Test | Description |
|------|-------------|
| **Wilcoxon signed-rank** | Pairwise significance testing |
| **Friedman test** | Overall model comparison |
| **Nemenyi post-hoc** | Pairwise comparisons with Critical Difference |
| **Bootstrap CI** | 95% confidence intervals |

---

## 📁 Project Structure

```
tabfm-benchmark/
├── configs/                     # 📋 Configuration files
│   ├── datasets.yaml           # 15 OpenML datasets with metadata
│   ├── models.yaml             # Model configs & hyperparameter search spaces
│   └── experiment.yaml         # Experiment settings
│
├── src/                        # 🛠️ Source code
│   ├── data/                   # Data loading (OpenML)
│   │   └── loader.py
│   │
│   ├── models/                 # Model wrappers
│   │   ├── tabpfn_model.py    # TabPFN (zero-shot)
│   │   ├── xgboost_model.py   # XGBoost (tuned)
│   │   ├── lightgbm_model.py  # LightGBM (tuned)
│   │   ├── catboost_model.py  # CatBoost (tuned)
│   │   └── mlp_model.py       # MLP (tuned)
│   │
│   ├── evaluation/             # Metrics & evaluation
│   │   ├── metrics.py         # All performance, calibration, fairness metrics
│   │   ├── statistical.py     # Friedman, Nemenyi, bootstrap CI
│   │   └── profiling.py       # Memory & timing profiling
│   │
│   └── viz/                    # Visualizations
│       ├── leaderboard.py     # Heatmaps & ranking plots
│       ├── statistical.py    # CD diagrams, confidence intervals
│       └── curves.py          # ROC/PR curves
│
├── experiments/                # 🏃 Experiment runner
│   ├── run_benchmark.py       # Main benchmark script
│   ├── results/
│   │   ├── raw/               # Individual JSON results per dataset
│   │   │   └── 20260705_*.json
│   │   └── aggregated/        # Aggregated CSV
│   │       └── results.csv
│   └── analysis/              # Result analysis scripts (if any)
│
├── reports/                    # 📊 Generated reports
│   ├── summary.md             # Detailed analysis report ⭐ START HERE FOR RESULTS
│   ├── linkedin_post.md       # LinkedIn summary
│   ├── figures/               # Generated visualizations
│   │   ├── auc_heatmap.png    # ROC AUC comparison
│   │   ├── average_ranks.png  # Model ranking
│   │   ├── calibration_comparison.png  # ECE/MCE plots
│   │   ├── confidence_intervals.png    # Bootstrap CI
│   │   ├── memory_comparison.png       # Memory usage
│   │   ├── timing_comparison.png       # Runtime charts
│   │   ├── win_loss_matrix.png         # Pairwise wins
│   │   └── cd_diagram.png              # Critical Difference
│   └── metrics_documentation.md       # Detailed metric definitions
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── METRICS_DOCUMENTATION.md     # Detailed metrics guide
└── CONTRIBUTING.md            # Contribution guidelines
```

---

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

# Quick test with fewer seeds (faster)
python experiments/run_benchmark.py --dataset credit-g --n_seeds 3

# Run with custom config
python experiments/run_benchmark.py --config my_config.yaml
```

### Output Locations

| Output | Location | Description |
|--------|----------|-------------|
| Raw Results | `experiments/results/raw/` | JSON files per dataset |
| Aggregated | `experiments/results/aggregated/results.csv` | All metrics in CSV |
| Figures | `reports/figures/` | Generated visualizations |
| Analysis | `reports/summary.md` | Detailed results |

---

## 📈 Understanding Your Results

### 📊 Where to Find What

**I want to see...** | **Go to...** | **File**
---------------------|-------------|--------
Overall model comparison | [reports/summary.md](reports/summary.md) | Summary of all metrics
ROC-AUC heatmap | Reports/Figures | `auc_heatmap.png`
Model rankings | Reports/Figures | `average_ranks.png`
Calibration analysis | Reports/Figures | `calibration_comparison.png`
Statistical significance | Reports/Figures | `cd_diagram.png`, `confidence_intervals.png`
Timing comparison | Reports/Figures | `timing_comparison.png`
Memory usage | Reports/Figures | `memory_comparison.png`
Pairwise wins/losses | Reports/Figures | `win_loss_matrix.png`
Raw numbers | experiments/results/aggregated/ | `results.csv`

### 📖 How to Interpret Results

#### Performance Metrics (Higher = Better)
- **ROC AUC > 0.9**: Excellent discrimination
- **ROC AUC 0.8-0.9**: Good discrimination
- **ROC AUC 0.7-0.8**: Fair discrimination
- **ROC AUC < 0.7**: Poor discrimination

#### Calibration Metrics (Lower = Better)
- **ECE < 0.05**: Well calibrated
- **ECE 0.05-0.10**: Acceptable calibration
- **ECE > 0.10**: Poor calibration

#### Timing
- **TabPFN**: ~10-20s (no tuning)
- **MLP**: ~15-35s (tuned)
- **LightGBM**: ~60-100s (tuned)
- **XGBoost**: ~80-120s (tuned)
- **CatBoost**: ~200-400s (tuned)

---

## 🏆 Benchmark Results

> 📋 **Full Results**: See [reports/summary.md](reports/summary.md) for complete analysis.

### Quick Summary

| Model | ROC-AUC | Speed (avg) | Calibration (ECE) | Best Use Case |
|-------|---------|-------------|-------------------|---------------|
| **TabPFN** 🥇 | 0.932 | 5.2s ⚡ | 0.048 | Rapid prototyping |
| **SAP-RPT-1** 🥈 | 0.891 | 9.4s ⚡ | — | Semantics-rich datasets |
| **CatBoost** 🥉 | 0.887 | 330.6s | 0.040 | Categorical features |
| **XGBoost** | 0.888 | 62.5s | 0.034 ✅ | Production systems |
| **LightGBM** | 0.889 | 213.3s | 0.048 | Large datasets |
| **MLP** | 0.866 | 33.0s | 0.052 | Real-time inference |

> Results across 15 OpenML datasets, 5 random seeds each. Speed = mean total wall-clock time
> per dataset-seed (includes Optuna tuning for tree models). SAP-RPT-1 added July 2026.

### Key Findings

1. **TabPFN achieves 12.1× speedup** over XGBoost with the highest mean AUC (0.932)
2. **SAP-RPT-1 ranks #2 zero-shot** — statistically tied with tuned CatBoost, XGBoost, LightGBM (Wilcoxon p > 0.05), 6.7× faster than XGBoost
3. **XGBoost has best calibration** - critical for probability-based decisions
4. **Traditional models excel** on specific domains (finance, healthcare)

### When to Use Which Model

#### ✅ Choose TabPFN When:
- Need rapid prototyping (15× faster)
- Small datasets (N < 1,000)
- High-dimensional data (p > 50)
- Exploring data characteristics before full tuning

#### ✅ Choose XGBoost When:
- Production systems requiring calibration
- Consistent performance across domains
- Probability-based decision making
- Balanced accuracy matters

#### ✅ Choose LightGBM When:
- Large datasets (speed critical)
- Memory-constrained environments
- Accept slight accuracy trade-off for speed

#### ✅ Choose CatBoost When:
- Many categorical features
- Need native categorical handling
- Can afford longer training time

#### ✅ Choose MLP When:
- Real-time inference needed
- Need integration with deep learning pipelines
- Feature-rich data with learned representations

---

## 🧪 Datasets

15 OpenML binary classification datasets, curated for TabPFN's sweet spot:

| Dataset | Domain | Samples | Features | Notes |
|---------|--------|---------|----------|-------|
| credit-g | Finance | 1,000 | 20 | German Credit - imbalanced |
| diabetes | Healthcare | 768 | 8 | Pima Indians Diabetes |
| spambase | NLP | 4,601 | 57 | Email spam |
| banknote-auth | Security | 1,372 | 4 | Low-dimensional |
| hill-valley | Synthetic | 1,212 | 100 | High-dimensional stress test |
| wdbc | Healthcare | 569 | 30 | Breast Cancer - separable |
| qsar-biodeg | Chemistry | 1,055 | 41 | Molecular QSAR |
| titanic | Historical | 891 | 11 | Classic dataset |
| ozone-level | Environment | 2,536 | 72 | High feature count |
| kc1 | Software | 2,109 | 21 | Defect prediction - imbalanced |
| pc1 | Software | 1,109 | 21 | Defect prediction |
| sick | Healthcare | 3,772 | 37 | Thyroid disease |
| phoneme | Audio | 5,404 | 5 | Low-dimensional |
| australian | Finance | 690 | 14 | Credit approval |
| kc2 | Software | 1,180 | 21 | Defect prediction |

---

## 🔬 Methodology

1. **Data Split**: 80% train / 20% test (stratified)
2. **Cross-Validation**: 5-fold stratified for hyperparameter tuning
3. **Seeds**: Random seeds for reproducibility
4. **TabPFN**: Subsamples if >8000 rows (CPU limit)
5. **Statistical Tests**: Wilcoxon (p < 0.05), Friedman, Nemenyi

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ideas for Contributions

- [ ] Add more datasets
- [ ] Implement additional models (FT-Transformer, SAINT, NODE)
- [ ] Add regression support
- [ ] Multi-seed analysis
- [ ] Ensemble methods
- [ ] Your ideas welcome! 🎉

---

## 📝 License

MIT License — feel free to use for research or projects.

---

## 📚 Citations

```bibtex
@misc{tabfm-benchmark,
  author = {Siri1702},
  title = {TabFM Benchmark: Tabular Foundation Models vs Gradient Boosted Trees},
  year = {2026},
  url = {https://github.com/Siri1702/tabfm-benchmark}
}
```

---

<p align="center">
  ⭐ <strong>Star this repo</strong> if you find it useful for your research!
</p>