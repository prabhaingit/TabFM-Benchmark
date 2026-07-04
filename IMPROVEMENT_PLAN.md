# TabFM Benchmark - Phasewise Improvisation Plan

## Executive Summary

This document outlines a comprehensive improvement plan to transform the TabFM Benchmark from a solid foundation into a **best-in-class research experiment project** that addresses critical gaps in existing benchmarks like TabArena. The plan is organized into **4 phases** with clear deliverables and milestones.

---

## Phase 1: Foundation & Infrastructure Enhancement

**Timeline**: 2-3 weeks  
**Goal**: Strengthen the core benchmark infrastructure and add missing essential capabilities

### 1.1 Complete Full Benchmark Execution
- [ ] Run full benchmark on all 15 datasets with 10 seeds
- [x] Generate complete statistical significance analysis
- [ ] Create comprehensive visualization dashboard (Plotly/Dash)

### 1.2 Add Statistical Rigor
- [x] Implement Friedman test for overall model comparison
- [x] Add Nemenyi post-hoc test for pairwise comparisons
- [x] Compute confidence intervals (bootstrap/CI) for all metrics
- [x] Add critical difference (CD) diagrams

### 1.3 Enhanced Evaluation Metrics
- [x] Add calibration metrics (ECE, reliability diagrams)
- [x] Implement ROC curves and PR curves per dataset
- [x] Add per-class metrics (precision, recall per class)
- [x] Compute fairness metrics (demographic parity, equalized odds)

### 1.4 Timing Analysis Expansion
- [x] Memory usage tracking
- [x] GPU memory (if using CUDA for TabPFN)
- [x] CPU/GPU utilization profiling

---

## Phase 2: Addressing TabArena Limitations

**Timeline**: 3-4 weeks  
**Goal**: Directly address the key limitations identified in TabArena and other benchmarks

### 2.1 Expand Dataset Coverage
- [ ] **Small Data Regime**: Add 10-15 datasets with N < 500 samples (inspired by PMLBmini)
  - Explore: small medical datasets, rare disease datasets
- [ ] **Large Data Regime**: Add 3-5 datasets with N > 50,000 samples
  - Test TabPFN subsampling strategy effectiveness
- [ ] **Temporal Data**: Add 3-5 datasets with time-based splits
  - Address distribution shift scenarios
- [ ] **Feature-Rich Datasets**: Add datasets with >100 features after preprocessing

### 2.2 Advanced Model Coverage
- [x] Add **TabFM** (Google's zero-shot tabular foundation model) - IMPLEMENTED ✓
- [ ] Add **FT-Transformer** (from rtdl package)
- [ ] Add **TabR** (retrieval-augmented)
- [ ] Add **SAINT** (self-attention with row/column embeddings)
- [ ] Add **NODE** (Neural Oblivious Decision Ensembles)
- [ ] Add **TabDPT** (if available - foundation model)

### 2.3 Hyperparameter Analysis
- [ ] Study variance of random hyperparameter choices
- [ ] Compare random search vs Bayesian optimization
- [ ] Analyze sensitivity to key hyperparameters per model

### 2.4 Robustness Analysis
- [ ] Add noise injection experiments
- [ ] Test feature perturbation robustness
- [ ] Test missing value handling

---

## Phase 3: Novel Research Contributions

**Timeline**: 4-6 weeks  
**Goal**: Add capabilities that provide NEW insights beyond existing benchmarks

### 3.1 Few-Shot Learning Analysis (CRITICAL GAP)
- [ ] **Systematic few-shot evaluation**: Test all models at N = 50, 100, 200, 500, 1000
- [ ] Create learning curves (performance vs training set size)
- [ ] Identify "crossover points" where DL surpasses GBDT
- [ ] Publish findings on data efficiency characteristics

### 3.2 Context Window Analysis for Foundation Models
- [ ] Analyze TabPFN performance vs context size
- [ ] Test different subsampling strategies
- [ ] Study context efficiency (performance per training sample used)
- [ ] Investigate context contamination risks

### 3.3 Feature Type Sensitivity
- [ ] Categorical feature cardinality impact
- [ ] Numerical feature distribution impact (skewed vs normal)
- [ ] Mixed feature type handling
- [ ] High-cardinality categorical handling

### 3.4 Meta-Learning Analysis
- [ ] Build meta-features for datasets
- [ ] Predict best model for new datasets
- [ ] Analyze which dataset characteristics favor which models

### 3.5 Ensemble Analysis
- [ ] Study simple averaging vs weighted ensembles
- [ ] Test cross-model ensembles (TabPFN + GBDT)
- [ ] Analyze ensemble diversity and complementarity

---

## Phase 4: Publication-Ready Research

**Timeline**: 4-6 weeks  
**Goal**: Transform into a complete research artifact with novel insights

### 4.1 Novel Insights to Discover
- [ ] **When do Tabular Foundation Models Really Work?** 
  - Define decision boundaries based on dataset characteristics
- [ ] **Cost-Benefit Analysis Framework**
  - Create trade-off curves (performance vs compute)
  - Provide decision guidance for practitioners
- [ ] **Beyond AUC: Practical Metrics**
  - Focus on imbalanced data scenarios
  - Analyze precision-recall trade-offs

### 4.2 Reproducibility & Documentation
- [ ] Complete experiment tracking with MLflow
- [ ] Generate complete experiment manifests
- [ ] Add Docker/container support for reproducibility
- [ ] Create experiment registry

### 4.3 Visualization Dashboard
- [ ] Interactive dashboard (Plotly/Dash)
- [ ] Learning curve visualization
- [ ] Statistical significance heatmaps
- [ ] Runtime performance comparison charts

### 4.4 Publication Artifacts
- [ ] Generate LaTeX tables for paper submission
- [ ] Create supplementary materials
- [ ] Write methodology section
- [ ] Draft key findings and conclusions

---

## Implementation Roadmap

```
Phase 1: Foundation          [Week 1-3]
├── Complete baseline runs
├── Statistical rigor
└── Enhanced metrics

Phase 2: TabArena Gaps       [Week 4-7]
├── Dataset expansion
├── Model expansion
└── Robustness testing

Phase 3: Novel Research      [Week 8-13]
├── Few-shot analysis
├── Context analysis
├── Meta-learning
└── Ensemble studies

Phase 4: Publication         [Week 14-19]
├── Dashboard
├── Documentation
├── Novel insights
└── Paper artifacts
```

---

## Key Research Questions to Answer

Based on the analysis of TabArena and other benchmark limitations, here are the critical questions this enhanced benchmark should answer:

| # | Question | Why Important |
|---|----------|----------------|
| 1 | **At what training set size does TabPFN stop being competitive?** | Defines practical applicability |
| 2 | **How does context size (training data used) affect TabPFN performance?** | Understands foundation model behavior |
| 3 | **Which dataset characteristics predict model success?** | Enables smart model selection |
| 4 | **Can ensemble of TabPFN + GBDT beat both individually?** | Practical hybrid approaches |
| 5 | **How do models handle temporal distribution shift?** | Real-world reliability |
| 6 | **What is the compute/performance trade-off frontier?** | Practical decision-making |

---

## Technical Improvements Required

### New Dependencies
```python
# Statistical tests
scikit-posthocs  # Already in requirements

# Visualization & Dashboard
plotly
dash

# Advanced models
rtdl  # For FT-Transformer, MLP, TabR

# Meta-learning
scipy.optimize  # For meta-model fitting

# Experiment tracking
mlflow  # Already in requirements
```

### New Files to Create
1. `src/data/meta_features.py` - Dataset meta-feature extraction
2. `src/analysis/few_shot.py` - Few-shot learning analysis
3. `src/analysis/learning_curves.py` - Learning curve generation
4. `src/analysis/ensembles.py` - Ensemble analysis
5. `src/viz/dashboard.py` - Interactive dashboard

---

## References

- [TabArena: A Living Benchmark](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
- [PMLBmini: Data-Scarce Applications](https://arxiv.org/pdf/2409.01635)
- [TABRED: Temporal Shift & Feature-Rich Gaps](https://openreview.net/pdf?id=L14sqcrUC3)
- [TabR: Retrieval-Augmented Tabular DL](https://arxiv.org/pdf/2307.14338)
- [Data-Centric Tabular Evaluation](https://proceedings.neurips.cc/paper_files/paper/2024/file/ae00e5ce7142d02e30a8235ede1ec6fc-Paper-Datasets_and_Benchmarks_Track.pdf)

---

*Generated: June 2026*
*Project: TabFM Benchmark*
*Author: Siri1702*