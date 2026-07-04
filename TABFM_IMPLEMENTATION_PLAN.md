# TabFM Implementation Plan

Google's TabFM integration into the TabFM Benchmark project.

---

## Analysis Summary

### ✅ Feasibility: YES

Google's TabFM is fully compatible with the benchmark:
- **License**: Apache 2.0 (free for commercial use)
- **API**: scikit-learn compatible (TabFMClassifier/TabFMRegressor)
- **Python**: 3.11+ required
- **Installation**: Simple pip install
- **Input**: pandas DataFrame (compatible)
- **Output**: predict() and predict_proba() available

### Key Similarities & Differences vs TabPFN

| Aspect | TabPFN (Current) | TabFM (Google) |
|--------|-----------------|----------------|
| **Approach** | In-context learning | In-context learning |
| **Input** | numpy arrays | pandas DataFrame |
| **Features** | Numerical only (needs encoding) | Mixed num + categorical (native) |
| **Backend** | PyTorch | JAX or PyTorch |
| **Python** | Any version | 3.11+ required |
| **Tuning** | Zero-shot | Zero-shot |
| **License** | Apache 2.0 | Apache 2.0 |

---

## Implementation Phases

### Phase 1: Preparation (1-2 days)

- [ ] 1.1 Add Python 3.11+ requirement to documentation
- [ ] 1.2 Add tabfm to requirements.txt
- [ ] 1.3 Test tabfm installation

### Phase 2: Model Wrapper (2-3 days)

- [ ] 2.1 Create `src/models/tabfm_model.py` wrapper
- [ ] 2.2 Implement TabFMWrapper class
- [ ] 2.3 Implement run_tabfm_benchmark function
- [ ] 2.4 Test wrapper on sample data

### Phase 3: Integration (1-2 days)

- [ ] 3.1 Update configs/models.yaml
- [ ] 3.2 Update configs/experiment.yaml (if needed)
- [ ] 3.3 Modify experiments/run_benchmark.py
- [ ] 3.4 Add TabFM to model registry

### Phase 4: Testing (1-2 days)

- [ ] 4.1 Test on single dataset (credit-g)
- [ ] 4.2 Test both JAX and PyTorch backends
- [ ] 4.3 Handle edge cases (categorical features, missing values)
- [ ] 4.4 Verify all metrics computed correctly

### Phase 5: Full Benchmark Run (1-2 days)

- [ ] 5.1 Run benchmark with TabFM on all datasets
- [ ] 5.2 Run with multiple seeds (10)
- [ ] 5.3 Collect results and generate visualizations
- [ ] 5.4 Compare TabFM vs TabPFN vs GBDT performance

### Phase 6: Documentation (1 day)

- [ ] 6.1 Update README with TabFM
- [ ] 6.2 Add TabFM to benchmark overview
- [ ] 6.3 Update results interpretation guide

---

## Implementation Checklist

| Task | Effort | Priority |
|------|--------|----------|
| Add Python 3.11+ requirement to docs | 0.5 hr | 🔴 High |
| Add tabfm to requirements.txt | 0.5 hr | 🔴 High |
| Create `src/models/tabfm_model.py` wrapper | 2-3 hr | 🔴 High |
| Update configs/models.yaml | 1 hr | 🔴 High |
| Update run_benchmark.py for TabFM | 1 hr | 🔴 High |
| Test on 1 dataset | 1 hr | 🔴 High |
| Full benchmark run | 2-4 hr | 🟡 Medium |
| Update README with TabFM | 1 hr | 🟡 Medium |

---

## Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Python 3.11+ requirement** | Create separate venv or note in requirements.txt |
| **Context length limits** | Test on large datasets, may need sampling |
| **Mixed feature handling** | TabFM handles natively - but verify compatibility |
| **Backend comparison** | Test both JAX and PyTorch, use faster one |
| **Memory usage** | Monitor during testing - may need optimization |

---

## Model Wrapper Code Structure

```python
# src/models/tabfm_model.py

class TabFMWrapper:
    """Wrapper for Google's TabFM model."""
    
    def __init__(self, backend: str = "jax"):
        self.backend = backend
        self.model = None
        self.is_fitted = False
        
    def fit(self, X, y):
        # Convert to DataFrame
        # Load model
        # Fit with in-context learning
        pass
        
    def predict(self, X):
        pass
        
    def predict_proba(self, X):
        pass


def run_tabfm_benchmark(X_train, y_train, X_test, y_test, ...):
    """Run TabFM on single train/test split."""
    pass
```

---

## Configuration Updates

### models.yaml
```yaml
tabfm_jax:
  name: TabFM (JAX)
  type: tabular_fm
  tuning_required: false
  
tabfm_pytorch:
  name: TabFM (PyTorch)
  type: tabular_fm
  tuning_required: false
```

---

## Expected Outcomes

After implementation:
- TabFM (JAX) and TabFM (PyTorch) in model comparison
- ROC AUC, timing, calibration metrics for TabFM
- Visualizations showing TabFM vs TabPFN vs GBDT
- Full analysis of zero-shot model performance

---

*Created: July 2026*
*Project: TabFM Benchmark*