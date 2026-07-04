# Metrics Documentation - Comprehensive Guide

This guide serves as an all-in-one reference for understanding the TabFM Benchmark's evaluation framework. It explains what each metric captures intuitively, how it's calculated, how to interpret results, and how multi-dataset/multi-seed results are aggregated.

---

## Table of Contents

1. [How Results Are Collected and Aggregated](#1-how-results-are-collected-and-aggregated)
2. [Performance Metrics](#2-performance-metrics)
3. [Calibration Metrics](#3-calibration-metrics)
4. [Per-Class Metrics](#4-per-class-metrics)
5. [Fairness & Quality Metrics](#5-fairness--quality-metrics)
6. [Timing & Resource Metrics](#6-timing--resource-metrics)
7. [Statistical Tests](#7-statistical-tests)
8. [Visualizations Explained](#8-visualizations-explained)
9. [Putting It All Together](#9-putting-it-all-together)

---

## 1. How Results Are Collected and Aggregated

### 1.1 The Experiment Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SINGLE EXPERIMENT RUN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Dataset → Train/Test Split → For Each Model:                              │
│    1. Hyperparameter Tuning (Optuna - except TabPFN)                        │
│    2. Train Model on Training Set                                           │
│    3. Predict on Test Set                                                    │
│    4. Compute ALL metrics                                                    │
│    5. Store as RunResult                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │  results.csv (per seed)     │
                    │  - model, dataset, seed     │
                    │  - all metrics computed     │
                    └─────────────────────────────┘
```

### 1.2 Aggregation Process

For each model on each dataset, experiments are run with **multiple random seeds**. Here's how aggregation works:

```python
# Raw results structure: one row per (model, dataset, seed)
# Example: 5 models × 3 datasets × 10 seeds = 150 rows

# Aggregation in compute_metrics():
results_df.groupby(["dataset", "model"])[metric].mean()  # Mean across seeds

# For statistical tests, we use the mean metric per (dataset, model)
pivot = results_df.groupby(["dataset", "model"]).mean()
# Now each row = one dataset, each column = one model
```

### 1.3 Why Multiple Seeds?

| Aspect | What It Captures |
|--------|------------------|
| **Random seed** | Model initialization, data shuffling |
| **Dataset variation** | How model performs across different data distributions |
| **Combined** | True performance + stability of the model |

A model with high variance across seeds (e.g., std > 0.05 in AUC) is **unstable** — its performance depends heavily on random initialization.

### 1.4 What Gets Stored

Each `RunResult` contains:

```
model_name          # e.g., "TabPFN", "XGBoost"
dataset_name        # e.g., "credit-g"
seed                # Random seed used
n_train, n_test     # Sample counts
n_features          # Feature count
y_test              # True labels
y_proba             # Predicted probabilities
y_pred              # Binary predictions (threshold=0.5)
fit_time_sec        # Training time
predict_time_sec    # Inference time
tuning_time_sec     # HP optimization time (0 for TabPFN)
peak_memory_mb      # Maximum memory used
error               # None if successful, error message if failed
```

---

## 2. Performance Metrics

These metrics answer: **"How well does the model predict?"**

### 2.1 ROC AUC (roc_auc)

**What it helps capture:**
> "Can the model distinguish between positive and negative classes? If I rank all predictions by confidence, how likely is a randomly chosen positive to rank higher than a randomly chosen negative?"

**Intuitive Definition:**
Imagine you randomly pick one positive sample and one negative sample. ROC AUC is the probability that your model gives a higher score to the positive sample than to the negative one.

**How it's calculated:**
```
ROC AUC = (number of concordant pairs) / (total pairs)

Where a concordant pair = positive gets higher score than negative
      = area under ROC curve plotting TPR vs FPR at every threshold
```

**How to interpret:**
| ROC AUC | Interpretation | Practical Meaning |
|---------|---------------|-------------------|
| 1.0 | Perfect | Model perfectly separates classes |
| 0.9-1.0 | Excellent | Very strong discriminative ability |
| 0.7-0.9 | Good | Useful for most applications |
| 0.5-0.7 | Fair | Better than random but weak |
| 0.5 | Random | No discrimination ability |

**In results.csv:** Column `roc_auc`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 2.2 Average Precision / PR-AUC (avg_precision)

**What it helps capture:**
> "When the model says 'I'm confident', how often is it actually correct? Especially important when positive class is rare."

**Intuitive Definition:**
Area under the Precision-Recall curve. Precision at threshold t = "Of all predictions above t, what fraction are actually positive?" PR-AUC summarizes this across all possible thresholds.

**Why it matters more than ROC AUC for imbalanced data:**
- ROC AUC can be misleading when negatives outnumber positives
- PR-AUC directly measures your ability to find the rare positives
- A model that predicts all negatives will have 0.5 ROC AUC but undefined PR-AUC

**How to interpret:**
| PR-AUC | Interpretation |
|--------|---------------|
| 1.0 | Perfect precision at all recall levels |
| > 0.8 | Excellent - most confident predictions are correct |
| 0.5-0.8 | Good - useful for ranking |
| < 0.5 | Poor - worse than random |

**In results.csv:** Column `avg_precision`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 2.3 F1 Macro (f1_macro)

**What it helps capture:**
> "How balanced is the model's performance between the two classes? Does it perform well on both positives AND negatives?"

**Intuitive Definition:**
The harmonic mean of precision and recall, computed separately for each class then averaged equally. Treats both classes with equal importance regardless of their frequency.

**Formula:**
```
F1_class0 = 2 × (precision_0 × recall_0) / (precision_0 + recall_0)
F1_class1 = 2 × (precision_1 × recall_1) / (precision_1 + recall_1)
F1_macro  = (F1_class0 + F1_class1) / 2
```

**How to interpret:**
| F1 Macro | Interpretation |
|----------|---------------|
| 1.0 | Perfect on both classes |
| 0.8-1.0 | Excellent balanced performance |
| 0.5-0.8 | Good, but may have class imbalance issues |
| < 0.5 | Poor - model struggles on one or both classes |

**Example:**
A model that predicts all positives will have F1_class1 = 1.0 but F1_class0 = 0, giving F1_macro = 0.5 — correctly flagging the poor balance.

**In results.csv:** Column `f1_macro`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 2.4 Brier Score (brier_score)

**What it helps capture:**
> "How accurate are the probability estimates? If the model says 80%, does it actually happen 80% of the time?"

**Intuitive Definition:**
The mean squared error between predicted probabilities and actual outcomes. Measures how well-calibrated the probabilities are.

**Formula:**
```
Brier Score = (1/n) × Σ (predicted_probability_i - actual_outcome_i)²

Where actual_outcome = 0 or 1
```

**How to interpret:**
| Brier Score | Interpretation |
|-------------|---------------|
| 0.0 | Perfect probability estimates |
| 0.01-0.05 | Excellent calibration |
| 0.05-0.10 | Good calibration |
| 0.10-0.20 | Fair calibration |
| 0.25 | Random guessing (for 50/50 classes) |
| > 0.25 | Worse than random |

**Key insight:** Lower is better. A well-calibrated model has both good discrimination (high ROC AUC) AND accurate probabilities (low Brier Score).

**In results.csv:** Column `brier_score`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 2.5 Log Loss (log_loss_val)

**What it helps capture:**
> "How much is the model 'surprised' by its predictions? Heavily penalizes confident wrong predictions."

**Intuitive Definition:**
Negative log-likelihood of the predictions. Measures how surprised the model is by its mistakes. Penalizes confident wrong predictions exponentially more than uncertain ones.

**Formula:**
```
Log Loss = -(1/n) × Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]

Where p_i = predicted probability for sample i
```

**How to interpret:**
| Log Loss | Interpretation |
|----------|---------------|
| 0.0 | Perfect predictions |
| < 0.5 | Good calibration |
| 0.5-1.0 | Moderate performance |
| > 2.0 | Poor predictions |

**Key difference from Brier Score:**
- Log Loss heavily penalizes **confident wrong predictions**
- Brier Score penalizes all errors equally (squared)
- A model that predicts 0.99 for a negative sample gets hit much harder by Log Loss

**In results.csv:** Column `log_loss_val`
**Aggregation:** Mean across seeds per (dataset, model)

---

## 3. Calibration Metrics

These metrics answer: **"Are the probabilities accurate? Does 80% confidence mean 80% accuracy?"**

### 3.1 Expected Calibration Error (ECE)

**What it helps capture:**
> "On average, how far off are my confidence scores from actual accuracy? If I bucket predictions by confidence, do the accuracies match the confidences?"

**Intuitive Definition:**
Divide predictions into bins based on confidence (e.g., 0-0.1, 0.1-0.2, ...). In each bin, compare:
- The average confidence assigned
- The actual accuracy achieved

ECE is the weighted average of these differences.

**Formula:**
```
ECE = Σ (|B| / n) × |accuracy(B) - confidence(B)|

Where B = set of predictions in each confidence bin
```

**How it's calculated (step-by-step):**
1. Bin predictions into 10 (or 15) equal-width bins
2. For each bin, compute:
   - Bin confidence = mean of predicted probabilities
   - Bin accuracy = mean of true labels (actual positive rate)
3. Weight each bin by its size (fraction of total predictions)
4. Sum the absolute differences

**How to interpret:**
| ECE | Interpretation |
|-----|---------------|
| 0.0 | Perfectly calibrated |
| < 0.05 | Well-calibrated |
| 0.05-0.10 | Acceptable calibration |
| 0.10-0.15 | Poor calibration |
| > 0.15 | Very poorly calibrated |

**Example:**
If all predictions with 80-90% confidence only achieve 60% accuracy, that's a 20% calibration error in that bin.

**In results.csv:** Columns `ece_10` (10 bins), `ece_15` (15 bins)
**Aggregation:** Mean across seeds per (dataset, model)

---

### 3.2 Maximum Calibration Error (MCE)

**What it helps capture:**
> "What's my worst bin? Even if most bins are good, one extreme miscalibration could be dangerous."

**Intuitive Definition:**
The largest calibration error across all bins. Identifies the worst-case miscalibration.

**How to interpret:**
| MCE | Interpretation |
|-----|---------------|
| 0.0 | Perfect in all bins |
| < 0.10 | Good worst-case |
| 0.10-0.20 | Moderate worst-case |
| > 0.20 | Dangerous miscalibration in some range |

**Why it matters:**
In medical or safety applications, you need to know not just average calibration but whether there's a range where predictions are completely wrong.

**In results.csv:** Column `max_calibration_error`
**Aggregation:** Mean across seeds per (dataset, model)

---

## 4. Per-Class Metrics

These metrics answer: **"How does the model perform on each class separately?"**

### 4.1 Sensitivity (Recall) - (sensitivity)

**What it helps capture:**
> "Of all the actual positives, how many did I find?"

**Intuitive Definition:**
True Positive Rate. Also called Recall or Hit Rate.

**Formula:**
```
Sensitivity = TP / (TP + FN)
           = (Positives correctly identified) / (Total actual positives)
```

**How to interpret:**
| Sensitivity | Meaning |
|-------------|---------|
| 1.0 | Found ALL positives |
| 0.8 | Missed 20% of positives |
| 0.5 | Missed half of positives |

**When to prioritize:**
- Medical screening (don't miss sick patients)
- Fraud detection (catch fraudulent transactions)
- High recall cost of missing true positives is high

**In results.csv:** Column `sensitivity`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 4.2 Specificity - (specificity)

**What it helps capture:**
> "Of all the actual negatives, how many did I correctly identify?"

**Intuitive Definition:**
True Negative Rate. Also called Selectivity.

**Formula:**
```
Specificity = TN / (TN + FP)
            = (Negatives correctly identified) / (Total actual negatives)
```

**How to interpret:**
| Specificity | Meaning |
|-------------|---------|
| 1.0 | Correctly rejected ALL negatives |
| 0.8 | 20% false positives |
| 0.5 | Half of negatives incorrectly flagged positive |

**When to prioritize:**
- When false alarms are costly
- Spam detection (don't mark legitimate email as spam)
- Quality control (don't reject good products)

**In results.csv:** Column `specificity`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 4.3 Precision/PPV - (ppv)

**What it helps capture:**
> "When I say positive, how often am I correct?"

**Intuitive Definition:**
Positive Predictive Value. The precision for the positive class.

**Formula:**
```
PPV = TP / (TP + FP)
    = (True positives) / (All positive predictions)
```

**How to interpret:**
| PPV | Meaning |
|-----|---------|
| 1.0 | Every positive prediction is correct |
| 0.8 | 20% false alarms |
| 0.5 | Half of positive predictions are wrong |

**Precision-Recall Trade-off:**
- High precision = low false positive rate = few false alarms
- High recall = low false negative rate = find most positives
- F1 balances both

**In results.csv:** Column `ppv`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 4.4 Per-Class Precision, Recall, F1

**What it helps capture:**
> "How does the model perform on each class individually? Are there disparities?"

**Intuitive Definition:**
Metrics computed separately for Class 0 (negative) and Class 1 (positive).

| Metric | Class 0 Formula | Class 1 Formula |
|--------|-----------------|----------------|
| Precision | TN/(TN+FN) | TP/(TP+FP) |
| Recall | TN/(TN+FP) | TP/(TP+FN) |
| F1 | 2×P×R/(P+R) | 2×P×R/(P+R) |

**How to interpret:**
- Large disparity between class_0 and class_1 metrics indicates bias
- A model with 0.9 precision on class 0 but 0.5 on class 1 treats classes differently
- This matters for fairness and imbalanced data

**In results.csv:** Columns `precision_class_0`, `precision_class_1`, `recall_class_0`, `recall_class_1`, `f1_class_0`, `f1_class_1`
**Aggregation:** Mean across seeds per (dataset, model)

---

## 5. Fairness & Quality Metrics

These metrics answer: **"Does the model treat different groups fairly? Is its quality robust across scenarios?"**

### 5.1 Balanced Accuracy - (balanced_accuracy)

**What it helps capture:**
> "How well does the model perform accounting for class imbalance?"

**Intuitive Definition:**
Average of sensitivity and specificity. Treats both classes equally regardless of their frequency in the data.

**Formula:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

**Why not just use accuracy?**
With 99% negatives and 1% positives:
- A model predicting all negatives gets 99% accuracy
- But balanced accuracy would be 50% (sensitivity=0, specificity=0.99)

**How to interpret:**
| Balanced Accuracy | Interpretation |
|-------------------|----------------|
| 1.0 | Perfect on both classes |
| 0.8-1.0 | Good balanced performance |
| 0.5 | Random (no discriminative power) |

**In results.csv:** Column `balanced_accuracy`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 5.2 Matthews Correlation Coefficient (MCC)

**What it helps capture:**
> "What's the correlation between predictions and actual outcomes? A balanced measure that works even with severe imbalance."

**Intuitive Definition:**
A correlation coefficient for classification. Returns values from -1 to +1 where +1 is perfect prediction, 0 is random, -1 is total disagreement.

**Formula:**
```
MCC = (TP × TN - FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**How to interpret:**
| MCC | Interpretation |
|-----|----------------|
| 1.0 | Perfect prediction |
| 0.7-1.0 | Good correlation |
| 0.3-0.7 | Moderate |
| 0.0 | Random prediction |
| -1.0 | Total disagreement (predicts opposite) |

**Why MCC over accuracy?**
- Accuracy can be misleading with imbalanced data
- MCC uses all four confusion matrix values
- MCC is -1 when predictions are always opposite (not just 0% accuracy)

**In results.csv:** Column `matthews_corrcoef`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 5.3 Demographic Parity - (demographic_parity)

**What it helps capture:**
> "What fraction of predictions are positive? Does this match the underlying prevalence?"

**Intuitive Definition:**
Overall positive prediction rate. The proportion of samples predicted as the positive class.

**Formula:**
```
Demographic Parity = (TP + FP) / (TP + TN + FP + FN)
                   = Mean of binary predictions
```

**How to interpret:**
- If true prevalence (class balance) is 20%, reasonable values are near 20%
- Values far from prevalence could indicate bias or poor calibration
- This is the weakest fairness metric (doesn't guarantee fairness)

**In results.csv:** Column `demographic_parity`
**Aggregation:** Mean across seeds per (dataset, model)

---

### 5.4 TPR/FPR Disparity

**What it helps capture:**
> "Does the model perform equally well across different subgroups?"

**Intuitive Definition:**
When a sensitive attribute is available (not currently in benchmark), compute TPR and FPR for each group. Disparity = max difference between groups.

**Currently in benchmark:**
- `tpr_overall`: Overall True Positive Rate
- `fpr_overall`: Overall False Positive Rate

**How to interpret:**
- TPR should be high (model finds positives)
- FPR should be low (model doesn't falsely flag negatives)
- Large gaps between TPR and FPR indicate room for threshold tuning

**In results.csv:** Columns `tpr_overall`, `fpr_overall`
**Aggregation:** Mean across seeds per (dataset, model)

---

## 6. Timing & Resource Metrics

These metrics answer: **"How much time and resources does the model require?"**

### 6.1 Timing Breakdown

| Metric | What It Captures | TabPFN | GBDT |
|--------|------------------|--------|------|
| `tuning_time_sec` | Hyperparameter search | **0** | ~130-230s |
| `fit_time_sec` | Model training | ~12s | ~130-230s |
| `predict_time_sec` | Inference | ~6s | ~0.01-0.2s |
| `total_wall_time_sec` | End-to-end | ~19s | ~330-460s |

**How to interpret:**
- **TabPFN:** Zero tuning time is its key advantage
- **GBDT:** Tuning dominates total time
- **Speedup ratio:** total_wall_time(XGBoost) / total_wall_time(TabPFN)

**Aggregation:** Mean across seeds per (dataset, model)

---

### 6.2 Memory Metrics

| Metric | What It Captures |
|--------|------------------|
| `peak_memory_mb` | Maximum memory used during execution |
| `memory_delta_mb` | Change in memory from start to peak |
| `gpu_memory_mb` | GPU memory allocated (None if CPU-only) |

**How to interpret:**
- Higher memory for large datasets or complex models
- Memory delta shows the "cost" of loading the model
- GPU memory = None means CPU-only execution

**Aggregation:** Mean across seeds per (dataset, model)

---

## 7. Statistical Tests

These tests answer: **"Are the differences between models statistically significant, or just random variation?"**

### 7.1 Friedman Test

**What it helps capture:**
> "Is there ANY significant difference between models across multiple datasets?"

**Intuitive Definition:**
Non-parametric alternative to ANOVA. Tests whether multiple models perform differently when evaluated on the same datasets.

**When to use:**
- Compare 3+ models
- Same datasets used for all models
- Non-parametric (doesn't assume normal distribution)

**How it's calculated:**
```
1. For each dataset, rank models by performance (1=best, k=worst)
2. Compute average rank for each model
3. Friedman statistic uses rank sums to test if ranks differ significantly
4. p < 0.05 means: "At least one model is significantly different"
```

**How to interpret:**
| Result | Meaning |
|--------|---------|
| p < 0.05 | Significant differences exist → proceed to pairwise tests |
| p ≥ 0.05 | No significant difference → models perform similarly |

**In implementation:**
```python
result = friedman_test(results_df, metric="roc_auc")
# Returns: statistic, p_value, significant, n_datasets, n_models, avg_ranks
```

---

### 7.2 Nemenyi Post-hoc Test

**What it helps capture:**
> "Which specific pairs of models are significantly different?"

**Intuitive Definition:**
Pairwise comparison after significant Friedman test. Uses Critical Difference (CD) to determine significance.

**Critical Difference (CD):**
```
CD = q_α × √(k(k+1)/(6n))

Where k = number of models, n = number of datasets
```

**How to interpret:**
- If rank difference between two models > CD: **significantly different**
- If rank difference < CD: **not significantly different**
- Produces p-values for each pair

**Visual interpretation (CD Diagram):**
- Models are plotted at their average ranks
- A horizontal bar shows the CD
- Models whose rank difference is less than CD are connected (not significantly different)

**In implementation:**
```python
comparisons = nemenyi_cd_test(results_df, metric="roc_auc")
# Returns: model_a, model_b, rank_diff, critical_difference, significant, p_value
```

---

### 7.3 Bootstrap Confidence Intervals

**What it helps capture:**
> "What's the uncertainty in my performance estimate? How confident can I be in the mean?"

**Intuitive Definition:**
Resampling method to estimate confidence intervals. Creates many bootstrap samples by resampling with replacement.

**How it's calculated:**
```
1. Take n samples (results per seed)
2. Resample with replacement 1000 times
3. Compute mean for each bootstrap sample
4. Find 2.5th and 97.5th percentiles = 95% CI
```

**How to interpret:**
- **95% CI:** If we repeated the experiment many times, 95% of intervals would contain the true mean
- **Narrow intervals:** Precise estimate (low variance across seeds)
- **Wide intervals:** Uncertain estimate (high variance)
- **Non-overlapping CIs:** Strong evidence of difference

**In results.csv:** Not a direct column, computed via function
**Visualization:** `reports/figures/confidence_intervals.png`

---

### 7.4 Wilcoxon Signed-Rank Test

**What it helps capture:**
> "Is model A significantly better than model B on a pairwise basis?"

**Intuitive Definition:**
Non-parametric paired test. Compares two models on the same datasets.

**Requirements:**
- Same datasets for both models
- At least 5 datasets for meaningful comparison
- Paired differences (same datasets)

**How to interpret:**
| p-value | Meaning |
|---------|---------|
| < 0.05 | Significant difference |
| ≥ 0.05 | No significant difference |
| better column | Which model performs better on average |

**In implementation:**
```python
comparisons = wilcoxon_pairwise(results_df, metric="roc_auc")
# Returns: model_a, model_b, p_value, significant, better, mean_a, mean_b
```

---

### 7.5 Average Rank Table

**What it helps capture:**
> "What's the overall ranking of models? Which model wins most often?"

**Intuitive Definition:**
For each dataset, rank models by performance. Average rank across all datasets.

**How to calculate:**
```
For each dataset:
  - Rank models: best performing = rank 1, second best = rank 2, etc.
  - Use average method for ties

Average rank = mean of ranks across all datasets
```

**How to interpret:**
| Average Rank | Meaning |
|--------------|----------|
| ~1.0 | Best on almost every dataset |
| ~2.0 | Usually second best |
| ~3.0 | Middle of the pack |
| Higher = worse |

**In implementation:**
```python
ranks = average_rank_table(results_df, metric="roc_auc")
# Returns: model, average_rank, n_datasets
```

---

## 8. Visualizations Explained

All figures are saved to `reports/figures/`:

### 8.1 AUC Heatmap (auc_heatmap.png)

**What it shows:**
Matrix of ROC AUC values. Rows = datasets, Columns = models. Color intensity shows performance.

**How it's calculated:**
1. Filter results to `roc_auc` metric
2. Pivot: rows=datasets, cols=models, values=mean(roc_auc)
3. Color map: darker = higher AUC

**How to read:**
- **Row comparison:** How each model performs on the same dataset
- **Column comparison:** How the same model performs across datasets
- **Color legend:** Check the scale (typically 0.5-1.0)
- **Dark squares:** Strong performance
- **Light squares:** Weak performance

---

### 8.2 Average Ranks (average_ranks.png)

**What it shows:**
Bar chart of average ranks across all datasets.

**How it's calculated:**
1. For each dataset, rank models by ROC AUC (1=best)
2. Compute average rank across all datasets
3. Plot as horizontal bar chart

**How to read:**
- **Lower bar = better rank** (rank 1 is best)
- **Color:** Different models have different colors
- **Interpretation:** TabPFN with avg rank 2.1 means it's typically the 2nd best model

---

### 8.3 Calibration Comparison (calibration_comparison.png)

**What it shows:**
ECE and MCE comparison across models.

**How it's calculated:**
1. Group by model
2. Compute mean ECE_10, ECE_15, and MCE
3. Plot as grouped bar chart

**How to read:**
- **Shorter bars = better calibration** (lower ECE/MCE)
- **Multiple bars per model:** ECE with 10 bins, ECE with 15 bins, MCE
- **TabPFN comparison:** Often has lower ECE than tuned GBDT models

---

### 8.4 Confidence Intervals (confidence_intervals.png)

**What it shows:**
95% bootstrap confidence intervals for each model's ROC AUC.

**How it's calculated:**
1. Group results by model
2. Bootstrap resample 1000 times
3. Compute 2.5th and 97.5th percentiles
4. Plot as horizontal error bars

**How to read:**
- **Dot:** Mean ROC AUC
- **Line:** 95% CI (range of likely values)
- **Non-overlapping intervals:** Strong evidence of difference
- **Overlapping intervals:** No significant difference (may still be statistically significant with proper test)

---

### 8.5 Memory Comparison (memory_comparison.png)

**What it shows:**
Peak memory usage and memory delta across models.

**How it's calculated:**
1. Group by model
2. Compute mean peak_memory_mb and memory_delta_mb
3. Plot as grouped bar chart

**How to read:**
- **Bar height:** Memory in MB
- **Comparison:** Lower is better for deployment
- **Note:** TabPFN may use more memory due to context storage

---

### 8.6 Timing Comparison (timing_comparison.png)

**What it shows:**
Runtime breakdown by model and timing category.

**How it's calculated:**
1. Group by model
2. Compute mean tuning_time, fit_time, predict_time
3. Plot as stacked or grouped bar chart

**How to read:**
- **Stacking:** Shows total time breakdown
- **TabPFN:** Minimal tuning time (0), moderate fit time
- **GBDT models:** Large tuning time dominates
- **Practical insight:** Time to solution = tuning + fit + predict

---

### 8.7 Win/Loss Matrix (win_loss_matrix.png)

**What it shows:**
Pairwise head-to-head comparisons.

**How it's calculated:**
1. For each dataset, compare models pairwise
2. Count wins (model A > model B), losses, ties
3. Create three heatmaps

**How to read:**
- **Wins heatmap:** Row model beats Column model
- **Losses heatmap:** Row model loses to Column model  
- **Ties heatmap:** Models perform equally
- **Diagonal:** Not applicable (model vs itself)
- **Interpretation:** A model that wins more than loses is generally better

---

## 9. Putting It All Together

### 9.1 Quick Analysis Workflow

```python
import pandas as pd

# Load results
df = pd.read_csv("experiments/results/aggregated/results.csv")

# 1. Overall performance
perf = df.groupby("model")[["roc_auc", "avg_precision", "f1_macro"]].mean()
print("=== Performance ===")
print(perf.sort_values("roc_auc", ascending=False))

# 2. Calibration
cal = df.groupby("model")[["ece_10", "brier_score"]].mean()
print("\n=== Calibration (lower is better) ===")
print(cal.sort_values("ece_10"))

# 3. Timing
timing = df.groupby("model")[["tuning_time_sec", "total_wall_time_sec"]].mean()
print("\n=== Timing (seconds) ===")
print(timing.sort_values("total_wall_time_sec"))

# 4. Statistical significance
from src.evaluation.statistical import friedman_test, nemenyi_cd_test

friedman_result = friedman_test(df, metric="roc_auc")
print("\n=== Friedman Test ===")
print(f"Significant: {friedman_result['significant']}, p-value: {friedman_result['p_value']:.4f}")

if friedman_result['significant']:
    nemenyi = nemenyi_cd_test(df, metric="roc_auc")
    print("\n=== Significant Pairwise Differences ===")
    print(nemenyi[nemenyi['significant']])
```

### 9.2 What Matters Most?

| Your Goal | Key Metrics to Focus On |
|-----------|------------------------|
| **Best raw performance** | ROC AUC, PR-AUC |
| **Reliable probabilities** | ECE, Brier Score, Log Loss |
| **Fast deployment** | Total wall time, Predict time |
| **Balanced across classes** | F1 Macro, Balanced Accuracy |
| **Statistical confidence** | Friedman test + Nemenyi + CIs |
| **Memory constraints** | Peak memory, Memory delta |

### 9.3 Interpreting TabPFN vs GBDT

| Metric | TabPFN Typically | GBDT Typically |
|--------|------------------|-----------------|
| ROC AUC | Competitive | Slightly higher with tuning |
| ECE | **Lower** (better calibrated) | Higher |
| Tuning time | **0** | 130-230s |
| Total time | **16× faster** | Baseline |
| Stability | Good (low variance) | Good |

**Key Insight:** TabPFN trades ~2-3% ROC AUC for 16× speedup and zero tuning effort. The ECE advantage suggests better probability calibration.

---

*Document generated: July 2026*
*Project: TabFM Benchmark*
*Author: Siri1702*