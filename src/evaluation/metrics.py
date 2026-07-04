"""
Computes all evaluation metrics from a RunResult.
Returns a flat dict ready for pandas DataFrame construction.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from src.models.base import RunResult  # noqa — import from base


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures the difference between predicted confidence and actual accuracy.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_proba[in_bin])
            ece += (bin_count / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute calibration metrics: ECE, MCE (Maximum Calibration Error).
    """
    y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)

    ece_10 = compute_ece(y_true, y_proba, n_bins=10)
    ece_15 = compute_ece(y_true, y_proba, n_bins=15)

    # Maximum Calibration Error
    bin_edges = np.linspace(0, 1, 15 + 1)
    max_calibration_error = 0.0
    for i in range(len(bin_edges) - 1):
        in_bin = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_proba[in_bin])
            max_calibration_error = max(max_calibration_error, abs(bin_acc - bin_conf))

    return {
        "ece_10": ece_10,
        "ece_15": ece_15,
        "max_calibration_error": max_calibration_error,
    }


def compute_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute ROC curve and PR curve metrics.
    Returns threshold-independent metrics.
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = fpr[optimal_idx] if optimal_idx < len(fpr) else 0.5

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": optimal_threshold,
        "optimal_tpr": tpr[optimal_idx] if optimal_idx < len(tpr) else np.nan,
        "optimal_fpr": fpr[optimal_idx] if optimal_idx < len(fpr) else np.nan,
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute per-class precision, recall, and F1.
    """
    # Per-class precision and recall
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "precision_class_0": precision[0] if len(precision) > 0 else np.nan,
        "precision_class_1": precision[1] if len(precision) > 1 else np.nan,
        "recall_class_0": recall[0] if len(recall) > 0 else np.nan,
        "recall_class_1": recall[1] if len(recall) > 1 else np.nan,
        "f1_class_0": f1[0] if len(f1) > 0 else np.nan,
        "f1_class_1": f1[1] if len(f1) > 1 else np.nan,
    }


def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              sensitive_attribute: np.ndarray = None) -> dict:
    """
    Compute fairness metrics if sensitive attribute is available.

    Metrics computed:
    - Demographic Parity: P(Y_pred=1)
    - Equalized Odds: TPR and FPR by group
    - Balanced Accuracy
    - Matthews Correlation Coefficient
    """
    if sensitive_attribute is None:
        # Return basic fairness-adjacent metrics without group info
        return {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "demographic_parity": np.mean(y_pred),
            "tpr_overall": np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1),
            "fpr_overall": np.sum((y_true == 0) & (y_pred == 1)) / max(np.sum(y_true == 0), 1),
        }

    # Compute group-wise metrics
    unique_groups = np.unique(sensitive_attribute)
    tpr_by_group = []
    fpr_by_group = []
    pp_by_group = []

    for group in unique_groups:
        mask = sensitive_attribute == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        if np.sum(y_true_g == 1) > 0:
            tpr_g = np.sum((y_true_g == 1) & (y_pred_g == 1)) / np.sum(y_true_g == 1)
        else:
            tpr_g = np.nan

        if np.sum(y_true_g == 0) > 0:
            fpr_g = np.sum((y_true_g == 0) & (y_pred_g == 1)) / np.sum(y_true_g == 0)
        else:
            fpr_g = np.nan

        pp_g = np.mean(y_pred_g)

        tpr_by_group.append(tpr_g)
        fpr_by_group.append(fpr_g)
        pp_by_group.append(pp_g)

    # Compute disparities (max - min across groups)
    tpr_disparity = np.nanmax(tpr_by_group) - np.nanmin(tpr_by_group)
    fpr_disparity = np.nanmax(fpr_by_group) - np.nanmin(fpr_by_group)
    pp_disparity = np.nanmax(pp_by_group) - np.nanmin(pp_by_group)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "demographic_parity": np.mean(y_pred),
        "demographic_parity_disparity": pp_disparity,
        "tpr_disparity": tpr_disparity,
        "fpr_disparity": fpr_disparity,
    }


def compute_metrics(result: RunResult) -> dict:
    """
    Given a RunResult, compute all metrics and return as a flat dict.
    Returns NaN for metrics that cannot be computed (e.g., error run).

    Includes:
    - Core metrics (ROC-AUC, PR-AUC, F1, Brier, Log Loss)
    - Calibration metrics (ECE, MCE)
    - ROC/PR curve metrics
    - Per-class metrics
    - Fairness-related metrics
    """
    base = {
        "model": result.model_name,
        "dataset": result.dataset_name,
        "seed": result.seed,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": result.n_features,
        "fit_time_sec": result.fit_time_sec,
        "predict_time_sec": result.predict_time_sec,
        "total_time_sec": result.fit_time_sec + result.predict_time_sec,
        "tuning_time_sec": result.tuning_time_sec,
        "total_wall_time_sec": result.total_wall_time_sec,
        "throughput_rows_per_sec": result.throughput_rows_per_sec,
        "ms_per_test_row": (result.predict_time_sec / result.n_test) * 1000 if result.n_test > 0 else None,
        "peak_memory_mb": result.peak_memory_mb,
        "memory_delta_mb": result.memory_delta_mb,
        "gpu_memory_mb": result.gpu_memory_mb,
        "error": result.error,
    }

    if result.error is not None:
        # Fill all metrics with NaN for failed runs
        base.update({k: np.nan for k in [
            "roc_auc", "avg_precision", "f1_macro", "pr_auc",
            "brier_score", "log_loss_val", "balanced_accuracy",
            "ece_10", "ece_15", "max_calibration_error",
            "precision_class_0", "precision_class_1",
            "recall_class_0", "recall_class_1",
            "f1_class_0", "f1_class_1",
            "peak_memory_mb", "memory_delta_mb", "gpu_memory_mb",
        ]})
        return base

    y = result.y_test
    yp = result.y_proba
    yh = result.y_pred

    # Clip probabilities to avoid log(0)
    yp_clipped = np.clip(yp, 1e-7, 1 - 1e-7)

    # Core metrics
    base.update({
        "roc_auc":       roc_auc_score(y, yp),
        "avg_precision": average_precision_score(y, yp),
        "f1_macro":      f1_score(y, yh, average="macro", zero_division=0),
        "brier_score":   brier_score_loss(y, yp),
        "log_loss_val":  log_loss(y, yp_clipped),
    })

    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    base.update({
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,  # recall
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv":         tp / (tp + fp) if (tp + fp) > 0 else np.nan,  # precision
    })

    # Calibration metrics
    calibration = compute_calibration_metrics(y, yp_clipped)
    base.update(calibration)

    # ROC/PR curve metrics
    curve_metrics = compute_roc_pr_curves(y, yp)
    base.update(curve_metrics)

    # Per-class metrics
    per_class = compute_per_class_metrics(y, yh)
    base.update(per_class)

    # Fairness metrics (without sensitive attribute - basic version)
    fairness = compute_fairness_metrics(y, yh)
    base.update(fairness)

    return base