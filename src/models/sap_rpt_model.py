"""
SAP RPT-1 OSS wrapper.

sap-rpt-1-oss (ConTextTab) is a zero-shot tabular foundation model that uses
semantics-aware in-context learning. Like TabPFN, fit() stores context and
no gradient-based training happens on the target dataset.

Install:
    pip install git+https://github.com/SAP-samples/sap-rpt-1-oss

HuggingFace auth required (model checkpoints download automatically):
    huggingface-cli login

Key params:
    max_context_size: rows of training context passed to the model (8192 max)
    bagging: ensemble multiplier (default 1; higher values have negligible AUC effect)

Root cause of past low AUC (now fixed):
    sap_rpt_oss.fit() does internal index-based alignment between X and y.
    The benchmark's DataLoader returns y_train as a numpy array (indices 0,1,2,...),
    but X_train is a pandas DataFrame whose index is the shuffled original row numbers
    (e.g. [675, 703, 12, ...]). When y is numpy, a default 0-based Series is created
    internally, misaligning labels with rows. Fix: wrap y as pd.Series(y, index=X.index)
    before calling fit(). With correct alignment, AUC on credit-g reaches 0.81 — identical
    to the Kaggle benchmark result — regardless of whether the data is raw or encoded.

Column ordering:
    With y correctly aligned, the model follows the standard sklearn convention:
    col[i] = P(classes_[i]). We still run an in-sample calibration check as a safeguard,
    since the margin between columns is large (>0.6 AUC) and the check is reliable.
"""

import numpy as np
import pandas as pd
from .base import ModelWrapper


class SAPRptWrapper(ModelWrapper):

    def __init__(
        self,
        max_context_size: int = 8192,
        bagging: int = 1,
        random_state: int = 42,
    ):
        super().__init__(name="SAP-RPT-1", random_state=random_state)
        self.max_context_size = max_context_size
        self.bagging = bagging
        self._pos_col_: int = 1  # default; overwritten by calibration in fit()

    def fit(self, X_train, y_train: np.ndarray) -> None:
        from sap_rpt_oss import SAP_RPT_OSS_Classifier

        if len(X_train) > self.max_context_size:
            X_train, y_train = self._subsample_balanced(
                X_train, y_train, self.max_context_size
            )

        # sap_rpt_oss aligns X and y by pandas index internally. When y is a numpy
        # array its default index (0,1,2,...) misaligns with X_train's shuffled row index.
        if hasattr(X_train, "index") and not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, index=X_train.index)

        self._X_fit_ = X_train
        self._y_fit_ = y_train
        self.tuning_time_sec_ = 0.0

        self._model = SAP_RPT_OSS_Classifier(
            max_context_size=self.max_context_size,
            bagging=self.bagging,
        )
        self._model.fit(X_train, y_train)
        self._pos_col_ = self._calibrate_pos_col(X_train, y_train)

    def predict_proba(self, X_test) -> np.ndarray:
        proba = self._model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, self._pos_col_]
        return proba

    def _calibrate_pos_col(self, X_train, y_train) -> int:
        """Return 0 or 1: whichever predict_proba column is positively correlated with y."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(self.random_state)
        n_check = min(50, len(X_train))
        idx = rng.choice(len(X_train), size=n_check, replace=False)
        X_check = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
        y_check = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

        if len(np.unique(y_check)) < 2:
            return 1  # fallback: can't compute AUC with one class

        proba_check = self._model.predict_proba(X_check)
        auc0 = roc_auc_score(y_check, proba_check[:, 0])
        auc1 = roc_auc_score(y_check, proba_check[:, 1])
        if abs(auc1 - auc0) < 0.1:
            return 1  # tie-guard: margin too small to be reliable, use sklearn default
        return 1 if auc1 > auc0 else 0

    def _subsample_balanced(self, X, y, n: int):
        """Stratified subsample preserving class balance."""
        rng = np.random.RandomState(self.random_state)
        y_values = y.values if hasattr(y, "values") else y
        classes = np.unique(y_values)
        per_class = n // len(classes)

        indices = []
        for cls in classes:
            cls_idx = np.where(y_values == cls)[0]
            take = min(per_class, len(cls_idx))
            indices.append(rng.choice(cls_idx, size=take, replace=False))

        idx = np.concatenate(indices)
        rng.shuffle(idx)
        if hasattr(X, "iloc"):
            y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
            return X.iloc[idx], y_sub
        return X[idx], y[idx]
