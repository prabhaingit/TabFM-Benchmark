"""
TabPFN wrapper.

Critical design choices:
- TabPFN is ZERO-SHOT: we call fit() but it doesn't train, it stores context
- N_ensemble_configurations controls the number of augmented forward passes
- For datasets > tabpfn_max_rows, we subsample the training context
- TabPFN v2 handles missing values internally — we pass preprocessed data anyway
"""

import numpy as np
from tabpfn import TabPFNClassifier
from .base import ModelWrapper


class TabPFNWrapper(ModelWrapper):

    def __init__(
        self,
        device: str = "cpu",
        n_ensemble_configurations: int = 8,
        max_context_size: int = 8000,   # subsample if train > this
        random_state: int = 42,
    ):
        super().__init__(name="TabPFN", random_state=random_state)
        self.device = device
        self.n_ensemble_configurations = n_ensemble_configurations
        self.max_context_size = max_context_size

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        For TabPFN, 'fit' = storing the context (X_train, y_train).
        If the context is too large, we subsample — preserving class balance.
        """
        if len(X_train) > self.max_context_size:
            X_train, y_train = self._subsample_balanced(
                X_train, y_train, self.max_context_size
            )
        self.tuning_time_sec_ = 0.0

        self._model = TabPFNClassifier(
            n_estimators=8,
            device=self.device,
        )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(X_test)
        # Handle binary case: return probability of class 1
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)  # fallback for edge cases

    def _subsample_balanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stratified subsample to keep class balance when training set is too large.
        Uses numpy for speed — no sklearn dependency here.
        """
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        per_class = n // len(classes)
        
        indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            take = min(per_class, len(cls_idx))
            indices.append(rng.choice(cls_idx, size=take, replace=False))
        
        idx = np.concatenate(indices)
        rng.shuffle(idx)
        return X[idx], y[idx]