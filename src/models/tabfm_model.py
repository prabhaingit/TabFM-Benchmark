"""
TabFM wrapper for the TabFM Benchmark.

Google's TabFM (Tabular Foundation Model) is a zero-shot foundation model
that uses in-context learning to make predictions on tabular data.

Key features:
- ZERO-SHOT: No hyperparameter tuning required
- Supports mixed numerical and categorical features natively
- Uses in-context learning (reads training data as context)
- Available in JAX and PyTorch backends

Requirements:
- Python >= 3.11
- tabfm package installed

Reference: https://github.com/google-research/tabfm
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
import tabfm
import torch

from .base import ModelWrapper


class TabFMClassifierWrapper(ModelWrapper):
    """
    Wrapper for Google's TabFM classifier.

    TabFM is a zero-shot tabular foundation model that uses in-context learning.
    It stores the training data as context and makes predictions without
    gradient-based training on the specific dataset.

    Note: TabFM requires Python >= 3.11
    """

    def __init__(
        self,
        backend: str = "pytorch",
        device: str = "cpu",
        random_state: int = 42,
    ):
        """
        Initialize TabFM wrapper.

        Args:
            backend: "jax" or "pytorch" for model loading
            device: "cpu" or "cuda" (GPU support depends on backend)
            random_state: Random seed for reproducibility
        """
        super().__init__(name=f"TabFM ({backend.upper()})", random_state=random_state)
        self.backend = backend
        self.device = device
        self._model = None
        self._classifier = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit TabFM model using in-context learning.

        For TabFM, 'fit' = storing the context (X_train, y_train) for ICL.
        The model doesn't do gradient-based training on the dataset.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
        """
        # TabFM requires pandas DataFrame input
        # Convert numpy array to DataFrame
        if isinstance(X_train, np.ndarray):
            X_train = self._numpy_to_dataframe(X_train)

        # Ensure y_train is numpy array
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        elif not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)

        # Load model and classifier
        self._load_model()

        # Fit the classifier (stores context for ICL)
        self._classifier.fit(X_train, y_train)

        # TabFM is zero-shot - no tuning time
        self.tuning_time_sec_ = 0.0

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all classes.

        Args:
            X_test: Test features, shape (n_samples, n_features)

        Returns:
            For binary: probabilities for positive class, shape (n_samples,)
            For multi-class: probabilities for all classes, shape (n_samples, n_classes)
        """
        if self._classifier is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to DataFrame if needed
        if isinstance(X_test, np.ndarray):
            X_test = self._numpy_to_dataframe(X_test)

        # Get probabilities from model
        proba = self._classifier.predict_proba(X_test)

        # Handle different output shapes
        if proba.ndim == 2 and proba.shape[1] > 1:
            # Multi-class: return full probability matrix
            return proba
        elif proba.shape[1] == 2:
            # Binary classification: return probability of class 1
            return proba[:, 1]
        else:
            # Fallback: flatten
            return proba.ravel()

    def _load_model(self):
        """Load TabFM model from pretrained weights."""
        if self._model is not None:
            return

        try:
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

            self._model = tabfm_v1_0_0.load(model_type='classification')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)


            # Create classifier
            from tabfm import TabFMClassifier
            self._classifier = TabFMClassifier(model=self._model)

        except ImportError as e:
            raise ImportError(
                f"TabFM not installed. Install with: pip install tabfm\n"
                f"Also requires Python >= 3.11\n"
                f"Error: {e}"
            )

    def _numpy_to_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """
        Convert numpy array to pandas DataFrame with generic column names.

        TabFM requires DataFrame input with column names.
        """
        n_features = X.shape[1]
        columns = [f"col_{i}" for i in range(n_features)]
        return pd.DataFrame(X, columns=columns)


class TabFMJAXWrapper(TabFMClassifierWrapper):
    """TabFM with JAX backend."""

    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(backend="jax", random_state=random_state, **kwargs)


class TabFMPyTorchWrapper(TabFMClassifierWrapper):
    """TabFM with PyTorch backend."""

    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(backend="pytorch", random_state=random_state, **kwargs)


# Convenience function for quick testing
def test_tabfm_installation():
    """Quick test to verify TabFM is installed and working."""
    try:
        from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
        model = tabfm_v1_0_0.load()
        from tabfm import TabFMClassifier
        clf = TabFMClassifier(model=model)

        # Simple test
        X_train = pd.DataFrame({
            "age": [25.0, 35.0, 45.0, 55.0],
            "income": [50000, 75000, 100000, 125000]
        })
        y_train = np.array([0, 0, 1, 1])

        clf.fit(X_train, y_train)

        X_test = pd.DataFrame({
            "age": [30.0],
            "income": [60000]
        })

        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        print(f"✓ TabFM test successful!")
        print(f"  Prediction: {pred}")
        print(f"  Probability: {proba}")
        return True

    except ImportError as e:
        print(f"✗ TabFM not available: {e}")
        return False
    except Exception as e:
        print(f"✗ TabFM test failed: {e}")
        return False


if __name__ == "__main__":
    test_tabfm_installation()