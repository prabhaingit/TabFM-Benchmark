"""
sklearn MLP wrapper with StandardScaler (required for gradient-based models).
Uses Optuna for hidden layer architecture and regularisation search.
"""

import numpy as np
import optuna
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from .base import ModelWrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Architecture choices as discrete options
ARCHITECTURES = [
    (64,), (128,), (256,),
    (128, 64), (256, 128), (256, 128, 64),
]


class MLPWrapper(ModelWrapper):

    def __init__(self, n_trials: int = 30, n_cv_folds: int = 3,
                 random_state: int = 42, timeout: int = 90):
        super().__init__(name="MLP", random_state=random_state)
        self.n_trials = n_trials
        self.n_cv_folds = n_cv_folds
        self.timeout = timeout
        self.best_params_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        def objective(trial):
            arch_idx = trial.suggest_int("arch_idx", 0, len(ARCHITECTURES) - 1)
            params = {
                "hidden_layer_sizes": ARCHITECTURES[arch_idx],
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-2, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 200, 500),
                "early_stopping": True,
                "random_state": self.random_state,
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**params)),
            ])
            cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True,
                                 random_state=self.random_state)
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()
        
        t_tune_start = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.tuning_time_sec_ = time.perf_counter() - t_tune_start
        self.best_params_ = study.best_params

        arch = ARCHITECTURES[self.best_params_.pop("arch_idx")]
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=arch,
                early_stopping=True,
                random_state=self.random_state,
                **self.best_params_,
            )),
        ])
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1]