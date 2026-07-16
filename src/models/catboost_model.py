import numpy as np
import optuna
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .base import ModelWrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)


class CatBoostWrapper(ModelWrapper):

    def __init__(self, n_trials: int = 50, n_cv_folds: int = 3,
                 random_state: int = 42, timeout: int = 120):
        super().__init__(name="CatBoost", random_state=random_state)
        self.n_trials = n_trials
        self.n_cv_folds = n_cv_folds
        self.timeout = timeout
        self.best_params_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 32, 128),
                "random_seed": self.random_state,
                "task_type": "GPU",
                "verbose": 0,
            }
            clf = CatBoostClassifier(**params)
            cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True,
                                 random_state=self.random_state)
            scores = cross_val_score(clf, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=1)
            return scores.mean()
        
        t_tune_start = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=1)
        self.tuning_time_sec_ = time.perf_counter() - t_tune_start
        self.best_params_ = study.best_params

        self._model = CatBoostClassifier(
            **self.best_params_,
            random_seed=self.random_state,
            task_type="GPU",
            verbose=0,
        )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1]