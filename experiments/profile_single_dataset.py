"""
Single-dataset profiling run.
Captures: fit_time, predict_time, AUC, CPU%, RAM MB, GPU util%, GPU memory MB
for every model on credit-g (1 seed).

Usage:
    python experiments/profile_single_dataset.py
"""
import sys, time, threading, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import psutil
import torch
import numpy as np

from src.data.loader import DataLoader
from src.models.tabpfn_model import TabPFNWrapper
from src.models.xgboost_model import XGBoostWrapper
from src.models.lightgbm_model import LightGBMWrapper
from src.models.catboost_model import CatBoostWrapper
from src.models.mlp_model import MLPWrapper
from src.models.sap_rpt_model import SAPRptWrapper
from src.evaluation.metrics import compute_metrics

SAMPLE_INTERVAL = 0.5   # seconds between resource samples


def _gpu_stats():
    """Return (util_pct, mem_used_mb) from nvidia-smi, or (0,0) if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip().split(",")
        return float(out[0].strip()), float(out[1].strip())
    except Exception:
        return 0.0, 0.0


class ResourceSampler:
    """Samples CPU%, RAM MB, GPU util%, GPU MB in a background thread."""

    def __init__(self):
        self._samples = []
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                cpu = self._proc.cpu_percent(interval=None)
                ram = self._proc.memory_info().rss / 1024 / 1024
                gpu_util, gpu_mem = _gpu_stats()
                self._samples.append((cpu, ram, gpu_util, gpu_mem))
            except Exception:
                pass
            time.sleep(SAMPLE_INTERVAL)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def summary(self):
        if not self._samples:
            return {"cpu_mean": 0, "cpu_peak": 0,
                    "ram_mean": 0, "ram_peak": 0,
                    "gpu_util_mean": 0, "gpu_util_peak": 0,
                    "gpu_mem_mean": 0, "gpu_mem_peak": 0}
        cpus  = [s[0] for s in self._samples]
        rams  = [s[1] for s in self._samples]
        gutil = [s[2] for s in self._samples]
        gmem  = [s[3] for s in self._samples]
        return {
            "cpu_mean_pct":  round(np.mean(cpus),  1),
            "cpu_peak_pct":  round(np.max(cpus),   1),
            "ram_mean_mb":   round(np.mean(rams),  0),
            "ram_peak_mb":   round(np.max(rams),   0),
            "gpu_util_mean_pct": round(np.mean(gutil), 1),
            "gpu_util_peak_pct": round(np.max(gutil),  1),
            "gpu_mem_mean_mb":   round(np.mean(gmem),  0),
            "gpu_mem_peak_mb":   round(np.max(gmem),   0),
        }


def profile_model(model, X_train, y_train, X_test, y_test, model_name):
    sampler = ResourceSampler()

    # --- FIT ---
    sampler.start()
    t0 = time.perf_counter()
    try:
        model.fit(X_train, y_train)
        fit_err = None
    except Exception as e:
        fit_err = str(e)
    fit_time = time.perf_counter() - t0
    sampler.stop()
    fit_stats = sampler.summary()

    if fit_err:
        return {"model": model_name, "error": fit_err}

    # --- PREDICT ---
    sampler.start()
    t1 = time.perf_counter()
    try:
        proba = model.predict_proba(X_test)
        pred_err = None
    except Exception as e:
        pred_err = str(e)
    predict_time = time.perf_counter() - t1
    sampler.stop()
    pred_stats = sampler.summary()

    if pred_err:
        return {"model": model_name, "fit_time_s": round(fit_time, 2), "error": pred_err}

    from sklearn.metrics import roc_auc_score
    auc = round(roc_auc_score(y_test, proba), 4)

    return {
        "model":            model_name,
        "auc":              auc,
        "fit_time_s":       round(fit_time, 2),
        "predict_time_s":   round(predict_time, 3),
        "total_time_s":     round(fit_time + predict_time, 2),
        # FIT resource usage
        "fit_cpu_mean%":    fit_stats["cpu_mean_pct"],
        "fit_cpu_peak%":    fit_stats["cpu_peak_pct"],
        "fit_ram_peak_mb":  fit_stats["ram_peak_mb"],
        "fit_gpu_util%":    fit_stats["gpu_util_mean_pct"],
        "fit_gpu_mem_mb":   fit_stats["gpu_mem_peak_mb"],
        # PREDICT resource usage
        "pred_cpu_mean%":   pred_stats["cpu_mean_pct"],
        "pred_ram_peak_mb": pred_stats["ram_peak_mb"],
        "pred_gpu_util%":   pred_stats["gpu_util_mean_pct"],
        "pred_gpu_mem_mb":  pred_stats["gpu_mem_peak_mb"],
    }


def main():
    print("Loading credit-g (OpenML 31)...")
    loader = DataLoader(test_size=0.2, random_state=42)
    ds = loader.load_openml(31, "credit-g")
    print(f"  Train: {len(ds.X_train)} rows  Test: {len(ds.X_test)} rows  "
          f"Features: {ds.n_features}  GPU: {torch.cuda.get_device_name(0)}\n")

    models = [
        (TabPFNWrapper(device="cuda", random_state=0),   "TabPFN (GPU)"),
        (XGBoostWrapper(n_trials=15, random_state=0),    "XGBoost (CPU)"),
        (LightGBMWrapper(n_trials=15, random_state=0),   "LightGBM (CPU)"),
        (CatBoostWrapper(n_trials=15, random_state=0),   "CatBoost (GPU)"),
        (MLPWrapper(n_trials=10, random_state=0),        "MLP (CPU)"),
        (SAPRptWrapper(random_state=0),                  "SAP-RPT-1 (GPU)"),
    ]

    results = []
    for model, name in models:
        print(f"  Running {name}...", flush=True)
        r = profile_model(model, ds.X_train, ds.y_train,
                          ds.X_test, ds.y_test, name)
        results.append(r)
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    AUC={r['auc']}  fit={r['fit_time_s']}s  "
                  f"predict={r['predict_time_s']}s  "
                  f"CPU_peak={r['fit_cpu_peak%']}%  "
                  f"RAM_peak={r['fit_ram_peak_mb']}MB  "
                  f"GPU_util={r['fit_gpu_util%']}%  "
                  f"GPU_mem={r['fit_gpu_mem_mb']}MB")

    # --- Summary table ---
    print("\n" + "=" * 110)
    print(f"{'Model':<20} {'AUC':>6} {'Fit(s)':>8} {'Pred(s)':>8} {'Total(s)':>9} "
          f"{'CPU%(mean)':>11} {'CPU%(peak)':>11} {'RAM_pk(MB)':>11} "
          f"{'GPU_util%':>10} {'GPU_mem(MB)':>12}")
    print("-" * 110)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<20}  ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:<20} {r['auc']:>6} {r['fit_time_s']:>8.1f} "
            f"{r['predict_time_s']:>8.3f} {r['total_time_s']:>9.1f} "
            f"{r['fit_cpu_mean%']:>11} {r['fit_cpu_peak%']:>11} "
            f"{r['fit_ram_peak_mb']:>11.0f} "
            f"{r['fit_gpu_util%']:>10} {r['fit_gpu_mem_mb']:>12.0f}"
        )
    print("=" * 110)
    print(f"\n(Predict-time GPU util — "
          + "  ".join(f"{r['model'].split()[0]}:{r.get('pred_gpu_util%','N/A')}%" for r in results if 'error' not in r)
          + ")")


if __name__ == "__main__":
    main()
