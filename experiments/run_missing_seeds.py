"""
Run seeds 1-4 for the 5 baseline models on the 7 datasets that only have seed 0.

Context: The July 5 run was a single-seed sweep (seed 0 only). The July 16 re-run
added seeds 1-4 but only for 8 datasets. This script fills in the gap for the
remaining 7: australian, kc1, kc2, ozone-level, pc1, phoneme, sick.

Writes raw JSON to experiments/results/raw/ and merges into
experiments/results/aggregated/results.csv (deduplication-safe).

Usage:
    python experiments/run_missing_seeds.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.evaluation.metrics import compute_metrics
from src.models.catboost_model import CatBoostWrapper
from src.models.lightgbm_model import LightGBMWrapper
from src.models.mlp_model import MLPWrapper
from src.models.xgboost_model import XGBoostWrapper

# tabpfn_model.py imports from tabpfn_client but installed package is tabpfn v8+
# Patch the import before loading the wrapper
import tabpfn
import sys as _sys
import types as _types
_tc = _types.ModuleType("tabpfn_client")
_tc.TabPFNClassifier = tabpfn.TabPFNClassifier
_sys.modules["tabpfn_client"] = _tc

from src.models.tabpfn_model import TabPFNWrapper  # noqa: E402

MISSING_DATASETS = [
    {"id": 40536, "name": "australian"},
    {"id": 1067,  "name": "kc1"},
    {"id": 1063,  "name": "kc2"},
    {"id": 1487,  "name": "ozone-level"},
    {"id": 1068,  "name": "pc1"},
    {"id": 1489,  "name": "phoneme"},
    {"id": 1050,  "name": "sick"},
]

MISSING_SEEDS = [1, 2, 3, 4]

AGG_CSV = Path("experiments/results/aggregated/results.csv")
RAW_DIR = Path("experiments/results/raw")


def build_baseline_models(seed: int) -> list:
    return [
        TabPFNWrapper(device="cpu", random_state=seed),
        XGBoostWrapper(n_trials=15, random_state=seed),
        LightGBMWrapper(n_trials=15, random_state=seed),
        CatBoostWrapper(n_trials=15, random_state=seed),
        MLPWrapper(n_trials=10, random_state=seed),
    ]


def already_done(existing_df: pd.DataFrame, dataset: str, model: str, seed: int) -> bool:
    if existing_df.empty:
        return False
    mask = (
        (existing_df["dataset"] == dataset)
        & (existing_df["model"] == model)
        & (existing_df["seed"] == seed)
    )
    return mask.any()


def main():
    loader = DataLoader(test_size=0.2, random_state=42)

    existing_df = pd.read_csv(AGG_CSV) if AGG_CSV.exists() else pd.DataFrame()
    print(f"Loaded {len(existing_df)} existing rows from results.csv")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_new = []

    for ds in MISSING_DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds['name']} (OpenML {ds['id']})")
        print(f"{'='*60}")

        dataset = loader.load_openml(ds["id"], ds["name"])
        ds_results = []

        for seed in tqdm(MISSING_SEEDS, desc=f"  Seeds", leave=False):
            models = build_baseline_models(seed)
            for model in models:
                if already_done(existing_df, ds["name"], model.name, seed):
                    print(f"    {model.name:12s} | seed={seed}  -- already done, skip")
                    continue

                print(f"    {model.name:12s} | seed={seed}", end=" ", flush=True)
                result = model.run(
                    dataset.X_train, dataset.y_train,
                    dataset.X_test,  dataset.y_test,
                    dataset_name=ds["name"],
                    seed=seed,
                )
                metrics = compute_metrics(result)
                ds_results.append(metrics)
                all_new.append(metrics)

                status = (
                    f"AUC={metrics.get('roc_auc', float('nan')):.4f}"
                    if metrics.get("error") is None
                    else f"ERROR: {metrics['error']}"
                )
                print(f"==> {status} ({result.fit_time_sec:.1f}s fit)")

        # Save raw JSON for this dataset immediately (don't lose work on crash)
        if ds_results:
            raw_path = RAW_DIR / f"{run_id}_missing_seeds_{ds['name']}.json"
            with open(raw_path, "w") as f:
                json.dump(ds_results, f, indent=2, default=str)
            print(f"  -> Saved raw: {raw_path.name}")

    if not all_new:
        print("\nNothing new to add — all seeds already present.")
        return

    # Merge into aggregated CSV
    new_df = pd.DataFrame(all_new)
    merged = pd.concat([existing_df, new_df], ignore_index=True)

    # Safety: deduplicate on (model, dataset, seed) keeping last (new) entry
    merged = merged.drop_duplicates(subset=["model", "dataset", "seed"], keep="last")
    merged.to_csv(AGG_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Done. Added {len(all_new)} new rows.")
    print(f"Aggregated CSV now has {len(merged)} rows.")
    print(f"{'='*60}")

    # Coverage check
    pivot = merged.pivot_table(
        index=["dataset", "seed"], columns="model", values="roc_auc", aggfunc="first"
    )
    complete = pivot.dropna()
    print(f"\nComplete cases (all models non-null): {len(complete)} / {len(pivot)}")


if __name__ == "__main__":
    main()
