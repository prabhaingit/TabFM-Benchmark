"""
Run SAP-RPT-1 on the 7 datasets where it has no results yet, then merge
into experiments/results/aggregated/results.csv.

Usage:
    python experiments/run_saprpt_missing.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from src.data.loader import DataLoader
from src.models.sap_rpt_model import SAPRptWrapper
from src.evaluation.metrics import compute_metrics

MISSING_DATASETS = [
    {"id": 40536, "name": "australian"},
    {"id": 1067,  "name": "kc1"},
    {"id": 1063,  "name": "kc2"},
    {"id": 1487,  "name": "ozone-level"},
    {"id": 1068,  "name": "pc1"},
    {"id": 1489,  "name": "phoneme"},
    {"id": 1050,  "name": "sick"},
]

N_SEEDS = 5
TEST_SIZE = 0.2
AGG_PATH = "experiments/results/aggregated/results.csv"
RAW_DIR = Path("experiments/results/raw")


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_new = []

    for ds in MISSING_DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds['name']} (OpenML {ds['id']})")
        print(f"{'='*60}")

        loader = DataLoader(test_size=TEST_SIZE, random_state=42)
        dataset = loader.load_openml(ds["id"], ds["name"])

        ds_results = []
        for seed in tqdm(range(N_SEEDS), desc=f"  Seeds for {ds['name']}", leave=False):
            model = SAPRptWrapper(random_state=seed)
            print(f"    SAP-RPT-1    | seed={seed}", end=" ", flush=True)
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

        # Save raw results per dataset
        raw_path = RAW_DIR / f"{run_id}_saprpt_{ds['name']}.json"
        with open(raw_path, "w") as f:
            json.dump(ds_results, f, indent=2, default=str)
        print(f"  Saved raw -> {raw_path.name}")

    # Merge into aggregated results
    print(f"\nMerging {len(all_new)} new results into {AGG_PATH} ...")
    existing = pd.read_csv(AGG_PATH)
    new_df = pd.DataFrame(all_new)

    # Drop any stale SAP-RPT-1 rows for these datasets (shouldn't be any, but safe)
    mask = (existing["model"] == "SAP-RPT-1") & (existing["dataset"].isin([d["name"] for d in MISSING_DATASETS]))
    existing = existing[~mask]

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged.to_csv(AGG_PATH, index=False)
    print(f"  Aggregated results: {len(existing)} existing + {len(new_df)} new = {len(merged)} total")
    print("\nDone.")


if __name__ == "__main__":
    main()
