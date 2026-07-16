"""
Main benchmark runner.

Usage:
    python experiments/run_benchmark.py --datasets credit-g diabetes spambase
    python experiments/run_benchmark.py --all           # run all 15 datasets
    python experiments/run_benchmark.py --dataset credit-g --n_seeds 5  # quick test
    python experiments/run_benchmark.py --all --force   # force re-run all datasets

Results are saved to experiments/results/raw/ as JSON and
aggregated to experiments/results/aggregated/results.csv

Incremental Run Feature:
- Datasets that have already been run are automatically skipped
- New datasets are merged with existing aggregated results
- Use --force to re-run datasets that already have results

New Features:
- Friedman test for overall model comparison
- Nemenyi post-hoc test for pairwise comparisons
- Bootstrap confidence intervals
- Critical Difference (CD) diagrams
- Memory profiling (CPU/GPU)
- Enhanced metrics (calibration, per-class, fairness)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.tabpfn_model import TabPFNWrapper
from src.models.xgboost_model import XGBoostWrapper
from src.models.lightgbm_model import LightGBMWrapper
from src.models.catboost_model import CatBoostWrapper
from src.models.mlp_model import MLPWrapper
from src.models.sap_rpt_model import SAPRptWrapper
from src.models.tabfm_model import TabFMJAXWrapper, TabFMPyTorchWrapper
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical import (
    wilcoxon_pairwise,
    average_rank_table,
    friedman_test,
    nemenyi_cd_test,
    compute_bootstrap_ci,
)
from src.viz.leaderboard import plot_auc_heatmap, plot_average_ranks
from src.viz.leaderboard import plot_timing_comparison, plot_memory_comparison
from src.viz.statistical import (
    plot_critical_difference_diagram,
    plot_confidence_intervals,
    plot_statistical_significance_heatmap,
    plot_win_loss_matrix,
)
from src.viz.curves import plot_calibration_comparison
from src.evaluation.profiling import get_system_info


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_existing_datasets() -> set:
    """
    Find datasets that have already been run (at least one seed exists).
    Returns a set of dataset names that have existing results.
    """
    raw_dir = Path("experiments/results/raw")
    existing = set()

    if not raw_dir.exists():
        return existing

    # Check for JSON files that contain dataset results
    # Filename format: YYYYMMDD_HHMMSS_datasetname.json
    for json_file in raw_dir.glob("*.json"):
        # Split by underscore, format is: 20260628_121641_credit-g
        # We need to skip the first two parts (date_time)
        parts = json_file.stem.split("_")
        if len(parts) >= 3:
            # Join all parts after the second one (in case dataset has underscores)
            dataset_name = "_".join(parts[2:])
            existing.add(dataset_name)

    return existing


def load_existing_aggregated_results() -> pd.DataFrame:
    """
    Load the most recent aggregated results CSV.
    Prefers results.csv if it exists, otherwise loads the most recent timestamped file.
    Returns empty DataFrame if none exists.
    """
    agg_dir = Path("experiments/results/aggregated")

    if not agg_dir.exists():
        return pd.DataFrame()

    # First, check for results.csv (the merged/incremental results file)
    results_csv = agg_dir / "results.csv"
    if results_csv.exists():
        print(f"Loading existing results from: results.csv")
        return pd.read_csv(results_csv)

    # Fall back to the most recent timestamped results file
    csv_files = list(agg_dir.glob("results_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    # Sort by modification time and get the most recent
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading existing results from: {latest_csv.name}")

    return pd.read_csv(latest_csv)


def build_models(exp_cfg: dict, seed: int) -> list:
    """Instantiate all model wrappers for a given seed."""
    return [
        TabPFNWrapper(device="cpu", random_state=seed),
        XGBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 50),
                       random_state=seed),
        LightGBMWrapper(n_trials=exp_cfg.get("n_optuna_trials", 50),
                        random_state=seed),
        CatBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 50),
                        random_state=seed),
        MLPWrapper(n_trials=exp_cfg.get("n_optuna_trials", 30),
                   random_state=seed),
        SAPRptWrapper(random_state=seed),
        # TabFM - EXCLUDED: not working yet, will add back later
        # TabFMJAXWrapper(random_state=seed),
    ]


def run_single(dataset_cfg: dict, exp_cfg: dict, run_id: str = None) -> list[dict]:
    """
    Run all models × all seeds on a single dataset.
    Returns a list of metric dicts (one per model × seed).
    Supports caching to skip already completed runs.
    """
    loader = DataLoader(
        test_size=exp_cfg.get("test_size", 0.2),
        random_state=42,  # fixed split seed — the same train/test for ALL model seeds
    )
    dataset = loader.load_openml(
        dataset_id=dataset_cfg["id"],
        dataset_name=dataset_cfg["name"],
    )

    # Skip datasets that are too large for TabPFN even with subsampling
    if dataset.n_rows > exp_cfg.get("max_rows", 10000):
        print(f"  Skipping {dataset_cfg['name']}: {dataset.n_rows} rows > limit")
        return []

    seeds = list(range(exp_cfg.get("n_seeds", 5)))
    all_metrics = []

    # Check for cached results
    cache_enabled = exp_cfg.get("cache_results", True)
    cached_path = f"experiments/results/raw/{run_id}_{dataset_cfg['name']}.json" if run_id else None

    # Load cached if exists
    if cache_enabled and cached_path and Path(cached_path).exists():
        with open(cached_path, "r") as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} cached results for {dataset_cfg['name']}")
        return cached

    for seed in tqdm(seeds, desc=f"  Seeds for {dataset_cfg['name']}", leave=False):
        models = build_models(exp_cfg, seed)
        for model in models:
            print(f"    {model.name:12s} | seed={seed}", end=" ")
            result = model.run(
                dataset.X_train, dataset.y_train,
                dataset.X_test,  dataset.y_test,
                dataset_name=dataset_cfg["name"],
                seed=seed,
            )
            metrics = compute_metrics(result)
            all_metrics.append(metrics)

            status = f"AUC={metrics.get('roc_auc', float('nan')):.4f}" \
                     if metrics.get("error") is None \
                     else f"ERROR: {metrics['error']}"
            print(f"==> {status} ({result.fit_time_sec:.1f}s fit)")

    return all_metrics


def _generate_reports(results_df: pd.DataFrame):
    """
    Generate reports and visualisations from results.
    Called incrementally after each dataset completes.
    """
    if results_df.empty:
        return

    n_datasets = results_df["dataset"].nunique()
    print(f"  -> Generating reports for {n_datasets} datasets...")

    # Basic rankings
    try:
        rank_df = average_rank_table(results_df, metric="roc_auc")
    except Exception as e:
        print(f"  Warning: Could not generate rank table: {e}")
        return

    # Visualisations - AUC heatmap
    try:
        plot_auc_heatmap(results_df, save_path="reports/figures/auc_heatmap.png")
    except Exception as e:
        print(f"  Warning: Could not generate AUC heatmap: {e}")

    # Average ranks
    try:
        plot_average_ranks(rank_df, save_path="reports/figures/average_ranks.png")
    except Exception as e:
        print(f"  Warning: Could not generate average ranks plot: {e}")

    # Timing comparison
    try:
        plot_timing_comparison(results_df, save_path="reports/figures/timing_comparison.png")
    except Exception as e:
        print(f"  Warning: Could not generate timing comparison: {e}")

    # Memory comparison
    try:
        plot_memory_comparison(results_df, save_path="reports/figures/memory_comparison.png")
    except Exception as e:
        print(f"  Warning: Could not generate memory comparison: {e}")

    # Win/Loss matrix
    try:
        plot_win_loss_matrix(results_df, metric="roc_auc", save_path="reports/figures/win_loss_matrix.png")
    except Exception as e:
        print(f"  Warning: Could not generate win/loss matrix: {e}")

    # Calibration comparison (if ECE available)
    if "ece_10" in results_df.columns:
        try:
            plot_calibration_comparison(results_df, save_path="reports/figures/calibration_comparison.png")
        except Exception as e:
            print(f"  Warning: Could not generate calibration comparison: {e}")

    print(f"  -> Reports updated successfully")


def main():
    parser = argparse.ArgumentParser(description="Run TabFM benchmark")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Force re-run even if dataset has existing results")
    args = parser.parse_args()

    # Load configs
    dataset_cfgs = load_config("configs/datasets.yaml")["datasets"]
    exp_cfg      = load_config("configs/experiment.yaml")["experiment"]

    if args.n_seeds:
        exp_cfg["n_seeds"] = args.n_seeds
    if args.datasets:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in args.datasets]

    Path("experiments/results/raw").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/aggregated").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Check for existing datasets
    existing_datasets = get_existing_datasets()
    print(f"Already run datasets: {existing_datasets if existing_datasets else 'None'}")

    # Load existing aggregated results (if any)
    existing_df = load_existing_aggregated_results()
    if not existing_df.empty:
        print(f"Loaded {len(existing_df)} existing results from previous runs")

    # Filter out datasets that have already been run (unless --force is used)
    if not args.force:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] not in existing_datasets]
        if not dataset_cfgs:
            if existing_df.empty:
                print("\nNo datasets to run and no existing results found.")
                return
            print("\n✓ All requested datasets have already been run. Loading existing results...")
            results_df = existing_df
        else:
            print(f"\nSkipping {len(existing_datasets)} already-run datasets: {list(existing_datasets)}")
    else:
        print("\n--force flag: re-running all datasets (will overwrite raw results)")
        dataset_cfgs = dataset_cfgs

    # Run new datasets
    all_results = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_datasets_run = []

    # Track cumulative results for incremental updates
    cumulative_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()

    for dataset_cfg in dataset_cfgs:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_cfg['name']} (OpenML {dataset_cfg['id']})")
        print(f"{'='*60}")

        metrics_list = run_single(dataset_cfg, exp_cfg)
        all_results.extend(metrics_list)
        new_datasets_run.append(dataset_cfg['name'])

        # Save raw results after each dataset (don't lose work if crashes)
        raw_path = f"experiments/results/raw/{run_id}_{dataset_cfg['name']}.json"
        with open(raw_path, "w") as f:
            json.dump(metrics_list, f, indent=2, default=str)

        # INCREMENTAL: Update aggregated results after each dataset
        if metrics_list:
            new_df = pd.DataFrame(metrics_list)

            if not cumulative_df.empty:
                # Merge: append new results to existing ones
                # Check for duplicates (same model, dataset, seed) and keep only new ones
                existing_keys = set(zip(
                    cumulative_df['model'],
                    cumulative_df['dataset'],
                    cumulative_df['seed']
                ))

                # Filter new results to exclude any that already exist
                new_unique = new_df[
                    ~new_df.apply(lambda row: (row['model'], row['dataset'], row['seed']) in existing_keys, axis=1)
                ]

                if len(new_unique) > 0:
                    results_df = pd.concat([cumulative_df, new_unique], ignore_index=True)
                else:
                    results_df = cumulative_df
            else:
                results_df = new_df

            # Update cumulative_df for next iteration
            cumulative_df = results_df.copy()

            # Save aggregated results after each dataset
            agg_path = "experiments/results/aggregated/results.csv"
            results_df.to_csv(agg_path, index=False)
            print(f"  -> Updated aggregated results: {len(results_df)} total results")

            # INCREMENTAL: Generate reports after each dataset
            _generate_reports(results_df)

    # Final merge for any remaining results
    if all_results and not dataset_cfgs:
        # This handles the case where no new datasets were run but existing_df exists
        new_df = pd.DataFrame(all_results)

        if not existing_df.empty:
            existing_keys = set(zip(
                existing_df['model'],
                existing_df['dataset'],
                existing_df['seed']
            ))

            new_unique = new_df[
                ~new_df.apply(lambda row: (row['model'], row['dataset'], row['seed']) in existing_keys, axis=1)
            ]

            if len(new_unique) > 0:
                results_df = pd.concat([existing_df, new_unique], ignore_index=True)
            else:
                results_df = existing_df
        else:
            results_df = new_df
    elif not all_results and not dataset_cfgs:
        # Use existing results
        results_df = existing_df
    else:
        # Already handled in the loop above
        results_df = load_existing_aggregated_results()

    if results_df.empty:
        print("No results available. Check your dataset configs.")
        return

    # Save aggregated results (use a consistent name, not timestamp-based)
    agg_path = "experiments/results/aggregated/results.csv"
    results_df.to_csv(agg_path, index=False)
    print(f"\nSaved aggregated results ==> {agg_path}")

    # Count unique datasets to determine what analyses are valid
    n_datasets = results_df["dataset"].nunique()
    print(f"\nDatasets processed: {n_datasets}")

    # Statistical analysis
    print("\n--- Statistical Analysis ---")
    rank_df = average_rank_table(results_df, metric="roc_auc")
    print(rank_df.to_string(index=False))

    # Wilcoxon test requires at least 5 datasets for meaningful comparison
    wilcoxon_df = wilcoxon_pairwise(results_df, metric="roc_auc")
    if n_datasets >= 5 and not wilcoxon_df.empty:
        print("\nWilcoxon pairwise tests (significant pairs):")
        print(wilcoxon_df[wilcoxon_df["significant"]].to_string(index=False))
    else:
        print("\nWilcoxon pairwise tests skipped (need ≥5 datasets)")

    # Visualisations
    print("\n--- Generating Visualisations ---")
    plot_auc_heatmap(results_df, save_path="reports/figures/auc_heatmap.png")
    plot_average_ranks(rank_df, save_path="reports/figures/average_ranks.png")
    plot_timing_comparison(results_df, save_path="reports/figures/timing_comparison.png")
    plot_memory_comparison(results_df, save_path="reports/figures/memory_comparison.png")

    # Statistical Analysis - Enhanced
    print("\n--- Enhanced Statistical Analysis ---")

    # Friedman test requires at least 2 datasets
    friedman_result = friedman_test(results_df, metric="roc_auc")
    print(f"\nFriedman Test:")
    if "error" not in friedman_result:
        print(f"  Statistic: {friedman_result['statistic']:.4f}")
        print(f"  p-value: {friedman_result['p_value']:.4f}")
        print(f"  Significant: {friedman_result['significant']}")
        print(f"  Average ranks: {friedman_result['avg_ranks']}")
    else:
        print(f"  {friedman_result['error']}")

    # Nemenyi post-hoc test requires at least 2 datasets
    nemenyi_df = nemenyi_cd_test(results_df, metric="roc_auc")
    if not nemenyi_df.empty:
        print(f"\nNemenyi Post-hoc Test (significant pairs):")
        sig_pairs = nemenyi_df[nemenyi_df["significant"]]
        if not sig_pairs.empty:
            print(sig_pairs.to_string(index=False))
        else:
            print("  No significantly different pairs found")

        # Compute CD for diagram
        if not nemenyi_df.empty:
            cd = nemenyi_df["critical_difference"].iloc[0]
            plot_critical_difference_diagram(rank_df, cd, save_path="reports/figures/cd_diagram.png")
    else:
        print("\nNemenyi test skipped (need ≥2 datasets)")

    # Bootstrap confidence intervals - works for any number of datasets
    print("\n--- Bootstrap Confidence Intervals (95%) ---")
    ci_df = compute_bootstrap_ci(results_df, metric="roc_auc", n_bootstrap=1000)
    if not ci_df.empty:
        print(ci_df.to_string(index=False))
        plot_confidence_intervals(ci_df, save_path="reports/figures/confidence_intervals.png")
    else:
        print("  No bootstrap results available")

    # Win/Loss matrix - works for any number but most useful with multiple datasets
    if not wilcoxon_df.empty:
        plot_statistical_significance_heatmap(wilcoxon_df, save_path="reports/figures/significance_heatmap.png")
    plot_win_loss_matrix(results_df, metric="roc_auc", save_path="reports/figures/win_loss_matrix.png")

    # Calibration comparison
    if "ece_10" in results_df.columns:
        plot_calibration_comparison(results_df, save_path="reports/figures/calibration_comparison.png")

    # Print system info for reproducibility
    print("\n--- System Information ---")
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    # Timing summary
    print("\n--- Timing Summary (mean across all datasets + seeds) ---")
    timing_summary = (
    results_df
    .groupby("model")[["tuning_time_sec", "fit_time_sec", "predict_time_sec", "total_wall_time_sec"]]
    .mean()
    .round(2)
)

    # Add speedup column only if XGBoost exists in results
    if "XGBoost" in timing_summary.index:
        xgb_time = timing_summary.loc["XGBoost", "total_wall_time_sec"]
        if pd.notna(xgb_time) and xgb_time > 0:
            timing_summary["speedup_vs_xgboost"] = (
                xgb_time / timing_summary["total_wall_time_sec"]
            ).round(1)
        else:
            timing_summary["speedup_vs_xgboost"] = np.nan
    print(timing_summary.to_string())

    # Memory summary
    if "peak_memory_mb" in results_df.columns:
        print("\n--- Memory Summary (mean across all datasets + seeds) ---")
        memory_summary = (
            results_df
            .groupby("model")[["peak_memory_mb", "memory_delta_mb"]]
            .mean()
            .round(1)
        )
        print(memory_summary.to_string())

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()