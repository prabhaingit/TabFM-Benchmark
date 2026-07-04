"""
Generate the main leaderboard visualisation:
- Heatmap of model × dataset AUC scores
- Summary bar chart of average ranks
- Win/loss/tie matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# Dark-mode style matching the project's visual identity
DARK_BG    = "#0d0f14"
SURFACE    = "#13161e"
ACCENT     = "#5b8dee"
ACCENT2    = "#a78bfa"
ACCENT3    = "#34d399"
TEXT_COLOR = "#e2e8f0"
MUTED      = "#8892a4"

MODEL_COLORS = {
    "TabPFN":   ACCENT,
    "XGBoost":  ACCENT3,
    "LightGBM": ACCENT2,
    "CatBoost": "#f59e0b",
    "MLP":      "#f87171",
}


def plot_auc_heatmap(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/auc_heatmap.png",
    figsize: tuple = (14, 8),
):
    """
    Heatmap: rows=datasets, columns=models, cells=mean AUC.
    TabPFN column is highlighted to draw the eye.
    """
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    
    # Sort datasets by TabPFN performance (descending) so the story is clear
    if "TabPFN" in pivot.columns:
        pivot = pivot.sort_values("TabPFN", ascending=False)

    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    cmap = sns.color_palette("mako", as_cmap=True)
    
    # Normalise each row so relative performance is visible
    pivot_norm = pivot.sub(pivot.min(axis=1), axis=0)
    pivot_norm = pivot_norm.div(pivot_norm.max(axis=1) + 1e-9, axis=0)

    im = ax.imshow(
        pivot_norm.values,
        cmap=cmap,
        aspect="auto",
        vmin=0, vmax=1,
    )

    # Annotate cells with actual AUC values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if pivot_norm.iloc[i, j] > 0.5 else TEXT_COLOR
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="600")

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color=TEXT_COLOR, fontsize=10, fontweight="600")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=MUTED, fontsize=9)

    # Highlight TabPFN column with a border
    if "TabPFN" in pivot.columns:
        tabpfn_col = list(pivot.columns).index("TabPFN")
        ax.axvline(tabpfn_col - 0.5, color=ACCENT, lw=2, alpha=0.8)
        ax.axvline(tabpfn_col + 0.5, color=ACCENT, lw=2, alpha=0.8)

    ax.set_title(
        f"Model × Dataset AUC Heatmap\n(cells normalised row-wise; absolute AUC shown)",
        color=TEXT_COLOR, fontsize=12, pad=16, fontweight="600",
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved heatmap → {save_path}")


def plot_average_ranks(
    rank_df: pd.DataFrame,
    save_path: str = "reports/figures/average_ranks.png",
):
    """
    Horizontal bar chart of average ranks. Lower = better.
    """
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    models = rank_df["model"].tolist()
    ranks  = rank_df["average_rank"].tolist()
    colors = [MODEL_COLORS.get(m, MUTED) for m in models]

    bars = ax.barh(models, ranks, color=colors, height=0.55, alpha=0.9)
    
    for bar, rank in zip(bars, ranks):
        ax.text(rank + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{rank:.2f}", va="center", color=TEXT_COLOR, fontsize=10)

    ax.set_xlabel("Average Rank (lower = better)", color=MUTED, fontsize=10)
    ax.set_title("Average Rank Across All Datasets", color=TEXT_COLOR,
                 fontsize=12, fontweight="600")
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(MUTED)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved rank chart → {save_path}")

def plot_timing_comparison(
    results_df: pd.DataFrame,
    save_path: str = "reports/figures/timing_comparison.png",
):
    """
    Stacked bar chart: tuning | fit | predict time per model.
    This is the visual that makes TabPFN's advantage undeniable.
    """
    timing = (
        results_df
        .groupby("model")[["tuning_time_sec", "fit_time_sec", "predict_time_sec"]]
        .mean()
        .loc[["TabPFN", "XGBoost", "LightGBM", "CatBoost", "MLP"]]  # fixed order
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # Left: stacked bar — absolute wall time
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    bottom = np.zeros(len(timing))
    colors_map = {"tuning_time_sec": "#374151", "fit_time_sec": ACCENT2, "predict_time_sec": ACCENT}
    labels_map = {"tuning_time_sec": "Hyperparameter tuning", "fit_time_sec": "Model fit", "predict_time_sec": "Inference"}

    for col in ["tuning_time_sec", "fit_time_sec", "predict_time_sec"]:
        vals = timing[col].values
        ax.bar(timing.index, vals, bottom=bottom,
               label=labels_map[col], color=colors_map[col], alpha=0.9)
        bottom += vals

    ax.set_title("Total Wall-Clock Time (mean across datasets + seeds)",
                 color=TEXT_COLOR, fontsize=11, fontweight="600")
    ax.set_ylabel("Seconds", color=MUTED)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(MUTED)
    ax.legend(facecolor=SURFACE, labelcolor=TEXT_COLOR, fontsize=9)

    # Right: throughput (rows/sec at inference)
    ax2 = axes[1]
    ax2.set_facecolor(DARK_BG)
    throughput = results_df.groupby("model")["throughput_rows_per_sec"].mean()
    colors_t = [MODEL_COLORS.get(m, MUTED) for m in throughput.index]
    ax2.bar(throughput.index, throughput.values, color=colors_t, alpha=0.9)
    ax2.set_title("Inference Throughput (rows / second)",
                  color=TEXT_COLOR, fontsize=11, fontweight="600")
    ax2.set_ylabel("Rows/sec", color=MUTED)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.spines[:].set_color(MUTED)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved timing chart → {save_path}")


def plot_memory_comparison(
    results_df: pd.DataFrame,
    save_path: str = "reports/figures/memory_comparison.png",
):
    """
    Bar chart comparing memory usage across models.
    Shows peak memory and memory delta.
    """
    if "peak_memory_mb" not in results_df.columns:
        print("  No memory data available, skipping memory visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # Left: peak memory usage
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    memory = results_df.groupby("model")["peak_memory_mb"].mean().sort_values(ascending=True)
    colors = [MODEL_COLORS.get(m, MUTED) for m in memory.index]
    bars = ax.barh(memory.index, memory.values, color=colors, alpha=0.85)

    for bar, val in zip(bars, memory.values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} MB", va="center", color=TEXT_COLOR, fontsize=10)

    ax.set_xlabel("Peak Memory Usage (MB)", color=MUTED, fontsize=11)
    ax.set_title("Peak Memory Usage", color=TEXT_COLOR, fontsize=12, fontweight="600")
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(MUTED)

    # Right: memory delta (memory increase during execution)
    ax2 = axes[1]
    ax2.set_facecolor(DARK_BG)
    if "memory_delta_mb" in results_df.columns:
        memory_delta = results_df.groupby("model")["memory_delta_mb"].mean().sort_values(ascending=True)
        colors_d = [MODEL_COLORS.get(m, MUTED) for m in memory_delta.index]
        bars2 = ax2.barh(memory_delta.index, memory_delta.values, color=colors_d, alpha=0.85)

        for bar, val in zip(bars2, memory_delta.values):
            ax2.text(val + 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f} MB", va="center", color=TEXT_COLOR, fontsize=10)

        ax2.set_xlabel("Memory Delta (MB)", color=MUTED, fontsize=11)
        ax2.set_title("Memory Increase During Execution", color=TEXT_COLOR, fontsize=12, fontweight="600")
    else:
        ax2.set_facecolor(DARK_BG)
        ax2.text(0.5, 0.5, "Memory delta not tracked", transform=ax2.transAxes,
                 ha="center", va="center", color=MUTED, fontsize=12)

    ax2.tick_params(colors=MUTED)
    ax2.spines[:].set_color(MUTED)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved memory chart → {save_path}")