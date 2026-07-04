"""
Statistical visualization for benchmark results:
- Critical Difference (CD) diagram
- Confidence interval plots
- Statistical significance heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Dark-mode style matching the project's visual identity
DARK_BG = "#0d0f14"
SURFACE = "#13161e"
ACCENT = "#5b8dee"
ACCENT2 = "#a78bfa"
ACCENT3 = "#34d399"
TEXT_COLOR = "#e2e8f0"
MUTED = "#8892a4"

MODEL_COLORS = {
    "TabPFN": ACCENT,
    "XGBoost": ACCENT3,
    "LightGBM": ACCENT2,
    "CatBoost": "#f59e0b",
    "MLP": "#f87171",
}


def plot_critical_difference_diagram(
    rank_df: pd.DataFrame,
    cd: float,
    save_path: str = "reports/figures/cd_diagram.png",
    alpha: float = 0.05,
):
    """
    Draw a Critical Difference (CD) diagram.

    The CD diagram shows:
    - Average ranks of models on the axis
    - A bar showing the critical difference
    - Lines connecting models whose rank difference exceeds CD (not significantly different)

    Reference: Demsar, J. (2006). Statistical comparisons of classifiers
    """
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    models = rank_df["model"].tolist()
    avg_ranks = rank_df["average_rank"].tolist()
    n_models = len(models)

    # Plot rank axis
    ax.set_xlim(1, n_models + 0.5)
    ax.set_ylim(0, 2)

    # Draw axis line
    ax.axhline(y=1, color=MUTED, linewidth=1.5)

    # Add tick marks for ranks
    for r in range(1, n_models + 1):
        ax.plot([r, r], [0.95, 1.05], color=MUTED, linewidth=1)
        ax.text(r, 0.85, str(r), ha="center", va="top", color=MUTED, fontsize=10)

    ax.text(n_models / 2 + 0.5, 0.75, "Average Rank", ha="center", color=MUTED, fontsize=11)

    # Plot models at their average ranks
    y_base = 1.2
    for i, (model, rank) in enumerate(zip(models, avg_ranks)):
        color = MODEL_COLORS.get(model, MUTED)

        # Draw model marker
        ax.scatter(rank, y_base, s=200, color=color, zorder=5, edgecolor=SURFACE, linewidth=2)

        # Add model name
        angle = 0 if rank < n_models / 2 else 0
        ax.text(rank, y_base + 0.15, model, ha="center", va="bottom",
                color=color, fontsize=10, fontweight="600", rotation=0)

    # Draw CD bar
    cd_y = 0.4
    cd_start = 1
    cd_end = 1 + cd

    ax.plot([cd_start, cd_end], [cd_y, cd_y], color=ACCENT, linewidth=3)
    ax.plot([cd_start, cd_end], [cd_y - 0.02, cd_y + 0.02], color=ACCENT, linewidth=1)
    ax.plot([cd_start, cd_end], [cd_y - 0.02, cd_y + 0.02], color=ACCENT, linewidth=1)

    # Vertical tick marks at CD endpoints
    ax.plot([cd_start, cd_start], [cd_y - 0.05, cd_y + 0.05], color=ACCENT, linewidth=2)
    ax.plot([cd_end, cd_end], [cd_y - 0.05, cd_y + 0.05], color=ACCENT, linewidth=2)

    # Add CD label
    ax.text((cd_start + cd_end) / 2, cd_y - 0.1, f"CD = {cd:.2f}",
            ha="center", va="top", color=ACCENT, fontsize=11, fontweight="600")

    # Add significance note
    ax.text(0.98, 0.05, f"α = {alpha}\nModels connected by line\nare not significantly different",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=9, style="italic")

    ax.set_title("Critical Difference Diagram", color=TEXT_COLOR, fontsize=13, fontweight="600", pad=20)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved CD diagram → {save_path}")


def plot_confidence_intervals(
    ci_df: pd.DataFrame,
    metric: str = "ROC-AUC",
    save_path: str = "reports/figures/confidence_intervals.png",
):
    """
    Plot confidence intervals for each model.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    models = ci_df["model"].tolist()
    means = ci_df["mean"].tolist()
    ci_lower = ci_df["ci_lower"].tolist()
    ci_upper = ci_df["ci_upper"].tolist()

    y_pos = np.arange(len(models))

    for i, (model, mean, low, high) in enumerate(zip(models, means, ci_lower, ci_upper)):
        color = MODEL_COLORS.get(model, MUTED)

        # Error bar
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]],
                   fmt="o", color=color, capsize=5, capthick=2, markersize=10, linewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel(f"{metric} (95% CI)", color=MUTED, fontsize=11)
    ax.set_title("Model Performance with 95% Confidence Intervals", color=TEXT_COLOR, fontsize=13, fontweight="600")
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(MUTED)
    ax.invert_yaxis()

    # Add grid
    ax.xaxis.grid(True, alpha=0.3, color=MUTED)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved CI plot → {save_path}")


def plot_statistical_significance_heatmap(
    wilcoxon_df: pd.DataFrame,
    save_path: str = "reports/figures/significance_heatmap.png",
    metric: str = "ROC-AUC",
):
    """
    Create a heatmap showing pairwise statistical significance.
    """
    # Get unique models
    all_models = set(wilcoxon_df["model_a"].tolist() + wilcoxon_df["model_b"].tolist())
    models = sorted(all_models)

    # Create significance matrix
    sig_matrix = pd.DataFrame(index=models, columns=models, data=1.0)  # 1 = not significant

    for _, row in wilcoxon_df.iterrows():
        if row["significant"]:
            sig_matrix.loc[row["model_a"], row["model_b"]] = 0
            sig_matrix.loc[row["model_b"], row["model_a"]] = 0

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    cmap = plt.cm.colors.ListedColormap([ACCENT3, MUTED])
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(sig_matrix.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, color=TEXT_COLOR, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color=TEXT_COLOR, fontsize=10)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                text = "-"
            else:
                text = "✓" if sig_matrix.iloc[i, j] == 0 else "✗"
            ax.text(j, i, text, ha="center", va="center",
                    color=ACCENT3 if sig_matrix.iloc[i, j] == 0 else MUTED,
                    fontsize=12)

    ax.set_title(f"Pairwise Significance ({metric})\n✓ = significantly different, ✗ = not significant",
                 color=TEXT_COLOR, fontsize=12, fontweight="600", pad=16)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=ACCENT3, label="Significant (p < 0.05)"),
        mpatches.Patch(facecolor=MUTED, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor=SURFACE, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved significance heatmap → {save_path}")


def plot_win_loss_matrix(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/win_loss_matrix.png",
):
    """
    Create a win/loss/tie matrix showing how often each model beats others.
    """
    # Pivot: rows=datasets, cols=models, values=mean metric
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )

    models = pivot.columns.tolist()
    n_models = len(models)

    # Initialize matrices
    wins = pd.DataFrame(0, index=models, columns=models, dtype=int)
    losses = pd.DataFrame(0, index=models, columns=models, dtype=int)
    ties = pd.DataFrame(0, index=models, columns=models, dtype=int)

    for _, row in pivot.iterrows():
        for m1 in models:
            for m2 in models:
                if m1 != m2:
                    v1, v2 = row[m1], row[m2]
                    if pd.isna(v1) or pd.isna(v2):
                        continue
                    if v1 > v2:
                        wins.loc[m1, m2] += 1
                    elif v1 < v2:
                        losses.loc[m1, m2] += 1
                    else:
                        ties.loc[m1, m2] += 1

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK_BG)

    for ax, (matrix, title, cmap) in zip(axes, [
        (wins, "Wins", "Greens"),
        (losses, "Losses", "Reds"),
        (ties, "Ties", "Blues"),
    ]):
        ax.set_facecolor(DARK_BG)
        sns = __import__("seaborn").heatmap(
            matrix, annot=True, fmt="d", cmap=cmap, ax=ax,
            cbar=False, linewidths=0.5, linecolor=MUTED,
            xticklabels=models, yticklabels=models,
        )
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="600")
        ax.tick_params(colors=MUTED)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(f"Win/Loss/Tie Matrix ({metric})", color=TEXT_COLOR, fontsize=14, fontweight="600")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved win/loss matrix → {save_path}")