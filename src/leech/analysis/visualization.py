"""
Visualization utilities for analysis results.

Create publication-quality plots for model comparisons, feature importance,
and ablation studies.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger("leech.analysis.visualization")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def plot_model_comparison(
    comparison_results: dict,
    output_path: Path,
    metrics: list[str] | None = None,
) -> None:
    """
    Create bar plot comparing models across metrics.

    Args:
        comparison_results: Dictionary from compare_models()
        output_path: Path to save plot
        metrics: Metrics to plot (default: accuracy, precision, recall, f1, roc_auc)
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    model_names = list(comparison_results.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=False):
        values = [comparison_results[model].get(metric, 0.0) for model in model_names]

        bars = ax.bar(
            range(len(model_names)), values, color=sns.color_palette("husl", len(model_names))
        )

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.upper()} Comparison")
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Comparison plot saved to {output_path}")


def plot_feature_importance(
    importance_scores: dict,
    output_path: Path,
) -> None:
    """
    Plot feature importance scores.

    Args:
        importance_scores: Dictionary from compute_feature_importance()
        output_path: Path to save plot
    """
    num_plots = len(importance_scores)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))

    if num_plots == 1:
        axes = [axes]

    for ax, (feature_name, scores) in zip(axes, importance_scores.items(), strict=False):
        if scores.ndim == 1:
            # 1D scores (e.g., signal importance over time)
            ax.plot(scores, linewidth=1.5)
            ax.set_xlabel("Position")
        elif scores.ndim == 2:
            # 2D scores (e.g., sequence or features over kmer positions)
            im = ax.imshow(scores, aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("K-mer Position")
            ax.set_ylabel("Feature Channel")

        ax.set_title(feature_name.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")


def plot_ablation_results(
    ablation_results: dict,
    output_path: Path,
) -> None:
    """
    Plot sequence ablation test results.

    Args:
        ablation_results: Dictionary from test_sequence_ablation()
        output_path: Path to save plot
    """
    conditions = ["normal", "zeros", "random"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Absolute values
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.25

    for i, condition in enumerate(conditions):
        if condition in ablation_results:
            values = [ablation_results[condition][m] for m in metrics]
            ax1.bar(x + i * width, values, width, label=condition.capitalize())

    ax1.set_ylabel("Score")
    ax1.set_title("Ablation Test: Absolute Performance")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([m.capitalize() for m in metrics])
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Performance drops
    ax2 = axes[1]
    drop_conditions = ["zeros_drop", "random_drop"]
    drop_metrics = [f"{m}_drop" for m in metrics]

    x2 = np.arange(len(metrics))
    for i, drop_cond in enumerate(drop_conditions):
        if drop_cond in ablation_results:
            values = [ablation_results[drop_cond][dm] for dm in drop_metrics]
            label = drop_cond.replace("_drop", "").capitalize()
            ax2.bar(x2 + i * width, values, width, label=label)

    ax2.set_ylabel("Performance Drop")
    ax2.set_title("Ablation Test: Performance Degradation")
    ax2.set_xticks(x2 + width / 2)
    ax2.set_xticklabels([m.capitalize() for m in metrics])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Ablation plot saved to {output_path}")


def plot_roc_curves(
    models_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        models_data: Dict mapping model_name -> (fpr, tpr)
        output_path: Path to save plot
    """
    from sklearn.metrics import auc

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = sns.color_palette("husl", len(models_data))

    for (model_name, (fpr, tpr)), color in zip(models_data.items(), colors, strict=False):
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curves saved to {output_path}")


def plot_confusion_matrices(
    models_cm: dict[str, np.ndarray],
    output_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """
    Plot confusion matrices for multiple models side-by-side.

    Args:
        models_cm: Dict mapping model_name -> confusion_matrix
        output_path: Path to save plot
        class_names: Names for classes (default: ["Uncharged", "Charged"])
    """
    if class_names is None:
        class_names = ["Uncharged", "Charged"]

    num_models = len(models_cm)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))

    if num_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, models_cm.items(), strict=False):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f"{model_name}\nConfusion Matrix")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrices saved to {output_path}")
