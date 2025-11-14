"""
Model comparison utilities.

Compare multiple trained models on the same test set and generate
comparison reports with metrics, visualizations, and statistical tests.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("leech.analysis.comparison")


def compare_models(
    model_dirs: list[Path],
    test_data_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """
    Compare multiple trained models on the same test set.

    Args:
        model_dirs: List of directories containing trained models (model_best.pt)
        test_data_path: Path to test dataset
        output_dir: Output directory for comparison results
        device: Device for inference

    Returns:
        Comparison results dictionary
    """
    from leech.evaluation import evaluate_model

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_path = model_dir / "model_best.pt"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        logger.info(f"Evaluating {model_name}...")

        # Run evaluation
        metrics_path = output_dir / f"{model_name}_metrics.json"
        metrics = evaluate_model(
            model_path=model_path,
            test_data_path=test_data_path,
            output_path=metrics_path,
            device=device,
        )

        results[model_name] = metrics

    # Save comparison results
    comparison_path = output_dir / "comparison_results.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Comparison results saved to {comparison_path}")

    # Generate comparison summary
    summary = _generate_comparison_summary(results)
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    logger.info(f"Comparison summary saved to {summary_path}")

    return results


def load_comparison_results(comparison_path: Path) -> dict:
    """
    Load comparison results from JSON file.

    Args:
        comparison_path: Path to comparison_results.json

    Returns:
        Comparison results dictionary
    """
    with open(comparison_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _generate_comparison_summary(results: dict) -> str:
    """
    Generate text summary of comparison results.

    Args:
        results: Comparison results dictionary

    Returns:
        Formatted summary string
    """
    lines = ["Model Comparison Summary", "=" * 80, ""]

    # Extract metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())

    # Create table header
    header = f"{'Model':<30} " + " ".join(f"{m:>10}" for m in metrics)
    lines.append(header)
    lines.append("-" * 80)

    # Add rows for each model
    for model_name in model_names:
        model_metrics = results[model_name]
        row = f"{model_name:<30} "
        row += " ".join(f"{model_metrics.get(m, 0.0):>10.4f}" for m in metrics)
        lines.append(row)

    lines.append("")

    # Rank models by metrics
    lines.append("Rankings:")
    lines.append("-" * 80)

    for metric in metrics:
        ranked = sorted(
            model_names,
            key=lambda m: results[m].get(metric, 0.0),
            reverse=True,
        )
        lines.append(f"{metric.upper():>15}: {' > '.join(ranked)}")

    lines.append("")

    # Statistical significance (if multiple models)
    if len(model_names) > 1:
        lines.append("Statistical Tests:")
        lines.append("-" * 80)
        lines.append("(Implement pairwise t-tests or McNemar's test here)")
        lines.append("")

    return "\n".join(lines)


def compare_training_curves(
    model_dirs: list[Path],
    output_path: Path,
) -> None:
    """
    Compare training curves across models.

    Args:
        model_dirs: List of model directories with metrics.json
        output_path: Path to save comparison plot
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_dir in model_dirs:
        history_path = model_dir / "metrics.json"
        if not history_path.exists():
            continue

        with open(history_path) as f:
            history = json.load(f)

        model_name = model_dir.name
        epochs = range(1, len(history["train_loss"]) + 1)

        # Plot loss
        ax1.plot(epochs, history["train_loss"], label=f"{model_name} (train)", alpha=0.7)
        if "val_loss" in history:
            ax1.plot(
                epochs, history["val_loss"], label=f"{model_name} (val)", alpha=0.7, linestyle="--"
            )

        # Plot accuracy
        ax2.plot(epochs, history["train_acc"], label=f"{model_name} (train)", alpha=0.7)
        if "val_acc" in history:
            ax2.plot(
                epochs, history["val_acc"], label=f"{model_name} (val)", alpha=0.7, linestyle="--"
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Training curves saved to {output_path}")
