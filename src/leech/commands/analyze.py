"""
Handlers for 'leech analyze' command group.

This module contains the business logic for analysis commands.
CLI commands in cli.py should delegate to these functions.
"""

import logging
from pathlib import Path

logger = logging.getLogger("leech.commands.analyze")


def handle_compare(
    model_dirs: list[Path],
    test_data: Path,
    output_dir: Path,
    device: str = "cuda",
    plot: bool = True,
) -> dict:
    """
    Handle model comparison command.

    Args:
        model_dirs: List of model directories to compare
        test_data: Path to test dataset
        output_dir: Output directory for results
        device: Device for evaluation
        plot: Whether to generate plots

    Returns:
        Comparison results dictionary
    """
    from leech.analysis.comparison import compare_models, compare_training_curves
    from leech.analysis.visualization import plot_model_comparison

    logger.info(f"Comparing {len(model_dirs)} models on {test_data}")

    # Run comparison
    results = compare_models(
        model_dirs=model_dirs,
        test_data_path=test_data,
        output_dir=output_dir,
        device=device,
    )

    if plot:
        # Generate comparison plot
        plot_path = output_dir / "comparison_plot.png"
        plot_model_comparison(results, plot_path)

        # Generate training curves comparison
        curves_path = output_dir / "training_curves.png"
        compare_training_curves(model_dirs, curves_path)

    logger.info(f"Comparison complete. Results saved to {output_dir}")

    return results


def handle_feature_importance(
    model_path: Path,
    test_data: Path,
    output_dir: Path,
    device: str = "cuda",
    method: str = "gradient",
    plot: bool = True,
) -> dict:
    """
    Handle feature importance analysis command.

    Args:
        model_path: Path to trained model
        test_data: Path to test dataset
        output_dir: Output directory for results
        device: Device for computation
        method: Importance method ("gradient" or "integrated_gradients")
        plot: Whether to generate plots

    Returns:
        Feature importance scores
    """
    from leech.analysis.feature_importance import compute_feature_importance
    from leech.analysis.visualization import plot_feature_importance

    logger.info(f"Computing feature importance for {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute importance
    importance_path = output_dir / "feature_importance.npz"
    importance = compute_feature_importance(
        model_path=model_path,
        test_data_path=test_data,
        output_path=importance_path,
        device=device,
        method=method,
    )

    if plot:
        plot_path = output_dir / "feature_importance_plot.png"
        plot_feature_importance(importance, plot_path)

    logger.info(f"Feature importance analysis complete. Results saved to {output_dir}")

    return importance


def handle_sequence_ablation(
    model_path: Path,
    test_data: Path,
    output_dir: Path,
    device: str = "cuda",
    plot: bool = True,
) -> dict:
    """
    Handle sequence ablation test command.

    Args:
        model_path: Path to trained model
        test_data: Path to test dataset
        output_dir: Output directory for results
        device: Device for computation
        plot: Whether to generate plots

    Returns:
        Ablation test results
    """
    from leech.analysis.sequence_ablation import test_sequence_ablation
    from leech.analysis.visualization import plot_ablation_results

    logger.info(f"Running sequence ablation test for {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ablation test
    ablation_path = output_dir / "sequence_ablation.json"
    results = test_sequence_ablation(
        model_path=model_path,
        test_data_path=test_data,
        output_path=ablation_path,
        device=device,
    )

    if plot and "error" not in results:
        plot_path = output_dir / "sequence_ablation_plot.png"
        plot_ablation_results(results, plot_path)

    logger.info(f"Sequence ablation test complete. Results saved to {output_dir}")

    return results


def handle_branch_contribution(
    model_dirs: dict[str, Path],
    test_data: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """
    Handle branch contribution analysis.

    Compares models with different branch combinations to quantify
    the contribution of each branch (signal, sequence, features).

    Args:
        model_dirs: Dict mapping model_type -> model_directory
                   Expected keys: "base", "signal_seq", "signal_feat", "full"
        test_data: Path to test dataset
        output_dir: Output directory for results
        device: Device for evaluation

    Returns:
        Branch contribution analysis results
    """
    from leech.analysis.comparison import compare_models

    logger.info("Analyzing branch contributions...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    model_list = list(model_dirs.values())
    results = compare_models(
        model_dirs=model_list,
        test_data_path=test_data,
        output_dir=output_dir,
        device=device,
    )

    # Compute branch contributions
    contributions = {}

    # Feature branch contribution
    if "base" in results and "full" in results:
        contributions["features"] = {
            metric: results["full"][metric] - results["base"][metric]
            for metric in ["accuracy", "f1", "roc_auc"]
            if metric in results["full"]
        }

    # Sequence branch contribution (for signal-features models)
    if "signal_feat" in results and "full" in results:
        contributions["sequence"] = {
            metric: results["full"][metric] - results["signal_feat"][metric]
            for metric in ["accuracy", "f1", "roc_auc"]
            if metric in results["full"]
        }

    # Save contribution analysis
    import json

    contribution_path = output_dir / "branch_contributions.json"
    with open(contribution_path, "w") as f:
        json.dump(contributions, f, indent=2)

    logger.info(f"Branch contribution analysis complete. Results saved to {output_dir}")

    return contributions
