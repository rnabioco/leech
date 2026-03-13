"""
Handlers for model evaluation commands.

This module contains the business logic for testing models on holdout data.
"""

import logging
from pathlib import Path

from rich.console import Console

logger = logging.getLogger("leech.commands.eval")
console = Console()


def handle_test(
    model: Path,
    test_data: Path,
    output: Path,
    device: str = "cuda",
    batch_size: int = 512,
) -> None:
    """
    Handle the test command logic.

    Args:
        model: Trained model file (.pt)
        test_data: Test dataset config (JSON)
        output: Output metrics file (JSON)
        device: Device for inference
        batch_size: Batch size for evaluation
    """
    from leech.evaluation import evaluate_model

    logger.info(f"Testing model: {model}")
    logger.info(f"Test data: {test_data}")
    logger.info(f"Output: {output}")

    evaluate_model(
        model_path=model,
        test_data_path=test_data,
        output_path=output,
        device=device,
        batch_size=batch_size,
    )

    console.print("[bold green]Testing complete![/bold green]")
    logger.info(f"Results saved to {output}")
