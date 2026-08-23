"""
Handlers for model evaluation commands.

This module contains the business logic for testing models on holdout data.
"""

import logging
from pathlib import Path

from leech.cli_config import make_console

logger = logging.getLogger("leech.commands.eval")
console = make_console()


def handle_test(
    model: Path,
    test_data: Path,
    output: Path,
    device: str = "cuda",
    batch_size: int = 512,
    emit_scores: Path | None = None,
) -> None:
    """
    Handle the test command logic.

    Args:
        model: Trained model file (.pt)
        test_data: Test dataset config (JSON)
        output: Output metrics file (JSON)
        device: Device for inference
        batch_size: Batch size for evaluation
        emit_scores: Optional .npz for per-chunk scores
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
        emit_scores=emit_scores,
    )

    console.print("[bold green]Testing complete![/bold green]")
    logger.info(f"Results saved to {output}")
    if emit_scores is not None:
        logger.info(f"Per-chunk scores saved to {emit_scores}")
