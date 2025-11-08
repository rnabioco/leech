"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import argparse
import logging
from pathlib import Path

from leech.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
)
from leech.logging_config import setup_logging

# Setup logging for CLI
logger = logging.getLogger("leech.cli")

# Model choices for CLI
MODEL_CHOICES = [
    "ConvLSTMDwell",
    "ConvLSTMBase",
    "TransformerDwell",
    "ConvOnly",
    "TCNDwell",
    "ResNetDwell",
]


def add_training_args(parser):
    """Add common training arguments to parser."""
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu"], help="Device for training"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    return parser


def add_model_args(parser):
    """Add common model arguments to parser."""
    parser.add_argument(
        "--model",
        type=str,
        default="ConvLSTMDwell",
        choices=MODEL_CHOICES,
        help="Model architecture",
    )
    return parser


def add_prepare_parser(subparsers):
    """Add 'prepare' command for data preparation."""
    parser = subparsers.add_parser("prepare", help="Prepare training data from POD5 and BAM files")
    parser.add_argument("--pod5", required=True, type=Path, help="POD5 file with raw signal")
    parser.add_argument(
        "--bam", required=True, type=Path, help="BAM file with alignments and mv tags"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path, help="Output directory for training chunks"
    )
    parser.add_argument(
        "--motif",
        type=str,
        default=None,
        help='Sequence motif to extract (e.g., "CCA" for tRNA 3\' end)',
    )
    parser.add_argument(
        "--motif-offset", type=int, default=0, help="Offset within motif for focus base"
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Label for all chunks from this file (0=uncharged, 1=charged)",
    )
    parser.add_argument("--min-mapq", type=int, default=10, help="Minimum mapping quality")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="signal+dwell+levels",
        choices=["signal", "signal+dwell", "signal+levels", "signal+dwell+levels"],
        help="Feature set to extract",
    )
    parser.set_defaults(func=run_prepare)


def add_train_parser(subparsers):
    """Add 'train' command for model training."""
    parser = subparsers.add_parser("train", help="Train a model on prepared data")
    parser.add_argument(
        "--train-data", required=True, type=Path, help="Training dataset config (JSON)"
    )
    parser.add_argument(
        "--val-data", type=Path, default=None, help="Validation dataset config (JSON)"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-config", type=Path, default=None, help="Model hyperparameters (JSON)"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path, help="Output directory for model and logs"
    )
    add_training_args(parser)
    parser.set_defaults(func=run_train)


def add_test_parser(subparsers):
    """Add 'test' command for model evaluation."""
    parser = subparsers.add_parser("test", help="Test a trained model")
    parser.add_argument("--model", required=True, type=Path, help="Trained model file (.pt)")
    parser.add_argument("--test-data", required=True, type=Path, help="Test dataset config (JSON)")
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output metrics file (JSON)"
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.set_defaults(func=run_test)


def add_infer_parser(subparsers):
    """Add 'infer' command for inference on new data."""
    parser = subparsers.add_parser("infer", help="Run inference on new data")
    parser.add_argument("--model", required=True, type=Path, help="Trained model file (.pt)")
    parser.add_argument("--pod5", required=True, type=Path, help="POD5 file with raw signal")
    parser.add_argument("--bam", required=True, type=Path, help="BAM file with alignments")
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output BAM with predictions"
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.set_defaults(func=run_infer)


def add_grid_search_parser(subparsers):
    """Add 'grid-search' command for chunk context optimization."""
    parser = subparsers.add_parser(
        "grid-search", help="Run grid search over chunk contexts for model optimization"
    )
    parser.add_argument("--train-data", required=True, type=Path, help="Training dataset (.npz)")
    parser.add_argument("--val-data", type=Path, default=None, help="Validation dataset (.npz)")
    add_model_args(parser)
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path, help="Output directory for grid results"
    )
    parser.add_argument(
        "--context-grid",
        type=str,
        required=True,
        help="Comma-separated context values to test (e.g., '200,500,1000,2000,5000')",
    )
    parser.add_argument(
        "--left-contexts",
        type=str,
        default=None,
        help="Override left contexts (comma-separated). If not provided, uses --context-grid",
    )
    parser.add_argument(
        "--right-contexts",
        type=str,
        default=None,
        help="Override right contexts (comma-separated). If not provided, uses --context-grid",
    )
    parser.add_argument(
        "--kmer-context", type=int, default=5, help="K-mer context for sequence encoding"
    )
    add_training_args(parser)
    parser.set_defaults(func=run_grid_search)


def run_prepare(args):
    """Execute data preparation."""

    from leech.data_prep import extract_training_chunks, iter_bam_with_pod5, save_chunks

    logger.info(f"Preparing data from {args.pod5} and {args.bam}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for read in iter_bam_with_pod5(args.bam, args.pod5, min_mapq=args.min_mapq):
        read_chunks = extract_training_chunks(
            read, motif=args.motif, motif_offset=args.motif_offset, label=args.label
        )
        chunks.extend(read_chunks)

    logger.info(f"Extracted {len(chunks)} training chunks")

    # Save chunks
    output_file = args.output_dir / "chunks.npz"
    save_chunks(chunks, output_file)
    logger.info(f"Saved to {output_file}")


def run_train(args):
    """Execute model training."""
    from leech.training import train_model

    logger.info(f"Training {args.model} model")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Output: {args.output_dir}")

    # Load model config if provided
    model_kwargs = {}
    if args.model_config is not None:
        import json

        with open(args.model_config) as f:
            model_kwargs = json.load(f)

    # Train model
    history = train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        **model_kwargs,
    )

    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {history.get('val_acc', [0])[-1]:.4f}")
    logger.info(f"Models saved to {args.output_dir}")


def run_test(args):
    """Execute model testing."""
    from leech.evaluation import evaluate_model

    logger.info(f"Testing model: {args.model}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output: {args.output}")

    # Run evaluation
    evaluate_model(
        model_path=args.model,
        test_data_path=args.test_data,
        output_path=args.output,
        device=args.device,
    )

    logger.info("Testing complete!")
    logger.info(f"Results saved to {args.output}")


def run_infer(args):
    """Execute inference."""
    from leech.inference import run_inference

    logger.info(f"Running inference with model: {args.model}")
    logger.info(f"Input: {args.pod5}, {args.bam}")
    logger.info(f"Output: {args.output}")

    # Run inference
    run_inference(
        model_path=args.model,
        pod5_path=args.pod5,
        bam_path=args.bam,
        output_path=args.output,
        device=args.device,
    )

    logger.info("Inference complete!")
    logger.info(f"Predictions saved to {args.output}")


def run_grid_search(args):
    """Execute grid search over chunk contexts."""
    from leech.gridsearch import GridSearchConfig, run_grid_search

    # Parse context grids
    if args.left_contexts is not None:
        left_contexts = [int(x.strip()) for x in args.left_contexts.split(",")]
    else:
        left_contexts = [int(x.strip()) for x in args.context_grid.split(",")]

    if args.right_contexts is not None:
        right_contexts = [int(x.strip()) for x in args.right_contexts.split(",")]
    else:
        right_contexts = [int(x.strip()) for x in args.context_grid.split(",")]

    logger.info(f"Starting grid search with {len(left_contexts)} x {len(right_contexts)} grid points")
    logger.info(f"Left contexts: {left_contexts}")
    logger.info(f"Right contexts: {right_contexts}")

    # Create config
    config = GridSearchConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_name=args.model,
        output_dir=args.output_dir,
        left_contexts=left_contexts,
        right_contexts=right_contexts,
        kmer_context=args.kmer_context,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )

    # Run grid search
    summary_path = run_grid_search(config)

    logger.info("=" * 80)
    logger.info("Grid search complete!")
    logger.info(f"Results saved to: {summary_path}")
    logger.info("=" * 80)


def main():
    """Main CLI entry point."""
    # Setup logging
    setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="leech",
        description="Enhanced classification from nanopore signals",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(
        title="commands", description="Available commands", dest="command", required=True
    )

    # Add subcommands
    add_prepare_parser(subparsers)
    add_train_parser(subparsers)
    add_test_parser(subparsers)
    add_infer_parser(subparsers)
    add_grid_search_parser(subparsers)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
