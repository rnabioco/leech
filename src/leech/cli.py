"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import argparse
from pathlib import Path


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
    parser.add_argument(
        "--model",
        type=str,
        default="ConvLSTMDwell",
        choices=["ConvLSTMDwell", "ConvLSTMBase"],
        help="Model architecture",
    )
    parser.add_argument(
        "--model-config", type=Path, default=None, help="Model hyperparameters (JSON)"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path, help="Output directory for model and logs"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
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
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.set_defaults(func=run_infer)


def run_prepare(args):
    """Execute data preparation."""

    from leech.data_prep import extract_training_chunks, iter_bam_with_pod5

    print(f"Preparing data from {args.pod5} and {args.bam}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for read in iter_bam_with_pod5(args.bam, args.pod5, min_mapq=args.min_mapq):
        read_chunks = extract_training_chunks(
            read, motif=args.motif, motif_offset=args.motif_offset, label=args.label
        )
        chunks.extend(read_chunks)

    print(f"Extracted {len(chunks)} training chunks")

    # Save chunks (implement serialization)
    output_file = args.output_dir / "chunks.npz"
    # TODO: Implement chunk serialization
    print(f"Saved to {output_file}")


def run_train(args):
    """Execute model training."""
    print(f"Training {args.model} model")
    print(f"Train data: {args.train_data}")
    print(f"Output: {args.output_dir}")

    # TODO: Implement training loop
    print("Training not yet implemented - coming next!")


def run_test(args):
    """Execute model testing."""
    print(f"Testing model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output}")

    # TODO: Implement testing
    print("Testing not yet implemented - coming next!")


def run_infer(args):
    """Execute inference."""
    print(f"Running inference with model: {args.model}")
    print(f"Input: {args.pod5}, {args.bam}")
    print(f"Output: {args.output}")

    # TODO: Implement inference
    print("Inference not yet implemented - coming next!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="leech",
        description="Learning Enhanced Aminoacylation Classification from Hanopore signals",
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

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
