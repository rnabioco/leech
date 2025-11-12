#!/usr/bin/env python3
"""
Generate a figure illustrating the relationship between nanopore signal,
move tables, and dwell times.

This script creates a publication-quality figure similar to squigualiser's
move_table_annotation.png, showing how move tables map signal to bases
and enable dwell time calculation.

Usage:
    python scripts/generate_move_table_figure.py [--output docs/figures/move_table_diagram.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def generate_synthetic_signal(bases: str, mean_dwell: int = 10, noise_level: float = 5.0):
    """Generate synthetic nanopore signal for a sequence.

    Args:
        bases: DNA/RNA sequence
        mean_dwell: Average samples per base
        noise_level: Standard deviation of noise

    Returns:
        signal: Synthetic signal array
        true_dwells: Actual dwell times per base
    """
    # Base-specific signal levels (arbitrary picoamps)
    base_levels = {"A": 85, "C": 75, "G": 65, "T": 95, "U": 95}

    signal = []
    true_dwells = []

    for base in bases:
        # Random dwell time around mean
        dwell = int(np.random.normal(mean_dwell, 2))
        dwell = max(3, dwell)  # At least 3 samples
        true_dwells.append(dwell)

        # Generate signal for this base
        level = base_levels.get(base, 80)
        base_signal = np.random.normal(level, noise_level, dwell)
        signal.extend(base_signal)

    return np.array(signal), true_dwells


def create_move_table(dwells: list[int], stride: int = 5):
    """Create a move table from dwell times.

    Args:
        dwells: List of dwell times (samples per base)
        stride: Basecaller stride (downsampling factor)

    Returns:
        moves: Binary move array (1 = new base, 0 = same base)
    """
    moves = []
    for dwell in dwells:
        # First position is a move
        moves.append(1)
        # Remaining positions are non-moves
        num_strides = (dwell // stride) - 1
        moves.extend([0] * num_strides)

    return moves


def create_figure(output_path: Path):
    """Create the move table illustration figure."""
    # Generate example data
    sequence = "ATCGATCG"
    stride = 5
    signal, true_dwells = generate_synthetic_signal(sequence, mean_dwell=12)
    moves = create_move_table(true_dwells, stride=stride)

    # Calculate positions
    signal_positions = np.arange(len(signal))
    stride_positions = np.arange(0, len(signal), stride)

    # Reconstruct dwell times from move table (what leech does)
    base_positions = []
    current_pos = 0
    for i, move in enumerate(moves):
        if move == 1:
            base_positions.append(i * stride)

    base_positions.append(len(signal))  # Add end position
    reconstructed_dwells = np.diff(base_positions)

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # === Panel A: Raw Signal ===
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(signal_positions, signal, color="#2E86AB", linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("Signal (pA)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "A. Raw Nanopore Signal (from POD5 file)", fontsize=12, fontweight="bold", loc="left"
    )
    ax1.set_xlim(0, len(signal))
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xticklabels([])

    # Annotate signal regions for first few bases
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    cumsum_pos = 0
    for i, (base, dwell) in enumerate(zip(sequence[:4], true_dwells[:4])):
        ax1.axvspan(cumsum_pos, cumsum_pos + dwell, alpha=0.15, color=colors[i % len(colors)])
        ax1.text(
            cumsum_pos + dwell / 2,
            ax1.get_ylim()[1] - 5,
            f"{base}\n{dwell} samples",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)], alpha=0.3),
        )
        cumsum_pos += dwell

    # === Panel B: Stride Positions ===
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.eventplot(stride_positions, colors="#E63946", linewidths=1.5)
    ax2.set_ylabel("Stride\nPositions", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"B. Basecaller Stride Positions (stride={stride})",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlim(0, len(signal))
    ax2.grid(True, alpha=0.3, linestyle="--", axis="x")
    ax2.set_xticklabels([])

    # Annotate some stride positions
    for i, pos in enumerate(stride_positions[:8]):
        ax2.text(pos, 0.3, str(i), ha="center", fontsize=8, color="#E63946")

    # === Panel C: Move Table ===
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)

    # Draw move table as colored boxes
    box_height = 0.6
    for i, (pos, move) in enumerate(zip(stride_positions[: len(moves)], moves)):
        color = "#06D6A0" if move == 1 else "#CCCCCC"
        rect = patches.Rectangle(
            (pos - stride / 2, -box_height / 2),
            stride,
            box_height,
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            alpha=0.7,
        )
        ax3.add_patch(rect)

        # Add move value text
        if i < 20:  # Only annotate first 20
            ax3.text(
                pos, 0, str(move), ha="center", va="center", fontsize=8, fontweight="bold"
            )

    ax3.set_ylabel("Move Table\n(mv tag)", fontsize=11, fontweight="bold")
    ax3.set_title(
        "C. Move Table from BAM (1=new base, 0=same base)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_yticks([])
    ax3.set_xlim(0, len(signal))
    ax3.grid(True, alpha=0.3, linestyle="--", axis="x")
    ax3.set_xticklabels([])

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="#06D6A0", edgecolor="black", label="Move (1)"),
        patches.Patch(facecolor="#CCCCCC", edgecolor="black", label="Stay (0)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # === Panel D: Basecalled Sequence + Dwell Times ===
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)

    # Draw bases with dwell times
    cumsum_pos = 0
    for i, (base, dwell) in enumerate(zip(sequence, reconstructed_dwells)):
        color = colors[i % len(colors)]

        # Draw rectangle for base
        rect = patches.Rectangle(
            (cumsum_pos, -0.3), dwell, 0.6, linewidth=1.5, edgecolor="black", facecolor=color
        )
        ax4.add_patch(rect)

        # Add base label
        ax4.text(
            cumsum_pos + dwell / 2,
            0.5,
            base,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        # Add dwell time below
        ax4.text(
            cumsum_pos + dwell / 2,
            -0.6,
            f"{dwell}",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color="#555555",
        )

        cumsum_pos += dwell

    ax4.set_ylabel("Bases +\nDwell Times", fontsize=11, fontweight="bold")
    ax4.set_title(
        "D. Basecalled Sequence with Dwell Times (samples per base)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax4.set_ylim(-1, 1)
    ax4.set_yticks([])
    ax4.set_xlim(0, len(signal))
    ax4.set_xlabel("Signal Sample Index", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle="--", axis="x")

    # Add overall title
    fig.suptitle(
        "Move Table Decoding: From Nanopore Signal to Dwell Times",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # Add explanation box
    explanation = (
        "Leech extracts dwell times by:\n"
        "1. Reading raw signal from POD5\n"
        "2. Parsing move table (mv tag) from BAM\n"
        f"3. Computing signal-to-base mapping (stride={stride})\n"
        "4. Calculating dwell = samples per base\n"
        "→ Dwell differences reveal modifications!"
    )

    fig.text(
        0.98,
        0.02,
        explanation,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF9E6", edgecolor="#E6B800", linewidth=2),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate move table illustration figure")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/move_table_diagram.png"),
        help="Output path for figure (default: docs/figures/move_table_diagram.png)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate figure
    output_path = create_figure(args.output)

    print(f"\nTo use in documentation, add to your markdown:")
    print(f'![Move Table Diagram](../figures/{output_path.name})')


if __name__ == "__main__":
    main()
