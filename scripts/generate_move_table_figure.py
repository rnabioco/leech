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

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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


def create_move_table(dwells: list[int], stride: int = 5, signal_len: int = None):
    """Create a move table from dwell times.

    Args:
        dwells: List of dwell times (samples per base)
        stride: Basecaller stride (downsampling factor)
        signal_len: Total signal length (to ensure moves span full axis)

    Returns:
        moves: Binary move array (1 = new base, 0 = same base)
    """
    # Calculate cumulative positions for each base
    base_starts = [0]
    for dwell in dwells:
        base_starts.append(base_starts[-1] + dwell)

    # Determine total number of stride positions
    if signal_len is None:
        signal_len = sum(dwells)
    num_stride_positions = (signal_len + stride - 1) // stride

    # For each stride position, find which base it belongs to
    def get_base_idx(stride_pos):
        for idx, start in enumerate(base_starts[:-1]):
            if start <= stride_pos < base_starts[idx + 1]:
                return idx
        return len(base_starts) - 2  # Last base

    # Create moves: 1 if first stride position of a base, 0 otherwise
    moves = []
    prev_base_idx = -1
    for stride_idx in range(num_stride_positions):
        stride_pos = stride_idx * stride
        curr_base_idx = get_base_idx(stride_pos)

        # Move if we're on a new base
        is_move = curr_base_idx != prev_base_idx
        moves.append(1 if is_move else 0)
        prev_base_idx = curr_base_idx

    return moves


def create_figure(output_path: Path):
    """Create the move table illustration figure."""
    # Generate example data
    sequence = "ATCGATCG"
    stride = 5
    signal, true_dwells = generate_synthetic_signal(sequence, mean_dwell=12)
    moves = create_move_table(true_dwells, stride=stride, signal_len=len(signal))

    # Calculate positions
    signal_positions = np.arange(len(signal))
    stride_positions = np.arange(0, len(signal), stride)

    # Reconstruct dwell times from move table (what leech does)
    base_positions = []
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
    # Add points on top of the signal line
    ax1.scatter(signal_positions, signal, color="#2E86AB", s=15, alpha=0.6, zorder=3)
    ax1.set_ylabel("Signal (pA)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "A. Raw Nanopore Signal (from POD5 file)", fontsize=12, fontweight="bold", loc="left"
    )
    ax1.set_xlim(0, len(signal))
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xticklabels([])

    # Annotate signal regions for ALL bases using reconstructed dwells
    # (to ensure alignment with move table and Panel D)
    # Use consistent colors for each base type
    base_colors = {
        "A": "#FF6B6B",
        "C": "#4ECDC4",
        "G": "#45B7D1",
        "T": "#FFA07A",
        "U": "#FFA07A",  # Same as T
    }
    cumsum_pos = 0
    for _i, (base, dwell) in enumerate(zip(sequence, reconstructed_dwells, strict=True)):
        color = base_colors.get(base, "#CCCCCC")  # Default gray for unknown bases
        ax1.axvspan(cumsum_pos, cumsum_pos + dwell, alpha=0.15, color=color)
        ax1.text(
            cumsum_pos + dwell / 2,
            ax1.get_ylim()[1] - 5,
            f"{base}\n{dwell} samples",
            ha="center",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.3},
        )
        cumsum_pos += dwell

    # Add vertical lines at base boundaries
    for pos in base_positions[:-1]:
        ax1.axvline(pos, color="red", linestyle="--", linewidth=1, alpha=0.4)

    # === Panel B: Stride Positions ===
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)

    # Draw stride intervals as boxes to show they span the x-axis
    box_height = 0.3
    for i in range(len(stride_positions)):
        start_pos = stride_positions[i]

        # Calculate width: use stride for all boxes except possibly the last one
        if i < len(stride_positions) - 1:
            box_width = stride
        else:
            # Last box extends to end of signal
            box_width = len(signal) - start_pos

        # Alternate colors for visual clarity
        color = "#FFE5E5" if i % 2 == 0 else "#FFF0F0"
        rect = patches.Rectangle(
            (start_pos, -box_height / 2),
            box_width,
            box_height,
            linewidth=0.5,
            edgecolor="#E63946",
            facecolor=color,
            alpha=0.6,
        )
        ax2.add_patch(rect)

    # Draw vertical marks at stride positions
    ax2.eventplot([stride_positions], colors="#E63946", linewidths=2.0, linelengths=0.6)

    ax2.set_ylabel("Stride\nIntervals", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"B. Basecaller Stride Intervals (stride={stride}, basecaller samples every {stride} signal points)",
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
        ax2.text(pos, 0.35, str(i), ha="center", fontsize=8, color="#E63946", fontweight="bold")

    # Add vertical lines at base boundaries
    for pos in base_positions[:-1]:
        ax2.axvline(pos, color="red", linestyle="--", linewidth=1, alpha=0.4)

    # === Panel C: Move Table ===
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)

    # Draw move table as colored boxes
    box_height = 0.6
    for i, move in enumerate(moves):
        pos = i * stride
        color = "#06D6A0" if move == 1 else "#CCCCCC"

        # Calculate width: use stride for all boxes except the last one
        # Last box extends to the end of the signal
        if i == len(moves) - 1:
            box_width = len(signal) - pos
        else:
            box_width = stride

        rect = patches.Rectangle(
            (pos, -box_height / 2),
            box_width,
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
                pos + box_width / 2,
                0,
                str(move),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    # Add vertical lines at base boundaries to show alignment
    for pos in base_positions[:-1]:  # Exclude the last position (end of signal)
        ax3.axvline(pos, color="red", linestyle="--", linewidth=1.5, alpha=0.6)

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
    for _i, (base, dwell) in enumerate(zip(sequence, reconstructed_dwells, strict=True)):
        color = base_colors.get(base, "#CCCCCC")  # Use same base_colors mapping from Panel A

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

    # Add vertical lines at base boundaries
    for pos in base_positions[:-1]:
        ax4.axvline(pos, color="red", linestyle="--", linewidth=1, alpha=0.4)

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
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#FFF9E6",
            "edgecolor": "#E6B800",
            "linewidth": 2,
        },
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

    print("\nTo use in documentation, add to your markdown:")
    print(f"![Move Table Diagram](../figures/{output_path.name})")


if __name__ == "__main__":
    main()
