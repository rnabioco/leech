# Understanding Move Tables and Dwell Times

This guide provides a detailed explanation of how leech extracts dwell times from move tables and why this is critical for detecting modified bases.

## Overview

Move tables are a key innovation from Oxford Nanopore Technologies' basecallers (dorado/guppy) that enable base-level alignment of raw signal. Understanding move tables is essential to understanding how leech works.

## The Nanopore Sequencing Process

```
┌─────────────────────────────────────────────────────────────┐
│                    Nanopore Sequencing                       │
│                                                              │
│  DNA/RNA ────▶ Nanopore ────▶ Current Changes ────▶ Signal  │
│                   Pore                  (pA)         (POD5)  │
│                                                              │
│  As molecule moves through pore, current changes based on   │
│  the bases inside the pore (k-mer dependent)                │
└─────────────────────────────────────────────────────────────┘
```

The raw signal is sampled at **4000 Hz** (4000 samples per second), producing a continuous stream of picoamp measurements.

## The Basecalling Challenge

The basecaller must convert this continuous signal into discrete bases (A, C, G, T/U). However:

1. **Signal is k-mer dependent**: The current is influenced by ~5-9 bases in the pore simultaneously
2. **Variable translocation speed**: Bases move through at different rates
3. **No 1:1 mapping**: Multiple signal samples correspond to each base

## Move Tables: The Solution

Move tables solve this by recording **when the basecaller transitions between bases**.

### Move Table Format

```
Move Table (mv tag in BAM):
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ stride │ move_0 │ move_1 │ move_2 │ move_3 │  ...   │
├────────┼────────┼────────┼────────┼────────┼────────┤
│   5    │   1    │   0    │   0    │   1    │  ...   │
└────────┴────────┴────────┴────────┴────────┴────────┘

stride:  Basecaller downsampling factor (typically 5)
move_i:  1 = new base, 0 = same base as previous
```

### Visual Example

Let's trace through a concrete example:

```
Raw Signal (4000 Hz samples):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0   5   10  15  20  25  30  35  40  45  50  55  60  65
 │   │   │   │   │   │   │   │   │   │   │   │   │   │
 ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼

Basecaller Stride Positions (every 5 samples):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0       5       10      15      20      25      30      35
 │       │       │       │       │       │       │       │
 ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼

Move Table: [5, 1,      0,      0,      1,      1,      0,      1]
                ↓       ↓       ↓       ↓       ↓       ↓       ↓
Bases:          A───────A───────A───────T───────C───────C───────G

Dwell Times (samples per base):
   Base A: samples 0-15  = 15 samples
   Base T: samples 15-20 = 5 samples
   Base C: samples 20-30 = 10 samples
   Base G: samples 30-35 = 5 samples
```

## Algorithm: Computing Dwell Times

Here's how leech converts a move table into dwell times:

```python
def compute_dwell_times(move_table):
    """
    Convert move table to dwell times.

    Args:
        move_table: [stride, move_0, move_1, ..., move_n]

    Returns:
        dwells: Array of samples per base
    """
    stride = move_table[0]
    moves = move_table[1:]

    # Find positions where moves occur (move == 1)
    base_positions = []
    for i, move in enumerate(moves):
        if move == 1:
            signal_pos = (i + 1) * stride  # +1 because of stride offset
            base_positions.append(signal_pos)

    # Add end position
    base_positions.append(len(signal))

    # Compute dwells as differences
    dwells = np.diff(base_positions)

    return dwells
```

### Step-by-Step Example

```
Input:
  move_table = [5, 1, 0, 0, 1, 1, 0, 1]
  signal_len = 35

Step 1: Extract stride and moves
  stride = 5
  moves = [1, 0, 0, 1, 1, 0, 1]

Step 2: Find base positions (where move == 1)
  Index 0: move=1 → position = (0+1)*5 = 5
  Index 1: move=0 → (skip)
  Index 2: move=0 → (skip)
  Index 3: move=1 → position = (3+1)*5 = 20
  Index 4: move=1 → position = (4+1)*5 = 25
  Index 5: move=0 → (skip)
  Index 6: move=1 → position = (6+1)*5 = 35

  base_positions = [5, 20, 25, 35]

Step 3: Compute dwells
  dwell[0] = 20 - 5  = 15 samples (base A)
  dwell[1] = 25 - 20 = 5 samples  (base T)
  dwell[2] = 35 - 25 = 10 samples (base C)

  dwells = [15, 5, 10]
```

## Why Dwell Times Matter

### Signal Alone: Limited Resolution

```
Signal (continuous):
    ╭─────╮     ╭───╮   ╭─────╮
────╯     ╰─────╯   ╰───╯     ╰────
    A A A       T     C C
    └─────┘     └─┘   └───┘
   Hard to determine exact base boundaries!
```

### Sequence Alone: No Signal Information

```
Sequence:  A T C G
           │ │ │ │
           No information about signal characteristics
           or translocation kinetics
```

### Dwell Times: The Bridge

```
Dwell times connect signal to bases:

Signal:    [─────────────────][───][─────────]
Bases:            A              T       C
Dwells:          15              5      10

Modified bases ────▶ Different kinetics ────▶ Different dwells!
```

## Biological Significance

### Example: Charged vs Uncharged tRNAs

```
┌─────────────────────────────────────────────────────┐
│            Uncharged tRNA (no amino acid)           │
├─────────────────────────────────────────────────────┤
│  Base:     C      C      A      G      G      C     │
│  Dwell:    10     12     11     10     12     11    │
│  Signal:   ▁▁▁▁  ▃▃▃▃  ▂▂▂▂  ▁▁▁▁  ▃▃▃▃  ▂▂▂▂    │
│                                                     │
│  → Normal translocation kinetics                   │
│  → Consistent dwell times                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│         Charged tRNA (with amino acid attached)     │
├─────────────────────────────────────────────────────┤
│  Base:     C      C      A      G      G      C     │
│  Dwell:    10     12     18     15     12     11    │
│  Signal:   ▁▁▁▁  ▃▃▃▃  ▂▂▂▂  ▁▁▁▁  ▃▃▃▃  ▂▂▂▂    │
│                    ↑↑↑↑  ↑↑↑↑                       │
│                  Modified site!                     │
│                                                     │
│  → Altered translocation at modification site      │
│  → Increased dwell times at specific positions     │
│  → Detectable by machine learning!                 │
└─────────────────────────────────────────────────────┘
```

The amino acid attachment causes:
1. **Structural changes** in the tRNA
2. **Different interactions** with the nanopore
3. **Altered translocation kinetics**
4. **Distinctive dwell patterns** at and near the modification site

## Multi-Modal Features

Leech combines three complementary feature types:

```
┌──────────────────┬─────────────────┬──────────────────┐
│  Signal Features │ Sequence Context│  Dwell Features  │
├──────────────────┼─────────────────┼──────────────────┤
│  • Mean(signal)  │  • K-mer        │  • Raw dwell     │
│  • Std(signal)   │  • One-hot enc. │  • Dwell ratios  │
│  • Median        │  • Position     │  • Dwell z-score │
│  • Range         │  • Context      │  • Neighbors     │
└──────────────────┴─────────────────┴──────────────────┘
            ↓                ↓                ↓
       ┌────────────────────────────────────────┐
       │      Multi-Branch Neural Network       │
       │   (Conv-LSTM with 3 input branches)    │
       └────────────────────────────────────────┘
                           ↓
              ┌──────────────────────┐
              │  Classification:     │
              │  Charged vs Uncharged│
              └──────────────────────┘
```

## Practical Considerations

### 1. Edge Cases

```
Problem: Reads may start/end mid-base

Solution: Filter chunks requiring full context
         (signal_context + kmer_context on both sides)
```

### 2. Indels in Reference Alignment

```
Problem: CIGAR operations may disrupt signal-to-base mapping

Reference:  A T C - - G T
Query:      A T C G A G T
                  ↑ ↑
            Insertion affects downstream positions

Solution: --skip-motif-indels flag skips motif sites with indels
```

### 3. Basecaller Stride Variations

```
Different basecallers use different strides:
  • Guppy:  stride = 5
  • Dorado: stride = 5 (default) or 8

Leech automatically detects from mv[0]
```

## Generating Move Table Diagrams

You can generate custom diagrams using the provided script:

```bash
# Generate with default synthetic data
python scripts/generate_move_table_figure.py

# Specify output location
python scripts/generate_move_table_figure.py --output my_diagram.png

# Set random seed for reproducibility
python scripts/generate_move_table_figure.py --seed 123
```

## Further Reading

- **ONT Move Table Specification**: [Dorado documentation](https://github.com/nanoporetech/dorado)
- **Remora Analysis**: [03-REMORA_ANALYSIS.md](03-REMORA_ANALYSIS.md)
- **Feature Extraction**: [API Reference - Features](../api/features.md)

## Summary

Move tables are essential for leech because they:

1. ✅ **Enable base-level signal alignment** without complex HMM inference
2. ✅ **Provide dwell times** that reveal translocation kinetics
3. ✅ **Bridge sequence and signal** for multi-modal learning
4. ✅ **Detect modifications** through altered kinetic patterns
5. ✅ **Scale efficiently** for large datasets

Understanding this relationship is key to understanding how leech achieves high accuracy in detecting tRNA aminoacylation and other modifications.
