# Understanding Move Tables

Move tables are the foundation of dwell time feature extraction in leech.
They encode how raw nanopore signal maps to basecalled sequence, enabling
computation of per-base dwell times that distinguish charged from uncharged
tRNAs.

## Move table overview

The figure below illustrates how leech extracts dwell times from move tables:

![Move Table Decoding](../figures/move_table_diagram.png)

**Panel A** shows the raw nanopore signal with colored regions indicating different bases. **Panel B** displays stride positions where the basecaller samples the signal. **Panel C** shows the move table (from BAM `mv` tag) with 1s indicating new bases and 0s indicating the pore is still reading the same base. **Panel D** combines the sequence with per-base dwell times calculated from the move table.

Modified bases (like charged tRNAs) often exhibit **different translocation kinetics** through the nanopore, resulting in distinctive dwell time patterns that leech models can learn to recognize.

## Nanopore signal and basecalling

Oxford Nanopore sequencers sample ionic current at 4000 Hz as a nucleic acid
strand translocates through the pore. The measured current at any instant
depends on the ~5--9 bases occupying the pore (the k-mer context), not a single
base. This k-mer dependence means that each base influences multiple consecutive
signal samples, and the signal at any position reflects contributions from
neighboring bases.

Translocation speed is not constant. The motor protein ratchets the strand
through the pore one base at a time, but the time spent on each base (the
**dwell time**) varies depending on sequence context, secondary structure, and
chemical modifications. A basecaller (e.g., Dorado) processes the raw signal
and produces both the base sequence and a **move table** recording which signal
positions correspond to base transitions.

## Move table format

The move table is stored in BAM files as the `mv` tag. Its structure:

```
mv:B:c,stride,move_0,move_1,move_2,...
```

- **First element**: the stride, which is the basecaller's downsampling factor
  (typically 5 for Dorado, though some configurations use 6 or 8)
- **Remaining elements**: a binary array where `1` marks the start of a new
  base and `0` means the current base continues

The `ns` tag stores the total number of signal samples for the read. An
optional `ts` tag records the number of signal samples trimmed from the
start (the adapter/primer region).

## Computing dwell times

To convert a move table into per-base dwell times:

1. Extract the stride from `mv[0]` and the binary moves from `mv[1:]`
2. Find positions where `move == 1` (base boundaries)
3. Convert move-space positions to signal-space: `signal_idx = (position + 1) * stride`
4. Compute dwell as the difference between consecutive signal indices

### Worked example

Given a move table and 35 signal samples:

```
mv = [5, 1, 0, 0, 1, 1, 0, 1]
     ↑              ↑
     stride=5       binary moves
```

The moves array is `[1, 0, 0, 1, 1, 0, 1]`. Positions where `move == 1` are
indices 0, 3, 4, and 6.

Converting to signal indices:

| Move index | Signal index           | Base |
|------------|------------------------|------|
| 0          | (0 + 1) × 5 = 5       | A    |
| 3          | (3 + 1) × 5 = 20      | T    |
| 4          | (4 + 1) × 5 = 25      | C    |
| 6          | (6 + 1) × 5 = 35      | G    |

Per-base dwell times (differences between consecutive signal indices):

| Base | Signal range | Dwell (samples) |
|------|-------------|-----------------|
| A    | 5--20       | 15              |
| T    | 20--25      | 5               |
| C    | 25--35      | 10              |

At 4000 Hz, these correspond to 3.75 ms, 1.25 ms, and 2.50 ms respectively.
The variation in dwell times across bases reflects differences in translocation
kinetics at each sequence context.

In leech, the `MoveTable` class in `features.py` handles this conversion:

```python title="Python" linenums="1"
from leech.features import MoveTable

move_table = MoveTable(mv_tag=[5, 1, 0, 0, 1, 1, 0, 1])
seq_to_sig = move_table.to_seq_to_sig_map()
dwells = np.diff(seq_to_sig)
```

## Biological significance

Aminoacylation (charging) of a tRNA attaches an amino acid to the 3' CCA tail.
This chemical modification alters the tRNA's physical properties as it passes
through the nanopore, producing measurable changes in dwell time at and near
the attachment site.

A typical dwell pattern comparison at the CCA motif:

| Position | Uncharged (samples) | Charged (samples) |
|----------|--------------------|--------------------|
| C        | 10                 | 10                 |
| C        | 12                 | 12                 |
| A        | 11                 | 18                 |
| (3' end) | 10                 | 15                 |

The increased dwell at the CCA tail in charged tRNAs reflects the attached
amino acid's effect on translocation kinetics. Different amino acids produce
distinct dwell signatures depending on their size, charge, and hydrophobicity,
which forms the basis for amino acid discrimination.

## Practical considerations

### Edge cases

Reads may begin or end in the middle of a base. Leech requires sufficient
context on both sides of a motif site (default: 200 signal samples left/right,
5 bases for k-mer context). Chunks that cannot satisfy this context requirement
are discarded.

### Indels at motif sites

When using reference-based motif search, leech maps motif positions from the
reference to query coordinates via the CIGAR string. If the alignment contains
insertions or deletions within the motif region, the mapping may be unreliable.
The `--skip-motif-indels` flag filters out such reads.

### Stride variations

The stride is automatically detected from the first element of the `mv` tag.
Common values:

| Basecaller      | Stride |
|-----------------|--------|
| Dorado (default)| 5      |
| Dorado (alt)    | 8      |
| Guppy           | 5      |

No manual configuration is needed; leech reads the stride from each BAM record.

### Required BAM tags

For leech to extract dwell features, BAM records must contain:

- `mv` -- the move table (required)
- `ns` -- total number of signal samples (required)
- `ts` -- trim offset in signal samples (optional; defaults to 0)

These tags are produced by Dorado and Guppy when run with move table output
enabled.
