# Classification Tasks

Leech is a general library for training classifiers on motif-centered
nanopore signal: the task, the classes, and what counts as "identity" for a
conditional or hierarchical model are all yours to define. This page works
through the concrete example that drove leech's development -- tRNA
aminoacylation ("charging") state and amino acid identity from direct RNA
sequencing -- because it exercises binary classification, hard multi-way
discrimination, and identity-conditioned modeling strategies that generalize
to other motif-centered classification tasks (e.g., other base modifications
identified by a known sequence context).

Within that example, leech supports two tasks of increasing difficulty: binary
charged/uncharged classification and amino acid discrimination. This page
describes both tasks, the modeling strategies for each, and how tRNA identity
can be used to simplify the amino acid problem.

## Charged vs. uncharged classification

The simpler task is distinguishing charged (aminoacylated) from uncharged tRNAs.
Aminoacylation is expected to shift signal amplitude at the CCA tail, which
should be large enough for signal-only models to detect with reasonable
accuracy.

!!! warning "Illustrative, not measured"
    The accuracy ranges in the table below are expectations used to frame the
    task, not results. No results file in this repository backs them. Measure
    your own with `leech eval test`.

| Model         | Expected accuracy (illustrative) | Key signal         |
|---------------|-------------------|--------------------|
| ConvLSTMBase  | ~75--80% (unmeasured) | Signal amplitude   |
| ConvLSTMDwell | ~85--90% (unmeasured) | Amplitude + dwell  |

Dwell features still improve performance on this task because they capture
translocation kinetics that amplitude alone misses, but the binary task is
largely solvable without them. The more challenging and scientifically
interesting task is identifying *which* amino acid is attached.

## Amino acid discrimination

The harder, more scientifically interesting task in this example is
classifying which of the 20 standard amino acids is attached to a charged
tRNA. This is substantially harder than binary classification because the
signal differences between amino acids are small.

Consider three amino acids. The values below are **illustrative numbers
chosen to show the shape of the problem, not measurements**:

| Amino acid | Signal amplitude (pA, illustrative) | Dwell (samples, illustrative) |
|------------|----------------------|------------------|
| Ala        | 94                   | 8                |
| Gly        | 93                   | 6                |
| Trp        | 98                   | 15               |

While Trp is distinguishable by amplitude, Ala and Gly differ by only 1 pA --
within read-to-read noise. In this illustration their dwell times differ
substantially, which is why dwell features are hypothesized to help. More
broadly, amino acid physical properties are hypothesized to relate to dwell
time:

| Property         | Hypothesized relation to dwell |
|------------------|--------------------------------|
| Molecular weight | Positive                       |
| Volume           | Positive                       |
| Charge           | Weaker                         |

This is the expected physical basis for dwell-based discrimination: larger
amino acids may cause more drag during translocation, increasing dwell time at
the CCA tail. No correlation coefficients have been measured or recorded in
this repository.

### Expected confusion patterns

Some amino acid pairs are inherently difficult to separate due to chemical
similarity:

- **High confusion**: Ile/Leu, Asp/Glu, Ser/Thr
- **Low confusion**: Gly/Trp, Lys/Asp

## tRNA-conditional strategies

At inference time, the tRNA identity is known from sequence alignment. This
knowledge can dramatically simplify the classification problem because
aminoacyl-tRNA synthetases (aaRSs) are highly specific: mischarging occurs
almost exclusively between chemically similar amino acids. Instead of a 20-way
classification, the problem reduces to a binary or few-way task.

### Common mischarging pairs

Each tRNA is charged by a specific aaRS that occasionally mischarges with a
chemically similar amino acid:

| tRNA      | Cognate AA | Near-cognate AAs | Basis                |
|-----------|------------|-------------------|----------------------|
| tRNA-Ile  | Ile        | Val, Leu          | Branched-chain       |
| tRNA-Asp  | Asp        | Glu               | Both acidic          |
| tRNA-Glu  | Glu        | Asp               | Both acidic          |
| tRNA-Phe  | Phe        | Tyr               | Both aromatic        |
| tRNA-Tyr  | Tyr        | Phe               | Both aromatic        |
| tRNA-Ser  | Ser        | Thr               | Both have hydroxyl   |
| tRNA-Thr  | Thr        | Ser               | Both have hydroxyl   |
| tRNA-Val  | Val        | Ile               | Branched-chain       |
| tRNA-Leu  | Leu        | Ile, Val          | Branched-chain       |

tRNAs not listed (Ala, Arg, Asn, Cys, Gln, Gly, His, Lys, Met, Pro, Trp) have
highly specific synthetases with rare mischarging.

### Modeling strategies

Four approaches leverage tRNA identity:

**tRNA-specific binary classifiers.** Train one classifier per tRNA family that
distinguishes the cognate amino acid from its known near-cognates. For example,
a tRNA-Ile classifier performs 3-way classification (Ile vs. Val vs. Leu). This
is the simplest approach and works well when training data is available for each
tRNA.

**Hierarchical classification.** A two-stage pipeline: first identify the tRNA
(from sequence), then apply a tRNA-specific second-stage classifier for amino
acid identity. This separates the problems cleanly.

**Conditional input.** A single model receives tRNA identity as an additional
input (e.g., a learned embedding). The model uses a loss mask so that only the
cognate and near-cognate amino acids contribute to the loss for each tRNA. This
allows shared feature learning across tRNAs.

**Shared encoder with tRNA-specific heads.** A common ConvLSTM encoder
processes signal, sequence, and dwell features. The encoder output feeds
tRNA-specific classification heads, each performing binary or few-way
classification. This balances parameter sharing with task-specific outputs.

| Strategy              | Advantages                          | Disadvantages                  |
|-----------------------|-------------------------------------|--------------------------------|
| Binary per tRNA       | Simple, high accuracy per task      | Requires per-tRNA training data|
| Hierarchical          | Clean separation of concerns        | Error propagation between stages|
| Conditional input     | Single model, shared features       | Complex loss masking           |
| Shared encoder + heads| Shared learning, specific outputs   | More parameters                |

## Multi-task learning

A multi-task model predicts both charging state and amino acid identity
simultaneously using a shared encoder with two output heads:

- **Charging head** -- binary classification (charged vs. uncharged)
- **Amino acid head** -- 21-way classification (20 amino acids + uncharged)

The combined loss uses a weighted sum:

```
total_loss = 0.7 * charging_loss + 0.3 * aa_loss
```

The charging task is easier and provides a strong gradient signal early in
training, while the amino acid task drives the model to learn finer-grained
representations. The shared encoder benefits from both objectives.

!!! tip
    At inference on biological data where the amino acid is unknown, the
    multi-task model provides both predictions: whether the tRNA is charged,
    and if so, which amino acid is attached.

## Expected performance

!!! warning "Illustrative, not measured"
    The ranges in the table below are expectations, not findings. No results
    file in this repository backs them.

Performance depends on data source. Synthetic data (purified tRNAs charged
with individual amino acids) provides clean training signal. Biological data
introduces variability from cellular context, mixed populations, and post-
transcriptional modifications.

| Model           | Synthetic 20-way | Bio zero-shot | Bio fine-tuned |
|-----------------|------------------|---------------|----------------|
| ConvLSTMBase    | ~55--65%         | --            | --             |
| ConvLSTMDwell   | ~72--82%         | ~42--58%      | ~62--72%       |
| tRNA-conditional| ~90--97%         | ~80--90%      | ~90--95%       |

The expected gap between 20-way and tRNA-conditional performance motivates the
value of incorporating biological knowledge about aaRS specificity.

!!! note
    These ranges are rough expectations of the task's difficulty. Performance on
    biological data depends on how well synthetic training data represents
    in vivo conditions. Fine-tuning on a small amount of biological data
    is expected to close much of the domain gap.
