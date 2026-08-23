# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`leech` (**L**earning **E**nhanced **E**lectrical **C**lassifiers from **H**anopore signals) is a Python library for training machine learning models on nanopore signal data. It extends [Remora](https://github.com/nanoporetech/remora) with dwell time features extracted from move tables (`mv` tag in BAM files) to classify modified bases, specifically for distinguishing charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Installation
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# CLI only (default — no snakemake)
uv sync

# With pipeline/Snakemake support
uv sync --extra pipeline

# All extras (dev, notebook, pipeline)
uv sync --all-extras
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_features.py

# Run with coverage
uv run pytest --cov=leech --cov-report=term-missing

# Run specific test function
uv run pytest tests/test_features.py::test_compute_dwell_times -v
```

### Linting and Formatting
```bash
# Check formatting
uv run ruff format --check .

# Format code
uv run ruff format .

# Lint with ruff
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type checking with ty
uv run ty check src/leech/
```

### Running the CLI

The CLI is organized into workflow-based command groups:
- `leech data` - Prepare and process training data
- `leech model` - Train and optimize models
- `leech eval` - Evaluate and analyze models
- `leech predict` - Run inference on new data

```bash
# Prepare training data (sequential)
uv run leech data prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/

# Prepare training data (parallel - recommended for large datasets)
# Use --workers to specify number of parallel processes
# Use --chunk-size to control batch size (default: 100 reads per batch)
uv run leech data prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/ \
  --workers 8 --chunk-size 100

# Merge and split data
uv run leech data merge -i label1=file1.npz -i label2=file2.npz --output-dir merged/

# Train model
uv run leech model train --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/

# Optimize hyperparameters
uv run leech model optimize --train-data chunks/train.npz --val-data chunks/val.npz \
  --context-grid 200,500,1000 --output-dir grid_results/ --parallel 4

# Evaluate model
uv run leech eval test --model models/model_best.pt --test-data chunks/test.json --output metrics.json

# Compare models
uv run leech eval compare -m models/model1/ -m models/model2/ -t chunks/test.npz -o comparison/

# Analyze feature importance
uv run leech eval importance -m models/model_best.pt -t chunks/test.npz -o importance/

# Sequence ablation testing
uv run leech eval ablation -m models/model_best.pt -t chunks/test.npz -o ablation/

# Bundle pairwise models for deployment
uv run leech model bundle --model-dir results/models/pairwise/ --output bundle.pt --version 1.0.0

# Inspect bundle metadata
uv run leech model bundle-info --bundle bundle.pt

# Calibrate model probabilities (Platt scaling)
uv run leech model calibrate --model-dir models/ --val-data chunks/val.npz

# Export model as standalone TorchScript
uv run leech model export --model-dir models/ -o exported_model.pt

# Run inference (single model)
uv run leech predict --model models/ --pod5 reads.pod5 --bam alignments.bam --output predictions.bam

# Run inference (bundled models, aggregated prediction)
uv run leech predict --bundle bundle.pt --all --pod5 reads.pod5 --bam alignments.bam --output predictions.bam
```

### Adding Dependencies
```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade
```

## Jupyter Notebook Conventions

**IMPORTANT**: When working with Jupyter notebooks in this project:

1. **Plotting library**: Use **plotnine** (ggplot2 for Python) for ALL visualizations
   - ❌ DO NOT use matplotlib
   - ❌ DO NOT use seaborn
   - ✓ Use plotnine exclusively

2. **Display plots**: Use `plot.show()` to display plots inline
   - ✓ `plot.show()` - displays plot inline in notebook
   - ❌ `print(plot)` - DO NOT use
   - ❌ `plt.savefig()` - DO NOT save to PNG files in notebooks

3. **Example**:
   ```python
   import plotnine as p9

   plot = (
       p9.ggplot(data, p9.aes(x="position", y="value"))
       + p9.geom_line()
       + p9.labs(title="My Plot")
   )
   plot.show()  # Display inline - DO NOT use print(plot)
   ```

4. **Rationale**:
   - Plotnine provides consistent, declarative grammar of graphics
   - Better suited for publication-quality scientific plots
   - Easier to maintain consistent styling across notebooks

## Architecture

### Core Data Flow

1. **Input**: POD5 files (raw nanopore signal) + BAM files (basecalls with `mv` move table tags)
2. **Feature Extraction** (`features.py`):
   - Parse move tables to compute per-base dwell times
   - Extract signal level statistics (mean, median, std, range) per base
   - Normalize raw signal using median-MAD (default), z-score, quantile, or pa_scaling
   - Optional signal map refinement via kmer level tables (`signal_refine.py`)
3. **Data Preparation** (`io/`, `preparation/`, `chunking/`):
   - Read BAM + POD5 files (`io/bam_reader.py`, `io/pod5_reader.py`)
   - Search for motifs in reference or basecalled sequence (`io/motif_search.py`)
   - Extract training chunks centered on motifs (e.g., "CCAGGC" for tRNA 3' end) (`chunking/extractor.py`)
   - Serialize chunks for training (`chunking/serialization.py`)
4. **Model Training**: PyTorch models with three input branches (signal, sequence, dwell/level features)
   - Loss functions: BCE, focal loss, cross-entropy (`losses.py`)
   - Augmentation: mixup (signal jitter + scale)
   - LR scheduling: reduce-on-plateau, cosine annealing with warmup
5. **Model Bundling**: Package multiple pairwise models into a single versioned .pt file for deployment
6. **Output**: Trained models (.pt files), model bundles, and predictions (BAM with modification probabilities)

### Parallel Processing

The `prepare` command processes batches of reads concurrently on either backend:

**Implementation** (`preparation/parallel.py`, `preparation/orchestrator.py`):
- `collect_read_infos_from_bam()`: First pass to collect lightweight read metadata from BAM
- Worker functions: Process batches of reads in parallel
- `prepare_training_data_parallel()`: Main parallel orchestration with configurable workers and chunk size

**Usage**:
```bash
# Use --workers N to enable parallel processing (N > 1)
# Use --chunk-size M to control batch size (default: 100 reads)
uv run leech data prepare --pod5 data.pod5 --bam alignments.bam \
  --output-dir chunks/ --workers 8 --chunk-size 100
```

**Performance**:
- Expected speedup: 3-6x on typical multi-core machines
- CPU-bound tasks (feature extraction): near-linear speedup with cores
- I/O-bound tasks (POD5 reading): moderate speedup (2-4x)
- Memory-efficient: Chunks reads into batches to avoid loading entire dataset

**Implementation details**:
- Two-pass design: (1) collect read metadata from BAM, (2) parallel POD5 + feature extraction
- Batching via `chunk_size` prevents memory issues with large datasets
- Reference sequences are passed to workers for reference-based motif search

**Two backends, both concurrent across batches.** `prepare` picks one in
`prepare_training_data_parallel()`:

| | dispatch | why |
|---|---|---|
| Rust (`leech_core` present, config supported) | `_iter_rust_batches` — `ThreadPoolExecutor(num_workers)` | the Rust call releases the GIL for the whole call, POD5 I/O included, so threads give real parallelism without pickling chunks back |
| Python (fallback) | `_iter_python_batches` — `mp.Pool(num_workers)` | each process holds its own cached POD5 reader |

`--workers` sets the number of batches in flight on **both**. Both iterators
yield `(n_reads, chunks)`; the caller does progress and accounting once.

**Do not turn either dispatcher back into a serial `for` loop.** This is the
single easiest mistake to make in this module and it has been made before
(issue #176): the Rust call looks self-parallelizing because per-read work
inside it is rayon-parallel, but nearly all the wall clock on a large POD5 is
POD5 I/O, which is one sequential stream of page faults per call. Driving it
serially leaves one read outstanding at a time and loses ~10-80x to the process
pool. `tests/test_prepare_dispatch.py` fails if this regresses.

### POD5 access from Rust

`rust/src/pod5_cache.rs` holds a process-global cache of open `escapepod_signal::Reader`s.
**Always go through `cached_reader(path)`; never call `Reader::open` directly.**

A `Reader` caches its read-id index in a `OnceLock` on itself, and
`reads_by_ids` without an index falls back to a single-threaded scan of the
entire reads table (all 22 columns) that can only stop once every target is
found. Reads arrive in BAM order, which is unrelated to POD5 storage order, so
that scan effectively runs to end-of-file — per batch. `cached_reader` opens
each POD5 once per process and warms the index once, in memory, from a scan
projected to the read_id column. It writes no `.p5s` sidecar and does not
require one.

The Python side does the equivalent in `io/pod5_reader.py`, which caches an
entered `DatasetReader` per process (entering is what warms the index).

Phase 1 (POD5 I/O) in `inference.rs` and `training.rs` runs inside `py.detach`.
Keep it that way — holding the GIL across the I/O serializes every caller
thread and makes batch-level concurrency impossible.

### Backend parity: when a chunk is dropped

**The only reason to drop a focus base is that it has no signal boundaries** —
`base_idx < 0` or `base_idx >= num_bases`, where `num_bases` is
`len(seq_to_sig_map) - 1`, *not* the sequence length. Everything else pads:
the k-mer window with `N`, dwell/features/signal with zeros. That rule lives in
`LeechRead.get_chunk`, and `extract_training_chunks_from_read` (`training.rs`)
and its twin in `inference.rs` must match it exactly.

Adding a `continue` to either Rust loop is how issue #185 happened. The guard
there also rejected a k-mer window overhanging the sequence, which under
`anchor="reference"` is the *aligned reference slice* — so every read whose
alignment stopped within `kmer_context` of the motif lost its only chunk. That
was ~1% of a production corpus, silently, and skewed toward supplementary and
indel-heavy alignments: the population the classifier most needs.

The two definitions of `num_bases` are not interchangeable.
`compute_ref_to_signal` strips trailing non-match CIGAR ops, so the map can be
shorter than the reference slice; `LeechRead.num_mapped_bases` is the focus
bound, `num_bases` is the sequence bound.

Which bases a read contributes chunks at is `chunking.find_focus_bases`, one
function, called by both backends. `parallel.py` used to carry its own copy and
it drifted. Do not add a second one.

**`seq_to_sig_map` and `sequence_with_kmer_context` come from the SIGNAL
window, not the k-mer window.** `get_chunk` locates the covered bases with two
`searchsorted` calls over the read's map, then snaps the partially overlapping
first and last bases to the window edges (`[0] = 0`, `[-1] = chunk_len`). The
k-mer window spans a different number of bases, so building these from
`kmer_start`/`kmer_end` disagrees on every chunk — that was issue #186, and it
went unnoticed because nothing compared the fields. Rust does it in
`signal_mapping::chunk_signal_kmer_inputs`, used by both `training.rs` and
`inference.rs`. Only `--seq-encoding signal_kmer` reads them, which is why a
full divergence was invisible.

`tests/test_parallel_prep.py::TestEdgeWindowParity` and
`::TestSignalKmerFieldParity` fail if any of this regresses; the backends are
held to chunk-set equality (not overlap) and to identical `signal_kmer`
encodings.

### Key Classes and Functions

**`MoveTable` (features.py)**
- Parses move table from BAM `mv` tag
- `to_seq_to_sig_map()`: converts moves to base→signal index mapping
- Core data structure for dwell time computation

**`BamReader` (io/bam_reader.py)**
- Reads BAM files and extracts alignment information
- Handles move table tags and quality filtering
- Coordinates with POD5Reader for signal extraction

**`POD5Reader` (io/pod5_reader.py)**
- Reads raw signal from POD5 files
- Maps read IDs from BAM to POD5 signal data

**`ChunkExtractor` (chunking/extractor.py)**
- Extracts training chunks centered on motifs
- Handles signal/sequence/feature context windows
- Creates chunk dictionaries with all features aligned

**`MotifSearcher` (io/motif_search.py)**
- Searches for motifs in reference or basecalled sequences
- Maps motif positions from reference to query coordinates using CIGAR
- Filters reads with indels in motif regions (optional)

**Config-driven models (`models/nn.py`, `models/configs/*.toml`)**
- `models/nn.py` is a bonito-style layer registry: `register`, `to_dict`,
  `from_dict`, plus `Serial`, `Stack`, `Parallel` and `Graph`.
- `Graph` is the multi-branch container: nodes are installed as *direct*
  attributes of the root module, so a TOML-declared architecture has the same
  `state_dict()` keys as the hand-written class it replaced.
- TOML values may be expressions: `"${lstm_hidden * 2}"`, `"${DEFAULT_KMER_LEN}"`.
  A node may be conditional (`when = "${has_features}"`), which is how one
  config serves several registry names via `[[variants]]`.
- Nodes execute in declaration order. When module-construction order must
  differ from dataflow order (as for the TCN Motor / DwellAttn variants, whose
  classes appended layers after their parent's head), declare an explicit
  `build_order` — it fixes the `state_dict()` key order and the seeded-init
  RNG order without changing execution.
- The ConvLSTM and TCN families (22 of 29 registry names) are declared this way;
  `kind = "graph"` is the only config kind.
- Adding a normalization/pooling variant should be a `[[variants]]` entry, not
  a new class.

**`ConvLSTMDwell` (models/configs/conv_lstm.toml)**
- PyTorch model with three branches:
  - Signal branch: Conv1d on raw signal
  - Sequence branch: Conv1d on one-hot encoded k-mers (or signal_kmer encoding)
  - Feature branch: Conv1d on dwell+level features (NEW vs. Remora)
- Branches merge → BiLSTM → FC output
- Compare with `ConvLSTMBase` (no dwell features) to measure impact

**Remora-compatible models** (`models/conv_lstm_remora.py`)
- `ConvLSTMRemora`: Remora-compatible architecture with dwell features
- `ConvLSTMRemoraBase`: Remora-compatible architecture without dwell features
- `RemoraModelWrapper` (`models/remora_compat.py`): Wraps Remora models for leech inference

**Model bundling** (`util.py`)
- Bundle multiple trained models into a single versioned .pt file
- Supports pairwise and one-vs-all aggregation modes
- Provenance tracking: motif, motif_offset, base_justify stored per model

**`PlattScaling` / `calibrate_model()` (calibration.py)**
- Post-hoc Platt scaling for probability calibration
- Fits two parameters (a, b) per model: sigmoid(a*logit + b)
- Essential for one-vs-all bundles with different positive-class rates

### Module Organization

```
src/leech/           # Main package source
├── cli.py           # Command-line interface (rich-click based)
├── cli_config.py    # Rich-click styling configuration
├── cli_options.py   # Shared option decorators for CLI commands
├── commands/        # CLI command handlers
│   ├── prepare.py   # Prepare command implementation
│   ├── merge_split.py  # Merge-and-split command
│   ├── train.py     # Train command handler
│   ├── bundle.py    # Bundle, bundle-info, export handlers
│   ├── calibrate.py # Platt scaling calibration handler
│   ├── eval.py      # Test command handler
│   ├── optimize.py  # Grid search optimization handler
│   ├── predict.py   # Inference/predict handler
│   └── analyze.py   # Analysis command handlers (importance, ablation)
├── confounds.py     # Confound mappings for adversarial training
├── io/              # Input/output operations
│   ├── bam_reader.py    # BAM file reading
│   ├── pod5_reader.py   # POD5 signal reading
│   ├── motif_search.py  # Motif searching in sequences
│   ├── reference.py     # Reference sequence handling
│   └── tsv_writer.py   # TSV prediction output writer
├── preparation/     # Data preparation orchestration
│   ├── orchestrator.py  # Main preparation logic
│   ├── parallel.py      # Parallel processing
│   ├── reader.py        # Read iteration
│   └── encoding.py      # Sequence encoding
├── chunking/        # Training chunk extraction
│   ├── extractor.py     # Chunk extraction logic
│   └── serialization.py # Save/load chunks
├── splitting/       # Data splitting
│   └── splitter.py  # Train/val/test split
├── features.py      # MoveTable, dwell times, signal levels, normalization
├── dataset.py       # PyTorch Dataset classes for loading chunks
├── training.py      # Training loop with Trainer class
├── evaluation.py    # Model evaluation and testing
├── inference.py     # Inference engine (multi-model, Remora compat)
├── gridsearch.py    # Grid search for chunk context optimization
├── util.py          # Helper functions (model loading, bundling, metrics)
├── losses.py        # Loss function implementations (BCE, focal, cross-entropy)
├── calibration.py   # Post-hoc Platt scaling for model calibration
├── signal_refine.py # Signal map refinement via kmer level tables
├── _rust_accel.py   # Rust acceleration wrapper for vectorized operations
├── configs.py       # Dataclass-based configuration management
├── constants.py     # Project-wide constants and defaults
├── logging_config.py  # Logging setup
└── models/          # Model architectures
    ├── __init__.py            # Model registry and get_model()
    ├── nn.py                  # Layer registry + composition primitives
    ├── config_loader.py       # TOML -> model class (torch-free discovery)
    ├── configs/               # TOML-declared architectures
    │   ├── conv_lstm.toml         # ConvLSTMBase/BN, ConvLSTMDwell/BN
    │   ├── conv_lstm_attn.toml    # the six *Attn variants
    │   ├── tcn_dwell.toml         # TCNDwell + GN/LN
    │   ├── tcn_dwell_residual.toml        # TCNDwellResidual + GN/LN
    │   ├── tcn_dwell_split_residual.toml  # TCNDwellSplitResidual + LN
    │   ├── tcn_dwell_residual_motor.toml  # + motor-region pooling
    │   └── tcn_dwell_residual_dwell_attn.toml  # + dwell-only cross-attention
    ├── components.py          # Reusable components (conv branches, TCN blocks)
    ├── inference_wrapper.py   # Inference wrapper pattern
    ├── conv_lstm_remora.py    # ConvLSTMRemora / ConvLSTMRemoraBase
    ├── transformer_dwell.py   # TransformerDwell architecture
    ├── resnet_dwell.py        # ResNetDwell architecture
    ├── conv_only.py           # ConvOnly architecture
    ├── signal_cnn.py          # SignalCNN (signal-only classifier)
    └── remora_compat.py       # RemoraModelWrapper for Remora model inference

tests/               # pytest tests
```

## Implementation Details

### Feature Engineering
- **Dwell times**: Number of signal samples per base, computed from move table using `np.diff(seq_to_sig_map)`
- **Dwell offset**: Tunable motor-sensor offset (`dwell_offset` param) for correcting the physical delay between motor and pore
- **Base justification**: Controls signal chunk centering relative to the focus base (`start`, `center`, `end`)
- **Signal normalization**: Median-MAD (default) is robust to outliers; z-score, quantile, and pa_scaling (physics-aware) methods available
- **Signal map refinement**: Optional kmer-level-table-based refinement of base boundaries (`--refine-signal-map`, `--kmer-table`)
- **Signal orientation**: RNA signals are reversed by default (POD5 stores 3'→5', basecaller expects 5'→3'); use `--no-reverse-signal` for DNA
- **Sequence encoding**: `base_onehot` (default 4-channel) or `signal_kmer` (36-dimensional signal-level kmers)
- **Feature concatenation**: Models expect 3 inputs: (signal, sequence, features) where features combines dwell and signal statistics

### Platt Scaling and Balance-Groups
- **Platt scaling** (`calibration.py`): Post-hoc calibration fitting `sigmoid(a*logit + b)` on validation data. Essential for one-vs-all bundles where models trained with different positive-class rates must produce comparable probabilities for argmax aggregation.
- **Balance-groups sampling**: When `--balance-groups` is enabled, training balances sampling across `source_group` labels (tracked per chunk) so each group contributes equally per epoch. Prevents dominant groups from biasing the model.
- **K-fold cross-validation**: `leech data merge --k-fold K` creates K stratified read-level splits for cross-validation. Bundle discovery auto-selects the best fold by validation F1 when k-fold directories are present.

### Move Table Format
The BAM `mv` tag format:
- First element: stride (basecaller downsampling factor, typically 5)
- Remaining elements: binary array where 1 = new base, 0 = same base
- Convert to signal indices: `signal_idx = move_position * stride + trim_offset` (Remora convention)
- Final entry is `num_samples` (from `ns` tag)

### Training Data Structure
Training chunks are dictionaries:
```python
{
    'signal': np.ndarray,      # Raw signal chunk [signal_len]
    'sequence': str,           # K-mer context sequence
    'dwell': np.ndarray,       # Per-base dwell times [kmer_len]
    'features': np.ndarray,    # Stacked features [num_features, kmer_len]
    'label': int,              # 0=uncharged, 1=charged
    'read_id': str,
    'base_idx': int,
    'source_group': str,       # Source group label (e.g., amino acid identity)
}
```

### Snakemake Integration
The Snakemake workflow is included in this repository under `pipeline/`. It provides production-ready pipelines for:
- Charged vs uncharged tRNA classification
- Pairwise amino acid classification
- Grid search optimization
- Model comparison across architectures
- HPC cluster integration (SLURM/LSF)

The workflow is designed to integrate with the leech CLI commands and supports both local and cluster execution.

## Important Constraints

1. **Move table requirement**: BAM files MUST have `mv` and `ns` tags (from dorado/guppy basecaller)
2. **Read ID matching**: POD5 read IDs must match BAM query names exactly
3. **Motif-based extraction**: Training focuses on specific motifs (e.g., "CCAGGC" for tRNA); motif_offset specifies the focus base within the motif
   - **Reference-based search (default)**: Searches for motif in reference sequence, maps to query via CIGAR. Avoids bias from basecalling errors at modification sites.
   - **Basecalled search**: Searches in basecalled sequence (backward compatible). Use `--motif-reference bam` to enable.
4. **Anchor modes**: `reference` (default) anchors chunks to reference coordinates via CIGAR (matches Remora `--reference-anchored` behavior); `basecall` anchors to basecalled coordinates
5. **Reference sequences**: For reference-based motif search, BAM must contain @SQ sequences in header OR provide `--reference-fasta` path
6. **Signal orientation**: RNA signals are auto-reversed (3'→5' to 5'→3'). Use `--no-reverse-signal` for DNA data
7. **Edge handling**: Chunks require sufficient context (default: 200 samples left/right for signal, 5 bases for k-mer)
8. **Feature alignment**: All three model inputs (signal, sequence, features) must be temporally aligned after convolution layers

## Dependencies

- **PyTorch** (2.5+): Neural network training
- **pod5** (0.3+): Reading ONT POD5 format
- **pysam** (0.22+): BAM file parsing
- **rich-click**: CLI framework (click + rich formatting)
- **polars**: Fast dataframe operations
- **numpy/scipy/scikit-learn** (1.5+): Numerical operations and ML utilities
- **pydantic**: Config validation
- **ruff**: Linting and formatting (replaces black/flake8)
- **ty**: Type checking (replaces mypy)

## Current Status

The codebase is feature-complete (v0.6.1):
- ✓ Feature extraction with dwell offset tuning and signal map refinement
- ✓ 29 model architectures: ConvLSTM (Base/Dwell × BN/GN/LN/Attn), TCN (Dwell/DwellGN/DwellLN/DwellResidual/DwellResidualGN/DwellResidualLN/DwellResidualMotor/DwellResidualDwellAttn/DwellSplitResidual/DwellSplitResidualLN), Transformer (Dwell/DwellResidual), ResNet, ConvOnly, SignalCNN
- ✓ Config-driven model layer: bonito-style layer registry (`models/nn.py`) + TOML architecture declarations (`models/configs/`)
- ✓ CLI organized into 4 command groups: `data` (prepare, merge), `model` (train, optimize, bundle, bundle-info, calibrate, export), `eval` (test, compare, importance, ablation), `predict`
- ✓ Training with focal loss, mixup augmentation, cosine annealing, gradient clipping, adversarial training, CL regression
- ✓ Grid search with range syntax, parallel execution, dwell offset tuning
- ✓ Model bundling for multi-model pairwise deployment
- ✓ Inference engine with leech and Remora model auto-detection
- ✓ Parallel data preparation and parallel inference
- ✓ Reference-anchored mode matching Remora convention
- ✓ Rust acceleration via escapepod-rs for 10x faster inference
- ✓ Snakemake workflow for production pipelines
- ✓ Platt scaling and multiclass temperature scaling for model calibration
- ✓ Balance-groups sampling for equal source group contribution
- ✓ K-fold cross-validation with read-level stratification
- ✓ TorchScript export for standalone model deployment
- ✓ TSV prediction output for downstream analysis
- ✓ `--signal-context` CLI option for asymmetric signal windows
- ✓ `--min-confidence` / `--min-margin` predict thresholds
- ✓ Cross-layer augmentation (time mask, shift, feature noise)
- ✓ Auto-read anchor and reference_fasta from model config at predict time
- ✓ `--backend auto|rust|python` on predict; `auto` falls back to Python (with a
  warning) for options the Rust pipeline cannot honor, `rust` raises instead
- ✓ `--no-require-query-mapping` for modifications that mis-call the motif
- ✓ Process-global POD5 reader cache in Rust (`rust/src/pod5_cache.rs`) and
  concurrent batch dispatch on both prepare backends
- ✓ Chunk-set parity between the prepare backends, one shared
  `find_focus_bases`, and a logged read yield on both
- ✓ Identical `signal_kmer` encodings across backends (shared
  `chunk_signal_kmer_inputs`)

All core functionality is implemented and ready for use.

## Git Conventions

- Do not add `Co-Authored-By` lines to commit messages.
