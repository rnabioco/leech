# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`leech` (**L**earning **E**nhanced **E**lectrical **C**lassifiers from **H**anopore signals) is a Python library for training machine learning models on nanopore signal data. It extends [Remora](https://github.com/nanoporetech/remora) with dwell time features extracted from move tables (`mv` tag in BAM files) to classify modified bases, specifically for distinguishing charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Installation

Released on PyPI as two distributions: `leech` (pure Python) and `leech-core`
(the `abi3` Rust extension, pulled by the `rust` extra). End users install
`uv add "leech[rust]"`. From a checkout:

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

# Train a CTC-CRF sequence model (emits a sequence, not a label)
uv run leech model train-crf --corpus corpus/ldx16 --output-dir models/crf/ \
  --epochs 32 --batch-size 256

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

**The feature window resolves in exactly one place:
`chunking.resolve_feature_window`.** `feature_start`/`feature_end` are signed
offsets from the focus base (end inclusive) and are optional — `None` means the
k-mer window, `±kmer_context`. `0` is a legitimate start (features begin *at*
the focus, the right-only window for tRNA 3' ends), so the fallback test must be
`is None`; `feature_start or -kmer_context` silently widens the window, which
was issue #189. Rust honors whatever it is passed, but the Python side stamped
the resolved value onto the chunk with `or -5` and got it wrong on every chunk
of a `--feature-start 0` run.

Nothing about the arrays gives that away — they have the requested width and
contents. The stored value is what `dataset.py` slices the k-mer window out of
the feature array with (`kmer_start = -kmer_context - feature_start`) and what
`training.py` copies into the model config for `predict`, so a wrong one shifts
training and inference by `kmer_context` bases with no shape error. `data
prepare` now logs the resolved window and records it in `prepare_config.json`.

`tests/test_parallel_prep.py::TestEdgeWindowParity`,
`::TestSignalKmerFieldParity`, and `::TestFeatureWindowParity` fail if any of
this regresses; the backends are held to chunk-set equality (not overlap), to
identical `signal_kmer` encodings, and to the same stored feature window.

**`tests/test_backend_parity.py` is the net that catches what those miss.** It
serializes both backends through `save_chunks` and compares *every array in the
npz*, over a matrix of anchors, refinement settings, `base_justify` values,
feature windows, `scale_iters` and signal contexts. It fails on any chunk field
it has not been told how to compare, so adding a field to the chunk format
without classifying it is a test failure rather than a silent gap. That
property is the point: every divergence so far (#185 counts, #186 signal_kmer,
#189 window width, #193 values) was invisible to the check written for the
previous one. Dwells and int/string fields compare exactly; float feature rows
get a tolerance, because Python accumulates per-base statistics in float64 and
casts while Rust accumulates in float32 — real noise of ~1e-7, against
divergences that have all moved values by 0.5% or more.

**Refinement is on by default and both backends must drive escapepod
identically.** `signal_refine.py` and `rust/src/inference_pipeline/refinement.rs`
each pass their own settings to escapepod's `refine_signal_map`, and escapepod's
Python binding defaults `dwell_target` to 4.0 while leech wants 0.0 (resolve it
from the read's own median dwell). Passing that explicitly on both sides is not
optional — leaving the default in place on the Python side made every dwell and
every level-derived feature differ between backends for four releases (#193).
`--scale-iters -1` means "no refinement" on both; do not clamp it to 0, which
escapepod reads as one DP pass.

Since escapepod v0.15.0 the settings themselves are escapepod's
`RefineSettings::move_table_refinement` preset (escapepod-rs#257), so there is
no longer a literal here to drift. **Do not hand-build `RefineSettings` in this
crate again** — that is what the preset exists to prevent. Four primitives now
come from upstream rather than being held locally: the preset, the POD5 reader
cache (`escapepod_signal::cached_reader`), per-base statistics
(`features::span_stats`, configured `SpanFill::Zero` / `SpanBounds::Clamp` /
`MedianConvention::SortPartialCmp`), and the move-table and CIGAR coordinate
mapping (`escapepod_signal::mapping`).

### The corpus is written and merged in exactly one place each

**`_merge_arrays_by_split` is the only merge.** All four public entry points
(`merge_and_split_chunks`, `merge_and_kfold_split_chunks`,
`merge_and_split_multiclass`, `merge_and_kfold_split_multiclass`) go through it.
One of them used to carry an inline copy "to avoid re-reading per fold", and
that copy never learned about the CSR `seq_to_sig_values`/`seq_to_sig_offsets`
pair — it masked a flat, one-row-per-*entry* array with the per-chunk boolean
mask, so `data merge --k-fold` on multiclass inputs raised `IndexError` on every
corpus written after the CSR change. **Do not inline a second merge.** The
cache it was buying cost 3.95x the corpus in RAM to save a sequential re-read
this storage does at 724 MB/s.

**Inputs must agree on their member set.** A file missing `focus_signal_pos` (or
the residual channel) contributes no rows for that member and full rows for
every other one, and the result is a corpus whose columns are off by the length
of the short file — silently, in one input order, and as an `IndexError` on load
in the other. The merge validates this up front and asserts every output member
has one row per chunk before `np.savez`.

**`prepare` never holds the corpus as a list.** `ChunkNpzWriter`/`ChunkSpool`
take batches as they are extracted and spool each member to disk; peak is ~0.25x
the payload rather than ~2.5x. The `.npz` it assembles is byte-compatible with
`save_chunks`, which stays for callers that already have a list. The corpus is
written twice (spill, then `.npz`) — that disk cost is the trade, and both
prepare paths log it at the start of a run.

**`None` is written as `""`, never as the string "None".** `chunk.get(field, "")`
returns the default only when the key is *missing*; a key present with the value
`None` — what `--label` gives when it is not passed, and what `load_chunks`
gives for an absent group — reached `np.array(..., dtype=str)` and became the
four-character string. That made a save/load round trip non-idempotent, so a
merge renamed empty source groups to a group *called* "None" that
`--balance-groups` then weighted like a real one. `_text()` in
`chunking/serialization.py` is the one place this is enforced; both write paths
share it through `iter_chunk_columns`.

**Do not use `index_select` / `Tensor.copy_` in the loader.** They hand the
memcpy to `at::parallel_for`. On an idle node that is marginally faster than
numpy; on a loaded one it measured 115 ms/batch against 0.25 ms — 460x worse —
and a training loop contends with its own forward/backward for that pool, while
grid search runs N such processes. `LeechDataset`'s block fill and
`__getitems__` gather go through numpy for this reason.

### CTC-CRF: a second task, and five rules that do not announce themselves

`leech.crf` maps a signal window to a *sequence* rather than a label — a CRF
over `n_base ** state_len` states whose Viterbi traceback emits one base per
move. The formulation is ONT's, introduced in bonito; the architecture is
SeqTagger's published parameters. Ported here from `escapepod_models.crf`
(rnabioco/escapepod-models#40), which now imports it from leech.

**The equivalence checks that prove this code correct live in escapepod-models**
(`scripts/ldx/analysis/verify_crf_*.py`), because they run the reference
implementation as the oracle and leech carries no dependency on it. Keep them
there, and do not "simplify" them by dropping that comparison — it is the
evidence.

**`import leech.crf` must pull only torch and numpy, and only on demand.** Not
pysam, polars, escapepod, sklearn or click — ever. escapepod-models installs
leech `--no-deps` into a conda-forge pixi environment precisely so its solver
never has to reconcile leech's POD5/BAM stack against conda's pytorch, and the
CRF path is all it needs; one convenience import at the top of `encoder.py`
turns that into an install that breaks at first use. The config path is eager
and everything else is lazy (PEP 562, as in `leech.models`), so
`from leech.crf import DEFAULT_CONFIG` costs no torch import: escapepod-models'
`ldxlib` exposes it as a module constant and is imported by two dozen scripts
that want only edit distances and panel lookups. `tests/test_crf_package.py`
fails if either property regresses.

**The model cannot emit the first `state_len` bases of its target.** They fix
the initial state and nothing else, so a `target_len` target decodes to
`target_len - state_len` bases *at any window width* — widening the signal
window recovers exactly nothing. Match decodes against `target[state_len:]`;
matching the full-length target calls the same sequence but inflates every edit
distance and compresses the confidence margin that ranking depends on. Size
targets so the sacrificial bases come from a constant prefix.

**Blank is entry 0 of each state group, and the score width is 1280.**
`score_index = state * (n_base + 1) + label`, `label == 0` meaning stay. 1024 is
the *linear layer's* width (`n_base ** (state_len + 1)`); the blank is spliced in
per state afterwards. This is the layout `escapepod-demux`'s Rust decoder
assumes — move the blank to the end of each group and every shape still lines
up while every call is wrong. Output is also **time-major** `(T, N, n_score)`,
the opposite of the boundary CNN's batch-major `[B, 2, L]` in the same stack.

**The loss runs in fp32, outside autocast, and `_UNREACHABLE` is -1e30, not
-inf.** The lattice scan accumulates over `chunk // stride` timesteps and fp16
loses the tail of that sum; autocast the encoder, where the matmuls are, and
cast back before the loss. And `logaddexp(-inf, -inf)` is `-inf` forward but
differentiates to `nan`, which poisons every upstream gradient — the loss looks
perfect and training silently does nothing. A large finite floor underflows to
zero weight against any real path while keeping the backward pass finite.

**The decode is two passes and both are load-bearing.** Log-semiring
forward/backward for per-timestep edge posteriors, then max-semiring over
`log(post + 1e-8)` for the argmax edge. A one-pass Viterbi over the raw encoder
scores is a different and worse decode, and is the obvious thing to simplify
away. The floor goes on the probability, not the log, so it cannot be folded
into the softmax.

**The manifest is the seam, and vocabulary stays on the far side of it.**
`leech.crf.manifest` takes one table — `read_id, pod5, anchor_end, target` plus
optional `label`/`group`/`batch`/`quality_score`/`quality_margin`/`split` — and
nothing about where those facts came from. Which reads belong to which barcode,
which flowcell, how a label's trustworthiness was scored: that is the producing
project's business. `target` is the **resolved sequence**; a class name may ride
along in `label` for reporting, but nothing here looks one up. escapepod-models'
extractor took a `--panel` argument purely to turn a class name into a target
string, and that one thread is what kept it tied to a single assay.

Two rules the manifest exists to enforce, both silent when broken:

- **Label quality travels as numbers, never as a `keep` boolean.** The gate is
  applied at training time so it stays sweepable — gating the ldx labels moved
  accuracy from 0.875 to 0.97, and a boolean decided at extraction would mean
  re-cutting an 8 GB corpus per threshold. `quality_coverage()` is there because
  an *unscored* read cannot pass a gate and is therefore dropped without a word:
  a partially scored table once cut a corpus from 56% to 13.5% of its reads,
  non-randomly.
- **`anchor_end` and `target` are coupled.** `check_geometry` refuses a window
  too short to hold its target rather than warning, because a short window
  trains, converges, and quietly discriminates on fewer bases than designed.
  Pass a *measured* `samples_per_base` — leech has dwell times — not a constant.

**Corpus planning is separate from extraction, and that is where the rules
are.** `crf.plan_corpus` decides which reads and which split without touching a
POD5; `crf.build_corpus` streams their signal to a memmap. Four rules, all
silent when broken: a cap only caps if every class can reach it (`per_group="auto"`
= the rarest class's *trainable* depth, test fraction reserved first); the split
is carved before capping and ranked per class **globally across batches**, since
per-`(batch, class)` ranking multiplies the cap by the batch count whenever
classes are crossed with batch; batches are **interleaved, not concatenated**,
or the whole test set comes from whichever batch sorts first and the headline
number measures batch; and sharding happens **after** planning, so each shard
keeps its share of one global split. Validated against the production ldx
manifest: 1.14M rows plan to 391,174 reads, the same count that repo's extractor
reports, with train balanced exactly across all 16 groups.

The analytic forward-backward in `_analytic.py` is the loss path; the plain
scans in `loss.py` are the readable reference the tests check it against, and
the Triton kernels check against those. Keep all three — the fallback chain is
what makes a wrong kernel visible.

### ONNX export: one exporter, and what a graph cannot carry

`leech.onnx_export` serves both the classifier arms (`leech model export
--format onnx`) and the CRF encoder (`leech.crf.export`). It exists because
`torch.export` makes a model loadable by anything with PyTorch and by nothing
else — escapepod-rs consumes ONNX and could not load a leech model at all.

**Always the dynamo exporter, opset 18.** `dynamo=False` is the obvious first
attempt and fails on these architectures with `Unsupported: ONNX export of
operator adaptive_avg_pool1d`. That is an exporter limitation, not a model
problem, but the message reads like one; `tests/test_onnx_export.py` pins the
working path so a regression to the legacy exporter says so.

**Every export writes a contract beside the graph**, because two things a
consumer needs are invisible in it: which input is which (arity and channel
counts are visible, roles are not), and what the output means — leech
classifiers emit a *single BCE logit*, not a two-class softmax, and reading it
as the latter makes every call wrong without erroring. The CRF's contract adds
standardisation (in neither the config nor the checkpoint) and the emitted
references (`target[state_len:]`, computed from the `state_len` the encoder
declares so no caller can pass full-length targets by hand).

**The `signal_kmer` sequence input is not in the graph, and that is deliberate.**
It is `encode_signal_kmer` output — a scatter of the one-hot k-mer context along
the signal axis, built in the *dataset* from the base-to-signal map. Two options
were on the table: bake it into the graph as an ONNX prefix, or keep it a
documented call. **It stays a documented call**, because the scatter needs the
base-to-signal map, which comes from the move table — an ONNX prefix would still
have to take that map as an input, so it moves where the scatter happens without
removing the consumer's obligation, and it would bake `signal_kmer_context` into
the graph. The contract names the function and its parameters rather than
leaving them to be re-derived — the same choice escapepod-rs's charging bundle
makes by carrying its recipe in `metadata.json`.

**But "one definition" is not yet true, and the fix is upstream.** `leech-core`
ships the encoder, and it is tempting to say a consumer should just call it —
except `leech_core` is `crate-type = ["cdylib"]`, a Python extension module, so
escapepod-rs cannot link it. The primitive itself
(`rust/src/encoding.rs::encode_signal_kmer_inner`) is pure, dependency-free and
carries no model vocabulary: sequence ints, a base-to-signal map, a signal
length and a k-mer context in, a `(4 * kmer_len, signal_len)` scatter out. That
is an `escapepod-signal` primitive by every rule this stack already applies —
and escapepod-signal owns `mapping`, which *produces* the map this consumes, so
today the producer is upstream and the consumer is downstream. Moving it is
rnabioco/escapepod-rs#271. Until it does,
any Rust consumer has to transcribe it, which is a second definition, and the
mitigation is a cross-language golden of the kind the CRF decode and the
charging features already have.

**Verification crosses the serialization boundary.** `verify_onnx` runs
onnxruntime against torch and returns the max absolute difference, which is what
an in-process assert cannot check. The CRF encoder measures 3.58e-07 against a
float32 eps of 1.19e-07; the production classifier arms measured 4.77e-07 and
1.19e-06 (rnabioco/leech#217). It also pins onnxruntime's thread pool, whose
default affinity call fails inside any cgroup-restricted allocation and floods
stderr.

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

**Model bundling** (`bundling.py`)
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
│   ├── benchmark.py # Training-step benchmark handler
│   ├── release_model.py  # model release/list/fetch handlers
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
├── release/         # `leech model release/list/fetch` (GitHub Releases)
│   ├── spec.py          # ReleaseSpec: YAML model-release specification
│   ├── notes.py         # Release-note rendering
│   └── github.py        # gh CLI wrapper
├── crf/             # CTC-CRF sequence models (torch + numpy ONLY)
│   ├── encoder.py       # CrfEncoder: signal -> transition scores
│   ├── loss.py          # CtcCrfLoss + the readable reference scans
│   ├── _analytic.py     # Analytic forward-backward (autograd Functions)
│   ├── _triton.py       # Optional CUDA lattice kernels
│   ├── decode.py        # Two-pass Viterbi -> sequences
│   ├── config.py        # Architecture TOML reader
│   ├── _flags.py        # LEECH_* / ESCAPEPOD_* switches
│   └── configs/         # Packaged crf_ctc.toml (the shipped geometry)
├── features.py      # MoveTable, dwell times, signal levels, normalization
├── dataset.py       # PyTorch Dataset classes, collate_fn, DataLoader sizing
├── training.py      # Training loop with Trainer class
├── evaluation.py    # Model evaluation and testing
├── inference/       # Inference engine (multi-model, Remora compat)
│   ├── single.py        # Single-model inference
│   ├── bundle.py        # Bundled multi-model inference
│   ├── aggregation.py   # Pairwise / one-vs-all vote aggregation
│   └── helpers.py       # Shared inference helpers
├── gridsearch.py    # Grid search for chunk context optimization
├── bundling.py      # Bundle models into one versioned .pt
├── model_loading.py # Checkpoint loading, seeding
├── model_export.py  # TorchScript / torch.export packaging
├── metrics.py       # Metric computation, printing, serialization
├── profiling.py     # Per-phase step timing for `model benchmark`
├── losses.py        # Loss function implementations (BCE, focal, cross-entropy)
├── calibration.py   # Post-hoc Platt scaling for model calibration
├── signal_refine.py # Signal map refinement via kmer level tables
├── _rust_accel.py   # Rust acceleration wrapper for vectorized operations
├── configs.py       # Dataclass-based configuration management
├── constants.py     # Project-wide constants and defaults
├── logging_config.py  # Logging setup
├── data/            # Packaged kmer level table (rna004_9mer_levels_v1)
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

The codebase is feature-complete (v0.7.0):
- ✓ Feature extraction with dwell offset tuning and signal map refinement
- ✓ 29 model architectures: ConvLSTM (Base/Dwell × BN/GN/LN/Attn), TCN (Dwell/DwellGN/DwellLN/DwellResidual/DwellResidualGN/DwellResidualLN/DwellResidualMotor/DwellResidualDwellAttn/DwellSplitResidual/DwellSplitResidualLN), Transformer (Dwell/DwellResidual), ResNet, ConvOnly, SignalCNN
- ✓ Config-driven model layer: bonito-style layer registry (`models/nn.py`) + TOML architecture declarations (`models/configs/`)
- ✓ CLI organized into 4 command groups: `data` (prepare, merge), `model` (train, train-crf, optimize, bundle, bundle-info, calibrate, export), `eval` (test, compare, importance, ablation), `predict`
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
- ✓ Field-by-field backend parity enforced in CI (`tests/test_backend_parity.py`)
- ✓ One extraction-sequence rule (`chunking.extraction_sequence`) shared by
  prepare and predict; `base_justify`, `dwell_offset` and
  `require_query_mapping` reach the Rust inference path
- ✓ One DataLoader worker rule (`dataset.resolve_dataloader_workers`) for
  training, validation and eval: auto on GPU (capped by the job's CPU
  allocation), serial on CPU, never workers inside a daemonic pool worker;
  `eval test` takes `--num-workers`
- ✓ CTC-CRF sequence models (`leech.crf`): encoder, training objective with an
  analytic forward-backward, optional Triton lattice kernels, and the two-pass
  Viterbi decode — ported from escapepod-models, torch + numpy only. Plus the
  manifest seam, the corpus builder (`plan_corpus`/`build_corpus`), the trainer
  (`CrfTrainer`, `leech model train-crf`) and the ONNX export. The metrics/eval
  half is not here yet.
- ✓ ONNX export for the classifier arms and the CRF encoder
  (`leech model export --format onnx`, `leech.crf.export`), dynamo exporter at
  opset 18, each with a contract sidecar and a round-trip check against torch

All core functionality is implemented and ready for use.

## Git Conventions

- Do not add `Co-Authored-By` lines to commit messages.
