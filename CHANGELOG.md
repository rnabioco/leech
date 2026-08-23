# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The Rust `data prepare` backend stamped the wrong feature window onto every
  chunk when `--feature-start 0` was requested (#189). `_prepare_batch_rust`
  resolved the value it stored with `config.chunk.feature_start or -5`, and `0`
  — features beginning *at* the focus base, the right-only window used for tRNA
  3' ends — is falsy, so the chunk recorded `feature_start=-5`.

  The extraction itself was correct: Rust receives the window and cuts the
  requested bases, so the arrays have the right shape and the right contents.
  What was wrong is the number the corpus reports about them, and that number is
  load-bearing — `dataset.py` slices the k-mer window out of the feature array
  at `(-kmer_context) - feature_start`, and `model train` copies it into the
  model config that `predict` later reads. A corpus prepared this way trains and
  infers on a window shifted by `kmer_context` bases, silently, with no shape
  error anywhere. **If you prepared with the Rust backend and an explicit
  `--feature-start 0`, re-prepare** (or rewrite `feature_starts` in the `.npz`).

  Only a falsy start triggered it; `--feature-start -15` was always stored
  correctly, which is why it went unnoticed. Both backends now resolve the
  window through one `resolve_feature_window`, and `TestFeatureWindowParity`
  holds them to the same stored metadata and the same array widths across four
  windows.

### Changed

- `data prepare` logs the resolved feature window (`[+0, +20] relative to the
  focus base, 21 bases wide`) and records `feature_start_resolved`,
  `feature_end_resolved`, and `feature_width` in `prepare_config.json`. The
  requested values are optional and default to the k-mer window, so "asked for
  21 bases, got 11" was previously visible only by diffing two corpora field by
  field (#189). `data merge` now warns when inputs disagree on the resolved
  window.

## [0.6.1] - 2026-08-23

Two silent parity bugs between the `data prepare` backends. **If you built a
corpus with the Rust backend on 0.6.0 or earlier, re-prepare it** — it is
missing ~1% of reads, non-randomly, and if you trained with
`--seq-encoding signal_kmer` the sequence context differed from what the Python
backend would have produced.

### Fixed

- The Rust `data prepare` backend silently dropped ~1% of reads that the Python
  backend kept — non-randomly, biased toward supplementary-aligned and
  indel-heavy reads (#185). On a 1,052,751-read production corpus the two
  backends differed by 9,772 reads, Rust's output a strict subset of Python's,
  with nothing logged and exit 0.

  `extract_training_chunks_from_read` skipped any focus base whose k-mer window
  ran off either end of the sequence, where `LeechRead.get_chunk` pads it with
  `N`. Under `--anchor reference` the sequence is the *aligned reference slice*,
  so a read whose alignment stops within `--kmer-context` (default 5) of the
  motif lost its only chunk — exactly the population produced by supplementary
  alignments and by the elevated indels that `--no-require-query-mapping`
  exists to keep. Rust now pads instead of skipping, and drops a focus base only
  when it has no signal boundaries, which is the Python rule.

  The same guard sat in `inference.rs`, so the same reads got no prediction on
  the Rust predict backend. Fixed alongside.

- `seq_to_sig_map` and `sequence_with_kmer_context` — the two chunk fields that
  drive `--seq-encoding signal_kmer` — never agreed between the prepare
  backends (#186). Not at edges: on every chunk, at every focus position.
  `LeechRead.get_chunk` derives them from the **signal** window, locating bases
  with two `searchsorted` calls over the read's map, and snaps the partially
  overlapping first and last bases to the window edges; Rust derived them from
  the **k-mer** window, which spans a different number of bases, and truncated
  rather than `N`-padding. Rust now uses the Python definition — the one every
  trained `signal_kmer` model has seen — via a shared
  `chunk_signal_kmer_inputs`, in both `training.rs` and `inference.rs`. The
  `(4 * kmer_len, signal_len)` encodings the model consumes are now identical
  between backends.

- `LeechRead.get_chunk` bounded the focus base on the sequence length while
  indexing `seq_to_sig_map`, which is shorter whenever `compute_ref_to_signal`
  strips trailing non-match CIGAR ops. The resulting `IndexError` made the
  prepare workers drop the whole read rather than the one chunk. It now bounds
  on `num_mapped_bases`, the same quantity Rust uses.

### Changed

- Both prepare backends now go through one `find_focus_bases`, instead of the
  Rust path carrying a second copy of the rule in `parallel.py`. The copies had
  drifted: the Rust one searched the *basecall* under `--anchor reference`,
  where positions are reference-relative, and its all-bases fallback ranged over
  the basecall's length rather than the reference slice's.

- `data prepare` logs a read yield — reads that produced chunks over reads seen
  — on both backends, and `stats["reads_with_motif"]` counts reads rather than
  chunks, which is what it counts in the sequential path. A backend that yields
  less than the other now says so in the log.

### Documentation

- Document the read yield line in the data preparation guide, and add a
  troubleshooting entry for a run that produces fewer chunks than a previous one
  or than the other backend.

## [0.6.0] - 2026-08-23

### Fixed

- `data prepare` on the Rust path was 10-80x slower than the Python
  multiprocessing fallback on a large POD5 (#176), so installing `leech_core`
  made the step slower with nothing in the log to say so. Three compounding
  causes, all fixed:
  - **The POD5 reader was opened once per batch.** `escapepod_signal::Reader`
    caches its read-id index in a `OnceLock` on itself, so a fresh reader threw
    the index away every time, and `reads_by_ids` fell back to a
    single-threaded scan of the whole reads table — all 22 columns, stopping
    only once every target was found. Reads arrive in BAM order, which is
    unrelated to POD5 storage order, so that scan ran to end-of-file on each
    batch. On a 145 GB POD5 on BeeGFS a single 1000-read batch had not finished
    after 13.5 minutes, spending it in uninterruptible sleep in
    `folio_wait_bit_common` at 0.006 of a core and ~119 major faults/s. The new
    `rust/src/pod5_cache.rs` opens each POD5 once per process and warms its
    index once, in memory, from a read_id-only projected scan; it writes no
    `.p5s` sidecar and needs none. All four `Reader::open` sites now use it.
  - **Phase 1 held the GIL.** `py.detach` wrapped only per-read processing, so
    POD5 I/O serialized every caller thread and no caller could overlap
    batches. Phase 1 now runs GIL-free in both `training.rs` and `inference.rs`.
  - **Batches were dispatched from a serial `for` loop.** `--workers` selected
    a rayon pool size but nothing dispatched batches concurrently, leaving one
    POD5 read outstanding at a time while the Python fallback had `--workers` of
    them. Both backends now expose the same `(n_reads, chunks)` iterator, with
    `num_workers` batches in flight — threads for Rust, processes for Python —
    and a bounded window so the BAM is not pulled into memory up front.
    `tests/test_prepare_dispatch.py` fails if the loop comes back.

- `--signal-norm` is no longer silently ignored when the Rust extraction path
  is active. `rust/src/inference_pipeline/processing.rs` always applies
  median-MAD — its `PipelineConfig` carries no normalization field — but
  neither the training gate (`preparation/parallel.py`) nor the inference gate
  (`inference/helpers.py`) checked the method, so `--signal-norm zscore` with
  `leech_core` installed produced median-MAD chunks instead. Measured on the
  tRNA fixtures, every chunk differed, by up to 1.77 in normalized-signal
  units. Because `predict` reads `signal_norm` back out of the model config,
  this also meant a model trained with a non-default norm was served under a
  different one. Both gates now consult `rust_supports_norm_method()`:
  `--backend auto` falls back to Python with a warning, and `--backend rust`
  raises rather than mis-normalizing. `median_mad` runs are unaffected.
- `recover_softclip_signal` is likewise no longer silently dropped on the Rust
  path. Recovery fills chunk-window samples outside the aligned region from the
  full pre-crop signal, which the Rust `ProcessedRead` discards when it crops —
  so enabling the flag with `leech_core` active produced the zero-padding the
  flag exists to avoid. `prepare` warned about this for `--workers > 1` but
  never gated on it, and the inference path did not warn at all. Both now route
  to Python automatically (`--backend rust` raises). The prepare warning is
  downgraded to info and no longer tells users to pass `--workers 1`, since
  the fallback is now automatic; enabling the flag costs throughput, not
  correctness.

### Changed

- escapepod bumped to **v0.14.0** — `rust/Cargo.toml` returns to a tag pin
  (from the `1c55668f` rev that took the unreleased kmer-level primitives) and
  the `escapepod` floor moves to `>=0.14.0`. v0.14.0 fixes the upstream half of
  #176 (escapepod-rs#251): `reads_by_ids`, `find_signal_rows_by_ids` and
  `find_signal_rows_with_calibration_by_ids` used to take the indexed path only
  when a `.p5s` sidecar existed on disk, and otherwise ran a full 22-column
  scan of the reads table that never built the index it declined — so even a
  cached reader re-scanned on every call. The scan variants are gone and all
  three route through `read_index()`. `escapepod-pod5` also gained `tracing`,
  so an index build now reports that it is happening and what it cost.
  `rust/src/pod5_cache.rs` keeps its reader cache, which is what makes the
  index survive across batches, but its index warm-up is now belt-and-braces
  rather than load-bearing; its docs no longer describe the removed scan
  fallback as current behaviour.
- Rust prepare-backend selection moved into
  `preparation.parallel.rust_prepare_unsupported_reason()`, a pure function
  returning the reason Rust cannot serve a given `PrepareConfig` (or `None`).
  It collects the pre-existing `focus_map` bypass alongside the two new
  capability checks, so the rules can be unit-tested without running a
  preparation pass — the previous inline gate could only be exercised by
  spawning the multiprocessing pool.

### Added

- `--no-require-query-mapping` on `data prepare`, for modifications that
  mis-call the motif they are measured at (#175). By default a motif found in
  the reference is accepted only if it also maps cleanly to query coordinates
  through the CIGAR. Under `--anchor reference` that mapping is used *only* to
  accept or reject — the returned coordinate is reference-relative and the query
  coordinates are discarded — so the check is a quality gate, not a requirement
  of window placement, and reads failing it can be kept without moving any
  chunk. This matters because the gate selects on the label: on aminoacyl-tRNA
  the adduct mis-calls the CCA junction, dropping 28% of charged reads against
  6% of uncharged, before any model sees the data. Only valid with
  `--anchor reference`; combining it with `--anchor basecall` raises, since
  there the returned coordinate *is* the query start.
- `--emit-scores PATH` on `leech eval test` writes the per-chunk `read_ids`,
  `labels` and `probs` behind the metrics to an `.npz` (#181). `evaluate_model`
  already computed a probability for every chunk and then dropped it, leaving
  the confusion matrix — a summary at ONE threshold — as the only artifact.
  That made AUPRC for the minority class, per-group error breakdowns, paired
  model comparison, calibration, and any other operating point unanswerable
  without re-running inference. Off by default. The join to read ids is
  positional and is checked, not trusted: a length mismatch raises rather than
  writing a well-formed file full of misattributed scores.
- `compute_metrics` now attaches a `threshold_sweep` reporting the best
  operating point rather than only the caller's implied 0.5 (#180). Three
  points — `at_youden`, `at_mcc`, `at_f1` — each with threshold, TPR, FPR, MCC,
  F1, Youden's J and called-positive fraction, plus `prevalence` beside them.
  The default threshold is the wrong choice whenever training and evaluation
  see different class ratios, which `--oversample-minority` guarantees: on a
  tRNA charging corpus at 13.0% positive, the same predictions scored 0.6066
  precision at observed prevalence against 0.9114 at 50/50, so the model was
  being blamed for the class ratio. `at_youden` is prevalence-invariant and is
  the right default when the deployment ratio is unknown or is itself what is
  being measured.
- `data prepare`'s startup line now names the dispatch, not just the backend:
  `[Rust (rayon), 32 batches in flight via threads]`. `leech_core` is a separate
  package from `leech`, so `maturin develop` can leave a freshly built extension
  beside a stale `leech` — new Rust, old serial driver — and the only symptom is
  being slow. `[Rust (rayon)]` alone predates #178 and does not distinguish the
  two; a build without "batches in flight" is the stale pairing.
- `data prepare` logs achieved reads/s on every progress line and in the final
  summary, on both backends, and names the backend in each. The #176 regression
  was invisible in the logs until the allocation ran out; a rate to compare
  against would have surfaced it in the first minute.
- Rust/Python chunk parity is now tested for `anchor="reference"`, not just
  `anchor="basecall"`. The test helper already accepted the parameter but no
  call site ever passed it, leaving the reference-anchored path — the default,
  and the one that crops the signal to the aligned region and runs
  `compute_ref_to_signal` — with no parity coverage. The two backends agree
  exactly on signal, sequence and dwells there, and to float32 rounding on
  features; the assertions now also cover features, which are what the model
  actually consumes.

## [0.5.0] - 2026-08-17

### Added

- Config-driven model layer (bonito-style). `leech/models/nn.py` provides a
  layer registry with `register` / `to_dict` / `from_dict` and the composition
  primitives `Serial`, `Stack`, `Parallel` (nestable fan-out + concat) and
  `Graph` (flat named dataflow for leech's multi-input, multi-branch models).
  Architectures are declared in TOML under `leech/models/configs/`.
- `Graph` accepts an optional `build_order`: node declaration order still
  drives execution, but layers are constructed and installed in `build_order`.
  This is what lets a config reproduce a class whose module-construction order
  was not its dataflow order (a subclass appending layers after its parent's
  head), keeping both `state_dict()` key order and seeded initialization.
- Layer registry entries for the TCN family: `tcn`, `temporalblock`,
  `normmlphead`, `normproj`, `slice`, `signalchannel` and `rangemeanpool`.
- `TCNDwellResidualMotor`, `TCNDwellResidualLNMotor`,
  `TCNDwellResidualDwellAttn` and `TCNDwellResidualLNDwellAttn` are now
  registered models (29 total).
- `SignalCNN`, a signal-only classifier for tasks where only the raw signal
  carries the label (e.g. barcode demultiplexing from adapter signal). It
  ignores the sequence and feature inputs in `forward()` while still honoring
  the standard batch contract, so it reuses the existing
  Trainer / `collate_fn` / `ModelInferenceWrapper` stack rather than
  duplicating it. Adds `SignalDataset` and
  `compute_class_weights_from_labels`.
- Model release workflow: `leech model release`, `leech model list` and
  `leech model fetch` publish trained bundles to GitHub Releases, browse
  what is available, and download them. Release metadata comes from a YAML
  spec that merges editorial fields (description, organism, metrics) with
  technical bundle metadata (architecture, pairs) into generated release
  notes. Tag convention: `model-{name}-v{version}`. Implemented in the new
  `leech/release/` subpackage over the `gh` CLI, so no new Python
  dependency.
- Training-loop features ported from bonito, all off by default:
  `--grad-accum-split N` splits a batch into N sub-batches and steps the
  optimizer once (larger effective batch without the memory);
  `--quantile-grad-clip` clips at 2x the median of the last 100 gradient norms
  instead of a fixed threshold; `--save-optim-every N` writes optimizer state
  only every N epochs while still writing weights on every save.
- `--confound` now accepts an inline `source:mapping[:table]` spec (e.g.
  `source_group:lookup:groups.json`) in addition to the built-in aliases, and
  a new `--confound-config` reads a JSON/YAML confound definition. Both
  resolve to one canonical token at the CLI boundary.

### Changed

- The k-mer level mechanics moved down to escapepod-signal
  (rnabioco/escapepod-rs#204/#205), which is now their canonical home:
  `leech_core`'s `extract_levels` delegates there (same f64 numerics), the
  dead `rough_rescale` binding (per-base-mean variant, no callers) is
  removed, and a new `rough_rescale_quantile` binding delegates the quantile
  rough rescale — `leech.signal_refine.rough_rescale_quantile` now dispatches
  to it for float32 inputs when `leech_core` is available. escapepod pins the
  parity with leech's NumPy implementations bit-for-bit in golden-vector
  tests, so leech and `escpod` can no longer drift on the k-mer residual's
  definition. Pure-Python fallbacks are unchanged. escapepod-signal is
  pinned by rev until the next escapepod release.
- The ConvLSTM family (10 registry names) is now declared in
  `models/configs/conv_lstm.toml` and `conv_lstm_attn.toml` instead of the
  hand-written `models/conv_lstm.py` / `conv_lstm_attn.py`, which were removed.
- The TCN family (12 registry names) is now declared in
  `models/configs/tcn_dwell.toml`, `tcn_dwell_residual.toml`,
  `tcn_dwell_split_residual.toml`, `tcn_dwell_residual_motor.toml` and
  `tcn_dwell_residual_dwell_attn.toml`. The five hand-written modules
  (`models/tcn_dwell*.py`) were removed; their reusable `TemporalBlock` and
  `TCN` blocks moved to `models/components.py`. `state_dict()` keys and their
  order, seeded initialization, and forward outputs are unchanged — existing
  checkpoints and bundles load as before. (The two `*Motor` names are the one
  exception to seeded-init parity: their old class built a classifier head and
  immediately discarded it, consuming random numbers the graph does not. Keys,
  key order and outputs still match, and those names were never in the registry
  before this release, so no checkpoint depends on it.)
- `kind = "class"` configs are gone; `kind = "graph"` is now the only config
  kind, so there is a single config semantics rather than two.
- Renamed `leech.features.normalize_signal` to `normalize_read_signal`. The old
  name collided with `escapepod.normalize_signal`, which is a *different*
  transform (int16 input, no 1.4826 scale factor) that would silently rescale
  every signal in the pipeline if swapped in by name.
- `median_mad` normalization now returns `float32` rather than `float64`,
  matching the Rust path. Values agree with the previous output to f32 precision
  (~7e-8 relative).
- The `ac` BAM tag now always carries the winning class probability. It
  previously carried `1.0 - conf` for reads filtered by `--min-confidence` /
  `--min-margin`, so a filtered read at `max_prob=0.4` reported `ac=0.6` and
  read as *more* confident than a read that passed at 0.5. The
  below-threshold sentinel in the `aa` tag is now the only signal that a call
  was filtered, and it moved to `constants.BELOW_THRESHOLD_LABEL`.
- Bundles now record `pair_labels` (pair → `[negative, positive]`), derived
  from each model's own `label_map`. `resolve_pair_labels()` prefers that map
  and falls back — warning once — to the old `pair.split("_", 1)` for bundles
  built before the field existed.
- The adversarial (gradient-reversal) training path is now config-driven. A
  new `confounds.py` describes a confound as `source` (which chunk field) ×
  `mapping` (`identity` or `lookup`), replacing the hardcoded `disc_base` /
  `trna_id` branch in `training.py` and the two parallel maps in `dataset.py`.
  `--confound disc_base` and `--confound trna_id` behave exactly as before.
- The model registry is now a torch-free `_MODEL_SPECS` table
  (name → submodule, class). Model classes — and therefore torch — load only
  on actual access, so `leech model train -h` no longer pays a ~10s torch
  import just to render its `--model` choices.
- Training now uses fused SDPA attention (`need_weights=False` at every
  `nn.MultiheadAttention` call site, which is what lets PyTorch 2.x dispatch
  to the flash / memory-efficient kernel), fused AdamW on CUDA, and TF32
  matmuls (`torch.set_float32_matmul_precision("high")`, matching the eval
  and inference paths). Behavior-preserving; existing checkpoints load and
  run identically. The training `DataLoader` generator is also seeded from
  the run seed, so shuffle order and per-worker augmentation RNG are
  reproducible regardless of how much global RNG model init consumed.
- `--scheduler cosine` is now a single `LambdaLR` warmup-plus-cosine schedule
  that owns its own warmup, replacing a manual linear warmup bolted onto
  `CosineAnnealingLR`. The LR now holds at the `1e-6` floor past the epoch
  budget instead of cycling back up. `reduce_on_plateau` is unchanged.
- Resuming from a checkpoint with no optimizer state now warns and continues
  with a fresh optimizer instead of raising `KeyError`.
- `escapepod` moved from the optional `pod5` extra into required dependencies —
  `leech.io.pod5_reader` and `leech.features` both import it unconditionally, so
  a base install previously failed on `import leech.io`. The `pod5` extra is
  retained as a no-op alias.
- Dropped the `escapepod-rs` git submodule. Now that the repository is public,
  `escapepod-signal` is a tag-pinned git dependency in `rust/Cargo.toml` and the
  `escapepod` Python package installs from PyPI as a prebuilt wheel, so a plain
  `git clone` + `uv sync` is enough to build leech.
- Bumped `escapepod` to v0.6.3, which fixes a `mad_normalize` abort on constant
  (dead-pore or flat) signal that could kill a long run on a single bad read.
- CI no longer needs the `ESCAPEPOD_PAT` secret, so dependabot now tracks GitHub
  Actions and the `rust/` cargo manifest too.
- The pipeline's `motif` config key is now required rather than defaulting to
  `"CCA"`. A wrong motif does not fail — it silently produces a whole run of
  chunks centered on the wrong base — and the shipped config, `prepare.smk`
  and `diagnose.smk` disagreed on the default. The motif offset default drops
  to 0.

### Removed

- The `diagnose` Snakemake rule and its `all_diagnose` target. Both referenced
  files removed in a60eb09 (`../envs/leech.yaml`,
  `scripts/diagnose_signal_orientation.py`) and could not run. This also
  retires the `label != "uncharged"` filter, the one place the DAG branched on
  a hardcoded class name.

### Fixed

- Signal-map refinement corrupted every level-derived feature, in two ways.
  `DWELL_TARGET` was hardcoded to 4.0 samples/base while RNA004 at 130 bps and
  4 kHz sits near 31, so the asymmetric penalty treated every base as ~8x too
  long; the target now resolves from the read's own move-table median. And
  rough rescale rewrote the signal with escapepod's affine fit, discarding the
  shared median-MAD normalization the per-base stats, k-mer residuals and
  trained models are calibrated against — the fit is estimated on a chunk
  sitting largely in a constant 3' adapter, where it is weakly identified
  (observed scales from 15 to 1084, frequently negative, i.e. sign-flipping
  the read). Refinement now takes only the refined boundaries. Measured on
  tRNA-Met chunks, per-base level vs the expected 9-mer level went from
  r = -0.03 to r = +0.82, and Met/HPG discrimination recovered from AUC 0.681
  to 0.822.
- Four architectures listed in `ModelInferenceWrapper.FEATURE_MODELS` were
  never added to the model registry, so `get_model()` rejected them while the
  inference path assumed they existed. They are registered now, with explicit
  constructor signatures — previously their `**kwargs` constructors made
  `model_loading._instantiate_model` silently drop the `motor_*` /
  `num_dwell_*` parameters when rebuilding a model from a saved config.
- POD5 read lookups now use escapepod's read-id index. `DatasetReader` was
  constructed but never entered, and entering it is what warms the index — so
  every `reads(selection=...)` call re-scanned the whole reads table, making a
  per-read lookup O(reads-in-file). Affects both the cached module-level reader
  and `POD5Reader`.
- `median_mad` normalization no longer returns an all-`NaN` signal for a
  constant (dead-pore or flat) read. It now delegates to
  `escapepod.mad_normalize`, the same routine `leech_core` already used, so the
  Python and Rust normalization paths agree by construction rather than by two
  parallel implementations.
- Class names containing `_` no longer produce garbage aggregation keys.
  The aggregators recovered class names with `pair.split("_", 1)`, but class
  names are free-form (they come from directory names and `label_map`), so
  `aggregate_one_vs_all` silently inverted the prediction for e.g.
  `notzeta_zeta`.
- The `grid_search` pipeline rule passed `--param-grid` and `--max-epochs`,
  neither of which `leech model optimize` accepts, so it failed at argument
  parsing every time it was scheduled. The rule, its config block and
  `docs/pipeline.md` now match what `optimize` actually searches (signal
  context windows and dwell offset), and a `grid_search_parallel` rule makes
  the existing `--parallel` option reachable.

## [0.4.1] - 2026-07-20

### Documentation

- Mark leech as alpha-quality in the README and docs front page

## [0.4.0] - 2026-07-12

### Added

- `--focus-tsv` for per-read labels and externally-anchored chunk extraction
- POD5 directory (not just a single file) accepted as a preparation source
- `leech model benchmark` for training-step profiling
- Soft-clipped signal recovery at reference-anchored chunk edges
- Kmer table fingerprint captured in model config for provenance

### Changed

- Bumped `escapepod-rs` to v0.6.0
- Delegated Python signal-map refinement to escapepod with a reproducible Theil-Sen seed
- Access POD5 signal via escapepod's `DatasetReader`
- Delegated banded DP and MAD normalization to `escapepod-signal`
- Improved multiclass training stability (memory, optimizer, selection metric, `disc_base` confound)
- Faster preparation: bypass the Rust pipeline and early-skip reads when a focus map is set

### Fixed

- Compute multiclass AUROC per-class to avoid the sum-to-1 trap

### Documentation

- Migrated the docs site to `zensical.toml`

## [0.3.2] - 2026-04-20

### Added

- `trna_id` adversarial confound for full isoacceptor identity debiasing
- Rust-accelerated training chunk extraction with rayon parallelism
- Dwell template features, motor/dwell-attn models, and rough-rescale option
- `--no-compress` flag and streaming BAM for data preparation
- `--split-by` for group-level train/test splits
- `signal_mode`, TCNDwellSplitResidual, and per-channel augmentation
- `--copy-tags` option to copy BAM tags into TSV predict output
- leech version and git commit captured in model config and bundles
- Rust monolithic extraction for bundle inference
- `leech_core` parallelization via `mp.Pool` workers for bundle inference

### Changed

- Bumped `escapepod-rs` to v0.1.3 (SSSE3 SIMD SVB16, audit-driven hot-path optimizations, dynamic versioning)
- Replaced `pod5` Python package with `escapepod` bindings for POD5 I/O
- `escapepod` moved to optional `pod5` extra for pixi compatibility
- Config.json is now the source of truth for model construction
- Replaced hardcoded model allowlists with signature introspection
- Consolidated `models/` from 27 to 13 files
- Split `util.py` into `model_loading`, `model_export`, `bundling`, `metrics`
- Split `inference.py` into `inference/` package
- Split Rust `inference_pipeline.rs` into 8 submodules
- Queue-based extraction pipeline for improved GPU inference throughput

### Fixed

- Module-level `np` shadowing from local numpy imports
- `_parse_and_validate_inputs` return type annotation
- Plumb `dwell_template_table` through calibrate/eval/predict
- Byte-identical Rust↔Python parity for CIGAR ref-to-signal mapping
- Pass `signal_in_channels` to model during calibration
- Resolve remaining clippy warnings in Rust code
- Excluded `vulture_whitelist.py` from ruff linting
- CI submodule checkout with `ESCAPEPOD_PAT`; ruff format compliance
- Benchmark script ruff lint errors

## [0.3.1] - 2026-03-20

### Added

- TSV prediction output (`TsvPredictionWriter`) as alternative to BAM tag output for multiclass models
- escapepod-rs integration for ~10x faster Rust-accelerated inference via POD5 batch reads
- Multiclass temperature scaling calibration with ECE improvement gating
- Adversarial training with gradient reversal layer and confound maps for discriminator base debiasing
- CL regression head for multi-task charging level prediction
- Cross-layer augmentation: time masking, cross-layer shift, per-channel feature noise
- Mega-batch streaming inference with double-buffered GPU pipeline
- `--signal-context` CLI option for `leech data prepare` to set asymmetric signal windows
- `--min-confidence` and `--min-margin` thresholds for `leech predict`
- `--oversample-minority` flag for class imbalance handling
- Auto-read `anchor` and `reference_fasta` from model config at predict time
- `am` (margin) BAM tag in predict output
- `enable_repr_capture()` for `ModelInferenceWrapper` internal activations
- Centralized Rich console with wide fallback for SLURM batch jobs
- TCNDwellResidualGN and TCNDwellResidualLN model variants (22 total architectures)
- Label smoothing and cosine annealing scheduler options
- `py.typed` PEP 561 marker for type checker support
- vulture dead code detection in dev tooling

### Changed

- K-fold merge now caches input files in RAM for faster processing
- Array-level merge without zlib compression for merge step
- Disable DataLoader workers for validation to reduce memory usage
- Checkpoint multiclass models on `val_f1` instead of `val_acc`
- Development status upgraded from Alpha to Beta
- Version string now uses `importlib.metadata` with git hash fallback
- Removed seaborn from notebook extras (plotnine-only policy)
- Bumped leech_core to v0.3.0 and Rust edition 2024

### Fixed

- Store `focus_signal_pos` in chunks for asymmetric `signal_context`
- Enable TF32 matmul precision in inference path
- Training summary reports actual best `val_acc`/F1 instead of last epoch
- Use `tolist()` for multiclass `pos_weight` serialization in config
- Handle k-fold directories in multiclass bundle discovery
- `label_map.json` lookup for k-fold adversarial training
- Use `reads_by_ids()` for O(1) indexed POD5 lookup instead of full scan
- Gate multiclass temperature scaling on ECE improvement
- Route multiclass through parallel inference path
- FASTA index race condition under parallel SLURM jobs
- Label smoothing no longer alters labels used for metrics
- Propagate `pa_mean`, `pa_stdev`, `skip_motif_indels`, and refiner params through config chain
- Align signal map refinement `scale_iters` between prepare and inference
- Auto-read `base_justify` from model config in single-model inference
- Use reference-based motif search in inference
- Signal kmer coordinate adjustment and `reference_fasta` plumbing
- Default `skip_motif_indels` to `False` everywhere

### Performance

- Prefetch pipeline with rayon contention fix in inference
- Sub-batch extraction with async BAM writes for GPU pipelining
- Multi-threaded extraction in sequential inference path
- Optimized inference pipeline for fast prediction

### Removed

- Dead code: `config.py` (replaced by `configs.py`), `calibrate_model_temperature()`, `_is_leech_export()`, `load_predictions_from_bam()`, `prepare_chunks_with_context()`, `extract_disc_bases_from_fasta()`, `handle_branch_contribution()`, `display_logo()`
- Unused constants: `DEFAULT_DWELL_MARGIN_LEFT/RIGHT`, `DEFAULT_NUM_WORKERS`, `DEFAULT_SEQ_ENCODING`, `DEFAULT_MIN_MAPQ`, `DEFAULT_MOTIF`, `DEFAULT_MOTIF_OFFSET`, `DEFAULT_REMORA_NUM_OUT`
- Unused Rust accel imports: `_rs_test_process_read`, `_rs_extract_levels`, `_rs_rough_rescale`
- Unused IO methods: `BAMReader.get_header()`, `BAMReader.count_alignments()`, `POD5Reader.get_signals_batch()`, `POD5Reader.iter_all_reads()`, `ReferenceManager.get_all_sequences()`

## [0.3.0] - 2026-03-14

### Added

- 12 new model architectures (20 total): BatchNorm (BN), Attention, GroupNorm (GN), LayerNorm (LN) variants for ConvLSTM; TCNDwellGN, TCNDwellLN, TCNDwellResidual; TransformerDwellResidual
- Multi-channel signal input (`signal_in_channels`) across all architectures for 2-channel (raw + kmer residual) models
- Kmer residual features: `kmer_expected`, `kmer_residual`, `kmer_residual_abs` from kmer level table lookup
- Signal map refinement rewritten to match Remora's banded Viterbi algorithm with iterative Theil-Sen rescaling
- Signal-level kmer encoding (`signal_kmer`) as default sequence encoding
- Reference-anchored mode (`--anchor reference`) and `pa_scaling` normalization
- Composable config dataclasses (`configs.py`) replacing 6-layer parameter threading
- `feature_start`/`feature_end` parameters replacing confusing `dwell_margin` params
- Dwell cross-attention in TransformerDwell, TCNDwell, ResNetDwell, ConvOnly
- Multi-class classification with confidence-weighted and tournament pairwise aggregation
- K-fold cross-validation with stratified read-level splits (`--k-fold`)
- Balance-groups sampling for equal source group contribution per epoch
- Platt scaling calibration (`leech model calibrate`) with guardrails and best-fold selection
- TorchScript export (`leech model export`) for standalone model deployment
- Rust-accelerated signal statistics via PyO3 (`leech-core` crate)
- `check-rust` CLI command to verify Rust extension availability
- Auto-load bundled kmer table when `--refine-signal-map` has no `--kmer-table`
- `--reference-fasta` support in `leech predict` for reference-anchored bundle inference
- Remora-compatible model variants (ConvLSTMRemora, ConvLSTMRemoraBase)
- Parallel inference with batch POD5 reads in workers
- GitHub Actions release workflow with platform wheel builds (linux x86_64/aarch64, macOS x86_64/arm64)

### Changed

- Default sequence encoding from `base_onehot` to `signal_kmer`
- Default `scale_iters` from 0 to 2 for signal map refinement
- Migrate model export from `torch.jit` to `torch.export` (PyTorch 2+)
- Unify `--anchor` flag across `prepare` and `predict` commands
- Lazy-load `MODEL_REGISTRY` to speed up CLI help (5.3s to 0.6s)
- Lazy imports in `__init__.py` to cut CLI startup from ~9s to ~0.5s
- Rust signal stats 217x faster than NumPy; pre-tensorize dataset; flatten serialization
- Enable `torch.compile` on CPU/GPU, TF32 matmul precision, `inference_mode` in eval
- Optimize test suite runtime from 287s to ~20s
- Speed up `leech eval test` with GPU optimizations and larger batch size
- Refactor CLI handlers into `commands/` subpackage
- Upgrade PyO3 and rust-numpy from 0.23 to 0.28

### Fixed

- Ensure `model_best.pt` always exists after training resume
- Load all checkpoints/bundles to CPU first to avoid device mismatch
- Batch bundle inference for GPU utilization
- Normalize `signal_in_channels` in architecture config comparison
- Return reference-relative coords from ReferenceMotifSearcher when `anchor=reference`
- Prevent `num_out` from leaking into model constructors that don't accept it
- Resolve constructor params for `**kwargs` subclasses in `_instantiate_model`
- Set `num_workers=0` in evaluation DataLoader to reduce memory usage
- Fix `torch.compile` `_orig_mod` prefix in checkpoint loading
- Fix missing `model_best.pt` when resuming completed training
- Fix bundle discovery for k-fold CV and batch size for small datasets
- Fix evaluation to use softmax for cross-entropy (multi-output) models
- Fix stale FASTA index by regenerating .fai before opening
- Fix DataLoader workers in parallel grid search
- Rename ResNetDwell `bn1/bn2` to `norm1/norm2` to match checkpoint migration
- Strip explicit kwargs from model config to prevent duplicates

## [0.2.0] - 2026-03-06

### Added

- Tunable `dwell_offset` hyperparameter for motor-sensor offset correction
- `base_justify` parameter to control signal chunk centering ("start", "center", "end")
- Range syntax (`start:stop:step`) for grid search context parameters
- `best_params.json` output from grid search for Snakemake integration
- Self-documenting run summary for pipeline runs
- Motor-pore offset analysis notebook

### Changed

- Reorganize CLI into workflow-based command groups: `data`, `model`, `eval`, `predict`
- Speed up grid search with CPU optimizations and parallel execution
- Migrate docs from MkDocs + Material to Zensical
- Switch LSF profile to use snakemake-executor-plugin-lsf
- Consolidate guides into 3 professional documentation pages
- Replace mypy with ty for type checking
- Deduplicate `_TRAINING_PARAMS` set and extract `_instantiate_model()` helper in util.py
- Consolidate `FEATURE_MODELS` set (dataset.py now references ModelInferenceWrapper)
- Standardize project acronym across docs, pyproject.toml, and CLAUDE.md

### Fixed

- O(n²) BAM scan in inference.py: build alignment dict in one pass instead of rescanning per read
- Off-by-one error in `to_seq_to_sig_map` to match Remora convention
- Default `min_mapq` filtering that drops most tRNA reads
- TypeError from grid search context params passed to model constructor
- Python badge in docs/index.md now shows 3.12+ (matches requires-python)

### Removed

- Stale `data_prep.py.bak` backup file

## [0.1.0-alpha] - 2025-11-13

Initial alpha release of leech for aa-tRNA-seq nanopore signal classification.

### Features

- Complete CLI with 6 commands: `prepare`, `merge-and-split`, `train`, `test`, `infer`, `grid-search`
- Feature extraction from POD5 and BAM files with move table parsing for dwell time computation
- Six model architectures: ConvLSTMDwell, ConvLSTMBase, TransformerDwell, ConvOnly, TCNDwell, ResNetDwell
- Parallel data preparation with multiprocessing support (8 workers default)
- Reference-based motif search to prevent training bias from basecalling errors
- Read-level data splitting to prevent leakage in multi-sample datasets
- Class weighting for imbalanced datasets
- CPU/GPU training support with automatic device detection
- Rich CLI with progress bars and modern interface
- Grid search for chunk context optimization
- Automated GitHub release workflow

### Data Preparation

- Multi-sample merge-and-split with label=file syntax
- TSV-based comparison specifications for batch processing
- Parallel POD5/BAM processing with configurable workers and chunk size
- Reference-based and basecalled motif search strategies
- Optional indel filtering at motif sites

### Training & Evaluation

- Training with early stopping, checkpointing, and validation
- Comprehensive metrics: accuracy, precision, recall, F1, ROC AUC, confusion matrix
- Grid search over signal context parameters
- Model checkpoint management with best model tracking

### Documentation

- Complete MkDocs documentation site with API reference
- CLI usage guide with all commands documented
- Architecture documentation and ADRs (Architecture Decision Records)
- Guides for cluster setup (Alpine/SLURM, Bodhi/LSF)
- Troubleshooting and implementation guides

### Development

- Complete test suite with pytest
- Modern tooling: uv for dependencies, ruff for linting, ty for type checking
- GitHub Actions CI/CD with linting, testing, and documentation deployment
- Snakemake pipeline for production workflows
