# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`leech.crf.evaluate`: decode a corpus, match it to references, report per
  group.** The generic half of CRF evaluation — what a *panel* is (which classes
  exist, which share a flowcell) stays with whatever defines the panel; what
  arrives here is a reference set, a grouping and a corpus.

  Three rules it holds rather than leaving to callers, because each produces a
  plausible-looking wrong number:

  - **Match what the model emits.** `emitted_references` applies
    `target[state_len:]` once. Scoring against full-length targets forces
    `state_len` leading deletions into every alignment, which inflates every
    distance *and compresses the margin* — an aligner puts those deletions where
    they help most, discounting wrong references more than the right one.
  - **Report per group.** When classes are crossed with batch, one pooled table
    measures batch. `balanced_recall` takes the grouping as an argument (only
    the caller knows whether their classes are confounded) and **raises** when
    no group has reads, because a null headline serializes fine and ships.
  - **Balanced, not raw.** A pooled accuracy over unbalanced classes is
    dominated by the deepest class.

  `lev_vs_refs` scores one decode against the whole reference set at once, which
  is the shape of every evaluation loop; scoring R references one at a time is R
  DP tables per read. Its vectorisation recovers the serial insertion term
  exactly (`j + cummin(tmp[k] - k)`), and that identity is asserted against the
  scalar implementation on random strings rather than assumed. edlib is used
  where importable, with the pure-Python fallback kept under its own name so a
  test compares the two rather than comparing edlib with itself.

  Validated end to end on the production ldx corpus: 16 references at emitted
  length 44 from 48, 4000 held-out reads decoded, and per-flowcell reporting
  that correctly finds 8 classes in each — the pilot's code-flowcell confound,
  which is exactly why pooling would be wrong.

### Fixed

- **`signal_kmer` degraded to `base_onehot` on the strength of one chunk, and
  the checkpoint did not record that it had** (#230). Three separable defects
  that combined into a run which finishes, looks fine, and trained on a
  different model input than was asked for.

  The `--seq-encoding` default is `signal_kmer`, so a corpus written by one
  leech version and read by another that finds no base-to-signal maps warns and
  carries on. That reached production through a project running two pixi
  environments: merges ran from the updated one, training from the stale one,
  and eight arms trained on `(4, kmer_len)` instead of `(36, signal_len)`. They
  were caught only because an unrelated memory ceiling OOM-killed the jobs; at
  the previous ceiling all eight would have completed with AUROC values in the
  right range from the wrong representation.

  - **The decision was made from chunk 0.** It is now a whole-corpus count —
    one vectorised pass over the CSR offsets and the context column, data
    already in memory — and the message says what fraction is affected. Chunk 0
    was wrong in both directions: one empty first row flipped a whole corpus to
    `base_onehot`, and a corpus whose first row was fine encoded every later row
    that had no map from data that isn't there. Those came out as **all-zero
    sequence channels**, per chunk, with nothing raised and nothing logged —
    the one defect of the three that was still silent rather than loud.

    A corpus carries the maps for every chunk or for none. None is the
    version-skew case the fallback exists for; anything in between is damage,
    and it now raises rather than resolving, because both available answers are
    the same silent representation change — encoding the uncovered rows anyway
    gives them the all-zero channels above, and switching the whole corpus on
    the strength of a few bad rows throws the encoding away for every good one.
    `--seq-encoding base_onehot` reads such a corpus if that is what you want.
  - **An encoding named on the command line is no longer substituted.**
    `--seq-encoding signal_kmer` over a corpus that cannot supply it now stops
    the run; taking the default still falls back, which is the case the fallback
    exists for. `--encoding-fallback` / `--no-encoding-fallback` overrides the
    choice either way (`LeechDataset(allow_encoding_fallback=...)` in code).
  - **The config records what the run used, not what it asked for.** The model
    is built from `LeechDataset.effective_seq_encoding` and `config.json` stores
    it, so a fallen-back arm is auditable after the fact — one field, where
    before there was no way to find out at all. That also keeps #217's ONNX
    contract honest: it is derived from the same config, and exists precisely so
    a non-Python consumer can trust the input spec. A config that misstates its
    encoding is refused at export rather than published, which
    `test_onnx_export.py` now pins.

  Before this, the middle state was the worst of the three: the dataset half
  fell back and the model half did not, so a run died at the first forward pass
  with a channel-count `RuntimeError` naming neither the corpus nor the
  encoding. Now the two halves agree, and the disagreement is reported where it
  happens.

- **CRF training: the batch order no longer replays the split's shuffle.**
  `resolve_split` seeds a generator from `seed`, and on the corpus-split and
  held-out-batch paths it shuffles an array of the *same length* the epoch loop
  goes on to shuffle. `CrfTrainer.train` then opened a second
  `default_rng(seed)`, which replayed that generator's first draw exactly — so
  epoch 1 trained on `pi(pi(train))`, a batch order determined by the split
  rather than independent of it, and every later epoch was the split stream
  shifted by one.

  Nothing about a run looks wrong when this happens: the order is still a
  permutation and the loss still falls, which is why it needed a test rather
  than an eye. `epoch_order_rng` spawns a distinct stream from the same seed,
  so runs stay reproducible and the two orders are independent.

  This changes which batches a given seed sees, so numbers from a seed will not
  reproduce across this release. Batch order alone moves a 32-epoch run's final
  training loss by more than 2x — measured on this corpus with one loop body and
  only the RNG stream varied — so treat a seed as one draw from that spread,
  not as a fixed point.

- **The k-mer context padding branch was compared by nothing.** At the default
  `signal_context=(200, 200)` no fixture read's context window runs off the end
  of the read — 0 of 18 chunks contain an `N` — so `test_backend_parity.py`,
  which otherwise compares every array in the npz between the two `data prepare`
  backends, only ever exercised the in-range path.

  That is the failure shape this stack keeps hitting: a golden that missed a
  wrong fallback because all 19 fixture reads took the other branch, caught only
  against a real corpus in 4 reads out of 842. It mattered more once the
  windowing became a call into escapepod-signal (#222), since the two sides pad
  against bounds that *sound* different (`len(sequence)` in Python, the sequence
  slice in Rust) and nothing compared them where it shows.

  `test_kmer_context_padding` widens the signal window until the branch is
  reached (9 of 18 chunks at 2000, all 18 at 6000) and asserts the padding
  actually happened — without which a fixture change that stopped producing edge
  chunks would leave it passing while testing nothing. The backends do agree,
  and now demonstrably rather than by inspection.

### Changed

- **The signal-level k-mer encoding comes from escapepod-signal** rather than
  being held here (escapepod-rs#271 / #272; requires escapepod 0.16.0).
  `rust/src/encoding.rs` and `sequence_to_int` are now calls into
  `escapepod_signal::seq_encoding`.

  leech held the only copy of this rule, inside a `crate-type = ["cdylib"]`
  Python extension module — so a native runtime for a leech `signal_kmer` model
  could not link it and had to transcribe it, which is a second definition that
  diverges silently. It is also the natural pair to `escapepod_signal::mapping`,
  which *produces* the base-to-signal map the encoding consumes: the producing
  half was already upstream and the consuming half was not.

  This is a delegation, so the only acceptable outcome is identity: 198 parity
  tests pass unchanged, including `test_backend_parity.py`, which compares every
  array in the npz between the Rust and Python backends against a Python
  reference this change does not touch.

  The k-mer *context slice* delegates too, via `sequence_bases_with_context`
  (escapepod-rs#274, escapepod 0.16.1). It could not at first: leech needs the
  window as **bases**, since the corpus serializes `sequence_with_kmer_context`
  as a string, where upstream only offered ints. Upstream now exposes both forms
  over one windowing rule, with `sequence_to_int(bases) == ints` pinned by a
  test there — so the three halves of the signal-level k-mer path (the map, the
  window, the encoding) all live in `escapepod-signal` and none is duplicated
  here.

  That third one is the highest-stakes of the three: it is where `before` and
  `after` are not interchangeable, and swapping them displaces every k-mer
  silently because the encoder only sees the total width. It is also the most
  directly checkable — `sequence_with_kmer_context` is one of the fields
  `test_backend_parity.py` compares array-by-array between backends.

  Only `rust/Cargo.toml`'s git tag moves to v0.16.1; the `escapepod` Python pin
  stays `>=0.16.0`, because the new function is in the Rust crate and not in the
  Python bindings.

### Added

- **ONNX export, for the classifier arms and the CRF encoder** (#217).
  `leech model export --format onnx` beside the existing `--format torch`
  (unchanged default), and `leech.crf.export.export_crf_onnx`. `torch.export`
  makes a model loadable by anything with PyTorch and by nothing else; a runtime
  consuming ONNX — which is what escapepod-rs runs — could not load a leech
  model at all.

  Both use the **dynamo exporter at opset 18**. `dynamo=False`, the obvious
  first attempt, fails on these architectures with an `adaptive_avg_pool1d`
  error that reads like a model problem and is an exporter limitation; that is
  documented where someone will hit it, and a regression test pins it.

  Each export writes a **contract** beside the graph, carrying the two things a
  consumer needs and cannot recover from it: which input is which (including
  that the `signal_kmer` sequence input is built in the dataset, not the model,
  and that `leech-core` ships that encoder), and what the output means — a
  single BCE logit, not a two-class softmax. The CRF's contract additionally
  carries standardisation, which is in neither the config nor the checkpoint,
  and its emitted references (`target[state_len:]`), computed from the
  `state_len` the encoder declares.

  Verified across the serialization boundary rather than in process:
  onnxruntime against torch, 3.58e-07 for the CRF encoder against a float32 eps
  of 1.19e-07.

  New `onnx` extra (`onnx`, `onnxruntime`, `onnxscript`). CI installs it so the
  round-trip tests run rather than skip.

- **`leech model train-crf`**, the CLI for the CTC-CRF trainer. Sits beside
  `model train` rather than in a group of its own: it is the same workflow step,
  a different task. Every `CrfTrainConfig` field is exposed, and a test asserts
  each option actually reaches the config — a click option that silently does
  not is how a sweep ends up running the default every time.

  The summary reports the emission rule (`target_len -> emits target_len -
  state_len`) because widening the window to get a longer decode is the mistake
  it prevents, and prints a second table only when the run discarded steps or
  saw non-finite gradients, since a discarded step is otherwise invisible.

### Added

- **`leech.crf.training`: a CTC-CRF trainer.** `CrfTrainer` runs the schedule
  and writes `model.pt` plus a `model.json` sidecar. The sidecar is not optional:
  the standardisation constants live in neither the architecture config nor the
  checkpoint, so weights alone cannot be used correctly.

  Separate from `leech.training.Trainer`, which is classification-locked through
  `pos_weight`, `num_out`, BCE/focal/CE and AUROC/F1 checkpointing — a sequence
  task shares none of it, and forcing it through would put the production
  classifier path at risk. The decisions are split out as plain functions
  (`compute_standardisation`, `apply_quality_gate`, `resolve_split`,
  `encode_targets`, `select_checkpoint`) so they are testable without a GPU; the
  loop is mechanical, the decisions are where runs go wrong quietly.

  Verified against production data, not just fixtures: standardisation over the
  391,174 x 3000 ldx corpus reproduces the shipped model's recorded constants
  (61.8216743766 / 9.5716818880) with a delta of **0.000e+00** on both, and the
  train/test counts match what `plan_corpus` derives independently from the
  manifest (283,296 / 107,878). A two-epoch GPU run trains at ~40 s/epoch with
  loss falling 0.4446 -> 0.0639.


- **`leech.crf.corpus`: cut a CRF training corpus from a manifest.**
  `plan_corpus` decides which reads and in which split, touching no POD5;
  `build_corpus` extracts their signal, streaming it to a memory-mappable
  `<out>_X.npy` beside a `<out>_meta.npz`. The signal is never held in RAM as a
  whole — an 80-plex corpus is tens of gigabytes, and the memmap is what makes
  the size a disk question instead of an allocation that fails.

  The two stages are separate because everything subtle is in the plan, and a
  corpus planned wrongly still trains and still reports a number. Four rules,
  each pinned by a test: a cap only caps if every class can reach it
  (`per_group="auto"` is the rarest class's *trainable* depth, with the test
  fraction reserved first); the split is carved before capping and ranked per
  class globally across batches, since per-`(batch, class)` ranking multiplies
  the cap by the batch count whenever classes are crossed with batch; batches
  are interleaved rather than concatenated, or the whole held-out set comes from
  whichever batch sorts first and the headline number measures batch; and
  sharding happens after planning, so every shard keeps its share of one global
  split. Extracting nothing, or less than half the plan, is a hard error with a
  different message for each — the causes differ, and a 0-row corpus otherwise
  exits cleanly and reaches a GPU job.

  Validated against the production ldx manifest: 1,139,602 rows plan to 391,174
  reads at `chunk=3000`, the same count escapepod-models' extractor reports for
  that input, with the training pool balanced exactly across all 16 groups and
  held-out reads drawn from both flowcells.

  `load_corpus` / `load_corpus_meta` read both this layout and the legacy
  single-`.npz` one, so corpora written before the split layout keep loading.

## [0.8.0] - 2026-08-25

Minor rather than patch: one new capability, no change to anything that
existed. Everything in 0.7.0 behaves identically — `leech.crf` is additive, and
nothing outside it was touched.

### Added

- **`leech.crf`: CTC-CRF sequence models — encoder, training objective and
  decode.** A second task alongside the chunk classifiers: where a classifier
  maps a signal window to a label, a CRF maps one to a *sequence*, over
  `n_base ** state_len` states whose Viterbi traceback emits one base per move.
  Ported unchanged from `escapepod_models.crf`, which now imports it from here.
  The formulation is ONT's, introduced in bonito; the architecture is
  SeqTagger's published parameters (Genome Res 35:956). The equivalence checks
  that prove it correct stay in escapepod-models, where the reference
  implementation is available to run as the oracle. The port is pinned by an A/B against
  the pre-move copy: encoder forward, loss forward *and* backward, the analytic
  forward-backward's posteriors, and the decoded strings are all bit-identical
  on CPU and on GPU (Triton kernels included).

  Contents: `CrfEncoder` (3 convs -> 5 alternating LSTMs -> linear -> tanh/scale
  -> per-state blank splice), `CtcCrfLoss` with an analytic forward-backward
  that replaces a 300-step autograd replay with one elementwise multiply,
  optional Triton lattice kernels, the two-pass Viterbi `decode_batch`, and the
  packaged architecture config `leech/crf/configs/crf_ctc.toml`.

  The subpackage imports **only torch and numpy, and only on demand** — the
  config path is eager and the rest is lazy (PEP 562, as in `leech.models`), so
  `from leech.crf import DEFAULT_CONFIG` costs no torch import. That is what
  lets escapepod-models install leech `--no-deps` into a conda-forge pixi
  environment, and it is enforced by `tests/test_crf_package.py`.

  Not included yet: the trainer, the ONNX export, and the corpus paths.

- **`leech.crf.manifest`: the seam between a corpus's vocabulary and its signal.**
  One table, one row per read — `read_id, pod5, anchor_end, target` plus optional
  `label`/`group`/`batch`/`quality_score`/`quality_margin`/`split` — and nothing
  about where those facts came from. `target` is the *resolved* sequence: a class
  name may ride along in `label` for reporting, but nothing here looks one up.
  Everything above the manifest is vocabulary (panels, codes, oligos, gates) and
  belongs to whatever project defines it; everything below is signal ML.

  Two rules it exists to enforce, both silent when broken. Label quality travels
  as **numbers, never a `keep` boolean**, so the gate stays sweepable at training
  time — gating one panel's labels moved accuracy from 0.875 to 0.97, and a
  boolean decided at extraction would mean re-cutting an 8 GB corpus per
  threshold; `quality_coverage()` reports partial scoring, because an unscored
  read cannot pass a gate and is dropped without a word (this once cut a corpus
  to a non-random 13.5%). And `check_geometry` **raises** on a window too short
  to hold its target rather than warning, because a short window trains,
  converges, and quietly discriminates on fewer bases than designed.

### Fixed

- **README claimed 20 model architectures; the registry has 29.** The count had
  not moved as families were added. It also did not mention the CRF task at all,
  which is now a section of its own.

## [0.7.0] - 2026-08-25

### Fixed

- **Optional text fields wrote `None` as the literal string `"None"`.**
  `chunk.get("label", "")` returns the default only when the key is *missing*,
  so a key present with the value `None` — which is what `data prepare`
  produces when `--label` is not passed, and what `load_chunks` produces for an
  absent source group — reached `np.array(..., dtype=str)` and was stringified.
  Two consequences: an unlabelled prepare stored the label `"None"` for every
  chunk, and a save/load round trip was not idempotent (`"" -> None ->
  "None"`), so merging a corpus renamed its empty source groups to a group
  *called* `"None"` — which `--balance-groups` then weighted like a real one,
  and which pairwise relabelling (matching on the stored label) matched
  against nothing, leaving `label_int` at -1 so the dataset dropped those
  chunks. `None` is now written as `""`, which is the convention every reader
  already honours. **Corpora written before this fix keep their `"None"`
  strings** — the readers are not changed to reinterpret them, because a group
  legitimately named "None" would then be silently destroyed. Check with
  `np.load(f)["source_groups"]` and re-prepare or rewrite if affected.
- **Calling `run_inference` twice in one process could hang forever.** The
  sequential path shut its `ThreadPoolExecutor`s down with `wait=False`,
  leaving worker threads alive past the return; the parallel path then forks an
  `mp.Pool`, and a fork inherits the memory of a process with running threads —
  including any lock those threads hold — but not the threads themselves, so
  nothing releases it. `num_workers=0` followed by `num_workers>0` deadlocked
  with no error and no timeout. The CLI never hit this (one predict per
  process); anyone scripting the Python API did. The pools are drained by that
  point anyway, so they now shut down with `wait=True`.
- **`data merge --k-fold` crashed on multiclass inputs.**
  `merge_and_kfold_split_multiclass` carried its own inline copy of the merge,
  and that copy never learned about the CSR base-to-signal members added in
  0.6.8 — it masked `seq_to_sig_values` (one row per map *entry*) with the
  per-chunk mask and raised `IndexError: boolean index did not match indexed
  array`. Every k-fold multiclass merge of a corpus written by the current
  `save_chunks` failed. It now calls the shared `_merge_arrays_by_split`, like
  the binary k-fold path always did. No test covered any merge entry point,
  which is why the drift shipped; `tests/test_splitter_merge.py` now covers all
  four.
- **Merging corpora with different member sets wrote a misaligned column.**
  An input missing `focus_signal_pos` (or the residual channel) contributed no
  rows for that member but full rows for every other one, and nothing checked
  the row counts agreed. Depending on input order the result either raised
  `IndexError` on load or — silently — gave every affected chunk another read's
  focus position, and so the wrong asymmetric signal crop. Mismatched inputs are
  now rejected up front, naming the file and the missing member, and every
  output member is asserted to have one row per chunk before it is written.
- **The per-base mean absorbed any signal past the end of the map.**
  `np.add.reduceat` segments on starts alone and runs its final segment to the
  end of the array, so `level_mean` for the last mapped base summed the whole
  tail: a map of `[0, 3, 6]` over ten samples reported 135.33 instead of 2.0.
  Median, std and range come from the explicit loop over both boundaries and
  were always right, which is why nothing caught it. Python fallback only
  (`HAS_RUST` False) — what an install without the `rust` extra runs, and what
  any caller of the exported `compute_signal_levels` with a partial map gets.
  Values on a map that covers its signal are bit-identical to before.

- **`LeechDataset` no longer holds three copies of the corpus while it loads**
  (#211). `load_chunks` read every npz member, the tensorize loop built one
  tensor per chunk from them, and `torch.stack` allocated the whole contiguous
  output while that list was still alive — a 41 GB npz peaked at 116 GB and
  hit the 120 GB cgroup limit before epoch 1. The fields now fill a
  preallocated tensor in bounded batches (`torch.stack(..., out=)`), and the
  arrays are read from the npz in row blocks rather than materialised, so the
  numpy source is never resident alongside the tensors built from it. Measured
  on a 300k-chunk corpus (1.8 GB npz, 1.6 GB of output tensors): peak RSS
  6.19 GB -> 2.36 GB, load time 21.5 s -> 19.7 s, tensors bit-identical.
- **Only the members a run consumes are decompressed.** `signal_residuals_flat`
  is skipped for `--signal-mode signal`, `features_flat` for models without a
  feature branch, and the base-to-signal maps unless `--seq-encoding
  signal_kmer` asks for them — up to 20 GB of decompression that used to
  happen on every load regardless.
- **Chunk metadata is stored as columns, not a dict per chunk** (#211). The
  dicts measured 780 bytes each — 5.2 GB for a 6.7M-chunk corpus — holding a
  handful of small integers and a few hundred distinct strings. `ChunkTable`
  keeps the npz's own arrays (text packed to bytes, integers narrowed) and
  hands out a row view on demand: 112 B/chunk measured, with no conversion
  transient, and `dataset.chunks` still reads as a sequence of mappings.
- **`load_chunks`'s docstring no longer claims the data is memory-mapped.**
  `np.load` never maps a zip member, compressed or not; it is always a full
  read, which is what made this path look lazy when it was not.

### Changed

- **Merging chunk files with different member sets is now an error.** It used
  to produce a corpus that was silently wrong (above). Corpora prepared by
  different leech versions must be re-prepared, or merged within their vintage.

- **`seq_to_sig_maps` is stored as `seq_to_sig_values` + `seq_to_sig_offsets`**
  (CSR: row `i` is `values[offsets[i]:offsets[i+1]]`) instead of a pickled
  object array. The old member cost one Python ndarray per chunk to unpickle
  and could not be read in row blocks. `load_chunks`, `data merge` and the
  dataset still read the legacy member, so existing corpora stay valid — but a
  file written by this version and read by leech <= 0.6.7 has no
  `seq_to_sig_maps`, so a `signal_kmer` run on that older version falls back to
  `base_onehot` (with the warning it already emits).

### Performance

- **`prepare` writes the corpus as it is extracted instead of accumulating it.**
  Both backends used to extend one list until every batch was done and only
  then call `save_chunks`, so peak held the per-chunk dicts, their arrays, and
  the stacked copy at once. Batches now spool to disk through `ChunkSpool` and
  the `.npz` is assembled at the end. Measured on 100k chunks / 231 MB of
  arrays: peak **2.46x -> 0.27x** of the payload without a split, **2.17x ->
  0.25x** with one, at unchanged wall time. The corpus is written twice (spill,
  then `.npz`), so the output directory needs room for it twice over; both
  paths log this at the start of a run.
- **`save_chunks` no longer duplicates every field it stacks.**
  `np.stack(...).astype(np.float32)` copied the array it had just built —
  `astype` copies by default, and the chunks were already float32 — and every
  stacked member was held until `np.savez` returned. Members are now stacked
  and written one at a time with `copy=False`: peak **1.99x -> 0.97x** of the
  payload (100k chunks), and `np.stack(200k x 540).astype(...)` alone drops
  824 MB -> 443 MB.
- **The merge holds one output at a time, not the whole corpus.**
  `_merge_arrays_by_split` accumulated every sliced array for every split
  across every input, then concatenated with all of it still alive. It now
  counts kept rows in a header-only first pass, preallocates one array per
  output member, and fills from `iter_npz_row_blocks`. 240k chunks / 661 MB,
  4 inputs to 3 splits: peak **1.80x -> 1.10x** of the payload, wall 9.04 s ->
  6.71 s. The k-fold multiclass path no longer caches every input file in RAM
  either: 120k chunks, `k_fold=3`, peak **3.95x -> 2.07x**.
- **`LeechDataset` builds its tensors a row block at a time.** The tensorize
  loop ran per chunk — one `torch.tensor` per label, one `np.stack` per
  signal/residual pair, one row view per chunk — over arrays that arrive in
  blocks of ~1,500 rows. Metadata now comes off the `ChunkTable` columns in one
  slice per block and only signal and features are handled per block. Measured
  on 200k chunks with production shapes (540-sample signal + residual, 12x21
  features): construction **57.7 -> 26.3 us/chunk** with `signal_kmer`,
  **66.9 -> 23.7 us/chunk** with `base_onehot`, peak RSS unchanged (1.99 ->
  1.95 GB). 2,216 output tensors across 28 option combinations are
  bit-identical.
- **The loader fetches a batch at a time.** `LeechDataset.__getitems__` returns
  an already-collated batch and `collate_fn` passes it through, replacing 256
  per-sample `__getitem__` calls and a `torch.stack`. Batch 256,
  `num_workers=0`: **100,959 -> 925,425 chunks/s**. Per-sample randomness in
  augmentation is preserved; cross-layer shift/time-mask and the list-fallback
  path still go per sample. `signal_kmer` gets the construction win but not the
  loader win — its per-sample `encode_signal_kmer` still dominates.
- **Splitting reads are mapped to splits in one pass.** The masks were built
  with a `str()` comprehension over the whole read-id column plus one
  membership comprehension per split. 500k rows over 3 splits: **756 -> 176 ms**.
- **Inference stages its host-to-device copies through pinned memory.**
  `np.stack(...)` then a synchronous `.to(device)` from pageable memory blocked
  the GPU thread for the whole copy. Batch 512 on an A30: the copy itself is
  **1.31x** faster for `base_onehot` and **1.85x** for `signal_kmer` (42.5 MB
  per batch). End-to-end `predict` moves 1.0-1.03x — it is extraction-bound —
  so this shows up only when the GPU thread is the bottleneck.
- **`ReadInfo` rebuilds the reference sequence on demand.**
  `get_reference_sequence()` ran in the constructor for every read whether or
  not the run was reference-anchored. Construction drops **6.45 -> 4.80 us** on
  139 nt reads and **43.8 -> 10.5 us** (4.2x) on 6.8 kb reads. The value is
  still materialised before pickling, so the multiprocessing prepare path is
  unaffected.

### Internal

- **One batch accumulator instead of three.** `single.py`'s two extraction
  paths and `bundle.py` each carried their own four parallel buffers, size
  check, flush and mega-batch write. `BatchAccumulator` and
  `prepare_signal_channels` are now shared; `single.py` drops 1400 -> 1259
  lines. Output BAMs are byte-identical across {multiclass, binary, bundle} x
  {rust, python} x {0, 2 workers}.
- **The four merge functions share their common shape.** `_collect_read_index`,
  `_assign_splits`, `_assign_kfold_splits` and friends replace four copies of
  scan-ids / assign / merge / build-result. Public signatures and returned
  dicts are unchanged; outputs verified member-for-member identical across 17
  scenarios.
- **Feature channel order is resolved once per read**, not rebuilt per chunk
  from a dict merge inside `get_chunk`. The order is unchanged and now pinned
  by name, by row, and by value against the Rust pipeline's own order.
- **The tally passes over chunk metadata read columns.** `max(label_int)`,
  the source-group and label counts, the sampler weights and `_crop_starts`
  each built a row view per chunk. 200k chunks: `max(label_int)` 97.9 -> 0.03 ms,
  the focus-position loop 124.2 -> 0.08 ms.

## [0.6.7] - 2026-08-24

Promotes `0.6.7-rc.1` unchanged — no commits landed between the two tags. The
release candidate exercised the new PyPI path end to end, so this is the first
version installable with `uv add "leech[rust]"` / `pip install "leech[rust]"`
rather than from a checkout.

### Added

- **`leech` and `leech-core` publish to PyPI automatically on a `v*` tag**
  (#210), via Trusted Publishing (OIDC) — no API token, no repository secret.
  Publishing was previously a manual `uv publish`, and `leech-core` was never
  published at all, so the `rust` extra could not resolve for anyone outside
  the workspace.
- **Two release gates.** `check-version` fails the tag before anything builds
  if it disagrees with either declared version — a PyPI upload cannot be
  replaced, so a wrong version reaching the index is permanent. `test` runs the
  suite at the tagged revision, which nothing did before: CI triggers on pushes
  to `main` and on PRs, never on a tag.

### Changed

- **`leech-core` ships one stable-ABI wheel per platform** (pyo3 `abi3-py312`)
  rather than one per interpreter, so a new CPython release no longer needs a
  new `leech` release to get a wheel. Wheels cover manylinux x86_64 and
  aarch64; other platforms build from the sdist, which needs a Rust toolchain
  and network access to github.com.
- **The `rust` extra pins `leech-core` exactly.** `check_rust()` only *warns*
  on a mismatch, so for a PyPI install the pin is the only thing preventing a
  current `leech` from pairing with a stale extension — the hazard that let
  `leech_core` sit at `0.3.0` across ten releases.

### Fixed

- **A correctly paired pre-release reported a version mismatch.** The two
  halves report versions in different dialects: `leech`'s through
  `importlib.metadata` in PEP 440 normal form, `leech_core`'s from
  `env!("CARGO_PKG_VERSION")` as literal Cargo semver. A final release spells
  the same in both, so the raw `==` looked correct until the first
  pre-release — where it warned on every install and failed the new `test`
  gate, making pre-releases unpublishable. Found by the `0.6.7-rc.1`
  rehearsal.

## [0.6.7-rc.1] - 2026-08-24

First release candidate. `leech` is public, and both distributions now publish
to PyPI automatically from a `v*` tag — this rc exists to exercise that path
end to end before a final release depends on it. It is a pre-release, so
`pip install leech` will not resolve to it.

### Added

- **`leech` and `leech-core` publish to PyPI on tag** (#210), via Trusted
  Publishing (OIDC) — no API token, no repository secret. Install with
  `uv add "leech[rust]"` or `pip install "leech[rust]"`. Publishing was
  previously a manual `uv publish`, and `leech-core` was never published at
  all, so `leech[rust]` could not resolve for anyone outside the workspace.
- **Two release gates that did not exist.** `check-version` fails the tag
  before anything builds if it disagrees with either declared version — a PyPI
  upload cannot be replaced, so a wrong version reaching the index is
  permanent. `test` runs the suite at the tagged revision, which nothing did:
  CI triggers on pushes to `main` and on PRs, never on a tag.

### Changed

- **`leech-core` ships one stable-ABI wheel per platform** (pyo3 `abi3-py312`)
  instead of one per interpreter. It loads on CPython 3.12 and every later 3.x,
  so a new CPython release no longer needs a new `leech` release to get a wheel.
  Wheels cover manylinux x86_64 and aarch64.
- **The `rust` extra pins `leech-core` exactly.** `check_rust()` only *warns* on
  a mismatch, so for a PyPI install the pin is the only thing preventing a
  current `leech` from pairing with a stale extension — the hazard that let
  `leech_core` sit at `0.3.0` across ten releases. The version now lives in
  three files and the test suite enforces all three agree with each other and
  with the tag.

### Fixed

- **A correctly paired pre-release reported a version mismatch.** The two halves
  report versions in different dialects: `leech`'s arrives via
  `importlib.metadata` in PEP 440 normal form (`0.6.7rc1`), `leech_core`'s from
  `env!("CARGO_PKG_VERSION")` as the literal Cargo semver (`0.6.7-rc.1`). A
  final release spells the same in both, so the raw `==` comparison looked
  correct right up to the first rc — where it warned on every install and failed
  the version-pairing test, and so would have failed the new `test` gate and
  made pre-releases unpublishable. `rust_version_mismatch()` now compares
  normalized forms, without taking a dependency on `packaging` (which is not a
  runtime dependency, and whose presence in dev environments is exactly how this
  would have come back).

## [0.6.6] - 2026-08-24

Patch release completing the DataLoader-worker fix started in 0.6.5. `eval test`
was fixed there; the in-training validation loader and both `calibration.py`
loaders were not, so a GPU still went near-idle once per epoch.

### Fixed

- **Validation loader starved the GPU once per epoch** (#207). `train` resolved
  workers for the training loader and hardcoded `num_workers=0` for validation
  three lines below. On a 1,176,763-chunk validation set that was ~5 minutes of
  near-idle GPU at every epoch boundary — roughly 75 minutes across a 15-epoch
  run, scaling with validation size.
- **`calibration.py` passed a literal 0 through on CUDA.** Both loaders took
  `num_workers: int = 0` and forwarded it unresolved, so `0` meant "no workers"
  rather than AUTO.

### Added

- `resolve_val_dataloader_workers`, beside `resolve_dataloader_workers`. Same
  rule, with one scoped exception: a dataset that fell back to per-chunk lists
  keeps 0 workers. `LeechDataset` stacks into contiguous buffers precisely so a
  fork COW-shares them; only the `_try_stack` fallback multiplies peak RSS. That
  exception wins even over an explicit `--num-workers N`, because OOM is not a
  throughput tradeoff — which is what the previous blanket 0 was protecting, at
  the cost of every other run.
- A guard test: `num_workers` may not be a bare literal anywhere in the package.
  It must come from a resolver, from a local named for what it carries, or
  carry a call-site marker `dataloader-workers: unresolved` with a reason. Two
  markers exist — `commands/benchmark.py` (worker count is the variable under
  test) and the legacy `SignalCNN` path (`SignalDataset` has no
  `_signals_tensor`, so the validation guard would force 0 and change
  behaviour).

  The guard is checked against the value wherever it is bound, not against
  `DataLoader(...)` call sites. Two earlier versions failed their own mutation
  test: a file-scoped allow-list exempted a whole file so the #207 bug passed,
  and a call-site check also passed it because that bug lives in a kwargs dict
  reaching the loader via `**`, in a function that resolves a different loader.

## [0.6.5] - 2026-08-24

Performance and build-correctness release. `leech eval test` was feeding the GPU
from a single process and now uses DataLoader workers, and `leech_core`'s
version tracks `leech`'s so `uv` can no longer restore a stale compiled
extension over a current build.

### Fixed

- **`leech eval test` fed the GPU from a single core.** The eval DataLoader was
  built with `num_workers` pinned to 0 and no flag to change it, so collate, the
  host-to-device copy and the forward pass all ran serially in one process: 8%
  GPU utilisation on an A5000 over a 7,835,334-chunk test set, against 98% for
  `model train` on the same corpus and the same card — same dataset class, same
  collate function, the only difference being that training had workers.

  The rule for sizing a loader now lives in exactly one place,
  `dataset.resolve_dataloader_workers`, which training carried inline and
  evaluation did not have at all. Its semantics are training's: `0` means
  *auto*, auto is 0 on CPU (workers there would compete with the compute) and
  >0 on CUDA, and a daemonic process — a grid-search `mp.Pool` worker — always
  gets 0, because it cannot spawn children.

  Auto is now also capped by the CPUs the process may actually run on
  (`sched_getaffinity`, which respects the Slurm cpuset). Without that, the
  pipeline's GPU eval rules, which request `cpus_per_task=2`, would have forked
  8 workers onto 2 cores. An explicit `--num-workers N` is honoured as given.

- **`leech_core`'s version now tracks `leech`'s.** It sat at `0.3.0` from v0.3.1
  to v0.6.4 — ten releases, spanning #176, #185, #187, #188, #192, #195, #200
  and #202 — while the Rust changed underneath it. That is not cosmetic: `uv`
  keys its archive cache on the version string, so `uv sync` could restore a
  compiled extension built from *any* earlier revision that shared it, over a
  current build. Observed doing exactly that: 43 tests failing with pre-#188
  behaviour (`chunk_signal_kmer_inputs` no longer snapping `map[0] = 0`) against
  an up-to-date working tree.

  `rust/Cargo.toml` is the single source; `rust/pyproject.toml` takes it via
  `dynamic = ["version"]` rather than carrying a third copy to keep in sync.

- **`leech_core` exports `__version__`, and `check_rust()` reports a mismatch.**
  The two are separate distributions built from one repository, so an extension
  compiled at one revision can sit alongside a `leech` from another. That
  pairing does not raise — it produces different numbers, which is how #176
  stayed hidden (new Rust, old serial driver). `check_rust()` printed a bare
  `leech_core` with no version at all; it now names it and says which half to
  rebuild.

### Added

- `--num-workers` on `leech eval test`, so the auto default can be overridden
  where it is wrong (default `0` = auto, as on `model train`).

- `tests/test_rust_version_pairing.py`: asserts the two declared versions agree
  in the source tree, that `rust/pyproject.toml` defers rather than pinning a
  third copy, and that the *installed* extension matches the tree — the last of
  which is the stale-build hazard itself.

### Changed

- The release process (`.claude/commands/release.md`) bumps both versions and
  re-verifies `check_rust()` afterwards, so this cannot drift again by omission.

### Documentation

- Four `leech model` subcommands were missing from the CLI reference entirely --
  `benchmark`, `release`, `list` and `fetch`. All four are now documented with
  their options and defaults taken from the click definitions.

- `CLAUDE.md`'s module tree named `util.py` and `inference.py`, neither of which
  has existed since they were split into `bundling.py` / `model_loading.py` /
  `model_export.py` / `metrics.py` and the `inference/` package. Thirteen
  modules were unlisted; the tree is now checked against the source.

- The docs workflow watched `mkdocs.yml` for changes. The site has built with
  zensical since the migration, so edits to `zensical.toml` never triggered a
  deploy.

## [0.6.4] - 2026-08-24

Dependency and internal-consolidation release. No user-facing behaviour change;
the only numeric movement is float32 rounding in level features, described
below. **Requires `escapepod >= 0.15.0`.**


### Changed

- **escapepod bumped to v0.15.0, and four locally-held primitives handed back to
  it.** All four are things leech was carrying only because escapepod had no
  home for them; each was filed upstream during the Rust/Python audit and each
  is now adopted. Net: 391 lines of Rust deleted against 102 added, with no
  behaviour change beyond float32 rounding.

  - **The refinement settings are escapepod's preset** (escapepod-rs#257).
    `refinement.rs::build_settings` was a 28-line `RefineSettings` literal
    duplicating the one inside escapepod's Python binding; the two drifted on
    `dwell_target` and that is what caused #193. It is now
    `RefineSettings::move_table_refinement(half_bandwidth, n_iters, seed)`,
    which is field-for-field identical to what leech was building — verified
    before switching — so the drift is now structurally impossible rather than
    test-guarded.

  - **The POD5 reader cache is escapepod's** (escapepod-rs#258). leech's
    process-global `OnceLock<Mutex<HashMap<String, Arc<Reader>>>>` is replaced
    by `escapepod_signal::cached_reader`, which warms the read-id index before
    publishing the entry and opens outside the lock — the two properties that
    made leech's version worth having. The batch-signal helper stays here and
    builds on it.

  - **Per-base statistics are `features::span_stats`** (escapepod-rs#260), with
    `SpanFill::Zero`, `SpanBounds::Clamp` and `MedianConvention::SortPartialCmp`
    to preserve leech's semantics exactly. escapepod computes in `f64` prefix
    sums where leech accumulated in `f32`, so level features move by at most
    7e-07 (`level_mean`) and 1.2e-07 (`level_std`); `level_median` and
    `level_range` are bit-identical. Both leech paths moved together, and
    Python-vs-Rust agreement is still exactly zero. leech's `median_f32` — kept
    only to match `numpy.median`'s even-length tie-break — is deleted, since
    that rule is now `MedianConvention`.

  - **Move-table and CIGAR mapping are `escapepod_signal::mapping`**
    (escapepod-rs#259). `build_seq_to_sig_map` and `compute_ref_to_signal` now
    delegate to `seq_to_signal_from_moves` and `ref_to_signal`. Verified
    byte-identical on every alignment in the tRNA fixtures before switching, and
    the Rust-vs-numpy parity test still passes. leech retains only the BAM
    op-code to `CigarKind` table, since escapepod takes the typed enum.

  - **The Python half takes the preset too.** With `escapepod` 0.15.0 now on
    PyPI, the floor moves to `>=0.15.0` and `SigMapRefiner.refine` stops passing
    `dwell_target` at all — on 0.15.0 the default means "take the preset", which
    is what leech_core takes. Pinning one field on the Python side while the
    Rust side took the whole preset would have left the two halves free to drift
    apart again the moment the preset changed, which is the exact shape of #193.
    The floor is load-bearing rather than cosmetic: on 0.14.0 omitting the
    argument silently reinstates the fixed `4.0` this override existed to
    correct, and the backend parity suite fails loudly if that happens.

## [0.6.3] - 2026-08-23

Backend parity. An audit of the Rust/Python boundary found eight divergences,
one of them on a code path that is **on by default**. Read the first entry
below before reusing any prepared corpus.

### Fixed

- The Python and Rust `data prepare` backends refined signal maps differently
  whenever `--refine-signal-map` was on, which is the **default**. Both halves
  of #168 had been applied to `leech_core` only:

  - `SigMapRefiner.refine` called escapepod's `refine_signal_map` without
    `dwell_target`, taking its fixed `4.0` default. RNA004 at 130 bps and 4 kHz
    sits near 31 samples/base, so the asymmetric dwell penalty treated every
    base as ~8x too long and moved the boundaries accordingly. leech_core has
    passed `0.0` (resolve the target from the read's own median dwell) since
    #168.
  - `SigMapRefiner.refine` then rewrote the signal with the fitted
    `(scale, shift, drift)`. That replaces one shared median-MAD transform with
    a per-read fit estimated on a chunk that sits largely in a constant 3'
    adapter, where the fit is weakly identified — observed scales ran from 15 to
    1084 and were frequently negative. leech_core stopped applying it in #168.

  Measured on the tRNA fixtures, the two backends disagreed on every chunk:
  max |signal delta| 3.44 in normalized units, every dwell different, max
  |feature delta| 3.57. After the fix: signal delta 0, dwells identical,
  features within float32 rounding.

  Which backend ran was decided by whether `leech_core` was installed and by
  `rust_prepare_unsupported_reason`, so a corpus's features depended on the
  install; `predict` splits the same way via `check_rust_extraction_available`,
  so a model could be trained and served on different transforms.
  **Re-prepare any corpus built with the Python backend and refinement on**
  (that is: without `leech_core` installed, or with `--workers 1`, a non
  median-MAD `--signal-norm`, `--recover-softclip-signal`, or a focus TSV).

- `--scale-iters -1` meant two different things. Python skipped the banded DP
  and rough-rescaled the signal instead; Rust clamped to `0`, which escapepod
  reads as "one DP pass without rescaling", so it refined the map. With the
  fitted rescale no longer applied, the Python behaviour is a no-op on both
  outputs, so `-1` now means "no refinement" on both backends.

- Refiner settings reached the Rust `data prepare` backend incompletely.
  `_prepare_batch_rust` asked the `SigMapRefiner` for `kmer_center_idx`, an
  attribute it does not have, so `getattr(..., -1)` pinned the Rust path to
  escapepod's `kmer_len / 2` default however the k-mer centre was configured —
  while the Python backend used the configured value. Half-bandwidth and
  scale-iters are now read off the same object too, rather than off
  `SignalConfig`, whose `refine_*` fields could only agree with the refiner by
  convention.

- `compute_kmer_residual_features` and the signal-residual channel extracted
  expected levels at `kmer_len // 2` while refinement used the refiner's
  `center_idx`, so a non-default centre offset every residual feature against
  the boundaries that produced it. Both now take the refiner's value, as the
  Rust pipeline already did.

- `prepare_config.json` recorded `refine_half_bandwidth`,
  `refine_do_rough_rescale` and `refine_kmer_center_idx` as dataclass defaults
  rather than what ran, because `data prepare` built the refiner without them.
  `model train` copies these into the model config and `predict` rebuilds a
  refiner from them, so the provenance chain carried defaults end to end.

- Reads whose signal map is shorter than their sequence were dropped by the
  Python `data prepare` backend and kept by the Rust one. Under
  `anchor="reference"` the sequence is the aligned reference slice while the map
  comes from `compute_ref_to_signal`, which strips trailing non-match CIGAR ops,
  so an alignment ending in a deletion has `num_mapped_bases < num_bases`. Both
  `compute_kmer_residual_features` and `compute_signal_residual` then handed
  numpy a length mismatch; the `ValueError` propagated out of `build_leech_read`
  and the workers' `except` turned it into losing the whole read. This is #185's
  failure mode with the backends swapped, and it selects the same population:
  indel-heavy and supplementary alignments. Levels are now fitted to the
  mapped-base grid on both sides (`levels_for_mapped_bases`), matching what the
  Rust pipeline did by zipping.

- The Rust pipeline emitted `kmer_expected` at full sequence length while
  deriving `kmer_residual` / `kmer_residual_abs` at the shorter mapped-base
  length, so one read could produce feature rows of two different widths and
  chunk extraction's `safe_end <= feat_row.len()` guard would zero some rows and
  not others. All three are now mapped-base width. `compute_signal_residual`
  also indexes levels with `.get` instead of `[i]`, since an out-of-range index
  inside a rayon worker is a panic that takes the whole batch down.

- `--no-rough-rescale` now warns that it is not honored. Refinement is
  delegated to escapepod, whose `refine_signal_map` always applies its
  least-squares rough rescale and exposes no switch, so neither backend could
  act on the flag.

- `predict` dropped `base_justify` on the Rust extraction path.
  `build_rust_extraction_kwargs` carried no such key, so the Rust signature's
  `"center"` default silently overrode a model trained with
  `--base-justify start`/`end` — which moves the focus sample within the base
  and so shifts every signal window. `data prepare` passed it correctly; only
  `predict` lost it, and nothing validates it, so the symptom was degraded
  accuracy with no error.

- `dwell_offset` was inert on the Rust `predict` path, and a wide feature
  window was passed to the model at full width. The Rust extractor returns
  features over the whole requested window, exactly as a Python chunk does, but
  the Rust consumers appended them to the batch without the narrowing the
  Python path applies. `validate_inference_shapes` checks feature *count*, not
  *width*, so this did not raise.

- The bundle's Python path appended dwell template channels *after* narrowing
  to the k-mer window, while training (`dataset.py`) appends before. The
  templates were keyed to the stored window's column 0 but applied to an array
  that had already been shifted out from under them.

  All four copies of this transform are now one function,
  `prepare_inference_features`, which mirrors `ChunkDataset._prepare_features`
  and raises when the requested window does not fit the stored one — training
  already raised for the same condition rather than sliding the window.

- Two of `predict`'s three extraction paths searched for the motif in the
  basecall while cutting chunks in reference coordinates. Motif positions index
  whatever sequence chunks come from, which under `anchor="reference"` is the
  aligned reference slice. `ReferenceMotifSearcher` ignores its `sequence`
  argument, which is why this was invisible — but `predict` picks the searcher
  with `mode="fasta" if reference_sequences else "bam"`, so a run without a
  reference FASTA gets the *basecalled* searcher, where the argument decides
  the answer. The rule now lives in one place, `chunking.extraction_sequence`,
  which `data prepare` also goes through.

- `require_query_mapping` did not reach `predict`. It is recorded in
  `prepare_config.json` but was neither copied into the model config by `model
  train` nor read by either inference entry point, so a corpus prepared with
  `--no-require-query-mapping` was scored with the gate back on — a different
  read population than the model was trained on, and on aminoacyl-tRNA a
  label-correlated one (the adduct mis-calls the CCA junction, dropping 28% of
  charged reads against 6% of uncharged).

- `encode_kmer` mapped only ACGT, so a `U` encoded as an all-zero column while
  every other base encoder in the tree — `features.sequence_to_int`,
  `encoding.seq_to_int`, and both Rust encoders — folds U onto T. The same base
  produced two different model inputs depending on which encoder the path
  reached.

### Added

- `tests/test_backend_parity.py`: a field-by-field comparison of the two
  `data prepare` backends over the fixtures, across a matrix of anchors,
  refinement settings, `base_justify` values, feature windows, `scale_iters`
  values and signal contexts — including the exact flag set from #193.

  It serializes both backends through `save_chunks` and compares **every array
  in the npz**, failing on any field it has not been told how to compare. Every
  divergence so far was invisible to the check that caught the previous one
  (#185 counts, #186 signal_kmer fields, #189 window width, #193 values),
  because each check was written one field at a time. This one extends itself:
  adding a field to the chunk format without classifying it is a test failure.

### Changed

- One per-base statistics implementation in Rust instead of two.
  `signal_stats::compute_signal_stats` (the Python fast path) and
  `inference_pipeline::features::compute_per_base_stats` (the Rust extraction
  path) were near-identical copies that disagreed on negative map entries: the
  first cast `i64` to `usize` raw and skipped the base, the second clamps to 0
  and computes over the truncated span. The pyfunction is now a wrapper.

- One POD5 batch-read helper instead of four copies. Parsing read ids as UUIDs,
  `reads_by_ids`, `get_signal_bulk` was written out in `pod5_io` twice and in
  both pipeline entry points — four places to forget `cached_reader`, which is
  the one thing that must not be got wrong there (#176).

- `SigMapRefiner` warns when `algo` or `sd_params` are set. Neither reaches the
  DP any more: `refine` delegates to escapepod, which builds its own settings
  and uses an asymmetric dwell penalty rather than leech's short-dwell table.

- The backend parity test (`tests/test_parallel_prep.py`) now parametrizes
  `refine_signal_map` and `base_justify` instead of pinning them to `False` and
  `"center"`. Pinning them is why the divergence above survived four releases:
  the only test comparing the backends opted out of the default configuration.

## [0.6.2] - 2026-08-23

Two silent correctness bugs, both of the same shape: a value that looked like
an absent default but wasn't. **Re-prepare any Rust-backend corpus built with
an explicit `--feature-start 0`**, and note that read-level splits no longer
reproduce from pre-0.6.2 seeds.

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

- Read-level splits depended on the order chunks happened to arrive in, not
  just on the seed. `split_chunks_by_read` shuffled `list(read_to_chunks)` —
  chunk arrival order — and the merge and k-fold splitters shuffled a list
  built from a `set`, whose iteration order is PYTHONHASHSEED-dependent. The
  Python `data prepare` backend returns batches through `imap_unordered`, so a
  seeded split was already not reproducible run to run on that path. All five
  sites now sort before shuffling, which is what `_split_by_group` already did.

  **Splits change for existing seeds.** A split regenerated after this fix will
  not match one generated before it, so do not re-split a corpus whose models
  are already trained without re-training — the previous train/test boundary is
  not recoverable.

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
