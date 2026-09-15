//! Training chunk extraction, parallel processing, and PyO3 entry point.
//!
//! Like `inference.rs`, `extract_training_chunks_from_read` is a thin adapter
//! over `escapepod_signal::chunk` (rnabioco/leech#258). One difference from
//! the inference path: the training format always records the chunk-local
//! `seq_to_sig_map` and `sequence_with_kmer_context` so a corpus can be
//! re-encoded to `signal_kmer` later even when it was extracted with
//! `base_onehot` -- `chunk::cut_chunk` only produces those for a chunk it cut
//! *as* `SeqEncoding::SignalKmer`, so this file calls the now-`pub`
//! `chunk::signal_kmer_inputs` directly, a second time, with the same
//! `sig_start`/`sig_end` `cut_chunk` used internally (seam 1 of #258).

use std::collections::HashMap;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use escapepod_signal::chunk;
use escapepod_signal::seq_encoding::KmerContext;

use crate::kmer_levels::KmerLevels;

use super::types::{PipelineConfig, TrainingChunkResult, build_anchor, build_config};

/// Raw per-base dwell, straight from the map -- not one of `ChunkSpec`'s
/// `FeatureChannel`s (those are gated by `compute_features`), but
/// `TrainingChunkResult.dwell` is populated unconditionally, matching the
/// pre-port behaviour. Trivial arithmetic on `ProcessedRead`'s own public
/// field, not a second copy of anything `read_rows` computes.
fn raw_dwells(processed: &chunk::ProcessedRead) -> Vec<f32> {
    (0..processed.n_bases())
        .map(|i| (processed.seq_to_sig[i + 1] - processed.seq_to_sig[i]) as f32)
        .collect()
}

/// Slice `values` (one entry per base) over the feature window around
/// `base_idx`, left-padded with zero so offset 0 always lands on the anchor
/// base -- the same windowing `cut_chunk` applies to `spec.feature_channels`,
/// applied here to the raw dwell array `TrainingChunkResult.dwell` carries
/// independent of `compute_features`.
fn window_over_bases(
    values: &[f32],
    base_idx: i64,
    offsets: (i64, i64),
    width: usize,
    n_bases: usize,
) -> Vec<f32> {
    let fs = (base_idx + offsets.0).max(0) as usize;
    let fe = ((base_idx + offsets.1 + 1) as usize).min(n_bases);
    let left_pad = ((0i64 - (base_idx + offsets.0)).max(0)) as usize;
    let mut out = vec![0.0f32; width];
    if fs < fe {
        let n = (fe - fs).min(width.saturating_sub(left_pad));
        out[left_pad..left_pad + n].copy_from_slice(&values[fs..fs + n]);
    }
    out
}

/// Resolve `(L, R)` base offsets around `base_idx` to a sample interval via
/// the read's own base-to-signal map (`--signal-context-bases`, issue #278).
///
/// `chunking.resolve_signal_context_bases` (Python) is the canonical
/// definition; this mirrors it exactly and the two are held equal by
/// `tests/test_backend_parity.py`'s bases-context matrix row. `sample_start`
/// is the first sample of base `base_idx - L` and `sample_end` is the first
/// sample past base `base_idx + R`, both clamped to the read's own mapped
/// span (base 0 through `n_bases - 1`) -- clamping, not dropping, is the
/// guard `--signal-context-bases` promises at either edge of a read (the one
/// allowed drop rule, CLAUDE.md, is unaffected: only `base_idx` itself being
/// unmapped drops a chunk, checked by the caller before this runs).
///
/// `left_bases`/`right_bases` are meant to be `>= 0` -- the CLI and
/// `handle_prepare` both refuse a negative value before any read is touched
/// -- but `lo_base`/`hi_base` are independently clamped to
/// `[0, n_bases - 1]` regardless of sign, matching
/// `chunking.resolve_signal_context_bases` (Python) exactly. Without this a
/// large-magnitude negative `left_bases` pushes `lo_base` past `n_bases`,
/// and indexing `seq_to_sig` with it is a hard **panic** here (unlike
/// Python's `IndexError`) -- one that, per issue #265's zero-tolerance
/// policy, aborts the whole in-flight batch and discards every chunk
/// `ChunkSpool` has already spooled to disk.
fn resolve_signal_context_bases(
    seq_to_sig: &[i64],
    base_idx: i64,
    left_bases: i64,
    right_bases: i64,
    n_bases: usize,
) -> (i64, i64) {
    let last_base = (n_bases - 1) as i64;
    let lo_base = (base_idx - left_bases).clamp(0, last_base) as usize;
    let hi_base = (base_idx + right_bases).clamp(0, last_base) as usize;
    (seq_to_sig[lo_base], seq_to_sig[hi_base + 1])
}

/// The anchor's offset within the emitted `signal_len`-wide array.
///
/// A constant (`fixed_signal_context_left`) in sample mode: `left + right ==
/// signal_len` always there, so `cut_chunk`'s internal `place_window` never
/// crops. In base-defined mode the number of samples spanning `L..R` bases
/// varies read to read and chunk to chunk, landing on either side of
/// `signal_len`, so the offset is resolved per chunk here -- replicating
/// `place_window`'s own pad (left-aligned, zero-fill right) / centre-crop
/// split (`escapepod_signal::chunk`) rather than the array contents
/// themselves, which `cut_chunk` already cut.
fn resolve_chunk_focus_signal_pos(
    win_left: i64,
    win_right: i64,
    signal_len: usize,
    bases_mode: bool,
    fixed_signal_context_left: i64,
) -> i64 {
    if !bases_mode {
        return fixed_signal_context_left;
    }
    let requested = win_left + win_right;
    if requested <= signal_len as i64 {
        win_left
    } else {
        let crop = (requested - signal_len as i64) / 2;
        win_left - crop
    }
}

/// Extract training-format chunks from a processed read.
fn extract_training_chunks_from_read(
    processed: &chunk::ProcessedRead,
    rid: &str,
    positions: &[i64],
    cfg: &PipelineConfig<'_>,
    signal_context_bases: Option<(i64, i64)>,
) -> Vec<TrainingChunkResult> {
    let rows = chunk::read_rows(processed, &cfg.spec);
    let n_bases = processed.n_bases();
    let dwells = raw_dwells(processed);
    let num_features = cfg.spec.feature_channels.len();
    let dwell_width = cfg.spec.feature_width();
    let n_signal_channels = cfg.spec.signal_channels.len();
    let ctx = KmerContext {
        before: cfg.skmer_ctx.0,
        after: cfg.skmer_ctx.1,
    };

    positions
        .iter()
        .filter_map(|&base_idx| {
            if base_idx < 0 || base_idx as usize >= n_bases {
                return None;
            }

            // Base-defined window (issue #278): a per-chunk ChunkSpec clone
            // carrying the resolved SAMPLE window for this base, so a fast
            // and a slow read read the same BASES of context. `cut_chunk`
            // (escapepod_signal::chunk) still does the actual pad/centre-crop
            // into the fixed `signal_len` -- no local reimplementation of
            // that rule.
            let mut per_base_spec;
            let spec_for_cut: &chunk::ChunkSpec = if let Some((lb, rb)) = signal_context_bases {
                let bi = base_idx as usize;
                let (sample_start, sample_end) =
                    resolve_signal_context_bases(&processed.seq_to_sig, base_idx, lb, rb, n_bases);
                let focus = cfg
                    .training_spec
                    .base_justify
                    .focus(processed.seq_to_sig[bi], processed.seq_to_sig[bi + 1]);
                per_base_spec = cfg.training_spec.clone();
                per_base_spec.signal_context = (focus - sample_start, sample_end - focus);
                &per_base_spec
            } else {
                &cfg.training_spec
            };

            // `spec_for_cut` cuts with `seq_encoding: None` (inherited from
            // `cfg.training_spec`) so `cut_chunk` cannot drop this chunk over
            // a failed internal `signal_kmer_inputs` call -- training builds
            // its own `sequence`/`seq_to_sig_map` below and never reads
            // `c.sequence`. Only a focus base with no signal boundaries is
            // skipped, already guarded above (issue #185); `rows` was built
            // from `cfg.spec`, which is fine since `read_rows` doesn't
            // consult `seq_encoding` or `signal_context` at all.
            let c = chunk::cut_chunk(processed, &rows, spec_for_cut, base_idx)?;

            let (signal, signal_residual) = if n_signal_channels > 1 {
                let (sig, res) = c.signal.split_at(cfg.spec.signal_len);
                (sig.to_vec(), res.to_vec())
            } else {
                (c.signal, Vec::new())
            };

            // Always computed regardless of `spec.seq_encoding`, keyed off the
            // REQUESTED (pre-crop) SIGNAL window -- the same pair
            // `cut_chunk`'s own SignalKmer branch would hand
            // `signal_kmer_inputs` internally, not the post-crop window
            // `place_window` actually placed (issue #186).
            let (win_left, win_right) = spec_for_cut.signal_context;
            let sig_start = c.focus_signal_pos - win_left;
            let sig_end = c.focus_signal_pos + win_right;
            let (seq_to_sig_map, ctx_bytes) =
                chunk::signal_kmer_inputs(processed, sig_start, sig_end, cfg.spec.signal_len, ctx)
                    .unwrap_or_default();
            let sequence_with_kmer_context = String::from_utf8(ctx_bytes).unwrap_or_default();

            let dwell = window_over_bases(
                &dwells,
                base_idx,
                cfg.spec.feature_offsets,
                dwell_width,
                n_bases,
            );

            let focus_signal_pos = resolve_chunk_focus_signal_pos(
                win_left,
                win_right,
                cfg.spec.signal_len,
                signal_context_bases.is_some(),
                cfg.signal_context_left,
            );

            Some(TrainingChunkResult {
                signal,
                sequence: kmer_seq(processed, base_idx, cfg.kmer_context),
                dwell,
                features: c.features,
                num_features,
                read_id: rid.to_string(),
                base_idx,
                focus_signal_pos,
                seq_to_sig_map,
                sequence_with_kmer_context,
                signal_residual,
            })
        })
        .collect()
}

/// The `kmer_context`-wide, `N`-padded base window around `base_idx` -- what
/// leech's training format always stores as `sequence`, independent of which
/// `seq_encoding` this run actually used (mirrors `SeqEncoding::BaseOneHot`'s
/// own windowing in `cut_chunk`, over bytes rather than a one-hot tensor, but
/// keyed off `kmer_context` directly rather than `spec.seq_encoding` /
/// `spec.feature_offsets` -- see the comment on `PipelineConfig::kmer_context`).
fn kmer_seq(processed: &chunk::ProcessedRead, base_idx: i64, kmer_context: i64) -> String {
    let context = kmer_context.max(0);
    let lo = base_idx - context;
    let width = 2 * context + 1;
    (0..width)
        .map(|k| {
            usize::try_from(lo + k)
                .ok()
                .and_then(|u| processed.sequence.get(u).copied())
                .map(|b| b as char)
                .unwrap_or('N')
        })
        .collect()
}

/// Process one read for training chunk extraction.
#[allow(clippy::too_many_arguments)]
fn process_one_read_training(
    raw_i16: &[i16],
    rid: &str,
    sequence: &str,
    mv: &[u8],
    stride: u32,
    ns: u64,
    trim: i64,
    positions: &[i64],
    cfg: &PipelineConfig<'_>,
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&str>,
    signal_context_bases: Option<(i64, i64)>,
) -> Vec<TrainingChunkResult> {
    let inputs = chunk::ReadInputs {
        raw: raw_i16,
        moves: mv,
        stride,
        trim,
        num_samples: ns,
    };
    let mut cigar_buf = Vec::new();
    let anchor = build_anchor(cfg, sequence, cigar_ops, ref_seq, &mut cigar_buf);

    match chunk::process_read(inputs, anchor, &cfg.process) {
        Some(processed) => {
            extract_training_chunks_from_read(&processed, rid, positions, cfg, signal_context_bases)
        }
        None => vec![],
    }
}

/// Run per-read training extraction in parallel (rayon, GIL released),
/// then convert results to Python dicts with numpy arrays.
///
/// Returns `(chunks, n_missing_from_pod5)`. The second figure counts
/// `read_ids` this POD5 has no signal for -- a read the caller submitted and
/// this call silently produced nothing for. It is reported separately because
/// the two ways a submitted read can yield no chunk are not the same event
/// (issue #325): a read absent from the POD5 is an *expected* exclusion when
/// the POD5 was deliberately pre-filtered to a subset of the BAM's reads
/// (`escpod bam-filter`, the pipeline's own Filter stage), while a read that
/// *was* found and still produced nothing is the zero-output signature
/// `_iter_rust_batches` treats as a failure (issue #267/#258). Without this
/// count the Python driver cannot tell them apart, and a selective POD5
/// filter trips `MAX_FAILED_READ_FRACTION` on every batch.
#[allow(clippy::too_many_arguments)]
fn _process_and_convert_training<'py>(
    py: Python<'py>,
    signal_map: &HashMap<String, Vec<i16>>,
    read_ids: &[String],
    sequences: &[String],
    mv_arrays: &[&[u8]],
    mv_strides: &[u32],
    num_samples_list: &[u64],
    trim_offsets: &[i64],
    motif_positions: &[Vec<i64>],
    cfg: &PipelineConfig<'_>,
    cigar_tuples: &Option<Vec<Vec<(u32, u32)>>>,
    reference_sequences: &Option<Vec<Option<String>>>,
    signal_context_bases: Option<(i64, i64)>,
) -> PyResult<(Vec<Py<PyAny>>, usize)> {
    let n_reads = read_ids.len();

    // Counted over the SUBMITTED ids, so it is directly comparable against
    // `n_submitted` on the Python side: `n_missing_from_pod5 == n_submitted`
    // means the whole batch was filtered out of this POD5, which is not a
    // failure at all.
    let n_missing_from_pod5 = read_ids
        .iter()
        .filter(|rid| !signal_map.contains_key(rid.as_str()))
        .count();

    // Phase 2: Per-read processing (parallel via rayon, GIL released)
    let all_chunks: Vec<Vec<TrainingChunkResult>> = py.detach(|| {
        (0..n_reads)
            .into_par_iter()
            .map(|i| {
                let rid = &read_ids[i];
                let raw_i16 = match signal_map.get(rid.as_str()) {
                    Some(s) => s,
                    None => return vec![],
                };
                let cigar = cigar_tuples
                    .as_ref()
                    .and_then(|c| c.get(i))
                    .map(|v| v.as_slice());
                let rseq = reference_sequences
                    .as_ref()
                    .and_then(|r| r.get(i))
                    .and_then(|s| s.as_deref());

                process_one_read_training(
                    raw_i16,
                    rid,
                    &sequences[i],
                    mv_arrays[i],
                    mv_strides[i],
                    num_samples_list[i],
                    trim_offsets[i],
                    &motif_positions[i],
                    cfg,
                    cigar,
                    rseq,
                    signal_context_bases,
                )
            })
            .collect()
    });

    // Phase 3: Convert to Python dicts (needs GIL)
    let mut results: Vec<Py<PyAny>> = Vec::new();
    for chunks in all_chunks {
        for c in chunks {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("signal", c.signal.into_pyarray(py))?;
            dict.set_item("sequence", &c.sequence)?;
            dict.set_item("read_id", &c.read_id)?;
            dict.set_item("base_idx", c.base_idx)?;
            dict.set_item("focus_signal_pos", c.focus_signal_pos)?;

            // Dwell array
            dict.set_item("dwell", c.dwell.into_pyarray(py))?;

            // Features as 2D array [num_features, dwell_width]
            if !c.features.is_empty() && c.num_features > 0 {
                let dw = c.features.len() / c.num_features;
                let arr = numpy::ndarray::Array2::from_shape_vec((c.num_features, dw), c.features)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Feature error: {e}"))
                    })?;
                dict.set_item("features", arr.into_pyarray(py))?;
            } else {
                let empty = numpy::ndarray::Array2::<f32>::zeros((0, 0));
                dict.set_item("features", empty.into_pyarray(py))?;
            }

            // seq_to_sig_map
            if !c.seq_to_sig_map.is_empty() {
                dict.set_item("seq_to_sig_map", c.seq_to_sig_map.into_pyarray(py))?;
            }

            // sequence_with_kmer_context
            if !c.sequence_with_kmer_context.is_empty() {
                dict.set_item("sequence_with_kmer_context", &c.sequence_with_kmer_context)?;
            }

            // Signal residual
            if !c.signal_residual.is_empty() {
                dict.set_item("signal_residual", c.signal_residual.into_pyarray(py))?;
            }

            results.push(dict.into());
        }
    }

    Ok((results, n_missing_from_pod5))
}

// ---------------------------------------------------------------------------
// PyO3 entry point -- training chunks (POD5 I/O + processing)
// ---------------------------------------------------------------------------

/// Extract training-format chunks for a batch of reads in one Rust call.
///
/// Returns `(chunks, n_missing_from_pod5)`: a list of Python dicts, each
/// containing numpy arrays for signal, dwell, features, and metadata, and the
/// number of submitted `read_ids` this POD5 holds no signal for. Labels and
/// other Python-side metadata are attached by the caller.
///
/// The second element exists so the caller can tell "this POD5 was pre-filtered
/// to a subset of the BAM" (benign, and the whole point of the pipeline's
/// Filter stage) from "reads were found and the pipeline still produced
/// nothing" (a real failure). See `_process_and_convert_training` and issue
/// #325.
///
/// Per-read processing is parallelized with rayon (GIL released).
#[pyfunction]
#[pyo3(signature = (
    pod5_path,
    read_ids,
    sequences,
    mv_strides,
    mv_arrays,
    num_samples_list,
    trim_offsets,
    signal_context_left,
    signal_context_right,
    kmer_context,
    motif_positions,
    signal_len,
    compute_features,
    reverse_signal = true,
    feature_start = None,
    feature_end = None,
    anchor = "basecall",
    cigar_tuples = None,
    reference_sequences = None,
    seq_encoding = "base_onehot",
    signal_kmer_context = None,
    refine_signal_map = false,
    kmer_table = None,
    kmer_len = 9,
    kmer_center_idx = -1,
    refine_half_bandwidth = 5,
    refine_scale_iters = 2,
    signal_in_channels = 1,
    base_justify = "center",
    signal_context_bases_left = None,
    signal_context_bases_right = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn extract_training_chunks<'py>(
    py: Python<'py>,
    pod5_path: &str,
    read_ids: Vec<String>,
    sequences: Vec<String>,
    mv_strides: Vec<u32>,
    mv_arrays: Vec<PyReadonlyArray1<'py, u8>>,
    num_samples_list: Vec<u64>,
    trim_offsets: Vec<i64>,
    signal_context_left: i64,
    signal_context_right: i64,
    kmer_context: i64,
    motif_positions: Vec<Vec<i64>>,
    signal_len: usize,
    compute_features: bool,
    reverse_signal: bool,
    feature_start: Option<i64>,
    feature_end: Option<i64>,
    anchor: &str,
    cigar_tuples: Option<Vec<Vec<(u32, u32)>>>,
    reference_sequences: Option<Vec<Option<String>>>,
    seq_encoding: &str,
    signal_kmer_context: Option<(usize, usize)>,
    refine_signal_map: bool,
    kmer_table: Option<&KmerLevels>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
    base_justify: &str,
    signal_context_bases_left: Option<i64>,
    signal_context_bases_right: Option<i64>,
) -> PyResult<(Vec<Py<PyAny>>, usize)> {
    let signal_context_bases = match (signal_context_bases_left, signal_context_bases_right) {
        (Some(l), Some(r)) => Some((l, r)),
        _ => None,
    };
    let n_reads = read_ids.len();
    if sequences.len() != n_reads
        || mv_strides.len() != n_reads
        || mv_arrays.len() != n_reads
        || num_samples_list.len() != n_reads
        || trim_offsets.len() != n_reads
        || motif_positions.len() != n_reads
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same length",
        ));
    }

    // Zero-copy borrow of each read's move table -- replaces the old
    // `mt.moves.tolist()` -> `Vec<Vec<u8>>` conversion (~10M PyLong
    // round-trips per 1,000-read batch, issue #259). `mv_arrays` outlives
    // `mv_slices` (it is not moved or dropped before this function returns),
    // so the borrow is valid for the rest of the call, including inside
    // `py.detach` below.
    let mv_slices: Vec<&[u8]> = mv_arrays
        .iter()
        .enumerate()
        .map(|(i, a)| {
            a.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "mv_arrays[{i}] (read_id={:?}) is not contiguous: {e}",
                    read_ids.get(i)
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let cfg = build_config(
        reverse_signal,
        anchor,
        seq_encoding,
        signal_kmer_context,
        signal_context_left,
        signal_context_right,
        kmer_context,
        signal_len,
        compute_features,
        feature_start,
        feature_end,
        refine_signal_map,
        kmer_table,
        kmer_len,
        kmer_center_idx,
        refine_half_bandwidth,
        refine_scale_iters,
        signal_in_channels,
        base_justify,
    )?;

    // Phase 1: POD5 I/O. The GIL is released for the whole phase — it is pure
    // Rust I/O touching no Python objects, and holding it here would serialize
    // every caller thread against the slowest network read in the batch,
    // making it impossible to overlap batches (issue #176).
    // Shared, already-indexed reader; see `pod5_cache::read_signals_by_ids`.
    let signal_map: HashMap<String, Vec<i16>> = py
        .detach(|| crate::pod5_cache::read_signal_map_by_ids(pod5_path, &read_ids))
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    _process_and_convert_training(
        py,
        &signal_map,
        &read_ids,
        &sequences,
        &mv_slices,
        &mv_strides,
        &num_samples_list,
        &trim_offsets,
        &motif_positions,
        &cfg,
        &cigar_tuples,
        &reference_sequences,
        signal_context_bases,
    )
}
