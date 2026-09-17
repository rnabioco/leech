//! Inference chunk extraction, parallel processing, and PyO3 entry points.
//!
//! `process_one_read` and `_process_and_convert` are thin adapters over
//! `escapepod_signal::chunk`: one `chunk::process_read` and `chunk::read_rows`
//! call per read, one `chunk::cut_chunk` call per focus position
//! (rnabioco/leech#258). The per-read signal processing, per-base statistics,
//! refinement and sequence encoding that used to be duplicated here now live
//! upstream; this file's job is marshalling PyO3 inputs in and numpy arrays
//! out.

use std::collections::HashMap;

use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use escapepod_signal::chunk;

use crate::kmer_levels::KmerLevels;
use crate::pod5_io::PreloadedSignals;

#[cfg(feature = "test-utils")]
use super::signal_mapping::compute_ref_to_signal;
use super::types::{
    ChunkResult, PipelineConfig, build_anchor, build_config, resolve_signal_context_bases,
};

/// One inference chunk returned to Python: (signal, seq_encoding, features?, read_id, base_idx).
type InferenceChunkPy = (
    Py<PyArray1<f32>>,
    Py<PyArray2<f32>>,
    Option<Py<PyArray2<f32>>>,
    String,
    i64,
);

/// Test helper return: (norm_signal, sig_map, dwells, features_2d).
#[cfg(feature = "test-utils")]
type TestProcessReadResult = (
    Py<PyArray1<f32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f32>>,
    Py<PyArray2<f32>>,
);

/// Process one read for inference: `process_read` -> `read_rows` -> one
/// `cut_chunk` per requested focus position.
#[allow(clippy::too_many_arguments)]
fn process_one_read(
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
) -> Vec<ChunkResult> {
    let inputs = chunk::ReadInputs {
        raw: raw_i16,
        moves: mv,
        stride,
        trim,
        num_samples: ns,
    };
    let mut cigar_buf = Vec::new();
    let anchor = build_anchor(cfg, sequence, cigar_ops, ref_seq, &mut cigar_buf);

    let processed = match chunk::process_read(inputs, anchor, &cfg.process) {
        Some(p) => p,
        None => return vec![],
    };
    let rows = chunk::read_rows(&processed, &cfg.spec);
    let n_bases = processed.n_bases();
    let num_features = cfg.spec.feature_channels.len();
    let dwell_width = cfg.spec.feature_width();

    positions
        .iter()
        .filter_map(|&base_idx| {
            if base_idx < 0 || base_idx as usize >= n_bases {
                return None;
            }

            // Base-defined window (issue #278/#341): a per-chunk `ChunkSpec`
            // clone carrying the resolved SAMPLE window for this base, so a
            // fast and a slow read see the same BASES of context. Mirrors
            // `training::extract_training_chunks_from_read` exactly -- see
            // that function's comment for why `signal_context` is the only
            // field that needs overriding: `cut_chunk` derives its `focus`
            // from `base_justify` and then `sig_start = focus - left`,
            // `sig_end = focus + right`, so setting
            // `(focus - sample_start, sample_end - focus)` reproduces the
            // requested `[sample_start, sample_end)` interval exactly, and
            // every other field (seq_encoding, feature_channels, ...) rides
            // along unchanged via `.clone()`.
            let mut per_base_spec;
            let spec_for_cut: &chunk::ChunkSpec = if let Some((lb, rb)) = signal_context_bases {
                let bi = base_idx as usize;
                let (sample_start, sample_end) =
                    resolve_signal_context_bases(&processed.seq_to_sig, base_idx, lb, rb, n_bases);
                let focus = cfg
                    .spec
                    .base_justify
                    .focus(processed.seq_to_sig[bi], processed.seq_to_sig[bi + 1]);
                per_base_spec = cfg.spec.clone();
                per_base_spec.signal_context = (focus - sample_start, sample_end - focus);
                &per_base_spec
            } else {
                &cfg.spec
            };

            // The only reason to drop a focus base is that it has no signal
            // boundaries -- `cut_chunk` returns `None` for exactly that (and,
            // under `SeqEncoding::SignalKmer`, when the window covers no base
            // at all -- the same "continue" the pre-port code used for an
            // empty `chunk_signal_kmer_inputs` result). A k-mer window that
            // merely overhangs the sequence is `N`-padded internally, not
            // dropped -- see CLAUDE.md on issue #185.
            let c = chunk::cut_chunk(&processed, &rows, spec_for_cut, base_idx)?;
            Some(ChunkResult {
                signal: c.signal,
                seq_enc: c.sequence,
                seq_rows: c.sequence_rows,
                seq_cols: c.sequence_cols,
                features: (num_features > 0).then_some(c.features),
                num_features,
                dwell_width,
                read_id: rid.to_string(),
                base_idx,
            })
        })
        .collect()
}

/// Run per-read processing in parallel (rayon, GIL released) then convert
/// results to numpy arrays.  Shared by [`extract_inference_chunks`] and
/// [`extract_chunks_from_preloaded`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn _process_and_convert<'py>(
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
) -> PyResult<Vec<InferenceChunkPy>> {
    let n_reads = read_ids.len();

    // --- Phase 2: Per-read processing (parallel via rayon, GIL released) ---
    let all_chunks: Vec<Vec<ChunkResult>>;
    {
        let pool_result: Vec<Vec<ChunkResult>> = py.detach(|| {
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

                    process_one_read(
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
        all_chunks = pool_result;
    }

    // --- Phase 3: Convert to numpy (needs GIL) ---
    let mut results = Vec::new();
    for chunks in all_chunks {
        for c in chunks {
            let sig_py = c.signal.into_pyarray(py).unbind();
            let seq_arr =
                numpy::ndarray::Array2::from_shape_vec((c.seq_rows, c.seq_cols), c.seq_enc)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Seq error: {e}"))
                    })?;
            let seq_py = seq_arr.into_pyarray(py).unbind();
            let feat_py = if let Some(flat) = c.features {
                let arr =
                    numpy::ndarray::Array2::from_shape_vec((c.num_features, c.dwell_width), flat)
                        .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Feat error: {e}"))
                    })?;
                Some(arr.into_pyarray(py).unbind())
            } else {
                None
            };
            results.push((sig_py, seq_py, feat_py, c.read_id, c.base_idx));
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// PyO3 entry point -- monolithic (POD5 I/O + processing)
// ---------------------------------------------------------------------------

/// Extract inference-ready chunks for a batch of reads in one Rust call.
///
/// Full production pipeline: POD5 -> normalize -> reference anchoring ->
/// signal refinement -> features -> chunk extraction -> sequence encoding.
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
pub fn extract_inference_chunks<'py>(
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
) -> PyResult<Vec<InferenceChunkPy>> {
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

    // Zero-copy borrow of each read's move table -- see the matching comment
    // in `training::extract_training_chunks` (issue #259).
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

    // --- Phase 1: POD5 I/O (indexed read lookup + bulk extract), GIL released ---
    // Pure Rust I/O over no Python objects, so the GIL is dropped for the whole
    // phase; see `crate::pod5_cache` for why the reader must be shared.
    // Shared, already-indexed reader; see `pod5_cache::read_signals_by_ids`.
    let signal_map: HashMap<String, Vec<i16>> = py
        .detach(|| crate::pod5_cache::read_signal_map_by_ids(pod5_path, &read_ids))
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    _process_and_convert(
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

// ---------------------------------------------------------------------------
// PyO3 entry point -- from preloaded signals (prefetch pipeline)
// ---------------------------------------------------------------------------

/// Extract inference-ready chunks using pre-fetched POD5 signals.
///
/// Identical to [`extract_inference_chunks`] but skips Phase 1 (POD5 I/O).
/// The `preloaded` handle is produced by [`preload_pod5_signals`] which can
/// run in a background thread to overlap I/O with processing + GPU inference.
#[pyfunction]
#[pyo3(signature = (
    preloaded,
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
pub fn extract_chunks_from_preloaded<'py>(
    py: Python<'py>,
    preloaded: &PreloadedSignals,
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
) -> PyResult<Vec<InferenceChunkPy>> {
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

    // Zero-copy borrow of each read's move table -- see the matching comment
    // in `training::extract_training_chunks` (issue #259).
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

    // Skip Phase 1 -- use preloaded signals directly
    _process_and_convert(
        py,
        &preloaded.signals,
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

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Process a single read through the production pipeline (no POD5, for
/// testing): `chunk::process_read` + `chunk::read_rows`, no table, no
/// refinement, the same 9-channel feature set inference/training build when
/// `compute_features` is on and no level table is configured. Routes through
/// production rather than re-implementing it, so `tests/test_rust_python_parity.py`
/// measures the real pipeline (rnabioco/leech#258, #261).
#[cfg(feature = "test-utils")]
#[pyfunction]
#[pyo3(signature = (raw_signal, mv_array, stride, trim_offset, num_samples, reverse_signal = true))]
pub fn _test_process_read<'py>(
    py: Python<'py>,
    raw_signal: Vec<i16>,
    mv_array: Vec<u8>,
    stride: u32,
    trim_offset: i64,
    num_samples: u64,
    reverse_signal: bool,
) -> PyResult<TestProcessReadResult> {
    let inputs = chunk::ReadInputs {
        raw: &raw_signal,
        moves: &mv_array,
        stride,
        trim: trim_offset,
        num_samples,
    };
    let process_cfg = chunk::ProcessConfig {
        reverse_signal,
        normalization: chunk::SignalNorm::MedianMad,
        levels: None,
        refine: None,
    };
    // No sequence content is needed: every channel below (dwell family, level
    // span stats) is derived from `signal`/`seq_to_sig` alone.
    let processed =
        chunk::process_read(inputs, chunk::Anchor::Query { sequence: b"" }, &process_cfg)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Empty signal after trimming")
            })?;
    if processed.n_bases() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("No bases"));
    }

    let spec = chunk::ChunkSpec {
        feature_channels: vec![
            chunk::FeatureChannel::Dwell,
            chunk::FeatureChannel::DwellLog,
            chunk::FeatureChannel::DwellMean,
            chunk::FeatureChannel::DwellStd,
            chunk::FeatureChannel::DwellRatio,
            chunk::FeatureChannel::LevelMean,
            chunk::FeatureChannel::LevelMedian,
            chunk::FeatureChannel::LevelStd,
            chunk::FeatureChannel::LevelRange,
        ],
        ..chunk::ChunkSpec::default()
    };
    let rows = chunk::read_rows(&processed, &spec);
    let num_bases = processed.n_bases();

    let sig_py = processed.signal.into_pyarray(py).unbind();
    let map_py = processed.seq_to_sig.into_pyarray(py).unbind();
    let dwell_py = rows.feature_rows[0].clone().into_pyarray(py).unbind();

    let n_feats = rows.feature_rows.len();
    let mut feat_flat = vec![0.0f32; n_feats * num_bases];
    for (f_idx, row) in rows.feature_rows.iter().enumerate() {
        feat_flat[f_idx * num_bases..(f_idx + 1) * num_bases].copy_from_slice(row);
    }
    let feat_arr = numpy::ndarray::Array2::from_shape_vec((n_feats, num_bases), feat_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let feat_py = feat_arr.into_pyarray(py).unbind();

    Ok((sig_py, map_py, dwell_py, feat_py))
}

/// Expose compute_ref_to_signal for direct Python<->Rust comparison testing.
#[cfg(feature = "test-utils")]
#[pyfunction]
#[pyo3(signature = (query_to_sig, cigar_ops))]
pub fn _test_ref_to_signal(
    query_to_sig: Vec<i64>,
    cigar_ops: Vec<(u32, u32)>,
) -> PyResult<Vec<i64>> {
    Ok(compute_ref_to_signal(&query_to_sig, &cigar_ops))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_default_config(
        kmer_context: i64,
        feature_start: Option<i64>,
        feature_end: Option<i64>,
    ) -> PipelineConfig<'static> {
        build_config(
            true,
            "reference",
            "base_onehot",
            None,
            200,
            200,
            kmer_context,
            400,
            true,
            feature_start,
            feature_end,
            true,
            None,
            9,
            4,
            5,
            2,
            1,
            "center",
        )
        .expect("valid config")
    }

    #[test]
    fn base_onehot_context_matches_kmer_context() {
        let cfg = build_default_config(5, None, None);
        assert_eq!(
            cfg.spec.seq_encoding,
            chunk::SeqEncoding::BaseOneHot { context: 5 }
        );
        let cfg = build_default_config(0, None, None);
        assert_eq!(
            cfg.spec.seq_encoding,
            chunk::SeqEncoding::BaseOneHot { context: 0 }
        );
    }

    #[test]
    fn training_spec_never_uses_signal_kmer_encoding_even_when_requested() {
        // training.rs cuts with `cfg.training_spec`, never `cfg.spec`, so
        // `cut_chunk` can never take the `SeqEncoding::SignalKmer` branch (and
        // therefore can never drop a chunk over a failed internal
        // `signal_kmer_inputs` call) on the training path -- regardless of
        // what encoding the run actually requested. This is what makes the
        // "the whole crash class from issue #185, reintroduced through
        // training's shared spec" bug structurally impossible to reintroduce,
        // rather than merely untriggered by the current test fixtures.
        let cfg = build_config(
            true,
            "reference",
            "signal_kmer",
            None,
            200,
            200,
            5,
            400,
            true,
            None,
            None,
            true,
            None,
            9,
            4,
            5,
            2,
            1,
            "center",
        )
        .expect("valid config");
        assert_eq!(
            cfg.spec.seq_encoding,
            chunk::SeqEncoding::SignalKmer {
                ctx: escapepod_signal::seq_encoding::KmerContext {
                    before: 4,
                    after: 4
                }
            }
        );
        assert_eq!(cfg.training_spec.seq_encoding, chunk::SeqEncoding::None);
        // Every other field stays in sync between the two specs -- only
        // `seq_encoding` diverges.
        assert_eq!(cfg.spec.signal_context, cfg.training_spec.signal_context);
        assert_eq!(
            cfg.spec.feature_channels,
            cfg.training_spec.feature_channels
        );
    }

    #[test]
    fn feature_window_defaults_to_the_kmer_context_when_unset() {
        // feature_start/feature_end default to +/-kmer_context.
        let cfg = build_default_config(5, None, None);
        assert_eq!(cfg.spec.feature_offsets, (-5, 5));
        assert_eq!(cfg.spec.feature_width(), 11);
    }

    #[test]
    fn feature_window_honors_an_explicit_feature_start_of_zero() {
        // feature_start = 0 is a legitimate, non-default window (features
        // begin AT the focus base -- the right-only window for tRNA 3' ends)
        // and must not be mistaken for "unset". `Option::unwrap_or` does not
        // have that failure mode because `Some(0)` is not `None`, but the
        // derivation is worth pinning directly (issue #189).
        let cfg = build_default_config(5, Some(0), None);
        assert_eq!(cfg.spec.feature_offsets, (0, 5));
        assert_eq!(cfg.spec.feature_width(), 6);
    }

    #[test]
    fn feature_window_honors_a_fully_explicit_feature_window() {
        let cfg = build_default_config(5, Some(-2), Some(3));
        assert_eq!(cfg.spec.feature_offsets, (-2, 3));
        assert_eq!(cfg.spec.feature_width(), 6);
    }

    #[test]
    fn feature_window_honors_an_explicit_feature_end_with_start_defaulted() {
        // feat_start/feat_end resolve independently -- exercise the mirror
        // case of the test above (only feature_end explicit) so a regression
        // that defaults only one side correctly cannot hide behind the
        // other's coverage.
        let cfg = build_default_config(5, None, Some(2));
        assert_eq!(cfg.spec.feature_offsets, (-5, 2));
        assert_eq!(cfg.spec.feature_width(), 8);
    }

    // These three assert `.is_err()` only, not the message text: `PyErr`'s
    // `Display` materializes the underlying Python exception object, which
    // needs a running interpreter (`Python::with_gil`) -- unavailable to a
    // plain `#[test]` binary without pyo3's `auto-initialize` feature (not
    // enabled here, since it would also affect the real extension build).
    // `PyValueError::new_err` itself stays lazy and needs no interpreter, so
    // `.is_err()` alone still proves the acceptance criterion: these raise
    // rather than silently choosing a default geometry.

    #[test]
    fn an_unrecognised_anchor_is_a_value_error_not_a_silent_default() {
        let result = build_config(
            true,
            "bogus",
            "base_onehot",
            None,
            200,
            200,
            5,
            400,
            true,
            None,
            None,
            true,
            None,
            9,
            4,
            5,
            2,
            1,
            "center",
        );
        assert!(result.is_err());
    }

    #[test]
    fn an_unrecognised_seq_encoding_is_a_value_error() {
        let result = build_config(
            true,
            "reference",
            "bogus",
            None,
            200,
            200,
            5,
            400,
            true,
            None,
            None,
            true,
            None,
            9,
            4,
            5,
            2,
            1,
            "center",
        );
        assert!(result.is_err());
    }

    /// rnabioco/escapepod-rs#388 / rnabioco/leech#343: predict's Rust path
    /// (`process_one_read`) relies entirely on `cut_chunk`'s internal
    /// `SignalKmer` arm -- unlike `training.rs`, which makes its own
    /// corrected second call -- so it can only be fixed by the pinned
    /// `escapepod-signal` crate itself. This test exercises the *exact*
    /// per-chunk wiring `process_one_read` uses (the same
    /// `resolve_signal_context_bases` + `base_justify.focus` +
    /// `per_base_spec.signal_context` construction, copied verbatim below)
    /// against a hand-verified fixture, rather than trusting the pin bump.
    ///
    /// Fixture: `signal = [0..10) as f32`, `seq_to_sig = [0,2,4,6,8,10]` (5
    /// bases of 2 samples each), `sequence = b"ACGTA"`. `base_idx=2` ('G'),
    /// `signal_context_bases=(1,3)` resolves to the sample window `[2, 10)`
    /// (8 samples) -- wider than `signal_len=4`, forcing the centre-crop
    /// branch (`crop = (8-4)/2 = 2`, placed window `[4, 8)` ==
    /// `signal[4..8] = [4,5,6,7]`).
    ///
    /// Base 2's real samples are `seq_to_sig[2..4) = [4, 6)` -- i.e. raw
    /// indices 4 and 5, which land at *placed-window-relative* columns 0-1
    /// (not 2-3, which is what the pre-#388 pre-crop origin would have
    /// produced). Row `4*kmer_pos(1) + base('G'=2) = 6` is base 2's
    /// own-identity channel (kmer_pos=1 is the centre of a `before=1,
    /// after=1` context); it must be hot at columns `[0, 2)` and zero at
    /// `[2, 4)`.
    #[test]
    fn signal_kmer_predict_path_aligns_through_centre_crop() {
        let processed = chunk::ProcessedRead {
            signal: (0..10).map(|i| i as f32).collect(),
            seq_to_sig: vec![0, 2, 4, 6, 8, 10],
            sequence: b"ACGTA".to_vec(),
            levels: None,
        };
        let mut cfg = build_config(
            true,
            "reference",
            "signal_kmer",
            Some((1, 1)),
            200,
            200,
            5,
            4, // signal_len -- narrower than the 8-sample requested window
            false,
            None,
            None,
            true,
            None,
            9,
            4,
            5,
            2,
            1,
            "start",
        )
        .expect("valid config");
        cfg.spec.signal_len = 4;

        let n_bases = processed.n_bases();
        let base_idx: i64 = 2;
        let bi = base_idx as usize;
        let rows = chunk::read_rows(&processed, &cfg.spec);

        // Verbatim copy of process_one_read's per-chunk spec construction.
        let (sample_start, sample_end) =
            resolve_signal_context_bases(&processed.seq_to_sig, base_idx, 1, 3, n_bases);
        assert_eq!((sample_start, sample_end), (2, 10));
        let focus = cfg
            .spec
            .base_justify
            .focus(processed.seq_to_sig[bi], processed.seq_to_sig[bi + 1]);
        let mut per_base_spec = cfg.spec.clone();
        per_base_spec.signal_context = (focus - sample_start, sample_end - focus);

        let c = chunk::cut_chunk(&processed, &rows, &per_base_spec, base_idx)
            .expect("chunk should be produced");

        assert_eq!(c.signal, vec![4.0, 5.0, 6.0, 7.0]);

        let channels = 4 * 3; // KmerContext(1,1) -> kmer_len 3
        assert_eq!(c.sequence.len(), channels * cfg.spec.signal_len);
        let row = 4 * 1 + 2; // kmer_pos=1 (centre), base 'G' = 2
        let start = row * cfg.spec.signal_len;
        assert_eq!(
            &c.sequence[start..start + cfg.spec.signal_len],
            &[1.0, 1.0, 0.0, 0.0],
            "base 2's own-identity channel must be hot where its real signal \
             (indices 4-5, placed-window columns 0-1) actually landed, not \
             shifted by the crop amount"
        );
    }

    #[test]
    fn an_unrecognised_base_justify_is_a_value_error() {
        let result = build_config(
            true,
            "reference",
            "base_onehot",
            None,
            200,
            200,
            5,
            400,
            true,
            None,
            None,
            true,
            None,
            9,
            4,
            5,
            2,
            1,
            "bogus",
        );
        assert!(result.is_err());
    }
}
