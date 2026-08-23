//! Inference chunk extraction, parallel processing, and PyO3 entry points.

use std::collections::HashMap;

use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::encoding::encode_signal_kmer_inner;
use crate::pod5_io::PreloadedSignals;

use super::features::{
    compute_dwell_features, compute_per_base_stats, encode_base_onehot, sequence_to_int,
};
use super::numeric::normalize_median_mad;
use super::processing::process_read_signal;
use super::signal_mapping::{
    build_seq_to_sig_map, chunk_signal_kmer_inputs, compute_ref_to_signal,
};
use super::types::{BaseJustify, ChunkResult, PipelineConfig, ProcessedRead};

/// One inference chunk returned to Python: (signal, seq_encoding, features?, read_id, base_idx).
type InferenceChunkPy = (
    Py<PyArray1<f32>>,
    Py<PyArray2<f32>>,
    Option<Py<PyArray2<f32>>>,
    String,
    i64,
);

/// Test helper return: (norm_signal, sig_map, dwells, features_2d).
type TestProcessReadResult = (
    Py<PyArray1<f32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f32>>,
    Py<PyArray2<f32>>,
);

/// Process one read for inference. Uses shared signal processing,
/// then extracts inference-encoded chunks.
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
    cfg: &PipelineConfig,
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&str>,
) -> Vec<ChunkResult> {
    let processed = match process_read_signal(
        raw_i16, sequence, mv, stride, ns, trim, cfg, cigar_ops, ref_seq,
    ) {
        Some(p) => p,
        None => return vec![],
    };

    let ProcessedRead {
        ref norm_signal,
        ref seq_to_sig,
        ref use_sequence,
        ref features_data,
        num_features,
        ref sig_residual,
        ..
    } = processed;
    let num_bases = seq_to_sig.len().saturating_sub(1);

    // Extract inference chunks
    let seq_bytes = use_sequence.as_bytes();
    let mut results = Vec::new();

    for &base_idx in positions {
        // Same rule as training extraction and as `LeechRead.get_chunk`: only
        // a focus base without signal boundaries is dropped. A k-mer window
        // overhanging the sequence is 'N'-padded, which both encoders below
        // already handle (all-zero column / -1 skipped). Skipping it instead
        // silently withheld predictions for reads whose aligned region stops
        // near the motif -- the training-side half of this was issue #185.
        if base_idx < 0 || base_idx as usize >= num_bases {
            continue;
        }
        let bi = base_idx as usize;
        let kmer_start = base_idx - cfg.kmer_ctx;
        let kmer_end = base_idx + cfg.kmer_ctx + 1;

        let focus_sig = cfg
            .base_justify
            .focus_pos(seq_to_sig[bi], seq_to_sig[bi + 1]);
        let sig_start_pos = focus_sig - cfg.signal_context_left;
        let sig_end_pos = focus_sig + cfg.signal_context_right;
        let actual_len = (sig_end_pos - sig_start_pos) as usize;

        let mut sig_chunk = vec![0.0f32; cfg.signal_len];
        if actual_len <= cfg.signal_len {
            let src_lo = sig_start_pos.max(0) as usize;
            let src_hi = sig_end_pos.min(norm_signal.len() as i64) as usize;
            let dst_off = (sig_start_pos.max(0) - sig_start_pos) as usize;
            if src_lo < src_hi {
                let n = (src_hi - src_lo).min(cfg.signal_len - dst_off);
                sig_chunk[dst_off..dst_off + n].copy_from_slice(&norm_signal[src_lo..src_lo + n]);
            }
        } else {
            let crop_start = (actual_len - cfg.signal_len) / 2;
            let abs_start = (sig_start_pos + crop_start as i64).max(0) as usize;
            let abs_end = (abs_start + cfg.signal_len).min(norm_signal.len());
            let n = abs_end - abs_start;
            sig_chunk[..n].copy_from_slice(&norm_signal[abs_start..abs_end]);
        }

        // Multi-channel
        let final_signal = if let Some(residual) = sig_residual {
            let mut res_chunk = vec![0.0f32; cfg.signal_len];
            if actual_len <= cfg.signal_len {
                let src_lo = sig_start_pos.max(0) as usize;
                let src_hi = sig_end_pos.min(residual.len() as i64) as usize;
                let dst_off = (sig_start_pos.max(0) - sig_start_pos) as usize;
                if src_lo < src_hi {
                    let n = (src_hi - src_lo).min(cfg.signal_len - dst_off);
                    res_chunk[dst_off..dst_off + n].copy_from_slice(&residual[src_lo..src_lo + n]);
                }
            } else {
                let crop_start = (actual_len - cfg.signal_len) / 2;
                let abs_start = (sig_start_pos + crop_start as i64).max(0) as usize;
                let abs_end = (abs_start + cfg.signal_len).min(residual.len());
                let n = abs_end - abs_start;
                res_chunk[..n].copy_from_slice(&residual[abs_start..abs_end]);
            }
            let mut stacked = Vec::with_capacity(2 * cfg.signal_len);
            stacked.extend_from_slice(&sig_chunk);
            stacked.extend_from_slice(&res_chunk);
            stacked
        } else {
            sig_chunk
        };

        // Sequence encoding
        let (seq_flat, seq_rows, seq_cols) = if cfg.use_signal_kmer {
            // Derived from the SIGNAL window, matching `LeechRead.get_chunk`,
            // which is what the Python inference path feeds the same encoder.
            // Deriving them from the k-mer window disagreed with it on every
            // chunk (issue #186).
            let (kmer_before, kmer_after) = cfg.skmer_ctx;
            let (chunk_sig_map, ctx_bytes) = chunk_signal_kmer_inputs(
                seq_to_sig,
                seq_bytes,
                sig_start_pos,
                sig_end_pos,
                norm_signal.len(),
                cfg.signal_len,
                cfg.skmer_ctx,
            );
            if chunk_sig_map.is_empty() {
                continue;
            }
            let seq_ints = sequence_to_int(&ctx_bytes);
            let enc_dim = 4 * (kmer_before + 1 + kmer_after);
            let flat = encode_signal_kmer_inner(
                &seq_ints,
                &chunk_sig_map,
                cfg.signal_len,
                kmer_before,
                kmer_after,
            );
            (flat, enc_dim, cfg.signal_len)
        } else {
            // 'N' past either end, so the window is always `kmer_win` wide and
            // the encoder emits an all-zero column there.
            let kmer_seq: Vec<u8> = (kmer_start..kmer_end)
                .map(|i| match usize::try_from(i) {
                    Ok(u) if u < seq_bytes.len() => seq_bytes[u],
                    _ => b'N',
                })
                .collect();
            let flat = encode_base_onehot(&kmer_seq);
            (flat, 4, cfg.kmer_win)
        };

        // Features
        let feat_arr: Option<Vec<f32>> = if let Some(all_feats) = features_data {
            let fs = base_idx + cfg.feat_start;
            let fe = base_idx + cfg.feat_end + 1;
            let safe_start = fs.max(0) as usize;
            let safe_end = fe.min(num_bases as i64) as usize;
            let left_pad = (0i64 - fs).max(0) as usize;
            let mut flat = vec![0.0f32; num_features * cfg.dwell_width];
            for (f_idx, feat_row) in all_feats.iter().enumerate() {
                if safe_start < safe_end && safe_end <= feat_row.len() {
                    let n = (safe_end - safe_start).min(cfg.dwell_width - left_pad);
                    for k in 0..n {
                        flat[f_idx * cfg.dwell_width + left_pad + k] = feat_row[safe_start + k];
                    }
                }
            }
            Some(flat)
        } else {
            None
        };

        results.push(ChunkResult {
            signal: final_signal,
            seq_enc: seq_flat,
            seq_rows,
            seq_cols,
            features: feat_arr,
            num_features,
            dwell_width: cfg.dwell_width,
            read_id: rid.to_string(),
            base_idx,
        });
    }

    results
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
    mv_arrays: &[Vec<u8>],
    mv_strides: &[u32],
    num_samples_list: &[u64],
    trim_offsets: &[i64],
    motif_positions: &[Vec<i64>],
    cfg: &PipelineConfig,
    cigar_tuples: &Option<Vec<Vec<(u32, u32)>>>,
    reference_sequences: &Option<Vec<Option<String>>>,
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
                        &mv_arrays[i],
                        mv_strides[i],
                        num_samples_list[i],
                        trim_offsets[i],
                        &motif_positions[i],
                        cfg,
                        cigar,
                        rseq,
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

/// Build a PipelineConfig from the common parameters shared by all entry points.
#[allow(clippy::too_many_arguments)]
fn build_config(
    reverse_signal: bool,
    anchor: &str,
    seq_encoding: &str,
    signal_kmer_context: Option<(usize, usize)>,
    signal_context_left: i64,
    signal_context_right: i64,
    kmer_context: i64,
    signal_len: usize,
    compute_features: bool,
    feature_start: Option<i64>,
    feature_end: Option<i64>,
    refine_signal_map: bool,
    kmer_table: Option<HashMap<String, f64>>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
    base_justify: &str,
) -> PipelineConfig {
    let kmer_ctx = kmer_context;
    PipelineConfig {
        reverse_signal,
        use_reference: anchor == "reference",
        use_signal_kmer: seq_encoding == "signal_kmer",
        skmer_ctx: signal_kmer_context.unwrap_or((4, 4)),
        signal_context_left,
        signal_context_right,
        kmer_ctx,
        kmer_win: (2 * kmer_ctx + 1) as usize,
        signal_len,
        compute_features,
        feat_start: feature_start.unwrap_or(-kmer_ctx),
        feat_end: feature_end.unwrap_or(kmer_ctx),
        dwell_width: (feature_end.unwrap_or(kmer_ctx) - feature_start.unwrap_or(-kmer_ctx) + 1)
            as usize,
        refine_signal_map,
        kmer_table,
        kmer_len,
        kmer_center_idx,
        refine_half_bandwidth,
        refine_scale_iters,
        signal_in_channels,
        base_justify: BaseJustify::from_str(base_justify),
    }
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
))]
#[allow(clippy::too_many_arguments)]
pub fn extract_inference_chunks<'py>(
    py: Python<'py>,
    pod5_path: &str,
    read_ids: Vec<String>,
    sequences: Vec<String>,
    mv_strides: Vec<u32>,
    mv_arrays: Vec<Vec<u8>>,
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
    kmer_table: Option<HashMap<String, f64>>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
    base_justify: &str,
) -> PyResult<Vec<InferenceChunkPy>> {
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
    );

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
        &mv_arrays,
        &mv_strides,
        &num_samples_list,
        &trim_offsets,
        &motif_positions,
        &cfg,
        &cigar_tuples,
        &reference_sequences,
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
))]
#[allow(clippy::too_many_arguments)]
pub fn extract_chunks_from_preloaded<'py>(
    py: Python<'py>,
    preloaded: &PreloadedSignals,
    read_ids: Vec<String>,
    sequences: Vec<String>,
    mv_strides: Vec<u32>,
    mv_arrays: Vec<Vec<u8>>,
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
    kmer_table: Option<HashMap<String, f64>>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
    base_justify: &str,
) -> PyResult<Vec<InferenceChunkPy>> {
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
    );

    // Skip Phase 1 -- use preloaded signals directly
    _process_and_convert(
        py,
        &preloaded.signals,
        &read_ids,
        &sequences,
        &mv_arrays,
        &mv_strides,
        &num_samples_list,
        &trim_offsets,
        &motif_positions,
        &cfg,
        &cigar_tuples,
        &reference_sequences,
    )
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Process a single read through the Rust pipeline (no POD5, for testing).
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
    let trim_start = trim_offset.max(0) as usize;
    let trim_end = (num_samples as usize).min(raw_signal.len());
    if trim_start >= trim_end {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Empty signal after trimming",
        ));
    }

    let mut trimmed_f32: Vec<f32> = raw_signal[trim_start..trim_end]
        .iter()
        .map(|&x| x as f32)
        .collect();
    let mut sig_map = build_seq_to_sig_map(&mv_array, stride, trim_offset, num_samples);

    if reverse_signal {
        trimmed_f32.reverse();
        let sig_len = trimmed_f32.len() as i64;
        sig_map.reverse();
        for val in sig_map.iter_mut() {
            *val = sig_len - *val;
        }
    }

    let norm_signal = normalize_median_mad(&trimmed_f32);
    let num_bases = sig_map.len().saturating_sub(1);
    if num_bases == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("No bases"));
    }

    let dwells: Vec<f32> = (0..num_bases)
        .map(|j| (sig_map[j + 1] - sig_map[j]) as f32)
        .collect();
    let (means, medians, stds, ranges) = compute_per_base_stats(&norm_signal, &sig_map);
    let mut all_feats = compute_dwell_features(&dwells);
    all_feats.push(means);
    all_feats.push(medians);
    all_feats.push(stds);
    all_feats.push(ranges);

    let sig_py = norm_signal.into_pyarray(py).unbind();
    let map_py = sig_map.into_pyarray(py).unbind();
    let dwell_py = dwells.into_pyarray(py).unbind();

    let n_feats = all_feats.len();
    let mut feat_flat = vec![0.0f32; n_feats * num_bases];
    for (f_idx, row) in all_feats.iter().enumerate() {
        feat_flat[f_idx * num_bases..(f_idx + 1) * num_bases].copy_from_slice(row);
    }
    let feat_arr = numpy::ndarray::Array2::from_shape_vec((n_feats, num_bases), feat_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let feat_py = feat_arr.into_pyarray(py).unbind();

    Ok((sig_py, map_py, dwell_py, feat_py))
}

/// Expose compute_ref_to_signal for direct Python<->Rust comparison testing.
#[pyfunction]
#[pyo3(signature = (query_to_sig, cigar_ops))]
pub fn _test_ref_to_signal(
    query_to_sig: Vec<i64>,
    cigar_ops: Vec<(u32, u32)>,
) -> PyResult<Vec<i64>> {
    Ok(compute_ref_to_signal(&query_to_sig, &cigar_ops))
}
