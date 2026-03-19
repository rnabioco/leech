//! Monolithic per-read feature extraction for the inference hot path.
//!
//! Replaces ~10 Python steps per read with a single Rust call:
//! POD5 signal → normalize → reference anchoring → signal refinement →
//! dwell/signal features → chunk extraction → sequence encoding.
//!
//! Supports all production config options: reference anchoring, signal map
//! refinement, signal_kmer encoding, and multi-channel signal (kmer residual).

use std::collections::{HashMap, HashSet};

use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::core_pipeline::*;
use crate::encoding::encode_signal_kmer_inner;
use crate::pod5_io::PreloadedSignals;
use crate::signal_refine::extract_levels_inner;

// ---------------------------------------------------------------------------
// Per-read result (plain Rust, no PyO3 — safe for rayon)
// ---------------------------------------------------------------------------

struct ChunkResult {
    signal: Vec<f32>,
    seq_enc: Vec<f32>,
    seq_rows: usize,
    seq_cols: usize,
    features: Option<Vec<f32>>,
    num_features: usize,
    dwell_width: usize,
    read_id: String,
    base_idx: i64,
}

/// Process one read entirely in Rust. Returns extracted chunks.
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
    // Config (shared across reads)
    cfg: &PipelineConfig,
    // Optional per-read data
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&str>,
) -> Vec<ChunkResult> {
    let trim_start = trim.max(0) as usize;
    let trim_end = (ns as usize).min(raw_i16.len());
    if trim_start >= trim_end {
        return vec![];
    }

    let mut trimmed_f32: Vec<f32> = raw_i16[trim_start..trim_end]
        .iter()
        .map(|&x| x as f32)
        .collect();
    let mut query_to_sig = build_seq_to_sig_map(mv, stride, trim, ns);

    if cfg.reverse_signal {
        trimmed_f32.reverse();
        let sig_len_val = trimmed_f32.len() as i64;
        query_to_sig.reverse();
        for val in query_to_sig.iter_mut() {
            *val = sig_len_val - *val;
        }
    }

    let mut norm_signal = normalize_median_mad(&trimmed_f32);

    // Reference anchoring
    let (mut seq_to_sig, use_sequence) = if cfg.use_reference {
        if let (Some(cigar), Some(rseq)) = (cigar_ops, ref_seq) {
            let ref_to_sig = compute_ref_to_signal(&query_to_sig, cigar);
            if ref_to_sig.len() < 2 {
                return vec![];
            }
            let sig_start = ref_to_sig[0].max(0) as usize;
            let sig_end = (*ref_to_sig.last().unwrap()).min(norm_signal.len() as i64) as usize;
            if sig_start >= sig_end {
                return vec![];
            }
            norm_signal = norm_signal[sig_start..sig_end].to_vec();
            let shifted: Vec<i64> = ref_to_sig.iter().map(|&v| v - sig_start as i64).collect();
            (shifted, rseq.to_string())
        } else {
            (query_to_sig.clone(), sequence.to_string())
        }
    } else {
        (query_to_sig.clone(), sequence.to_string())
    };

    let num_bases = seq_to_sig.len().saturating_sub(1);
    if num_bases == 0 {
        return vec![];
    }

    // Signal refinement
    let mut expected_levels_f64: Option<Vec<f64>> = None;
    if cfg.refine_signal_map {
        if let Some(ref kt) = cfg.kmer_table {
            refine_signal_map_pipeline(
                &mut norm_signal, &mut seq_to_sig, &use_sequence,
                kt, cfg.kmer_len, cfg.kmer_center_idx,
                cfg.refine_half_bandwidth, cfg.refine_scale_iters,
            );
            expected_levels_f64 = Some(extract_levels_inner(
                &use_sequence, kt, cfg.kmer_len, cfg.kmer_center_idx,
            ));
        }
    }

    let num_bases = seq_to_sig.len().saturating_sub(1);
    if num_bases == 0 {
        return vec![];
    }

    // Features
    let dwells: Vec<f32> = (0..num_bases)
        .map(|j| (seq_to_sig[j + 1] - seq_to_sig[j]) as f32)
        .collect();

    let features_data: Option<Vec<Vec<f32>>> = if cfg.compute_features {
        let (means, medians, stds, ranges) = compute_per_base_stats(&norm_signal, &seq_to_sig);
        let mut feats = compute_dwell_features(&dwells);
        feats.push(means);
        feats.push(medians);
        feats.push(stds);
        feats.push(ranges);
        if let Some(ref levels) = expected_levels_f64 {
            let (ke, kr, kra) = compute_kmer_residual_features(&norm_signal, &seq_to_sig, levels, num_bases);
            feats.push(ke);
            feats.push(kr);
            feats.push(kra);
        }
        Some(feats)
    } else {
        None
    };
    let num_features = features_data.as_ref().map(|f| f.len()).unwrap_or(0);

    let sig_residual: Option<Vec<f32>> = if cfg.signal_in_channels > 1 {
        expected_levels_f64.as_ref().map(|levels| {
            compute_signal_residual(&norm_signal, &seq_to_sig, levels, num_bases)
        })
    } else {
        None
    };

    // Extract chunks
    let seq_bytes = use_sequence.as_bytes();
    let mut results = Vec::new();

    for &base_idx in positions {
        let bi = base_idx as usize;
        let kmer_start = base_idx - cfg.kmer_ctx;
        let kmer_end = base_idx + cfg.kmer_ctx + 1;
        if kmer_start < 0 || kmer_end > seq_bytes.len() as i64 || bi >= num_bases {
            continue;
        }

        let focus_sig = (seq_to_sig[bi] + seq_to_sig[bi + 1]) / 2;
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
        let final_signal = if let Some(ref residual) = sig_residual {
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
            let chunk_kmer_start = (base_idx - cfg.kmer_ctx) as usize;
            let chunk_kmer_end = (base_idx + cfg.kmer_ctx + 1) as usize;
            let (kmer_before, kmer_after) = cfg.skmer_ctx;
            let seq_with_ctx_start = chunk_kmer_start.saturating_sub(kmer_before);
            let seq_with_ctx_end = (chunk_kmer_end + kmer_after).min(seq_bytes.len());
            let seq_with_ctx = &seq_bytes[seq_with_ctx_start..seq_with_ctx_end];
            let seq_ints = sequence_to_int(seq_with_ctx);
            let map_start = chunk_kmer_start;
            let map_end = (chunk_kmer_end + 1).min(seq_to_sig.len());
            if map_start >= map_end { continue; }
            let chunk_sig_map: Vec<i64> = seq_to_sig[map_start..map_end]
                .iter().map(|&v| v - sig_start_pos).collect();
            let enc_dim = 4 * (kmer_before + 1 + kmer_after);
            let flat = encode_signal_kmer_inner(&seq_ints, &chunk_sig_map, cfg.signal_len, kmer_before, kmer_after);
            (flat, enc_dim, cfg.signal_len)
        } else {
            let kmer_seq = &seq_bytes[kmer_start as usize..kmer_end as usize];
            let flat = encode_base_onehot(kmer_seq);
            (flat, 4, cfg.kmer_win)
        };

        // Features
        let feat_arr: Option<Vec<f32>> = if let Some(ref all_feats) = features_data {
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

// ---------------------------------------------------------------------------
// Shared Phase 2 (rayon) + Phase 3 (numpy) — called by both entry points
// ---------------------------------------------------------------------------

/// Run per-read processing in parallel (rayon, GIL released) then convert
/// results to numpy arrays.  Shared by [`extract_inference_chunks`] and
/// [`extract_chunks_from_preloaded`].
#[allow(clippy::too_many_arguments)]
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
) -> PyResult<
    Vec<(
        Py<PyArray1<f32>>,
        Py<PyArray2<f32>>,
        Option<Py<PyArray2<f32>>>,
        String,
        i64,
    )>,
> {
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
            let seq_arr = numpy::ndarray::Array2::from_shape_vec((c.seq_rows, c.seq_cols), c.seq_enc)
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
// PyO3 entry point — monolithic (POD5 I/O + processing)
// ---------------------------------------------------------------------------

/// Extract inference-ready chunks for a batch of reads in one Rust call.
///
/// Full production pipeline: POD5 → normalize → reference anchoring →
/// signal refinement → features → chunk extraction → sequence encoding.
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
    dwell_offset = 0,
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
))]
#[allow(clippy::too_many_arguments, unused_variables)]
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
    dwell_offset: i64,
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
) -> PyResult<
    Vec<(
        Py<PyArray1<f32>>,
        Py<PyArray2<f32>>,
        Option<Py<PyArray2<f32>>>,
        String,
        i64,
    )>,
> {
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

    let kmer_ctx = kmer_context;

    // Build shared config
    let cfg = PipelineConfig {
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
        dwell_width: (feature_end.unwrap_or(kmer_ctx) - feature_start.unwrap_or(-kmer_ctx) + 1) as usize,
        refine_signal_map,
        kmer_table,
        kmer_len,
        kmer_center_idx,
        refine_half_bandwidth,
        refine_scale_iters,
        signal_in_channels,
    };

    // --- Phase 1: POD5 I/O (scan reads, UUID filter, early termination, bulk extract) ---
    let target_uuids: HashSet<escapepod::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod::Uuid::parse_str(s).ok())
        .collect();
    let reader = escapepod::Reader::open(pod5_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to open POD5: {e}"))
    })?;

    let mut to_extract: Vec<(String, Vec<u64>)> = Vec::with_capacity(target_uuids.len());
    for read_result in reader.reads().map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to iterate reads: {e}"))
    })? {
        let read = read_result.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to parse read: {e}"))
        })?;
        if target_uuids.contains(&read.read_id) {
            to_extract.push((read.read_id.to_string(), read.signal_rows));
            if to_extract.len() == target_uuids.len() {
                break;
            }
        }
    }

    let bulk_signals = reader.get_signal_bulk(&to_extract).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Signal extraction failed: {e}"))
    })?;
    let signal_map: HashMap<String, Vec<i16>> = bulk_signals.into_iter().collect();

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
// PyO3 entry point — from preloaded signals (prefetch pipeline)
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
    dwell_offset = 0,
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
))]
#[allow(clippy::too_many_arguments, unused_variables)]
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
    dwell_offset: i64,
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
) -> PyResult<
    Vec<(
        Py<PyArray1<f32>>,
        Py<PyArray2<f32>>,
        Option<Py<PyArray2<f32>>>,
        String,
        i64,
    )>,
> {
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

    let kmer_ctx = kmer_context;

    let cfg = PipelineConfig {
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
    };

    // Skip Phase 1 — use preloaded signals directly
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
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>, Py<PyArray1<f32>>, Py<PyArray2<f32>>)> {
    let trim_start = trim_offset.max(0) as usize;
    let trim_end = (num_samples as usize).min(raw_signal.len());
    if trim_start >= trim_end {
        return Err(pyo3::exceptions::PyValueError::new_err("Empty signal after trimming"));
    }

    let mut trimmed_f32: Vec<f32> = raw_signal[trim_start..trim_end].iter().map(|&x| x as f32).collect();
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

    let dwells: Vec<f32> = (0..num_bases).map(|j| (sig_map[j + 1] - sig_map[j]) as f32).collect();
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
