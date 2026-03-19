//! Rust-accelerated data preparation pipeline.
//!
//! Replaces Python multiprocessing with in-process rayon parallelism for
//! extracting training chunks from POD5/BAM data. Reuses all shared processing
//! logic from `core_pipeline` (same code path as inference).
//!
//! Entry point: `extract_training_chunks_batch()` — called from Python via PyO3.

use std::collections::{HashMap, HashSet};

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;

use crate::core_pipeline::*;
use crate::encoding::encode_signal_kmer_inner;
use crate::motif_search;
use crate::signal_refine::extract_levels_inner;

// ---------------------------------------------------------------------------
// Per-read result (plain Rust, no PyO3 — safe for rayon)
// ---------------------------------------------------------------------------

struct TrainingChunkResult {
    signal: Vec<f32>,              // (signal_len,) or (2*signal_len,) with residual
    sequence: String,              // kmer sequence
    dwell: Vec<f32>,               // (dwell_width,)
    features: Vec<f32>,            // (num_features * dwell_width,) row-major
    num_features: usize,
    dwell_width: usize,
    base_idx: i64,
    read_id: String,
    feature_start: i64,
    feature_end: i64,
    seq_to_sig_map: Vec<i64>,     // variable length
    sequence_with_kmer_context: String,
    signal_residual: Option<Vec<f32>>, // (signal_len,)
    // Sequence encoding for onehot/signal_kmer
    seq_enc: Vec<f32>,
    seq_rows: usize,
    seq_cols: usize,
}

// ---------------------------------------------------------------------------
// Per-read processing — extends inference pipeline with motif search
// ---------------------------------------------------------------------------

/// Training-specific pipeline config (extends PipelineConfig with motif search).
struct TrainingConfig {
    /// Base pipeline config (shared with inference)
    base: PipelineConfig,
    /// Motif search parameters
    motif: Option<String>,
    motif_offset: i32,
    motif_reference: String,  // "bam" or "fasta"
    skip_motif_indels: bool,
    allow_edge_indels: bool,
    /// Normalization method
    norm_method: String,
}

/// Process one read for training chunk extraction.
///
/// Similar to inference `process_one_read` but:
/// 1. Performs motif search (inference receives positions pre-computed)
/// 2. Returns richer per-chunk output (seq_to_sig_map, feature_start/end, etc.)
/// 3. Supports all normalization methods (inference only uses median_mad)
#[allow(clippy::too_many_arguments)]
fn process_one_read_training(
    raw_i16: &[i16],
    rid: &str,
    sequence: &str,
    mv: &[u8],
    stride: u32,
    ns: u64,
    trim: i64,
    cfg: &TrainingConfig,
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&str>,
    ref_name: Option<&str>,
    ref_start: Option<u64>,
    ref_end: Option<u64>,
    ref_seq_dict: Option<&HashMap<String, String>>,
    cal_offset: f32,
    cal_scale: f32,
) -> Vec<TrainingChunkResult> {
    let trim_start = trim.max(0) as usize;
    let trim_end = (ns as usize).min(raw_i16.len());
    if trim_start >= trim_end {
        return vec![];
    }

    // --- Normalization ---
    let mut trimmed_f32: Vec<f32> = if cfg.norm_method == "pa_scaling" {
        // pa_scaling needs raw DAC values (before trimming to f32)
        let pa_mean = 0.0f32; // These come from the Python side via PipelineConfig
        let pa_stdev = 1.0f32;
        // For pa_scaling, we normalize the raw i16 directly
        normalize_pa_scaling(
            &raw_i16[trim_start..trim_end],
            cal_offset,
            cal_scale,
            pa_mean,
            pa_stdev,
        )
    } else {
        raw_i16[trim_start..trim_end]
            .iter()
            .map(|&x| x as f32)
            .collect()
    };

    let mut query_to_sig = build_seq_to_sig_map(mv, stride, trim, ns);

    if cfg.base.reverse_signal {
        trimmed_f32.reverse();
        let sig_len_val = trimmed_f32.len() as i64;
        query_to_sig.reverse();
        for val in query_to_sig.iter_mut() {
            *val = sig_len_val - *val;
        }
    }

    let mut norm_signal = if cfg.norm_method == "pa_scaling" {
        trimmed_f32.clone() // already normalized
    } else {
        normalize_signal(&trimmed_f32, &cfg.norm_method)
    };

    // --- Reference anchoring ---
    let (mut seq_to_sig, use_sequence) = if cfg.base.use_reference {
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

    // --- Signal refinement ---
    let mut expected_levels_f64: Option<Vec<f64>> = None;
    if cfg.base.refine_signal_map {
        if let Some(ref kt) = cfg.base.kmer_table {
            refine_signal_map_pipeline(
                &mut norm_signal, &mut seq_to_sig, &use_sequence,
                kt, cfg.base.kmer_len, cfg.base.kmer_center_idx,
                cfg.base.refine_half_bandwidth, cfg.base.refine_scale_iters,
            );
            expected_levels_f64 = Some(extract_levels_inner(
                &use_sequence, kt, cfg.base.kmer_len, cfg.base.kmer_center_idx,
            ));
        }
    }

    let num_bases = seq_to_sig.len().saturating_sub(1);
    if num_bases == 0 {
        return vec![];
    }

    // --- Features ---
    let dwells: Vec<f32> = (0..num_bases)
        .map(|j| (seq_to_sig[j + 1] - seq_to_sig[j]) as f32)
        .collect();

    let features_data: Option<Vec<Vec<f32>>> = if cfg.base.compute_features {
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

    let sig_residual: Option<Vec<f32>> = if cfg.base.signal_in_channels > 1 {
        expected_levels_f64.as_ref().map(|levels| {
            compute_signal_residual(&norm_signal, &seq_to_sig, levels, num_bases)
        })
    } else {
        None
    };

    // --- Motif search (training-specific) ---
    let positions: Vec<i64> = if let Some(ref motif) = cfg.motif {
        if cfg.motif_reference == "fasta" {
            // Reference-based motif search
            if let (Some(cigar), Some(rn), Some(rs), Some(re)) =
                (cigar_ops, ref_name, ref_start, ref_end)
            {
                // Look up full reference sequence from dict
                let full_ref = ref_seq_dict
                    .and_then(|d| d.get(rn))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                if full_ref.is_empty() {
                    return vec![];
                }
                let anchor = if cfg.base.use_reference { "reference" } else { "basecall" };
                motif_search::reference_motif_search(
                    full_ref,
                    rs as usize,
                    re as usize,
                    cigar,
                    rs,
                    motif,
                    cfg.motif_offset,
                    cfg.skip_motif_indels,
                    cfg.allow_edge_indels,
                    anchor,
                )
            } else {
                return vec![];
            }
        } else {
            // Basecalled motif search
            motif_search::basecalled_motif_search(sequence, motif, cfg.motif_offset)
        }
    } else {
        // No motif — use all bases (avoiding edges)
        (5..num_bases.saturating_sub(5)).map(|i| i as i64).collect()
    };

    if positions.is_empty() {
        return vec![];
    }

    // --- Extract chunks ---
    let seq_bytes = use_sequence.as_bytes();
    let mut results = Vec::new();
    let pcfg = &cfg.base;

    for &base_idx in &positions {
        let bi = base_idx as usize;
        let kmer_start = base_idx - pcfg.kmer_ctx;
        let kmer_end = base_idx + pcfg.kmer_ctx + 1;
        if kmer_start < 0 || kmer_end > seq_bytes.len() as i64 || bi >= num_bases {
            continue;
        }

        let focus_sig = (seq_to_sig[bi] + seq_to_sig[bi + 1]) / 2;
        let sig_start_pos = focus_sig - pcfg.signal_context_left;
        let sig_end_pos = focus_sig + pcfg.signal_context_right;
        let actual_len = (sig_end_pos - sig_start_pos) as usize;

        // --- Signal chunk extraction ---
        let mut sig_chunk = vec![0.0f32; pcfg.signal_len];
        let mut seq_to_sig_offset: i64 = 0;
        if actual_len <= pcfg.signal_len {
            let src_lo = sig_start_pos.max(0) as usize;
            let src_hi = sig_end_pos.min(norm_signal.len() as i64) as usize;
            let dst_off = (sig_start_pos.max(0) - sig_start_pos) as usize;
            if sig_start_pos < 0 {
                seq_to_sig_offset = -sig_start_pos;
            }
            if src_lo < src_hi {
                let n = (src_hi - src_lo).min(pcfg.signal_len - dst_off);
                sig_chunk[dst_off..dst_off + n].copy_from_slice(&norm_signal[src_lo..src_lo + n]);
            }
        } else {
            let crop_start = (actual_len - pcfg.signal_len) / 2;
            let abs_start = (sig_start_pos + crop_start as i64).max(0) as usize;
            let abs_end = (abs_start + pcfg.signal_len).min(norm_signal.len());
            let n = abs_end - abs_start;
            sig_chunk[..n].copy_from_slice(&norm_signal[abs_start..abs_end]);
        }

        // Multi-channel signal (with residual)
        let final_signal = if let Some(ref residual) = sig_residual {
            let mut res_chunk = vec![0.0f32; pcfg.signal_len];
            if actual_len <= pcfg.signal_len {
                let src_lo = sig_start_pos.max(0) as usize;
                let src_hi = sig_end_pos.min(residual.len() as i64) as usize;
                let dst_off = (sig_start_pos.max(0) - sig_start_pos) as usize;
                if src_lo < src_hi {
                    let n = (src_hi - src_lo).min(pcfg.signal_len - dst_off);
                    res_chunk[dst_off..dst_off + n].copy_from_slice(&residual[src_lo..src_lo + n]);
                }
            } else {
                let crop_start = (actual_len - pcfg.signal_len) / 2;
                let abs_start = (sig_start_pos + crop_start as i64).max(0) as usize;
                let abs_end = (abs_start + pcfg.signal_len).min(residual.len());
                let n = abs_end - abs_start;
                res_chunk[..n].copy_from_slice(&residual[abs_start..abs_end]);
            }
            let mut stacked = Vec::with_capacity(2 * pcfg.signal_len);
            stacked.extend_from_slice(&sig_chunk);
            stacked.extend_from_slice(&res_chunk);
            stacked
        } else {
            sig_chunk
        };

        // Separate signal residual for training output
        let signal_residual_chunk: Option<Vec<f32>> = if let Some(ref residual) = sig_residual {
            let mut res_chunk = vec![0.0f32; pcfg.signal_len];
            if actual_len <= pcfg.signal_len {
                let src_lo = sig_start_pos.max(0) as usize;
                let src_hi = sig_end_pos.min(residual.len() as i64) as usize;
                let dst_off = (sig_start_pos.max(0) - sig_start_pos) as usize;
                if src_lo < src_hi {
                    let n = (src_hi - src_lo).min(pcfg.signal_len - dst_off);
                    res_chunk[dst_off..dst_off + n].copy_from_slice(&residual[src_lo..src_lo + n]);
                }
            } else {
                let crop_start = (actual_len - pcfg.signal_len) / 2;
                let abs_start = (sig_start_pos + crop_start as i64).max(0) as usize;
                let abs_end = (abs_start + pcfg.signal_len).min(residual.len());
                let n = abs_end - abs_start;
                res_chunk[..n].copy_from_slice(&residual[abs_start..abs_end]);
            }
            Some(res_chunk)
        } else {
            None
        };

        // --- Sequence encoding ---
        let (seq_flat, seq_rows, seq_cols) = if pcfg.use_signal_kmer {
            let chunk_kmer_start = (base_idx - pcfg.kmer_ctx) as usize;
            let chunk_kmer_end = (base_idx + pcfg.kmer_ctx + 1) as usize;
            let (kmer_before, kmer_after) = pcfg.skmer_ctx;
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
            let flat = encode_signal_kmer_inner(&seq_ints, &chunk_sig_map, pcfg.signal_len, kmer_before, kmer_after);
            (flat, enc_dim, pcfg.signal_len)
        } else {
            let kmer_seq = &seq_bytes[kmer_start as usize..kmer_end as usize];
            let flat = encode_base_onehot(kmer_seq);
            (flat, 4, pcfg.kmer_win)
        };

        // --- Kmer sequence string ---
        let kmer_seq_str = std::str::from_utf8(&seq_bytes[kmer_start as usize..kmer_end as usize])
            .unwrap_or("")
            .to_string();

        // --- Feature extraction ---
        let (feat_arr, feat_dwell) = if let Some(ref all_feats) = features_data {
            let fs = base_idx + pcfg.feat_start;
            let fe = base_idx + pcfg.feat_end + 1;
            let safe_start = fs.max(0) as usize;
            let safe_end = fe.min(num_bases as i64) as usize;
            let left_pad = (0i64 - fs).max(0) as usize;
            let mut flat = vec![0.0f32; num_features * pcfg.dwell_width];
            for (f_idx, feat_row) in all_feats.iter().enumerate() {
                if safe_start < safe_end && safe_end <= feat_row.len() {
                    let n = (safe_end - safe_start).min(pcfg.dwell_width - left_pad);
                    for k in 0..n {
                        flat[f_idx * pcfg.dwell_width + left_pad + k] = feat_row[safe_start + k];
                    }
                }
            }

            // Extract dwell slice
            let mut dwell_chunk = vec![0.0f32; pcfg.dwell_width];
            if safe_start < safe_end && safe_end <= dwells.len() {
                let n = (safe_end - safe_start).min(pcfg.dwell_width - left_pad);
                dwell_chunk[left_pad..left_pad + n].copy_from_slice(&dwells[safe_start..safe_start + n]);
            }

            (flat, dwell_chunk)
        } else {
            (vec![], vec![0.0f32; pcfg.dwell_width])
        };

        // --- Chunk-relative seq_to_sig_map ---
        // Find sequence bases that fall within the signal window
        let chunk_seq_start = seq_to_sig.iter()
            .position(|&v| v >= sig_start_pos.max(0))
            .unwrap_or(0)
            .saturating_sub(1);
        let chunk_seq_end = seq_to_sig.iter()
            .position(|&v| v >= sig_end_pos)
            .unwrap_or(seq_to_sig.len())
            .min(seq_to_sig.len());
        let chunk_seq_start = chunk_seq_start.max(0);
        let chunk_map: Vec<i64> = seq_to_sig[chunk_seq_start..chunk_seq_end]
            .iter()
            .map(|&v| v - sig_start_pos + seq_to_sig_offset)
            .collect();

        // --- Extended sequence for signal_kmer encoding ---
        let (kmer_before, kmer_after) = pcfg.skmer_ctx;
        let ext_start = chunk_seq_start as i64 - kmer_before as i64;
        let ext_end = (chunk_seq_end as i64 + kmer_after as i64).min(seq_bytes.len() as i64);
        let mut seq_with_kmer_ctx = String::new();
        for i in ext_start..ext_end {
            if i >= 0 && (i as usize) < seq_bytes.len() {
                seq_with_kmer_ctx.push(seq_bytes[i as usize] as char);
            } else {
                seq_with_kmer_ctx.push('N');
            }
        }

        results.push(TrainingChunkResult {
            signal: final_signal,
            sequence: kmer_seq_str,
            dwell: feat_dwell,
            features: feat_arr,
            num_features,
            dwell_width: pcfg.dwell_width,
            base_idx,
            read_id: rid.to_string(),
            feature_start: pcfg.feat_start,
            feature_end: pcfg.feat_end,
            seq_to_sig_map: chunk_map,
            sequence_with_kmer_context: seq_with_kmer_ctx,
            signal_residual: signal_residual_chunk,
            seq_enc: seq_flat,
            seq_rows,
            seq_cols,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// PyO3 result type
// ---------------------------------------------------------------------------

/// Columnar result from `extract_training_chunks_batch`.
///
/// All arrays have the same leading dimension N (number of chunks).
/// This avoids per-chunk Python object overhead.
#[pyclass]
pub struct TrainingBatchResult {
    #[pyo3(get)]
    num_chunks: usize,
    #[pyo3(get)]
    signals: Py<PyAny>,           // ndarray (N, signal_out_len) f32
    #[pyo3(get)]
    sequences: Vec<String>,      // (N,)
    #[pyo3(get)]
    dwells: Py<PyAny>,            // ndarray (N, dwell_width) f32
    #[pyo3(get)]
    features: Py<PyAny>,          // ndarray (N, num_feat, dwell_width) f32
    #[pyo3(get)]
    read_ids: Vec<String>,       // (N,)
    #[pyo3(get)]
    base_indices: Py<PyAny>,      // ndarray (N,) i64
    #[pyo3(get)]
    feature_starts: Py<PyAny>,    // ndarray (N,) i64
    #[pyo3(get)]
    feature_ends: Py<PyAny>,      // ndarray (N,) i64
    #[pyo3(get)]
    seq_to_sig_maps: Vec<Py<PyAny>>,  // N variable-length ndarray i64
    #[pyo3(get)]
    sequences_with_kmer_context: Vec<String>,  // (N,)
    #[pyo3(get)]
    signal_residuals: Option<Py<PyAny>>,  // ndarray (N, signal_len) f32 or None
    #[pyo3(get)]
    seq_encodings: Vec<Py<PyAny>>,  // N ndarray (rows, cols) f32
}

#[pymethods]
impl TrainingBatchResult {
    fn __len__(&self) -> usize {
        self.num_chunks
    }
}

// ---------------------------------------------------------------------------
// PyO3 entry point
// ---------------------------------------------------------------------------

/// Extract training chunks for a batch of reads in one Rust call.
///
/// Full pipeline: POD5 → normalize → reference anchoring → signal refinement →
/// features → motif search → chunk extraction → sequence encoding.
///
/// Per-read processing is parallelized with rayon (GIL released).
///
/// Returns a tuple of columnar arrays for efficient consumption in Python.
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
    signal_len,
    compute_features,
    reverse_signal = true,
    feature_start = None,
    feature_end = None,
    anchor = "reference",
    cigar_tuples = None,
    reference_sequences = None,
    motif = None,
    motif_offset = 0,
    motif_reference = "fasta",
    ref_seq_dict = None,
    ref_names = None,
    ref_starts = None,
    ref_ends = None,
    skip_motif_indels = false,
    allow_edge_indels = false,
    norm_method = "median_mad",
    base_justify = "center",
    seq_encoding = "base_onehot",
    signal_kmer_context = None,
    refine_signal_map = false,
    kmer_table = None,
    kmer_len = 9,
    kmer_center_idx = -1,
    refine_half_bandwidth = 5,
    refine_scale_iters = 2,
    signal_in_channels = 1,
    cal_offsets = None,
    cal_scales = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn extract_training_chunks_batch<'py>(
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
    signal_len: usize,
    compute_features: bool,
    reverse_signal: bool,
    feature_start: Option<i64>,
    feature_end: Option<i64>,
    anchor: &str,
    cigar_tuples: Option<Vec<Vec<(u32, u32)>>>,
    reference_sequences: Option<Vec<Option<String>>>,
    motif: Option<String>,
    motif_offset: i32,
    motif_reference: &str,
    ref_seq_dict: Option<HashMap<String, String>>,
    ref_names: Option<Vec<Option<String>>>,
    ref_starts: Option<Vec<u64>>,
    ref_ends: Option<Vec<u64>>,
    skip_motif_indels: bool,
    allow_edge_indels: bool,
    norm_method: &str,
    base_justify: &str,
    seq_encoding: &str,
    signal_kmer_context: Option<(usize, usize)>,
    refine_signal_map: bool,
    kmer_table: Option<HashMap<String, f64>>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
    cal_offsets: Option<Vec<f32>>,
    cal_scales: Option<Vec<f32>>,
) -> PyResult<TrainingBatchResult> {
    let n_reads = read_ids.len();
    if sequences.len() != n_reads
        || mv_strides.len() != n_reads
        || mv_arrays.len() != n_reads
        || num_samples_list.len() != n_reads
        || trim_offsets.len() != n_reads
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same length",
        ));
    }

    let _ = base_justify; // TODO: support start/end justify (currently always center)

    let kmer_ctx = kmer_context;

    // Build config
    let cfg = TrainingConfig {
        base: PipelineConfig {
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
        },
        motif,
        motif_offset,
        motif_reference: motif_reference.to_string(),
        skip_motif_indels,
        allow_edge_indels,
        norm_method: norm_method.to_string(),
    };

    // --- Phase 1: POD5 I/O ---
    let target_uuids: HashSet<escapepod::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod::Uuid::parse_str(s).ok())
        .collect();
    let reader = escapepod::Reader::open(pod5_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to open POD5: {e}"))
    })?;

    let mut to_extract: Vec<(String, Vec<u64>)> = Vec::with_capacity(target_uuids.len());
    let mut cal_data: HashMap<String, (f32, f32)> = HashMap::new();

    for read_result in reader.reads().map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to iterate reads: {e}"))
    })? {
        let read = read_result.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to parse read: {e}"))
        })?;
        if target_uuids.contains(&read.read_id) {
            let rid = read.read_id.to_string();
            cal_data.insert(rid.clone(), (read.calibration_offset, read.calibration_scale));
            to_extract.push((rid, read.signal_rows));
            if to_extract.len() == target_uuids.len() {
                break;
            }
        }
    }

    let bulk_signals = reader.get_signal_bulk(&to_extract).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Signal extraction failed: {e}"))
    })?;
    let signal_map: HashMap<String, Vec<i16>> = bulk_signals.into_iter().collect();

    // --- Phase 2: Per-read processing (parallel via rayon, GIL released) ---
    let all_chunks: Vec<Vec<TrainingChunkResult>>;
    {
        let pool_result: Vec<Vec<TrainingChunkResult>> = py.detach(|| {
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
                    let rname = ref_names
                        .as_ref()
                        .and_then(|r| r.get(i))
                        .and_then(|s| s.as_deref());
                    let rstart = ref_starts.as_ref().and_then(|r| r.get(i)).copied();
                    let rend = ref_ends.as_ref().and_then(|r| r.get(i)).copied();

                    // Calibration: prefer per-read from Python, fallback to POD5
                    let (co, cs) = if let (Some(offsets), Some(scales)) = (&cal_offsets, &cal_scales) {
                        (offsets[i], scales[i])
                    } else {
                        cal_data.get(rid.as_str()).copied().unwrap_or((0.0, 1.0))
                    };

                    process_one_read_training(
                        raw_i16,
                        rid,
                        &sequences[i],
                        &mv_arrays[i],
                        mv_strides[i],
                        num_samples_list[i],
                        trim_offsets[i],
                        &cfg,
                        cigar,
                        rseq,
                        rname,
                        rstart,
                        rend,
                        ref_seq_dict.as_ref(),
                        co,
                        cs,
                    )
                })
                .collect()
        });
        all_chunks = pool_result;
    }

    // --- Phase 3: Flatten and convert to numpy (needs GIL) ---
    let flat_chunks: Vec<TrainingChunkResult> = all_chunks.into_iter().flatten().collect();
    let n_chunks = flat_chunks.len();

    if n_chunks == 0 {
        // Return empty arrays
        let dw = cfg.base.dwell_width;
        let nf = if cfg.base.compute_features { 9 } else { 0 }; // estimate
        let sig_out_len = if cfg.base.signal_in_channels > 1 { signal_len * 2 } else { signal_len };

        return Ok(TrainingBatchResult {
            num_chunks: 0,
            signals: numpy::ndarray::Array2::<f32>::zeros((0, sig_out_len)).into_pyarray(py).unbind().into_any().into(),
            sequences: vec![],
            dwells: numpy::ndarray::Array2::<f32>::zeros((0, dw)).into_pyarray(py).unbind().into_any().into(),
            features: numpy::ndarray::Array3::<f32>::zeros((0, nf, dw)).into_pyarray(py).unbind().into_any().into(),
            read_ids: vec![],
            base_indices: numpy::ndarray::Array1::<i64>::zeros(0).into_pyarray(py).unbind().into_any().into(),
            feature_starts: numpy::ndarray::Array1::<i64>::zeros(0).into_pyarray(py).unbind().into_any().into(),
            feature_ends: numpy::ndarray::Array1::<i64>::zeros(0).into_pyarray(py).unbind().into_any().into(),
            seq_to_sig_maps: vec![],
            sequences_with_kmer_context: vec![],
            signal_residuals: None,
            seq_encodings: vec![],
        });
    }

    // Determine dimensions from first chunk
    let sig_out_len = flat_chunks[0].signal.len();
    let dw = flat_chunks[0].dwell_width;
    let nf = flat_chunks[0].num_features;

    // Build columnar arrays
    let mut signals_flat = Vec::with_capacity(n_chunks * sig_out_len);
    let mut dwells_flat = Vec::with_capacity(n_chunks * dw);
    let mut feats_flat = Vec::with_capacity(n_chunks * nf * dw);
    let mut base_indices = Vec::with_capacity(n_chunks);
    let mut feat_starts = Vec::with_capacity(n_chunks);
    let mut feat_ends = Vec::with_capacity(n_chunks);
    let mut chunk_sequences = Vec::with_capacity(n_chunks);
    let mut chunk_read_ids = Vec::with_capacity(n_chunks);
    let mut chunk_sig_maps: Vec<Py<PyAny>> = Vec::with_capacity(n_chunks);
    let mut chunk_seq_kmer_ctx = Vec::with_capacity(n_chunks);
    let mut has_residual = false;
    let mut residuals_flat: Vec<f32> = Vec::new();
    let mut seq_encodings: Vec<Py<PyAny>> = Vec::with_capacity(n_chunks);

    for c in &flat_chunks {
        signals_flat.extend_from_slice(&c.signal);
        dwells_flat.extend_from_slice(&c.dwell);
        feats_flat.extend_from_slice(&c.features);
        base_indices.push(c.base_idx);
        feat_starts.push(c.feature_start);
        feat_ends.push(c.feature_end);
        chunk_sequences.push(c.sequence.clone());
        chunk_read_ids.push(c.read_id.clone());
        chunk_sig_maps.push(c.seq_to_sig_map.clone().into_pyarray(py).unbind().into_any().into());
        chunk_seq_kmer_ctx.push(c.sequence_with_kmer_context.clone());

        if c.signal_residual.is_some() {
            has_residual = true;
        }

        // Sequence encoding
        let seq_arr = numpy::ndarray::Array2::from_shape_vec((c.seq_rows, c.seq_cols), c.seq_enc.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Seq enc error: {e}")))?;
        seq_encodings.push(seq_arr.into_pyarray(py).unbind().into_any().into());
    }

    // Collect residuals if any
    if has_residual {
        residuals_flat.reserve(n_chunks * signal_len);
        for c in &flat_chunks {
            if let Some(ref res) = c.signal_residual {
                residuals_flat.extend_from_slice(res);
            } else {
                residuals_flat.extend(std::iter::repeat(0.0f32).take(signal_len));
            }
        }
    }

    // Convert to numpy
    let signals_arr = numpy::ndarray::Array2::from_shape_vec((n_chunks, sig_out_len), signals_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Signal shape error: {e}")))?;
    let dwells_arr = numpy::ndarray::Array2::from_shape_vec((n_chunks, dw), dwells_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Dwell shape error: {e}")))?;
    let feats_arr = if nf > 0 {
        numpy::ndarray::Array3::from_shape_vec((n_chunks, nf, dw), feats_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Feature shape error: {e}")))?
    } else {
        numpy::ndarray::Array3::<f32>::zeros((n_chunks, 0, dw))
    };
    let residuals_py = if has_residual {
        let arr = numpy::ndarray::Array2::from_shape_vec((n_chunks, signal_len), residuals_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Residual shape error: {e}")))?;
        Some(arr.into_pyarray(py).unbind().into_any().into())
    } else {
        None
    };

    Ok(TrainingBatchResult {
        num_chunks: n_chunks,
        signals: signals_arr.into_pyarray(py).unbind().into_any().into(),
        sequences: chunk_sequences,
        dwells: dwells_arr.into_pyarray(py).unbind().into_any().into(),
        features: feats_arr.into_pyarray(py).unbind().into_any().into(),
        read_ids: chunk_read_ids,
        base_indices: base_indices.into_pyarray(py).unbind().into_any().into(),
        feature_starts: feat_starts.into_pyarray(py).unbind().into_any().into(),
        feature_ends: feat_ends.into_pyarray(py).unbind().into_any().into(),
        seq_to_sig_maps: chunk_sig_maps,
        sequences_with_kmer_context: chunk_seq_kmer_ctx,
        signal_residuals: residuals_py,
        seq_encodings,
    })
}
