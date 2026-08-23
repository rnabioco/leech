//! Training chunk extraction, parallel processing, and PyO3 entry point.

use std::collections::HashMap;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::processing::process_read_signal;
use super::signal_mapping::chunk_signal_kmer_inputs;
use super::types::{BaseJustify, PipelineConfig, ProcessedRead, TrainingChunkResult};

/// Extract training-format chunks from a processed read.
fn extract_training_chunks_from_read(
    processed: &ProcessedRead,
    rid: &str,
    positions: &[i64],
    cfg: &PipelineConfig,
) -> Vec<TrainingChunkResult> {
    let ProcessedRead {
        ref norm_signal,
        ref seq_to_sig,
        ref use_sequence,
        ref dwells,
        ref features_data,
        num_features,
        ref sig_residual,
    } = *processed;
    let num_bases = seq_to_sig.len().saturating_sub(1);
    let seq_bytes = use_sequence.as_bytes();
    let mut results = Vec::new();

    for &base_idx in positions {
        // The ONLY reason to drop a focus base is that it has no signal
        // boundaries -- `seq_to_sig[bi + 1]` below must exist. A k-mer window
        // that overhangs either end of the sequence is padded with 'N', which
        // is what `LeechRead.get_chunk` does; skipping it instead silently
        // loses whole reads whose aligned region stops within `kmer_ctx` of
        // the motif (issue #185). In reference-anchored mode the sequence is
        // the aligned reference slice, so that is the supplementary-aligned
        // and indel-heavy population, not a random ~1%.
        if base_idx < 0 || base_idx as usize >= num_bases {
            continue;
        }
        let bi = base_idx as usize;
        let kmer_start = base_idx - cfg.kmer_ctx;
        let kmer_end = base_idx + cfg.kmer_ctx + 1;
        let kmer_len = (kmer_end - kmer_start) as usize;

        // Signal chunk (same logic as inference)
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

        // Signal residual chunk
        let res_chunk = if let Some(ref residual) = *sig_residual {
            let mut rc = vec![0.0f32; cfg.signal_len];
            if actual_len <= cfg.signal_len {
                let src_lo = sig_start_pos.max(0) as usize;
                let src_hi = sig_end_pos.min(residual.len() as i64) as usize;
                let dst_off = (sig_start_pos.max(0) - sig_start_pos) as usize;
                if src_lo < src_hi {
                    let n = (src_hi - src_lo).min(cfg.signal_len - dst_off);
                    rc[dst_off..dst_off + n].copy_from_slice(&residual[src_lo..src_lo + n]);
                }
            } else {
                let crop_start = (actual_len - cfg.signal_len) / 2;
                let abs_start = (sig_start_pos + crop_start as i64).max(0) as usize;
                let abs_end = (abs_start + cfg.signal_len).min(residual.len());
                let n = abs_end - abs_start;
                rc[..n].copy_from_slice(&residual[abs_start..abs_end]);
            }
            rc
        } else {
            vec![]
        };

        // Kmer sequence, 'N'-padded past either end of the sequence so the
        // window is always `kmer_len` wide (matches `LeechRead.get_chunk`).
        let kmer_seq: String = (kmer_start..kmer_end)
            .map(|i| match usize::try_from(i) {
                Ok(u) if u < seq_bytes.len() => seq_bytes[u] as char,
                _ => 'N',
            })
            .collect();

        // Dwell slice for this kmer window
        let dwell_fs = (base_idx + cfg.feat_start).max(0) as usize;
        let dwell_fe = ((base_idx + cfg.feat_end + 1) as usize).min(num_bases);
        let dwell_left_pad = ((0i64 - (base_idx + cfg.feat_start)).max(0)) as usize;
        let mut dwell_chunk = vec![0.0f32; cfg.dwell_width];
        if dwell_fs < dwell_fe {
            let n = (dwell_fe - dwell_fs).min(cfg.dwell_width - dwell_left_pad);
            dwell_chunk[dwell_left_pad..dwell_left_pad + n]
                .copy_from_slice(&dwells[dwell_fs..dwell_fs + n]);
        }

        // Features (flattened [num_features * dwell_width])
        let feat_flat = if let Some(ref all_feats) = *features_data {
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
            flat
        } else {
            vec![]
        };

        // Focus signal position -- match the Python convention of always
        // using signal_context_left so downstream code can rely on a fixed
        // offset regardless of edge padding.
        let focus_signal_pos = cfg.signal_context_left;

        // Chunk-local seq_to_sig_map and its context sequence, the two inputs
        // signal_kmer encoding consumes. Both key off the SIGNAL window, not
        // the k-mer window -- see `chunk_signal_kmer_inputs`. Always computed,
        // as Python does, so a chunk can be re-encoded either way later.
        let (chunk_sig_map, ctx_bytes) = chunk_signal_kmer_inputs(
            seq_to_sig,
            seq_bytes,
            sig_start_pos,
            sig_end_pos,
            norm_signal.len(),
            cfg.signal_len,
            cfg.skmer_ctx,
        );
        let seq_with_ctx = String::from_utf8(ctx_bytes).unwrap_or_default();

        results.push(TrainingChunkResult {
            signal: sig_chunk,
            sequence: kmer_seq,
            dwell: dwell_chunk,
            features: feat_flat,
            num_features,
            kmer_len,
            read_id: rid.to_string(),
            base_idx,
            focus_signal_pos,
            seq_to_sig_map: chunk_sig_map,
            sequence_with_kmer_context: seq_with_ctx,
            signal_residual: res_chunk,
        });
    }

    results
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
    cfg: &PipelineConfig,
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&str>,
) -> Vec<TrainingChunkResult> {
    match process_read_signal(
        raw_i16, sequence, mv, stride, ns, trim, cfg, cigar_ops, ref_seq,
    ) {
        Some(processed) => extract_training_chunks_from_read(&processed, rid, positions, cfg),
        None => vec![],
    }
}

/// Run per-read training extraction in parallel (rayon, GIL released),
/// then convert results to Python dicts with numpy arrays.
#[allow(clippy::too_many_arguments)]
fn _process_and_convert_training<'py>(
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
) -> PyResult<Vec<Py<PyAny>>> {
    let n_reads = read_ids.len();

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

    Ok(results)
}

// ---------------------------------------------------------------------------
// PyO3 entry point -- training chunks (POD5 I/O + processing)
// ---------------------------------------------------------------------------

/// Extract training-format chunks for a batch of reads in one Rust call.
///
/// Returns a list of Python dicts, each containing numpy arrays for signal,
/// dwell, features, and metadata. Labels and other Python-side metadata
/// are attached by the caller.
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
pub fn extract_training_chunks<'py>(
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
) -> PyResult<Vec<Py<PyAny>>> {
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
        base_justify: BaseJustify::from_str(base_justify),
    };

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
