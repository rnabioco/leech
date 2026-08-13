//! Shared per-read signal processing pipeline.

use crate::signal_refine::extract_levels_inner;

use super::features::{
    compute_dwell_features, compute_kmer_residual_features, compute_per_base_stats,
    compute_signal_residual,
};
use super::numeric::normalize_median_mad;
use super::refinement::refine_signal_map_pipeline;
use super::signal_mapping::{build_seq_to_sig_map, compute_ref_to_signal};
use super::types::{PipelineConfig, ProcessedRead};

/// Process one read's signal: trim -> reverse -> normalize -> anchor ->
/// refine -> compute features. Shared by inference and training paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_read_signal(
    raw_i16: &[i16],
    sequence: &str,
    mv: &[u8],
    stride: u32,
    ns: u64,
    trim: i64,
    cfg: &PipelineConfig,
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&str>,
) -> Option<ProcessedRead> {
    let trim_start = trim.max(0) as usize;
    let trim_end = (ns as usize).min(raw_i16.len());
    if trim_start >= trim_end {
        return None;
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
                return None;
            }
            let sig_start = ref_to_sig[0].max(0) as usize;
            let sig_end = (*ref_to_sig.last().unwrap()).min(norm_signal.len() as i64) as usize;
            if sig_start >= sig_end {
                return None;
            }
            norm_signal = norm_signal[sig_start..sig_end].to_vec();
            let shifted: Vec<i64> = ref_to_sig.iter().map(|&v| v - sig_start as i64).collect();
            (shifted, rseq.to_string())
        } else {
            (query_to_sig, sequence.to_string())
        }
    } else {
        (query_to_sig, sequence.to_string())
    };

    let num_bases = seq_to_sig.len().saturating_sub(1);
    if num_bases == 0 {
        return None;
    }

    // Signal refinement
    let mut expected_levels_f64: Option<Vec<f64>> = None;
    if cfg.refine_signal_map
        && let Some(ref kt) = cfg.kmer_table
    {
        refine_signal_map_pipeline(
            &norm_signal,
            &mut seq_to_sig,
            &use_sequence,
            kt,
            cfg.kmer_len,
            cfg.kmer_center_idx,
            cfg.refine_half_bandwidth,
            cfg.refine_scale_iters,
        );
        expected_levels_f64 = Some(extract_levels_inner(
            &use_sequence,
            kt,
            cfg.kmer_len,
            cfg.kmer_center_idx,
        ));
    }

    let num_bases = seq_to_sig.len().saturating_sub(1);
    if num_bases == 0 {
        return None;
    }

    // Dwells
    let dwells: Vec<f32> = (0..num_bases)
        .map(|j| (seq_to_sig[j + 1] - seq_to_sig[j]) as f32)
        .collect();

    // Features
    let features_data: Option<Vec<Vec<f32>>> = if cfg.compute_features {
        let (means, medians, stds, ranges) = compute_per_base_stats(&norm_signal, &seq_to_sig);
        let mut feats = compute_dwell_features(&dwells);
        let kmer_residuals = expected_levels_f64
            .as_ref()
            .map(|levels| compute_kmer_residual_features(&means, levels));
        feats.push(means);
        feats.push(medians);
        feats.push(stds);
        feats.push(ranges);
        if let Some((ke, kr, kra)) = kmer_residuals {
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
        expected_levels_f64
            .as_ref()
            .map(|levels| compute_signal_residual(&norm_signal, &seq_to_sig, levels, num_bases))
    } else {
        None
    };

    Some(ProcessedRead {
        norm_signal,
        seq_to_sig,
        use_sequence,
        dwells,
        features_data,
        num_features,
        sig_residual,
    })
}
