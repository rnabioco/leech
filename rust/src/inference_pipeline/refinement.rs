//! Signal refinement pipeline.
//!
//! Delegates the full signal-to-sequence refinement to escapepod-signal's
//! canonical `resquiggle::refine_signal_map` rather than maintaining leech's
//! own hand-ported copy. The settings below reproduce leech's pipeline
//! (fixed banding, least-squares rough rescale over the 0.05–0.95 quantiles
//! with 10-base clipping, Theil-Sen inter-iteration rescale) with escapepod's
//! purpose-built asymmetric dwell penalty (quadratic below target, log above)
//! in place of leech's short-dwell table.

use std::collections::HashMap;

use escapepod_signal::resquiggle::{
    BandingAlgo, RefineAlgo, RefineSettings, RescaleAlgo, RescaleFilterParams, RoughRescaleAlgo,
    refine_signal_map,
};

use crate::signal_refine::extract_levels_inner;

/// Preferred dwell time (signal samples/base) for the asymmetric dwell penalty.
const DWELL_TARGET: f32 = 4.0;
/// Strength of the dwell penalty.
const DWELL_WEIGHT: f32 = 0.5;
/// Max points sampled for the Theil-Sen inter-iteration rescale.
const THEIL_SEN_MAX_POINTS: usize = 200;
/// Bases clipped from each end during rough rescale.
const ROUGH_CLIP_BASES: usize = 10;

/// Build the refinement settings matching leech's pipeline.
fn build_settings(half_bandwidth: i32, scale_iters: i32) -> RefineSettings {
    // scale_iters: <0 handled by the caller (rough-rescale only). 0 -> a single
    // DP pass with no rescale; >0 -> that many passes with rescale. This mirrors
    // escapepod's `n_refinement_iters` (0 => 1 iter no rescale; n => n w/ rescale).
    let n_refinement_iters = scale_iters.max(0) as usize;
    let filter = RescaleFilterParams::default(); // matches leech's Theil-Sen filter
    RefineSettings {
        refinement_algo: RefineAlgo::DwellPenalty {
            target: DWELL_TARGET,
            weight: DWELL_WEIGHT,
        },
        n_refinement_iters,
        half_bandwidth: half_bandwidth.max(0) as usize,
        adjust_band_min_size: 2,
        rescale_algo: RescaleAlgo::TheilSen {
            filter,
            max_points: THEIL_SEN_MAX_POINTS,
        },
        rough_rescale_algo: RoughRescaleAlgo::LeastSquares {
            quantiles: RoughRescaleAlgo::default_quantiles(),
            clip_bases: ROUGH_CLIP_BASES,
            use_base_center: true,
        },
        normalize_levels: false,
        banding_algo: BandingAlgo::Fixed,
    }
}

/// Full signal refinement pipeline (delegates to escapepod-signal).
///
/// Mutates `signal` (rescaled to the level model) and `seq_to_sig_map`
/// (refined boundaries) in place. On any failure the inputs are left
/// unchanged, matching the previous pipeline's early-return behaviour.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine_signal_map_pipeline(
    signal: &mut Vec<f32>,
    seq_to_sig_map: &mut Vec<i64>,
    sequence: &str,
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    kmer_center_idx: i32,
    half_bandwidth: i32,
    scale_iters: i32,
) {
    let levels_f32: Vec<f32> = extract_levels_inner(sequence, kmer_to_level, kmer_len, kmer_center_idx)
        .iter()
        .map(|&v| v as f32)
        .collect();
    if levels_f32.is_empty() || seq_to_sig_map.len() != levels_f32.len() + 1 {
        return;
    }

    // escapepod works in usize signal coordinates; bail if the map is malformed.
    if seq_to_sig_map.iter().any(|&v| v < 0) {
        return;
    }
    let map_usize: Vec<usize> = seq_to_sig_map.iter().map(|&v| v as usize).collect();
    let last = *map_usize.last().unwrap();
    if last > signal.len() || map_usize[0] >= last {
        return;
    }

    let settings = build_settings(half_bandwidth, scale_iters);

    // Signal is already median-MAD normalized, so start from identity scaling
    // and let rough rescale derive the level-matching transform.
    let result = match refine_signal_map(&settings, signal, &map_usize, &levels_f32, 1.0, 0.0) {
        Ok(r) => r,
        Err(_) => return,
    };

    if result.seq_to_signal_map.len() != seq_to_sig_map.len() {
        return;
    }

    // Rescale the signal by the final (shift, scale, drift) so downstream
    // per-base stats and residuals are on the level model's scale — the same
    // signal escapepod used in its final DP pass.
    if result.scale.abs() > 1e-10 {
        for (i, s) in signal.iter_mut().enumerate() {
            *s = (*s - result.shift - result.drift * i as f32) / result.scale;
        }
    }
    *seq_to_sig_map = result.seq_to_signal_map.iter().map(|&v| v as i64).collect();
}
