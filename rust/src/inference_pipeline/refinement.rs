//! Signal refinement pipeline.
//!
//! Delegates the full signal-to-sequence refinement to escapepod-signal's
//! canonical `resquiggle::refine_signal_map`, configured by escapepod's own
//! `RefineSettings::move_table_refinement` preset.
//!
//! **Do not hand-build the settings here.** This file used to carry a struct
//! literal that duplicated the one inside escapepod's Python binding, each with
//! a comment asserting it matched the other. They drifted on `dwell_target` --
//! escapepod's binding hardcoded 4.0, this side passed the per-read sentinel --
//! and because the dwell penalty is asymmetric, a target ~8x too low did not
//! merely weaken the prior, it dragged boundaries toward dwells the pore never
//! produced. The two backends refined the same reads to different boundaries
//! for four releases (leech #193, escapepod-rs#257). One preset, both callers.

use std::collections::HashMap;

use escapepod_signal::resquiggle::{RefineSettings, refine_signal_map};

use crate::signal_refine::extract_levels_inner;

/// Fixed seed for the Theil-Sen subsample so refinement is reproducible on long
/// reads. Must match leech's Python `REFINE_SUBSAMPLE_SEED`.
const REFINE_SUBSAMPLE_SEED: u64 = 42;

/// Build the refinement settings.
///
/// `scale_iters` is clamped at 0 only because escapepod reads 0 as "one DP pass
/// without rescaling"; a *negative* value means "no refinement at all" and is
/// handled by the caller (`process_read_signal`), which skips this function
/// entirely rather than clamping. Clamping there instead would refine the map
/// on this path while Python left it untouched.
fn build_settings(half_bandwidth: i32, scale_iters: i32) -> RefineSettings {
    RefineSettings::move_table_refinement(
        half_bandwidth.max(0) as usize,
        scale_iters.max(0) as usize,
        Some(REFINE_SUBSAMPLE_SEED),
    )
}

/// Full signal refinement pipeline (delegates to escapepod-signal).
///
/// Refines `seq_to_sig_map` (base boundaries) in place; `signal` is read but
/// never rewritten, so the caller's normalization is preserved — see the note
/// at the end of this function. On any failure the map is left unchanged,
/// matching the previous pipeline's early-return behaviour.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine_signal_map_pipeline(
    signal: &[f32],
    seq_to_sig_map: &mut Vec<i64>,
    sequence: &str,
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    kmer_center_idx: i32,
    half_bandwidth: i32,
    scale_iters: i32,
) {
    let levels_f32: Vec<f32> =
        extract_levels_inner(sequence, kmer_to_level, kmer_len, kmer_center_idx)
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

    // Take the refined boundaries, keep our own normalization.
    //
    // Deliberately do NOT rescale `signal` by the fitted (shift, scale, drift).
    // The caller hands in a median-MAD normalized signal — one transform shared
    // by every read — and that is what per-base stats, k-mer residuals and the
    // trained models are calibrated against. Re-normalizing here replaces it
    // with a *per-read* transform fitted on this chunk alone, which only helps
    // if the fit is reliable for every read.
    //
    // It is not. These chunks sit largely in a constant 3' adapter, so the
    // expected levels barely vary and the fit is weakly identified: observed
    // scales ranged from 15 to 1084 and were frequently negative. escapepod now
    // rejects the worst of those, but rejection is itself per-read, so the reads
    // that still fit end up on a different scale from the reads that do not —
    // and cross-read comparability is exactly what k-mer residuals depend on.
    // Measured on tRNA-Met chunks, per-base level vs expected k-mer level:
    // r = +0.72 keeping our normalization, +0.03 applying the fitted one.
    *seq_to_sig_map = result.seq_to_signal_map.iter().map(|&v| v as i64).collect();
}
