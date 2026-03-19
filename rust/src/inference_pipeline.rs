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

use crate::encoding::encode_signal_kmer_inner;
use crate::pod5_io::PreloadedSignals;
use crate::signal_refine::{extract_levels_inner, seq_banded_dp_inner};

// ---------------------------------------------------------------------------
// Numeric helpers — O(n) median via select_nth_unstable
// ---------------------------------------------------------------------------

const F32_CMP: fn(&f32, &f32) -> std::cmp::Ordering =
    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
const F64_CMP: fn(&f64, &f64) -> std::cmp::Ordering =
    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

fn median_f32(data: &mut [f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
    let mid = n / 2;
    data.select_nth_unstable_by(mid, F32_CMP);
    if n % 2 == 0 {
        let hi = data[mid];
        // The max of the lower partition is the other median element
        let lo = data[..mid].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lo + hi) / 2.0
    } else {
        data[mid]
    }
}

fn median_f64(data: &mut [f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
    let mid = n / 2;
    data.select_nth_unstable_by(mid, F64_CMP);
    if n % 2 == 0 {
        let hi = data[mid];
        let lo = data[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo + hi) / 2.0
    } else {
        data[mid]
    }
}

fn percentile_f64(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[allow(dead_code)]
fn quantile_f32(data: &[f32], q: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = q * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f32) as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f32;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Piecewise linear interpolation (equivalent to np.interp).
fn linear_interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    if xp.is_empty() || fp.is_empty() {
        return vec![0.0; x.len()];
    }
    let n = xp.len();
    let mut result = Vec::with_capacity(x.len());
    let mut j = 0usize; // current segment
    for &xi in x {
        if xi <= xp[0] {
            result.push(fp[0]);
        } else if xi >= xp[n - 1] {
            result.push(fp[n - 1]);
        } else {
            while j < n - 2 && xp[j + 1] < xi {
                j += 1;
            }
            let t = (xi - xp[j]) / (xp[j + 1] - xp[j]);
            result.push(fp[j] + t * (fp[j + 1] - fp[j]));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

fn normalize_median_mad(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }
    let mut sorted = signal.to_vec();
    let med = median_f32(&mut sorted);
    let mut deviations: Vec<f32> = signal.iter().map(|&x| (x - med).abs()).collect();
    let mad = median_f32(&mut deviations);
    let scale = mad * 1.4826;
    if scale < 1e-10 {
        return signal.iter().map(|&x| x - med).collect();
    }
    signal.iter().map(|&x| (x - med) / scale).collect()
}

// ---------------------------------------------------------------------------
// Move table → seq_to_sig_map
// ---------------------------------------------------------------------------

fn build_seq_to_sig_map(mv_array: &[u8], stride: u32, trim_offset: i64, num_samples: u64) -> Vec<i64> {
    let mut sig_map: Vec<i64> = mv_array
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 1)
        .map(|(i, _)| i as i64 * stride as i64) // trimmed-space coordinates
        .collect();
    let trimmed_len = num_samples as i64 - trim_offset;
    sig_map.push(trimmed_len);
    sig_map
}

// ---------------------------------------------------------------------------
// CIGAR → reference-to-signal mapping
// ---------------------------------------------------------------------------

fn make_ref_to_query_mapping(cigar_ops: &[(u32, u32)]) -> Vec<f64> {
    const MATCH: [bool; 9] = [true, false, false, false, false, false, false, true, true];
    const REF_CONSUME: [bool; 9] = [true, false, true, true, false, false, false, true, true];
    const QUERY_CONSUME: [bool; 9] = [true, true, false, false, true, false, false, true, true];

    // Strip trailing non-match ops
    let mut cigar: Vec<(u32, u32)> = cigar_ops.to_vec();
    while !cigar.is_empty() && !MATCH.get(cigar.last().unwrap().0 as usize).copied().unwrap_or(false) {
        cigar.pop();
    }
    if cigar.is_empty() {
        return vec![0.0];
    }

    // Compute cumulative ref and query positions, collect match blocks
    let mut ref_pos = 0u64;
    let mut query_pos = 0u64;
    let mut ref_knots = vec![0.0f64];
    let mut query_knots = vec![0.0f64];

    for &(op, len) in &cigar {
        let op_idx = op as usize;
        let is_match = MATCH.get(op_idx).copied().unwrap_or(false);
        let consumes_ref = REF_CONSUME.get(op_idx).copied().unwrap_or(false);
        let consumes_query = QUERY_CONSUME.get(op_idx).copied().unwrap_or(false);

        if is_match && len > 0 {
            // Knot at start of match block
            ref_knots.push(ref_pos as f64);
            query_knots.push(query_pos as f64);
            // Knot at end-1 of match block
            ref_knots.push((ref_pos + len as u64 - 1) as f64);
            query_knots.push((query_pos + len as u64 - 1) as f64);
        }

        if consumes_ref {
            ref_pos += len as u64;
        }
        if consumes_query {
            query_pos += len as u64;
        }
    }

    // Final knots
    ref_knots.push(ref_pos as f64);
    query_knots.push(query_pos as f64);

    // Interpolate to get mapping for every ref position
    let x: Vec<f64> = (0..=ref_pos).map(|i| i as f64).collect();
    linear_interp(&x, &ref_knots, &query_knots)
}

fn compute_ref_to_signal(query_to_sig: &[i64], cigar_ops: &[(u32, u32)]) -> Vec<i64> {
    let ref_to_query = make_ref_to_query_mapping(cigar_ops);

    // Interpolate float query positions through query_to_sig mapping
    let xp: Vec<f64> = (0..query_to_sig.len()).map(|i| i as f64).collect();
    let fp: Vec<f64> = query_to_sig.iter().map(|&v| v as f64).collect();
    let result_f64 = linear_interp(&ref_to_query, &xp, &fp);

    result_f64.iter().map(|&v| v.floor() as i64).collect()
}

// ---------------------------------------------------------------------------
// Band computation (matches leech's Python implementation)
// ---------------------------------------------------------------------------

fn compute_sig_band(bps: &[i32], levels: &[f64], bhw: i32) -> (Vec<i32>, Vec<i32>) {
    let seq_len = levels.len();
    let sig_len = (bps[seq_len] - bps[0]) as usize;

    // Build seq_indices: for each signal position, which base it belongs to
    let mut seq_indices = Vec::with_capacity(sig_len);
    for base in 0..seq_len {
        let dwell = (bps[base + 1] - bps[base]) as usize;
        for _ in 0..dwell {
            seq_indices.push(base as i32);
        }
    }

    let mut band_lo = vec![0i32; sig_len];
    let mut band_hi = vec![0i32; sig_len];
    for s in 0..sig_len {
        band_lo[s] = (seq_indices[s] - bhw).max(0);
        band_hi[s] = (seq_indices[s] + bhw + 1).min(seq_len as i32);
    }

    // Handle NaN levels: route through NaN regions
    for s in 0..sig_len {
        let base = seq_indices[s] as usize;
        if base < levels.len() && levels[base].is_nan() {
            band_lo[s] = seq_indices[s];
            band_hi[s] = seq_indices[s] + 1;
        }
    }

    // Enforce monotonicity
    for s in 1..sig_len {
        band_lo[s] = band_lo[s].max(band_lo[s - 1]);
    }
    for s in (0..sig_len - 1).rev() {
        band_hi[s] = band_hi[s].min(band_hi[s + 1]);
    }

    (band_lo, band_hi)
}

fn convert_to_seq_band(sig_band_lo: &[i32], sig_band_hi: &[i32], seq_len: usize) -> (Vec<i32>, Vec<i32>) {
    let sig_len = sig_band_lo.len() as i32;
    let mut seq_lo = vec![0i32; seq_len];
    let mut seq_hi = vec![sig_len; seq_len];

    // Upper signal coords define lower sequence boundaries
    for s in 1..sig_band_hi.len() {
        if sig_band_hi[s] != sig_band_hi[s - 1] {
            let base = sig_band_hi[s - 1] as usize;
            if base < seq_len {
                seq_lo[base] = s as i32;
            }
        }
    }
    for b in 1..seq_len {
        seq_lo[b] = seq_lo[b].max(seq_lo[b - 1]);
    }

    // Lower signal coords define upper sequence boundaries
    for s in 1..sig_band_lo.len() {
        if sig_band_lo[s] != sig_band_lo[s - 1] {
            let base = sig_band_lo[s] as usize;
            if base > 0 && base - 1 < seq_len {
                seq_hi[base - 1] = s as i32;
            }
        }
    }
    for b in (0..seq_len - 1).rev() {
        seq_hi[b] = seq_hi[b].min(seq_hi[b + 1]);
    }

    (seq_lo, seq_hi)
}

fn adjust_seq_band(seq_lo: &mut [i32], seq_hi: &mut [i32], min_step: i32) {
    let n = seq_lo.len();
    if n == 0 {
        return;
    }

    // Fix starts: sweep right-to-left
    let band_min = seq_lo[0];
    for i in (0..n - 1).rev() {
        if seq_lo[i] > seq_lo[i + 1] - min_step {
            seq_lo[i] = seq_lo[i + 1] - min_step;
        }
    }
    seq_lo[0] = band_min;
    let mut i = 1;
    while i < n && seq_lo[i] <= seq_lo[i - 1] {
        seq_lo[i] = seq_lo[i - 1] + 1;
        i += 1;
    }

    // Fix ends: sweep left-to-right
    let band_max = seq_hi[n - 1];
    for i in 1..n {
        if seq_hi[i] < seq_hi[i - 1] + min_step {
            seq_hi[i] = seq_hi[i - 1] + min_step;
        }
    }
    seq_hi[n - 1] = band_max;
    let mut i = n as i32 - 2;
    while i >= 0 && seq_hi[i as usize] >= seq_hi[i as usize + 1] {
        seq_hi[i as usize] = seq_hi[i as usize + 1] - 1;
        i -= 1;
    }
}

// ---------------------------------------------------------------------------
// Signal refinement pipeline (matches SigMapRefiner.refine)
// ---------------------------------------------------------------------------

fn rough_rescale_quantile(signal: &[f32], expected: &[f64], sig_map: &[i64], clip_bases: usize) -> Vec<f32> {
    let num_bases = sig_map.len().saturating_sub(1);
    if num_bases == 0 {
        return signal.to_vec();
    }

    // Compute center-of-base signal values
    let mut centers_sig = Vec::with_capacity(num_bases);
    let mut centers_lvl = Vec::with_capacity(num_bases);
    let lo = clip_bases;
    let hi = if num_bases > clip_bases * 2 { num_bases - clip_bases } else { num_bases };

    for i in lo..hi {
        let center = ((sig_map[i] + sig_map[i + 1]) / 2) as usize;
        if center < signal.len() {
            centers_sig.push(signal[center] as f64);
            centers_lvl.push(expected[i]);
        }
    }

    if centers_sig.len() < 3 {
        return signal.to_vec();
    }

    // Quantile-based lstsq fit
    let quants: Vec<f64> = (1..20).map(|i| i as f64 * 0.05).collect(); // 0.05..0.95
    let mut sorted_sig = centers_sig.clone();
    sorted_sig.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_lvl = centers_lvl.clone();
    sorted_lvl.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sig_qs: Vec<f64> = quants.iter().map(|&q| percentile_f64(&sorted_sig, q * 100.0)).collect();
    let lvl_qs: Vec<f64> = quants.iter().map(|&q| percentile_f64(&sorted_lvl, q * 100.0)).collect();

    // Fit: level = shift + scale * signal  →  [1, sig_q] * [shift, scale] = lvl_q
    let n = sig_qs.len() as f64;
    let sum_x: f64 = sig_qs.iter().sum();
    let sum_y: f64 = lvl_qs.iter().sum();
    let sum_xx: f64 = sig_qs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = sig_qs.iter().zip(lvl_qs.iter()).map(|(&x, &y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return signal.to_vec();
    }
    let scale = (n * sum_xy - sum_x * sum_y) / denom;
    let shift = (sum_y - scale * sum_x) / n;

    if scale.abs() < 1e-10 {
        return signal.to_vec();
    }

    signal.iter().map(|&s| (scale * s as f64 + shift) as f32).collect()
}

fn theil_sen_rescale(
    signal: &[f32],
    levels: &[f64],
    sig_map: &[i64],
    edge_filter: usize,
) -> Vec<f32> {
    let num_bases = sig_map.len().saturating_sub(1);
    if num_bases < 20 {
        return signal.to_vec();
    }

    // Compute per-base means and dwells
    let mut sig_means = vec![0.0f64; num_bases];
    let mut dwells = vec![0i64; num_bases];
    for i in 0..num_bases {
        let start = sig_map[i] as usize;
        let end = sig_map[i + 1] as usize;
        dwells[i] = (end as i64) - (start as i64);
        if end > start && end <= signal.len() {
            let sum: f64 = signal[start..end].iter().map(|&s| s as f64).sum();
            sig_means[i] = sum / (end - start) as f64;
        }
    }

    // Filter: dwell percentiles, edge bases, min absolute level
    let mut sorted_dwells: Vec<f64> = dwells.iter().map(|&d| d as f64).collect();
    sorted_dwells.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dwell_lo = percentile_f64(&sorted_dwells, 10.0);
    let dwell_hi = percentile_f64(&sorted_dwells, 90.0);
    let mean_level: f64 = levels.iter().sum::<f64>() / levels.len() as f64;

    let mut filt_means = Vec::new();
    let mut filt_levels = Vec::new();
    for i in 0..num_bases {
        if i < edge_filter || i >= num_bases - edge_filter {
            continue;
        }
        let d = dwells[i] as f64;
        if d <= dwell_lo || d >= dwell_hi {
            continue;
        }
        if (levels[i] - mean_level).abs() < 0.2 {
            continue;
        }
        if sig_means[i].is_nan() {
            continue;
        }
        filt_means.push(sig_means[i]);
        filt_levels.push(levels[i]);
    }

    if filt_means.len() < 10 {
        return signal.to_vec();
    }

    // Subsample for Theil-Sen if too many
    let max_pts = 1000;
    if filt_means.len() > max_pts {
        // Deterministic subsample (every nth)
        let step = filt_means.len() / max_pts;
        let mut sub_m = Vec::with_capacity(max_pts);
        let mut sub_l = Vec::with_capacity(max_pts);
        for i in (0..filt_means.len()).step_by(step.max(1)) {
            sub_m.push(filt_means[i]);
            sub_l.push(filt_levels[i]);
            if sub_m.len() >= max_pts {
                break;
            }
        }
        filt_means = sub_m;
        filt_levels = sub_l;
    }

    // Compute all pairwise slopes
    let mut slopes = Vec::new();
    for i in 0..filt_means.len() {
        for j in (i + 1)..filt_means.len() {
            let dx = filt_means[j] - filt_means[i];
            if dx.abs() > 1e-12 {
                slopes.push((filt_levels[j] - filt_levels[i]) / dx);
            }
        }
    }
    if slopes.is_empty() {
        return signal.to_vec();
    }

    let slope = median_f64(&mut slopes);
    let mut residuals: Vec<f64> = filt_levels
        .iter()
        .zip(filt_means.iter())
        .map(|(&l, &m)| l - slope * m)
        .collect();
    let intercept = median_f64(&mut residuals);

    if slope.abs() < 1e-10 {
        return signal.to_vec();
    }

    signal.iter().map(|&s| (slope * s as f64 + intercept) as f32).collect()
}

/// Full signal refinement pipeline matching SigMapRefiner.refine().
fn refine_signal_map_pipeline(
    signal: &mut Vec<f32>,
    seq_to_sig_map: &mut Vec<i64>,
    sequence: &str,
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    kmer_center_idx: i32,
    half_bandwidth: i32,
    scale_iters: i32,
) {
    let levels_f64 = extract_levels_inner(sequence, kmer_to_level, kmer_len, kmer_center_idx);

    // Step 1: Rough rescale (quantile-based)
    *signal = rough_rescale_quantile(signal, &levels_f64, seq_to_sig_map, 10);

    // Step 2: Iterative banded DP refinement
    if scale_iters < 0 {
        return;
    }

    let mut work_signal = signal.clone();
    let n_iters = if scale_iters == 0 { 1 } else { scale_iters as usize };

    for _iteration in 0..n_iters {
        // Trim signal to mapped region
        let sig_start = seq_to_sig_map[0];
        let sig_end = *seq_to_sig_map.last().unwrap();
        if sig_start < 0 || sig_end as usize > work_signal.len() {
            return;
        }
        let trimmed_signal: Vec<f32> = work_signal[sig_start as usize..sig_end as usize].to_vec();
        let local_map: Vec<i32> = seq_to_sig_map.iter().map(|&v| (v - sig_start) as i32).collect();

        let levels_f32: Vec<f32> = levels_f64.iter().map(|&v| v as f32).collect();
        let seq_len = levels_f32.len();
        if seq_len == 0 || local_map.len() != seq_len + 1 {
            return;
        }

        // Compute band
        let (sig_bl, sig_bh) = compute_sig_band(&local_map, &levels_f64, half_bandwidth);
        if sig_bl.is_empty() {
            return;
        }
        let (mut seq_lo, mut seq_hi) = convert_to_seq_band(&sig_bl, &sig_bh, seq_len);
        adjust_seq_band(&mut seq_lo, &mut seq_hi, 2);

        // Validate band
        if seq_lo[0] != 0 || seq_hi.last().copied() != Some(trimmed_signal.len() as i32) {
            return;
        }

        // Replace NaN levels with 0
        let mut temp_levels = levels_f32.clone();
        for l in temp_levels.iter_mut() {
            if l.is_nan() {
                *l = 0.0;
            }
        }

        // Short dwell penalty: weight * (d - target)^2 for d < limit
        let (target, limit, weight) = (4i32, 3i32, 0.5f32);
        let sd_pen: Vec<f32> = (0..limit).map(|d| weight * ((d - target) * (d - target)) as f32).collect();

        // Run DP
        let path = seq_banded_dp_inner(
            &trimmed_signal,
            &temp_levels,
            &seq_lo,
            &seq_hi,
            &sd_pen,
            true, // dwell_penalty
        );

        // Validate monotonicity
        let monotonic = path.windows(2).all(|w| w[1] > w[0]);
        if !monotonic {
            return;
        }

        // Update sig_map
        *seq_to_sig_map = path.iter().map(|&v| v as i64 + sig_start).collect();

        // Inter-iteration rescaling
        if scale_iters > 0 {
            work_signal = theil_sen_rescale(&work_signal, &levels_f64, seq_to_sig_map, 10);
        }
    }

    *signal = work_signal;
}

// ---------------------------------------------------------------------------
// Per-base features
// ---------------------------------------------------------------------------

fn compute_per_base_stats(
    signal: &[f32],
    seq_to_sig: &[i64],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_bases = seq_to_sig.len().saturating_sub(1);
    let mut means = vec![0.0f32; num_bases];
    let mut medians = vec![0.0f32; num_bases];
    let mut stds = vec![0.0f32; num_bases];
    let mut ranges = vec![0.0f32; num_bases];
    let sig_len = signal.len() as i64;
    let mut sort_buf = Vec::new();

    for i in 0..num_bases {
        let start = seq_to_sig[i].max(0).min(sig_len) as usize;
        let end = seq_to_sig[i + 1].max(0).min(sig_len) as usize;
        if start >= end {
            continue;
        }
        let slice = &signal[start..end];
        let n = slice.len() as f32;
        let mut sum = 0.0f32;
        let mut mn = f32::MAX;
        let mut mx = f32::MIN;
        for &v in slice {
            sum += v;
            if v < mn { mn = v; }
            if v > mx { mx = v; }
        }
        means[i] = sum / n;
        ranges[i] = mx - mn;
        let mean = means[i];
        let var_sum: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum();
        stds[i] = (var_sum / n).sqrt();
        sort_buf.clear();
        sort_buf.extend_from_slice(slice);
        medians[i] = median_f32(&mut sort_buf);
    }
    (means, medians, stds, ranges)
}

fn compute_dwell_features(dwells: &[f32]) -> Vec<Vec<f32>> {
    let n = dwells.len();
    if n == 0 {
        return vec![vec![]; 5];
    }
    let window = 5usize;
    let pad = window / 2;
    let eps = 1e-6f32;

    let padded_len = n + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    for val in padded[..pad].iter_mut() {
        *val = dwells[0];
    }
    padded[pad..pad + n].copy_from_slice(dwells);
    for val in padded[pad + n..].iter_mut() {
        *val = dwells[n - 1];
    }

    let mut dwell_mean = vec![0.0f32; n];
    let mut dwell_std = vec![0.0f32; n];
    for i in 0..n {
        let win = &padded[i..i + window];
        let sum: f32 = win.iter().sum();
        let mean = sum / window as f32;
        dwell_mean[i] = mean;
        let var: f32 = win.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / window as f32;
        dwell_std[i] = var.sqrt();
    }

    let dwell_raw = dwells.to_vec();
    let dwell_log: Vec<f32> = dwells.iter().map(|&d| (d + eps).ln()).collect();
    let dwell_ratio: Vec<f32> = dwells.iter().zip(dwell_mean.iter()).map(|(&d, &m)| d / (m + eps)).collect();

    vec![dwell_raw, dwell_log, dwell_mean, dwell_std, dwell_ratio]
}

// ---------------------------------------------------------------------------
// Kmer residual features + signal residual
// ---------------------------------------------------------------------------

fn compute_kmer_residual_features(
    signal: &[f32],
    sig_map: &[i64],
    expected_levels: &[f64],
    num_bases: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut observed_mean = vec![0.0f32; num_bases];
    for i in 0..num_bases {
        let start = sig_map[i].max(0) as usize;
        let end = sig_map[i + 1].max(0) as usize;
        if end > start && end <= signal.len() {
            let sum: f32 = signal[start..end].iter().sum();
            observed_mean[i] = sum / (end - start) as f32;
        }
    }

    let kmer_expected: Vec<f32> = expected_levels.iter().map(|&v| v as f32).collect();
    let kmer_residual: Vec<f32> = observed_mean.iter().zip(kmer_expected.iter()).map(|(&o, &e)| o - e).collect();
    let kmer_residual_abs: Vec<f32> = kmer_residual.iter().map(|&r| r.abs()).collect();

    (kmer_expected, kmer_residual, kmer_residual_abs)
}

fn compute_signal_residual(
    signal: &[f32],
    sig_map: &[i64],
    expected_levels: &[f64],
    num_bases: usize,
) -> Vec<f32> {
    let sig_len = signal.len();
    let mut residual = vec![0.0f32; sig_len];
    for i in 0..num_bases {
        let start = sig_map[i].max(0) as usize;
        let end = sig_map[i + 1].max(0) as usize;
        let end = end.min(sig_len);
        let expected = expected_levels[i] as f32;
        if expected.abs() > 1e-12 {
            for s in start..end {
                residual[s] = signal[s] - expected;
            }
        }
    }
    residual
}

// ---------------------------------------------------------------------------
// Sequence encoding helpers
// ---------------------------------------------------------------------------

fn sequence_to_int(sequence: &[u8]) -> Vec<i8> {
    sequence
        .iter()
        .map(|&c| match c {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ => -1,
        })
        .collect()
}

fn encode_base_onehot(sequence: &[u8]) -> Vec<f32> {
    let seq_len = sequence.len();
    let mut encoding = vec![0.0f32; 4 * seq_len];
    for (i, &c) in sequence.iter().enumerate() {
        let idx = match c {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ => continue,
        };
        encoding[idx * seq_len + i] = 1.0;
    }
    encoding
}

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

/// Shared config for all reads in a batch (avoids per-read cloning).
struct PipelineConfig {
    reverse_signal: bool,
    use_reference: bool,
    use_signal_kmer: bool,
    skmer_ctx: (usize, usize),
    signal_context_left: i64,
    signal_context_right: i64,
    kmer_ctx: i64,
    kmer_win: usize,
    signal_len: usize,
    compute_features: bool,
    feat_start: i64,
    feat_end: i64,
    dwell_width: usize,
    refine_signal_map: bool,
    kmer_table: Option<HashMap<String, f64>>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
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

    let matched_reads = reader.reads_by_ids(&target_uuids).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to look up reads: {e}"))
    })?;
    let to_extract: Vec<(String, Vec<u64>)> = matched_reads
        .into_iter()
        .map(|r| (r.read_id.to_string(), r.signal_rows))
        .collect();

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
