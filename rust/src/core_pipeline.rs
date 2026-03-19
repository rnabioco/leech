//! Shared processing functions used by both inference and training pipelines.
//!
//! Extracted from `inference_pipeline.rs` to avoid duplication. Contains:
//! - Numeric helpers (median, percentile, interpolation)
//! - Signal normalization
//! - Move table → seq_to_sig_map conversion
//! - CIGAR → reference-to-signal mapping
//! - Band computation for signal refinement
//! - Signal refinement pipeline
//! - Per-base feature computation
//! - Sequence encoding helpers

use std::collections::HashMap;

use crate::signal_refine::{extract_levels_inner, seq_banded_dp_inner};

// ---------------------------------------------------------------------------
// Numeric helpers — O(n) median via select_nth_unstable
// ---------------------------------------------------------------------------

pub(crate) const F32_CMP: fn(&f32, &f32) -> std::cmp::Ordering =
    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
pub(crate) const F64_CMP: fn(&f64, &f64) -> std::cmp::Ordering =
    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

pub(crate) fn median_f32(data: &mut [f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
    let mid = n / 2;
    data.select_nth_unstable_by(mid, F32_CMP);
    if n % 2 == 0 {
        let hi = data[mid];
        let lo = data[..mid].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lo + hi) / 2.0
    } else {
        data[mid]
    }
}

pub(crate) fn median_f64(data: &mut [f64]) -> f64 {
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

pub(crate) fn percentile_f64(sorted: &[f64], pct: f64) -> f64 {
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
pub(crate) fn quantile_f32(data: &[f32], q: f32) -> f32 {
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
pub(crate) fn linear_interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
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

pub(crate) fn normalize_median_mad(signal: &[f32]) -> Vec<f32> {
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

pub(crate) fn normalize_zscore(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }
    let n = signal.len() as f32;
    let mean: f32 = signal.iter().sum::<f32>() / n;
    let var: f32 = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    if std < 1e-10 {
        return signal.iter().map(|&x| x - mean).collect();
    }
    signal.iter().map(|&x| (x - mean) / std).collect()
}

pub(crate) fn normalize_quantile(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }
    // Winsorize at 1st/99th percentile, then median-MAD
    let mut sorted = signal.to_vec();
    sorted.sort_unstable_by(F32_CMP);
    let n = sorted.len();
    let lo_idx = (0.01 * (n - 1) as f32).floor() as usize;
    let hi_idx = (0.99 * (n - 1) as f32).ceil().min((n - 1) as f32) as usize;
    let lo_val = sorted[lo_idx];
    let hi_val = sorted[hi_idx];
    let clipped: Vec<f32> = signal.iter().map(|&x| x.max(lo_val).min(hi_val)).collect();
    normalize_median_mad(&clipped)
}

pub(crate) fn normalize_pa_scaling(
    raw_i16: &[i16],
    cal_offset: f32,
    cal_scale: f32,
    pa_mean: f32,
    pa_stdev: f32,
) -> Vec<f32> {
    if raw_i16.is_empty() || pa_stdev.abs() < 1e-10 {
        return raw_i16.iter().map(|&x| x as f32).collect();
    }
    raw_i16
        .iter()
        .map(|&x| {
            let pa = (x as f32 + cal_offset) * cal_scale;
            (pa - pa_mean) / pa_stdev
        })
        .collect()
}

/// Dispatch normalization by method name. For pa_scaling, the caller must
/// pre-normalize since it needs DAC values (not trimmed f32).
pub(crate) fn normalize_signal(signal: &[f32], method: &str) -> Vec<f32> {
    match method {
        "median_mad" => normalize_median_mad(signal),
        "zscore" | "z_score" => normalize_zscore(signal),
        "quantile" => normalize_quantile(signal),
        _ => normalize_median_mad(signal), // fallback
    }
}

// ---------------------------------------------------------------------------
// Move table → seq_to_sig_map
// ---------------------------------------------------------------------------

pub(crate) fn build_seq_to_sig_map(mv_array: &[u8], stride: u32, trim_offset: i64, num_samples: u64) -> Vec<i64> {
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

pub(crate) fn make_ref_to_query_mapping(cigar_ops: &[(u32, u32)]) -> Vec<f64> {
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
            ref_knots.push(ref_pos as f64);
            query_knots.push(query_pos as f64);
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

    ref_knots.push(ref_pos as f64);
    query_knots.push(query_pos as f64);

    let x: Vec<f64> = (0..=ref_pos).map(|i| i as f64).collect();
    linear_interp(&x, &ref_knots, &query_knots)
}

pub(crate) fn compute_ref_to_signal(query_to_sig: &[i64], cigar_ops: &[(u32, u32)]) -> Vec<i64> {
    let ref_to_query = make_ref_to_query_mapping(cigar_ops);

    let xp: Vec<f64> = (0..query_to_sig.len()).map(|i| i as f64).collect();
    let fp: Vec<f64> = query_to_sig.iter().map(|&v| v as f64).collect();
    let result_f64 = linear_interp(&ref_to_query, &xp, &fp);

    result_f64.iter().map(|&v| v.floor() as i64).collect()
}

// ---------------------------------------------------------------------------
// Band computation (matches leech's Python implementation)
// ---------------------------------------------------------------------------

pub(crate) fn compute_sig_band(bps: &[i32], levels: &[f64], bhw: i32) -> (Vec<i32>, Vec<i32>) {
    let seq_len = levels.len();
    let sig_len = (bps[seq_len] - bps[0]) as usize;

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

    for s in 0..sig_len {
        let base = seq_indices[s] as usize;
        if base < levels.len() && levels[base].is_nan() {
            band_lo[s] = seq_indices[s];
            band_hi[s] = seq_indices[s] + 1;
        }
    }

    for s in 1..sig_len {
        band_lo[s] = band_lo[s].max(band_lo[s - 1]);
    }
    for s in (0..sig_len - 1).rev() {
        band_hi[s] = band_hi[s].min(band_hi[s + 1]);
    }

    (band_lo, band_hi)
}

pub(crate) fn convert_to_seq_band(sig_band_lo: &[i32], sig_band_hi: &[i32], seq_len: usize) -> (Vec<i32>, Vec<i32>) {
    let sig_len = sig_band_lo.len() as i32;
    let mut seq_lo = vec![0i32; seq_len];
    let mut seq_hi = vec![sig_len; seq_len];

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

pub(crate) fn adjust_seq_band(seq_lo: &mut [i32], seq_hi: &mut [i32], min_step: i32) {
    let n = seq_lo.len();
    if n == 0 {
        return;
    }

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

pub(crate) fn rough_rescale_quantile(signal: &[f32], expected: &[f64], sig_map: &[i64], clip_bases: usize) -> Vec<f32> {
    let num_bases = sig_map.len().saturating_sub(1);
    if num_bases == 0 {
        return signal.to_vec();
    }

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

    let quants: Vec<f64> = (1..20).map(|i| i as f64 * 0.05).collect();
    let mut sorted_sig = centers_sig.clone();
    sorted_sig.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_lvl = centers_lvl.clone();
    sorted_lvl.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sig_qs: Vec<f64> = quants.iter().map(|&q| percentile_f64(&sorted_sig, q * 100.0)).collect();
    let lvl_qs: Vec<f64> = quants.iter().map(|&q| percentile_f64(&sorted_lvl, q * 100.0)).collect();

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

pub(crate) fn theil_sen_rescale(
    signal: &[f32],
    levels: &[f64],
    sig_map: &[i64],
    edge_filter: usize,
) -> Vec<f32> {
    let num_bases = sig_map.len().saturating_sub(1);
    if num_bases < 20 {
        return signal.to_vec();
    }

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

    let max_pts = 1000;
    if filt_means.len() > max_pts {
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
pub(crate) fn refine_signal_map_pipeline(
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

        let (sig_bl, sig_bh) = compute_sig_band(&local_map, &levels_f64, half_bandwidth);
        if sig_bl.is_empty() {
            return;
        }
        let (mut seq_lo, mut seq_hi) = convert_to_seq_band(&sig_bl, &sig_bh, seq_len);
        adjust_seq_band(&mut seq_lo, &mut seq_hi, 2);

        if seq_lo[0] != 0 || seq_hi.last().copied() != Some(trimmed_signal.len() as i32) {
            return;
        }

        let mut temp_levels = levels_f32.clone();
        for l in temp_levels.iter_mut() {
            if l.is_nan() {
                *l = 0.0;
            }
        }

        let (target, limit, weight) = (4i32, 3i32, 0.5f32);
        let sd_pen: Vec<f32> = (0..limit).map(|d| weight * ((d - target) * (d - target)) as f32).collect();

        let path = seq_banded_dp_inner(
            &trimmed_signal,
            &temp_levels,
            &seq_lo,
            &seq_hi,
            &sd_pen,
            true,
        );

        let monotonic = path.windows(2).all(|w| w[1] > w[0]);
        if !monotonic {
            return;
        }

        *seq_to_sig_map = path.iter().map(|&v| v as i64 + sig_start).collect();

        if scale_iters > 0 {
            work_signal = theil_sen_rescale(&work_signal, &levels_f64, seq_to_sig_map, 10);
        }
    }

    *signal = work_signal;
}

// ---------------------------------------------------------------------------
// Per-base features
// ---------------------------------------------------------------------------

pub(crate) fn compute_per_base_stats(
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

pub(crate) fn compute_dwell_features(dwells: &[f32]) -> Vec<Vec<f32>> {
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

pub(crate) fn compute_kmer_residual_features(
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

pub(crate) fn compute_signal_residual(
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

pub(crate) fn sequence_to_int(sequence: &[u8]) -> Vec<i8> {
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

pub(crate) fn encode_base_onehot(sequence: &[u8]) -> Vec<f32> {
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
// Shared pipeline config
// ---------------------------------------------------------------------------

/// Shared config for all reads in a batch (avoids per-read cloning).
pub(crate) struct PipelineConfig {
    pub reverse_signal: bool,
    pub use_reference: bool,
    pub use_signal_kmer: bool,
    pub skmer_ctx: (usize, usize),
    pub signal_context_left: i64,
    pub signal_context_right: i64,
    pub kmer_ctx: i64,
    pub kmer_win: usize,
    pub signal_len: usize,
    pub compute_features: bool,
    pub feat_start: i64,
    pub feat_end: i64,
    pub dwell_width: usize,
    pub refine_signal_map: bool,
    pub kmer_table: Option<HashMap<String, f64>>,
    pub kmer_len: usize,
    pub kmer_center_idx: i32,
    pub refine_half_bandwidth: i32,
    pub refine_scale_iters: i32,
    pub signal_in_channels: usize,
}
