//! Band computation and signal refinement pipeline.

use std::collections::HashMap;

use crate::signal_refine::{extract_levels_inner, seq_banded_dp_inner};

use super::numeric::{median_f64, percentile_f64};

fn compute_sig_band(bps: &[i32], levels: &[f64], bhw: i32) -> (Vec<i32>, Vec<i32>) {
    let seq_len = levels.len();
    let sig_len = (bps[seq_len] - bps[0]) as usize;

    let mut band_lo = vec![0i32; sig_len];
    let mut band_hi = vec![0i32; sig_len];

    // Running pointer to determine base index for each signal position,
    // avoiding a full Vec<i32> allocation of length sig_len.
    let mut base = 0usize;
    let mut base_end = (bps[1] - bps[0]) as usize;
    for s in 0..sig_len {
        while base + 1 < seq_len && s >= base_end {
            base += 1;
            base_end += (bps[base + 1] - bps[base]) as usize;
        }
        let b = base as i32;
        band_lo[s] = (b - bhw).max(0);
        band_hi[s] = (b + bhw + 1).min(seq_len as i32);
        // Handle NaN levels: route through NaN regions
        if base < levels.len() && levels[base].is_nan() {
            band_lo[s] = b;
            band_hi[s] = b + 1;
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

    // Fit: level = shift + scale * signal  ->  [1, sig_q] * [shift, scale] = lvl_q
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
    let max_pts = 200;
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
