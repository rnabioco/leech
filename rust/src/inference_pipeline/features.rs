//! Per-base feature computation and sequence encoding.

use super::numeric::median_f32;

pub(super) fn compute_per_base_stats(
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

pub(super) fn compute_dwell_features(dwells: &[f32]) -> Vec<Vec<f32>> {
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

pub(super) fn compute_kmer_residual_features(
    observed_means: &[f32],
    expected_levels: &[f64],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kmer_expected: Vec<f32> = expected_levels.iter().map(|&v| v as f32).collect();
    let kmer_residual: Vec<f32> = observed_means.iter().zip(kmer_expected.iter()).map(|(&o, &e)| o - e).collect();
    let kmer_residual_abs: Vec<f32> = kmer_residual.iter().map(|&r| r.abs()).collect();

    (kmer_expected, kmer_residual, kmer_residual_abs)
}

pub(super) fn compute_signal_residual(
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

pub(super) fn sequence_to_int(sequence: &[u8]) -> Vec<i8> {
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

pub(super) fn encode_base_onehot(sequence: &[u8]) -> Vec<f32> {
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
