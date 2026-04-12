//! Numeric helpers and normalization.

pub(super) const F32_CMP: fn(&f32, &f32) -> std::cmp::Ordering =
    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
pub(super) const F64_CMP: fn(&f64, &f64) -> std::cmp::Ordering =
    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

pub(super) fn median_f32(data: &mut [f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
    // Full sort to match numpy.median exactly (numpy sorts internally).
    // select_nth_unstable is faster but can pick different adjacent elements
    // for even-length arrays when float32 values are very close, causing
    // ~1e-7 normalization differences that cascade through refinement.
    data.sort_unstable_by(F32_CMP);
    let mid = n / 2;
    if n % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

pub(super) fn median_f64(data: &mut [f64]) -> f64 {
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

pub(super) fn percentile_f64(sorted: &[f64], pct: f64) -> f64 {
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
pub(super) fn quantile_f32(data: &[f32], q: f32) -> f32 {
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

pub(super) fn normalize_median_mad(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }
    let mut buf = signal.to_vec();
    let med = median_f32(&mut buf);
    // Reuse buf for MAD computation (overwrite with absolute deviations)
    for (v, &s) in buf.iter_mut().zip(signal.iter()) {
        *v = (s - med).abs();
    }
    let mad = median_f32(&mut buf);
    let scale = mad * 1.4826;
    if scale < 1e-10 {
        return signal.iter().map(|&x| x - med).collect();
    }
    signal.iter().map(|&x| (x - med) / scale).collect()
}
