//! Numeric helpers and normalization.

pub(super) const F32_CMP: fn(&f32, &f32) -> std::cmp::Ordering =
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
    if n.is_multiple_of(2) {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
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

/// Median-MAD normalization with the Gaussian scale factor (1.4826).
///
/// Delegates to `escapepod_signal::segmentation::mad_normalize_robust` so
/// this crate and escapepod-signal share the same implementation.
pub(super) fn normalize_median_mad(signal: &[f32]) -> Vec<f32> {
    escapepod_signal::segmentation::mad_normalize_robust(signal)
}
