//! Normalization.
//!
//! This module used to carry a `median_f32` that full-sorted rather than using
//! `select_nth_unstable`, to match `numpy.median`'s choice between the two
//! middle elements of an even-length span. That rule now lives upstream as
//! `MedianConvention::SortPartialCmp` and is selected in `features_stats`, so
//! there is nothing to keep in sync here — see escapepod-rs#260.

/// Median-MAD normalization with the Gaussian scale factor (1.4826).
///
/// Delegates to `escapepod_signal::segmentation::mad_normalize_robust` so
/// this crate and escapepod-signal share the same implementation.
pub(super) fn normalize_median_mad(signal: &[f32]) -> Vec<f32> {
    escapepod_signal::segmentation::mad_normalize_robust(signal)
}
