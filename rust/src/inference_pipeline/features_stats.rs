//! Per-base signal statistics, delegated to escapepod.

use escapepod_signal::features::{
    MedianConvention, Normalization, SpanBounds, SpanConfig, SpanFill, SpanScratch, SpanStatsOut,
    span_stats,
};

/// Per-base mean, median, population sd and range over `signal`.
///
/// Delegates to `escapepod_signal::features::span_stats`, which is the same
/// reduction done better: one pass over the covered region with `f64` prefix
/// sums and O(1) per span, rather than a fresh accumulation per base in `f32`.
///
/// The three policies below are the ones leech used to encode by re-implementing
/// the whole function (escapepod-rs#260):
///
/// - [`SpanFill::Zero`] — an unresolved base reads as 0, not `NaN`. These arrays
///   feed a neural network, where one `NaN` poisons the forward pass.
/// - [`SpanBounds::Clamp`] — a span that starts before the signal is truncated
///   into it rather than skipped. Under `anchor="reference"` a map entry can be
///   negative after the aligned-region crop, and the truncated span still holds
///   real signal. leech's two former copies disagreed here (one skipped, one
///   clamped), which is what made them a divergence rather than duplication.
/// - [`MedianConvention::SortPartialCmp`] — the `numpy.median` convention, so
///   the Rust path and the numpy fallback in `features.py` agree. escapepod
///   measures the two conventions as bit-identical over finite values; they
///   part company only on `NaN`, which [`SpanFill::Zero`] means we do not have.
///
/// The signal is already median-MAD normalized by `process_read_signal`, so the
/// normalization here is [`Normalization::None`].
pub(crate) fn compute_per_base_stats(
    signal: &[f32],
    seq_to_sig: &[i64],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_bases = seq_to_sig.len().saturating_sub(1);
    let mut means = vec![0.0f32; num_bases];
    let mut medians = vec![0.0f32; num_bases];
    let mut stds = vec![0.0f32; num_bases];
    let mut ranges = vec![0.0f32; num_bases];
    if num_bases == 0 {
        return (means, medians, stds, ranges);
    }

    let spans: Vec<[i64; 2]> = (0..num_bases)
        .map(|i| [seq_to_sig[i], seq_to_sig[i + 1]])
        .collect();

    // `dwell` is required by the output struct but leech computes dwells from
    // the map directly (they must stay exact integers); this buffer is discarded.
    let mut dwell_scratch = vec![0.0f32; num_bases];
    let mut scratch = SpanScratch::default();

    span_stats(
        signal,
        &spans,
        SpanConfig {
            norm: Normalization::None,
            fill: SpanFill::Zero,
            bounds: SpanBounds::Clamp,
            median: MedianConvention::SortPartialCmp,
        },
        &mut scratch,
        SpanStatsOut {
            dwell: &mut dwell_scratch,
            mean: &mut means,
            sd: &mut stds,
            median: Some(&mut medians),
            range: Some(&mut ranges),
        },
    );

    (means, medians, stds, ranges)
}
