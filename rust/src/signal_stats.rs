use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::inference_pipeline::compute_per_base_stats;

/// Compute per-base signal statistics: mean, median, std, range.
///
/// Replaces the Python loop in features.py:compute_signal_features() that
/// called np.median/np.std/np.max/np.min per base — very slow due to
/// per-call Python→C dispatch overhead on small slices (~5-50 samples).
///
/// Delegates to the inference pipeline's `compute_per_base_stats`, which is
/// what the Rust extraction path uses. This file used to carry a second,
/// near-identical implementation; the two differed on negative map entries
/// (this one cast `i64` to `usize` raw and skipped the base, leaving zeros;
/// the other clamps to 0 and computes over the truncated span), so the Python
/// fast path and the Rust pipeline could report different level features for
/// the same read. Two implementations of one statistic is one too many —
/// keep this a wrapper.
///
/// Args:
///     signal: Normalized signal array (float32)
///     seq_to_sig_map: Base-to-signal mapping (int64), length = num_bases + 1
///
/// Returns:
///     Tuple of (mean, median, std, range) arrays, each of length num_bases
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn compute_signal_stats<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    seq_to_sig_map: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let signal = signal.as_array();
    let sig_map = seq_to_sig_map.as_array();

    let sig_slice = signal.as_slice().expect("signal must be contiguous");
    let map_slice = sig_map.as_slice().expect("sig_map must be contiguous");

    let (means, medians, stds, ranges) = compute_per_base_stats(sig_slice, map_slice);

    Ok((
        Array1::from_vec(means).into_pyarray(py),
        Array1::from_vec(medians).into_pyarray(py),
        Array1::from_vec(stds).into_pyarray(py),
        Array1::from_vec(ranges).into_pyarray(py),
    ))
}
