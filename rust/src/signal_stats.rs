use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute per-base signal statistics: mean, median, std, range.
///
/// Replaces the Python loop in features.py:compute_signal_features() that
/// called np.median/np.std/np.max/np.min per base — very slow due to
/// per-call Python→C dispatch overhead on small slices (~5-50 samples).
///
/// Args:
///     signal: Normalized signal array (float32)
///     seq_to_sig_map: Base-to-signal mapping (int64), length = num_bases + 1
///
/// Returns:
///     Tuple of (mean, median, std, range) arrays, each of length num_bases
#[pyfunction]
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
    let sig_len = sig_slice.len();
    let num_bases = if map_slice.len() > 0 {
        map_slice.len() - 1
    } else {
        0
    };

    let mut means = Array1::<f32>::zeros(num_bases);
    let mut medians = Array1::<f32>::zeros(num_bases);
    let mut stds = Array1::<f32>::zeros(num_bases);
    let mut ranges = Array1::<f32>::zeros(num_bases);

    // Reusable buffer for median computation (avoids per-base allocation)
    let mut sort_buf: Vec<f32> = Vec::with_capacity(64);

    for i in 0..num_bases {
        let start = map_slice[i] as usize;
        let end = map_slice[i + 1] as usize;

        if start >= end || start >= sig_len {
            continue;
        }

        let end = end.min(sig_len);
        let segment = &sig_slice[start..end];
        let n = segment.len() as f32;

        // Single pass: mean, min, max
        let mut sum = 0.0f32;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &x in segment {
            sum += x;
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }
        let mean = sum / n;
        means[i] = mean;
        ranges[i] = max_val - min_val;

        // Second pass: variance (population std, matching numpy default)
        let mut var_sum = 0.0f32;
        for &x in segment {
            let d = x - mean;
            var_sum += d * d;
        }
        stds[i] = (var_sum / n).sqrt();

        // Median via sort (segments are typically small, ~5-50 samples)
        sort_buf.clear();
        sort_buf.extend_from_slice(segment);
        sort_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sort_buf.len() / 2;
        medians[i] = if sort_buf.len() % 2 == 0 {
            (sort_buf[mid - 1] + sort_buf[mid]) / 2.0
        } else {
            sort_buf[mid]
        };
    }

    Ok((
        means.into_pyarray(py),
        medians.into_pyarray(py),
        stds.into_pyarray(py),
        ranges.into_pyarray(py),
    ))
}
