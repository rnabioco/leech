use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use escapepod_signal::resquiggle::banded_dp_with_penalty_table;

/// Internal banded Viterbi DP — delegates to escapepod-signal.
pub(crate) fn seq_banded_dp_inner(
    signal: &[f32],
    levels: &[f32],
    band_lo: &[i32],
    band_hi: &[i32],
    sd_pen: &[f32],
    use_dwell_pen: bool,
) -> Vec<i32> {
    banded_dp_with_penalty_table(signal, levels, band_lo, band_hi, sd_pen, use_dwell_pen)
}

/// PyO3 wrapper for banded Viterbi DP.
#[pyfunction]
pub fn seq_banded_dp<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    levels: PyReadonlyArray1<'py, f32>,
    seq_band: PyReadonlyArray2<'py, i32>,
    short_dwell_penalty: PyReadonlyArray1<'py, f32>,
    algo: &str,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let signal = signal.as_slice()?;
    let levels = levels.as_slice()?;
    let sd_pen = short_dwell_penalty.as_slice()?;
    let seq_band_arr = seq_band.as_array();
    let seq_len = levels.len();

    let mut band_lo = vec![0i32; seq_len];
    let mut band_hi = vec![0i32; seq_len];
    for i in 0..seq_len {
        band_lo[i] = seq_band_arr[[0, i]];
        band_hi[i] = seq_band_arr[[1, i]];
    }

    let path = seq_banded_dp_inner(
        signal,
        levels,
        &band_lo,
        &band_hi,
        sd_pen,
        algo == "dwell_penalty",
    );
    let result = Array1::from_vec(path);
    Ok(result.into_pyarray(py))
}

/// Internal level extraction — delegates to escapepod-signal, the canonical
/// implementation (rnabioco/escapepod-rs#204). `center_idx < 0` selects the
/// default center `kmer_len / 2`.
pub(crate) fn extract_levels_inner(
    sequence: &str,
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    center_idx: i32,
) -> Vec<f64> {
    let center = usize::try_from(center_idx).ok();
    escapepod_signal::resquiggle::extract_levels(sequence, kmer_to_level, kmer_len, center)
}

/// PyO3 wrapper for level extraction.
#[pyfunction]
#[pyo3(signature = (sequence, kmer_to_level, kmer_len, center_idx = -1))]
pub fn extract_levels<'py>(
    py: Python<'py>,
    sequence: &str,
    kmer_to_level: HashMap<String, f64>,
    kmer_len: usize,
    center_idx: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let levels = extract_levels_inner(sequence, &kmer_to_level, kmer_len, center_idx);
    let result = Array1::from_vec(levels);
    Ok(result.into_pyarray(py))
}

/// PyO3 wrapper for the quantile rough rescale — delegates to
/// escapepod-signal's `rough_rescale_quantile`, the canonical implementation
/// (rnabioco/escapepod-rs#204), bit-for-bit compatible with leech's NumPy
/// version for float32 inputs.
#[pyfunction]
#[pyo3(signature = (signal, expected_levels, seq_to_sig_map, clip_bases = 10))]
pub fn rough_rescale_quantile<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    expected_levels: PyReadonlyArray1<'py, f32>,
    seq_to_sig_map: PyReadonlyArray1<'py, i64>,
    clip_bases: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let rescaled = escapepod_signal::resquiggle::rough_rescale_quantile(
        signal.as_slice()?,
        expected_levels.as_slice()?,
        seq_to_sig_map.as_slice()?,
        clip_bases,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Array1::from_vec(rescaled).into_pyarray(py))
}
