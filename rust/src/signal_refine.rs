use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
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

    let path = seq_banded_dp_inner(signal, levels, &band_lo, &band_hi, sd_pen, algo == "dwell_penalty");
    let result = Array1::from_vec(path);
    Ok(result.into_pyarray(py))
}

/// Internal level extraction (no PyO3 types).
pub(crate) fn extract_levels_inner(
    sequence: &str,
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    center_idx: i32,
) -> Vec<f64> {
    let seq_bytes = sequence.as_bytes();
    let seq_len = seq_bytes.len();
    let mut levels = vec![0.0f64; seq_len];
    let cidx = if center_idx < 0 {
        kmer_len / 2
    } else {
        center_idx as usize
    };

    if seq_len < kmer_len {
        return levels;
    }

    let seq_upper: Vec<u8> = seq_bytes
        .iter()
        .map(|&b| {
            let c = b.to_ascii_uppercase();
            if c == b'U' { b'T' } else { c }
        })
        .collect();

    for pos in 0..=(seq_len - kmer_len) {
        let kmer = std::str::from_utf8(&seq_upper[pos..pos + kmer_len]).unwrap_or("");
        if let Some(&level) = kmer_to_level.get(kmer) {
            levels[pos + cidx] = level;
        }
    }

    levels
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

/// Internal rough rescaling (no PyO3 types). Returns rescaled signal or original.
pub(crate) fn rough_rescale_inner(
    signal: &[f64],
    expected: &[f64],
    sig_map: &[i64],
) -> Vec<f64> {
    let num_bases = sig_map.len().saturating_sub(1);
    let mut observed = vec![0.0f64; num_bases];

    for i in 0..num_bases {
        let start = sig_map[i] as usize;
        let end = sig_map[i + 1] as usize;
        if end > start && end <= signal.len() {
            let sum: f64 = signal[start..end].iter().sum();
            observed[i] = sum / (end - start) as f64;
        } else if i > 0 {
            observed[i] = observed[i - 1];
        }
    }

    if num_bases > 1 {
        let n = num_bases as f64;
        let sum_x: f64 = expected.iter().sum();
        let sum_y: f64 = observed.iter().sum();
        let sum_xx: f64 = expected.iter().map(|&x| x * x).sum();
        let sum_xy: f64 = expected
            .iter()
            .zip(observed.iter())
            .map(|(&x, &y)| x * y)
            .sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-10 {
            let a = (n * sum_xy - sum_x * sum_y) / denom;
            let b = (sum_y - a * sum_x) / n;

            if a.abs() > 1e-10 {
                return signal.iter().map(|&s| (s - b) / a).collect();
            }
        }
    }

    signal.to_vec()
}

/// PyO3 wrapper for rough rescaling.
#[pyfunction]
pub fn rough_rescale<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    expected_levels: PyReadonlyArray1<'py, f64>,
    seq_to_sig_map: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sig = signal.as_slice()?;
    let exp = expected_levels.as_slice()?;
    let map = seq_to_sig_map.as_slice()?;
    let result = rough_rescale_inner(sig, exp, map);
    Ok(Array1::from_vec(result).into_pyarray(py))
}
