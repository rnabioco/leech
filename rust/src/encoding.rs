use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Internal signal-level kmer encoding. Returns flat row-major Vec of shape
/// `(4 * kmer_len, signal_len)`.
pub(crate) fn encode_signal_kmer_inner(
    seq_ints: &[i8],
    sig_map: &[i64],
    signal_len: usize,
    kmer_before: usize,
    kmer_after: usize,
) -> Vec<f32> {
    let kmer_len = kmer_before + 1 + kmer_after;
    let seq_len = sig_map.len().saturating_sub(1);
    let enc_dim = 4 * kmer_len;
    let mut enc = vec![0.0f32; enc_dim * signal_len];

    for kmer_pos in 0..kmer_len {
        let offset = 4 * kmer_pos;
        for seq_pos in 0..seq_len {
            let base_idx = seq_pos + kmer_pos;
            if base_idx >= seq_ints.len() {
                continue;
            }
            let base = seq_ints[base_idx];
            if base < 0 {
                continue;
            }
            let sig_start = (sig_map[seq_pos] as usize).min(signal_len);
            let sig_end = (sig_map[seq_pos + 1] as usize).min(signal_len);
            if sig_start < sig_end {
                let row = offset + base as usize;
                for s in sig_start..sig_end {
                    enc[row * signal_len + s] = 1.0;
                }
            }
        }
    }

    enc
}

/// PyO3 wrapper for signal-level kmer encoding.
#[pyfunction]
pub fn encode_signal_kmer<'py>(
    py: Python<'py>,
    sequence_ints: PyReadonlyArray1<'py, i8>,
    seq_to_sig_map: PyReadonlyArray1<'py, i64>,
    signal_len: i64,
    kmer_before: usize,
    kmer_after: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let seq_ints = sequence_ints.as_slice()?;
    let sig_map = seq_to_sig_map.as_slice()?;
    let sig_len = signal_len as usize;
    let kmer_len = kmer_before + 1 + kmer_after;

    let flat = encode_signal_kmer_inner(seq_ints, sig_map, sig_len, kmer_before, kmer_after);
    let arr = Array2::from_shape_vec((4 * kmer_len, sig_len), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}
