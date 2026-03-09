use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Signal-level kmer encoding: (4 * kmer_len, signal_len) float32.
///
/// For each signal position, encodes the kmer context centered on the base
/// occupying that position as a one-hot vector (4 channels per kmer position).
///
/// Args:
///     sequence_ints: Int-encoded bases with context,
///         len = seq_len + kmer_before + kmer_after.
///         Values 0-3 for ACGT, <0 for unknown (skipped).
///     seq_to_sig_map: Base-to-signal mapping for the core seq_len bases,
///         len = seq_len + 1. Values are signal indices.
///     signal_len: Length of the signal array.
///     kmer_before: Number of kmer context bases before center.
///     kmer_after: Number of kmer context bases after center.
///
/// Returns:
///     Encoding array of shape (4 * kmer_len, signal_len), dtype float32.
///     kmer_len = kmer_before + 1 + kmer_after.
#[pyfunction]
pub fn encode_signal_kmer<'py>(
    py: Python<'py>,
    sequence_ints: PyReadonlyArray1<'py, i8>,
    seq_to_sig_map: PyReadonlyArray1<'py, i64>,
    signal_len: i64,
    kmer_before: usize,
    kmer_after: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let seq_ints = sequence_ints.as_array();
    let sig_map = seq_to_sig_map.as_array();
    let sig_len = signal_len as usize;
    let kmer_len = kmer_before + 1 + kmer_after;
    let seq_len = sig_map.len() - 1;

    let mut enc = Array2::<f32>::zeros((4 * kmer_len, sig_len));

    let sig_map_slice = sig_map.as_slice().expect("sig_map must be contiguous");
    let seq_slice = seq_ints.as_slice().expect("seq_ints must be contiguous");

    for kmer_pos in 0..kmer_len {
        let offset = 4 * kmer_pos;
        for seq_pos in 0..seq_len {
            let base_idx = seq_pos + kmer_pos;
            if base_idx >= seq_slice.len() {
                continue;
            }
            let base = seq_slice[base_idx];
            if base < 0 {
                continue;
            }
            let sig_start = sig_map_slice[seq_pos] as usize;
            let sig_end = sig_map_slice[seq_pos + 1] as usize;
            let actual_start = if sig_start < sig_len {
                sig_start
            } else {
                sig_len
            };
            let actual_end = if sig_end < sig_len { sig_end } else { sig_len };
            if actual_start < actual_end {
                let row = offset + base as usize;
                for s in actual_start..actual_end {
                    enc[[row, s]] = 1.0;
                }
            }
        }
    }

    Ok(enc.into_pyarray(py))
}
