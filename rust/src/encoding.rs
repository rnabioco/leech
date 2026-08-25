use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Signal-level k-mer encoding, from `escapepod-signal`.
///
/// The rule used to live here, and it was the only copy — inside a `cdylib`
/// that Rust cannot link, so any native runtime for a leech `signal_kmer` model
/// had to transcribe it (rnabioco/escapepod-rs#271). It is the natural pair to
/// `escapepod_signal::mapping`, which *produces* the base-to-signal map this
/// consumes, so upstream now owns both halves and this is a call.
///
/// Returns a flat row-major `Vec` of shape `(4 * kmer_len, signal_len)`.
pub(crate) fn encode_signal_kmer_inner(
    seq_ints: &[i8],
    sig_map: &[i64],
    signal_len: usize,
    kmer_before: usize,
    kmer_after: usize,
) -> Vec<f32> {
    escapepod_signal::seq_encoding::encode_signal_kmer(
        seq_ints,
        sig_map,
        signal_len,
        escapepod_signal::seq_encoding::KmerContext {
            before: kmer_before,
            after: kmer_after,
        },
    )
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
