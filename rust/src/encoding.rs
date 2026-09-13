use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
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
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Batched form of [`encode_signal_kmer`], one pyo3 call for a whole
/// DataLoader batch instead of one per chunk (rnabioco/leech#260).
///
/// `sequence_ints` and `seq_to_sig_map` are the *padded rectangular* arrays
/// `LeechDataset` already stores (`(rows, W1)` int8, `(rows, W2)` int32): each
/// row is independently padded to its array's max width with sentinels the
/// per-row encoder already treats as "nothing here" — `seq_ints < 0` and
/// `seq_to_sig == signal_len` both cause a position to be skipped, and a
/// real (unpadded) `seq_to_sig_map` row's own last element is always
/// `signal_len` by construction (`LeechRead.get_chunk` snaps it to the
/// window edge), so the row's real/padded boundary is invisible to the
/// encoder. That means a uniform-stride CSR view of these rectangular
/// arrays — offset `i * W` for row `i` — is exact, with no need to track
/// each row's real (unpadded) length separately.
///
/// `seq_to_sig_map` arrives as `i32` (the dataset stores it that way to
/// halve its resident memory; values are always in `[0, signal_len]`) and is
/// widened to `i64` here, which is what `SignalKmerBatch` requires upstream.
#[pyfunction]
pub fn encode_signal_kmer_batch<'py>(
    py: Python<'py>,
    sequence_ints: PyReadonlyArray2<'py, i8>,
    seq_to_sig_map: PyReadonlyArray2<'py, i32>,
    signal_len: i64,
    kmer_before: usize,
    kmer_after: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let seq_ints_view = sequence_ints.as_array();
    let seq_to_sig_view = seq_to_sig_map.as_array();
    let (rows, w1) = seq_ints_view.dim();
    let (rows2, w2) = seq_to_sig_view.dim();
    if rows != rows2 {
        return Err(PyValueError::new_err(format!(
            "sequence_ints has {rows} rows but seq_to_sig_map has {rows2}"
        )));
    }
    let sig_len = signal_len as usize;
    let kmer_len = kmer_before + 1 + kmer_after;
    let channels = 4 * kmer_len;

    // Row-major flatten regardless of the input arrays' actual memory
    // layout — `.iter()` follows each array's logical (row, col) order.
    let seq_ints_flat: Vec<i8> = seq_ints_view.iter().copied().collect();
    let seq_to_sig_flat: Vec<i64> = seq_to_sig_view.iter().map(|&v| i64::from(v)).collect();
    let seq_ints_offsets: Vec<usize> = (0..=rows).map(|i| i * w1).collect();
    let seq_to_sig_offsets: Vec<usize> = (0..=rows).map(|i| i * w2).collect();

    let out = py.detach(|| {
        let batch = escapepod_signal::seq_encoding::SignalKmerBatch {
            seq_ints: &seq_ints_flat,
            seq_ints_offsets: &seq_ints_offsets,
            seq_to_signal: &seq_to_sig_flat,
            seq_to_signal_offsets: &seq_to_sig_offsets,
        };
        let ctx = escapepod_signal::seq_encoding::KmerContext {
            before: kmer_before,
            after: kmer_after,
        };
        let mut out = vec![0.0f32; rows * channels * sig_len];
        escapepod_signal::seq_encoding::encode_signal_kmer_batch_into(
            &batch, sig_len, ctx, &mut out,
        );
        out
    });

    let arr = Array3::from_shape_vec((rows, channels, sig_len), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}
