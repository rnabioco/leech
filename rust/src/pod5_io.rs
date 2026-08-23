//! Batch POD5 reading via escapepod-rs.
//!
//! Replaces Python pod5 library's `DatasetReader.reads()` with escapepod's
//! memory-mapped reader and bulk signal extraction.
//!
//! Read lookup goes through [`crate::pod5_cache`], which opens each POD5 once
//! per process and builds its read-id index once. That is load-bearing, not an
//! optimization: without an index every lookup degrades to a single-threaded
//! scan of the entire reads table, which on a large POD5 costs minutes per
//! batch. Never call `Reader::open` directly from this crate.

use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::pod5_cache::{read_signal_map_by_ids, read_signals_by_ids};

/// One read's POD5 result: (signal_i16, calibration_offset, calibration_scale).
type Pod5ReadResult = (Py<PyArray1<i16>>, f32, f32);

/// Read raw DAC signals for a batch of reads from a POD5 file.
///
/// Resolves the read IDs against the shared indexed reader, then bulk-extracts
/// every signal in one batch-grouped pass. The GIL is released for the whole
/// call, so several Python threads may read concurrently.
///
/// Returns a dict mapping `read_id` → `(signal_i16, cal_offset, cal_scale)`.
#[pyfunction]
pub fn read_pod5_batch<'py>(
    py: Python<'py>,
    pod5_path: &str,
    read_ids: Vec<String>,
) -> PyResult<HashMap<String, Pod5ReadResult>> {
    let raw = py
        .detach(|| read_signals_by_ids(pod5_path, &read_ids))
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    let mut results = HashMap::with_capacity(raw.len());
    for read in raw {
        let arr = read.signal.into_pyarray(py).unbind();
        results.insert(
            read.read_id,
            (arr, read.calibration_offset, read.calibration_scale),
        );
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Opaque preloaded signals handle for prefetch pipeline
// ---------------------------------------------------------------------------

/// Opaque handle holding raw i16 signals keyed by read ID.
///
/// Created by [`preload_pod5_signals`] and consumed by
/// `extract_chunks_from_preloaded` in the inference pipeline.
/// Stays in Rust memory — no numpy conversion, no calibration.
#[pyclass]
pub struct PreloadedSignals {
    pub(crate) signals: HashMap<String, Vec<i16>>,
}

#[pymethods]
impl PreloadedSignals {
    fn __len__(&self) -> usize {
        self.signals.len()
    }
}

/// Pre-read raw DAC signals from a POD5 file, returning an opaque handle.
///
/// Releases the GIL during I/O so a background thread can prefetch while
/// the main thread does other work (BAM metadata extraction, GPU inference).
/// The returned [`PreloadedSignals`] is consumed by
/// `extract_chunks_from_preloaded`.
#[pyfunction]
pub fn preload_pod5_signals(
    py: Python<'_>,
    pod5_path: &str,
    read_ids: Vec<String>,
) -> PyResult<PreloadedSignals> {
    // Release GIL during I/O-heavy POD5 reading so other Python threads
    // (e.g. main thread doing BAM metadata extraction) can proceed.
    let result = py.detach(|| read_signal_map_by_ids(pod5_path, &read_ids));

    match result {
        Ok(signals) => Ok(PreloadedSignals { signals }),
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
    }
}
