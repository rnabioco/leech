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

use std::collections::{HashMap, HashSet};

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::pod5_cache::cached_reader;

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
    let target_uuids: HashSet<escapepod_signal::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod_signal::Uuid::parse_str(s).ok())
        .collect();

    type RawBatch = Vec<(String, Vec<i16>, f32, f32)>;
    let raw: RawBatch = py
        .detach(|| -> Result<RawBatch, String> {
            let reader = cached_reader(pod5_path)?;

            let matched_reads = reader
                .reads_by_ids(&target_uuids)
                .map_err(|e| format!("Failed to look up reads: {e}"))?;

            let mut to_extract: Vec<(String, Vec<u64>)> = Vec::with_capacity(matched_reads.len());
            let mut calibrations: HashMap<String, (f32, f32)> =
                HashMap::with_capacity(matched_reads.len());
            for read in matched_reads {
                let rid = read.read_id.to_string();
                calibrations.insert(
                    rid.clone(),
                    (read.calibration_offset, read.calibration_scale),
                );
                to_extract.push((rid, read.signal_rows));
            }

            // Bulk extract all signals (batched decompression, much faster than per-read)
            let signals = reader
                .get_signal_bulk(&to_extract)
                .map_err(|e| format!("Signal extraction failed: {e}"))?;

            Ok(signals
                .into_iter()
                .map(|(rid, signal)| {
                    let (off, scale) = calibrations.get(&rid).copied().unwrap_or((0.0, 1.0));
                    (rid, signal, off, scale)
                })
                .collect())
        })
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    let mut results = HashMap::with_capacity(raw.len());
    for (rid, signal, cal_offset, cal_scale) in raw {
        let arr = signal.into_pyarray(py).unbind();
        results.insert(rid, (arr, cal_offset, cal_scale));
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
    let target_uuids: HashSet<escapepod_signal::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod_signal::Uuid::parse_str(s).ok())
        .collect();

    // Release GIL during I/O-heavy POD5 reading so other Python threads
    // (e.g. main thread doing BAM metadata extraction) can proceed.
    let result: Result<HashMap<String, Vec<i16>>, String> = py.detach(|| {
        let reader = cached_reader(pod5_path)?;

        let matched_reads = reader
            .reads_by_ids(&target_uuids)
            .map_err(|e| format!("Failed to look up reads: {e}"))?;

        let to_extract: Vec<(String, Vec<u64>)> = matched_reads
            .into_iter()
            .map(|r| (r.read_id.to_string(), r.signal_rows))
            .collect();

        let bulk_signals = reader
            .get_signal_bulk(&to_extract)
            .map_err(|e| format!("Signal extraction failed: {e}"))?;

        Ok(bulk_signals.into_iter().collect())
    });

    match result {
        Ok(signals) => Ok(PreloadedSignals { signals }),
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
    }
}
