//! Batch POD5 reading via escapepod-rs.
//!
//! Replaces Python pod5 library's `DatasetReader.reads()` with escapepod's
//! memory-mapped reader and bulk signal extraction for ~25-43x faster I/O.

use std::collections::{HashMap, HashSet};

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// Iterate reads, filter by UUID, and collect (read_id, signal_rows) pairs.
/// Includes calibration data when `with_cal` is true.
fn collect_matching_reads(
    reader: &escapepod::Reader,
    target_uuids: &HashSet<escapepod::Uuid>,
) -> Result<
    (
        Vec<(String, Vec<u64>)>,
        HashMap<String, (f32, f32)>,
    ),
    String,
> {
    let n = target_uuids.len();
    let mut to_extract: Vec<(String, Vec<u64>)> = Vec::with_capacity(n);
    let mut calibrations: HashMap<String, (f32, f32)> = HashMap::with_capacity(n);

    for read_result in reader
        .reads()
        .map_err(|e| format!("Failed to iterate reads: {e}"))?
    {
        let read = read_result.map_err(|e| format!("Failed to parse read: {e}"))?;
        if target_uuids.contains(&read.read_id) {
            let rid = read.read_id.to_string();
            calibrations.insert(rid.clone(), (read.calibration_offset, read.calibration_scale));
            to_extract.push((rid, read.signal_rows));
            if to_extract.len() == n {
                break; // Early termination — all targets found
            }
        }
    }

    Ok((to_extract, calibrations))
}

/// Read raw DAC signals for a batch of reads from a POD5 file.
///
/// Opens the file once, scans for matching read IDs, and bulk-extracts
/// all signals via escapepod's LRU-cached signal decompression.
///
/// Returns a dict mapping `read_id` → `(signal_i16, cal_offset, cal_scale)`.
#[pyfunction]
pub fn read_pod5_batch<'py>(
    py: Python<'py>,
    pod5_path: &str,
    read_ids: Vec<String>,
) -> PyResult<HashMap<String, (Py<PyArray1<i16>>, f32, f32)>> {
    let target_uuids: HashSet<escapepod::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod::Uuid::parse_str(s).ok())
        .collect();

    let reader = escapepod::Reader::open(pod5_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to open POD5 {pod5_path}: {e}"))
    })?;

    let (to_extract, calibrations) =
        collect_matching_reads(&reader, &target_uuids).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(e)
        })?;

    // Bulk extract all signals (batched decompression, much faster than per-read)
    let signals = reader.get_signal_bulk(&to_extract).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Signal extraction failed: {e}"))
    })?;

    let mut results = HashMap::with_capacity(signals.len());
    for (rid, signal) in signals {
        let (cal_offset, cal_scale) = calibrations.get(&rid).copied().unwrap_or((0.0, 1.0));
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
    let target_uuids: HashSet<escapepod::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod::Uuid::parse_str(s).ok())
        .collect();

    let pod5_path_owned = pod5_path.to_string();

    // Release GIL during I/O-heavy POD5 reading so other Python threads
    // (e.g. main thread doing BAM metadata extraction) can proceed.
    let result: Result<HashMap<String, Vec<i16>>, String> = py.detach(move || {
        let reader = escapepod::Reader::open(&pod5_path_owned)
            .map_err(|e| format!("Failed to open POD5 {pod5_path_owned}: {e}"))?;

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
