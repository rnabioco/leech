//! Batch POD5 reading via escapepod-rs.
//!
//! Replaces Python pod5 library's `DatasetReader.reads()` with escapepod's
//! memory-mapped reader and bulk signal extraction for ~25-43x faster I/O.

use std::collections::{HashMap, HashSet};

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

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

    // Projected read (cols 0,1,16,17), UUID-native filter, early termination
    let matched = reader.read_id_signal_rows_cal_filtered(&target_uuids).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read POD5 index: {e}"))
    })?;

    let mut to_extract: Vec<(String, Vec<u64>)> = Vec::with_capacity(matched.len());
    let mut calibrations: HashMap<String, (f32, f32)> = HashMap::with_capacity(matched.len());

    for (uuid, rows, cal_offset, cal_scale) in matched {
        let rid = uuid.to_string();
        calibrations.insert(rid.clone(), (cal_offset, cal_scale));
        to_extract.push((rid, rows));
    }

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
