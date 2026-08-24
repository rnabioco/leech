//! Batch signal reads against escapepod's process-global reader cache.
//!
//! This module used to carry its own `static OnceLock<Mutex<HashMap<String,
//! Arc<Reader>>>>`, because `Reader` caches its read-id index in a `OnceLock`
//! on the *instance*: a reader opened per batch throws the index away and
//! rebuilds it every time. On a 145 GB POD5 on a network filesystem that was
//! minutes of uninterruptible sleep per batch — the ~10-80x regression in
//! issue #176.
//!
//! escapepod 0.15.0 owns that cache now (escapepod-rs#258), in the crate where
//! the index lives, with the ordering and failure semantics that make it worth
//! having: the file is opened outside the lock so a slow open never blocks a
//! lookup on another path, publication goes through `entry` so a race cannot
//! leave two live readers for one file, and the index is warmed before the
//! entry is published so N workers hitting their first batch find it built
//! rather than piling into one lazy init.
//!
//! **Always go through [`read_signals_by_ids`] or [`escapepod_signal::cached_reader`];
//! never call `Reader::open` directly.**

use std::collections::{HashMap, HashSet};

use escapepod_signal::cached_reader;

/// One read's signal plus the calibration needed to convert it to pA.
pub(crate) struct BulkSignal {
    pub(crate) read_id: String,
    pub(crate) signal: Vec<i16>,
    pub(crate) calibration_offset: f32,
    pub(crate) calibration_scale: f32,
}

/// Resolve `read_ids` against the shared indexed reader and bulk-extract their
/// signals.
///
/// The three-step dance below — parse the ids as UUIDs, `reads_by_ids` to
/// locate the rows, `get_signal_bulk` to decompress them in one batch-grouped
/// pass — was written out four times across this crate, in `pod5_io` twice and
/// in the training and inference entry points. Four copies of the same lookup
/// is four places to forget [`cached_reader`], which is the one thing here that
/// must not be got wrong: `Reader::open` per call throws away the read-id index
/// and re-pays for it, which was the ~10-80x regression in issue #176.
///
/// **Callers must invoke this with the GIL released** (`py.detach`). It is pure
/// Rust I/O over no Python objects, and holding the GIL across it serializes
/// every caller thread against the slowest network read in the batch, making
/// batch-level concurrency impossible.
///
/// Reads that are not present in the file are simply absent from the result;
/// an unparseable read id is skipped rather than failing the batch, since a
/// BAM can carry query names that are not POD5 UUIDs at all.
pub(crate) fn read_signals_by_ids(
    pod5_path: &str,
    read_ids: &[String],
) -> Result<Vec<BulkSignal>, String> {
    let target_uuids: HashSet<escapepod_signal::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod_signal::Uuid::parse_str(s).ok())
        .collect();

    let reader =
        cached_reader(pod5_path).map_err(|e| format!("Failed to open POD5 {pod5_path}: {e}"))?;

    let matched_reads = reader
        .reads_by_ids(&target_uuids)
        .map_err(|e| format!("Failed to look up reads: {e}"))?;

    let mut to_extract: Vec<(String, Vec<u64>)> = Vec::with_capacity(matched_reads.len());
    let mut calibrations: HashMap<String, (f32, f32)> = HashMap::with_capacity(matched_reads.len());
    for read in matched_reads {
        let rid = read.read_id.to_string();
        calibrations.insert(
            rid.clone(),
            (read.calibration_offset, read.calibration_scale),
        );
        to_extract.push((rid, read.signal_rows));
    }

    let bulk_signals = reader
        .get_signal_bulk(&to_extract)
        .map_err(|e| format!("Signal extraction failed: {e}"))?;

    Ok(bulk_signals
        .into_iter()
        .map(|(read_id, signal)| {
            let (calibration_offset, calibration_scale) =
                calibrations.get(&read_id).copied().unwrap_or((0.0, 1.0));
            BulkSignal {
                read_id,
                signal,
                calibration_offset,
                calibration_scale,
            }
        })
        .collect())
}

/// [`read_signals_by_ids`] as a `read_id -> signal` map, for the callers that
/// do not need calibration.
pub(crate) fn read_signal_map_by_ids(
    pod5_path: &str,
    read_ids: &[String],
) -> Result<HashMap<String, Vec<i16>>, String> {
    Ok(read_signals_by_ids(pod5_path, read_ids)?
        .into_iter()
        .map(|b| (b.read_id, b.signal))
        .collect())
}
