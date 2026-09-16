//! Batch signal reads against escapepod's process-global dataset cache.
//!
//! This module used to carry its own `static OnceLock<Mutex<HashMap<String,
//! Arc<Reader>>>>`, because `Reader` caches its read-id index in a `OnceLock`
//! on the *instance*: a reader opened per batch throws the index away and
//! rebuilds it every time. On a 145 GB POD5 on a network filesystem that was
//! minutes of uninterruptible sleep per batch — the ~10-80x regression in
//! issue #176.
//!
//! escapepod 0.15.0 owns the per-file half of that cache (escapepod-rs#258):
//! `ReaderCache`/`cached_reader`, keyed one `Arc<Reader>` per file, opened
//! outside the lock with its index warmed before publication.
//! escapepod-rs#384 added `Dataset`/`cached_dataset` one level up — a single
//! file, a directory (a MinKNOW run), or a mix of both, presented as one
//! random-access-by-read-id source. Every file a `Dataset` touches is opened
//! through that *same* `ReaderCache`, never a second, dataset-private one, so
//! a single-file `--pod5` still shares its `Arc<Reader>` with any other path
//! that names the same file directly.
//!
//! **Always go through [`read_signals_by_ids`] or [`escapepod_signal::cached_dataset`];
//! never call `Reader::open` or `Dataset::open` directly.** There is
//! deliberately no file/directory branch anywhere in this module — a
//! single-file `Dataset` costs nothing over a bare `Reader`, so one code path
//! serves both shapes.

use std::collections::{HashMap, HashSet};

use escapepod_signal::cached_dataset;

/// One read's signal plus the calibration needed to convert it to pA.
pub(crate) struct BulkSignal {
    pub(crate) read_id: String,
    pub(crate) signal: Vec<i16>,
    pub(crate) calibration_offset: f32,
    pub(crate) calibration_scale: f32,
}

/// Resolve `read_ids` against the shared indexed dataset and bulk-extract
/// their signals.
///
/// `pod5_path` is a single POD5 file, a directory of them, or (via
/// [`escapepod_signal::cached_dataset`]'s single-root convenience form) a
/// mix — the same call handles all three, since every file a [`Dataset`]
/// touches is opened through the *same* per-file [`escapepod_signal::cached_reader`]
/// this module used before #384, which is the thing that must not be gotten
/// wrong here: reopening a file per call throws away its read-id index and
/// re-pays for it, the ~10-80x regression in issue #176.
///
/// **Callers must invoke this with the GIL released** (`py.detach`). It is pure
/// Rust I/O over no Python objects, and holding the GIL across it serializes
/// every caller thread against the slowest network read in the batch, making
/// batch-level concurrency impossible.
///
/// Reads that are not present in the dataset are simply absent from the
/// result; an unparseable read id is skipped rather than failing the batch,
/// since a BAM can carry query names that are not POD5 UUIDs at all.
///
/// [`Dataset`]: escapepod_signal::Dataset
pub(crate) fn read_signals_by_ids(
    pod5_path: &str,
    read_ids: &[String],
) -> Result<Vec<BulkSignal>, String> {
    let target_uuids: HashSet<escapepod_signal::Uuid> = read_ids
        .iter()
        .filter_map(|s| escapepod_signal::Uuid::parse_str(s).ok())
        .collect();

    let dataset = cached_dataset(pod5_path)
        .map_err(|e| format!("Failed to open POD5 source {pod5_path}: {e}"))?;

    let matched_reads = dataset
        .reads_by_ids(&target_uuids)
        .map_err(|e| format!("Failed to look up reads: {e}"))?;

    let mut calibrations: HashMap<String, (f32, f32)> = HashMap::with_capacity(matched_reads.len());
    for read in &matched_reads {
        calibrations.insert(
            read.read_id.to_string(),
            (read.calibration_offset, read.calibration_scale),
        );
    }

    let bulk_signals = dataset
        .decode_bulk(&matched_reads, usize::MAX)
        .map_err(|e| format!("Signal extraction failed: {e}"))?;

    Ok(bulk_signals
        .into_iter()
        .map(|(read_id, signal)| {
            let rid = read_id.to_string();
            let (calibration_offset, calibration_scale) =
                calibrations.get(&rid).copied().unwrap_or((0.0, 1.0));
            BulkSignal {
                read_id: rid,
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

#[cfg(test)]
mod tests {
    //! `Dataset::open`'s directory scan and `decode_bulk`'s owning-file
    //! bucketing are escapepod-rs#384's contract, already covered there
    //! (`crates/escapepod-pod5/tests/test_dataset_cache.rs`); what belongs
    //! here is proof that *this crate's* one entry point, `read_signals_by_ids`,
    //! is indifferent to whether `pod5_path` names a file or a directory —
    //! the whole point of collapsing the file/directory branch (issue #339).
    use std::collections::HashMap;
    use std::path::Path;

    use escapepod_signal::{EndReason, ReadData, RunInfoData, Uuid, Writer, WriterOptions};
    use tempfile::TempDir;

    use super::{read_signal_map_by_ids, read_signals_by_ids};

    fn make_run_info(acq_id: &str) -> RunInfoData {
        RunInfoData {
            acquisition_id: acq_id.to_string(),
            acquisition_start_time: 1_609_459_200_000,
            adc_max: 2047,
            adc_min: -2048,
            context_tags: HashMap::new(),
            experiment_name: "leech_test".to_string(),
            flow_cell_id: "FAK_LEECH".to_string(),
            flow_cell_product_code: "FLO-MIN106".to_string(),
            protocol_name: "leech_test_protocol".to_string(),
            protocol_run_id: "protocol_leech_test".to_string(),
            protocol_start_time: 1_609_459_200_000,
            sample_id: "leech_test_sample".to_string(),
            sample_rate: 4_000,
            sequencing_kit: "SQK-LSK109".to_string(),
            sequencer_position: "MN00000".to_string(),
            sequencer_position_type: "minion".to_string(),
            software: "leech-pod5-cache-test".to_string(),
            system_name: "leech_test_system".to_string(),
            system_type: "minion".to_string(),
            tracking_id: HashMap::new(),
        }
    }

    fn make_read(run_info_idx: u32, read_number: u32, num_samples: u64) -> ReadData {
        ReadData {
            read_id: Uuid::new_v4(),
            read_number,
            start_sample: (read_number as u64 - 1) * num_samples,
            channel: 1,
            well: 1,
            pore_type: "not_set".into(),
            calibration_offset: 0.5,
            calibration_scale: 0.95,
            median_before: 200.0,
            end_reason: EndReason::SignalPositive,
            end_reason_forced: false,
            run_info_index: run_info_idx,
            num_minknow_events: 100,
            tracked_scaling_scale: 1.0,
            tracked_scaling_shift: 0.0,
            predicted_scaling_scale: 1.0,
            predicted_scaling_shift: 0.0,
            num_reads_since_mux_change: 0,
            time_since_mux_change: 0.0,
            num_samples,
            open_pore_level: 220.0,
            expected_open_pore_level: 0.0,
            selected_read_level: 0.0,
            signal_rows: Vec::new(),
        }
    }

    /// A small deterministic (not random) signal so the test doesn't need an
    /// RNG dependency: a xorshift-style mix, seeded per read so distinct reads
    /// don't collide.
    fn synth_signal(n: usize, seed: u64) -> Vec<i16> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as i32) % 2000 - 1000) as i16
            })
            .collect()
    }

    /// Write `n_reads` synthetic reads to a single POD5 file, returning their
    /// ids as the `String`s [`read_signals_by_ids`] expects.
    fn write_fixture(
        path: &Path,
        acq_id: &str,
        n_reads: usize,
        samples_per_read: usize,
    ) -> Vec<String> {
        let mut writer = Writer::create(path, WriterOptions::default()).expect("Writer::create");
        let run_idx = writer
            .add_run_info(make_run_info(acq_id))
            .expect("add_run_info");
        let mut ids = Vec::with_capacity(n_reads);
        for i in 0..n_reads {
            let read = make_read(run_idx, i as u32 + 1, samples_per_read as u64);
            ids.push(read.read_id.to_string());
            let signal = synth_signal(samples_per_read, 0xA110 + i as u64);
            writer.add_read(read, &signal).expect("add_read");
        }
        writer.finish().expect("writer.finish");
        ids
    }

    #[test]
    fn directory_input_matches_calling_once_per_file() {
        let tmp = TempDir::new().unwrap();
        let path_a = tmp.path().join("a.pod5");
        let path_b = tmp.path().join("b.pod5");
        let ids_a = write_fixture(&path_a, "acq_a", 3, 400);
        let ids_b = write_fixture(&path_b, "acq_b", 3, 400);

        let all_ids: Vec<String> = ids_a.iter().chain(ids_b.iter()).cloned().collect();
        let dir = tmp.path().to_str().unwrap();
        let via_dir = read_signals_by_ids(dir, &all_ids).expect("directory read");
        assert_eq!(
            via_dir.len(),
            all_ids.len(),
            "every read across both files must come back from one directory call"
        );

        let mut expected: HashMap<String, (Vec<i16>, f32, f32)> = HashMap::new();
        for (path, ids) in [(&path_a, &ids_a), (&path_b, &ids_b)] {
            for b in read_signals_by_ids(path.to_str().unwrap(), ids).expect("per-file read") {
                expected.insert(
                    b.read_id,
                    (b.signal, b.calibration_offset, b.calibration_scale),
                );
            }
        }

        for b in via_dir {
            let (signal, offset, scale) = expected
                .get(&b.read_id)
                .expect("directory result must be one of the per-file reads");
            assert_eq!(
                &b.signal, signal,
                "signal must match a direct per-file read"
            );
            assert_eq!(b.calibration_offset, *offset);
            assert_eq!(b.calibration_scale, *scale);
        }
    }

    #[test]
    fn single_file_input_is_unaffected_by_the_dataset_switch() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("only.pod5");
        let ids = write_fixture(&path, "acq_only", 4, 250);

        let map = read_signal_map_by_ids(path.to_str().unwrap(), &ids).expect("single-file read");
        assert_eq!(map.len(), ids.len());
        for id in &ids {
            assert!(map.contains_key(id));
        }
    }
}
