//! Process-global cache of open POD5 readers.
//!
//! # Why this exists
//!
//! Every entry point that touches a POD5 does the same three things: open the
//! file, resolve a batch of read IDs to `(batch, row)` locations, then bulk-read
//! those rows. Only the third step is proportional to the batch. The second is
//! proportional to the *file* unless a read-id index is present, because
//! [`Reader::reads_by_ids`] falls back to
//! `reads_by_ids_scan` — a single-threaded walk of every batch in the reads
//! table, deserializing all 22 columns, that can only stop early once it has
//! found every target. Reads arrive here in BAM order, which bears no relation
//! to POD5 storage order, so "found every target" is in practice "reached the
//! end of the file".
//!
//! [`Reader`] caches that index in a `OnceLock` on itself, so opening a fresh
//! reader per batch throws it away and pays the full scan again. On a 145 GB
//! POD5 on a network filesystem that is minutes of uninterruptible sleep in
//! `folio_wait_bit_common` per batch, at ~0.6% of one core — which is exactly
//! how the Rust path came to be ~10-80x slower than the Python multiprocessing
//! fallback (issue #176). The Python path never had the problem because
//! `leech.io.pod5_reader` caches an entered `DatasetReader` per process.
//!
//! So: open each POD5 once per process, warm its read-id index once, and hand
//! out shared references. Lookups after that are a binary search per read.
//!
//! The warm-up builds the index **in memory**, from a scan projected to the
//! read_id column alone. It does not read, write, or require a `.p5s` sidecar.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use escapepod_signal::Reader;

/// Cache of readers keyed by the POD5 path string as it was passed in.
///
/// Entries live until the process exits. A `Reader` holds an mmap and the
/// read-id index, so the resident cost is the index (~24 bytes per read in the
/// file), not the file — a few tens of MB for a multi-million-read POD5.
static READERS: OnceLock<Mutex<HashMap<String, Arc<Reader>>>> = OnceLock::new();

/// Get the shared [`Reader`] for `path`, opening and indexing it on first use.
///
/// Call this instead of [`Reader::open`] everywhere. Repeat calls for the same
/// path are a hash lookup; the expensive open-and-index happens once per
/// process.
///
/// Errors from the index warm-up are **not** propagated: an un-indexable POD5
/// is still readable, just slowly, and failing the whole batch over it would be
/// worse than the slowdown. The open itself is fatal, as before.
pub fn cached_reader(path: &str) -> Result<Arc<Reader>, String> {
    let cache = READERS.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(reader) = cache.lock().unwrap().get(path) {
        return Ok(Arc::clone(reader));
    }

    // Opened outside the lock so a slow open on one path does not block lookups
    // for another. Two threads racing on the same path both open; `or_insert`
    // below keeps whichever landed first and drops the other. That costs one
    // redundant open in a rare race, and never a deadlock or a torn cache.
    let reader =
        Arc::new(Reader::open(path).map_err(|e| format!("Failed to open POD5 {path}: {e}"))?);

    // Warm the read-id index once. Without this, `reads_by_ids` takes the
    // full-scan path on *every* call, because a scan never populates the index
    // it skipped. See the module docs.
    if let Err(e) = reader.read_index() {
        eprintln!(
            "leech_core: could not build the read-id index for {path} ({e}); \
             read lookups will scan the reads table and be much slower"
        );
    }

    let mut guard = cache.lock().unwrap();
    Ok(Arc::clone(guard.entry(path.to_string()).or_insert(reader)))
}
