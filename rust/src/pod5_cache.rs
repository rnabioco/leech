//! Process-global cache of open POD5 readers.
//!
//! # Why this exists
//!
//! Every entry point that touches a POD5 does the same three things: open the
//! file, resolve a batch of read IDs to `(batch, row)` locations, then bulk-read
//! those rows. Only the third is proportional to the batch. The second is
//! proportional to the *file* the first time, because it has to build a read-id
//! index — and [`Reader`] caches that index in a `OnceLock` on itself, so a
//! reader opened per batch throws it away and pays for it again every time.
//!
//! On a 145 GB POD5 on a network filesystem that was minutes of uninterruptible
//! sleep in `folio_wait_bit_common` per batch, at ~0.6% of one core — the
//! ~10-80x gap against the Python multiprocessing fallback in issue #176. The
//! Python path never had the problem because `leech.io.pod5_reader` caches an
//! entered `DatasetReader` per process.
//!
//! So: open each POD5 once per process and hand out shared references. The
//! index is then built once, and every later lookup is a binary search.
//!
//! # What changed upstream
//!
//! escapepod v0.14.0 fixed the other half of #176 (escapepod-rs#251): before
//! it, `reads_by_ids` chose the indexed path only when a `.p5s` sidecar existed
//! on disk, and otherwise ran a full 22-column scan of the reads table that
//! *never built the index it declined* — so even a cached reader re-scanned on
//! every call. Since v0.14.0 the scan variants are gone and all three lookup
//! entry points route through `read_index()` unconditionally.
//!
//! That makes the warm-up below belt-and-braces rather than load-bearing, and
//! it is kept deliberately: building the index at insert time, before the
//! reader is visible to any other thread, keeps N worker threads from piling up
//! inside one `OnceLock::get_or_init` on their first batch. Caching the reader
//! itself remains essential either way.
//!
//! The index is built **in memory**, from a scan projected to the read_id
//! column. It neither reads nor writes a `.p5s` sidecar.

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

    // Build the read-id index once, here, while nothing else can see this
    // reader — so the first batch on each of N worker threads does not queue
    // inside `OnceLock::get_or_init`. See the module docs.
    if let Err(e) = reader.read_index() {
        eprintln!(
            "leech_core: could not build the read-id index for {path} ({e}); \
             read lookups will scan the reads table and be much slower"
        );
    }

    let mut guard = cache.lock().unwrap();
    Ok(Arc::clone(guard.entry(path.to_string()).or_insert(reader)))
}
