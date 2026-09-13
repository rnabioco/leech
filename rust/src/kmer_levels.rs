//! Owns the k-mer level table used for signal-map refinement, built once per
//! prepare/predict run rather than re-marshalled from a Python `dict` on every
//! batch call (issue #259).
//!
//! Every `extract_training_chunks` / `extract_inference_chunks` /
//! `extract_chunks_from_preloaded` call used to take
//! `kmer_table: Option<HashMap<String, f64>>` and pay a fresh
//! `dict -> HashMap<String, f64>` conversion under the GIL, before
//! `py.detach`, for the *same* 262,144-entry 9-mer table on every batch --
//! serializing the `ThreadPoolExecutor` workers `_iter_rust_batches` exists to
//! overlap. Measured: 0.0 ms without the table attached, 51.8 ms min / 71.7 ms
//! median with it (roughly 0.5 ms of GIL time per read at `chunk_size=100`).
//!
//! `KmerLevels` does that conversion exactly once, in Python, at the start of
//! a run (`leech._rust_accel.make_kmer_levels`), and every batch call
//! afterwards borrows the table by reference -- see
//! `inference_pipeline::types::PipelineConfig::kmer_table`.
//!
//! A `HashMap<String, f64>` was kept, rather than a dense `Vec<f32>` indexed by
//! a packed k-mer integer code, because the table is not guaranteed dense over
//! `4**k`: callers can hand in a table with missing entries or a k-mer length
//! other than 9, and a hash lookup keyed on the raw k-mer string is exactly
//! what `escapepod_signal::resquiggle::extract_levels` already expects. A
//! packed dense encoding would be a second representation to keep in sync with
//! that upstream contract for a table this codebase already treats as
//! opaque -- not worth it when the cost this fixes is one-time-per-run either
//! way.

use std::collections::HashMap;

use pyo3::prelude::*;

/// Opaque handle owning a k-mer -> expected-level table.
///
/// Built once per run in Python (`KmerLevels(kmer_to_level)`, or
/// `leech._rust_accel.make_kmer_levels`) and passed by reference into every
/// `extract_*` entry point. Follows the same build-once-borrow-many pattern as
/// [`crate::pod5_io::PreloadedSignals`].
///
/// `frozen` (no interior mutability -- there is no setter, and none is
/// planned) is what lets entry points take a plain `Option<&KmerLevels>`
/// parameter: pyo3 only implements `FromPyObject` for a bare `&T` reference
/// when `T: PyClass<Frozen = True> + Sync`, which composes with `Option<T>`'s
/// own blanket impl for the `kmer_table = None` default. Without `frozen`,
/// the reference would have to go through a `PyRef` runtime-borrow-tracked
/// guard instead -- sound (`PyCell` allows any number of concurrent shared
/// borrows, which is all every caller here ever takes), but pure ceremony for
/// data that is never mutated after construction.
#[pyclass(frozen)]
pub struct KmerLevels {
    pub(crate) table: HashMap<String, f64>,
}

#[pymethods]
impl KmerLevels {
    /// Convert a Python `dict[str, float]` to `HashMap<String, f64>` exactly
    /// once. This is the conversion issue #259 moved out of the per-batch hot
    /// path.
    #[new]
    fn new(table: HashMap<String, f64>) -> Self {
        Self { table }
    }

    fn __len__(&self) -> usize {
        self.table.len()
    }
}
