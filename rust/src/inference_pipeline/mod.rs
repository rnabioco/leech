//! Monolithic per-read feature extraction for the inference hot path.
//!
//! Replaces ~10 Python steps per read with a single Rust call:
//! POD5 signal -> normalize -> reference anchoring -> signal refinement ->
//! dwell/signal features -> chunk extraction -> sequence encoding.
//!
//! Supports all production config options: reference anchoring, signal map
//! refinement, signal_kmer encoding, and multi-channel signal (kmer residual).

mod features;
mod inference;
mod numeric;
mod processing;
mod refinement;
mod signal_mapping;
mod training;
mod types;

// Public API (unchanged from original inference_pipeline.rs)
pub use inference::_test_process_read;
pub use inference::_test_ref_to_signal;
pub use inference::extract_chunks_from_preloaded;
pub use inference::extract_inference_chunks;
pub use training::extract_training_chunks;
