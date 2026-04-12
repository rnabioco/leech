//! Shared types for the inference pipeline.

use std::collections::HashMap;

/// Result of signal processing for a single read. Shared between inference
/// and training chunk extraction.
pub(super) struct ProcessedRead {
    pub(super) norm_signal: Vec<f32>,
    pub(super) seq_to_sig: Vec<i64>,
    pub(super) use_sequence: String,
    pub(super) dwells: Vec<f32>,
    pub(super) features_data: Option<Vec<Vec<f32>>>,
    pub(super) num_features: usize,
    pub(super) sig_residual: Option<Vec<f32>>,
}

/// Per-chunk result for inference (pre-encoded arrays).
pub(super) struct ChunkResult {
    pub(super) signal: Vec<f32>,
    pub(super) seq_enc: Vec<f32>,
    pub(super) seq_rows: usize,
    pub(super) seq_cols: usize,
    pub(super) features: Option<Vec<f32>>,
    pub(super) num_features: usize,
    pub(super) dwell_width: usize,
    pub(super) read_id: String,
    pub(super) base_idx: i64,
}

/// Per-chunk result for training (raw arrays, no pre-encoding).
#[allow(dead_code)]
pub(super) struct TrainingChunkResult {
    pub(super) signal: Vec<f32>,
    pub(super) sequence: String,
    pub(super) dwell: Vec<f32>,
    pub(super) features: Vec<f32>,
    pub(super) num_features: usize,
    pub(super) kmer_len: usize,
    pub(super) read_id: String,
    pub(super) base_idx: i64,
    pub(super) focus_signal_pos: i64,
    /// Chunk-local seq_to_sig_map (for signal_kmer encoding at train time)
    pub(super) seq_to_sig_map: Vec<i64>,
    /// Extended sequence for signal_kmer (empty if not needed)
    pub(super) sequence_with_kmer_context: String,
    /// Signal residual chunk (empty if single channel)
    pub(super) signal_residual: Vec<f32>,
}

/// Shared config for all reads in a batch (avoids per-read cloning).
pub(super) struct PipelineConfig {
    pub(super) reverse_signal: bool,
    pub(super) use_reference: bool,
    pub(super) use_signal_kmer: bool,
    pub(super) skmer_ctx: (usize, usize),
    pub(super) signal_context_left: i64,
    pub(super) signal_context_right: i64,
    pub(super) kmer_ctx: i64,
    pub(super) kmer_win: usize,
    pub(super) signal_len: usize,
    pub(super) compute_features: bool,
    pub(super) feat_start: i64,
    pub(super) feat_end: i64,
    pub(super) dwell_width: usize,
    pub(super) refine_signal_map: bool,
    pub(super) kmer_table: Option<HashMap<String, f64>>,
    pub(super) kmer_len: usize,
    pub(super) kmer_center_idx: i32,
    pub(super) refine_half_bandwidth: i32,
    pub(super) refine_scale_iters: i32,
    pub(super) signal_in_channels: usize,
    /// Focus position within the base: "center" (default), "start", "end"
    pub(super) base_justify: BaseJustify,
}

#[derive(Clone, Copy)]
pub(super) enum BaseJustify {
    Center,
    Start,
    End,
}

impl BaseJustify {
    pub(super) fn from_str(s: &str) -> Self {
        match s {
            "start" => Self::Start,
            "end" => Self::End,
            _ => Self::Center,
        }
    }

    /// Compute focus signal position for a base given its signal boundaries.
    pub(super) fn focus_pos(self, sig_start: i64, sig_end: i64) -> i64 {
        match self {
            Self::Start => sig_start,
            Self::End => sig_end,
            Self::Center => (sig_start + sig_end) / 2,
        }
    }
}
