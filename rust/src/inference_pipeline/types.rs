//! Shared types for the inference pipeline glue: the leech-specific output
//! structs, and the one place a [`chunk::ProcessConfig`] + [`chunk::ChunkSpec`]
//! is built from the parameters every pyo3 entry point in this module takes.

use escapepod_signal::chunk::{self, FeatureChannel, SignalChannel};
use escapepod_signal::mapping::CigarOp;
use escapepod_signal::seq_encoding::KmerContext;

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use crate::kmer_levels::KmerLevels;

use super::signal_mapping::typed_cigar;

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
pub(super) struct TrainingChunkResult {
    pub(super) signal: Vec<f32>,
    pub(super) sequence: String,
    pub(super) dwell: Vec<f32>,
    pub(super) features: Vec<f32>,
    pub(super) num_features: usize,
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

/// Everything the three pyo3 entry points (`extract_inference_chunks`,
/// `extract_chunks_from_preloaded`, `extract_training_chunks`) need, built
/// once per batch in [`build_config`] and shared by every read in it.
///
/// `process` and `spec` are `escapepod_signal::chunk`'s own types -- this
/// struct is the leech-specific glue around them: which anchor to build per
/// read (`use_reference` -- the two variants take different borrowed data per
/// read, so the choice can't live in `process` itself), the signal-kmer
/// context leech's training format always records regardless of
/// `spec.seq_encoding` (`skmer_ctx`), and the constant leech writes for
/// `TrainingChunkResult.focus_signal_pos` instead of the chunk's actual
/// (per-read-varying) focus sample -- see seam 2 of rnabioco/leech#258.
#[derive(Debug)]
pub(super) struct PipelineConfig<'a> {
    pub(super) process: chunk::ProcessConfig<'a>,
    pub(super) spec: chunk::ChunkSpec,
    /// `spec` with `seq_encoding` forced to `None`, for training's `cut_chunk`
    /// call. Training discards `Chunk::sequence`/`sequence_rows`/
    /// `sequence_cols` entirely -- it builds its own `sequence` (`kmer_seq`)
    /// and gets `seq_to_sig_map`/`sequence_with_kmer_context` from a second,
    /// separate, non-dropping call to `chunk::signal_kmer_inputs`. Cutting
    /// with the real `spec.seq_encoding` (`SignalKmer` under the default
    /// `--seq-encoding signal_kmer`) would let `cut_chunk` drop the whole
    /// chunk whenever *that* internal `signal_kmer_inputs` call fails, for a
    /// reason training never acts on -- the pre-port training path never
    /// dropped a chunk for anything but an out-of-bounds `base_idx`. See
    /// CLAUDE.md's "Backend parity: when a chunk is dropped" (issue #185).
    pub(super) training_spec: chunk::ChunkSpec,
    pub(super) use_reference: bool,
    pub(super) skmer_ctx: (usize, usize),
    pub(super) signal_context_left: i64,
    /// The base-window half-width leech's training format always records
    /// as `sequence` (a `base_onehot`-shaped `N`-padded string), independent
    /// of `spec.seq_encoding` and of `spec.feature_offsets` -- neither
    /// carries this value on the `SignalKmer` branch, and `feature_offsets`
    /// is a caller-overridable window that may differ from `kmer_context`
    /// (issue #189's class of bug: two representations of a window width
    /// that can silently diverge).
    pub(super) kmer_context: i64,
}

/// Build the shared config from the parameters common to all three pyo3
/// entry points. The single place `PipelineConfig` is constructed --
/// `extract_inference_chunks`, `extract_chunks_from_preloaded` and
/// `extract_training_chunks` all call this rather than each building their
/// own (rnabioco/leech#258).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_config<'a>(
    reverse_signal: bool,
    anchor: &str,
    seq_encoding: &str,
    signal_kmer_context: Option<(usize, usize)>,
    signal_context_left: i64,
    signal_context_right: i64,
    kmer_context: i64,
    signal_len: usize,
    compute_features: bool,
    feature_start: Option<i64>,
    feature_end: Option<i64>,
    refine_signal_map: bool,
    kmer_table: Option<&'a KmerLevels>,
    kmer_len: usize,
    kmer_center_idx: i32,
    refine_half_bandwidth: i32,
    refine_scale_iters: i32,
    signal_in_channels: usize,
    base_justify: &str,
) -> PyResult<PipelineConfig<'a>> {
    let use_reference = match anchor {
        "reference" => true,
        "basecall" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "unrecognised anchor {other:?} (expected \"reference\" or \"basecall\")"
            )));
        }
    };
    let use_signal_kmer = match seq_encoding {
        "signal_kmer" => true,
        "base_onehot" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "unrecognised seq_encoding {other:?} (expected \"signal_kmer\" or \"base_onehot\")"
            )));
        }
    };
    let base_justify = chunk::BaseJustify::from_name(base_justify).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unrecognised base_justify {base_justify:?} (expected \"start\", \"center\" or \"end\")"
        ))
    })?;

    let skmer_ctx = signal_kmer_context.unwrap_or((4, 4));
    let kmer_table = kmer_table.map(|kt| &kt.table);
    // Levels are needed whenever refinement is configured with a table, and
    // extracted either way -- the k-mer residual channels need them whether
    // or not the boundaries actually moved. This is also what gates whether
    // the *_residual channels are emitted at all (below): requesting a
    // residual channel without a level table produces a narrower tensor than
    // asked for, matching the pre-port behaviour rather than a zeroed one, so
    // a caller who passes `signal_in_channels=2` without a table finds out
    // from the shape rather than getting silent zeros.
    let levels_available = refine_signal_map && kmer_table.is_some();

    let levels = levels_available.then(|| chunk::LevelModel {
        table: kmer_table.expect("levels_available implies kmer_table.is_some()"),
        kmer_len,
        // `center_idx < 0` is leech's "use the default" sentinel; escapepod's
        // own default is the same `kmer_len / 2` (resquiggle::extract_levels),
        // so resolving it here rather than passing `Option` through keeps
        // that one definition upstream instead of re-stating it.
        center_idx: usize::try_from(kmer_center_idx).unwrap_or(kmer_len / 2),
    });
    let refine = (levels_available && refine_scale_iters >= 0).then(|| chunk::RefineParams {
        half_bandwidth: refine_half_bandwidth.max(0) as usize,
        scale_iters: refine_scale_iters.max(0) as usize,
        seed: Some(chunk::DEFAULT_REFINE_SEED),
    });

    let process = chunk::ProcessConfig {
        reverse_signal,
        normalization: chunk::SignalNorm::MedianMad,
        levels,
        refine,
    };

    let feat_start = feature_start.unwrap_or(-kmer_context);
    let feat_end = feature_end.unwrap_or(kmer_context);

    let mut feature_channels = Vec::new();
    if compute_features {
        // Order matches the pre-port `features_data` stack exactly: dwell
        // family first, then level-span stats, then (when a table is in
        // play) the k-mer residual family. Parity tests compare this array
        // positionally, so the order is load-bearing.
        feature_channels.extend([
            FeatureChannel::Dwell,
            FeatureChannel::DwellLog,
            FeatureChannel::DwellMean,
            FeatureChannel::DwellStd,
            FeatureChannel::DwellRatio,
            FeatureChannel::LevelMean,
            FeatureChannel::LevelMedian,
            FeatureChannel::LevelStd,
            FeatureChannel::LevelRange,
        ]);
        if levels_available {
            feature_channels.extend([
                FeatureChannel::KmerExpected,
                FeatureChannel::KmerResidual,
                FeatureChannel::KmerResidualAbs,
            ]);
        }
    }

    let mut signal_channels = vec![SignalChannel::Current];
    if signal_in_channels > 1 && levels_available {
        signal_channels.push(SignalChannel::KmerResidual);
    }

    let seq_encoding_spec = if use_signal_kmer {
        chunk::SeqEncoding::SignalKmer {
            ctx: KmerContext {
                before: skmer_ctx.0,
                after: skmer_ctx.1,
            },
        }
    } else {
        chunk::SeqEncoding::BaseOneHot {
            context: kmer_context.max(0) as usize,
        }
    };

    let spec = chunk::ChunkSpec {
        signal_context: (signal_context_left, signal_context_right),
        signal_len,
        base_justify,
        signal_channels,
        seq_encoding: seq_encoding_spec,
        feature_offsets: (feat_start, feat_end),
        feature_channels,
        dwell_window: chunk::DEFAULT_DWELL_WINDOW,
    };

    let training_spec = chunk::ChunkSpec {
        seq_encoding: chunk::SeqEncoding::None,
        ..spec.clone()
    };

    Ok(PipelineConfig {
        process,
        spec,
        training_spec,
        use_reference,
        skmer_ctx,
        signal_context_left,
        kmer_context,
    })
}

/// Resolve `(L, R)` base offsets around `base_idx` to a sample interval via
/// the read's own base-to-signal map (`--signal-context-bases`, issue #278).
///
/// `chunking.resolve_signal_context_bases` (Python) is the canonical
/// definition; this mirrors it exactly and the two are held equal by
/// `tests/test_backend_parity.py`'s bases-context matrix row. `sample_start`
/// is the first sample of base `base_idx - L` and `sample_end` is the first
/// sample past base `base_idx + R`, both clamped to the read's own mapped
/// span (base 0 through `n_bases - 1`) -- clamping, not dropping, is the
/// guard `--signal-context-bases` promises at either edge of a read (the one
/// allowed drop rule, CLAUDE.md, is unaffected: only `base_idx` itself being
/// unmapped drops a chunk, checked by the caller before this runs).
///
/// `left_bases`/`right_bases` are meant to be `>= 0` -- the CLI and
/// `handle_prepare` both refuse a negative value before any read is touched
/// -- but `lo_base`/`hi_base` are independently clamped to
/// `[0, n_bases - 1]` regardless of sign, matching
/// `chunking.resolve_signal_context_bases` (Python) exactly. Without this a
/// large-magnitude negative `left_bases` pushes `lo_base` past `n_bases`,
/// and indexing `seq_to_sig` with it is a hard **panic** here (unlike
/// Python's `IndexError`) -- one that, per issue #265's zero-tolerance
/// policy, aborts the whole in-flight batch and discards every chunk
/// `ChunkSpool` has already spooled to disk.
///
/// Shared by `training::extract_training_chunks_from_read` and
/// `inference::process_one_read` -- both backends resolve a base-defined
/// window through this single definition rather than each carrying its own
/// copy (issue #341: the two used to diverge, with the inference path always
/// forced to Python instead of implementing this at all).
pub(super) fn resolve_signal_context_bases(
    seq_to_sig: &[i64],
    base_idx: i64,
    left_bases: i64,
    right_bases: i64,
    n_bases: usize,
) -> (i64, i64) {
    let last_base = (n_bases - 1) as i64;
    let lo_base = (base_idx - left_bases).clamp(0, last_base) as usize;
    let hi_base = (base_idx + right_bases).clamp(0, last_base) as usize;
    (seq_to_sig[lo_base], seq_to_sig[hi_base + 1])
}

/// Build the `Anchor` for one read: reference-anchored when the config asks
/// for it and both a CIGAR and a reference sequence are available, basecall
/// coordinates otherwise (matching the pre-port fallback for a read the caller
/// didn't supply alignment info for). Shared by `inference::process_one_read`
/// and `training::extract_training_chunks_from_read`. `cigar_buf` is threaded
/// in rather than returned so its lifetime outlives the `Anchor` borrowing it.
pub(super) fn build_anchor<'a>(
    cfg: &PipelineConfig<'_>,
    sequence: &'a str,
    cigar_ops: Option<&[(u32, u32)]>,
    ref_seq: Option<&'a str>,
    cigar_buf: &'a mut Vec<CigarOp>,
) -> chunk::Anchor<'a> {
    if cfg.use_reference
        && let (Some(cig), Some(rseq)) = (cigar_ops, ref_seq)
    {
        *cigar_buf = typed_cigar(cig);
        chunk::Anchor::Reference {
            sequence: rseq.as_bytes(),
            cigar: cigar_buf.as_slice(),
        }
    } else {
        chunk::Anchor::Query {
            sequence: sequence.as_bytes(),
        }
    }
}
