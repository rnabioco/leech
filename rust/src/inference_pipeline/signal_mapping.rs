//! Move table and CIGAR-based signal mapping.

use escapepod_signal::mapping::{CigarKind, CigarOp, ref_to_signal, seq_to_signal_from_moves};

/// Query->signal map from a basecaller move table.
///
/// Delegates to `escapepod_signal::mapping::seq_to_signal_from_moves`, which is
/// the `mv`/`ns`/`ts` tag convention with nothing leech-specific in it. Result
/// is in trimmed-signal coordinates, closed with `num_samples - trim_offset`.
pub(super) fn build_seq_to_sig_map(
    mv_array: &[u8],
    stride: u32,
    trim_offset: i64,
    num_samples: u64,
) -> Vec<i64> {
    seq_to_signal_from_moves(mv_array, stride, trim_offset, num_samples)
}

/// A BAM CIGAR op code (`pysam`'s `cigartuples` encoding) as a [`CigarKind`].
///
/// escapepod takes a typed `CigarKind` rather than the raw integer, which is
/// the right call -- but it means this table is the one place the numbering
/// still has to be written down. It is the SAM spec order, unchanged since the
/// format was defined: `MIDNSHP=X`. An unrecognised code maps to `Pad`, which
/// consumes neither query nor reference and so cannot shift a coordinate.
fn cigar_kind(op: u32) -> CigarKind {
    match op {
        0 => CigarKind::Match,
        1 => CigarKind::Insertion,
        2 => CigarKind::Deletion,
        3 => CigarKind::Skip,
        4 => CigarKind::SoftClip,
        5 => CigarKind::HardClip,
        7 => CigarKind::SequenceMatch,
        8 => CigarKind::SequenceMismatch,
        _ => CigarKind::Pad,
    }
}

/// Reference->signal map, by the Remora knot convention.
///
/// Delegates to `escapepod_signal::mapping::ref_to_signal`. leech used to carry
/// its own walk of the same convention -- trailing non-match ops stripped, 1:1
/// integer lookup inside match blocks, interpolation across indel gaps -- which
/// is short enough to retype and subtle enough to retype wrongly. The failure
/// mode is the reason it belongs upstream: a map built slightly differently
/// still refines, still produces per-base statistics and still scores, just
/// over a different set of samples than the caller thinks. Nothing errors.
pub(super) fn compute_ref_to_signal(query_to_sig: &[i64], cigar_ops: &[(u32, u32)]) -> Vec<i64> {
    let cigar: Vec<CigarOp> = cigar_ops
        .iter()
        .map(|&(op, len)| CigarOp::new(cigar_kind(op), len))
        .collect();
    ref_to_signal(query_to_sig, &cigar)
}

/// Chunk-local `seq_to_sig_map` and `N`-padded context sequence, the two
/// inputs `signal_kmer` encoding needs.
///
/// Mirrors the tail of `LeechRead.get_chunk`, which is the definition every
/// trained `signal_kmer` model has seen. Note what it keys off: the **signal**
/// window, located in the map with two binary searches, *not* the k-mer
/// window. Those select different numbers of bases — the signal window spans
/// however many bases fall inside `signal_context`, the k-mer window spans
/// exactly `2 * kmer_context + 1` — so deriving these from `kmer_start`/
/// `kmer_end` disagrees with Python on every chunk, which was issue #186.
///
/// Returns `(chunk_seq_to_sig, sequence_with_kmer_context)`, of length
/// `n + 1` and `n + kmer_before + kmer_after` for the `n` bases the window
/// covers — the shapes `encode_signal_kmer_inner` expects. Both are empty
/// when the read has no usable map.
pub(super) fn chunk_signal_kmer_inputs(
    seq_to_sig: &[i64],
    seq_bytes: &[u8],
    sig_start_pos: i64,
    sig_end_pos: i64,
    num_samples: usize,
    chunk_len: usize,
    skmer_ctx: (usize, usize),
) -> (Vec<i64>, Vec<u8>) {
    if seq_to_sig.len() < 2 {
        return (vec![], vec![]);
    }
    let num_bases_map = seq_to_sig.len() - 1;

    // Python clamps the window into the signal before locating bases, and it
    // is the clamped window that the searches use.
    let ss = sig_start_pos.max(0);
    let se = sig_end_pos.min(num_samples as i64);

    // searchsorted(map, ss, side="right") - 1: the base whose span contains ss.
    let seq_start = seq_to_sig.partition_point(|&v| v <= ss) as i64 - 1;
    // searchsorted(map, se, side="left"): the first boundary at or past se.
    let seq_end = seq_to_sig.partition_point(|&v| v < se) as i64;

    let seq_start = seq_start.max(0) as usize;
    let seq_end = (seq_end.clamp(0, seq_bytes.len() as i64) as usize).min(num_bases_map);
    if seq_start > seq_end {
        return (vec![], vec![]);
    }

    // Offsets are against the UNCLAMPED window start, so a chunk that
    // underflows the signal still reports positions relative to its own left
    // edge. (Python reaches the same value via `sig_start - seq_to_sig_offset`.)
    let mut map: Vec<i64> = seq_to_sig[seq_start..=seq_end]
        .iter()
        .map(|&v| v - sig_start_pos)
        .collect();
    // The first and last bases only partially overlap the window; Python snaps
    // them to its edges rather than letting them poke outside.
    let last = map.len() - 1;
    map[0] = 0;
    map[last] = chunk_len as i64;

    let (kmer_before, kmer_after) = skmer_ctx;
    let ctx_lo = seq_start as i64 - kmer_before as i64;
    let ctx_hi = seq_end as i64 + kmer_after as i64;
    let ctx: Vec<u8> = (ctx_lo..ctx_hi)
        .map(|i| match usize::try_from(i) {
            Ok(u) if u < seq_bytes.len() => seq_bytes[u],
            _ => b'N',
        })
        .collect();

    (map, ctx)
}
