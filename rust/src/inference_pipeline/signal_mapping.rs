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

    // The windowing comes from escapepod-signal, which also owns the map this
    // sits between and the encoding it feeds. Kept as BASES rather than ints
    // because the corpus serializes `sequence_with_kmer_context` as a string;
    // `sequence_ints_with_context` is the same window in the other alphabet,
    // and `sequence_to_int` of this is exactly that (escapepod-rs#274).
    //
    // This is the step where `before` and `after` are NOT interchangeable:
    // swapping them displaces every k-mer silently, and the encoder cannot
    // detect it because it only sees the total width.
    let (kmer_before, kmer_after) = skmer_ctx;
    let ctx = escapepod_signal::seq_encoding::sequence_bases_with_context(
        seq_bytes,
        seq_start,
        seq_end - seq_start,
        escapepod_signal::seq_encoding::KmerContext {
            before: kmer_before,
            after: kmer_after,
        },
    );

    (map, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cigar_kind_maps_the_sam_spec_ops() {
        assert_eq!(cigar_kind(0), CigarKind::Match);
        assert_eq!(cigar_kind(1), CigarKind::Insertion);
        assert_eq!(cigar_kind(2), CigarKind::Deletion);
        assert_eq!(cigar_kind(3), CigarKind::Skip);
        assert_eq!(cigar_kind(4), CigarKind::SoftClip);
        assert_eq!(cigar_kind(5), CigarKind::HardClip);
        assert_eq!(cigar_kind(7), CigarKind::SequenceMatch);
        assert_eq!(cigar_kind(8), CigarKind::SequenceMismatch);
    }

    #[test]
    fn cigar_kind_maps_padding_via_the_wildcard() {
        // Op 6 (`P`, Padding) IS an assigned SAM op -- it just has no
        // explicit match arm here, unlike 0-5/7/8. It reaches `Pad` only
        // through the wildcard, which happens to be correct: `P` already
        // consumes neither query nor reference, the same as `Pad`'s meaning
        // for a genuinely unrecognised code below.
        assert_eq!(cigar_kind(6), CigarKind::Pad);
    }

    #[test]
    fn cigar_kind_treats_an_unrecognized_op_as_consuming_nothing() {
        // 9 and 255 are outside the whole SAM `MIDNSHP=X` table (0-8) --
        // genuinely unknown, not merely un-matched like op 6 above. An
        // unrecognised code must not be treated as any op that advances a
        // coordinate -- mapping it to `Pad` (consumes neither query nor
        // reference) is what keeps a bad/future op code from silently
        // shifting `compute_ref_to_signal`'s output rather than erroring.
        for op in [9u32, 255] {
            let kind = cigar_kind(op);
            assert_eq!(kind, CigarKind::Pad, "op {op}");
            assert!(!kind.consumes_query(), "op {op}");
            assert!(!kind.consumes_reference(), "op {op}");
        }
    }

    // Four bases (map has 5 boundaries, 0..=40 in steps of 10).
    const MAP: [i64; 5] = [0, 10, 20, 30, 40];

    #[test]
    fn chunk_signal_kmer_inputs_underflowing_window_clamps_to_zero_and_snaps_edges() {
        // sig_start_pos is negative -- the window starts before the signal.
        // `ss = sig_start_pos.max(0)` clamps the search, but the returned map
        // stays offset against the UNCLAMPED start (Python reaches the same
        // value via `sig_start - seq_to_sig_offset`), and the first/last
        // covered bases snap to the chunk edges regardless.
        let seq_bytes = b"ACGT";
        let (map, ctx) = chunk_signal_kmer_inputs(&MAP, seq_bytes, -15, 25, 40, 40, (2, 2));

        assert_eq!(map, vec![0, 25, 35, 40], "map: {map:?}");
        assert!(!ctx.is_empty());
    }

    #[test]
    fn chunk_signal_kmer_inputs_window_past_end_clamps_to_num_samples() {
        // sig_end_pos exceeds num_samples -- the window runs off the end of
        // the signal. `se = sig_end_pos.min(num_samples)` clamps the search.
        let seq_bytes = b"ACGT";
        let (map, ctx) = chunk_signal_kmer_inputs(&MAP, seq_bytes, 15, 55, 40, 40, (2, 2));

        assert_eq!(map, vec![0, 5, 15, 40], "map: {map:?}");
        assert!(!ctx.is_empty());
    }

    #[test]
    fn chunk_signal_kmer_inputs_map_shorter_than_sequence_bounds_by_the_map() {
        // The map covers only 2 bases, but the sequence carries 6 -- an
        // alignment that stopped short of the full read (CLAUDE.md: "the map
        // can be shorter than the reference slice"). `seq_end` must clamp to
        // `num_bases_map`, not to `seq_bytes.len()`, or this indexes past
        // what the map actually describes.
        let short_map = [0i64, 10, 20];
        let seq_bytes = b"ACGTAC";
        let (map, ctx) = chunk_signal_kmer_inputs(&short_map, seq_bytes, 0, 20, 20, 20, (2, 2));

        assert_eq!(map, vec![0, 10, 20], "map: {map:?}");
        assert!(!ctx.is_empty());
    }

    #[test]
    fn chunk_signal_kmer_inputs_map_too_short_to_have_any_base_returns_empty() {
        // seq_to_sig.len() < 2: no base has both boundaries, so there is
        // nothing to chunk.
        let degenerate_map = [5i64];
        let seq_bytes = b"ACGT";
        let (map, ctx) =
            chunk_signal_kmer_inputs(&degenerate_map, seq_bytes, 0, 20, 20, 20, (2, 2));

        assert!(map.is_empty());
        assert!(ctx.is_empty());
    }
}
