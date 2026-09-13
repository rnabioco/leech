//! CIGAR-op translation for reference anchoring.
//!
//! Move-table mapping, reference-signal mapping and the chunk-local
//! signal_kmer inputs all moved to `escapepod_signal::chunk` and `::mapping`
//! (rnabioco/leech#258) -- `process_one_read`/`process_one_read_training` call
//! them directly. What's left here is BAM's CIGAR op numbering, which has
//! nowhere upstream to live: it is pysam's encoding of the SAM spec, not a
//! signal-processing rule.

#[cfg(feature = "test-utils")]
use escapepod_signal::mapping::ref_to_signal;
use escapepod_signal::mapping::{CigarKind, CigarOp};

/// A BAM CIGAR op code (`pysam`'s `cigartuples` encoding) as a [`CigarKind`].
///
/// escapepod takes a typed `CigarKind` rather than the raw integer, which is
/// the right call -- but it means this table is the one place the numbering
/// still has to be written down. It is the SAM spec order, unchanged since the
/// format was defined: `MIDNSHP=X`. An unrecognised code maps to `Pad`, which
/// consumes neither query nor reference and so cannot shift a coordinate.
pub(super) fn cigar_kind(op: u32) -> CigarKind {
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

/// `cigar_ops` (pysam tuples) as typed [`CigarOp`]s, ready for
/// `Anchor::Reference` or [`compute_ref_to_signal`].
pub(super) fn typed_cigar(cigar_ops: &[(u32, u32)]) -> Vec<CigarOp> {
    cigar_ops
        .iter()
        .map(|&(op, len)| CigarOp::new(cigar_kind(op), len))
        .collect()
}

/// Reference->signal map, by the Remora knot convention. Test-only: production
/// reaches the same upstream function through `Anchor::Reference` inside
/// `chunk::process_read`; this wrapper exists so `_test_ref_to_signal` (and
/// `tests/bench_cigar_parity.py`) can still probe it directly.
#[cfg(feature = "test-utils")]
pub(super) fn compute_ref_to_signal(query_to_sig: &[i64], cigar_ops: &[(u32, u32)]) -> Vec<i64> {
    ref_to_signal(query_to_sig, &typed_cigar(cigar_ops))
}

#[cfg(test)]
mod tests {
    use super::*;
    use escapepod_signal::chunk::{ProcessedRead, signal_kmer_inputs};
    use escapepod_signal::seq_encoding::KmerContext;

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

    // Four bases (map has 5 boundaries, 0..=40 in steps of 10). These three
    // cases moved here, ported to call `escapepod_signal::chunk::
    // signal_kmer_inputs` directly, when leech's own `chunk_signal_kmer_inputs`
    // (which they used to pin) was deleted in favor of the now-`pub` upstream
    // function (rnabioco/leech#258, rnabioco/escapepod-rs#380). The edge cases
    // -- an underflowing window, a window past the end, a map shorter than the
    // sequence -- are the ones a re-derivation gets wrong, so they stay
    // covered even though the implementation moved.
    fn read_with_map(map: &[i64], sequence: &[u8]) -> ProcessedRead {
        ProcessedRead {
            signal: vec![0.0; map.last().copied().unwrap_or(0).max(0) as usize],
            seq_to_sig: map.to_vec(),
            sequence: sequence.to_vec(),
            levels: None,
        }
    }

    #[test]
    fn signal_kmer_inputs_underflowing_window_clamps_to_zero_and_snaps_edges() {
        // sig_start is negative -- the window starts before the signal. The
        // returned map stays offset against the UNCLAMPED start, and the
        // first/last covered bases snap to the chunk edges regardless.
        let read = read_with_map(&[0, 10, 20, 30, 40], b"ACGT");
        let (map, ctx) = signal_kmer_inputs(
            &read,
            -15,
            25,
            40,
            KmerContext {
                before: 2,
                after: 2,
            },
        )
        .unwrap();

        assert_eq!(map, vec![0, 25, 35, 40], "map: {map:?}");
        assert!(!ctx.is_empty());
    }

    #[test]
    fn signal_kmer_inputs_window_past_end_clamps_to_num_samples() {
        // sig_end exceeds the signal length -- the window runs off the end.
        let read = read_with_map(&[0, 10, 20, 30, 40], b"ACGT");
        let (map, ctx) = signal_kmer_inputs(
            &read,
            15,
            55,
            40,
            KmerContext {
                before: 2,
                after: 2,
            },
        )
        .unwrap();

        assert_eq!(map, vec![0, 5, 15, 40], "map: {map:?}");
        assert!(!ctx.is_empty());
    }

    #[test]
    fn signal_kmer_inputs_map_shorter_than_sequence_bounds_by_the_map() {
        // The map covers only 2 bases, but the sequence carries 6 -- an
        // alignment that stopped short of the full read (CLAUDE.md: "the map
        // can be shorter than the reference slice"). The end must clamp to
        // the map's own base count, not to the sequence length.
        let read = read_with_map(&[0, 10, 20], b"ACGTAC");
        let (map, ctx) = signal_kmer_inputs(
            &read,
            0,
            20,
            20,
            KmerContext {
                before: 2,
                after: 2,
            },
        )
        .unwrap();

        assert_eq!(map, vec![0, 10, 20], "map: {map:?}");
        assert!(!ctx.is_empty());
    }

    #[test]
    fn signal_kmer_inputs_map_too_short_to_have_any_base_returns_none() {
        // seq_to_sig.len() < 2: no base has both boundaries, so there is
        // nothing to chunk.
        let read = read_with_map(&[5], b"ACGT");
        assert!(
            signal_kmer_inputs(
                &read,
                0,
                20,
                20,
                KmerContext {
                    before: 2,
                    after: 2
                }
            )
            .is_none()
        );
    }
}
