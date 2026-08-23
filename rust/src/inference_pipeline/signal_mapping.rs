//! Move table and CIGAR-based signal mapping.

pub(super) fn build_seq_to_sig_map(
    mv_array: &[u8],
    stride: u32,
    trim_offset: i64,
    num_samples: u64,
) -> Vec<i64> {
    let mut sig_map: Vec<i64> = mv_array
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 1)
        .map(|(i, _)| i as i64 * stride as i64) // trimmed-space coordinates
        .collect();
    let trimmed_len = num_samples as i64 - trim_offset;
    sig_map.push(trimmed_len);
    sig_map
}

/// Compute reference-to-signal mapping by directly walking the CIGAR.
///
/// Avoids the two-step float interpolation (ref->float_query->float_signal)
/// that caused 1-sample precision differences vs Python's `np.interp`.
///
/// Within CIGAR match blocks (M/=/X), ref->query is 1:1 integer, so we do a
/// direct `query_to_sig[query_pos]` lookup with zero float arithmetic.
/// For indel-gap positions between match blocks, we interpolate using the
/// integer knot coordinates (ratio of small ints -> exact in f64).
pub(super) fn compute_ref_to_signal(query_to_sig: &[i64], cigar_ops: &[(u32, u32)]) -> Vec<i64> {
    const MATCH: [bool; 9] = [true, false, false, false, false, false, false, true, true];
    const REF_CONSUME: [bool; 9] = [true, false, true, true, false, false, false, true, true];
    const QUERY_CONSUME: [bool; 9] = [true, true, false, false, true, false, false, true, true];

    let n_query = query_to_sig.len();
    if n_query == 0 {
        return vec![];
    }
    let last_query = n_query - 1;

    // Strip trailing non-match ops (remora convention)
    let mut cigar: Vec<(u32, u32)> = cigar_ops.to_vec();
    while !cigar.is_empty()
        && !MATCH
            .get(cigar.last().unwrap().0 as usize)
            .copied()
            .unwrap_or(false)
    {
        cigar.pop();
    }
    if cigar.is_empty() {
        return vec![query_to_sig[0]];
    }

    // Collect match blocks as (ref_start, ref_end_exclusive, query_start)
    // and total ref/query lengths.
    let mut ref_pos: i64 = 0;
    let mut query_pos: i64 = 0;
    let mut blocks: Vec<(i64, i64, i64)> = Vec::new();

    for &(op, len) in &cigar {
        let op_idx = op as usize;
        let is_match = MATCH.get(op_idx).copied().unwrap_or(false);
        let consumes_ref = REF_CONSUME.get(op_idx).copied().unwrap_or(false);
        let consumes_query = QUERY_CONSUME.get(op_idx).copied().unwrap_or(false);

        if is_match && len > 0 {
            blocks.push((ref_pos, ref_pos + len as i64, query_pos));
        }
        if consumes_ref {
            ref_pos += len as i64;
        }
        if consumes_query {
            query_pos += len as i64;
        }
    }

    let ref_len = ref_pos; // total reference length after stripping
    let total_query = query_pos;

    if blocks.is_empty() {
        return vec![query_to_sig[0]; (ref_len + 1) as usize];
    }

    // Helper: look up signal position for an exact integer query position
    let sig_at = |q: i64| -> i64 {
        let q = q.max(0).min(last_query as i64);
        query_to_sig[q as usize]
    };

    // Helper: interpolate signal for a fractional query position.
    // All inputs are integers -> ratio is exact in f64.
    let sig_interp = |q_num: i64, q_den: i64, q_base: i64| -> i64 {
        // q_float = q_base + q_num / q_den  (but we keep it as a ratio)
        let q_float = q_base as f64 + q_num as f64 / q_den as f64;
        if q_float <= 0.0 {
            return query_to_sig[0];
        }
        if q_float >= last_query as f64 {
            return query_to_sig[last_query];
        }
        let j = q_float.floor() as usize;
        let frac = q_float - j as f64;
        let lo = query_to_sig[j] as f64;
        let hi = query_to_sig[j + 1] as f64;
        (lo + frac * (hi - lo)).floor() as i64
    };

    let mut result = Vec::with_capacity((ref_len + 1) as usize);

    // Knots for gap interpolation follow the remora convention:
    // Each match block contributes two knots: (block_ref_start, block_query_start)
    // and (block_ref_end-1, block_query_end-1).  Between match blocks,
    // interpolate linearly between the end-1 knot of block A and the start
    // knot of block B.

    // Gap before first match block: interpolate between (0,0) and first block start
    let (first_ref, _, first_query) = blocks[0];
    if first_ref > 0 {
        // Knots: (0, 0) and (first_ref, first_query)
        for r in 0..first_ref {
            // t = r / first_ref, q = t * first_query
            result.push(sig_interp(r * first_query, first_ref, 0));
        }
    }

    for (bi, &(blk_ref_start, blk_ref_end, blk_query_start)) in blocks.iter().enumerate() {
        // Fill match block: direct integer lookup
        for r in blk_ref_start..blk_ref_end {
            let q = blk_query_start + (r - blk_ref_start);
            result.push(sig_at(q));
        }

        // Gap after this match block (before next block, or to end)
        let knot_ref_a = blk_ref_end - 1; // end-1 of this block
        let knot_query_a = blk_query_start + (blk_ref_end - 1 - blk_ref_start);

        let (knot_ref_b, knot_query_b) = if bi + 1 < blocks.len() {
            // Next block exists: interpolate to its start
            let (next_ref, _, next_query) = blocks[bi + 1];
            (next_ref, next_query)
        } else {
            // After last block: interpolate to (ref_len, total_query)
            (ref_len, total_query)
        };

        let gap_start = blk_ref_end;
        let gap_end = knot_ref_b; // exclusive (knot_ref_b itself is the next block start or ref_len)

        if gap_start < gap_end {
            let ref_span = knot_ref_b - knot_ref_a; // always > 0
            let query_span = knot_query_b - knot_query_a;
            for r in gap_start..gap_end {
                // t = (r - knot_ref_a) / ref_span
                // q = knot_query_a + t * query_span
                let dr = r - knot_ref_a;
                result.push(sig_interp(dr * query_span, ref_span, knot_query_a));
            }
        }
    }

    // Final entry: ref_len maps to total_query
    result.push(sig_at(total_query));

    // Ensure correct length (ref_len + 1)
    result.truncate((ref_len + 1) as usize);
    result
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
