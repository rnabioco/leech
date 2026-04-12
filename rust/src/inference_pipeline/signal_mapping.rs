//! Move table and CIGAR-based signal mapping.

pub(super) fn build_seq_to_sig_map(mv_array: &[u8], stride: u32, trim_offset: i64, num_samples: u64) -> Vec<i64> {
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
