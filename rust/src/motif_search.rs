//! Motif search functions for finding modification sites.
//!
//! Port of `leech.io.motif_search` to Rust. Supports:
//! - Simple substring search (`find_motif_in_sequence`)
//! - Reference-to-query coordinate mapping via CIGAR (`map_reference_to_query_coords`)
//! - Full reference-based motif search (`reference_motif_search`)

/// Find all occurrences of `motif` in `sequence[start..end]` (case-insensitive).
///
/// Returns positions relative to the full sequence (not the slice).
pub(crate) fn find_motif_in_sequence(sequence: &str, motif: &str, start: usize, end: usize) -> Vec<usize> {
    let end = end.min(sequence.len());
    if start >= end || motif.is_empty() || motif.len() > (end - start) {
        return vec![];
    }

    let region = &sequence.as_bytes()[start..end];
    let motif_bytes: Vec<u8> = motif.bytes().map(|b| b.to_ascii_uppercase()).collect();
    let motif_len = motif_bytes.len();
    let mut positions = Vec::new();

    for i in 0..=(region.len() - motif_len) {
        let mut matched = true;
        for j in 0..motif_len {
            if region[i + j].to_ascii_uppercase() != motif_bytes[j] {
                matched = false;
                break;
            }
        }
        if matched {
            positions.push(start + i);
        }
    }

    positions
}

/// Map reference coordinates to query coordinates using CIGAR operations.
///
/// Walks the CIGAR ops tracking ref_pos/query_pos to map a reference region
/// `[ref_start, ref_end)` to query coordinates.
///
/// If `skip_indels` is true, returns `None` when indels are found in the
/// mapped region. If `allow_edge_indels` is true, only indels in the core
/// region (excluding ±edge_tolerance from edges) cause rejection.
///
/// Returns `Some((query_start, query_end))` or `None` if mapping fails.
pub(crate) fn map_reference_to_query_coords(
    cigar_ops: &[(u32, u32)],
    ref_start_aligned: u64,
    ref_motif_start: usize,
    ref_motif_end: usize,
    skip_indels: bool,
    allow_edge_indels: bool,
) -> Option<(usize, usize)> {
    let mut ref_pos = ref_start_aligned as usize;
    let mut query_pos: usize = 0;

    let mut query_start: Option<usize> = None;
    let mut query_end: Option<usize> = None;
    let mut has_indel_in_region = false;

    let motif_len = ref_motif_end - ref_motif_start;
    let edge_tolerance = if allow_edge_indels {
        3usize.min(motif_len / 2)
    } else {
        0
    };

    for &(op, length) in cigar_ops {
        if query_start.is_some() && query_end.is_some() {
            break;
        }

        let len = length as usize;

        match op {
            // M/=/X: match/mismatch (consumes both ref and query)
            0 | 7 | 8 => {
                if query_start.is_none()
                    && ref_motif_start >= ref_pos
                    && ref_motif_start < ref_pos + len
                {
                    query_start = Some(query_pos + (ref_motif_start - ref_pos));
                }
                if query_end.is_none()
                    && ref_motif_end > ref_pos  // ref_motif_end - 1 >= ref_pos
                    && ref_motif_end <= ref_pos + len
                {
                    query_end = Some(query_pos + (ref_motif_end - ref_pos));
                }
                ref_pos += len;
                query_pos += len;
            }
            // I: insertion (consumes query only)
            1 => {
                if allow_edge_indels {
                    let core_start = ref_motif_start + edge_tolerance;
                    let core_end = ref_motif_end.saturating_sub(edge_tolerance);
                    if ref_pos >= core_start && ref_pos < core_end {
                        has_indel_in_region = true;
                    }
                } else if ref_pos >= ref_motif_start && ref_pos < ref_motif_end {
                    has_indel_in_region = true;
                }
                query_pos += len;
            }
            // D: deletion (consumes reference only)
            2 => {
                if allow_edge_indels {
                    let core_start = ref_motif_start + edge_tolerance;
                    let core_end = ref_motif_end.saturating_sub(edge_tolerance);
                    if ref_pos >= core_start && ref_pos + len > core_start && ref_pos < core_end {
                        has_indel_in_region = true;
                    }
                } else if ref_pos >= ref_motif_start && ref_pos + len > ref_motif_start && ref_pos < ref_motif_end {
                    has_indel_in_region = true;
                }
                ref_pos += len;
            }
            // S: soft clip (consumes query only)
            4 => {
                query_pos += len;
            }
            // N: ref skip (consumes reference only)
            3 => {
                ref_pos += len;
            }
            // H (5), P (6): don't consume either
            _ => {}
        }
    }

    let (qs, qe) = match (query_start, query_end) {
        (Some(s), Some(e)) => (s, e),
        _ => return None,
    };

    if skip_indels && has_indel_in_region {
        return None;
    }

    Some((qs, qe))
}

/// Full reference-based motif search.
///
/// Finds motif occurrences in the reference sequence within the aligned region,
/// maps each to query coordinates via CIGAR, applies motif_offset, and returns
/// focus base positions.
///
/// When `anchor` is "reference", returns positions relative to `ref_start`
/// (reference coordinates). When "basecall", returns query coordinates.
pub(crate) fn reference_motif_search(
    ref_seq: &str,
    ref_start: usize,
    ref_end: usize,
    cigar_ops: &[(u32, u32)],
    ref_start_aligned: u64,
    motif: &str,
    motif_offset: i32,
    skip_indels: bool,
    allow_edge_indels: bool,
    anchor: &str,
) -> Vec<i64> {
    let motif_len = motif.len();

    // Find motif positions in reference within aligned region
    let motif_positions = find_motif_in_sequence(ref_seq, motif, ref_start, ref_end);

    let mut focus_positions = Vec::new();

    for ref_motif_start in motif_positions {
        let ref_motif_end = ref_motif_start + motif_len;

        // Map to query coordinates
        let query_coords = map_reference_to_query_coords(
            cigar_ops,
            ref_start_aligned,
            ref_motif_start,
            ref_motif_end,
            skip_indels,
            allow_edge_indels,
        );

        let (query_start, _query_end) = match query_coords {
            Some(coords) => coords,
            None => continue,
        };

        // Length check (lenient for non-indel-skip, strict otherwise)
        let mapped_len = _query_end as i64 - query_start as i64;
        let length_ok = if skip_indels {
            mapped_len == motif_len as i64
        } else {
            (mapped_len - motif_len as i64).abs() <= 3
        };

        if !length_ok {
            continue;
        }

        // Apply offset and determine focus position
        if anchor == "reference" {
            // Reference coordinates relative to alignment start
            let pos = (ref_motif_start as i64 - ref_start_aligned as i64) + motif_offset as i64;
            focus_positions.push(pos);
        } else {
            // Query (basecall) coordinates
            let pos = query_start as i64 + motif_offset as i64;
            focus_positions.push(pos);
        }
    }

    focus_positions
}

/// Basecalled motif search: find motif in basecalled sequence directly.
///
/// Returns focus base positions (motif start + offset).
pub(crate) fn basecalled_motif_search(
    sequence: &str,
    motif: &str,
    motif_offset: i32,
) -> Vec<i64> {
    let positions = find_motif_in_sequence(sequence, motif, 0, sequence.len());
    positions.iter().map(|&p| p as i64 + motif_offset as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_motif_basic() {
        let positions = find_motif_in_sequence("ACGTCCAGGCTTCCAGGC", "CCAGGC", 0, 18);
        assert_eq!(positions, vec![4, 12]);
    }

    #[test]
    fn test_find_motif_case_insensitive() {
        let positions = find_motif_in_sequence("acgtCCAGGCtt", "ccaggc", 0, 12);
        assert_eq!(positions, vec![4]);
    }

    #[test]
    fn test_find_motif_with_range() {
        let positions = find_motif_in_sequence("CCAGGCTTCCAGGC", "CCAGGC", 2, 14);
        assert_eq!(positions, vec![8]);
    }

    #[test]
    fn test_find_motif_empty() {
        let positions = find_motif_in_sequence("ACGT", "CCAGGC", 0, 4);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_map_coords_simple_match() {
        // Simple 10M CIGAR
        let cigar = vec![(0u32, 10u32)]; // 10M
        let result = map_reference_to_query_coords(&cigar, 0, 2, 5, false, false);
        assert_eq!(result, Some((2, 5)));
    }

    #[test]
    fn test_map_coords_with_insertion() {
        // 5M1I5M
        let cigar = vec![(0, 5), (1, 1), (0, 5)];
        // Ref pos 0-4 → query 0-4, ref pos 5-9 → query 6-10
        let result = map_reference_to_query_coords(&cigar, 0, 6, 8, false, false);
        assert_eq!(result, Some((7, 9)));
    }

    #[test]
    fn test_map_coords_skip_indels() {
        // 3M1I3M — insertion at ref pos 3, in region [2,5)
        let cigar = vec![(0, 3), (1, 1), (0, 3)];
        let result = map_reference_to_query_coords(&cigar, 0, 2, 5, true, false);
        assert_eq!(result, None);
    }

    #[test]
    fn test_basecalled_search() {
        let positions = basecalled_motif_search("ACGTCCAGGCTT", "CCAGGC", 2);
        assert_eq!(positions, vec![6]); // motif at 4 + offset 2 = 6
    }
}
