use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

const LARGE_SCORE: f32 = 100.0;

/// Squared error score between a signal sample and expected level.
#[inline(always)]
fn score(s: f32, l: f32) -> f32 {
    let tmp = s - l;
    tmp * tmp
}

/// Standard Viterbi forward step for one base (minimizes squared error).
///
/// Decides move vs stay at each band position. Traceback stores number
/// of stays since last move.
fn banded_forward_vit_step(
    curr_scores: &mut [f32],
    curr_tb: &mut [i32],
    prev_scores: &[f32],
    curr_level: f32,
    curr_signal: &[f32],
    band_start_diff: i32,
) {
    let n_curr = curr_scores.len();
    let mut n_prev = prev_scores.len();

    // Effective start index into prev_scores after band_start_diff
    let prev_offset: usize;

    if band_start_diff == 0 {
        // Same start — move here impossible (0-length assignment)
        curr_scores[0] = LARGE_SCORE + prev_scores[n_prev - 1];
        curr_tb[0] = -1;
        prev_offset = 0;
    } else {
        let bsd = band_start_diff as usize;
        let base_score = score(curr_level, curr_signal[0]);
        curr_scores[0] = prev_scores[bsd - 1] + base_score;
        curr_tb[0] = 0;
        prev_offset = bsd;
        n_prev = n_prev.saturating_sub(bsd);
    }

    // If bands are same size, trim prev by one
    let effective_n_prev = if n_prev == n_curr { n_prev - 1 } else { n_prev };

    // Overlap region: move vs stay
    for bp in 1..=effective_n_prev {
        let base_score = score(curr_level, curr_signal[bp]);
        let move_score = prev_scores[prev_offset + bp - 1] + base_score;
        let stay_score = curr_scores[bp - 1] + base_score;
        if move_score < stay_score {
            curr_scores[bp] = move_score;
            curr_tb[bp] = 0;
        } else {
            curr_scores[bp] = stay_score;
            curr_tb[bp] = curr_tb[bp - 1] + 1;
        }
    }

    // Past overlap: forced stays
    for bp in (effective_n_prev + 1)..n_curr {
        let base_score = score(curr_level, curr_signal[bp]);
        curr_scores[bp] = curr_scores[bp - 1] + base_score;
        curr_tb[bp] = curr_tb[bp - 1] + 1;
    }
}

/// Viterbi forward step with short-dwell penalty for one base.
fn banded_forward_dwell_penalty_step(
    curr_scores: &mut [f32],
    curr_tb: &mut [i32],
    prev_scores: &[f32],
    curr_level: f32,
    curr_signal: &[f32],
    band_start_diff: i32,
    dwell_penalty: &[f32],
) {
    let n_curr = curr_scores.len();
    let n_prev = prev_scores.len();
    let n_pen = dwell_penalty.len();

    // Compute unpenalized scores for dwells >= penalty length
    let mut unpen_scores = vec![0.0f32; n_curr];
    let mut unpen_tb = vec![0i32; n_curr];
    banded_forward_vit_step(
        &mut unpen_scores,
        &mut unpen_tb,
        prev_scores,
        curr_level,
        curr_signal,
        band_start_diff,
    );

    for bp in 0..n_curr {
        // Past end of prev band by more than penalty range: forced stay
        if (bp as i32) + band_start_diff - (n_prev as i32) >= (n_pen as i32) {
            curr_scores[bp] = curr_scores[bp - 1] + score(curr_level, curr_signal[bp]);
            curr_tb[bp] = curr_tb[bp - 1] + 1;
            continue;
        }

        // Default: invalid
        curr_scores[bp] = LARGE_SCORE + prev_scores[n_prev - 1];
        curr_tb[bp] = -1;

        if bp == 0 && band_start_diff == 0 {
            continue;
        }

        let mut running_pos_score: f32 = 0.0;
        for dwell_idx in 0..n_pen {
            // Beginning of band reached
            if dwell_idx > bp || (band_start_diff == 0 && bp == dwell_idx) {
                break;
            }

            running_pos_score += score(curr_level, curr_signal[bp - dwell_idx]);

            // Check prev position is in range
            let prev_idx = (bp as i32) - (dwell_idx as i32) - 1 + band_start_diff;
            if prev_idx < 0 || prev_idx >= (n_prev as i32) {
                continue;
            }

            let pos_score =
                prev_scores[prev_idx as usize] + running_pos_score + dwell_penalty[dwell_idx];
            if pos_score < curr_scores[bp] {
                curr_scores[bp] = pos_score;
                curr_tb[bp] = dwell_idx as i32;
            }
        }

        // Check unpenalized score for dwell >= penalty length
        if bp >= n_pen {
            let pos_score = unpen_scores[bp - n_pen] + running_pos_score;
            if pos_score < curr_scores[bp] {
                curr_scores[bp] = pos_score;
                curr_tb[bp] = unpen_tb[bp - n_pen] + (n_pen as i32);
            }
        }
    }
}

/// Banded Viterbi DP for signal map refinement (Remora-compatible).
///
/// Implements the correct Viterbi algorithm with stay/move transitions
/// and optional short-dwell penalties. Minimizes squared error between
/// signal and expected levels within a banded search space.
///
/// Args:
///     signal: Float32 normalized signal values
///     levels: Float32 expected signal levels per base
///     seq_band: int32 array of shape (2, seq_len). Row 0 = lower band
///         boundaries in signal coords, row 1 = upper boundaries.
///     short_dwell_penalty: Float32 penalty array for short dwells
///     algo: "Viterbi" or "dwell_penalty"
///
/// Returns:
///     Int32 array of length seq_len+1 (refined seq_to_sig_map)
#[pyfunction]
pub fn seq_banded_dp<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    levels: PyReadonlyArray1<'py, f32>,
    seq_band: PyReadonlyArray2<'py, i32>,
    short_dwell_penalty: PyReadonlyArray1<'py, f32>,
    algo: &str,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let signal = signal.as_slice()?;
    let levels = levels.as_slice()?;
    let sd_pen = short_dwell_penalty.as_slice()?;
    let seq_band_arr = seq_band.as_array();
    let seq_len = levels.len();

    let use_dwell_pen = algo == "dwell_penalty";

    // Read band boundaries
    let mut band_lo = vec![0i32; seq_len];
    let mut band_hi = vec![0i32; seq_len];
    for i in 0..seq_len {
        band_lo[i] = seq_band_arr[[0, i]];
        band_hi[i] = seq_band_arr[[1, i]];
    }

    // Compute base offsets for ragged array
    let mut base_offsets = vec![0u32; seq_len + 1];
    for i in 0..seq_len {
        let bw = (band_hi[i] - band_lo[i]) as u32;
        base_offsets[i + 1] = base_offsets[i] + bw;
    }
    let band_len = base_offsets[seq_len] as usize;

    // Allocate ragged arrays
    let mut all_scores = vec![0.0f32; band_len];
    let mut traceback = vec![0i32; band_len];

    // First base: spoof prev_scores to force stays
    let first_bw = (band_hi[0] - band_lo[0]) as usize;
    let mut prev_scores = vec![f32::MAX; first_bw];
    prev_scores[0] = 0.0;

    if use_dwell_pen {
        banded_forward_dwell_penalty_step(
            &mut all_scores[..first_bw],
            &mut traceback[..first_bw],
            &prev_scores,
            levels[0],
            &signal[..first_bw],
            1,
            sd_pen,
        );
    } else {
        banded_forward_vit_step(
            &mut all_scores[..first_bw],
            &mut traceback[..first_bw],
            &prev_scores,
            levels[0],
            &signal[..first_bw],
            1,
        );
    }

    let mut prev_band_st = 0i32;
    let mut prev_bw = first_bw;
    let mut prev_offset = 0usize;

    // Process remaining bases
    for base_idx in 1..seq_len {
        let curr_band_st = band_lo[base_idx];
        let curr_band_en = band_hi[base_idx];
        let curr_bw = (curr_band_en - curr_band_st) as usize;
        let curr_offset = base_offsets[base_idx] as usize;

        // We need to pass slices of all_scores for both prev and curr.
        // Since prev and curr don't overlap in the ragged array, we can
        // split the array.
        let band_start_diff = curr_band_st - prev_band_st;

        // Copy prev scores out to avoid borrow conflict
        let prev_sc: Vec<f32> = all_scores[prev_offset..prev_offset + prev_bw].to_vec();

        let sig_slice = &signal[curr_band_st as usize..curr_band_en as usize];

        let (cs, ct) = {
            let scores_slice = &mut all_scores[curr_offset..curr_offset + curr_bw];
            let tb_slice = &mut traceback[curr_offset..curr_offset + curr_bw];
            (scores_slice, tb_slice)
        };

        if use_dwell_pen {
            banded_forward_dwell_penalty_step(
                cs,
                ct,
                &prev_sc,
                levels[base_idx],
                sig_slice,
                band_start_diff,
                sd_pen,
            );
        } else {
            banded_forward_vit_step(
                cs,
                ct,
                &prev_sc,
                levels[base_idx],
                sig_slice,
                band_start_diff,
            );
        }

        prev_band_st = curr_band_st;
        prev_bw = curr_bw;
        prev_offset = curr_offset;
    }

    // Traceback
    let sig_len = band_hi[seq_len - 1];
    let mut path = vec![0i32; seq_len + 1];
    path[0] = 0;
    path[seq_len] = sig_len;

    for base_idx in (1..seq_len).rev() {
        let sig_lookup_pos = path[base_idx + 1] - 1;
        let band_idx = sig_lookup_pos - band_lo[base_idx];
        let offset = base_offsets[base_idx] as usize + band_idx as usize;
        let next_sig_offset = traceback[offset];
        path[base_idx] = sig_lookup_pos - next_sig_offset;
    }

    let result = Array1::from_vec(path);
    Ok(result.into_pyarray(py))
}

/// Extract expected signal levels for each position in a sequence.
///
/// Matches remora's extract_levels: for each valid kmer window starting at pos,
/// looks up the expected level and assigns it to position pos + center_idx.
/// Edge positions where a full kmer cannot be formed are left at 0.
///
/// Args:
///     sequence: DNA/RNA sequence
///     kmer_to_level: Mapping from kmer string to expected level
///     kmer_len: Length of kmers in the table
///     center_idx: Position within the kmer that is the "center" base.
///                 Use -1 for kmer_len / 2 (default).
///
/// Returns:
///     Array of expected levels, length = len(sequence)
#[pyfunction]
#[pyo3(signature = (sequence, kmer_to_level, kmer_len, center_idx = -1))]
pub fn extract_levels<'py>(
    py: Python<'py>,
    sequence: &str,
    kmer_to_level: HashMap<String, f64>,
    kmer_len: usize,
    center_idx: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let seq_bytes = sequence.as_bytes();
    let seq_len = seq_bytes.len();
    let mut levels = Array1::<f64>::zeros(seq_len);
    let cidx = if center_idx < 0 {
        kmer_len / 2
    } else {
        center_idx as usize
    };

    if seq_len < kmer_len {
        return Ok(levels.into_pyarray(py));
    }

    // Pre-process sequence: uppercase and U->T
    let seq_upper: Vec<u8> = seq_bytes
        .iter()
        .map(|&b| {
            let c = b.to_ascii_uppercase();
            if c == b'U' { b'T' } else { c }
        })
        .collect();

    for pos in 0..=(seq_len - kmer_len) {
        let kmer = std::str::from_utf8(&seq_upper[pos..pos + kmer_len]).unwrap_or("");
        if let Some(&level) = kmer_to_level.get(kmer) {
            levels[pos + cidx] = level;
        }
    }

    Ok(levels.into_pyarray(py))
}

/// Rough rescaling of signal to match expected kmer levels.
///
/// Computes per-base mean signal and fits a linear transform to match
/// expected levels.
///
/// Args:
///     signal: Normalized signal array
///     expected_levels: Expected levels per base from kmer table
///     seq_to_sig_map: Base-to-signal mapping
///
/// Returns:
///     Rescaled signal
#[pyfunction]
pub fn rough_rescale<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    expected_levels: PyReadonlyArray1<'py, f64>,
    seq_to_sig_map: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = signal.as_array();
    let expected = expected_levels.as_array();
    let sig_map = seq_to_sig_map.as_array();

    let num_bases = sig_map.len() - 1;
    let mut observed = vec![0.0f64; num_bases];

    let sig_slice = signal.as_slice().expect("signal must be contiguous");

    for i in 0..num_bases {
        let start = sig_map[i] as usize;
        let end = sig_map[i + 1] as usize;
        if end > start {
            let sum: f64 = sig_slice[start..end].iter().sum();
            observed[i] = sum / (end - start) as f64;
        } else if i > 0 {
            observed[i] = observed[i - 1];
        }
    }

    // Fit linear transform via least squares: observed = a * expected + b
    if num_bases > 1 {
        let n = num_bases as f64;
        let sum_x: f64 = expected.iter().sum();
        let sum_y: f64 = observed.iter().sum();
        let sum_xx: f64 = expected.iter().map(|&x| x * x).sum();
        let sum_xy: f64 = expected
            .iter()
            .zip(observed.iter())
            .map(|(&x, &y)| x * y)
            .sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-10 {
            let a = (n * sum_xy - sum_x * sum_y) / denom;
            let b = (sum_y - a * sum_x) / n;

            if a.abs() > 1e-10 {
                let result: Array1<f64> = signal.mapv(|s| (s - b) / a);
                return Ok(result.into_pyarray(py));
            }
        }
    }

    Ok(signal.to_owned().into_pyarray(py))
}
