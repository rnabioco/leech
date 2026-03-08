use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Banded Viterbi DP for signal map refinement.
///
/// Finds optimal base boundaries within banded region that maximizes
/// likelihood of observed signal given expected levels.
///
/// The scoring function is -0.5 * (signal[t] - expected_level[b])^2
/// for each signal position t assigned to base b.
///
/// Args:
///     signal: Signal array (rescaled to match expected levels)
///     expected_levels: Expected signal level per base
///     band_starts: Start of valid signal range per base
///     band_ends: End of valid signal range per base
///     sig_len: Total signal length
///
/// Returns:
///     Refined seq_to_sig_map of length num_bases+1
#[pyfunction]
pub fn seq_banded_dp<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    expected_levels: PyReadonlyArray1<'py, f64>,
    band_starts: PyReadonlyArray1<'py, i64>,
    band_ends: PyReadonlyArray1<'py, i64>,
    sig_len: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let signal = signal.as_array();
    let expected = expected_levels.as_array();
    let starts = band_starts.as_array();
    let ends = band_ends.as_array();

    let num_bases = expected.len();
    let num_positions = num_bases + 1;

    // Compute max band width
    let max_band = starts
        .iter()
        .zip(ends.iter())
        .map(|(&s, &e)| (e - s) as usize + 1)
        .max()
        .unwrap_or(1);

    // DP tables - use flat Vec for performance
    let mut scores = vec![f64::NEG_INFINITY; num_positions * max_band];
    let mut traceback = vec![0i64; num_positions * max_band];

    // Helper: index into flat DP table
    let idx = |b: usize, j: usize| -> usize { b * max_band + j };

    // Initialize first boundary
    let start0 = starts[0] as usize;
    let end0 = (ends[0] as usize).min(sig_len);
    for t in start0..end0 {
        let j = t - start0;
        if j < max_band {
            scores[idx(0, j)] = 0.0;
        }
    }

    // Get signal as slice for fast access
    let sig_slice = signal.as_slice().expect("signal must be contiguous");

    // Forward pass
    for b in 1..num_positions {
        let b_start = starts[b] as usize;
        let b_end = (ends[b] as usize).min(sig_len);
        let prev_start = starts[b - 1] as usize;
        let prev_end = (ends[b - 1] as usize).min(sig_len);
        let level = expected[b - 1];

        for t in b_start..b_end {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_prev: i64 = -1;

            let prev_t_max = t.min(prev_end);
            for prev_t in prev_start..prev_t_max {
                let prev_j = prev_t - prev_start;
                if prev_j >= max_band {
                    continue;
                }
                let prev_score = scores[idx(b - 1, prev_j)];
                if prev_score == f64::NEG_INFINITY {
                    continue;
                }

                // Score for assigning signal[prev_t..t] to base b-1
                let mut seg_score = 0.0f64;
                for s in prev_t..t {
                    let diff = sig_slice[s] - level;
                    seg_score -= 0.5 * diff * diff;
                }

                let total = prev_score + seg_score;
                if total > best_score {
                    best_score = total;
                    best_prev = prev_t as i64;
                }
            }

            let j = t - b_start;
            if j < max_band {
                scores[idx(b, j)] = best_score;
                traceback[idx(b, j)] = best_prev;
            }
        }
    }

    // Traceback: find best final position
    let last_start = starts[num_positions - 1] as usize;
    let last_end = (ends[num_positions - 1] as usize).min(sig_len);
    let mut best_final_score = f64::NEG_INFINITY;
    let mut best_final_t = sig_len as i64;

    for t in last_start..last_end {
        let j = t - last_start;
        if j < max_band && scores[idx(num_positions - 1, j)] > best_final_score {
            best_final_score = scores[idx(num_positions - 1, j)];
            best_final_t = t as i64;
        }
    }

    // Reconstruct path
    let mut refined = Array1::<i64>::zeros(num_positions);
    refined[num_positions - 1] = best_final_t;

    for b in (1..num_positions).rev() {
        let t = refined[b] as usize;
        let b_start = starts[b] as usize;
        if t >= b_start {
            let j = t - b_start;
            if j < max_band {
                refined[b - 1] = traceback[idx(b, j)];
            } else {
                refined[b - 1] = starts[b - 1];
            }
        } else {
            refined[b - 1] = starts[b - 1];
        }
    }

    Ok(refined.into_pyarray(py))
}

/// Extract expected signal levels for each position in a sequence.
///
/// For each position i, looks up the kmer centered at i and returns the
/// expected signal level. Positions near the edges use truncated kmers
/// if available, otherwise use the nearest valid kmer's level.
///
/// Args:
///     sequence: DNA/RNA sequence
///     kmer_to_level: Mapping from kmer string to expected level
///     kmer_len: Length of kmers in the table
///
/// Returns:
///     Array of expected levels, length = len(sequence)
#[pyfunction]
pub fn extract_levels<'py>(
    py: Python<'py>,
    sequence: &str,
    kmer_to_level: HashMap<String, f64>,
    kmer_len: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let seq_bytes = sequence.as_bytes();
    let seq_len = seq_bytes.len();
    let mut levels = Array1::<f64>::zeros(seq_len);
    let half_k = kmer_len / 2;

    for i in 0..seq_len {
        let start = i as isize - half_k as isize;
        let end = i + half_k + 1;

        let (kmer_start, kmer_end) = if start < 0 || end > seq_len {
            let clamped_start = (start.max(0) as usize).min(seq_len.saturating_sub(kmer_len));
            (clamped_start, clamped_start + kmer_len)
        } else {
            (start as usize, end)
        };

        // Build kmer string, replacing U with T and uppercasing
        let kmer: String = seq_bytes[kmer_start..kmer_end]
            .iter()
            .map(|&b| {
                let c = (b as char).to_ascii_uppercase();
                if c == 'U' {
                    'T'
                } else {
                    c
                }
            })
            .collect();

        if let Some(&level) = kmer_to_level.get(&kmer) {
            levels[i] = level;
        } else if i > 0 {
            levels[i] = levels[i - 1];
        }
    }

    Ok(levels.into_pyarray(py))
}

/// Rough rescaling of signal to match expected kmer levels.
///
/// Computes per-base mean signal and fits a linear transform to match
/// expected levels. This makes the Viterbi scoring more accurate.
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

            // Apply inverse: rescaled = (signal - b) / a
            if a.abs() > 1e-10 {
                let result: Array1<f64> = signal.mapv(|s| (s - b) / a);
                return Ok(result.into_pyarray(py));
            }
        }
    }

    // Fallback: return copy of original signal
    Ok(signal.to_owned().into_pyarray(py))
}
