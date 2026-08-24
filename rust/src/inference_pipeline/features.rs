//! Per-base feature computation and sequence encoding.

pub(super) fn compute_dwell_features(dwells: &[f32]) -> Vec<Vec<f32>> {
    let n = dwells.len();
    if n == 0 {
        return vec![vec![]; 5];
    }
    let window = 5usize;
    let pad = window / 2;
    let eps = 1e-6f32;

    let padded_len = n + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    for val in padded[..pad].iter_mut() {
        *val = dwells[0];
    }
    padded[pad..pad + n].copy_from_slice(dwells);
    for val in padded[pad + n..].iter_mut() {
        *val = dwells[n - 1];
    }

    let mut dwell_mean = vec![0.0f32; n];
    let mut dwell_std = vec![0.0f32; n];
    for i in 0..n {
        let win = &padded[i..i + window];
        let sum: f32 = win.iter().sum();
        let mean = sum / window as f32;
        dwell_mean[i] = mean;
        let var: f32 = win.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / window as f32;
        dwell_std[i] = var.sqrt();
    }

    let dwell_raw = dwells.to_vec();
    let dwell_log: Vec<f32> = dwells.iter().map(|&d| (d + eps).ln()).collect();
    let dwell_ratio: Vec<f32> = dwells
        .iter()
        .zip(dwell_mean.iter())
        .map(|(&d, &m)| d / (m + eps))
        .collect();

    vec![dwell_raw, dwell_log, dwell_mean, dwell_std, dwell_ratio]
}

/// Per-base k-mer residual features, all three rows `observed_means.len()` wide.
///
/// `expected_levels` is indexed by *sequence* base and `observed_means` by
/// *mapped* base; under `anchor="reference"` an alignment ending in a
/// non-match CIGAR op makes the second shorter (see `LeechRead.num_mapped_bases`
/// and `levels_for_mapped_bases` on the Python side). Fitting the levels to the
/// mapped-base grid up front is what keeps all three rows the same width as
/// every other feature row: `kmer_expected` used to be emitted at the full
/// sequence length while the two derived rows were zipped down to the shorter
/// one, so a single read could produce feature rows of two different lengths
/// and the `safe_end <= feat_row.len()` guard in chunk extraction would zero
/// some of them and not others.
pub(super) fn compute_kmer_residual_features(
    observed_means: &[f32],
    expected_levels: &[f64],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_bases = observed_means.len();
    let mut kmer_expected = vec![0.0f32; num_bases];
    for (dst, &src) in kmer_expected.iter_mut().zip(expected_levels.iter()) {
        *dst = src as f32;
    }
    let kmer_residual: Vec<f32> = observed_means
        .iter()
        .zip(kmer_expected.iter())
        .map(|(&o, &e)| o - e)
        .collect();
    let kmer_residual_abs: Vec<f32> = kmer_residual.iter().map(|&r| r.abs()).collect();

    (kmer_expected, kmer_residual, kmer_residual_abs)
}

pub(super) fn compute_signal_residual(
    signal: &[f32],
    sig_map: &[i64],
    expected_levels: &[f64],
    num_bases: usize,
) -> Vec<f32> {
    let sig_len = signal.len();
    let mut residual = vec![0.0f32; sig_len];
    for i in 0..num_bases {
        let start = sig_map[i].max(0) as usize;
        let end = sig_map[i + 1].max(0) as usize;
        let end = end.min(sig_len);
        // `.get` rather than `[i]`: this runs inside a rayon worker under
        // `py.detach`, where an out-of-range index is a panic that takes the
        // whole batch with it. Levels are per sequence base and `num_bases` is
        // per mapped base; they are only equal by construction today.
        let expected = expected_levels.get(i).copied().unwrap_or(0.0) as f32;
        if expected.abs() > 1e-12 {
            for s in start..end {
                residual[s] = signal[s] - expected;
            }
        }
    }
    residual
}

pub(super) fn sequence_to_int(sequence: &[u8]) -> Vec<i8> {
    sequence
        .iter()
        .map(|&c| match c {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ => -1,
        })
        .collect()
}

pub(super) fn encode_base_onehot(sequence: &[u8]) -> Vec<f32> {
    let seq_len = sequence.len();
    let mut encoding = vec![0.0f32; 4 * seq_len];
    for (i, &c) in sequence.iter().enumerate() {
        let idx = match c {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ => continue,
        };
        encoding[idx * seq_len + i] = 1.0;
    }
    encoding
}
