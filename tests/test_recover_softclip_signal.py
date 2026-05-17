"""Edge-recovery for soft-clipped signal in ref-anchored mode.

R4 from the coordinate-positioning audit. In reference-anchored mode the
LeechRead's normalized signal is cropped to the aligned region, which
makes refinement and feature extraction work on in-distribution data but
forces ``get_chunk`` to zero-pad chunk windows that extend into
soft-clipped/unaligned signal.

When ``recover_softclip_signal=True`` and the LeechRead carries a full
pre-crop signal (``full_signal``) plus the crop offset (``signal_offset``),
``get_chunk`` should fill those zero-padded edges with the real samples
from the underlying read. The flag is opt-in to preserve Remora-compatible
zero-pad behavior by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from leech.chunking import LeechRead
from leech.configs import ChunkConfig


def _make_read_with_softclip(
    *,
    aligned_len: int = 200,
    left_softclip: int = 100,
    right_softclip: int = 100,
    seed: int = 0,
) -> LeechRead:
    """Build a synthetic ref-anchored LeechRead. ``full_signal`` is a
    distinct, non-zero waveform so the test can tell soft-clipped fill
    from default zero-pad."""
    rng = np.random.default_rng(seed)
    full_len = left_softclip + aligned_len + right_softclip

    # Use distinct value ranges per region so we can fingerprint where each
    # chunk sample came from:
    #   left soft-clip  -> values in [-3, -2]
    #   aligned region  -> values in [ 0,  1]
    #   right soft-clip -> values in [ 2,  3]
    full_signal = np.zeros(full_len, dtype=np.float32)
    full_signal[:left_softclip] = rng.uniform(-3.0, -2.0, left_softclip).astype(np.float32)
    full_signal[left_softclip : left_softclip + aligned_len] = rng.uniform(
        0.0, 1.0, aligned_len
    ).astype(np.float32)
    full_signal[left_softclip + aligned_len :] = rng.uniform(2.0, 3.0, right_softclip).astype(
        np.float32
    )

    cropped_signal = full_signal[left_softclip : left_softclip + aligned_len].copy()

    # 20 aligned bases evenly spaced across the aligned region
    num_bases = 20
    seq_to_sig_map = np.linspace(0, aligned_len, num_bases + 1, dtype=np.int64)
    dwells = np.diff(seq_to_sig_map)

    return LeechRead(
        read_id="softclip_test",
        sequence="A" * num_bases,
        signal=cropped_signal,
        seq_to_sig_map=seq_to_sig_map,
        dwells=dwells,
        dwell_features={"dwell": dwells.astype(np.float32)},
        signal_features={"level_mean": np.zeros(num_bases, dtype=np.float32)},
        full_signal=full_signal,
        signal_offset=left_softclip,
    )


def test_flag_off_preserves_zero_pad_at_left_edge():
    """Default behavior (flag off): underflowing chunks zero-pad on the left."""
    read = _make_read_with_softclip()
    cfg = ChunkConfig(
        base_justify="start",
        signal_context=(50, 50),
        kmer_context=3,
        recover_softclip_signal=False,
    )
    # base_idx=0 sits at the first aligned base; with signal_context[0]=50
    # the chunk underflows by 50 samples into the soft-clipped region.
    chunk = read.get_chunk(base_idx=0, config=cfg)
    assert chunk is not None
    sig = chunk["signal"]
    # First 50 samples are zero-padded; remaining samples come from the
    # cropped aligned-region signal (values in [0, 1]).
    assert np.all(sig[:50] == 0.0), "expected zero-pad in left underflow when flag is off"
    assert np.all((sig[50:] >= 0.0) & (sig[50:] <= 1.0)), (
        "right half should be aligned-region values"
    )


def test_flag_on_fills_left_edge_from_softclip():
    """With the flag on, the underflowed samples come from the left soft-clip
    region (distinct value range), not zeros."""
    read = _make_read_with_softclip()
    cfg = ChunkConfig(
        base_justify="start",
        signal_context=(50, 50),
        kmer_context=3,
        recover_softclip_signal=True,
    )
    chunk = read.get_chunk(base_idx=0, config=cfg)
    assert chunk is not None
    sig = chunk["signal"]
    # First 50 samples should now be from the left soft-clip region
    # (values in [-3, -2]); aligned region [0, 1] follows.
    assert np.all((sig[:50] >= -3.0) & (sig[:50] <= -2.0)), (
        "expected left-softclip values in underflow region when flag is on"
    )
    assert np.all((sig[50:] >= 0.0) & (sig[50:] <= 1.0)), "aligned-region values must be unchanged"


def test_flag_on_fills_right_edge_from_softclip():
    read = _make_read_with_softclip()
    cfg = ChunkConfig(
        base_justify="end",
        signal_context=(50, 50),
        kmer_context=3,
        recover_softclip_signal=True,
    )
    # Last aligned base (base_idx=19); end-justified chunk overflows past
    # the aligned region by 50 samples into the right soft-clip region.
    chunk = read.get_chunk(base_idx=19, config=cfg)
    assert chunk is not None
    sig = chunk["signal"]
    # Left half should be aligned-region; right half should be right-softclip
    # values (in [2, 3]).
    assert np.all((sig[:50] >= 0.0) & (sig[:50] <= 1.0)), (
        "aligned-region values must precede the overflow"
    )
    assert np.all((sig[50:] >= 2.0) & (sig[50:] <= 3.0)), (
        "expected right-softclip values in overflow region when flag is on"
    )


def test_flag_on_falls_back_to_zero_past_full_signal():
    """If the chunk window extends past even the full signal (i.e. there's
    no soft-clip data available either), we should still zero-pad those
    samples rather than crash."""
    # Only 10 samples of left soft-clip; chunk wants 50 samples of
    # underflow. The first 40 should be zero (past full_signal start), the
    # next 10 should be left-softclip, then aligned region.
    read = _make_read_with_softclip(left_softclip=10)
    cfg = ChunkConfig(
        base_justify="start",
        signal_context=(50, 50),
        kmer_context=3,
        recover_softclip_signal=True,
    )
    chunk = read.get_chunk(base_idx=0, config=cfg)
    assert chunk is not None
    sig = chunk["signal"]
    assert np.all(sig[:40] == 0.0), "samples past full_signal[0] should remain zero"
    assert np.all((sig[40:50] >= -3.0) & (sig[40:50] <= -2.0)), (
        "available soft-clip samples should still be filled"
    )


def test_flag_on_is_no_op_when_chunk_in_bounds():
    """A chunk that fits inside the aligned region must produce the same
    output regardless of the flag (the flag only kicks in on the underflow
    path)."""
    read = _make_read_with_softclip()
    base_idx = 10  # middle of the aligned region
    cfg_off = ChunkConfig(
        base_justify="center",
        signal_context=(20, 20),
        kmer_context=3,
        recover_softclip_signal=False,
    )
    cfg_on = ChunkConfig(
        base_justify="center",
        signal_context=(20, 20),
        kmer_context=3,
        recover_softclip_signal=True,
    )
    chunk_off = read.get_chunk(base_idx, config=cfg_off)
    chunk_on = read.get_chunk(base_idx, config=cfg_on)
    assert chunk_off is not None and chunk_on is not None
    np.testing.assert_array_equal(chunk_off["signal"], chunk_on["signal"])


def test_flag_on_is_no_op_when_full_signal_unavailable():
    """If LeechRead has no full_signal (e.g. basecall-anchored mode), the
    flag must not change behavior — chunks still zero-pad at edges."""
    read = _make_read_with_softclip()
    read.full_signal = None
    read.signal_offset = 0
    cfg = ChunkConfig(
        base_justify="start",
        signal_context=(50, 50),
        kmer_context=3,
        recover_softclip_signal=True,
    )
    chunk = read.get_chunk(base_idx=0, config=cfg)
    assert chunk is not None
    # Without full_signal the underflow falls back to zero-pad.
    assert np.all(chunk["signal"][:50] == 0.0)


@pytest.mark.parametrize("flag", [False, True])
def test_chunk_extraction_does_not_raise_at_extreme_edges(flag):
    """Regression guard: extraction at base_idx=0 and base_idx=num_bases-1
    with both flag settings completes without exception."""
    read = _make_read_with_softclip()
    cfg = ChunkConfig(
        base_justify="center",
        signal_context=(50, 50),
        kmer_context=3,
        recover_softclip_signal=flag,
    )
    assert read.get_chunk(base_idx=0, config=cfg) is not None
    assert read.get_chunk(base_idx=read.num_bases - 1, config=cfg) is not None
