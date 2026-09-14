"""Focus-relative sequence masking (`--mask-seq-left-of-focus`/`-right-of-focus`).

leech#256: with `seq_encoding: signal_kmer` the sequence branch receives
`sequences_with_kmer_context`, which begins with acceptor-stem bases 5' of a
3'-end motif and therefore identifies a tRNA's body outright -- every model
trained so far had tRNA identity leaked to it through the sequence branch.
`ChunkConfig.mask_seq_side` blanks ('N') sequence characters strictly to one
side of the focus base in both `sequence` (base_onehot) and
`sequence_with_kmer_context` (signal_kmer), reusing the existing "non-ACGT
maps to -1 and is skipped" convention (`sequence_to_int`, `encode_signal_kmer`)
rather than inventing a new one.

These tests pin the exact masking geometry `LeechRead.get_chunk` computes
(``_mask_focus_side``) against hand-derived expected windows -- replicating
the manual per-corpus "write N over the body-side characters" workaround the
issue documents as ground truth -- and confirm the encoders that consume the
masked strings really do treat the masked positions as "no base" (all-zero
one-hot channels), plus the Rust-backend gating that keeps the two prepare/
predict backends from silently disagreeing about a feature Rust does not
implement.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from leech.chunking import LeechRead
from leech.configs import ChunkConfig
from leech.features import encode_signal_kmer, sequence_to_int

# Matches leech.constants.DEFAULT_SIGNAL_KMER_CONTEXT -- the fixed kmer_before/
# kmer_after `get_chunk` always uses to build sequence_with_kmer_context.
_KMER_BEFORE, _KMER_AFTER = 4, 4


def _make_uniform_read(num_bases: int = 40, samples_per_base: int = 10) -> LeechRead:
    """A LeechRead with an evenly-spaced base-to-signal map and a sequence
    whose every position is distinguishable (so masking is easy to spot).
    """
    seq_to_sig_map = np.arange(
        0, (num_bases + 1) * samples_per_base, samples_per_base, dtype=np.int64
    )
    dwells = np.diff(seq_to_sig_map)
    signal = np.zeros(num_bases * samples_per_base, dtype=np.float32)
    sequence = "ACGT" * (num_bases // 4)
    assert len(sequence) == num_bases
    return LeechRead(
        read_id="mask_test",
        sequence=sequence,
        signal=signal,
        seq_to_sig_map=seq_to_sig_map,
        dwells=dwells,
        dwell_features={"dwell": dwells.astype(np.float32)},
        signal_features={"level_mean": np.zeros(num_bases, dtype=np.float32)},
    )


# ---------------------------------------------------------------------------
# Geometry: the "sequence" (base_onehot) k-mer window is always centered
# exactly on the focus base, independent of base_justify.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_justify", ["start", "center", "end"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_base_onehot_window_masks_around_center_focus(base_justify, side):
    read = _make_uniform_read()
    base_idx = 20
    kmer_context = 5
    cfg_plain = ChunkConfig(
        base_justify=base_justify, signal_context=(50, 50), kmer_context=kmer_context
    )
    cfg_masked = ChunkConfig(
        base_justify=base_justify,
        signal_context=(50, 50),
        kmer_context=kmer_context,
        mask_seq_side=side,
    )
    plain = read.get_chunk(base_idx, config=cfg_plain)
    masked = read.get_chunk(base_idx, config=cfg_masked)
    assert plain is not None and masked is not None

    unmasked_seq = plain["sequence"]
    masked_seq = masked["sequence"]
    assert len(unmasked_seq) == 2 * kmer_context + 1
    focus_idx = kmer_context  # always centered, regardless of base_justify

    # Focus base itself is never masked.
    assert masked_seq[focus_idx] == unmasked_seq[focus_idx]

    if side == "left":
        assert masked_seq[:focus_idx] == "N" * focus_idx
        assert masked_seq[focus_idx:] == unmasked_seq[focus_idx:]
    else:
        assert masked_seq[focus_idx + 1 :] == "N" * (len(unmasked_seq) - focus_idx - 1)
        assert masked_seq[: focus_idx + 1] == unmasked_seq[: focus_idx + 1]


# ---------------------------------------------------------------------------
# Geometry: sequence_with_kmer_context's focus position moves with
# base_justify (it is anchored to the SIGNAL window, not the base window) --
# pinned against the hand-derived formula from the issue's own workaround.
# ---------------------------------------------------------------------------


def test_signal_kmer_window_matches_hand_derived_geometry_end_justify():
    """Replicates the issue's own worked example: with base_justify="end",
    the focus is core index f (seq_to_sig[f+1] == focus_signal_pos) and the
    body-side characters are sequence_with_kmer_context[0 : f+kmer_before).
    """
    read = _make_uniform_read(num_bases=40, samples_per_base=10)
    base_idx = 20
    cfg = ChunkConfig(base_justify="end", signal_context=(50, 50), kmer_context=5)
    plain = read.get_chunk(base_idx, config=cfg)
    assert plain is not None

    # Hand-derived from the uniform geometry (10 samples/base, signal_context
    # (50, 50) = 5 bases either side): seq_start = base_idx - 4, so the focus
    # base's core-relative index is f = base_idx - seq_start = 4.
    seq_start = base_idx - 4
    seq_end = base_idx + 6
    f = base_idx - seq_start
    assert f == 4
    ext = plain["sequence_with_kmer_context"]
    assert len(ext) == (seq_end - seq_start) + _KMER_BEFORE + _KMER_AFTER  # core_len + 8
    focus_ext_idx = f + _KMER_BEFORE
    assert focus_ext_idx == 8

    masked = read.get_chunk(
        base_idx,
        config=ChunkConfig(
            base_justify="end", signal_context=(50, 50), kmer_context=5, mask_seq_side="left"
        ),
    )
    assert masked is not None
    masked_ext = masked["sequence_with_kmer_context"]
    assert masked_ext[:focus_ext_idx] == "N" * focus_ext_idx
    assert masked_ext[focus_ext_idx:] == ext[focus_ext_idx:]


@pytest.mark.parametrize("base_justify", ["start", "center", "end"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_signal_kmer_window_focus_char_never_masked(base_justify, side):
    read = _make_uniform_read()
    base_idx = 20
    cfg_plain = ChunkConfig(base_justify=base_justify, signal_context=(50, 50), kmer_context=5)
    cfg_masked = ChunkConfig(
        base_justify=base_justify, signal_context=(50, 50), kmer_context=5, mask_seq_side=side
    )
    plain = read.get_chunk(base_idx, config=cfg_plain)
    masked = read.get_chunk(base_idx, config=cfg_masked)
    assert plain is not None and masked is not None

    ext = plain["sequence_with_kmer_context"]
    masked_ext = masked["sequence_with_kmer_context"]
    assert len(ext) == len(masked_ext)

    # However the focus offset is derived, masking never touches it, and
    # every character strictly on the *other* side from the requested one is
    # left completely untouched -- so a diff of the two strings identifies
    # a single contiguous run that starts (side="left") or ends (side="right")
    # at the string boundary and does not swallow the whole string.
    diff = [i for i in range(len(ext)) if ext[i] != masked_ext[i]]
    assert diff, "masking should change at least one character for a non-trivial window"
    assert all(masked_ext[i] == "N" for i in diff)
    if side == "left":
        assert diff == list(range(diff[0], diff[-1] + 1))
        assert diff[0] == 0
        assert diff[-1] < len(ext) - 1  # never the whole string -- focus survives
    else:
        assert diff == list(range(diff[0], diff[-1] + 1))
        assert diff[-1] == len(ext) - 1
        assert diff[0] > 0


def test_mask_seq_side_none_is_a_no_op():
    read = _make_uniform_read()
    cfg_default = ChunkConfig(base_justify="end", signal_context=(50, 50), kmer_context=5)
    cfg_explicit_none = ChunkConfig(
        base_justify="end", signal_context=(50, 50), kmer_context=5, mask_seq_side=None
    )
    a = read.get_chunk(20, config=cfg_default)
    b = read.get_chunk(20, config=cfg_explicit_none)
    assert a["sequence"] == b["sequence"]
    assert a["sequence_with_kmer_context"] == b["sequence_with_kmer_context"]


# ---------------------------------------------------------------------------
# Downstream: masked positions really do carry no signal into either
# encoding -- the whole point of leech#256.
# ---------------------------------------------------------------------------


def test_masked_positions_are_all_zero_in_signal_kmer_encoding():
    """The "own base identity" channel -- kmer_pos == kmer_before, i.e. the
    center of each signal sample's k-mer context, which is exactly the base
    occupying that sample -- must be all zero for every core base whose
    character was masked, and must NOT be all zero for the focus base itself
    (a regression sanity check: an over-broad mask that also blanked the
    focus would pass every other assertion here).

    Every other (kmer_pos, channel) slot may legitimately be non-zero even
    over a masked base's own signal span -- `encode_signal_kmer` gives every
    signal sample the FULL k-mer window around it by design, so a masked
    (body) base's span can still show unmasked, real bases in its
    forward-looking context channels (e.g. the focus base itself, a few
    samples early). That is not a body-identity leak: leaking the tRNA body
    would mean some channel reveals what a masked position's *own* base was,
    which this test is the one that catches.
    """
    read = _make_uniform_read()
    base_idx = 20
    signal_len = 100
    kmer_context = (4, 4)
    cfg = ChunkConfig(
        base_justify="end", signal_context=(50, 50), kmer_context=5, mask_seq_side="left"
    )
    chunk = read.get_chunk(base_idx, config=cfg)
    assert chunk is not None

    seq_ints = sequence_to_int(chunk["sequence_with_kmer_context"])
    seq_to_sig = chunk["seq_to_sig_map"].astype(np.int64)
    enc = encode_signal_kmer(seq_ints, seq_to_sig, signal_len, kmer_context)
    kmer_len = 9
    assert enc.shape == (4 * kmer_len, signal_len)

    # Every masked base's sequence_to_int value is -1 (the "N" sentinel).
    focus_ext_idx = 8  # from the hand-derived geometry above (base_justify="end")
    assert np.all(seq_ints[:focus_ext_idx] == -1)
    assert seq_ints[focus_ext_idx] >= 0

    enc_by_kmer_pos = enc.reshape(kmer_len, 4, signal_len)
    center = _KMER_BEFORE  # kmer_pos whose channel is the base's own identity
    core_focus_idx = focus_ext_idx - _KMER_BEFORE  # == 4
    for core_i in range(core_focus_idx):  # masked core bases: 0, 1, 2, 3
        sig_start, sig_end = int(seq_to_sig[core_i]), int(seq_to_sig[core_i + 1])
        assert np.all(enc_by_kmer_pos[center, :, sig_start:sig_end] == 0.0), (
            f"core base {core_i} was masked but its own-identity channel fired"
        )
    # The focus base's own-identity channel is unmasked: exactly one 1.0 per
    # signal sample across its span.
    f_start, f_end = int(seq_to_sig[core_focus_idx]), int(seq_to_sig[core_focus_idx + 1])
    assert f_end > f_start
    assert np.all(enc_by_kmer_pos[center, :, f_start:f_end].sum(axis=0) == 1.0)


def test_masked_positions_are_all_zero_in_base_onehot_encoding():
    """Mirrors dataset.py's `_encode_sequence`: non-ACGT ('N') bytes map to
    an all-zero one-hot column, which is exactly what masking relies on for
    the base_onehot branch too.
    """
    read = _make_uniform_read()
    cfg = ChunkConfig(
        base_justify="end", signal_context=(50, 50), kmer_context=5, mask_seq_side="left"
    )
    chunk = read.get_chunk(20, config=cfg)
    assert chunk is not None
    seq_ints = sequence_to_int(chunk["sequence"])
    assert np.all(seq_ints[:5] == -1)  # masked left half (focus at index 5)
    assert np.all(seq_ints[5:] >= 0)  # focus + right half untouched


# ---------------------------------------------------------------------------
# Rust-backend gating: masking is Python-only, and both prepare and predict
# must fall back (or refuse) rather than silently ship an unmasked corpus /
# feed a live prediction the real, unmasked bases.
# ---------------------------------------------------------------------------


def test_rust_prepare_unsupported_reason_flags_mask_seq_side():
    from leech.configs import LabelConfig, MotifConfig, PrepareConfig, SignalConfig
    from leech.preparation.parallel import rust_prepare_unsupported_reason

    base_config = PrepareConfig(
        pod5_path="unused.pod5",
        signal=SignalConfig(),
        motif=MotifConfig(),
        chunk=ChunkConfig(),
        labeling=LabelConfig(),
    )
    assert rust_prepare_unsupported_reason(base_config) is None

    masked_config = PrepareConfig(
        pod5_path="unused.pod5",
        signal=SignalConfig(),
        motif=MotifConfig(),
        chunk=ChunkConfig(mask_seq_side="left"),
        labeling=LabelConfig(),
    )
    reason = rust_prepare_unsupported_reason(masked_config)
    assert reason is not None
    assert "mask_seq_side" in reason


def test_check_rust_extraction_available_gates_mask_seq_side():
    from leech._rust_accel import HAS_RUST
    from leech.inference.helpers import check_rust_extraction_available

    # Unmasked: rust gating is unaffected (may or may not be available on
    # this machine, but must not raise and must not be refused for masking).
    check_rust_extraction_available("auto", mask_seq_side=None)

    # Masked + backend=auto: never uses rust, regardless of availability.
    use_rust_masked, *_ = check_rust_extraction_available("auto", mask_seq_side="left")
    assert use_rust_masked is False

    if HAS_RUST:
        # Masked + backend=rust: refuses outright rather than silently
        # degrading -- only reachable when there is a Rust path to refuse;
        # otherwise "leech_core is not installed" fires first, which is
        # covered by the pre-existing --backend rust-without-Rust behavior.
        with pytest.raises(RuntimeError, match="mask_seq_side"):
            check_rust_extraction_available("rust", mask_seq_side="right")
    else:
        with pytest.raises(RuntimeError):
            check_rust_extraction_available("rust", mask_seq_side="right")


# ---------------------------------------------------------------------------
# Config round trip: prepare records it, train carries it forward, predict
# reads it back -- the "decide once, record it" contract this repo holds
# seq_encoding to (leech#230) applies here too.
# ---------------------------------------------------------------------------


def test_prepare_config_to_dict_records_mask_seq_side():
    from leech.configs import LabelConfig, MotifConfig, PrepareConfig, SignalConfig

    config = PrepareConfig(
        pod5_path="unused.pod5",
        signal=SignalConfig(),
        motif=MotifConfig(),
        chunk=ChunkConfig(mask_seq_side="right"),
        labeling=LabelConfig(),
    )
    assert config.to_dict()["mask_seq_side"] == "right"

    config_none = PrepareConfig(
        pod5_path="unused.pod5",
        signal=SignalConfig(),
        motif=MotifConfig(),
        chunk=ChunkConfig(),
        labeling=LabelConfig(),
    )
    assert config_none.to_dict()["mask_seq_side"] is None


def test_inference_spec_reads_mask_seq_side_from_model_config():
    from leech.inference.helpers import InferenceSpec

    config = {
        "motif": "CCAGGC",
        "signal_len": 100,
        "kmer_len": 11,
        "mask_seq_side": "left",
    }
    spec = InferenceSpec.from_config(config)
    assert spec.mask_seq_side == "left"

    config_absent = dict(config)
    del config_absent["mask_seq_side"]
    spec_absent = InferenceSpec.from_config(config_absent)
    assert spec_absent.mask_seq_side is None


# ---------------------------------------------------------------------------
# `data merge` must not silently combine differently-masked corpora --
# `_propagate_prepare_config`'s critical_keys check exists precisely to
# refuse (loudly) exactly the kind of geometry mismatch this is (issue #189
# for the general mechanism; leech#256 for mask_seq_side specifically).
# ---------------------------------------------------------------------------


def _write_sidecar(directory, mask_seq_side) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prepare_config.json").write_text(json.dumps({"mask_seq_side": mask_seq_side}))


@pytest.mark.parametrize("side_a,side_b", [("left", "right"), ("left", None), (None, "right")])
def test_merge_warns_on_mask_seq_side_mismatch(tmp_path, caplog, side_a, side_b):
    from leech.commands.merge_split import _propagate_prepare_config

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _write_sidecar(dir_a, side_a)
    _write_sidecar(dir_b, side_b)

    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="leech.commands.merge_split"):
        _propagate_prepare_config([dir_a / "train.npz", dir_b / "train.npz"], out_dir)

    assert any("mask_seq_side" in rec.message for rec in caplog.records), (
        "merging corpora with different mask_seq_side must warn -- silently "
        "combining them reintroduces the tRNA-body identity leak at the "
        "merge step instead of prepare"
    )
    written = json.loads((out_dir / "prepare_config.json").read_text())
    assert written["mask_seq_side"] == side_a  # documented "first input wins" behavior


def test_merge_silent_when_mask_seq_side_matches(tmp_path, caplog):
    from leech.commands.merge_split import _propagate_prepare_config

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _write_sidecar(dir_a, "left")
    _write_sidecar(dir_b, "left")

    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="leech.commands.merge_split"):
        _propagate_prepare_config([dir_a / "train.npz", dir_b / "train.npz"], out_dir)

    assert not any("mask_seq_side" in rec.message for rec in caplog.records)
    written = json.loads((out_dir / "prepare_config.json").read_text())
    assert written["mask_seq_side"] == "left"
