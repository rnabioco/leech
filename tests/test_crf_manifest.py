"""The manifest is the seam between a corpus's vocabulary and its signal.

Every check here corresponds to a way a CRF corpus has silently come out wrong
in practice: a missing column that surfaced an hour into a run, a partially
scored quality table that cut the corpus to a non-random 13.5%, a window too
short to hold its target, and a target compared at full length when the model
can only emit part of it.
"""

from __future__ import annotations

import polars as pl
import pytest

from leech.crf import (
    REQUIRED_COLUMNS,
    CrfManifest,
    check_geometry,
    emitted_target,
    load_manifest,
)

TARGET = "ACGTACGTACGTACGTACGT"


def _frame(n: int = 4, **overrides) -> pl.DataFrame:
    data = {
        "read_id": [f"read{i}" for i in range(n)],
        "pod5": ["reads.pod5"] * n,
        "anchor_end": [3000] * n,
        "target": [TARGET] * n,
    }
    data.update(overrides)
    return pl.DataFrame(data)


def _write(tmp_path, frame: pl.DataFrame, name="manifest.parquet"):
    path = tmp_path / name
    if path.suffix == ".parquet":
        frame.write_parquet(path)
    else:
        frame.write_csv(path, separator="\t" if path.suffix == ".tsv" else ",")
    return path


# ── the emission rule ──────────────────────────────────────────────────────


def test_emitted_target_drops_the_first_state_len_bases():
    assert emitted_target("AACCGGTT", 4) == "GGTT"


@pytest.mark.parametrize("state_len", [0, 1, 4, 8])
def test_emitted_length_is_target_minus_state_len(state_len):
    """The rule that does not change with window width."""
    assert len(emitted_target(TARGET, state_len)) == len(TARGET) - state_len


def test_emitted_target_rejects_a_negative_state_len():
    with pytest.raises(ValueError, match="state_len"):
        emitted_target(TARGET, -1)


# ── geometry ───────────────────────────────────────────────────────────────


def test_window_that_holds_the_target_is_accepted():
    check_geometry(window=3000, target_len=48, samples_per_base=56.0)


def test_window_too_short_for_the_target_is_refused():
    """Raises, not warns: a short window trains fine and is quietly wrong.

    This is how a 27-nt barcode came to be discriminated on 23 nt — the run
    converged and reported nothing unusual.
    """
    with pytest.raises(ValueError, match="cannot hold"):
        check_geometry(window=2000, target_len=48, samples_per_base=56.0)


def test_target_shorter_than_state_len_emits_nothing():
    with pytest.raises(ValueError, match="emits nothing"):
        check_geometry(window=100_000, target_len=4, samples_per_base=56.0)


def test_the_error_says_what_the_target_would_emit():
    """A message that only says "too short" invites widening the window, which
    is exactly the fix that recovers nothing."""
    with pytest.raises(ValueError) as exc:
        check_geometry(window=100, target_len=48, samples_per_base=56.0)
    assert "44" in str(exc.value)  # 48 - 4


# ── loading and validation ─────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["manifest.parquet", "manifest.tsv", "manifest.csv"])
def test_round_trips_every_supported_format(tmp_path, name):
    man = load_manifest(_write(tmp_path, _frame(), name))
    assert len(man) == 4
    assert set(REQUIRED_COLUMNS) <= set(man.columns)


def test_missing_required_column_names_it(tmp_path):
    path = _write(tmp_path, _frame().drop("target"))
    with pytest.raises(ValueError, match="missing required column"):
        load_manifest(path)


def test_missing_target_column_points_at_the_seam(tmp_path):
    """The common porting mistake is a table carrying a class name instead of a
    resolved sequence. The error has to say where that lookup belongs."""
    path = _write(tmp_path, _frame().drop("target"))
    with pytest.raises(ValueError, match="belongs"):
        load_manifest(path)


def test_absent_optional_column_is_fine(tmp_path):
    man = load_manifest(_write(tmp_path, _frame()))
    assert not man.has("batch")
    assert man.batches() == []


def test_required_optional_column_fails_up_front(tmp_path):
    """`require=` exists so a holdout run fails at load, not an hour in."""
    path = _write(tmp_path, _frame())
    with pytest.raises(ValueError, match="batch"):
        load_manifest(path, require=("batch",))


def test_duplicate_read_ids_are_refused(tmp_path):
    frame = _frame(2, read_id=["r0", "r0"])
    with pytest.raises(ValueError, match="duplicate read_id"):
        load_manifest(_write(tmp_path, frame))


def test_empty_manifest_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_manifest(_write(tmp_path, _frame(0)))


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.parquet")


def test_unknown_format_is_refused(tmp_path):
    path = tmp_path / "manifest.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="unrecognised manifest format"):
        load_manifest(path)


# ── quality coverage ───────────────────────────────────────────────────────


def test_full_quality_coverage(tmp_path):
    frame = _frame(4, quality_score=[70.0, 80.0, 90.0, 100.0])
    assert load_manifest(_write(tmp_path, frame)).quality_coverage() == 1.0


def test_partial_quality_coverage_is_reported(tmp_path):
    """An unscored read cannot pass a gate, so it is dropped silently — a
    partially scored manifest trains on a small non-random subset."""
    frame = _frame(4, quality_score=[70.0, None, None, None])
    assert load_manifest(_write(tmp_path, frame)).quality_coverage() == 0.25


def test_no_quality_column_means_full_coverage(tmp_path):
    """Nothing is gated, so nothing is lost — 0.0 would read as total failure."""
    assert load_manifest(_write(tmp_path, _frame())).quality_coverage() == 1.0


# ── convenience accessors ──────────────────────────────────────────────────


def test_batches_are_sorted_and_deduplicated(tmp_path):
    frame = _frame(4, batch=["b2", "b1", "b2", None])
    assert load_manifest(_write(tmp_path, frame)).batches() == ["b1", "b2"]


def test_target_lengths_surfaces_a_ragged_corpus(tmp_path):
    frame = _frame(2, target=[TARGET, TARGET[:-3]])
    assert load_manifest(_write(tmp_path, frame)).target_lengths() == {20, 17}


def test_manifest_carries_its_own_path(tmp_path):
    path = _write(tmp_path, _frame())
    assert load_manifest(path).path == path


def test_manifest_is_constructible_without_a_path():
    """In-memory manifests are legal; `path` is for error messages."""
    assert len(CrfManifest(frame=_frame())) == 4
