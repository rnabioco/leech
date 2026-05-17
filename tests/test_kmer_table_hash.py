"""Provenance fingerprint for the kmer level table.

R3 from the coordinate-positioning audit: signal-map refinement uses a
bundled kmer level table that ``leech`` loads via
``leech.data.get_kmer_table()``. Nothing in the train->inference handoff
records *which* table was used, so a ``leech`` upgrade that ships a
revised table will silently change refinement output at predict time.

These tests pin (a) the hash helper itself, (b) that the hash is written
into ``PrepareConfig.to_dict()``, and (c) that the inference-side warning
fires on drift.
"""

from __future__ import annotations

import logging
from pathlib import Path

from leech.configs import (
    ChunkConfig,
    MotifConfig,
    PrepareConfig,
    SignalConfig,
)
from leech.data import compute_kmer_table_sha256, get_kmer_table
from leech.inference.helpers import _warn_if_kmer_table_drifted


def test_compute_kmer_table_sha256_is_deterministic():
    """Same file -> same hash; hash matches stdlib hashlib output."""
    path = get_kmer_table()
    h1 = compute_kmer_table_sha256(path)
    h2 = compute_kmer_table_sha256(path)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex digest
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_kmer_table_sha256_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"hello world\n")
    f2.write_bytes(b"hello world!\n")
    assert compute_kmer_table_sha256(f1) != compute_kmer_table_sha256(f2)


def test_prepare_config_emits_kmer_table_sha256_when_refining():
    prep = PrepareConfig(
        pod5_path=Path("/dummy.pod5"),
        signal=SignalConfig(
            refine_signal_map=True,
            kmer_table_path=get_kmer_table(),
        ),
        motif=MotifConfig(motif="CCAGGC", motif_offset=2),
        chunk=ChunkConfig(),
    )
    d = prep.to_dict()
    assert "kmer_table_sha256" in d
    assert d["kmer_table_sha256"] == compute_kmer_table_sha256(get_kmer_table())


def test_prepare_config_omits_hash_when_refinement_disabled():
    """No refinement -> no hash, even if a path is set. Avoids fingerprinting
    an unused file (and avoids the I/O cost during prep serialization)."""
    prep = PrepareConfig(
        pod5_path=Path("/dummy.pod5"),
        signal=SignalConfig(
            refine_signal_map=False,
            kmer_table_path=get_kmer_table(),
        ),
        motif=MotifConfig(motif="CCAGGC", motif_offset=2),
        chunk=ChunkConfig(),
    )
    d = prep.to_dict()
    assert d["kmer_table_sha256"] is None


def test_prepare_config_omits_hash_when_path_unset():
    """Refinement enabled but no path captured -> emit None rather than crash."""
    prep = PrepareConfig(
        pod5_path=Path("/dummy.pod5"),
        signal=SignalConfig(
            refine_signal_map=True,
            kmer_table_path=None,
        ),
        motif=MotifConfig(motif="CCAGGC", motif_offset=2),
        chunk=ChunkConfig(),
    )
    d = prep.to_dict()
    assert d["kmer_table_sha256"] is None


def test_warn_if_kmer_table_drifted_no_op_on_match(caplog):
    """Matching hash -> no warning. The warning is reserved for the actual
    drift case so it stays visible when it matters."""
    path = get_kmer_table()
    matching_sha = compute_kmer_table_sha256(path)
    with caplog.at_level(logging.WARNING, logger="leech.inference"):
        _warn_if_kmer_table_drifted(matching_sha, path)
    assert not any("does not match" in r.message for r in caplog.records)


def test_warn_if_kmer_table_drifted_no_op_on_legacy_config(caplog):
    """Legacy models (trained before R3) have no sha256 in config -> skip
    the check rather than emitting a noisy warning on every predict run."""
    path = get_kmer_table()
    with caplog.at_level(logging.WARNING, logger="leech.inference"):
        _warn_if_kmer_table_drifted(None, path)
    assert not any("does not match" in r.message for r in caplog.records)


def test_warn_if_kmer_table_drifted_fires_on_mismatch(caplog):
    """A non-matching stored hash -> warning logged. This is the regression
    guard for the actual silent-drift scenario the field exists to catch."""
    path = get_kmer_table()
    bogus_sha = "0" * 64
    with caplog.at_level(logging.WARNING, logger="leech.inference"):
        _warn_if_kmer_table_drifted(bogus_sha, path)
    assert any("does not match" in r.message for r in caplog.records), (
        "expected drift warning when stored sha256 differs from live table"
    )
