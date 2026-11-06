"""``leech data prepare --spill-dir`` (issue #444).

Extraction is faked (no POD5/BAM): the fake feeds synthetic chunks to the
``chunk_sink`` ``handle_prepare`` hands it, which is the only seam the spill
directory touches.
"""

import tempfile
from pathlib import Path

import pytest
from test_chunk_writer import assert_npz_identical, make_chunks


def _run(tmp_path, monkeypatch, output_dir, **kwargs):
    from leech.commands.prepare import handle_prepare

    def fake_parallel(*, chunk_sink, **_):
        chunks = make_chunks(30)
        for i in range(0, len(chunks), 10):
            chunk_sink(chunks[i : i + 10])
        return [], {
            "total_reads": 30,
            "reads_with_motif": 30,
            "reads_without_motif": 0,
            "total_chunks": len(chunks),
            "failed_reads": 0,
            "failed_batches": 0,
        }

    monkeypatch.setattr("leech.preparation.prepare_training_data_parallel", fake_parallel)
    return handle_prepare(
        pod5=tmp_path / "fake.pod5",
        bam=tmp_path / "fake.bam",
        output_dir=output_dir,
        motif="CCAGGC",
        motif_reference="bam",
        refine_signal_map=False,
        workers=1,
        no_split=True,
        **kwargs,
    )


def test_prepare_spill_dir_output_identical(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, tmp_path / "plain")
    _run(tmp_path, monkeypatch, tmp_path / "spilled", spill_dir=tmp_path / "scratch")
    assert_npz_identical(tmp_path / "plain" / "all.npz", tmp_path / "spilled" / "all.npz")


def test_prepare_spill_dir_keeps_output_dir_clean(tmp_path, monkeypatch):
    spill_dirs = []
    real = tempfile.TemporaryFile

    def recording(*args, **kwargs):
        spill_dirs.append(Path(kwargs["dir"]).resolve())
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryFile", recording)
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    _run(tmp_path, monkeypatch, out, spill_dir=scratch)
    assert spill_dirs, "the spool never opened a spill file"
    assert set(spill_dirs) == {scratch.resolve()}
    assert out.resolve() not in spill_dirs


def test_prepare_spill_dir_created_when_missing(tmp_path, monkeypatch):
    scratch = tmp_path / "a" / "b" / "scratch"
    assert not scratch.exists()
    _run(tmp_path, monkeypatch, tmp_path / "out", spill_dir=scratch)
    assert scratch.is_dir()
    assert (tmp_path / "out" / "all.npz").exists()


def test_check_spill_space_refuses_when_estimate_exceeds_free(tmp_path):
    from leech.commands.prepare import _check_spill_space

    with pytest.raises(RuntimeError, match="free"):
        _check_spill_space(tmp_path, estimated_bytes=2**62)
    _check_spill_space(tmp_path, estimated_bytes=1)
