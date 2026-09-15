"""
Tests for how ``prepare`` dispatches batches to its backends.

These are regression guards for issue #176, where the Rust prepare path was
driven from a serial ``for`` loop. That left one POD5 read outstanding at a
time and made the "accelerated" backend ~10-80x slower than the multiprocessing
fallback on a large POD5. The per-batch work is stubbed out here — what is
under test is the driver, not the pipeline.

They need no POD5/BAM fixtures, which is deliberate: the property they protect
is easy to break and should be checked everywhere the suite runs.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from leech.preparation import parallel as par


def _drive(monkeypatch, work, *, num_workers, n_batches, batch_size=2):
    """Run ``_iter_rust_batches`` over ``n_batches`` stub batches.

    ``work`` stands in for ``_prepare_batch_rust_with_failures`` and must
    return its ``(chunks, n_failed_reads, n_submitted, n_missing_from_pod5)``
    shape -- see :class:`leech.preparation.parallel.BatchOutcome` for what the
    driver does with each field.
    """
    batches = [[object()] * batch_size for _ in range(n_batches)]
    monkeypatch.setattr(par, "iter_read_info_batches", lambda *a, **k: iter(batches))
    monkeypatch.setattr(par, "_prepare_batch_rust_with_failures", work)

    return list(
        par._iter_rust_batches(
            bam_path=None,
            config=None,
            motif_searcher=None,
            chunk_size=batch_size,
            min_mapq=0,
            num_workers=num_workers,
        )
    )


class TestRustBatchDispatchIsConcurrent:
    def test_batches_overlap(self, monkeypatch):
        """Several batches run at once. A serial driver cannot pass this."""
        lock = threading.Lock()
        live = 0
        peak = 0
        release = threading.Event()

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            # Block until the driver has had the chance to start others. A
            # serial driver never reaches 4 in flight and the watcher gives up.
            release.wait(timeout=20.0)
            with lock:
                live -= 1
            return [{"read_id": "x"}], 0, 1, 0

        def watcher():
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                with lock:
                    if peak >= 4:
                        break
                time.sleep(0.005)
            release.set()

        t = threading.Thread(target=watcher, daemon=True)
        t.start()
        results = _drive(monkeypatch, work, num_workers=4, n_batches=8)
        t.join(timeout=25.0)

        assert peak >= 4, f"only {peak} batch(es) ever overlapped -- the driver is serial"
        assert len(results) == 8

    def test_yields_in_bam_order_with_read_counts(self, monkeypatch):
        """Results arrive in submission order, each tagged with its read count."""

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            return [{"n": len(read_batch)}], 0, 1, 0

        results = _drive(monkeypatch, work, num_workers=4, n_batches=5, batch_size=3)

        assert [outcome.n_reads for outcome in results] == [3] * 5
        assert [outcome.chunks[0]["n"] for outcome in results] == [3] * 5

    def test_failed_batch_is_flagged_not_silently_absorbed(self, monkeypatch):
        """Issue #265: a bad batch must not end the run by itself (the
        dispatcher stays a generator; the caller decides what's fatal), but
        it also must not come back looking like a clean, motif-less batch.
        """
        calls = {"n": 0}
        lock = threading.Lock()

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            with lock:
                calls["n"] += 1
                n = calls["n"]
            if n == 2:
                raise RuntimeError("boom")
            return [{"ok": True}], 0, 1, 0

        results = _drive(monkeypatch, work, num_workers=2, n_batches=4)

        assert len(results) == 4
        failed = [r for r in results if r.batch_failed]
        ok = [r for r in results if not r.batch_failed]
        assert len(failed) == 1, "exactly one of the four batches was made to fail"
        assert len(ok) == 3
        # The failed batch reports its reads as failed rather than "no
        # motif" -- that is the whole point of the flag.
        (bad,) = failed
        assert bad.n_reads == 2
        assert bad.n_failed_reads == bad.n_reads
        assert bad.chunks == []
        # The other batches are unaffected: still 1 chunk, no failures.
        assert all(r.n_failed_reads == 0 for r in ok)
        assert sum(len(r.chunks) for r in ok) == 3

    def test_zero_yield_despite_submitted_reads_counts_as_failed(self, monkeypatch):
        """A batch that fed real reads into Rust (a motif match found) and
        got nothing back, with no exception, is the closest signal a Python
        driver has to "Rust silently dropped every read" -- it must not read
        as a clean no-motif batch either."""

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            return [], 0, 2, 0  # 2 found in the POD5, Rust returned nothing

        (outcome,) = _drive(monkeypatch, work, num_workers=1, n_batches=1)

        assert not outcome.batch_failed
        assert outcome.chunks == []
        assert outcome.n_failed_reads == 2

    def test_genuinely_motifless_batch_is_not_counted_as_failed(self, monkeypatch):
        """The other half of the same corner: nothing submitted (no read in
        the batch had a motif match) is a legitimate, non-failure outcome."""

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            return [], 0, 0, 0  # nothing submitted -- no motif anywhere in the batch

        (outcome,) = _drive(monkeypatch, work, num_workers=1, n_batches=1)

        assert not outcome.batch_failed
        assert outcome.chunks == []
        assert outcome.n_failed_reads == 0

    def test_batch_wholly_absent_from_the_pod5_is_not_a_failure(self, monkeypatch):
        """Issue #325: the third way a batch yields nothing.

        Every submitted read was simply not in this POD5 -- the expected
        shape when the POD5 was pre-filtered to a subset of the BAM's reads.
        Indistinguishable from the #267/#258 signature above without the
        per-call count Rust now returns, and it happens on EVERY batch of
        such a run, so treating it as failure aborts the whole thing.
        """

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            return [], 0, 3, 3  # 3 submitted, all 3 absent from the POD5

        (outcome,) = _drive(monkeypatch, work, num_workers=1, n_batches=1)

        assert not outcome.batch_failed
        assert outcome.chunks == []
        assert outcome.n_failed_reads == 0
        assert outcome.n_reads_missing_from_pod5 == 3

    def test_zero_yield_still_fails_for_the_reads_that_were_found(self, monkeypatch):
        """The #267/#258 guard survives partial POD5 coverage.

        One read of three was absent; the other two WERE found and still
        produced nothing, which is the genuine zero-output failure. Only
        those two count.
        """

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            return [], 0, 3, 1

        (outcome,) = _drive(monkeypatch, work, num_workers=1, n_batches=1)

        assert not outcome.batch_failed
        assert outcome.n_failed_reads == 2
        assert outcome.n_reads_missing_from_pod5 == 1

    def test_absent_reads_are_reported_even_when_chunks_come_back(self, monkeypatch):
        """A productive batch still reports what the POD5 was missing."""

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            return [{"read_id": "x"}], 0, 3, 2

        (outcome,) = _drive(monkeypatch, work, num_workers=1, n_batches=1)

        assert outcome.n_failed_reads == 0
        assert outcome.n_reads_missing_from_pod5 == 2

    def test_in_flight_window_is_bounded(self, monkeypatch):
        """The driver must not pull the whole BAM into memory up front."""
        num_workers = 3
        window = 2 * num_workers

        started = threading.Semaphore(0)
        proceed = threading.Event()
        consumed = {"n": 0}

        def work(read_batch, config, motif_searcher, kmer_levels=None):
            started.release()
            proceed.wait(timeout=20.0)
            return [], 0, 0, 0

        def counting_batches(*a, **k):
            for _ in range(500):
                consumed["n"] += 1
                yield [object(), object()]

        monkeypatch.setattr(par, "iter_read_info_batches", counting_batches)
        monkeypatch.setattr(par, "_prepare_batch_rust_with_failures", work)

        gen = par._iter_rust_batches(
            bam_path=None,
            config=None,
            motif_searcher=None,
            chunk_size=2,
            min_mapq=0,
            num_workers=num_workers,
        )
        drained = threading.Thread(target=lambda: list(gen), daemon=True)
        drained.start()
        try:
            # Once every worker is busy the window is full; the driver should
            # then stop pulling from the BAM until something completes.
            for _ in range(num_workers):
                assert started.acquire(timeout=20.0)
            time.sleep(0.25)
            assert consumed["n"] <= window + 1, (
                f"pulled {consumed['n']} batches from the BAM with a window of {window}"
            )
        finally:
            proceed.set()
            drained.join(timeout=25.0)


class TestThroughputMonitor:
    """The rate line that would have caught #176 in the first minute."""

    def test_reports_reads_per_second(self):
        monitor = par._ThroughputMonitor("Rust (rayon)")
        monitor.start = time.monotonic() - 10.0
        assert monitor.reads_per_second(1000) == pytest.approx(100.0, rel=0.05)

    def test_zero_elapsed_does_not_divide_by_zero(self):
        monitor = par._ThroughputMonitor("Python (multiprocessing)")
        monitor.start = time.monotonic()
        assert monitor.reads_per_second(0) == 0.0
        assert monitor.reads_per_second(10) > 0.0

    def test_progress_line_names_the_backend_and_the_rate(self, caplog):
        monitor = par._ThroughputMonitor("Rust (rayon)")
        monitor.start = time.monotonic() - 2.0
        with caplog.at_level("INFO", logger="leech.preparation.parallel"):
            monitor.log_progress(batches=5, total_reads=200, total_chunks=180)
        (record,) = caplog.records
        assert "Rust (rayon)" in record.message
        assert "reads/s" in record.message


class TestBackendSelection:
    """``--backend`` on ``data prepare`` (issue #177).

    It replaced the ``LEECH_DISABLE_RUST`` environment variable, which killed
    the ``leech_core`` import process-wide -- far broader than the one step it
    was meant to switch, and invisible to anything but a grep. Forcing a
    backend is a measurement tool: both produce identical chunks
    (``test_backend_parity.py``), so the only thing it changes is throughput.
    """

    @staticmethod
    def _config(monkeypatch, *, reason=None, available=True):
        monkeypatch.setattr(par, "rust_prepare_unsupported_reason", lambda cfg: reason)
        monkeypatch.setattr(par, "HAS_RUST", available)
        monkeypatch.setattr(par, "_rs_extract_training_chunks", object() if available else None)
        return object()

    def test_auto_takes_rust_when_it_can_serve(self, monkeypatch):
        cfg = self._config(monkeypatch)
        assert par._select_prepare_backend(cfg, "auto") is True

    def test_auto_falls_back_when_config_unsupported(self, monkeypatch):
        cfg = self._config(monkeypatch, reason="focus_map is set")
        assert par._select_prepare_backend(cfg, "auto") is False

    def test_auto_falls_back_when_unavailable(self, monkeypatch):
        cfg = self._config(monkeypatch, available=False)
        assert par._select_prepare_backend(cfg, "auto") is False

    def test_python_forces_the_pool_even_when_rust_is_ready(self, monkeypatch):
        cfg = self._config(monkeypatch)
        assert par._select_prepare_backend(cfg, "python") is False

    def test_rust_raises_rather_than_falling_back(self, monkeypatch):
        """A forced run that quietly took the other path measures nothing."""
        cfg = self._config(monkeypatch, reason="focus_map is set")
        with pytest.raises(RuntimeError, match="cannot serve this config"):
            par._select_prepare_backend(cfg, "rust")

    def test_rust_raises_when_leech_core_is_missing(self, monkeypatch):
        cfg = self._config(monkeypatch, available=False)
        with pytest.raises(RuntimeError, match="not importable"):
            par._select_prepare_backend(cfg, "rust")

    def test_unknown_choice_is_rejected(self, monkeypatch):
        cfg = self._config(monkeypatch)
        with pytest.raises(ValueError, match="unknown backend"):
            par._select_prepare_backend(cfg, "rusty")


class TestPrepareTrainingDataParallelFailureHandling:
    """Issue #265: the run itself -- not just the low-level dispatcher --
    must fail loudly when it was systematically broken, rather than finishing
    with exit code 0 and a pile of warnings. These drive
    ``prepare_training_data_parallel`` end to end, stubbing out only the
    per-batch dispatch (``_iter_python_batches``), so what's under test is the
    aggregation and raise logic, not either backend's pipeline.
    """

    @staticmethod
    def _run(monkeypatch, outcomes):
        monkeypatch.setattr(par, "_select_prepare_backend", lambda cfg, choice: False)
        monkeypatch.setattr(par, "_iter_python_batches", lambda *a, **k: iter(outcomes))
        return par.prepare_training_data_parallel(
            bam_path=Path("fake.bam"), config=object(), num_workers=1, chunk_size=2
        )

    def test_any_failed_batch_raises(self, monkeypatch):
        outcomes = [
            par.BatchOutcome(n_reads=2, chunks=[{"read_id": "r0"}]),
            par.BatchOutcome(n_reads=2, chunks=[], n_failed_reads=2, batch_failed=True),
        ]
        with pytest.raises(RuntimeError, match="failed outright"):
            self._run(monkeypatch, outcomes)

    def test_high_failed_read_fraction_raises(self, monkeypatch):
        """9 of 10 reads failed -- well past the 50% default threshold."""
        outcomes = [par.BatchOutcome(n_reads=10, chunks=[{"read_id": "r0"}], n_failed_reads=9)]
        with pytest.raises(RuntimeError, match="threshold"):
            self._run(monkeypatch, outcomes)

    def test_low_failed_read_fraction_does_not_raise(self, monkeypatch):
        """3 of 10 reads failed -- under the threshold, a healthy-ish run."""
        chunks_out = [{"read_id": f"r{i}"} for i in range(4)]
        outcomes = [par.BatchOutcome(n_reads=10, chunks=chunks_out, n_failed_reads=3)]
        chunks, stats = self._run(monkeypatch, outcomes)
        assert len(chunks) == 4
        assert stats["failed_reads"] == 3
        assert stats["failed_batches"] == 0
        # `reads_without_motif` must not double-count the failed reads.
        assert stats["reads_without_motif"] == 10 - 4 - 3

    def test_zero_reads_does_not_raise(self, monkeypatch):
        """An empty BAM (or a filter that matched nothing) is not a failure
        this check is meant to catch -- it returns a clean empty result."""
        chunks, stats = self._run(monkeypatch, [])
        assert chunks == []
        assert stats["total_chunks"] == 0
        assert stats["failed_reads"] == 0
        assert stats["failed_batches"] == 0
        assert stats["reads_missing_from_pod5"] == 0

    def test_reads_absent_from_the_pod5_do_not_trip_the_threshold(self, monkeypatch):
        """Issue #325: 9 of 10 reads not in the POD5 -- 90%, and fine.

        This is what a POD5 pre-filtered to one cognate region looks like
        against a full-scope BAM. Before the fix every one of those reads was
        counted as failed, so the run raised with no output at all even
        though the one present read extracted correctly.
        """
        outcomes = [
            par.BatchOutcome(n_reads=10, chunks=[{"read_id": "r0"}], n_reads_missing_from_pod5=9)
        ]
        chunks, stats = self._run(monkeypatch, outcomes)
        assert len(chunks) == 1
        assert stats["reads_missing_from_pod5"] == 9
        assert stats["failed_reads"] == 0
        # Absent reads belong to neither "no motif" nor "failed".
        assert stats["reads_without_motif"] == 0
        assert stats["reads_with_motif"] == 1

    def test_failed_fraction_is_measured_over_attempted_reads(self, monkeypatch):
        """Absent reads leave the denominator too, or they hide real failures.

        100 reads, 90 absent from the POD5, 6 of the remaining 10 failing:
        60% of what was actually attempted, but only 6% of the BAM. Measured
        against the BAM it never trips, and a genuinely broken pipeline
        finishes clean behind a pre-filtered POD5.
        """
        outcomes = [
            par.BatchOutcome(
                n_reads=100,
                chunks=[{"read_id": f"r{i}"} for i in range(4)],
                n_failed_reads=6,
                n_reads_missing_from_pod5=90,
            )
        ]
        with pytest.raises(RuntimeError, match="threshold"):
            self._run(monkeypatch, outcomes)

    def test_every_read_absent_returns_cleanly(self, monkeypatch):
        """Nothing attempted is not a failed fraction -- and not a crash.

        A wholly mismatched BAM/POD5 pair lands here with an empty
        denominator. It is still caught, one level up: zero chunks is what
        ``handle_prepare`` raises on, and the count is in the stats.
        """
        outcomes = [par.BatchOutcome(n_reads=10, chunks=[], n_reads_missing_from_pod5=10)]
        chunks, stats = self._run(monkeypatch, outcomes)
        assert chunks == []
        assert stats["total_chunks"] == 0
        assert stats["reads_missing_from_pod5"] == 10
        assert stats["failed_reads"] == 0

    def test_a_genuine_failure_behind_a_filtered_pod5_still_raises(self, monkeypatch):
        """Both buckets at once: the #265 guard must still fire."""
        outcomes = [
            par.BatchOutcome(
                n_reads=20,
                chunks=[],
                n_failed_reads=8,
                n_reads_missing_from_pod5=10,
            )
        ]
        with pytest.raises(RuntimeError, match="threshold"):
            self._run(monkeypatch, outcomes)
