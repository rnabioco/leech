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

import pytest

from leech.preparation import parallel as par


def _drive(monkeypatch, work, *, num_workers, n_batches, batch_size=2):
    """Run ``_iter_rust_batches`` over ``n_batches`` stub batches."""
    batches = [[object()] * batch_size for _ in range(n_batches)]
    monkeypatch.setattr(par, "iter_read_info_batches", lambda *a, **k: iter(batches))
    monkeypatch.setattr(par, "_prepare_batch_rust", work)

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

        def work(read_batch, config, motif_searcher):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            # Block until the driver has had the chance to start others. A
            # serial driver never reaches 4 in flight and the watcher gives up.
            release.wait(timeout=20.0)
            with lock:
                live -= 1
            return [{"read_id": "x"}]

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

        def work(read_batch, config, motif_searcher):
            return [{"n": len(read_batch)}]

        results = _drive(monkeypatch, work, num_workers=4, n_batches=5, batch_size=3)

        assert [n_reads for n_reads, _ in results] == [3] * 5
        assert [chunks[0]["n"] for _, chunks in results] == [3] * 5

    def test_failed_batch_is_skipped_not_fatal(self, monkeypatch):
        """A bad batch must not end the run, and its reads still count."""
        calls = {"n": 0}
        lock = threading.Lock()

        def work(read_batch, config, motif_searcher):
            with lock:
                calls["n"] += 1
                n = calls["n"]
            if n == 2:
                raise RuntimeError("boom")
            return [{"ok": True}]

        results = _drive(monkeypatch, work, num_workers=2, n_batches=4)

        assert len(results) == 4
        assert sum(len(chunks) for _, chunks in results) == 3
        assert all(n_reads == 2 for n_reads, _ in results)

    def test_in_flight_window_is_bounded(self, monkeypatch):
        """The driver must not pull the whole BAM into memory up front."""
        num_workers = 3
        window = 2 * num_workers

        started = threading.Semaphore(0)
        proceed = threading.Event()
        consumed = {"n": 0}

        def work(read_batch, config, motif_searcher):
            started.release()
            proceed.wait(timeout=20.0)
            return []

        def counting_batches(*a, **k):
            for _ in range(500):
                consumed["n"] += 1
                yield [object(), object()]

        monkeypatch.setattr(par, "iter_read_info_batches", counting_batches)
        monkeypatch.setattr(par, "_prepare_batch_rust", work)

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
