"""
Tests for RNA signal reversal (reverse_signal parameter).

Direct RNA sequencing acquires signal 3'->5', but the basecaller reverses
it internally to work 5'->3'. The move table (mv/ts/ns tags) references
the basecaller's reversed signal. We must reverse the POD5 signal to match.

These tests verify:
1. Signal reversal happens in iter_bam_with_pod5 (sequential path)
2. Signal reversal happens in the parallel worker
3. reverse_signal=False preserves original signal
4. CLI flag --no-reverse-signal correctly maps to reverse_signal=False
5. After reversal, seq_to_sig_map correctly indexes the signal
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

from leech.features import MoveTable

# ---------------------------------------------------------------------------
# Helpers: create deterministic asymmetric signal & mock BAM alignment
# ---------------------------------------------------------------------------


def _make_asymmetric_signal(n=1000):
    """Create a signal that is NOT a palindrome, so reversal is detectable."""
    rng = np.random.RandomState(123)
    signal = rng.randn(n).astype(np.float32)
    # Double-check it's not accidentally symmetric
    assert not np.array_equal(signal, signal[::-1])
    return signal


def _make_mock_alignment(read_id="test_read", num_bases=20, stride=5):
    """Create a mock pysam alignment with mv/ns/ts tags.

    Returns an alignment where every base gets exactly `stride` signal samples,
    with trim_offset=0, so signal boundaries are trivially predictable.

    The move array has num_bases * stride entries (one move per stride positions).
    Each entry corresponds to `stride` raw signal samples, so the total signal
    length is num_bases * stride * stride = num_bases * stride^2.
    """
    aln = MagicMock()
    aln.query_name = read_id
    aln.query_sequence = "ACGT" * (num_bases // 4) + "A" * (num_bases % 4)
    aln.mapping_quality = 60
    aln.reference_name = "chr1"
    aln.reference_start = 0
    aln.reference_end = num_bases
    aln.is_reverse = False
    aln.is_unmapped = False
    aln.is_secondary = False
    aln.is_supplementary = False

    # Build mv tag: [stride, 1, 0, 0, ..., 1, 0, 0, ...]
    # Each base = 1 move + (stride-1) stays → stride entries per base
    moves = []
    for _ in range(num_bases):
        moves.append(1)
        moves.extend([0] * (stride - 1))
    mv_tag = np.array([stride] + moves, dtype=np.int8)

    # num_samples = last move position * stride + stride = total signal length
    # Move positions are at indices [0, stride, 2*stride, ...] in the array.
    # Signal position = move_array_index * stride. The last base starts at
    # (num_bases-1)*stride * stride and ends at num_bases*stride*stride.
    num_samples = len(moves) * stride

    def has_tag(tag):
        return tag in ("mv", "ns")

    def get_tag(tag):
        if tag == "mv":
            return mv_tag
        if tag == "ns":
            return num_samples
        if tag == "ts":
            return 0
        raise KeyError(tag)

    aln.has_tag = has_tag
    aln.get_tag = get_tag

    return aln, num_samples


# ---------------------------------------------------------------------------
# 1. Sequential path: iter_bam_with_pod5
# ---------------------------------------------------------------------------


class TestIterBamWithPod5ReverseSignal:
    """Test signal reversal in the sequential reader path."""

    def _run_iter(self, reverse_signal, raw_signal, num_bases=20, stride=5):
        """Helper: patch I/O and run iter_bam_with_pod5, return first LeechRead."""
        from leech.configs import SignalConfig

        aln, num_samples = _make_mock_alignment(num_bases=num_bases, stride=stride)

        # Trim raw_signal to match num_samples
        raw = raw_signal[:num_samples].copy()

        mock_pod5_reader = MagicMock()
        mock_pod5_reader.get_signal.return_value = (raw, {"read_id": "test_read"})
        mock_pod5_reader.__enter__ = MagicMock(return_value=mock_pod5_reader)
        mock_pod5_reader.__exit__ = MagicMock(return_value=False)

        with (
            patch("leech.preparation.reader.POD5Reader", return_value=mock_pod5_reader),
            patch("leech.preparation.reader.iter_bam_batches", return_value=iter([[aln]])),
        ):
            from leech.preparation.reader import iter_bam_with_pod5

            reads = list(
                iter_bam_with_pod5(
                    Path("fake.bam"),
                    Path("fake.pod5"),
                    signal_config=SignalConfig(
                        reverse_signal=reverse_signal,
                        anchor="basecall",
                        refine_signal_map=False,
                    ),
                )
            )

        assert len(reads) == 1, f"Expected 1 read, got {len(reads)}"
        return reads[0], raw

    def test_reverse_signal_true_reverses(self):
        """With reverse_signal=True, the normalized signal should come from reversed raw."""
        raw_signal = _make_asymmetric_signal(10000)
        read, raw = self._run_iter(reverse_signal=True, raw_signal=raw_signal)

        # The signal stored in the LeechRead is normalized, so we can't compare
        # directly to raw. But we CAN verify it was normalized from the reversed
        # signal by checking the order: median-MAD normalization is monotonic
        # (order-preserving), so the normalized signal should have the same
        # ordering as the reversed raw signal.
        reversed_raw = raw[::-1]

        # Verify sign pattern matches reversed raw, not original
        # Use a robust check: correlation with reversed raw >> correlation with raw
        corr_reversed = np.corrcoef(read.signal, reversed_raw)[0, 1]
        corr_original = np.corrcoef(read.signal, raw)[0, 1]
        assert corr_reversed > 0.99, (
            f"Signal should correlate with reversed raw (r={corr_reversed:.4f})"
        )
        # The original should NOT correlate as highly (unless signal is symmetric)
        assert corr_reversed > corr_original, (
            "Signal correlates more with original than reversed — reversal not applied?"
        )

    def test_reverse_signal_false_preserves(self):
        """With reverse_signal=False, the normalized signal should match original raw order."""
        raw_signal = _make_asymmetric_signal(10000)
        read, raw = self._run_iter(reverse_signal=False, raw_signal=raw_signal)

        corr_original = np.corrcoef(read.signal, raw)[0, 1]
        corr_reversed = np.corrcoef(read.signal, raw[::-1])[0, 1]
        assert corr_original > 0.99, (
            f"Signal should correlate with original raw (r={corr_original:.4f})"
        )
        assert corr_original > corr_reversed, (
            "Signal correlates more with reversed than original — unexpected reversal?"
        )

    def test_default_is_reverse_true(self):
        """Default SignalConfig.reverse_signal should be True (RNA mode)."""
        from leech.configs import SignalConfig

        assert SignalConfig().reverse_signal is True


# ---------------------------------------------------------------------------
# 2. After reversal, seq_to_sig_map correctly indexes the signal
# ---------------------------------------------------------------------------


class TestReversedSignalIndexing:
    """Verify that seq_to_sig_map indexes correctly into reversed signal."""

    def test_signal_levels_match_after_reversal(self):
        """After reversal, extracting signal per base using seq_to_sig_map should work."""
        stride = 5
        num_bases = 4
        num_samples = num_bases * stride  # 20

        # Create signal where each base has a distinct level in the REVERSED signal
        # Reversed signal: base 0 -> samples [0:5), base 1 -> [5:10), etc.
        reversed_signal = np.zeros(num_samples, dtype=np.float32)
        reversed_signal[0:5] = 1.0
        reversed_signal[5:10] = 2.0
        reversed_signal[10:15] = 3.0
        reversed_signal[15:20] = 4.0

        # The POD5 stores the UN-reversed signal (3'->5')
        pod5_signal = reversed_signal[::-1]

        # Simulate what leech does: reverse, then index
        processed_signal = pod5_signal[::-1]  # This recovers reversed_signal
        np.testing.assert_array_equal(processed_signal, reversed_signal)

        # Build move table (all consecutive moves)
        moves = np.array([1] * num_bases, dtype=np.int8)
        mt = MoveTable(
            stride=stride,
            moves=moves,
            read_id="test",
            num_samples=num_samples,
            trim_offset=0,
        )
        seq_to_sig = mt.to_seq_to_sig_map()

        # Extract per-base signal levels
        for base_idx in range(num_bases):
            start = seq_to_sig[base_idx]
            end = seq_to_sig[base_idx + 1]
            base_signal = processed_signal[start:end]
            expected_level = float(base_idx + 1)
            np.testing.assert_allclose(
                base_signal.mean(),
                expected_level,
                err_msg=f"Base {base_idx}: expected level {expected_level}, "
                f"got {base_signal.mean()}",
            )

    def test_without_reversal_indexing_is_wrong(self):
        """Without reversal, seq_to_sig_map points to wrong signal regions."""
        stride = 5
        num_bases = 4
        num_samples = num_bases * stride

        # Same setup: distinct levels in the reversed (basecaller) signal space
        reversed_signal = np.zeros(num_samples, dtype=np.float32)
        reversed_signal[0:5] = 1.0
        reversed_signal[5:10] = 2.0
        reversed_signal[10:15] = 3.0
        reversed_signal[15:20] = 4.0

        pod5_signal = reversed_signal[::-1]

        # WITHOUT reversal: use pod5_signal directly (WRONG for RNA)
        moves = np.array([1] * num_bases, dtype=np.int8)
        mt = MoveTable(
            stride=stride,
            moves=moves,
            read_id="test",
            num_samples=num_samples,
            trim_offset=0,
        )
        seq_to_sig = mt.to_seq_to_sig_map()

        # Extract per-base signal from UN-reversed signal — should be WRONG
        mismatches = 0
        for base_idx in range(num_bases):
            start = seq_to_sig[base_idx]
            end = seq_to_sig[base_idx + 1]
            base_signal = pod5_signal[start:end]
            expected_level = float(base_idx + 1)
            if not np.isclose(base_signal.mean(), expected_level):
                mismatches += 1

        assert mismatches > 0, (
            "Without reversal, signal levels should NOT match — this means the test setup is wrong"
        )


# ---------------------------------------------------------------------------
# 3. Parallel worker path
# ---------------------------------------------------------------------------


class TestParallelWorkerReverseSignal:
    """Test signal reversal in the parallel processing path."""

    def test_worker_reverses_signal(self):
        """The parallel worker should reverse signal when reverse_signal=True."""
        from leech.configs import (
            ChunkConfig,
            LabelConfig,
            MotifConfig,
            PrepareConfig,
            SignalConfig,
        )
        from leech.features import MoveTable
        from leech.io.bam_reader import ReadInfo

        stride = 5
        num_bases = 100  # Need enough bases for default signal context (200+200)
        num_samples = num_bases * stride  # 500 samples

        # Create asymmetric signal
        raw_signal = _make_asymmetric_signal(num_samples)

        # Create a mock ReadInfo
        read_info = MagicMock(spec=ReadInfo)
        read_info.read_id = "test_read"
        read_info.sequence = "ACGT" * (num_bases // 4)
        read_info.mapping_quality = 60
        read_info.reference_name = "chr1"
        read_info.reference_start = 0
        read_info.reference_end = num_bases
        read_info.is_reverse = False
        read_info.cigar_tuples = [(0, num_bases)]
        read_info.reference_sequence = None

        # Build a real MoveTable
        moves = np.array([1] * num_bases, dtype=np.int8)
        mt = MoveTable(
            stride=stride,
            moves=moves,
            read_id="test_read",
            num_samples=num_samples,
            trim_offset=0,
        )
        read_info.to_move_table.return_value = mt

        # Mock POD5 read object
        mock_pod5_read = MagicMock()
        mock_pod5_read.signal = raw_signal.copy()
        mock_pod5_read.read_id = "test_read"
        mock_pod5_read.pore.channel = 1
        mock_pod5_read.pore.well = 1
        mock_pod5_read.pore.pore_type = "not_set"
        mock_pod5_read.calibration.offset = 0.0
        mock_pod5_read.calibration.scale = 1.0
        mock_pod5_read.run_info.sample_rate = 4000

        # Mock DatasetReader
        mock_dataset_reader = MagicMock()
        mock_dataset_reader.__enter__ = MagicMock(return_value=mock_dataset_reader)
        mock_dataset_reader.__exit__ = MagicMock(return_value=False)
        mock_dataset_reader.reads.return_value = iter([mock_pod5_read])

        with patch("leech.preparation.parallel.DatasetReader", return_value=mock_dataset_reader):
            from leech.preparation.parallel import _process_read_chunk_worker

            # Build PrepareConfig with reverse_signal=True
            config_rev = PrepareConfig(
                pod5_path=Path("fake.pod5"),
                signal=SignalConfig(
                    reverse_signal=True,
                    anchor="basecall",
                    refine_signal_map=False,
                ),
                motif=MotifConfig(motif=None, motif_reference="bam"),
                chunk=ChunkConfig(),
                labeling=LabelConfig(label="test", label_int=0),
            )
            chunks_reversed = _process_read_chunk_worker(([read_info], config_rev))

            # Reset mock for second call
            mock_pod5_read.signal = raw_signal.copy()
            mock_dataset_reader.reads.return_value = iter([mock_pod5_read])

            # Build PrepareConfig with reverse_signal=False
            config_no_rev = PrepareConfig(
                pod5_path=Path("fake.pod5"),
                signal=SignalConfig(
                    reverse_signal=False,
                    anchor="basecall",
                    refine_signal_map=False,
                ),
                motif=MotifConfig(motif=None, motif_reference="bam"),
                chunk=ChunkConfig(),
                labeling=LabelConfig(label="test", label_int=0),
            )
            chunks_not_reversed = _process_read_chunk_worker(([read_info], config_no_rev))

        # Both should produce chunks (no motif = extract all valid positions)
        assert len(chunks_reversed) > 0, "No chunks extracted with reverse_signal=True"
        assert len(chunks_not_reversed) > 0, "No chunks extracted with reverse_signal=False"

        # The signal in chunks should differ between reversed and not-reversed
        sig_rev = chunks_reversed[0]["signal"]
        sig_norev = chunks_not_reversed[0]["signal"]
        assert not np.allclose(sig_rev, sig_norev), (
            "Reversed and non-reversed chunks have identical signal — reversal not applied"
        )


# ---------------------------------------------------------------------------
# 4. Parameter threading through orchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorThreading:
    """Verify config-based functions accept the right parameters."""

    def test_prepare_training_data_accepts_config(self):
        """prepare_training_data should accept config: PrepareConfig."""
        import inspect

        from leech.preparation.orchestrator import prepare_training_data

        sig = inspect.signature(prepare_training_data)
        assert "config" in sig.parameters

    def test_prepare_training_data_with_split_accepts_config(self):
        """prepare_training_data_with_split should accept config: PrepareConfig."""
        import inspect

        from leech.preparation.orchestrator import prepare_training_data_with_split

        sig = inspect.signature(prepare_training_data_with_split)
        assert "config" in sig.parameters

    def test_prepare_training_data_parallel_accepts_config(self):
        """prepare_training_data_parallel should accept config: PrepareConfig."""
        import inspect

        from leech.preparation.parallel import prepare_training_data_parallel

        sig = inspect.signature(prepare_training_data_parallel)
        assert "config" in sig.parameters

    def test_run_inference_accepts_reverse_signal(self):
        """run_inference should accept reverse_signal kwarg."""
        import inspect

        from leech.inference import run_inference

        sig = inspect.signature(run_inference)
        assert "reverse_signal" in sig.parameters
        assert sig.parameters["reverse_signal"].default is True

    def test_handle_prepare_accepts_reverse_signal(self):
        """handle_prepare should accept reverse_signal kwarg."""
        import inspect

        from leech.commands.prepare import handle_prepare

        sig = inspect.signature(handle_prepare)
        assert "reverse_signal" in sig.parameters
        assert sig.parameters["reverse_signal"].default is True


# ---------------------------------------------------------------------------
# 5. CLI flag --no-reverse-signal
# ---------------------------------------------------------------------------


class TestCLIReverseSignalFlag:
    """Test that the CLI correctly wires --no-reverse-signal."""

    def test_prepare_no_reverse_signal_flag_exists(self):
        """The prepare command should have --no-reverse-signal flag."""
        from leech.cli import prepare

        param_names = [p.name for p in prepare.params]
        assert "no_reverse_signal" in param_names

    def test_predict_no_reverse_signal_flag_exists(self):
        """The predict command should have --no-reverse-signal flag."""
        from leech.cli import predict

        param_names = [p.name for p in predict.params]
        assert "no_reverse_signal" in param_names

    def test_prepare_passes_reverse_signal_false(self, tmp_path):
        """--no-reverse-signal should pass reverse_signal=False to handle_prepare."""
        from leech.cli import cli

        runner = CliRunner()

        # Create dummy files so Click's exists=True validation passes
        pod5_file = tmp_path / "test.pod5"
        bam_file = tmp_path / "test.bam"
        pod5_file.touch()
        bam_file.touch()
        out_dir = tmp_path / "out"

        with patch("leech.commands.handle_prepare") as mock_handle:
            mock_handle.return_value = {"n_chunks": 0, "n_train": 0, "n_val": 0, "n_test": 0}
            result = runner.invoke(
                cli,
                [
                    "data",
                    "prepare",
                    "--pod5",
                    str(pod5_file),
                    "--bam",
                    str(bam_file),
                    "--output-dir",
                    str(out_dir),
                    "--no-reverse-signal",
                ],
                catch_exceptions=False,
            )

            assert mock_handle.called, f"handle_prepare not called. Output: {result.output}"
            assert mock_handle.call_args.kwargs["reverse_signal"] is False

    def test_prepare_default_reverse_signal_true(self, tmp_path):
        """Without --no-reverse-signal, reverse_signal should be True."""
        from leech.cli import cli

        runner = CliRunner()

        pod5_file = tmp_path / "test.pod5"
        bam_file = tmp_path / "test.bam"
        pod5_file.touch()
        bam_file.touch()
        out_dir = tmp_path / "out"

        with patch("leech.commands.handle_prepare") as mock_handle:
            mock_handle.return_value = {"n_chunks": 0, "n_train": 0, "n_val": 0, "n_test": 0}
            result = runner.invoke(
                cli,
                [
                    "data",
                    "prepare",
                    "--pod5",
                    str(pod5_file),
                    "--bam",
                    str(bam_file),
                    "--output-dir",
                    str(out_dir),
                ],
                catch_exceptions=False,
            )

            assert mock_handle.called, f"handle_prepare not called. Output: {result.output}"
            assert mock_handle.call_args.kwargs["reverse_signal"] is True


# ---------------------------------------------------------------------------
# 6. Mathematical property: reversal is an involution
# ---------------------------------------------------------------------------


class TestReversalProperties:
    """Test mathematical properties of the signal reversal."""

    def test_double_reversal_is_identity(self):
        """Reversing twice should give back the original signal."""
        signal = _make_asymmetric_signal(500)
        reversed_once = signal[::-1]
        reversed_twice = reversed_once[::-1]
        np.testing.assert_array_equal(signal, reversed_twice)

    def test_reversal_preserves_length(self):
        """Reversed signal must have the same length."""
        signal = _make_asymmetric_signal(500)
        assert len(signal[::-1]) == len(signal)

    def test_reversal_preserves_statistics(self):
        """Reversal should not change mean, std, median, MAD."""
        signal = _make_asymmetric_signal(500)
        reversed_signal = signal[::-1]

        # Use atol for float32 precision (~1e-7)
        np.testing.assert_allclose(signal.mean(), reversed_signal.mean(), atol=1e-6)
        np.testing.assert_allclose(signal.std(), reversed_signal.std(), atol=1e-6)
        np.testing.assert_allclose(np.median(signal), np.median(reversed_signal), atol=1e-6)

        # MAD (median absolute deviation)
        mad_orig = np.median(np.abs(signal - np.median(signal)))
        mad_rev = np.median(np.abs(reversed_signal - np.median(reversed_signal)))
        np.testing.assert_allclose(mad_orig, mad_rev, atol=1e-6)

    def test_normalization_commutes_with_reversal(self):
        """median-MAD normalization should commute with reversal.

        norm(reverse(x)) == reverse(norm(x))

        This is important because it means the order of operations doesn't
        matter for the final normalized signal values — only the ordering changes.
        """
        from leech.features import normalize_signal

        signal = _make_asymmetric_signal(500)

        # Path 1: reverse then normalize
        reversed_signal = signal[::-1].copy()
        norm_of_reversed, _ = normalize_signal(reversed_signal, method="median_mad")

        # Path 2: normalize then reverse
        norm_of_original, _ = normalize_signal(signal.copy(), method="median_mad")
        reversed_norm = norm_of_original[::-1]

        np.testing.assert_allclose(
            norm_of_reversed,
            reversed_norm,
            rtol=1e-5,
            err_msg="Normalization does not commute with reversal",
        )
