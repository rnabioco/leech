"""
End-to-end pipeline integration tests: prepare → merge → train → eval → predict.

Uses real tRNA IVC fixtures (20 Ala reads) with randomly initialized models.
All tests are marked @pytest.mark.slow — run with `pytest --slow` to include.

Each test is independent: shared data preparation is cached via a module-scoped
fixture, but train/eval/predict each create their own model from scratch.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import TRNA_BAM, TRNA_FIXTURES_AVAILABLE, TRNA_POD5, TRNA_REF

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available"),
]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MOTIF = "CCAGGC"
MOTIF_OFFSET = 2
CLASS_NAMES = ["Ala", "Gly"]  # binary for fast training


# ---------------------------------------------------------------------------
# Module-scoped data preparation (runs once for all tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared_data(tmp_path_factory):
    """Run prepare on tRNA fixtures twice (Ala + Gly labels), return output dir.

    Both use the same 20 reads — labels differ so merge sees two classes.
    """
    from leech.commands.prepare import handle_prepare

    out = tmp_path_factory.mktemp("prepare")
    _common = {
        "pod5": TRNA_POD5,
        "bam": TRNA_BAM,
        "motif": MOTIF,
        "motif_offset": MOTIF_OFFSET,
        "reference_fasta": TRNA_REF,
        "no_split": True,
        "workers": 1,
        "reverse_signal": True,
        "refine_signal_map": False,
    }
    handle_prepare(output_dir=out / "ala", label="Ala", **_common)
    handle_prepare(output_dir=out / "gly", label="Gly", **_common)
    assert (out / "ala" / "all.npz").exists()
    assert (out / "gly" / "all.npz").exists()
    return out


@pytest.fixture(scope="module")
def merged_data(tmp_path_factory, prepared_data):
    """Merge Ala + Gly chunks and split at read level."""
    from leech.commands.merge_split import handle_merge_and_split

    ala_npz = prepared_data / "ala" / "all.npz"
    gly_npz = prepared_data / "gly" / "all.npz"
    out = tmp_path_factory.mktemp("merge")

    handle_merge_and_split(
        input_chunks=(f"Ala={ala_npz}", f"Gly={gly_npz}"),
        output_dir=out,
        train_split=0.6,
        val_split=0.2,
        seed=42,
    )
    assert (out / "train.npz").exists()
    assert (out / "val.npz").exists()
    assert (out / "test.npz").exists()
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_chunk_info(npz_path: Path) -> dict:
    """Load an npz and return basic stats.

    The serialization format uses *_flat keys (signals_flat, features_flat, etc.)
    — each element is a single chunk's array stored as an object array entry.
    """
    data = np.load(npz_path, allow_pickle=True)
    n = len(data["read_ids"])
    signal_len = data["signals_flat"][0].shape[-1] if n > 0 else 0
    kmer_len = len(data["sequences"][0]) if n > 0 else 0
    num_features = data["features_flat"][0].shape[0] if n > 0 else 0
    labels = set(data["labels"]) if "labels" in data else set()
    return {
        "n": n,
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "num_features": num_features,
        "labels": labels,
    }


def _train_binary_model(merged_dir: Path, output_dir: Path) -> Path:
    """Train a small binary model for 2 epochs, return output dir."""
    from leech.commands.train import handle_train

    info = _get_chunk_info(merged_dir / "train.npz")

    handle_train(
        train_data=merged_dir / "train.npz",
        val_data=merged_dir / "val.npz",
        model_name="ConvLSTMDwell",
        model_config=None,
        output_dir=output_dir,
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
        device="cpu",
        seed=42,
        early_stopping=0,
        use_class_weights=False,
        pos_weight=None,
        resume=None,
        weight_decay=0.0,
        max_grad_norm=1.0,
        scheduler="plateau",
        scheduler_patience=5,
        scheduler_factor=0.5,
        warmup_epochs=0,
        loss_type="bce",
        focal_gamma=2.0,
        label_smoothing=0.0,
        mixed_precision=False,
        augment_jitter=0.0,
        augment_scale_min=1.0,
        augment_scale_max=1.0,
        num_workers=0,
        motif=MOTIF,
        motif_offset=MOTIF_OFFSET,
        seq_encoding="base_onehot",
        signal_len=info["signal_len"],
        kmer_len=info["kmer_len"],
        num_features=info["num_features"],
        conv_channels=[4, 16, 32],
        lstm_hidden=16,
        dropout=0.0,
    )
    return output_dir


def _train_multiclass_model(merged_dir: Path, output_dir: Path) -> Path:
    """Train a small multiclass model for 2 epochs, return output dir."""
    from leech.training import train_model

    info = _get_chunk_info(merged_dir / "train.npz")

    train_model(
        train_data_path=merged_dir / "train.npz",
        val_data_path=merged_dir / "val.npz",
        model_name="ConvLSTMDwell",
        output_dir=output_dir,
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
        device="cpu",
        seed=42,
        early_stopping_patience=0,
        use_class_weights=False,
        loss_type="cross_entropy",
        num_workers=0,
        motif=MOTIF,
        motif_offset=MOTIF_OFFSET,
        seq_encoding="base_onehot",
        num_out=len(CLASS_NAMES),
        signal_len=info["signal_len"],
        kmer_len=info["kmer_len"],
        num_features=info["num_features"],
        conv_channels=[4, 16, 32],
        lstm_hidden=16,
        dropout=0.0,
    )
    return output_dir


# ---------------------------------------------------------------------------
# Tests: prepare
# ---------------------------------------------------------------------------


class TestPrepare:
    """Data preparation from BAM/POD5 fixtures."""

    def test_prepare_produces_chunks(self, prepared_data):
        """prepare --no-split produces all.npz with chunks."""
        info = _get_chunk_info(prepared_data / "ala" / "all.npz")
        assert info["n"] > 0, "No chunks extracted"
        assert info["signal_len"] > 0
        assert info["kmer_len"] > 0
        assert info["num_features"] > 0

    def test_prepare_config_sidecar(self, prepared_data):
        """prepare writes prepare_config.json with extraction parameters."""
        config_path = prepared_data / "ala" / "prepare_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config = json.load(f)
        assert config["motif"] == MOTIF
        assert config["motif_offset"] == MOTIF_OFFSET

    def test_prepare_labels_match(self, prepared_data):
        """Each prepare run labels chunks with the specified label."""
        ala_info = _get_chunk_info(prepared_data / "ala" / "all.npz")
        gly_info = _get_chunk_info(prepared_data / "gly" / "all.npz")
        assert ala_info["labels"] == {"Ala"}
        assert gly_info["labels"] == {"Gly"}


# ---------------------------------------------------------------------------
# Tests: merge
# ---------------------------------------------------------------------------


class TestMerge:
    """Merge and split at read level."""

    def test_merge_creates_splits(self, merged_data):
        """merge produces train/val/test .npz files."""
        for split in ["train", "val", "test"]:
            path = merged_data / f"{split}.npz"
            assert path.exists(), f"Missing {split}.npz"
            info = _get_chunk_info(path)
            assert info["n"] > 0, f"{split}.npz is empty"

    def test_merge_has_both_labels(self, merged_data):
        """Merged data contains both class labels."""
        info = _get_chunk_info(merged_data / "train.npz")
        assert info["labels"] == set(CLASS_NAMES)

    def test_merge_no_read_leakage(self, merged_data):
        """No read IDs shared between train and test splits."""
        train = np.load(merged_data / "train.npz", allow_pickle=True)
        test = np.load(merged_data / "test.npz", allow_pickle=True)
        train_ids = set(train["read_ids"])
        test_ids = set(test["read_ids"])
        assert train_ids.isdisjoint(test_ids), "Read leakage between train and test"


# ---------------------------------------------------------------------------
# Tests: train
# ---------------------------------------------------------------------------


class TestTrain:
    """Model training on prepared data."""

    def test_train_binary_produces_checkpoint(self, merged_data, tmp_path):
        """Binary training produces model_best.pt and summary.json."""
        model_dir = _train_binary_model(merged_data, tmp_path / "binary_model")
        assert (model_dir / "model_best.pt").exists()
        assert (model_dir / "summary.json").exists()
        assert (model_dir / "config.json").exists()

    def test_train_binary_summary_has_metrics(self, merged_data, tmp_path):
        """Summary JSON contains expected training metrics."""
        model_dir = _train_binary_model(merged_data, tmp_path / "binary_model")
        with open(model_dir / "summary.json") as f:
            summary = json.load(f)
        assert "best_val_f1" in summary or "best_val_acc" in summary
        assert "best_epoch" in summary

    def test_train_multiclass_produces_checkpoint(self, merged_data, tmp_path):
        """Multiclass training produces model_best.pt."""
        model_dir = _train_multiclass_model(merged_data, tmp_path / "mc_model")
        assert (model_dir / "model_best.pt").exists()
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        assert config["num_out"] == len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Tests: eval
# ---------------------------------------------------------------------------


class TestEval:
    """Model evaluation on held-out test data."""

    def test_eval_binary_produces_metrics(self, merged_data, tmp_path):
        """eval test on binary model produces JSON with accuracy/F1."""
        from leech.commands.eval import handle_test

        model_dir = _train_binary_model(merged_data, tmp_path / "eval_binary")
        metrics_path = tmp_path / "metrics.json"

        handle_test(
            model=model_dir,
            test_data=merged_data / "test.npz",
            output=metrics_path,
            device="cpu",
            batch_size=64,
        )

        assert metrics_path.exists()
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_eval_multiclass_produces_metrics(self, merged_data, tmp_path):
        """eval test on multiclass model produces JSON with accuracy/F1."""
        from leech.commands.eval import handle_test

        model_dir = _train_multiclass_model(merged_data, tmp_path / "eval_mc")
        metrics_path = tmp_path / "mc_metrics.json"

        handle_test(
            model=model_dir,
            test_data=merged_data / "test.npz",
            output=metrics_path,
            device="cpu",
            batch_size=64,
        )

        assert metrics_path.exists()
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Tests: predict (inference on raw BAM/POD5)
# ---------------------------------------------------------------------------


class TestPredict:
    """End-to-end inference writing tags to output BAM."""

    def test_predict_multiclass_writes_tags(self, merged_data, tmp_path):
        """Multiclass predict writes aa/ac/am/pn/pp tags to output BAM."""
        import pysam

        from leech.inference import run_inference
        from leech.model_loading import load_model_from_checkpoint

        model_dir = _train_multiclass_model(merged_data, tmp_path / "pred_mc")
        model, config = load_model_from_checkpoint(model_dir, device="cpu")

        # Add multiclass metadata that run_inference needs
        config["num_out"] = len(CLASS_NAMES)
        config["label_map"] = {name: i for i, name in enumerate(CLASS_NAMES)}
        config["refine_signal_map"] = False

        output_bam = tmp_path / "predictions.bam"
        run_inference(
            model_and_config=(model, config),
            pod5_path=TRNA_POD5,
            bam_path=TRNA_BAM,
            output_path=output_bam,
            device="cpu",
            batch_size=64,
            reverse_signal=True,
            reference_fasta=TRNA_REF,
            raw=False,
            min_confidence=0,
            min_margin=0,
        )

        with pysam.AlignmentFile(str(output_bam), "rb") as bam:
            tagged = [r for r in bam if r.has_tag("aa")]
        assert len(tagged) > 0
        for read in tagged:
            assert read.has_tag("ac")
            assert read.has_tag("am")
            assert read.has_tag("pn")
            assert read.has_tag("pp")
            assert read.get_tag("aa") in set(CLASS_NAMES) | {"unc"}

    def test_predict_binary_writes_tags(self, merged_data, tmp_path):
        """Binary predict writes tags to output BAM (num_out=1 path)."""
        import pysam

        from leech.inference import run_inference
        from leech.model_loading import load_model_from_checkpoint

        model_dir = _train_binary_model(merged_data, tmp_path / "pred_bin")
        model, config = load_model_from_checkpoint(model_dir, device="cpu")
        config["refine_signal_map"] = False

        output_bam = tmp_path / "bin_predictions.bam"
        run_inference(
            model_and_config=(model, config),
            pod5_path=TRNA_POD5,
            bam_path=TRNA_BAM,
            output_path=output_bam,
            device="cpu",
            batch_size=64,
            reverse_signal=True,
            reference_fasta=TRNA_REF,
            raw=False,
        )

        # Binary models use MP/ML tags, not aa/ac/am
        with pysam.AlignmentFile(str(output_bam), "rb") as bam:
            reads = list(bam)
        assert len(reads) > 0
        tagged = [r for r in reads if r.has_tag("MP")]
        assert len(tagged) > 0

    def test_predict_with_min_confidence(self, merged_data, tmp_path):
        """min_confidence=255 forces all reads to 'unc'."""
        import pysam

        from leech.inference import run_inference
        from leech.model_loading import load_model_from_checkpoint

        model_dir = _train_multiclass_model(merged_data, tmp_path / "pred_conf")
        model, config = load_model_from_checkpoint(model_dir, device="cpu")
        config["num_out"] = len(CLASS_NAMES)
        config["label_map"] = {name: i for i, name in enumerate(CLASS_NAMES)}
        config["refine_signal_map"] = False

        output_bam = tmp_path / "unc_predictions.bam"
        run_inference(
            model_and_config=(model, config),
            pod5_path=TRNA_POD5,
            bam_path=TRNA_BAM,
            output_path=output_bam,
            device="cpu",
            batch_size=64,
            reverse_signal=True,
            reference_fasta=TRNA_REF,
            min_confidence=255,
        )

        with pysam.AlignmentFile(str(output_bam), "rb") as bam:
            tagged = [r for r in bam if r.has_tag("aa")]
        assert len(tagged) > 0
        assert all(r.get_tag("aa") == "unc" for r in tagged)
