"""
Pytest fixtures for leech tests.

Provides reusable test data and mock objects.
"""

import numpy as np
import pytest
import torch

from leech.chunking import LeechRead, save_chunks
from leech.features import MoveTable


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Slow test (use --slow to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture
def sample_signal():
    """Generate a sample normalized signal."""
    # 1000 samples with some noise
    np.random.seed(42)
    signal = np.random.randn(1000).astype(np.float32)
    return signal


@pytest.fixture
def sample_sequence():
    """Generate a sample DNA sequence."""
    return "ACGTACGTACGTACGTACGT"


@pytest.fixture
def sample_move_table():
    """Generate a sample MoveTable.

    Creates a move table for 20 bases with:
    - stride = 5 (basecaller samples every 5 signal points)
    - First base starts at signal position 250 (allowing 200 left context)
    - Each base gets 30 signal samples (6 stride intervals)
    - Total signal = len(moves) * stride = 170 * 5 = 850, plus trim_offset=250 → 1100
    - We use trim_offset=250 so indices are into a 1100-sample signal
    """
    stride = 5
    # Create 20 moves (one per base) spread across the signal
    # Each base occupies 6 stride intervals (30 samples)
    moves = []
    for _ in range(20):  # 20 bases
        moves.append(1)  # Move to new base
        for _ in range(5):  # Stay on base for 5 more positions (6 total * 5 = 30 samples)
            moves.append(0)

    moves = np.array(moves, dtype=np.int8)
    # Total move table length = 120 entries → covers 120*5=600 signal samples
    # With trim_offset=250, signal indices run from 250 to 250+600=850
    # num_samples=850 so the last base extends to 850

    return MoveTable(
        stride=stride,
        moves=moves,
        read_id="test_read_001",
        num_samples=850,
        trim_offset=250,
    )


@pytest.fixture
def sample_leech_read(sample_signal, sample_sequence, sample_move_table):
    """Generate a sample LeechRead object."""
    seq_to_sig_map = sample_move_table.to_seq_to_sig_map()
    dwells = np.diff(seq_to_sig_map)

    # Create dwell and signal features (5 total to match model_config)
    dwell_features = {
        "dwell": dwells.astype(np.float32),
        "dwell_log": np.log(dwells + 1e-6).astype(np.float32),
    }

    num_bases = len(dwells)
    signal_features = {
        "level_mean": np.random.randn(num_bases).astype(np.float32),
        "level_median": np.random.randn(num_bases).astype(np.float32),
        "level_std": np.abs(np.random.randn(num_bases)).astype(np.float32),
    }

    return LeechRead(
        read_id="test_read_001",
        sequence=sample_sequence,
        signal=sample_signal,
        seq_to_sig_map=seq_to_sig_map,
        dwells=dwells,
        dwell_features=dwell_features,
        signal_features=signal_features,
        labels=None,
        metadata={"test": True},
    )


@pytest.fixture
def sample_chunks(sample_leech_read):
    """Generate sample training chunks.

    Creates chunks with signal_context=(200, 200) and kmer_context=5
    to match the standard testing configuration (signal_len=400, kmer_len=11).
    """
    chunks = []

    # Create a few chunks manually, being careful about boundaries
    num_bases = len(sample_leech_read.dwells)
    kmer_context = 5  # This creates sequences of length 11 (2*5 + 1)
    start_idx = kmer_context
    end_idx = min(15, num_bases - kmer_context)  # Leave room for kmer_context

    for base_idx in range(start_idx, end_idx):
        chunk = sample_leech_read.get_chunk(
            base_idx,
            signal_context=(200, 200),  # Standard context
            kmer_context=kmer_context,
        )
        if chunk is not None:
            chunk["read_id"] = sample_leech_read.read_id
            chunk["label_int"] = base_idx % 2  # Alternate numeric labels (0 or 1)
            chunk["label"] = "charged" if (base_idx % 2) == 1 else "uncharged"  # String label
            chunks.append(chunk)

    return chunks


@pytest.fixture
def temp_chunks_file(sample_chunks, tmp_path):
    """Create a temporary chunks file."""
    chunks_file = tmp_path / "test_chunks.npz"
    save_chunks(sample_chunks, chunks_file)
    return chunks_file


@pytest.fixture
def sample_batch():
    """Generate a sample batch for model testing."""
    batch_size = 4
    signal_len = 400
    kmer_len = 11
    num_features = 5

    return {
        "signal": torch.randn(batch_size, signal_len),
        "sequence": torch.randn(batch_size, 4, kmer_len),  # One-hot encoded
        "features": torch.randn(batch_size, num_features, kmer_len),
        "label": torch.randint(0, 2, (batch_size, 1)).float(),
    }


@pytest.fixture
def model_config():
    """Standard model configuration for testing."""
    return {
        "signal_len": 400,
        "kmer_len": 11,
        "num_features": 5,
        "conv_channels": [4, 16, 64],  # Smaller for faster tests
        "lstm_hidden": 32,  # Smaller for faster tests
        "dropout": 0.1,
    }


@pytest.fixture
def temp_model_dir(tmp_path, model_config):
    """Create a temporary directory with model config."""
    import json

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Save config
    config = {
        "model_name": "ConvLSTMDwell",
        **model_config,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "device": "cpu",
        "seed": 42,
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    return model_dir


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for metrics testing."""
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    # Make predictions somewhat correlated with true labels
    y_prob = np.where(y_true == 1, y_prob * 0.5 + 0.5, y_prob * 0.5)
    y_pred = (y_prob > 0.5).astype(int)

    return y_true, y_pred, y_prob


@pytest.fixture
def sample_model(model_config):
    """Generate a sample model for testing."""
    from leech.models import get_model

    model = get_model("ConvLSTMDwell", **model_config)
    return model


@pytest.fixture
def sample_dataloader(temp_chunks_file, model_config):
    """Generate a sample data loader for testing."""
    from torch.utils.data import DataLoader

    from leech.dataset import LeechDataset, collate_fn

    dataset = LeechDataset(
        chunk_path=temp_chunks_file,
        model_type="ConvLSTMDwell",
        signal_len=model_config["signal_len"],
        kmer_len=model_config["kmer_len"],
        seq_encoding="base_onehot",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return dataloader
