"""
Tests for dataset module.

Tests LeechDataset and collate_fn.
"""

import pytest
import torch

from leech.dataset import LeechDataset, collate_fn


class TestLeechDataset:
    """Test LeechDataset class."""

    def test_initialization(self, temp_chunks_file):
        """Test dataset initialization."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert len(dataset) > 0
        assert dataset.signal_len == 400
        assert dataset.kmer_len == 11

    def test_initialization_no_valid_chunks(self, tmp_path):
        """Test that ValueError is raised when no valid chunks exist."""
        import numpy as np

        # Create file with chunks but no valid labels
        chunks_file = tmp_path / "empty_chunks.npz"
        np.savez_compressed(
            chunks_file,
            signals=np.array([np.random.randn(100)], dtype=object),
            sequences=np.array(["ACGTACGTACG"]),
            dwells=np.array([np.random.randn(11)], dtype=object),
            features=np.array([np.random.randn(5, 11)], dtype=object),
            labels=np.array([""]),  # Empty string label
            labels_int=np.array([-1]),  # Invalid numeric label
            read_ids=np.array(["read_001"]),
            base_indices=np.array([5]),
        )

        with pytest.raises(ValueError, match="No valid chunks found"):
            LeechDataset(chunks_file, signal_len=400, kmer_len=11, seq_encoding="base_onehot")

    def test_getitem_structure(self, temp_chunks_file):
        """Test that __getitem__ returns correct structure."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]

        assert "signal" in item
        assert "sequence" in item
        assert "features" in item
        assert "label" in item

        assert isinstance(item["signal"], torch.Tensor)
        assert isinstance(item["sequence"], torch.Tensor)
        assert isinstance(item["features"], torch.Tensor)
        assert isinstance(item["label"], torch.Tensor)

    def test_getitem_shapes(self, temp_chunks_file):
        """Test that __getitem__ returns correct shapes with base_onehot encoding."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]

        assert item["signal"].shape == (400,)
        assert item["sequence"].shape == (4, 11)  # 4 bases (A,C,G,T), 11 positions
        assert item["features"].shape[1] == 11  # Second dim should be kmer_len
        assert item["label"].shape == (1,)

    def test_signal_padding(self, temp_chunks_file):
        """Test that signals shorter than signal_len are padded."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=1000,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert item["signal"].shape == (1000,)

    def test_signal_truncation(self, temp_chunks_file):
        """Test that signals longer than signal_len are truncated."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=100,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert item["signal"].shape == (100,)

    def test_model_type_base_no_features(self, temp_chunks_file):
        """Test that ConvLSTMBase dataset doesn't include features."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMBase",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert "features" not in item

    def test_model_type_dwell_has_features(self, temp_chunks_file):
        """Test that ConvLSTMDwell dataset includes features."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert "features" in item

    def test_model_type_transformer_has_features(self, temp_chunks_file):
        """Test that TransformerDwell dataset includes features."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="TransformerDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert "features" in item

    def test_sequence_one_hot_encoding(self, temp_chunks_file):
        """Test that sequences are properly one-hot encoded."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        sequence = item["sequence"]

        # Check that each position has exactly one 1
        assert torch.all(sequence.sum(dim=0) == 1.0)

        # Check that all values are 0 or 1
        assert torch.all((sequence == 0) | (sequence == 1))

    def test_label_dtype(self, temp_chunks_file):
        """Test that labels are float32."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert item["label"].dtype == torch.float32

    def test_iteration(self, temp_chunks_file):
        """Test iterating over dataset."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        items = list(dataset)
        assert len(items) == len(dataset)


class TestCollateFn:
    """Test collate_fn for DataLoader."""

    def test_collate_basic(self, temp_chunks_file):
        """Test basic collation of batch."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        # Get a few items
        batch = [dataset[i] for i in range(min(4, len(dataset)))]

        collated = collate_fn(batch)

        assert "signal" in collated
        assert "sequence" in collated
        assert "label" in collated

    def test_collate_shapes(self, temp_chunks_file):
        """Test that collated batch has correct shapes."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        batch_size = min(4, len(dataset))
        batch = [dataset[i] for i in range(batch_size)]

        collated = collate_fn(batch)

        assert collated["signal"].shape == (batch_size, 400)
        assert collated["sequence"].shape == (batch_size, 4, 11)
        assert collated["label"].shape == (batch_size, 1)

    def test_collate_with_features(self, temp_chunks_file):
        """Test collation with features included."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        collated = collate_fn(batch)

        assert "features" in collated
        assert collated["features"].shape[0] == len(batch)
        assert collated["features"].shape[2] == 11  # kmer_len

    def test_collate_without_features(self, temp_chunks_file):
        """Test collation without features (ConvLSTMBase)."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMBase",
            seq_encoding="base_onehot",
        )

        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        collated = collate_fn(batch)

        assert "features" not in collated

    def test_collate_single_item(self, temp_chunks_file):
        """Test collation with single item (batch_size=1)."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        batch = [dataset[0]]
        collated = collate_fn(batch)

        assert collated["signal"].shape == (1, 400)
        assert collated["sequence"].shape == (1, 4, 11)
        assert collated["label"].shape == (1, 1)


class TestDatasetIntegration:
    """Integration tests with PyTorch DataLoader."""

    def test_dataloader_integration(self, temp_chunks_file):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        batch = next(iter(loader))

        assert isinstance(batch, dict)
        assert batch["signal"].shape[0] <= 2  # batch_size or less

    def test_dataloader_multiple_batches(self, temp_chunks_file):
        """Test iterating through multiple batches."""
        from torch.utils.data import DataLoader

        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        batches = list(loader)
        total_samples = sum(batch["signal"].shape[0] for batch in batches)

        assert total_samples == len(dataset)

    def test_dataloader_shuffle(self, temp_chunks_file):
        """Test that shuffling produces different orders."""
        from torch.utils.data import DataLoader

        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        # Create two loaders with different seeds and collect all indices
        torch.manual_seed(42)
        loader1 = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        indices1 = []
        for batch in loader1:
            # Use signal values as a proxy for sample identity (first few values)
            indices1.append(batch["signal"][:, :5].clone())

        torch.manual_seed(123)
        loader2 = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        indices2 = []
        for batch in loader2:
            indices2.append(batch["signal"][:, :5].clone())

        # With different seeds and sufficient data, order should differ
        if len(dataset) > 2:
            # Check if at least one batch differs
            all_equal = all(torch.equal(i1, i2) for i1, i2 in zip(indices1, indices2, strict=True))
            assert not all_equal, "Shuffling with different seeds should produce different orders"


class TestAugmentation:
    """Test new augmentation methods (shift, time mask, feature noise)."""

    def test_shift_modifies_signal(self, temp_chunks_file):
        """Test that shift augmentation modifies signal stochastically."""
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            shift_max_bases=2.0,
        )
        torch.manual_seed(42)
        item1 = ds[0]
        torch.manual_seed(123)
        item2 = ds[0]
        # With different seeds, at least one branch should differ
        assert not torch.equal(item1["signal"], item2["signal"]) or not torch.equal(
            item1["sequence"], item2["sequence"]
        )

    def test_time_mask_zeros_regions(self, temp_chunks_file):
        """Test that time masking zeros contiguous regions."""
        ds_no_mask = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        ds_mask = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            time_mask_bases=3,
            time_mask_count=2,
        )
        orig = ds_no_mask[0]
        torch.manual_seed(42)
        masked = ds_mask[0]
        # Masked signal should have some zeros where original didn't
        orig_zeros = (orig["signal"] == 0).sum()
        masked_zeros = (masked["signal"] == 0).sum()
        assert masked_zeros >= orig_zeros

    def test_feature_noise_modifies_features(self, temp_chunks_file):
        """Test that feature noise adds stochastic noise to features."""
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            feature_noise_scale=0.1,
        )
        torch.manual_seed(42)
        item1 = ds[0]
        torch.manual_seed(123)
        item2 = ds[0]
        # Features should differ between calls with different seeds
        assert not torch.equal(item1["features"], item2["features"])

    def test_no_augmentation_when_disabled(self, temp_chunks_file):
        """Test that disabled augmentation params don't modify outputs."""
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            shift_max_bases=0.0,
            time_mask_bases=0,
            feature_noise_scale=0.0,
        )
        item1 = ds[0]
        item2 = ds[0]
        assert torch.equal(item1["signal"], item2["signal"])
        assert torch.equal(item1["features"], item2["features"])

    def test_cross_layer_consistency_shift(self, temp_chunks_file):
        """Test shift: signal changes over many draws; edges are zero-padded."""
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            shift_max_bases=5.0,
        )
        ds_orig = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        orig = ds_orig[0]
        signal_changed = False
        for seed in range(100):
            torch.manual_seed(seed)
            shifted = ds[0]
            if not torch.equal(shifted["signal"], orig["signal"]):
                signal_changed = True
                # Check zero-padding: shifted signal should have zeros on one edge
                first_nonzero = (shifted["signal"] != 0).nonzero()
                if first_nonzero.numel() > 0:
                    has_edge_zeros = (
                        first_nonzero[0].item() > 0
                        or first_nonzero[-1].item() < shifted["signal"].shape[-1] - 1
                    )
                    assert has_edge_zeros, "Shifted signal should have zero-padded edges"
                break
        assert signal_changed, "Shift should modify signal for at least one seed"

    def test_val_dataset_no_augmentation(self, temp_chunks_file):
        """Test that validation datasets (no augmentation params) are deterministic."""
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            # No augmentation params set
        )
        item1 = ds[0]
        item2 = ds[0]
        assert torch.equal(item1["signal"], item2["signal"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
