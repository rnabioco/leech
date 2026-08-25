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


class TestAsymmetricFocusPosition:
    """Test that asymmetric signal_context correctly places the focus base."""

    @staticmethod
    def _make_chunks(signal_len: int, focus_signal_pos: int | None, n: int = 4):
        """Create synthetic chunks with a known signal pattern.

        The signal is filled with a linear ramp so that the exact crop window
        can be verified by checking the first/last values of the output.
        """
        import numpy as np

        chunks = []
        kmer_len = 11
        for i in range(n):
            signal = np.arange(signal_len, dtype=np.float32)  # ramp 0..signal_len-1
            seq_to_sig = np.linspace(0, signal_len, kmer_len + 7, dtype=np.int64)
            chunk = {
                "signal": signal,
                "sequence": "A" * kmer_len,
                "dwell": np.ones(kmer_len, dtype=np.float32) * 10,
                "features": np.random.randn(5, kmer_len).astype(np.float32),
                "label": "pos",
                "label_int": i % 2,
                "read_id": f"read_{i}",
                "base_idx": 5,
                "feature_start": -5,
                "feature_end": 5,
                "source_group": "test",
                "cl_value": None,
                "seq_to_sig_map": seq_to_sig,
                "sequence_with_kmer_context": "A" * (kmer_len + 8),
            }
            if focus_signal_pos is not None:
                chunk["focus_signal_pos"] = focus_signal_pos
            chunks.append(chunk)
        return chunks

    def test_symmetric_no_focus_field(self):
        """Without focus_signal_pos, crop assumes center (backward compat)."""
        # Symmetric: 400-wide signal, left=90, right=90 → crop [110:290]
        chunks = self._make_chunks(signal_len=400, focus_signal_pos=None)
        ds = LeechDataset(
            chunks=chunks,
            signal_len=180,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            left_context=90,
            right_context=90,
        )
        sig = ds[0]["signal"]
        assert sig.shape[-1] == 180
        # Focus at center (200), crop [200-90 : 200+90] = [110:290]
        assert sig[0].item() == 110.0
        assert sig[-1].item() == 289.0

    def test_asymmetric_with_focus_field(self):
        """With focus_signal_pos=90, crop uses stored position."""
        # Asymmetric prepare [90, 450] → 540-wide signal, focus at 90
        chunks = self._make_chunks(signal_len=540, focus_signal_pos=90)
        ds = LeechDataset(
            chunks=chunks,
            signal_len=490,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            left_context=90,
            right_context=400,
        )
        sig = ds[0]["signal"]
        assert sig.shape[-1] == 490
        # Focus at 90, crop [90-90 : 90+400] = [0:490]
        assert sig[0].item() == 0.0
        assert sig[-1].item() == 489.0

    def test_asymmetric_without_focus_field_is_wrong(self):
        """Demonstrate the old bug: without focus_signal_pos on asymmetric data,
        center-assumption crops the wrong region."""
        # Same 540-wide signal but NO focus_signal_pos → falls back to center (270)
        chunks = self._make_chunks(signal_len=540, focus_signal_pos=None)
        ds = LeechDataset(
            chunks=chunks,
            signal_len=490,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            left_context=90,
            right_context=400,
        )
        sig = ds[0]["signal"]
        # Center fallback: crop [270-90 : 270+400] = [180:670]
        # Signal is only 540 wide, so [180:540] + 130 zeros
        assert sig[0].item() == 180.0  # WRONG: should be 0.0
        # Last non-zero should be 539 (end of signal), rest zero-padded
        assert sig[359].item() == 539.0
        assert sig[360].item() == 0.0  # zero-padded past signal end

    def test_focus_field_roundtrips_through_serialization(self, tmp_path):
        """Test that focus_signal_pos survives save→load."""
        from leech.chunking.serialization import load_chunks, save_chunks

        chunks = self._make_chunks(signal_len=540, focus_signal_pos=90)
        npz_path = tmp_path / "test.npz"
        save_chunks(chunks, npz_path)
        loaded = load_chunks(npz_path)

        assert loaded[0]["focus_signal_pos"] == 90
        assert loaded[-1]["focus_signal_pos"] == 90

    def test_focus_field_absent_in_old_data(self, tmp_path):
        """Test backward compat: old NPZ files without focus_signal_pos."""
        from leech.chunking.serialization import load_chunks, save_chunks

        chunks = self._make_chunks(signal_len=400, focus_signal_pos=None)
        # Remove the field before saving (simulate old data)
        for c in chunks:
            c.pop("focus_signal_pos", None)
        npz_path = tmp_path / "old.npz"
        save_chunks(chunks, npz_path)
        loaded = load_chunks(npz_path)

        assert "focus_signal_pos" not in loaded[0]

    def test_extractor_stores_focus_signal_pos(self, sample_leech_read):
        """Test that the chunk extractor stores focus_signal_pos."""
        chunk = sample_leech_read.get_chunk(
            base_idx=10,
            signal_context=(90, 450),
            kmer_context=5,
        )
        assert chunk is not None
        assert chunk["focus_signal_pos"] == 90

        # Symmetric context should also store it
        chunk_sym = sample_leech_read.get_chunk(
            base_idx=10,
            signal_context=(200, 200),
            kmer_context=5,
        )
        assert chunk_sym is not None
        assert chunk_sym["focus_signal_pos"] == 200


class TestBatchedFetch:
    """``__getitems__`` gathers a whole batch instead of one row at a time.

    ``DataLoader`` calls it in place of N ``__getitem__`` calls plus a
    ``collate_fn`` stack (torch >= 2.0). The gate is that it changes nothing:
    the batch it returns must equal the one the per-sample path produces, bit
    for bit, for every option that changes what a sample contains.
    """

    OPTIONS = {
        "signal_len": 400,
        "kmer_len": 11,
        "model_type": "ConvLSTMDwell",
        "seq_encoding": "base_onehot",
    }

    def _dataset(self, path, **overrides):
        return LeechDataset(chunk_path=path, **{**self.OPTIONS, **overrides})

    def _assert_same_batch(self, dataset, indices):
        batched = collate_fn(dataset.__getitems__(indices))
        per_sample = collate_fn([dataset[i] for i in indices])
        assert batched.keys() == per_sample.keys()
        for key in batched:
            assert batched[key].dtype == per_sample[key].dtype, key
            assert batched[key].shape == per_sample[key].shape, key
            assert torch.equal(batched[key], per_sample[key]), key

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"model_type": "ConvLSTMBase"},
            {"model_type": "TCNDwellResidualLN"},
            {"seq_encoding": "signal_kmer", "signal_kmer_context": (2, 2)},
            {"cl_regression": True},
            {"signal_mode": "signal"},
            {"dwell_offset": 1},
        ],
        ids=["default", "no_features", "wide_features", "signal_kmer", "cl", "signal_only", "off1"],
    )
    def test_batch_equals_per_sample(self, temp_chunks_file, overrides):
        dataset = self._dataset(temp_chunks_file, **overrides)
        assert dataset._batched_fetch
        self._assert_same_batch(dataset, list(range(len(dataset))))
        self._assert_same_batch(dataset, [len(dataset) - 1, 0, 1])  # out of order

    def test_batch_equals_per_sample_with_a_confound(self, temp_chunks_file):
        from leech.confounds import ConfoundEncoder

        encoder = ConfoundEncoder(
            name="grp", source="label_int", value_to_class={0: 0, 1: 1}, num_classes=2
        )
        dataset = self._dataset(temp_chunks_file, confound_encoder=encoder)
        assert "confound_label" in dataset[0]
        self._assert_same_batch(dataset, list(range(len(dataset))))

    def test_collate_passes_an_already_collated_batch_through(self, temp_chunks_file):
        dataset = self._dataset(temp_chunks_file)
        batch = dataset.__getitems__([0, 1])
        assert isinstance(batch, dict)
        assert collate_fn(batch) is batch

    def test_dataloader_yields_the_same_batches_either_way(self, temp_chunks_file):
        from torch.utils.data import DataLoader

        dataset = self._dataset(temp_chunks_file)

        def batches():
            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
            return [{k: v.clone() for k, v in b.items()} for b in loader]

        dataset._batched_fetch = True
        with_batched = batches()
        dataset._batched_fetch = False
        without = batches()

        assert len(with_batched) == len(without)
        for left, right in zip(with_batched, without, strict=True):
            assert left.keys() == right.keys()
            for key in left:
                assert torch.equal(left[key], right[key]), key

    def test_falls_back_to_a_list_when_a_field_is_not_stacked(self, temp_chunks_file):
        """Ragged fields keep the per-sample path; there is no batch to gather."""
        dataset = self._dataset(temp_chunks_file)
        dataset._batched_fetch = False
        fetched = dataset.__getitems__([0, 1])
        assert isinstance(fetched, list)
        assert collate_fn(fetched)["signal"].shape[0] == 2

    @pytest.mark.parametrize("option", ["shift_max_bases", "time_mask_bases"])
    def test_cross_layer_augmentation_keeps_the_per_sample_path(self, temp_chunks_file, option):
        """Shift and time mask draw one offset per sample and roll by it."""
        dataset = self._dataset(temp_chunks_file, **{option: 2 if option else 0})
        fetched = dataset.__getitems__([0, 1])
        assert isinstance(fetched, list)

    def test_batched_augmentation_draws_per_sample(self, temp_chunks_file):
        """One scale factor per row, not one for the whole batch.

        A batched ``uniform_(...).item()`` would scale every row by the same
        number — the augmentation would still "work" and every batch would be
        wrong in the same way, which is why this is asserted rather than eyeballed.
        """
        dataset = self._dataset(
            temp_chunks_file, augmentation={"jitter_std": 0.0, "scale_range": (0.5, 1.5)}
        )
        indices = list(range(len(dataset)))
        torch.manual_seed(0)
        batch = collate_fn(dataset.__getitems__(indices))

        stored = dataset._signals_tensor[indices]
        ratios = (batch["signal"] / stored).reshape(len(indices), -1)
        # Constant within a row (one factor per sample) ...
        assert torch.allclose(ratios.min(dim=1).values, ratios.max(dim=1).values, atol=1e-5)
        # ... and different between rows.
        per_row = ratios[:, 0]
        assert per_row.unique().numel() == len(indices)
        assert float(per_row.min()) >= 0.5 and float(per_row.max()) <= 1.5

    def test_batched_jitter_is_per_element(self, temp_chunks_file):
        dataset = self._dataset(temp_chunks_file, augmentation={"jitter_std": 0.05})
        indices = list(range(len(dataset)))
        torch.manual_seed(0)
        batch = collate_fn(dataset.__getitems__(indices))
        noise = batch["signal"] - dataset._signals_tensor[indices]
        assert noise.abs().max() > 0
        # Independent draws: no two rows share their noise vector.
        assert not torch.equal(noise[0], noise[1])
        assert abs(float(noise.std()) - 0.05) < 0.02

    def test_batched_feature_noise_is_per_element(self, temp_chunks_file):
        dataset = self._dataset(temp_chunks_file, feature_noise_scale=0.5)
        indices = list(range(len(dataset)))
        torch.manual_seed(0)
        batch = collate_fn(dataset.__getitems__(indices))
        noise = batch["features"] - dataset._features_tensor[indices]
        assert noise.abs().max() > 0
        assert not torch.equal(noise[0], noise[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
