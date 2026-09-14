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
            {"label_noise_rates": {"unknown": 0.3}},
        ],
        ids=[
            "default",
            "no_features",
            "wide_features",
            "signal_kmer",
            "cl",
            "signal_only",
            "off1",
            "label_noise_rate",
        ],
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


class TestStandardizeFeatures:
    """Corpus-wide per-channel feature mean/std for --standardize-features
    (issue #283). The dataset only ever computes these; it never applies
    them to its own output -- see the module docstring on
    ``LeechDataset.__init__``'s ``standardize_features`` argument."""

    def test_disabled_by_default(self, temp_chunks_file):
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert ds.feature_mean is None
        assert ds.feature_std is None

    def test_computes_per_channel_mean_and_std(self, temp_chunks_file):
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            standardize_features=True,
        )
        assert ds.feature_mean is not None
        assert ds.feature_std is not None
        num_features = ds._features_tensor.shape[1]
        assert ds.feature_mean.shape == (num_features,)
        assert ds.feature_std.shape == (num_features,)
        # Collapsed over BOTH the chunk axis and the window axis -- one
        # number per channel, not one per (channel, position) the way the
        # feature-noise std is.
        expected_mean = ds._features_tensor.mean(dim=(0, 2))
        expected_std = ds._features_tensor.std(dim=(0, 2))
        torch.testing.assert_close(ds.feature_mean, expected_mean)
        torch.testing.assert_close(ds.feature_std, expected_std)

    def test_not_computed_for_a_model_that_ignores_features(self, temp_chunks_file):
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="SignalCNN",
            seq_encoding="base_onehot",
            standardize_features=True,
        )
        assert ds.feature_mean is None
        assert ds.feature_std is None

    def test_dataset_output_is_unaffected(self, temp_chunks_file):
        """The transform lives in the model, not the dataset -- __getitem__
        must return the same raw features whether or not stats were
        computed."""
        plain = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        standardized = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            standardize_features=True,
        )
        torch.testing.assert_close(plain[0]["features"], standardized[0]["features"])


class TestTimeStretchPrimitives:
    """The pure numpy warp functions ``__getitems__`` composes (issue #281)."""

    def test_signal_identity_at_factor_one(self):
        import numpy as np

        from leech.dataset import _time_stretch_signal_rows

        rng = np.random.default_rng(0)
        signal = rng.standard_normal((3, 50)).astype(np.float32)
        factors = np.ones(3, dtype=np.float32)
        out = _time_stretch_signal_rows(signal, factors, focus_idx=25)
        assert np.array_equal(out, signal)

    def test_map_identity_at_factor_one(self):
        import numpy as np

        from leech.dataset import _time_stretch_seq_to_sig_rows

        seq_to_sig = np.array([[0, 20, 40, 60, 80, 100]], dtype=np.int32)
        factors = np.ones(1, dtype=np.float32)
        out = _time_stretch_seq_to_sig_rows(seq_to_sig, factors, focus_idx=50, signal_len=100)
        assert np.array_equal(out, seq_to_sig)

    def test_map_sentinel_is_never_scaled(self):
        """A value equal to signal_len (padding, or a row's own right-edge
        terminator) must stay signal_len exactly -- scaling it would turn a
        boundary marker `encode_signal_kmer_batch` relies on into a stray
        interior position."""
        import numpy as np

        from leech.dataset import _time_stretch_seq_to_sig_rows

        seq_to_sig = np.array([[10, 50, 100, 100]], dtype=np.int32)  # padded row
        factors = np.array([1.8], dtype=np.float32)
        out = _time_stretch_seq_to_sig_rows(seq_to_sig, factors, focus_idx=50, signal_len=100)
        assert out[0, 2] == 100
        assert out[0, 3] == 100

    def test_map_boundary_stretching_past_the_edge_clips_to_the_sentinel(self):
        import numpy as np

        from leech.dataset import _time_stretch_seq_to_sig_rows

        # focus=50, factor=3: a boundary at 90 would scale to 50+(90-50)*3=170,
        # well past the 100-sample window -- clip to signal_len (padding).
        seq_to_sig = np.array([[90, 100]], dtype=np.int32)
        factors = np.array([3.0], dtype=np.float32)
        out = _time_stretch_seq_to_sig_rows(seq_to_sig, factors, focus_idx=50, signal_len=100)
        assert out[0, 0] == 100

    def test_signal_and_map_stay_aligned(self):
        """The scaled map, used to index the stretched signal, recovers the
        signal value the un-stretched map pointed at -- the alignment
        acceptance criterion, checked directly against the pure functions.

        A linear ramp signal makes this exact up to the map value's rounding
        to the nearest integer sample: interpolating a linear function at any
        fractional position reproduces that position's value exactly, so the
        only error is from `round(scaled_position) != scaled_position`.
        """
        import numpy as np

        from leech.dataset import _time_stretch_seq_to_sig_rows, _time_stretch_signal_rows

        signal_len = 100
        focus_idx = 50
        factor = 1.2
        signal = np.arange(signal_len, dtype=np.float32).reshape(1, signal_len)
        factors = np.array([factor], dtype=np.float32)
        # Boundaries chosen so the scaled position stays in-window (no clip).
        original_boundaries = np.array([30, 40, 50, 60, 70], dtype=np.int32)
        seq_to_sig = original_boundaries.reshape(1, -1)

        stretched_signal = _time_stretch_signal_rows(signal, factors, focus_idx)
        stretched_map = _time_stretch_seq_to_sig_rows(seq_to_sig, factors, focus_idx, signal_len)

        for j, p in enumerate(original_boundaries):
            scaled = int(stretched_map[0, j])
            assert 0 <= scaled < signal_len
            recovered = stretched_signal[0, scaled]
            # Rounding the scaled position to the nearest sample introduces at
            # most 0.5 samples of source-position error, i.e. 0.5/factor here.
            assert abs(float(recovered) - float(p)) < 1.0

    def test_signal_stretch_widens_the_window_about_the_focus(self):
        """A factor > 1 samples further from the focus per output step."""
        import numpy as np

        from leech.dataset import _time_stretch_signal_rows

        signal = np.arange(100, dtype=np.float32).reshape(1, 100)
        out = _time_stretch_signal_rows(signal, np.array([2.0]), focus_idx=50)
        # Output sample 60 reads from source 50 + (60-50)/2 = 55.
        assert abs(float(out[0, 60]) - 55.0) < 1e-4
        # The focus sample itself never moves.
        assert abs(float(out[0, 50]) - 50.0) < 1e-4


class TestTimeStretch:
    """``LeechDataset(time_stretch_range=...)`` (issue #281)."""

    def test_disabled_by_default(self, temp_chunks_file):
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert ds._time_stretch_range == (1.0, 1.0)

    def test_one_one_is_bitwise_identity(self, temp_chunks_file):
        """(1.0, 1.0) must reproduce the un-augmented batch bit for bit."""
        baseline = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        stretched = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            time_stretch_range=(1.0, 1.0),
        )
        indices = list(range(len(baseline)))
        expected = collate_fn(baseline.__getitems__(indices))
        actual = collate_fn(stretched.__getitems__(indices))
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), key

    def test_one_one_is_bitwise_identity_with_signal_kmer(self, temp_chunks_file):
        baseline = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
            signal_kmer_context=(2, 2),
        )
        stretched = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
            signal_kmer_context=(2, 2),
            time_stretch_range=(1.0, 1.0),
        )
        indices = list(range(len(baseline)))
        expected = collate_fn(baseline.__getitems__(indices))
        actual = collate_fn(stretched.__getitems__(indices))
        for key in expected:
            assert torch.equal(expected[key], actual[key]), key

    def test_changes_the_signal(self, temp_chunks_file):
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            time_stretch_range=(1.5, 1.5),
        )
        indices = list(range(len(ds)))
        batch = collate_fn(ds.__getitems__(indices))
        assert not torch.equal(batch["signal"], ds._signals_tensor[indices])

    def test_anchors_on_focus_signal_pos_not_signal_len_over_two(self):
        """Regression: a corpus prepared with an asymmetric ``--signal-context``
        but trained without a train-time asymmetric crop (the common case --
        ``leech model train`` exposes no singular ``--left-context``/
        ``--right-context`` override) must anchor the stretch at the corpus's
        own ``focus_signal_pos``, not ``signal_len // 2``.

        E.g. ``data prepare --signal-context 100 300`` stores
        ``focus_signal_pos=100`` in a 400-sample window; assuming the focus
        sits at 200 (the symmetric-prepare center) silently decorrelates the
        stretched signal, map and features from the real motif position.
        """
        import numpy as np

        signal_len = 400
        true_focus = 100  # e.g. --signal-context 100 300
        wrong_focus = signal_len // 2  # the bug's assumption

        chunks = [
            {
                "signal": np.arange(signal_len, dtype=np.float32),
                "sequence": "A" * 11,
                "label": "pos",
                "label_int": i % 2,
                "read_id": f"read_{i}",
                "base_idx": 5,
                "focus_signal_pos": true_focus,
            }
            for i in range(3)
        ]

        ds = LeechDataset(
            chunks=chunks,
            signal_len=signal_len,
            kmer_len=11,
            model_type="ConvLSTMBase",  # no features; keep the fixture minimal
            seq_encoding="base_onehot",
            time_stretch_range=(2.0, 2.0),
        )
        # The realistic case this regresses: no asymmetric train-time crop.
        assert ds.left_context is None and ds.right_context is None

        rows = np.arange(3)
        batch = collate_fn(ds.__getitems__(list(rows)))
        original = ds._signals_tensor[rows]

        # The true focus sample must be exactly unchanged...
        assert torch.equal(batch["signal"][:, true_focus], original[:, true_focus])
        # ...while signal_len // 2 is NOT the invariant point: it moved, which
        # proves this test would have failed against the old
        # `focus_idx = signal_len // 2` assumption instead of passing vacuously.
        assert not torch.equal(batch["signal"][:, wrong_focus], original[:, wrong_focus])

    def test_anchors_on_focus_signal_pos_block_path(self, tmp_path):
        """Same regression as above, through the npz-backed block-wise filler."""
        import numpy as np

        from leech.chunking.serialization import save_chunks

        signal_len = 400
        true_focus = 100
        wrong_focus = signal_len // 2

        chunks = [
            {
                "signal": np.arange(signal_len, dtype=np.float32),
                "sequence": "A" * 11,
                "dwell": np.ones(11, dtype=np.float32),
                "features": np.zeros((2, 11), dtype=np.float32),
                "label": "pos",
                "label_int": i % 2,
                "read_id": f"read_{i}",
                "base_idx": 5,
                "focus_signal_pos": true_focus,
            }
            for i in range(3)
        ]
        npz_path = tmp_path / "asymmetric.npz"
        save_chunks(chunks, npz_path)

        ds = LeechDataset(
            npz_path,
            signal_len=signal_len,
            kmer_len=11,
            model_type="ConvLSTMBase",
            seq_encoding="base_onehot",
            time_stretch_range=(2.0, 2.0),
        )
        assert ds.left_context is None and ds.right_context is None

        rows = np.arange(3)
        batch = collate_fn(ds.__getitems__(list(rows)))
        original = ds._signals_tensor[rows]

        assert torch.equal(batch["signal"][:, true_focus], original[:, true_focus])
        assert not torch.equal(batch["signal"][:, wrong_focus], original[:, wrong_focus])

    def test_shares_one_factor_and_focus_across_signal_channels(self):
        """Regression for the ``(B, 2, L)`` branch (``signal_mode="both"`` on
        a corpus with a residual channel): the residual channel must warp
        with the exact same per-row factor and focus as the primary signal,
        not independently and not left unwarped.

        A constant offset between the two channels' ramps commutes exactly
        through linear interpolation only when both are resampled at the same
        fractional position with the same weights, so ``channel1 - channel0``
        staying exactly the offset everywhere is what proves they share one
        factor/focus; an independent draw, a focus mismatch, or one channel
        skipped would all show up as a non-constant difference.
        """
        import numpy as np

        signal_len = 400
        offset = 1000.0
        chunks = [
            {
                "signal": np.arange(signal_len, dtype=np.float32),
                "signal_residual": np.arange(signal_len, dtype=np.float32) + offset,
                "sequence": "A" * 11,
                "label": "pos",
                "label_int": i % 2,
                "read_id": f"read_{i}",
                "base_idx": 5,
                "focus_signal_pos": signal_len // 2,
            }
            for i in range(3)
        ]

        ds = LeechDataset(
            chunks=chunks,
            signal_len=signal_len,
            kmer_len=11,
            model_type="ConvLSTMBase",
            seq_encoding="base_onehot",
            signal_mode="both",
            time_stretch_range=(2.0, 2.0),
        )
        assert ds.signal_channels == 2

        batch = collate_fn(ds.__getitems__([0, 1, 2]))
        signal = batch["signal"]
        assert signal.shape == (3, 2, signal_len)

        # factor=2.0 about the center of a 400-sample ramp keeps every output
        # position's source lookup within [0, L), so nothing here is
        # zero-padded -- a clean exact check, no edge cases to special-case.
        diff = signal[:, 1, :] - signal[:, 0, :]
        assert torch.allclose(diff, torch.full_like(diff, offset), atol=1e-3)

        # And it actually warped -- not a no-op on the raw ramp.
        raw_ramp = torch.arange(signal_len, dtype=torch.float32).unsqueeze(0).repeat(3, 1)
        assert not torch.equal(signal[:, 0, :], raw_ramp)

    def test_feature_channel_rule(self, temp_chunks_file):
        """dwell/dwell_mean *= factor, dwell_log += log(factor), dwell_ratio
        and dwell_std unchanged -- exercised directly against
        ``_apply_time_stretch`` with a hand-built 5-channel feature tensor, so
        the assertion doesn't depend on ``temp_chunks_file``'s (deliberately
        reduced, for other tests) feature channel semantics.
        """
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=100,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            time_stretch_range=(1.5, 1.5),
        )
        import numpy as np

        batch, kmer_len = 4, 11
        dwell = torch.full((batch, kmer_len), 10.0)
        dwell_log = torch.log(dwell)
        dwell_mean = torch.full((batch, kmer_len), 12.0)
        dwell_std = torch.full((batch, kmer_len), 2.0)
        dwell_ratio = dwell / dwell_mean
        features = torch.stack([dwell, dwell_log, dwell_mean, dwell_std, dwell_ratio], dim=1)
        signal = torch.arange(100, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)

        _, _, features_out = ds._apply_time_stretch(signal, None, features, np.arange(batch))

        factor = 1.5
        assert torch.allclose(features_out[:, 0], dwell * factor)
        assert torch.allclose(features_out[:, 2], dwell_mean * factor)
        assert torch.allclose(
            features_out[:, 1], dwell_log + torch.log(torch.tensor(factor)), atol=1e-6
        )
        assert torch.equal(features_out[:, 3], dwell_std)
        assert torch.equal(features_out[:, 4], dwell_ratio)

    def test_feature_channel_rule_indices_match_compute_dwell_features(self):
        """``test_feature_channel_rule`` locks indices 0/1/2 to
        dwell/dwell_log/dwell_mean against a hand-built tensor; this locks
        that same assumption against the real producer
        (``compute_dwell_features``, whose dict order ``merge_feature_channels``
        places first in every chunk's feature rows -- see
        ``chunking/extractor.py``'s ``merge_feature_channels`` docstring and
        ``tests/test_data_prep.py::TestFeatureChannelOrder``, which locks the
        write side). If a future edit reorders ``compute_dwell_features``'
        dict, this is what would fail on the read side -- otherwise
        ``_apply_time_stretch`` would silently scale the wrong channels with
        no test catching it.
        """
        import numpy as np

        from leech.features import compute_dwell_features

        dwells = np.array([8, 10, 12, 9, 11], dtype=np.int64)
        channel_names = list(compute_dwell_features(dwells).keys())
        assert channel_names[:5] == [
            "dwell",
            "dwell_log",
            "dwell_mean",
            "dwell_std",
            "dwell_ratio",
        ]

    def test_focus_clamped_when_crop_excludes_it(self, temp_chunks_file):
        """A ``signal_len`` far smaller than the stored width can center-crop
        the focus out of the window entirely (crop_start > focus_signal_pos);
        the computed anchor must stay a valid index into the final tensor
        rather than going negative (or past its end), even though no anchor
        is really "correct" once the crop has already excluded the focus.
        """
        import numpy as np

        ds = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        # A stored width far larger than signal_len simulates
        # `data prepare --signal-context <small> <huge>` trained without a
        # matching asymmetric --left-context/--right-context override.
        huge_stored_len = 10 * ds.signal_len
        focus_idx = ds._time_stretch_focus_for_chunk(
            focus_signal_pos=10, stored_len=huge_stored_len
        )
        assert 0 <= focus_idx < ds.signal_len

        column = ds._time_stretch_focus_column(huge_stored_len)
        assert np.all(column >= 0) and np.all(column < ds.signal_len)

    def test_feature_scaling_skipped_below_five_channels(self, temp_chunks_file):
        """A corpus without the standard 5-channel dwell block is left alone
        rather than having an arbitrary column scaled as if it were dwell."""
        ds = LeechDataset(
            temp_chunks_file,
            signal_len=100,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            time_stretch_range=(1.5, 1.5),
        )
        import numpy as np

        batch, kmer_len = 3, 11
        features = torch.randn(batch, 2, kmer_len)
        signal = torch.arange(100, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)

        _, _, features_out = ds._apply_time_stretch(signal, None, features, np.arange(batch))
        assert torch.equal(features_out, features)

    def test_conflicts_with_shift_raises(self, temp_chunks_file):
        with pytest.raises(ValueError, match="shift_max_bases"):
            LeechDataset(
                temp_chunks_file,
                signal_len=400,
                kmer_len=11,
                model_type="ConvLSTMDwell",
                seq_encoding="base_onehot",
                time_stretch_range=(0.8, 1.25),
                shift_max_bases=2.0,
            )

    def test_conflicts_with_time_mask_raises(self, temp_chunks_file):
        with pytest.raises(ValueError, match="time_mask_bases"):
            LeechDataset(
                temp_chunks_file,
                signal_len=400,
                kmer_len=11,
                model_type="ConvLSTMDwell",
                seq_encoding="base_onehot",
                time_stretch_range=(0.8, 1.25),
                time_mask_bases=3,
            )

    @pytest.mark.parametrize("bad_range", [(1.2, 0.8), (0.0, 1.2), (-0.5, 1.2)])
    def test_invalid_range_raises(self, temp_chunks_file, bad_range):
        with pytest.raises(ValueError, match="time_stretch_range"):
            LeechDataset(
                temp_chunks_file,
                signal_len=400,
                kmer_len=11,
                model_type="ConvLSTMDwell",
                seq_encoding="base_onehot",
                time_stretch_range=bad_range,
            )

    def test_requires_batched_fetch(self):
        """A corpus that degrades to the per-chunk list fallback has no
        batched implementation of time-stretch to run -- raise at
        construction instead of silently skipping the augmentation."""
        import numpy as np

        chunks = []
        for i in range(2):
            chunk = {
                "signal": np.zeros(50, dtype=np.float32),
                "sequence": "A" * 11,
                "label": "pos",
                "label_int": i % 2,
                "read_id": f"read_{i}",
                "base_idx": 5,
            }
            if i == 0:
                # Only the first chunk carries a residual channel, so
                # _prepare_signal returns a (2, L) tensor for it and a plain
                # (L,) tensor for the other -- a shape mismatch that degrades
                # the signal field to a per-chunk list.
                chunk["signal_residual"] = np.zeros(50, dtype=np.float32)
            chunks.append(chunk)

        with pytest.raises(ValueError, match="batched __getitems__ fetch path"):
            LeechDataset(
                chunks=chunks,
                signal_len=50,
                kmer_len=11,
                model_type="ConvLSTMBase",
                seq_encoding="base_onehot",
                signal_mode="both",
                time_stretch_range=(1.2, 1.5),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
