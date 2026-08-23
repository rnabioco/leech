"""Every base encoder in the tree must agree on the alphabet.

There are five: two in Python (``preparation.encoding.encode_kmer`` and
``encoding.seq_to_int``), one more in ``features.sequence_to_int``, and two in
Rust (``sequence_to_int`` and ``encode_base_onehot`` in
``rust/src/inference_pipeline/features.rs``). They are reached by different
code paths for the same chunk depending on backend and ``seq_encoding``, so a
disagreement about a single letter is a disagreement about the model's input.

``encode_kmer`` mapped only ACGT, so U -- present in RNA references and some
basecaller output -- became an all-zero column there and a T everywhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from leech.features import sequence_to_int
from leech.preparation.encoding import encode_kmer, seq_to_int

ACGT = "ACGT"


class TestUracilFoldsOntoT:
    def test_encode_kmer_matches_t(self):
        np.testing.assert_array_equal(encode_kmer("ACGU").numpy(), encode_kmer("ACGT").numpy())

    def test_encode_kmer_lowercase_matches(self):
        np.testing.assert_array_equal(encode_kmer("acgu").numpy(), encode_kmer("ACGT").numpy())

    def test_sequence_to_int_matches_t(self):
        np.testing.assert_array_equal(sequence_to_int("ACGU"), sequence_to_int("ACGT"))

    def test_seq_to_int_matches_t(self):
        np.testing.assert_array_equal(seq_to_int("ACGU"), seq_to_int("ACGT"))


class TestEncodersAgree:
    @pytest.mark.parametrize("seq", ["ACGT", "ACGU", "acgt", "acgu", "ACGTN", "ACGUN"])
    def test_onehot_matches_int_encoding(self, seq):
        """``encode_kmer`` and ``sequence_to_int`` must pick the same row."""
        onehot = encode_kmer(seq).numpy()
        ints = sequence_to_int(seq)
        for col, base_int in enumerate(ints):
            column = onehot[:, col]
            if base_int < 0:  # unknown -> all-zero column
                assert column.sum() == 0, f"{seq!r} col {col}"
            else:
                assert column.argmax() == base_int and column.sum() == 1, f"{seq!r} col {col}"

    @pytest.mark.parametrize("seq", ["ACGT", "ACGU", "acgu", "ACGUN"])
    def test_matches_rust_encoder(self, seq):
        """The Rust one-hot encoder must produce the same array."""
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_encode_signal_kmer

        if not HAS_RUST or _rs_encode_signal_kmer is None:
            pytest.skip("leech_core not available")

        # encode_signal_kmer with zero context and one signal sample per base
        # reduces to a per-base one-hot, which is what encode_kmer produces.
        ints = sequence_to_int(seq).astype(np.int8)
        sig_map = np.arange(len(seq) + 1, dtype=np.int64)
        rust = np.asarray(_rs_encode_signal_kmer(ints, sig_map, len(seq), 0, 0))
        np.testing.assert_array_equal(rust, encode_kmer(seq).numpy())
