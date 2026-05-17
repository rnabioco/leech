"""Train/predict coordinate parity.

These tests guard against silent divergence between the chunk-extraction code
paths used during ``leech data prepare`` and ``leech predict``. Both call
``build_leech_read`` + ``LeechRead.get_chunk`` under the hood, but the
positioning parameters flow through three serialization steps in between:

  PrepareConfig -> prepare_config.json -> model config.json -> InferenceConfig

A field that gets dropped, renamed, or type-changed anywhere along that chain
would silently produce inference chunks that disagree with what the model was
trained on. The tests below pin the round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from leech.configs import (
    ChunkConfig,
    MotifConfig,
    PrepareConfig,
    SignalConfig,
)


def _make_prep_config() -> PrepareConfig:
    """Non-default values for every positioning-relevant field, so the
    round-trip is sensitive to drift."""
    return PrepareConfig(
        pod5_path=Path("/dummy.pod5"),
        signal=SignalConfig(
            reverse_signal=True,
            anchor="reference",
            norm_method="zscore",
            refine_signal_map=False,
            refine_scale_iters=3,
            refine_half_bandwidth=7,
            refine_do_rough_rescale=False,
            refine_kmer_center_idx=2,
            pa_mean=120.5,
            pa_stdev=14.0,
        ),
        motif=MotifConfig(
            motif="CCAGGC",
            motif_offset=2,
            motif_reference="fasta",
            skip_motif_indels=True,
        ),
        chunk=ChunkConfig(
            base_justify="end",
            feature_start=-7,
            feature_end=7,
            signal_context=(175, 225),
            kmer_context=5,
        ),
        reference_fasta=Path("/dummy.fa"),
    )


def test_prepare_config_persists_all_positioning_fields():
    """Every field inference reads via ``config.get(...)`` must survive
    PrepareConfig.to_dict()."""
    prep = _make_prep_config()
    d = prep.to_dict()

    # Signal-side fields read by single.py:307-327, 343-378, 436, 468-470
    assert d["anchor"] == "reference"
    assert d["reverse_signal"] is True
    assert d["signal_norm"] == "zscore"
    assert d["refine_signal_map"] is False
    assert d["refine_scale_iters"] == 3
    assert d["refine_half_bandwidth"] == 7
    assert d["refine_do_rough_rescale"] is False
    assert d["refine_kmer_center_idx"] == 2
    assert d["pa_mean"] == 120.5
    assert d["pa_stdev"] == 14.0

    # Motif-side fields read by single.py:296-298, 552-557
    assert d["motif"] == "CCAGGC"
    assert d["motif_offset"] == 2
    assert d["motif_reference"] == "fasta"
    assert d["skip_motif_indels"] is True

    # Chunk-side fields read by single.py:396-405, 431-433
    assert d["base_justify"] == "end"
    assert d["feature_start"] == -7
    assert d["feature_end"] == 7
    assert d["kmer_context"] == 5
    # signal_context is serialized as list (JSON-safe), reconstructed as tuple
    assert d["signal_context"] == [175, 225]

    # Reference fasta path persisted as string
    assert d["reference_fasta"] == "/dummy.fa"


def test_signal_context_train_to_inference_roundtrip():
    """training.py writes signal_context as left_context+right_context (training.py:1381-1382)
    and single.py reads those two and rebuilds the tuple (single.py:381-386).
    Verify the value survives that split-and-merge."""
    prep_sig_ctx = (175, 225)

    # Mirror what training.py writes
    train_cfg = {
        "left_context": prep_sig_ctx[0],
        "right_context": prep_sig_ctx[1],
        "signal_len": prep_sig_ctx[0] + prep_sig_ctx[1],
    }

    # Mirror what single.py reads (run_inference, lines 381-386)
    left_ctx = train_cfg.get("left_context")
    right_ctx = train_cfg.get("right_context")
    if left_ctx is not None and right_ctx is not None:
        reconstructed = (left_ctx, right_ctx)
    else:
        signal_len = train_cfg["signal_len"]
        reconstructed = (signal_len // 2, signal_len // 2)

    assert reconstructed == prep_sig_ctx


def test_chunk_extraction_invariant_under_roundtrip(sample_leech_read):
    """A chunk produced by the prep-side ChunkConfig must equal the chunk
    produced by an inference-side ChunkConfig reconstructed from the
    serialized prep config.

    The list->tuple conversion of signal_context is the historically
    fragile spot — if either side gets the type wrong, indexing math may
    still work but signal_context comparisons will silently diverge from
    expectations.
    """
    prep_chunk_cfg = ChunkConfig(
        base_justify="center",
        feature_start=-7,
        feature_end=7,
        signal_context=(200, 200),
        kmer_context=5,
    )
    prep = PrepareConfig(
        pod5_path=Path("/dummy.pod5"),
        chunk=prep_chunk_cfg,
    )
    d = prep.to_dict()

    # Reconstruct ChunkConfig the way single.py does (lines 628-634), going
    # through the same list-to-tuple coercion.
    inf_chunk_cfg = ChunkConfig(
        base_justify=d["base_justify"],
        feature_start=d["feature_start"],
        feature_end=d["feature_end"],
        signal_context=tuple(d["signal_context"]),
        kmer_context=d["kmer_context"],
    )

    # Extract chunks at several positions in the synthetic read.
    base_indices = [6, 8, 10, 12, 14]
    for base_idx in base_indices:
        prep_chunk = sample_leech_read.get_chunk(base_idx, config=prep_chunk_cfg)
        inf_chunk = sample_leech_read.get_chunk(base_idx, config=inf_chunk_cfg)
        assert prep_chunk is not None and inf_chunk is not None, (
            f"chunk extraction failed at base_idx={base_idx}"
        )

        # Numeric arrays must be bit-exact.
        np.testing.assert_array_equal(
            prep_chunk["signal"], inf_chunk["signal"], err_msg=f"signal @ {base_idx}"
        )
        np.testing.assert_array_equal(
            prep_chunk["dwell"], inf_chunk["dwell"], err_msg=f"dwell @ {base_idx}"
        )
        np.testing.assert_array_equal(
            prep_chunk["features"], inf_chunk["features"], err_msg=f"features @ {base_idx}"
        )
        np.testing.assert_array_equal(
            prep_chunk["seq_to_sig_map"],
            inf_chunk["seq_to_sig_map"],
            err_msg=f"seq_to_sig_map @ {base_idx}",
        )

        # Scalar fields must match exactly.
        assert prep_chunk["sequence"] == inf_chunk["sequence"]
        assert prep_chunk["focus_signal_pos"] == inf_chunk["focus_signal_pos"]
        assert prep_chunk["base_idx"] == inf_chunk["base_idx"]
        assert prep_chunk["feature_start"] == inf_chunk["feature_start"]
        assert prep_chunk["feature_end"] == inf_chunk["feature_end"]


def test_base_justify_modes_produce_distinct_focus(sample_leech_read):
    """Sanity check that the three base_justify modes actually produce
    different focus positions for a non-trivial dwell window. If they
    collapse to the same value, the round-trip test above would pass
    vacuously — this guards against that."""
    base_idx = 10
    chunks = {}
    for mode in ("start", "center", "end"):
        cfg = ChunkConfig(
            base_justify=mode,
            signal_context=(50, 50),
            kmer_context=3,
        )
        c = sample_leech_read.get_chunk(base_idx, config=cfg)
        assert c is not None
        chunks[mode] = c

    # The focus position WITHIN the chunk is always signal_context[0]
    # (extractor.py:252), so chunks["start"]["focus_signal_pos"] ==
    # chunks["end"]["focus_signal_pos"]. What differs is the underlying
    # signal samples that landed in the chunk window — different
    # focus_sig_pos in the parent read shifts the window.
    sig_start = chunks["start"]["signal"]
    sig_center = chunks["center"]["signal"]
    sig_end = chunks["end"]["signal"]

    # If start, center, end resolve to different parent-signal positions
    # (which they should for any base with dwell > 1), the chunks differ.
    assert not np.array_equal(sig_start, sig_center) or not np.array_equal(sig_center, sig_end), (
        "base_justify modes produced identical chunks; sample_leech_read "
        "may have a degenerate dwell at base_idx=10"
    )
