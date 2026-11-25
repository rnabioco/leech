"""Same-task N-seed logit-average ensemble and paired-seed statistics (#405)."""

import json
import statistics

import pytest
import torch

from leech.bundling import (
    LogitAverageEnsemble,
    create_seed_ensemble_bundle,
    list_bundle_models,
    load_seed_ensemble,
)
from leech.models import get_model
from leech.seed_stats import DEFAULT_SEEDS, paired_seed_differences

ARCH = {
    "signal_len": 64,
    "kmer_len": 11,
    "num_features": 9,
    "conv_channels": [4, 8, 16],
    "lstm_hidden": 8,
    "dropout": 0.0,
}
CONFIG = {
    "model_name": "ConvLSTMDwell",
    **ARCH,
    "refine_signal_map": False,
    "seq_encoding": "base_onehot",
    "motif": "CCAGGC",
    "motif_offset": 2,
}


def _member(tmp_path, seed, label_map=None):
    d = tmp_path / f"seed{seed}"
    d.mkdir()
    cfg = dict(CONFIG)
    cfg["label_map"] = label_map or {"uncharged": 0, "charged": 1}
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(seed)
    model = get_model("ConvLSTMDwell", **ARCH)
    torch.save({"model_state_dict": model.state_dict()}, d / "model_best.pt")
    return d, model


def _inputs():
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(3, 1, ARCH["signal_len"], generator=g),
        torch.randn(3, 4, ARCH["kmer_len"], generator=g),
        torch.randn(3, ARCH["num_features"], ARCH["kmer_len"], generator=g),
    )


def test_ensemble_forward_is_mean_of_member_logits():
    a = torch.nn.Linear(2, 1)
    b = torch.nn.Linear(2, 1)
    x = torch.randn(5, 2)
    ens = LogitAverageEnsemble([a, b])
    expected = (a(x) + b(x)) / 2
    assert torch.allclose(ens(x), expected)


def test_ensemble_rejects_empty():
    with pytest.raises(ValueError):
        LogitAverageEnsemble([])


def test_bundle_roundtrip_matches_manual_average(tmp_path):
    dirs, models = [], []
    for s in (1, 2, 3):
        d, m = _member(tmp_path, s)
        dirs.append(d)
        models.append(m.eval())
    out = create_seed_ensemble_bundle(dirs, tmp_path / "ens.pt", "0.0.1-test")
    meta = list_bundle_models(out)
    assert meta["comparison_type"] == "seed_ensemble"
    assert meta["num_models"] == 3
    ens, config = load_seed_ensemble(out, device="cpu")
    assert config["model_name"] == "ConvLSTMDwell"
    x = _inputs()
    with torch.no_grad():
        expected = torch.stack([m(*x) for m in models]).mean(0)
        got = ens(*x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_single_member_ensemble_refused(tmp_path):
    d, _ = _member(tmp_path, 1)
    with pytest.raises(ValueError, match="at least 2"):
        create_seed_ensemble_bundle([d], tmp_path / "e.pt", "0.0.1")


def test_different_task_refused(tmp_path):
    d1, _ = _member(tmp_path, 1)
    d2, _ = _member(tmp_path, 2, label_map={"Ala": 0, "Gly": 1})
    with pytest.raises(ValueError, match="same task"):
        create_seed_ensemble_bundle([d1, d2], tmp_path / "e.pt", "0.0.1")


def test_default_seed_count_is_six():
    assert len(DEFAULT_SEEDS) == 6


def _rows(base, arm, n=6):
    rows = []
    for s in range(1, n + 1):
        rows.append({"pair": "P", "arm": "baseline", "seed": s, "auroc": base[s - 1]})
        rows.append({"pair": "P", "arm": "x", "seed": s, "auroc": arm[s - 1]})
    return rows


def test_paired_differences_use_same_seed_and_report_sd():
    base = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    arm = [0.91, 0.93, 0.92, 0.96, 0.95, 0.99]
    out = paired_seed_differences(_rows(base, arm), "baseline", ["auroc"])
    (r,) = [r for r in out if r["arm"] == "x"]
    diffs = [a - b for a, b in zip(arm, base, strict=True)]
    assert r["n_paired_seeds"] == 6
    assert r["auroc_paired_diff_mean"] == pytest.approx(statistics.mean(diffs))
    assert r["auroc_paired_diff_sd"] == pytest.approx(statistics.stdev(diffs))
    assert not [r for r in out if r["arm"] == "baseline"]


def test_pairing_skips_unmatched_seeds_and_sd_nan_for_one_pair():
    rows = _rows([0.9] * 6, [0.95] * 6)
    rows = [r for r in rows if not (r["arm"] == "x" and r["seed"] > 1)]
    (r,) = paired_seed_differences(rows, "baseline", ["auroc"])
    assert r["n_paired_seeds"] == 1
    assert r["auroc_paired_diff_sd"] != r["auroc_paired_diff_sd"]  # NaN
