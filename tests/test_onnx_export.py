"""ONNX export, for the classifier arms and the CRF encoder.

The point of these tests is the **round trip**: exporting and then comparing
onnxruntime against torch across the serialization boundary catches what an
in-process assert cannot. They also pin the two things a consumer needs and
cannot read off the graph — which input is which, and what the output means.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from leech.crf import CrfEncoder, EncoderConfig  # noqa: E402
from leech.crf.export import export_crf_onnx, load_training_sidecar  # noqa: E402
from leech.model_export import export_single_model_onnx  # noqa: E402
from leech.models import get_model  # noqa: E402
from leech.onnx_export import OPSET, describe_inputs, export_onnx, verify_onnx  # noqa: E402

#: float32 machine epsilon. Anything at this order is rounding, not disagreement.
EPS = 1.1920929e-07


# ── the CRF encoder ────────────────────────────────────────────────────────


@pytest.fixture
def crf_run(tmp_path):
    """A checkpoint plus the sidecar a trainer would have written beside it."""
    cfg = EncoderConfig(chunk=300)
    torch.manual_seed(0)
    model = CrfEncoder(cfg)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "model.json").write_text(
        json.dumps(
            {"mean": 61.82, "std": 9.57, "chunk": 300, "selected_epoch": 7, "target_len": 48}
        )
    )
    return tmp_path, cfg, model


def test_crf_export_agrees_with_torch(crf_run):
    run, cfg, _ = crf_run
    export_crf_onnx(run / "model.pt", run / "export", sidecar=run)
    meta = json.loads((run / "export" / "metadata.json").read_text())
    diff = meta["verification"]["onnxruntime_vs_torch_max_abs_diff"]
    assert diff < 1e-5, f"max abs diff {diff:.3e} is too large to be rounding"


def test_crf_export_output_is_time_major(crf_run):
    """`(T, N, n_score)`, not `(N, T, n_score)`.

    The boundary CNN in the same stack is batch-major, so a consumer reusing
    that assumption silently transposes rather than failing.
    """
    import onnxruntime as ort

    run, cfg, _ = crf_run
    path = export_crf_onnx(run / "model.pt", run / "export", sidecar=run, verify=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    out = session.run(None, {"signal": np.random.randn(3, 1, cfg.chunk).astype(np.float32)})[0]
    assert out.shape == (cfg.chunk // cfg.stride, 3, cfg.n_score)


def test_crf_export_batch_axis_is_dynamic(crf_run):
    """Batch is axis 1 of the output because the graph is time-major — getting
    the dynamic axis wrong pins the graph to the batch it was traced at."""
    import onnxruntime as ort

    run, cfg, _ = crf_run
    path = export_crf_onnx(run / "model.pt", run / "export", sidecar=run, verify=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        out = session.run(
            None, {"signal": np.random.randn(batch, 1, cfg.chunk).astype(np.float32)}
        )[0]
        assert out.shape[1] == batch


def test_crf_metadata_carries_the_standardisation(crf_run):
    """It is in neither the config nor the checkpoint, so a consumer that cannot
    read it here cannot standardise and decodes silently worse."""
    run, _, _ = crf_run
    export_crf_onnx(run / "model.pt", run / "export", sidecar=run, verify=False)
    meta = json.loads((run / "export" / "metadata.json").read_text())
    assert meta["standardisation"]["mean"] == 61.82
    assert meta["standardisation"]["stdev"] == 9.57


def test_crf_export_without_standardisation_is_refused(crf_run):
    run, _, _ = crf_run
    with pytest.raises(ValueError, match="standardisation"):
        export_crf_onnx(run / "model.pt", run / "export", verify=False)


def test_crf_metadata_declares_the_score_layout(crf_run):
    """1280, blank at index 0 of each state group — the layout the Rust decoder
    assumes, where a wrong guess keeps every shape valid."""
    run, cfg, _ = crf_run
    export_crf_onnx(run / "model.pt", run / "export", sidecar=run, verify=False)
    crf = json.loads((run / "export" / "metadata.json").read_text())["crf"]
    assert crf["n_score"] == 1280
    assert crf["blank_index"] == 0
    assert crf["scores_per_state"] == cfg.n_base + 1


def test_references_are_written_as_what_the_model_emits(crf_run):
    """`target[state_len:]`. Matching full-length targets calls the same
    sequence but inflates every edit distance."""
    run, cfg, _ = crf_run
    target = "ACGT" * 12
    export_crf_onnx(
        run / "model.pt",
        run / "export",
        sidecar=run,
        verify=False,
        references={"code01": target},
    )
    meta = json.loads((run / "export" / "metadata.json").read_text())
    assert meta["references"]["code01"] == target[cfg.state_len :]
    assert len(meta["references"]["code01"]) == len(target) - cfg.state_len


def test_training_provenance_travels_with_the_export(crf_run):
    run, _, _ = crf_run
    export_crf_onnx(run / "model.pt", run / "export", sidecar=run, verify=False)
    meta = json.loads((run / "export" / "metadata.json").read_text())
    assert meta["training"]["selected_epoch"] == 7


def test_sidecar_loads_from_a_directory_or_a_file(crf_run):
    run, _, _ = crf_run
    assert load_training_sidecar(run) == load_training_sidecar(run / "model.json")


def test_missing_sidecar_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_training_sidecar(tmp_path)


# ── the classifier arms ────────────────────────────────────────────────────


@pytest.fixture
def model_dir(tmp_path, model_config):
    directory = tmp_path / "model"
    directory.mkdir()
    config = {"model_name": "ConvLSTMDwell", **model_config}
    (directory / "config.json").write_text(json.dumps(config))
    model = get_model("ConvLSTMDwell", **model_config)
    torch.save({"model_state_dict": model.state_dict()}, directory / "model_best.pt")
    return directory


def test_classifier_export_agrees_with_torch(tmp_path, model_dir):
    """The measurement #217 opened on: production arms land at 4.77e-07 and
    1.19e-06 against a float32 eps of 1.19e-07."""
    out = tmp_path / "model.onnx"
    export_single_model_onnx(model_dir, out)
    meta = json.loads(out.with_suffix(".json").read_text())
    diff = meta["verification"]["onnxruntime_vs_torch_max_abs_diff"]
    assert diff < 1e-4, f"max abs diff {diff:.3e}"
    assert meta["verification"]["float32_eps"] == pytest.approx(EPS)


def test_classifier_contract_names_every_input(tmp_path, model_dir):
    out = tmp_path / "model.onnx"
    export_single_model_onnx(model_dir, out, verify=False)
    meta = json.loads(out.with_suffix(".json").read_text())
    names = [i["name"] for i in meta["inputs"]]
    assert names == ["signal", "sequence", "features"]
    assert all(i["shape"][0] == "batch" for i in meta["inputs"])


def test_classifier_contract_states_the_output_is_one_bce_logit(tmp_path, model_dir):
    """Read as a two-class softmax, every call is wrong and nothing errors."""
    out = tmp_path / "model.onnx"
    export_single_model_onnx(model_dir, out, verify=False)
    meta = json.loads(out.with_suffix(".json").read_text())
    convention = meta["outputs"][0]["convention"]
    assert "SINGLE BCE logit" in convention and "sigmoid" in convention


def test_classifier_export_records_the_opset(tmp_path, model_dir):
    out = tmp_path / "model.onnx"
    export_single_model_onnx(model_dir, out, verify=False)
    assert json.loads(out.with_suffix(".json").read_text())["opset"] == OPSET


def test_signal_kmer_contract_names_the_encoder_the_graph_does_not_contain():
    """The 36-channel sequence input is built in the dataset, so a non-Python
    runtime must produce it — and reimplementing it creates a second definition
    that diverges silently."""
    config = {"seq_encoding": "signal_kmer", "signal_kmer_context": [4, 4]}
    example = (torch.randn(2, 2, 390), torch.randn(2, 36, 390), torch.randn(2, 12, 21))
    specs = describe_inputs(config, example)
    sequence = next(s for s in specs if s.name == "sequence")
    assert "encode_signal_kmer" in sequence.produced_by
    assert "leech-core" in sequence.produced_by
    assert specs[0].produced_by is None  # raw signal needs no explanation


def test_base_onehot_contract_does_not_mention_the_signal_kmer_encoder():
    config = {"seq_encoding": "base_onehot"}
    example = (torch.randn(2, 400), torch.randn(2, 4, 21))
    specs = describe_inputs(config, example)
    assert "one-hot" in specs[1].produced_by
    assert "encode_signal_kmer" not in specs[1].produced_by


def test_a_config_that_misstates_its_encoding_cannot_be_exported(tmp_path, model_config):
    """The contract is derived from config.json, so the config has to be true.

    #230: a run whose signal_kmer request fell back to base_onehot used to save
    ``seq_encoding: signal_kmer`` anyway. Such a checkpoint is refused here —
    the example inputs and the model are built from the same claim, and the
    weights do not fit it — rather than published as a contract declaring a
    36-channel input the model does not have. Training now records the
    effective encoding, so this state is unreachable from `leech model train`;
    the check is what keeps a hand-edited or older config from being trusted.
    """
    directory = tmp_path / "model"
    directory.mkdir()
    model = get_model("ConvLSTMDwell", **model_config)  # a base_onehot arm
    torch.save({"model_state_dict": model.state_dict()}, directory / "model_best.pt")
    misstated = {
        "model_name": "ConvLSTMDwell",
        **model_config,
        "seq_encoding": "signal_kmer",
        "signal_kmer_context": [4, 4],
    }
    (directory / "config.json").write_text(json.dumps(misstated))

    with pytest.raises(RuntimeError, match="size mismatch"):
        export_single_model_onnx(directory, tmp_path / "model.onnx", verify=False)


# ── the shared exporter ────────────────────────────────────────────────────


def test_export_uses_the_dynamo_path(tmp_path):
    """`dynamo=False` fails on these architectures with an `adaptive_avg_pool1d`
    error that reads like a model problem and is not one. If this ever regresses
    to the legacy exporter, this test is what says so."""

    class Pooled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(7)  # 7 does not divide 100

        def forward(self, x):
            return self.pool(x).flatten(1)

    model = Pooled().eval()
    example = (torch.randn(2, 3, 100),)
    path = export_onnx(model, example, tmp_path / "m.onnx", input_names=["x"], output_names=["y"])
    assert path.exists()
    assert verify_onnx(path, model, example, input_names=["x"]) < 1e-5


def test_the_legacy_exporter_still_refuses_the_aten_adaptive_pool(tmp_path):
    """The claim the module docstring rests on, measured rather than inherited.

    It matters that this is checked and not assumed: leech's own
    `AdaptiveAvgPool1d` no longer emits an aten adaptive pool, which could
    plausibly have retired the reason for pinning the dynamo exporter. It did
    not — `torch.jit.trace` makes `.shape[-1]` a Tensor, so the replacement
    takes its dynamic-length fallback and lands on the same aten op.
    """
    from leech.models.components import AdaptiveAvgPool1d

    example = (torch.randn(2, 3, 100),)
    for name, pool in (("aten", torch.nn.AdaptiveAvgPool1d(7)), ("leech", AdaptiveAvgPool1d(7))):
        model = torch.nn.Sequential(pool, torch.nn.Flatten(1)).eval()
        with pytest.raises(Exception, match="adaptive_avg_pool1d"):
            torch.onnx.export(
                model,
                example,
                str(tmp_path / f"{name}.onnx"),
                dynamo=False,
                opset_version=OPSET,
                input_names=["x"],
                output_names=["y"],
            )


# ── what a non-Python runtime needs, and onnxruntime never notices ─────────


def test_the_pool_exports_as_a_matmul_not_a_gather(tmp_path):
    """A non-dividing adaptive pool must not reach the graph as `GatherND`.

    390 -> 11 is the charging TCN's geometry. Dynamo open-codes the aten op as
    `Unsqueeze -> Transpose -> GatherND -> Transpose -> Where`, a rank-8 gather
    over an all-constant index that tract 0.23.5 cannot close — which is what
    made `charging_tcn_rna004@v0.1.0` unloadable by every released `escpod`
    (rnabioco/escapepod-models#96). onnxruntime runs it fine, so no round-trip
    check can see this; only the op list can.
    """
    import onnx

    from leech.models.components import AdaptiveAvgPool1d

    model = torch.nn.Sequential(AdaptiveAvgPool1d(11), torch.nn.Flatten(1)).eval()
    example = (torch.randn(2, 3, 390),)
    path = export_onnx(
        model, example, tmp_path / "pool.onnx", input_names=["x"], output_names=["y"]
    )
    ops = {n.op_type for n in onnx.load(str(path)).graph.node}
    assert "GatherND" not in ops
    assert "MatMul" in ops
    assert verify_onnx(path, model, example, input_names=["x"]) < 1e-5


def test_the_pool_matches_the_aten_op_it_replaces():
    """Same arithmetic, to float32 rounding, across the bin-width edge cases.

    `adaptive_avg_pool1d`'s bins are `[floor(j*L/K), ceil((j+1)*L/K))`, so
    widths differ by one whenever K does not divide L — 390 -> 11 gives bins of
    36 and 37. A matrix built on a different convention would agree on the
    dividing cases and quietly disagree on exactly the ones that matter.

    The grid includes `k > length`, which is not a degenerate case anyone should
    skip: `ResNetDwell` pools a length-4 map up to 11, so the op UPSAMPLES
    there. A range guard that rejected it shipped in the first draft of this
    and was caught by two model tests rather than by this one; the grid now
    covers it.
    """
    from leech.models.components import AdaptiveAvgPool1d

    torch.manual_seed(0)
    worst = 0.0
    for length in (3, 4, 11, 12, 37, 100, 390, 400, 1024):
        for k in (1, 5, 7, 11, 21):
            x = torch.randn(3, 8, length)
            want = torch.nn.functional.adaptive_avg_pool1d(x, k)
            got = AdaptiveAvgPool1d(k)(x)
            assert got.shape == want.shape
            worst = max(worst, float((want - got).abs().max()))
    assert worst < 10 * EPS, worst


def test_export_writes_no_value_info(tmp_path):
    """Dynamo records every intermediate's shape with the batch axis as a
    SYMBOL; a consumer that pins the batch then cannot unify, and tract fails
    at the first convolution with `Sym(batch) vs Val(1)`. Every graph escpod
    loads carries zero entries."""
    import onnx

    model = torch.nn.Sequential(torch.nn.Conv1d(3, 4, 3), torch.nn.Flatten(1)).eval()
    path = export_onnx(
        model,
        (torch.randn(2, 3, 32),),
        tmp_path / "conv.onnx",
        input_names=["x"],
        output_names=["y"],
    )
    assert list(onnx.load(str(path)).graph.value_info) == []


def test_strip_value_info_is_idempotent_and_reports(tmp_path):
    """It returns how many it dropped, so a caller can log a real number, and
    running it twice is not an error."""
    import onnx

    from leech.onnx_export import strip_value_info

    model = torch.nn.Sequential(torch.nn.Conv1d(3, 4, 3), torch.nn.Flatten(1)).eval()
    path = export_onnx(
        model,
        (torch.randn(2, 3, 32),),
        tmp_path / "again.onnx",
        input_names=["x"],
        output_names=["y"],
    )
    # export_onnx already stripped, so there is nothing left to drop.
    assert strip_value_info(path) == 0
    proto = onnx.load(str(path))
    proto.graph.value_info.extend(proto.graph.input)
    onnx.save(proto, str(path))
    assert strip_value_info(path) == len(proto.graph.input)
    assert list(onnx.load(str(path)).graph.value_info) == []


def test_verify_returns_the_actual_difference(tmp_path):
    model = torch.nn.Linear(4, 2).eval()
    example = (torch.randn(3, 4),)
    path = export_onnx(model, example, tmp_path / "lin.onnx", input_names=["x"], output_names=["y"])
    assert verify_onnx(path, model, example, input_names=["x"]) < 1e-5
