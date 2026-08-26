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


def test_verify_returns_the_actual_difference(tmp_path):
    model = torch.nn.Linear(4, 2).eval()
    example = (torch.randn(3, 4),)
    path = export_onnx(model, example, tmp_path / "lin.onnx", input_names=["x"], output_names=["y"])
    assert verify_onnx(path, model, example, input_names=["x"]) < 1e-5
