"""Label-divergence guard (issue #402, Part A).

Where two classes differ by library chemistry (adapters sharing a common arm,
then diverging), the sequence past the divergence is the label. The user
declares where it starts, ``P`` (0-based from the first base of ``--motif``);
leech derives ``D = P - motif_offset`` and refuses a window in which any stored
input can carry the identity of a base at focus-relative ``+D`` or beyond.

``P``/``D`` values below are fixtures, not claims about any library: 20 is
verified against the shipped dual-adapter panel (first test); 28 is used only
as an arbitrary larger fixture.
"""

import json
import sys
from pathlib import Path

import pytest
from test_chunk_writer import make_chunks

from leech import leakage
from leech.constants import DEFAULT_MIN_SAMPLES_PER_BASE

REPO = Path(__file__).resolve().parent.parent
PANEL = REPO / "pipeline" / "resources" / "references" / "yeast-biological-trna.fa"

P = 20  # first label-defining base, from the start of the motif
MOTIF = "CCAGGC"
D = 18  # P - motif_offset 2
LOOKAHEAD = 4  # the bundled 9-mer table

KMER_CHANNELS = ["kmer_expected", "kmer_residual", "kmer_residual_abs"]
NO_KMER = ["dwell", "dwell_log", "level_mean"]


def _window(**kw):
    """The window of the production-arm example, with the pinned-table defaults."""
    base = {
        "feature_start": 0,
        "feature_end": 12,
        "kmer_context": 5,
        "base_justify": "end",
        "signal_context": (90, 300),
        "signal_context_bases": None,
        "lookahead": LOOKAHEAD,
        "feature_channels": None,
        "mask_seq_side": None,
    }
    base.update(kw)
    return base


def _evaluate(window, *, floor=DEFAULT_MIN_SAMPLES_PER_BASE, override=False, offset=2, pos=P):
    return leakage.evaluate(
        pos,
        motif=MOTIF,
        motif_offset=offset,
        allow_override=override,
        rate_floor=floor,
        **window,
    )


def _offenders(window, *, floor=DEFAULT_MIN_SAMPLES_PER_BASE, d=D):
    reach = leakage.compute_reach(rate_floor=floor, **window)
    return {k: v for k, v in reach.items() if v >= d}


def _records(path):
    out, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            out[name] = ""
        elif name:
            out[name] += line.strip()
    return out


# ---------------------------------------------------------------------------
# The numbers this design rests on
# ---------------------------------------------------------------------------


def test_shipped_dual_adapter_panel_first_disagrees_at_position_20():
    """Pins P=20: the two adapters in the shipped panel diverge 20 bases into CCAGGC."""
    slices = {seq[seq.index(MOTIF) :][:43] for seq in _records(PANEL).values()}
    assert len(slices) == 2, "the panel holds exactly two adapter chemistries"
    one, two = sorted(slices)
    first = next(i for i, (x, y) in enumerate(zip(one, two, strict=True)) if x != y)
    assert first == P


def test_focus_relative_divergence_follows_motif_offset():
    assert leakage.focus_offset(P, 2) == 18
    assert leakage.focus_offset(P, 3) == 17
    assert _evaluate(_window(), offset=3, override=True)["focus_offset"] == 17
    assert _evaluate(_window(), offset=2)["focus_offset"] == 18


# ---------------------------------------------------------------------------
# The reach table
# ---------------------------------------------------------------------------

#: (id, window overrides, rate floor, expected refused inputs {name: reach} or None)
PINNED = [
    ("fe20-kmer", {"feature_end": 20}, 24, {"features": 20, "kmer_features": 24}),
    ("fe17-kmer", {"feature_end": 17}, 24, {"kmer_features": 21}),
    ("fe13-kmer", {"feature_end": 13}, 24, None),
    ("fe14-kmer", {"feature_end": 14}, 24, {"kmer_features": 18}),
    ("fe17-nokmer", {"feature_end": 17, "feature_channels": NO_KMER}, 24, None),
    ("fe18-nokmer", {"feature_end": 18, "feature_channels": NO_KMER}, 24, {"features": 18}),
    ("fe12-floor24", {"feature_end": 12}, 24, None),
    (
        "fe12-floor23",
        {"feature_end": 12},
        23,
        {"sequence_kmer": 18, "signal_residual": 18},
    ),
    (
        "fe12-floor15",
        {"feature_end": 12},
        15,
        {"signal": 20, "sequence_kmer": 24, "signal_residual": 24},
    ),
    ("bases-0-13", {"feature_end": 12, "signal_context_bases": (0, 13)}, 24, None),
    (
        "bases-0-14",
        {"feature_end": 12, "signal_context_bases": (0, 14)},
        24,
        {"sequence_kmer": 18, "signal_residual": 18},
    ),
    (
        "bases-0-17-blind",
        {
            "feature_end": 12,
            "signal_context_bases": (0, 17),
            "lookahead": None,
            "mask_seq_side": "right",
        },
        24,
        None,
    ),
    (
        "bases-0-18-blind",
        {
            "feature_end": 12,
            "signal_context_bases": (0, 18),
            "lookahead": None,
            "mask_seq_side": "right",
        },
        24,
        {"signal": 18},
    ),
    (
        "shipped-pipeline-default",
        {
            "feature_start": None,
            "feature_end": None,
            "base_justify": "center",
            "signal_context": (225, 225),
        },
        24,
        None,
    ),
]


@pytest.mark.parametrize(
    ("overrides", "floor", "expected"),
    [pytest.param(o, f, e, id=i) for i, o, f, e in PINNED],
)
def test_reach_table_pinned_examples(overrides, floor, expected):
    window = _window(**overrides)
    assert _offenders(window, floor=floor) == (expected or {})
    if expected is None:
        block = _evaluate(window, floor=floor)
        assert block["status"] == "clear"
    else:
        with pytest.raises(ValueError, match="label-divergence"):
            _evaluate(window, floor=floor)
        assert _evaluate(window, floor=floor, override=True)["status"] == "override"


def test_the_boundary_is_exclusive():
    """Reach D-1 passes; reach D is refused."""
    assert _offenders(_window(feature_end=12, feature_channels=NO_KMER), d=18) == {}
    r = leakage.compute_reach(**_window(feature_end=17, feature_channels=NO_KMER))
    assert r["features"] == 17 < D
    r = leakage.compute_reach(**_window(feature_end=18, feature_channels=NO_KMER))
    assert r["features"] == 18 >= D


def test_kmer_lookahead_is_read_from_the_refiner_not_hardcoded(tmp_path):
    # The bundled 9-mer table: L = 4.
    from leech.data import get_kmer_table

    assert leakage.table_kmer_len(get_kmer_table()) == 9
    assert leakage.bundled_lookahead() == LOOKAHEAD == leakage.refiner_lookahead(9)

    # A 7-mer table moves the verdict at the same window: 14 + 3 = 17 < 18.
    table = tmp_path / "k7.txt"
    table.write_text("AAAAAAA\t0.5\nCCCCCCC\t1.5\n")
    seven = leakage.table_kmer_len(table)
    assert seven == 7
    l7 = leakage.refiner_lookahead(seven)
    assert l7 == 3
    wide = _window(feature_end=14)
    assert _offenders({**wide, "lookahead": LOOKAHEAD}) == {"kmer_features": 18}
    assert _offenders({**wide, "lookahead": l7}) == {}


def test_no_refiner_means_no_sequence_derived_inputs():
    """Without a refiner the k-mer feature channels and the residual are not stored."""
    reach = leakage.compute_reach(**_window(feature_end=30, lookahead=None))
    assert "kmer_features" not in reach
    assert "signal_residual" not in reach
    assert reach["features"] == 30  # the non-k-mer rows still are


def test_mask_seq_side_right_discounts_only_the_sequence_branches():
    plain = leakage.compute_reach(**_window(feature_end=12))
    masked = leakage.compute_reach(**_window(feature_end=12, mask_seq_side="right"))
    assert masked["sequence"] == masked["sequence_kmer"] == 0
    for key in ("features", "kmer_features", "signal", "signal_residual"):
        assert masked[key] == plain[key]
    # "left" masks the 5' side: the downstream reach is unchanged.
    left = leakage.compute_reach(**_window(feature_end=12, mask_seq_side="left"))
    assert left == plain


def test_signal_context_bases_reach_is_exact():
    for r in (0, 5, 17, 40):
        reach = leakage.compute_reach(**_window(signal_context_bases=(3, r)))
        assert reach["signal"] == r
        assert reach["signal_residual"] == r + LOOKAHEAD
        assert reach["sequence_kmer"] == r + 4
    # and it ignores the rate floor entirely
    a = leakage.compute_reach(rate_floor=10, **_window(signal_context_bases=(0, 9)))
    b = leakage.compute_reach(rate_floor=40, **_window(signal_context_bases=(0, 9)))
    assert a == b


def test_sample_window_bound_uses_the_rate_floor_and_justify():
    def sig(justify, right, floor):
        return leakage.signal_reach(
            base_justify=justify, signal_context=(0, right), rate_floor=floor
        )

    # ceil(theta + R / s) - 1
    assert sig("end", 300, 24) == 13  # ceil(1 + 12.5) - 1
    assert sig("center", 300, 24) == 12  # ceil(0.5 + 12.5) - 1
    assert sig("start", 300, 24) == 12  # ceil(0 + 12.5) - 1
    assert sig("end", 300, 23) == 14
    assert sig("end", 300, 15) == 20
    assert sig("end", 240, 24) == 10  # exact multiple: ceil(11) - 1
    # a slower floor means a smaller bound
    assert sig("end", 300, 36) < sig("end", 300, 24)
    assert DEFAULT_MIN_SAMPLES_PER_BASE == 24.0


# ---------------------------------------------------------------------------
# Declaration contract and the refusal message
# ---------------------------------------------------------------------------


def test_declaration_requires_a_motif_and_refuses_focus_tsv(tmp_path):
    with pytest.raises(ValueError, match="--motif"):
        leakage.validate_declaration(P, motif=None, motif_offset=2)
    with pytest.raises(ValueError, match="--focus-tsv"):
        leakage.validate_declaration(P, motif=MOTIF, motif_offset=2, focus_tsv=tmp_path / "x.tsv")

    from leech.commands.prepare import handle_prepare

    base = {
        "pod5": tmp_path / "x.pod5",
        "bam": tmp_path / "x.bam",
        "output_dir": tmp_path / "out",
        "label_divergence_position": P,
    }
    with pytest.raises(ValueError, match="--motif"):
        handle_prepare(**base)
    tsv = tmp_path / "focus.tsv"
    tsv.write_text("read_id\tlabel_int\tanchor_sample\n")
    with pytest.raises(ValueError, match="--focus-tsv"):
        handle_prepare(**base, motif=MOTIF, focus_tsv=tsv)


def test_declaration_cli_refuses_missing_motif_and_focus_tsv(tmp_path):
    from click.testing import CliRunner

    from leech.cli import cli

    (tmp_path / "x.pod5").touch()
    (tmp_path / "x.bam").touch()
    args = [
        "data",
        "prepare",
        "--pod5",
        str(tmp_path / "x.pod5"),
        "--bam",
        str(tmp_path / "x.bam"),
        "--output-dir",
        str(tmp_path / "o"),
        "--label-divergence-position",
        "20",
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert "--motif" in result.output


@pytest.mark.parametrize(("pos", "offset"), [(2, 2), (1, 2), (0, 0), (5, 9)])
def test_nonpositive_focus_relative_divergence_is_refused(pos, offset):
    with pytest.raises(ValueError, match=r"must be >= \+1"):
        leakage.validate_declaration(pos, motif=MOTIF, motif_offset=offset)


def test_refusal_message_names_the_input_the_reach_and_the_override():
    with pytest.raises(ValueError) as err:
        _evaluate(_window(feature_end=20), floor=23)
    msg = str(err.value)
    assert f"position {P}" in msg
    assert MOTIF in msg
    assert "motif offset 2" in msg
    assert f"+{D}" in msg
    assert "features: reach +20" in msg
    assert "kmer_features: reach +24" in msg
    assert "23 samples/base" in msg  # the rate floor, since the signal window is sample-defined
    assert "--signal-context-bases" in msg
    assert "--allow-window-past-divergence" in msg
    assert "shorten" in msg


def test_refusal_omits_the_rate_floor_for_a_base_defined_window():
    with pytest.raises(ValueError) as err:
        _evaluate(_window(signal_context_bases=(0, 25)))
    assert "samples/base" not in str(err.value)


def test_override_is_recorded_as_status_override():
    block = _evaluate(_window(feature_end=20), override=True)
    assert block["status"] == "override"
    assert block["reach"]["features"] == 20
    assert _evaluate(_window(feature_end=12))["status"] == "clear"
    assert set(_evaluate(_window())) == {
        "position",
        "focus_offset",
        "motif",
        "motif_offset",
        "rate_floor",
        "reach",
        "status",
    }


# ---------------------------------------------------------------------------
# prepare: before any read, and what the sidecar records
# ---------------------------------------------------------------------------


def _fake_parallel(calls):
    def fake(*, chunk_sink, **_):
        calls.append("extract")
        chunks = make_chunks(6)
        chunk_sink(chunks)
        return [], {
            "total_reads": 6,
            "reads_with_motif": 6,
            "reads_without_motif": 0,
            "total_chunks": 6,
            "failed_reads": 0,
            "failed_batches": 0,
        }

    return fake


def _prepare(tmp_path, monkeypatch, **kwargs):
    from leech.commands.prepare import handle_prepare

    calls: list[str] = []
    monkeypatch.setattr("leech.preparation.prepare_training_data_parallel", _fake_parallel(calls))
    out = kwargs.pop("output_dir", tmp_path / "out")
    handle_prepare(
        pod5=tmp_path / "nonexistent.pod5",
        bam=tmp_path / "nonexistent.bam",
        output_dir=out,
        motif=MOTIF,
        motif_offset=2,
        motif_reference="bam",
        refine_signal_map=False,
        workers=1,
        no_split=True,
        **kwargs,
    )
    return out, calls


def test_prepare_refuses_before_touching_reads(tmp_path, monkeypatch):
    """The refusal precedes reference loading, the sidecar, and extraction."""
    opened = []
    monkeypatch.setattr(
        "leech.io.get_reference_sequences", lambda *a, **k: opened.append("reference")
    )
    from leech.commands.prepare import handle_prepare

    calls: list[str] = []
    monkeypatch.setattr("leech.preparation.prepare_training_data_parallel", _fake_parallel(calls))
    with pytest.raises(ValueError, match="label-divergence"):
        handle_prepare(
            pod5=tmp_path / "does-not-exist.pod5",
            bam=tmp_path / "does-not-exist.bam",
            output_dir=tmp_path / "out",
            motif=MOTIF,
            motif_offset=2,
            motif_reference="fasta",  # would load a reference, if reached
            refine_signal_map=False,
            feature_start=0,
            feature_end=20,
            label_divergence_position=P,
        )
    assert opened == [] and calls == []
    assert not (tmp_path / "out" / "prepare_config.json").exists()


def test_prepare_override_proceeds_and_records_it(tmp_path, monkeypatch):
    out, calls = _prepare(
        tmp_path,
        monkeypatch,
        feature_start=0,
        feature_end=20,
        label_divergence_position=P,
        allow_window_past_divergence=True,
    )
    assert calls == ["extract"]
    block = json.loads((out / "prepare_config.json").read_text())["label_divergence"]
    assert block["status"] == "override"
    assert block["reach"]["features"] == 20


def test_prepare_config_records_the_declaration_and_null_when_undeclared(tmp_path, monkeypatch):
    out, _ = _prepare(
        tmp_path,
        monkeypatch,
        output_dir=tmp_path / "declared",
        feature_start=0,
        feature_end=12,
        base_justify="center",
        signal_context=(225, 225),
        label_divergence_position=P,
    )
    block = json.loads((out / "prepare_config.json").read_text())["label_divergence"]
    assert block["position"] == P
    assert block["focus_offset"] == D
    assert block["motif"] == MOTIF
    assert block["motif_offset"] == 2
    assert block["rate_floor"] == 24.0
    assert block["status"] == "clear"
    # refinement is off here: no k-mer features, no residual
    assert set(block["reach"]) == {"features", "signal", "sequence", "sequence_kmer"}

    out2, _ = _prepare(tmp_path, monkeypatch, output_dir=tmp_path / "undeclared")
    sidecar = json.loads((out2 / "prepare_config.json").read_text())
    assert "label_divergence" in sidecar and sidecar["label_divergence"] is None


def test_override_without_a_declaration_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no effect without"):
        _prepare(tmp_path, monkeypatch, allow_window_past_divergence=True)


def test_leakage_module_is_torch_free():
    import subprocess

    code = (
        "import sys; import leech.leakage as m; "
        "m.compute_reach(); m.bundled_lookahead(); "
        "import leech.context_grid; "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _sidecar(directory, block):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"feature_channels": ["dwell"]}
    if block is not False:
        payload["label_divergence"] = block
    (directory / "prepare_config.json").write_text(json.dumps(payload))
    return directory / "all.npz"


def _block(position=P, status="clear", offset=2, reach=None):
    return {
        "position": position,
        "focus_offset": position - offset,
        "motif": MOTIF,
        "motif_offset": offset,
        "rate_floor": 24.0,
        "reach": reach or {"features": 12},
        "status": status,
    }


def test_merge_refuses_disagreeing_declarations(tmp_path):
    from leech.commands.merge_split import _propagate_prepare_config

    a = _sidecar(tmp_path / "a", _block(20))
    b = _sidecar(tmp_path / "b", _block(28))
    with pytest.raises(ValueError, match="label-divergence"):
        _propagate_prepare_config([a, b], tmp_path / "out")


def test_merge_refuses_declared_with_undeclared(tmp_path):
    from leech.commands.merge_split import _propagate_prepare_config

    a = _sidecar(tmp_path / "a", _block())
    for undeclared in (None, False):  # null block, and a sidecar predating the key
        b = _sidecar(tmp_path / "b", undeclared)
        with pytest.raises(ValueError, match=r"re-prepare"):
            _propagate_prepare_config([a, b], tmp_path / "out")
    # an input with no sidecar at all is undeclared too
    (tmp_path / "c").mkdir()
    with pytest.raises(ValueError, match="undeclared"):
        _propagate_prepare_config([a, tmp_path / "c" / "all.npz"], tmp_path / "out")


def test_merge_of_all_undeclared_inputs_is_unchanged(tmp_path, caplog):
    from leech.commands.merge_split import _propagate_prepare_config

    a = _sidecar(tmp_path / "a", None)
    b = _sidecar(tmp_path / "b", False)
    _propagate_prepare_config([a, b], tmp_path / "out")
    written = json.loads((tmp_path / "out" / "prepare_config.json").read_text())
    assert written.get("label_divergence") is None
    assert "label-divergence" not in caplog.text


def test_merge_of_agreeing_declarations_keeps_the_worst_status(tmp_path):
    from leech.commands.merge_split import _propagate_prepare_config

    a = _sidecar(tmp_path / "a", _block(status="clear", reach={"features": 12, "signal": 9}))
    b = _sidecar(tmp_path / "b", _block(status="override", reach={"features": 20}))
    _propagate_prepare_config([a, b], tmp_path / "out")
    block = json.loads((tmp_path / "out" / "prepare_config.json").read_text())["label_divergence"]
    assert block["status"] == "override"
    assert block["reach"] == {"features": 20, "signal": 9}


def test_kfold_merge_applies_the_same_check(tmp_path):
    from leech.chunking import save_chunks
    from leech.commands.merge_split import handle_merge_and_split_kfold

    paths = []
    for name, block in (("A", _block(20)), ("B", _block(28))):
        path = _sidecar(tmp_path / name, block)
        save_chunks(make_chunks(12), path)
        paths.append(f"{name}={path}")
    with pytest.raises(ValueError, match="label-divergence"):
        handle_merge_and_split_kfold(tuple(paths), tmp_path / "kfold", k_fold=3)


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------


def _train(temp_chunks_file, model_dir, block):
    from leech.training import train_model

    sidecar = temp_chunks_file.parent / "prepare_config.json"
    sidecar.write_text(json.dumps({"label_divergence": block} if block is not False else {}))
    train_model(
        train_data_path=temp_chunks_file,
        val_data_path=None,
        model_name="ConvLSTMDwell",
        output_dir=model_dir,
        epochs=1,
        batch_size=2,
        device="cpu",
        motif=MOTIF,
        seed=42,
    )
    return json.loads((model_dir / "config.json").read_text())


def test_train_config_json_carries_the_declaration(temp_chunks_file, tmp_path):
    block = _block()
    assert _train(temp_chunks_file, tmp_path / "m1", block)["label_divergence"] == block
    assert _train(temp_chunks_file, tmp_path / "m2", False)["label_divergence"] is None


def test_eval_metrics_carry_the_leakage_block(temp_chunks_file, tmp_path):
    from leech.evaluation import evaluate_model

    for name, block, expected in (
        ("declared", _block(status="override"), {"status": "override", "position": P}),
        ("undeclared", False, None),
    ):
        model_dir = tmp_path / name
        _train(temp_chunks_file, model_dir, block)
        metrics = evaluate_model(
            model_path=model_dir,
            test_data_path=temp_chunks_file,
            output_path=tmp_path / f"{name}.json",
            device="cpu",
            batch_size=2,
        )
        saved = json.loads((tmp_path / f"{name}.json").read_text())
        if expected is None:
            assert "leakage" not in metrics and "leakage" not in saved
        else:
            assert metrics["leakage"] == {**expected, "focus_offset": D}
            assert saved["leakage"] == metrics["leakage"]


def test_leakage_is_attached_on_the_multiclass_path_too():
    """Both `evaluate_model` return paths go through the one attach helper."""
    import inspect

    from leech import evaluation

    src = inspect.getsource(evaluation.evaluate_model)
    assert src.count("_attach_leakage(metrics, config)") == 2
    metrics: dict = {}
    evaluation._attach_leakage(metrics, {"label_divergence": _block(status="clear")})
    assert metrics["leakage"] == {"status": "clear", "position": P, "focus_offset": D}
    metrics = {}
    evaluation._attach_leakage(metrics, {"label_divergence": None})
    assert metrics == {}
