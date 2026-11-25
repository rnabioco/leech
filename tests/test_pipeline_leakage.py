"""The label-divergence guard in the Snakemake pipeline (issue #402, Part A).

CI never runs the pipeline rules, so the `common.smk` helpers are evaluated here
without Snakemake (the same technique as ``test_leakage_canary.py``): their
source is pulled out of the real file and exec'd against the real shipped
config, rather than restated.
"""

import ast
import importlib.util
from pathlib import Path

import pytest
import yaml

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
COMMON = PIPELINE / "workflow" / "rules" / "common.smk"
SCRIPTS = PIPELINE / "workflow" / "scripts"

HELPERS = ("build_divergence_arg", "check_label_divergence_config")


class WorkflowError(Exception):
    """Stand-in for snakemake.exceptions.WorkflowError."""


def _helpers(config: dict):
    tree = ast.parse(COMMON.read_text())
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert {n.name for n in wanted} == set(HELPERS), "common.smk lost a divergence helper"
    namespace = {"config": config, "WorkflowError": WorkflowError}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "common.smk", "exec"), namespace)
    return namespace["build_divergence_arg"], namespace["check_label_divergence_config"]


def _shipped_config(**overrides) -> dict:
    config = yaml.safe_load((PIPELINE / "config" / "config.yaml").read_text())
    config.update(yaml.safe_load((PIPELINE / "config" / "samples-alpine.yaml").read_text()))
    config.update(overrides)
    return config


def _declared(**overrides) -> dict:
    """A config that declares a divergence (motif CCAGGC at offset 2, P=20 -> D=18)."""
    base = {
        "label_divergence_position": 20,
        "motif": "CCAGGC",
        "motif_offset": 2,
        "chunk_context": [225, 225],
        "use_grid_search": False,
    }
    base.update(overrides)
    return _shipped_config(**base)


def test_missing_declaration_is_a_workflow_error():
    config = _shipped_config()
    del config["label_divergence_position"]
    arg, check = _helpers(config)
    for helper in (arg, check):
        with pytest.raises(WorkflowError, match="label_divergence_position"):
            helper()


def test_shipped_configs_set_the_key_where_it_belongs():
    """config.yaml documents the key without setting it; the samples file sets it."""
    base = yaml.safe_load((PIPELINE / "config" / "config.yaml").read_text())
    samples = yaml.safe_load((PIPELINE / "config" / "samples-alpine.yaml").read_text())
    assert "label_divergence_position" not in base
    assert "label_divergence_position" in samples
    assert base["allow_window_past_divergence"] is False
    assert "label_divergence_position" in (PIPELINE / "config" / "config.yaml").read_text()


def test_null_declaration_is_accepted():
    arg, check = _helpers(_shipped_config(label_divergence_position=None))
    assert arg() == ""
    assert check() == []


def test_declaration_renders_the_flags():
    arg, check = _helpers(_declared())
    assert arg() == "--label-divergence-position 20"
    assert [b["status"] for b in check()] == ["clear"]
    arg, _ = _helpers(_declared(allow_window_past_divergence=True))
    assert arg() == "--label-divergence-position 20 --allow-window-past-divergence"


def test_chunk_context_past_the_divergence_is_refused_at_dag_build():
    _, check = _helpers(_declared(chunk_context=[225, 600]))
    with pytest.raises(WorkflowError, match="chunk_context"):
        check()


def test_grid_search_right_context_past_the_divergence_is_refused_before_the_search():
    config = _declared(
        use_grid_search=True,
        grid_search={"context_grid": "200:1000:200", "dwell_offsets": "0"},
    )
    _, check = _helpers(config)
    with pytest.raises(WorkflowError, match="grid_search right context"):
        check()

    # an explicit right-context list is what is checked, not the symmetric fallback
    ok = _declared(
        use_grid_search=True,
        grid_search={"context_grid": "200:1000:200", "right_contexts": [100, 200]},
    )
    assert _helpers(ok)[1]()

    # off by default: the grid does not apply when it is not used
    assert _helpers(_declared(grid_search={"context_grid": "200:1000:200"}))[1]()


def test_override_key_lets_the_run_proceed():
    _, check = _helpers(_declared(chunk_context=[225, 600], allow_window_past_divergence=True))
    blocks = check()
    assert blocks and all(b["status"] == "override" for b in blocks)


def _load_script(name: str):
    pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["summarize_metrics", "compare_architectures"])
def test_leakage_status_is_an_explicit_column_never_blank(script):
    module = _load_script(script)
    absent = module.flatten_leakage({"accuracy": 0.9})
    assert absent["leakage_status"] == "undeclared"
    assert "leakage" not in absent
    for status in ("clear", "override"):
        flat = module.flatten_leakage(
            {"leakage": {"status": status, "position": 20, "focus_offset": 18}}
        )
        assert flat["leakage_status"] == status
        assert flat["leakage_focus_offset"] == 18


def test_comparison_table_surfaces_the_worst_status_per_architecture():
    module = _load_script("compare_architectures")
    import pandas as pd

    df = pd.DataFrame(
        [
            {"architecture": "A", "sample": "s1", "f1": 0.8, **_flat(module, "clear")},
            {"architecture": "A", "sample": "s2", "f1": 0.9, **_flat(module, "override")},
            {"architecture": "B", "sample": "s1", "f1": 0.7, **_flat(module, None)},
            {"architecture": "C", "sample": "s1", "f1": 0.6, **_flat(module, "clear")},
        ]
    )
    agg = module.aggregate_metrics(df)
    status = dict(zip(agg["architecture"], agg["leakage_status"], strict=True))
    assert status == {"A": "override", "B": "undeclared", "C": "clear"}
    table = module.create_comparison_table(agg)
    assert {"leakage_status", "leakage_focus_offset"} <= set(table.columns)
    assert table["leakage_status"].notna().all()


def _flat(module, status):
    leakage = None if status is None else {"status": status, "focus_offset": 18}
    data = {"leakage": leakage} if leakage else {}
    flat = module.flatten_leakage(data)
    return {k: flat[k] for k in ("leakage_status", "leakage_focus_offset")}
