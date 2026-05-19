"""Tests for generic confound-map helpers."""

import json
from pathlib import Path

import pytest

from leech.confounds import (
    build_label_confound_map,
    build_string_id_map,
    load_label_class_map,
)


# ---------------------------------------------------------------------------
# build_label_confound_map
# ---------------------------------------------------------------------------


def test_build_label_confound_map_basic() -> None:
    label_map = {"Ala": 0, "Cys": 1, "Gly": 2}
    class_map = {"Ala": 0, "Cys": 3, "Gly": 1}
    out = build_label_confound_map(label_map, class_map)
    assert out == {0: 0, 1: 3, 2: 1}


def test_build_label_confound_map_drops_missing() -> None:
    """Labels present in label_map but not class_map are omitted."""
    label_map = {"Ala": 0, "uncharged": 1, "Cys": 2}
    class_map = {"Ala": 7, "Cys": 9}
    out = build_label_confound_map(label_map, class_map)
    assert out == {0: 7, 2: 9}


def test_build_label_confound_map_requires_class_map() -> None:
    with pytest.raises(ValueError, match="class_map is required"):
        build_label_confound_map({"Ala": 0}, None)


def test_build_label_confound_map_coerces_to_int() -> None:
    """Class values are coerced to int (JSON-loaded strings should still work)."""
    out = build_label_confound_map({"x": 0}, {"x": "5"})  # type: ignore[arg-type]
    assert out == {0: 5}


# ---------------------------------------------------------------------------
# build_string_id_map
# ---------------------------------------------------------------------------


def test_build_string_id_map_assigns_contiguous_classes() -> None:
    names = ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]
    name_to_int, n = build_string_id_map(names)
    assert n == 3
    assert set(name_to_int.keys()) == {"alpha", "beta", "gamma"}
    assert sorted(name_to_int.values()) == [0, 1, 2]


def test_build_string_id_map_deterministic_ordering() -> None:
    """Sorted-by-string-value, not by first-seen, so runs are reproducible."""
    a, _ = build_string_id_map(["b", "a", "c"])
    b, _ = build_string_id_map(["c", "b", "a"])
    assert a == b == {"a": 0, "b": 1, "c": 2}


# ---------------------------------------------------------------------------
# load_label_class_map
# ---------------------------------------------------------------------------


def test_load_label_class_map_flat(tmp_path: Path) -> None:
    (tmp_path / "label_class_map.json").write_text(json.dumps({"Ala": 0, "Cys": 3}))
    out = load_label_class_map(tmp_path)
    assert out == {"Ala": 0, "Cys": 3}


def test_load_label_class_map_nested(tmp_path: Path) -> None:
    """Loader also accepts the nested provenance form written by the pipeline."""
    payload = {
        "label_to_class": {"Ala": 0, "Cys": 3},
        "class_to_seq": {"0": "AAAAAAAC", "3": "AAAAAAAT"},
        "num_classes": 2,
    }
    (tmp_path / "label_class_map.json").write_text(json.dumps(payload))
    out = load_label_class_map(tmp_path)
    assert out == {"Ala": 0, "Cys": 3}


def test_load_label_class_map_walks_parent(tmp_path: Path) -> None:
    """k-fold layout: map lives one directory above the fold-specific data."""
    (tmp_path / "label_class_map.json").write_text(json.dumps({"Ala": 0}))
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    assert load_label_class_map(fold_dir) == {"Ala": 0}


def test_load_label_class_map_missing(tmp_path: Path) -> None:
    assert load_label_class_map(tmp_path) is None
