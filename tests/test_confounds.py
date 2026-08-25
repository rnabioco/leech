"""Tests for confound mappings, in particular the per-AA disc_base extractor."""

import json
from pathlib import Path

import pytest

from leech.confounds import (
    BUILTIN_CONFOUNDS,
    DISC_BASE_TO_INT,
    NUM_DISC_BASES,
    ConfoundEncoder,
    ConfoundSpec,
    build_confound_encoder,
    build_confound_map,
    extract_disc_bases_from_fasta,
    load_confound_config,
    parse_confound_token,
)


@pytest.fixture
def fasta_path(tmp_path: Path) -> Path:
    """Three tRNAs where one AA (Arg) has two isoacceptors with conflicting disc_bases."""
    path = tmp_path / "test.fa"
    path.write_text(
        ">tRNA-Ala-GGC-1-1\n"
        "AAAAAAAACCCAGGCTTT\n"  # disc_base 'C' (one nt before CCAGGC)
        ">tRNA-Arg-ACG-1-1\n"
        "AAAAAAAAGCCAGGCTTT\n"  # disc_base 'G'
        ">tRNA-Arg-CCG-1-1\n"
        "AAAAAAAAGCCAGGCTTT\n"  # disc_base 'G' (same so Arg mode is unambiguous)
        ">tRNA-Cys-GCA-1-1\n"
        "AAAAAAAATCCAGGCTTT\n"  # disc_base 'T'
    )
    return path


def test_extract_disc_bases_basic(fasta_path: Path) -> None:
    disc_map = extract_disc_bases_from_fasta(fasta_path)
    assert disc_map == {"Ala": "C", "Arg": "G", "Cys": "T"}


def test_extract_disc_bases_modal_vote(tmp_path: Path) -> None:
    """When isoacceptors of one AA disagree, the modal base wins."""
    path = tmp_path / "mixed.fa"
    path.write_text(
        ">tRNA-Arg-ACG-1-1\n"
        "AAAAAAAACCCAGGC\n"  # C
        ">tRNA-Arg-CCG-1-1\n"
        "AAAAAAAATCCAGGC\n"  # T
        ">tRNA-Arg-CCT-1-1\n"
        "AAAAAAAACCCAGGC\n"  # C (modal)
    )
    disc_map = extract_disc_bases_from_fasta(path)
    assert disc_map == {"Arg": "C"}


def test_extract_disc_bases_rna_to_dna(tmp_path: Path) -> None:
    """U is silently mapped to T so RNA-style FASTAs work."""
    path = tmp_path / "rna.fa"
    path.write_text(">tRNA-Phe-AAA-1-1\nAAAAAAAAUCCAGGC\n")
    disc_map = extract_disc_bases_from_fasta(path)
    assert disc_map == {"Phe": "T"}


def test_extract_disc_bases_missing_motif(tmp_path: Path) -> None:
    path = tmp_path / "no_motif.fa"
    path.write_text(">tRNA-Lys-CTT-1-1\nAAAAAAAAGGGG\n")
    with pytest.raises(ValueError, match="not found"):
        extract_disc_bases_from_fasta(path)


def test_extract_disc_bases_motif_at_start(tmp_path: Path) -> None:
    path = tmp_path / "start.fa"
    path.write_text(">tRNA-Lys-CTT-1-1\nCCAGGCAAA\n")
    with pytest.raises(ValueError, match="position 0"):
        extract_disc_bases_from_fasta(path)


def test_extract_disc_bases_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.fa"
    path.write_text(">NotAtRNA\nAAAACCCAGGC\n")
    with pytest.raises(ValueError, match="Cannot parse"):
        extract_disc_bases_from_fasta(path)


def test_extract_disc_bases_feeds_build_confound_map(fasta_path: Path) -> None:
    """End-to-end: extracted disc_map feeds build_confound_map → int classes."""
    disc_map = extract_disc_bases_from_fasta(fasta_path)
    label_map = {"Ala": 0, "Arg": 1, "Cys": 2}
    confound_map = build_confound_map(label_map, disc_map)
    assert confound_map == {
        0: DISC_BASE_TO_INT["C"],
        1: DISC_BASE_TO_INT["G"],
        2: DISC_BASE_TO_INT["T"],
    }


# ---------------------------------------------------------------------------
# Generic, config-driven confound layer
# ---------------------------------------------------------------------------


def test_parse_confound_token_aliases() -> None:
    assert parse_confound_token("trna_id") is BUILTIN_CONFOUNDS["trna_id"]
    assert parse_confound_token("disc_base") is BUILTIN_CONFOUNDS["disc_base"]


def test_parse_confound_token_inline_identity() -> None:
    spec = parse_confound_token("reference_name:identity")
    assert spec.source == "reference_name"
    assert spec.mapping == "identity"
    assert spec.table is None


def test_parse_confound_token_inline_lookup() -> None:
    spec = parse_confound_token("source_group:lookup:groups.json")
    assert spec.source == "source_group"
    assert spec.mapping == "lookup"
    assert spec.table == "groups.json"


def test_parse_confound_token_unknown_alias() -> None:
    with pytest.raises(ValueError, match="Unknown confound"):
        parse_confound_token("nonsense")


def test_parse_confound_token_bad_mapping() -> None:
    with pytest.raises(ValueError, match="Unknown confound mapping"):
        parse_confound_token("reference_name:bogus")


def test_confound_spec_token_round_trip() -> None:
    for token in ("trna_id", "disc_base", "reference_name:identity", "g:lookup:t.json"):
        assert parse_confound_token(token).to_token() == token


def test_load_confound_config_json_list(tmp_path: Path) -> None:
    path = tmp_path / "conf.json"
    path.write_text(json.dumps(["trna_id"]))
    specs = load_confound_config(path)
    assert len(specs) == 1
    assert specs[0].to_token() == "trna_id"


def test_load_confound_config_yaml_dict(tmp_path: Path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text(
        "confounds:\n"
        "  - source: source_group\n"
        "    mapping: lookup\n"
        "    table: groups.json\n"
        "    name: aa_group\n"
    )
    specs = load_confound_config(path)
    assert len(specs) == 1
    assert specs[0].source == "source_group"
    assert specs[0].mapping == "lookup"
    assert specs[0].table == "groups.json"
    assert specs[0].label == "aa_group"


def test_build_encoder_identity() -> None:
    """Identity mapping assigns a contiguous class per distinct value (sorted)."""
    spec = ConfoundSpec(source="reference_name", mapping="identity")
    values = ["tRNA-Ser-CGA-1-1", "tRNA-Ala-GGC-1-1", "tRNA-Ser-CGA-1-1", None, ""]
    enc = build_confound_encoder(spec, source_values=values)
    assert enc is not None
    assert enc.num_classes == 2
    assert enc.value_to_class == {"tRNA-Ala-GGC-1-1": 0, "tRNA-Ser-CGA-1-1": 1}
    # encode reads the configured chunk field; unknown/missing -> -1
    assert enc.encode({"reference_name": "tRNA-Ala-GGC-1-1"}) == 0
    assert enc.encode({"reference_name": "tRNA-Ser-CGA-1-1"}) == 1
    assert enc.encode({"reference_name": "unseen"}) == -1
    assert enc.encode({}) == -1


def test_build_encoder_identity_no_values_returns_none() -> None:
    spec = ConfoundSpec(source="reference_name", mapping="identity")
    assert build_confound_encoder(spec, source_values=[]) is None


def test_build_encoder_lookup(tmp_path: Path) -> None:
    """Lookup maps raw values through a sidecar table, categories -> ints."""
    table = tmp_path / "groups.json"
    table.write_text(json.dumps({"Ala": "small", "Gly": "small", "Trp": "large"}))
    spec = ConfoundSpec(source="source_group", mapping="lookup", table="groups.json")
    enc = build_confound_encoder(spec, data_dir=tmp_path)
    assert enc is not None
    assert enc.num_classes == 2  # small, large
    # categories sorted: large=0, small=1
    assert enc.encode({"source_group": "Trp"}) == 0
    assert enc.encode({"source_group": "Ala"}) == 1
    assert enc.encode({"source_group": "Gly"}) == 1
    assert enc.encode({"source_group": "Unknown"}) == -1


def test_build_encoder_disc_base_matches_legacy(fasta_path: Path, tmp_path: Path) -> None:
    """The disc_base alias reproduces the legacy build_confound_map behaviour."""
    disc_map = extract_disc_bases_from_fasta(fasta_path)
    (tmp_path / "disc_base_map.json").write_text(json.dumps(disc_map))
    label_map = {"Ala": 0, "Arg": 1, "Cys": 2}

    enc = build_confound_encoder(
        BUILTIN_CONFOUNDS["disc_base"],
        label_map=label_map,
        data_dir=tmp_path,
    )
    assert enc is not None
    assert enc.num_classes == NUM_DISC_BASES
    assert enc.source == "label_int"
    assert enc.value_to_class == build_confound_map(label_map, disc_map)
    # encode keys on the chunk's label_int field
    assert enc.encode({"label_int": 0}) == DISC_BASE_TO_INT["C"]
    assert enc.encode({"label_int": 99}) == -1


def test_build_encoder_disc_base_needs_label_map() -> None:
    enc = build_confound_encoder(BUILTIN_CONFOUNDS["disc_base"], label_map=None)
    assert enc is None


def test_confound_encoder_direct() -> None:
    enc = ConfoundEncoder(name="c", source="source_group", value_to_class={"x": 0}, num_classes=1)
    assert enc.encode({"source_group": "x"}) == 0
    assert enc.encode({"source_group": "y"}) == -1


class TestIdentityFromNpzColumn:
    """`build_confound_encoder` takes the npz column itself, not a boxed list.

    `training.py` used to call `.tolist()` on the source column, which boxes one
    Python string per chunk -- ~400 MB on the 6.7M-chunk corpus in #211, held
    while both datasets are built. The guard here was `if not source_values:`,
    which raises "truth value of an array with more than one element is
    ambiguous" on an ndarray, so the array could not be passed at all.
    """

    def _spec(self):
        from leech.confounds import parse_confound_token

        return parse_confound_token("source_group:identity")

    def test_ndarray_column_builds_the_same_encoder_as_a_list(self):
        import numpy as np

        from leech.confounds import build_confound_encoder

        values = ["Ala", "Gly", "Ala", "Ser", "Gly"]
        from_list = build_confound_encoder(self._spec(), source_values=values)
        from_array = build_confound_encoder(self._spec(), source_values=np.array(values, dtype=str))

        assert from_list is not None and from_array is not None
        assert from_array.num_classes == from_list.num_classes
        assert from_array.value_to_class == from_list.value_to_class

    def test_empty_ndarray_disables_adversarial_training(self):
        import numpy as np

        from leech.confounds import build_confound_encoder

        assert build_confound_encoder(self._spec(), source_values=np.array([], dtype=str)) is None
        assert build_confound_encoder(self._spec(), source_values=None) is None

    def test_single_element_ndarray_is_not_mistaken_for_empty(self):
        import numpy as np

        from leech.confounds import build_confound_encoder

        encoder = build_confound_encoder(self._spec(), source_values=np.array(["Ala"], dtype=str))
        assert encoder is not None
        assert encoder.num_classes == 1
