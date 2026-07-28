"""Confound handling for adversarial (gradient-reversal) training.

Adversarial training decorrelates the learned representation from a *confound*:
some per-chunk categorical variable the model should NOT use. The gradient
reversal machinery (``losses.AdversarialHead``) only needs, per chunk, an
integer confound class (or ``-1`` to ignore) plus a class count. Everything
domain-specific lives in how that integer is derived from chunk metadata.

That derivation is described declaratively by a :class:`ConfoundSpec`:

* ``source`` -- which chunk field holds the raw value
  (e.g. ``"reference_name"``, ``"label_int"``, ``"source_group"``).
* ``mapping`` -- how a raw value becomes a class int:

  * ``"identity"`` -- each distinct observed value gets its own class
    (this is the generic form of the old ``trna_id`` mode).
  * ``"lookup"`` -- translate raw values through an external table
    ``{value: category}`` and map categories to contiguous ints
    (the generic form of the old ``disc_base`` mode).

Two built-in aliases reproduce the original tRNA-specific behaviour:

* ``trna_id``   == ``ConfoundSpec("reference_name", "identity")``
* ``disc_base`` == a ``label_int`` lookup through ``disc_base_map.json``.

The discriminator base (Sprinzl position 73, one nt 5' of the CCA tail) is a
tRNA-specific sequence confound; :func:`extract_disc_bases_from_fasta` derives
its per-amino-acid map at pipeline time. The training path never hardcodes
organism-specific assignments -- it only consumes the generic sidecar.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DISC_BASE_TO_INT: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

NUM_DISC_BASES = 4

logger = logging.getLogger("leech.confounds")


def extract_disc_bases_from_fasta(
    fasta_path: str | Path,
    motif: str = "CCAGGC",
) -> dict[str, str]:
    """Extract per-amino-acid discriminator base from a reference FASTA.

    For each tRNA record, locates ``motif`` (typically CCAGGC: the universal
    CCA tail + the first 3 nt of the 3' adaptor) and reads the base one
    position 5' of the motif — the Sprinzl-73 discriminator base.

    When multiple isoacceptors of the same amino acid disagree on the
    discriminator base (e.g. tRNA-Arg-ACG: A vs. tRNA-Arg-CCG: C), the
    modal base is chosen and a warning is logged. Adversarial training
    will still see the per-AA mode; per-isoacceptor disagreement is the
    motivation for the ``trna_id`` confound mode.

    Args:
        fasta_path: Path to a FASTA file with headers of the form
            ``>tRNA-{AminoAcid}-{anticodon}-{copy}-{variant}``.
        motif: Sequence that anchors the disc_base position. The base
            immediately upstream of the first occurrence of ``motif`` in
            each record is the discriminator base.

    Returns:
        ``{amino_acid: base}`` where ``base`` ∈ ``{"A", "C", "G", "T"}``.

    Raises:
        ValueError: if the motif is not found in a record, occurs at the
            start of a record (no upstream base), or yields a non-ACGT
            discriminator base.
    """
    fasta_path = Path(fasta_path)
    per_aa_bases: dict[str, list[str]] = defaultdict(list)
    per_aa_records: dict[str, list[str]] = defaultdict(list)

    header: str | None = None
    seq_parts: list[str] = []

    def _finalize(header: str | None, seq: str) -> None:
        if header is None:
            return
        # Header form: tRNA-{AA}-{anticodon}-...
        parts = header.split("-")
        if len(parts) < 2 or parts[0] != "tRNA":
            raise ValueError(
                f"Cannot parse amino acid from FASTA header '{header}' (expected 'tRNA-{{AA}}-...')"
            )
        aa = parts[1]
        seq_upper = seq.upper().replace("U", "T")
        pos = seq_upper.find(motif)
        if pos < 0:
            raise ValueError(
                f"Motif '{motif}' not found in record '{header}' (sequence length {len(seq)})"
            )
        if pos == 0:
            raise ValueError(
                f"Motif '{motif}' at position 0 in record '{header}' "
                "leaves no upstream base for the discriminator"
            )
        base = seq_upper[pos - 1]
        if base not in DISC_BASE_TO_INT:
            raise ValueError(f"Non-ACGT discriminator base '{base}' before motif in '{header}'")
        per_aa_bases[aa].append(base)
        per_aa_records[aa].append(header)

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                _finalize(header, "".join(seq_parts))
                header = line[1:].split()[0]  # strip ">" and any trailing comment
                seq_parts = []
            else:
                seq_parts.append(line)
        _finalize(header, "".join(seq_parts))

    disc_map: dict[str, str] = {}
    for aa, bases in per_aa_bases.items():
        counts = Counter(bases)
        modal_base, modal_count = counts.most_common(1)[0]
        if len(counts) > 1:
            logger.warning(
                "disc_base for %s varies across %d isoacceptors: %s; using mode '%s'. "
                "Consider --confound trna_id for per-isoacceptor adversarial training.",
                aa,
                len(bases),
                dict(counts),
                modal_base,
            )
        disc_map[aa] = modal_base
        logger.info("disc_base[%s] = %s (n=%d/%d)", aa, modal_base, modal_count, len(bases))

    return disc_map


def load_disc_base_map(data_dir: Path) -> dict[str, str] | None:
    """Load disc_base_map.json sidecar from a data directory.

    Looks in ``data_dir`` and ``data_dir/..`` (for k-fold subdirectories).

    Returns:
        ``{amino_acid: base}`` mapping, or None if not found.
    """
    for parent in [data_dir, data_dir.parent]:
        path = parent / "disc_base_map.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def build_confound_map(
    label_map: dict[str, int],
    disc_base_map: dict[str, str] | None = None,
) -> dict[int, int]:
    """Build mapping from label_int to confound class int.

    Args:
        label_map: ``{amino_acid_name: label_int}`` from ``label_map.json``.
        disc_base_map: ``{amino_acid: base}`` mapping.  Required — no
            hardcoded defaults.

    Returns:
        ``{label_int: confound_class_int}`` for amino acids with known
        discriminator bases.  Classes not found in *disc_base_map*
        (e.g. ``"uncharged"``) are omitted — the adversarial loss will
        ignore them via ``ignore_index=-1``.

    Raises:
        ValueError: If *disc_base_map* is None.
    """
    if disc_base_map is None:
        raise ValueError(
            "disc_base_map is required. Generate disc_base_map.json from the "
            "reference FASTA using the extract_disc_bases Snakemake rule."
        )

    confound_map: dict[int, int] = {}
    for aa_name, label_int in label_map.items():
        base = disc_base_map.get(aa_name)
        if base is not None:
            confound_map[label_int] = DISC_BASE_TO_INT[base]

    return confound_map


def build_trna_identity_map(
    reference_names: list[str],
) -> tuple[dict[str, int], int]:
    """Build a per-isoacceptor confound map from chunk reference names.

    Each unique tRNA reference name (e.g. ``tRNA-Ser-CGA-1-1``) becomes
    a distinct confound class. This is a much stronger adversarial target
    than the 4-class discriminator base, because the model is penalised
    for encoding *any* tRNA-body information — not just the single
    nucleotide at Sprinzl position 73.

    Args:
        reference_names: Array or list of per-chunk reference name strings.
            Typically read from ``npz["reference_names"]``.

    Returns:
        (ref_to_int, num_classes) where ``ref_to_int`` maps each unique
        reference name to a contiguous integer class, and ``num_classes``
        is the total number of unique tRNAs.
    """
    import numpy as np

    unique = sorted(set(np.asarray(reference_names).flat))
    ref_to_int = {name: i for i, name in enumerate(unique)}
    logger.info(f"tRNA identity confound: {len(ref_to_int)} unique isoacceptors")
    for name, idx in ref_to_int.items():
        logger.debug(f"  {name} -> {idx}")
    return ref_to_int, len(ref_to_int)


# ---------------------------------------------------------------------------
# Generic, config-driven confound layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfoundSpec:
    """Declarative description of an adversarial confound.

    Args:
        source: Chunk-dict field holding the raw confound value
            (e.g. ``"reference_name"``, ``"label_int"``, ``"source_group"``).
        mapping: ``"identity"`` (each distinct value is its own class) or
            ``"lookup"`` (translate values through an external ``table``).
        table: For ``mapping="lookup"``, path to a JSON/YAML sidecar mapping
            ``{raw_value: category}``. Resolved relative to the training data
            directory when not absolute. Ignored by the built-in ``disc_base``
            alias, which loads ``disc_base_map.json`` via :func:`load_disc_base_map`.
        name: Human-readable label used in logging. Defaults to ``source``.
        builtin: Internal marker for the ``disc_base`` alias, which composes
            the lookup through ``label_map`` and the fixed A/C/G/T class space.
    """

    source: str
    mapping: str = "identity"
    table: str | None = None
    name: str | None = None
    builtin: str | None = None

    def __post_init__(self) -> None:
        if self.mapping not in ("identity", "lookup"):
            raise ValueError(
                f"Unknown confound mapping '{self.mapping}' (expected 'identity' or 'lookup')"
            )
        if self.mapping == "lookup" and self.table is None and self.builtin is None:
            raise ValueError("confound mapping='lookup' requires a 'table' path")

    @property
    def label(self) -> str:
        return self.name or self.source

    def to_token(self) -> str:
        """Serialize to the compact string carried through the training config."""
        if self.builtin is not None:
            return self.builtin
        if self.mapping == "lookup":
            return f"{self.source}:lookup:{self.table}"
        return f"{self.source}:identity"


# Built-in aliases reproducing the original tRNA-specific confounds.
BUILTIN_CONFOUNDS: dict[str, ConfoundSpec] = {
    "trna_id": ConfoundSpec(
        source="reference_name", mapping="identity", name="trna_id", builtin="trna_id"
    ),
    "disc_base": ConfoundSpec(
        source="label_int", mapping="lookup", name="disc_base", builtin="disc_base"
    ),
}


def parse_confound_token(token: str) -> ConfoundSpec:
    """Parse a ``--confound`` string into a :class:`ConfoundSpec`.

    Accepts a built-in alias (``disc_base``, ``trna_id``) or an inline spec
    ``source:mapping[:table]`` (e.g. ``reference_name:identity`` or
    ``source_group:lookup:groups.json``).
    """
    token = token.strip()
    if ":" not in token:
        try:
            return BUILTIN_CONFOUNDS[token]
        except KeyError:
            raise ValueError(
                f"Unknown confound '{token}'. Use a built-in alias "
                f"({', '.join(sorted(BUILTIN_CONFOUNDS))}) or an inline spec "
                "'source:mapping[:table]'."
            ) from None
    parts = token.split(":")
    if len(parts) == 2:
        source, mapping = parts
        return ConfoundSpec(source=source, mapping=mapping)
    if len(parts) == 3:
        source, mapping, table = parts
        return ConfoundSpec(source=source, mapping=mapping, table=table)
    raise ValueError(f"Malformed confound spec '{token}' (expected 'source:mapping[:table]')")


def load_confound_config(path: str | Path) -> list[ConfoundSpec]:
    """Load a list of :class:`ConfoundSpec` from a JSON or YAML config file.

    Accepts a top-level list, or a mapping with a ``confounds:`` key. Each
    entry is either a built-in alias string or a mapping with ``source`` /
    ``mapping`` / ``table`` / ``name`` keys.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if isinstance(data, dict):
        data = data.get("confounds", data)
    if not isinstance(data, list):
        raise ValueError(
            f"Confound config '{path}' must be a list of confounds "
            "(or a mapping with a 'confounds:' list)."
        )

    specs: list[ConfoundSpec] = []
    for entry in data:
        if isinstance(entry, str):
            specs.append(parse_confound_token(entry))
        elif isinstance(entry, dict):
            if "source" not in entry:
                raise ValueError(f"Confound config entry missing 'source': {entry}")
            specs.append(
                ConfoundSpec(
                    source=entry["source"],
                    mapping=entry.get("mapping", "identity"),
                    table=entry.get("table"),
                    name=entry.get("name"),
                )
            )
        else:
            raise ValueError(f"Invalid confound config entry: {entry!r}")
    return specs


@dataclass
class ConfoundEncoder:
    """Maps a chunk to a confound class integer (``-1`` = ignore).

    Built once from the training data, then reused for train/val datasets so
    both share the same value-to-class assignment.
    """

    name: str
    source: str
    value_to_class: dict
    num_classes: int

    def encode(self, chunk: dict) -> int:
        return self.value_to_class.get(chunk.get(self.source), -1)


def _load_lookup_table(table: str, data_dir: Path | None) -> dict:
    """Load a ``{raw_value: category}`` sidecar for ``mapping='lookup'``."""
    path = Path(table)
    if not path.is_absolute() and data_dir is not None and not path.exists():
        path = data_dir / table
    text = Path(path).read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def build_confound_encoder(
    spec: ConfoundSpec,
    *,
    source_values: list | None = None,
    label_map: dict[str, int] | None = None,
    data_dir: Path | None = None,
) -> ConfoundEncoder | None:
    """Construct a :class:`ConfoundEncoder` from a spec and the training data.

    Args:
        spec: The confound description.
        source_values: Per-chunk raw values of ``spec.source`` from the training
            set. Required for ``mapping="identity"``.
        label_map: ``{class_name: label_int}`` -- required by the ``disc_base``
            built-in to compose the discriminator-base lookup.
        data_dir: Training data directory, used to resolve sidecar tables and
            to locate ``disc_base_map.json``.

    Returns:
        A ready encoder, or ``None`` (with a warning) when required inputs are
        missing -- letting the caller disable adversarial training gracefully.
    """
    # Built-in discriminator-base confound: label_int -> A/C/G/T class.
    if spec.builtin == "disc_base":
        if label_map is None:
            logger.warning("confound 'disc_base' needs a label_map; disabling adversarial training")
            return None
        disc_base_map = load_disc_base_map(data_dir) if data_dir is not None else None
        confound_map = build_confound_map(label_map, disc_base_map=disc_base_map)
        logger.info(f"Adversarial confound 'disc_base': {confound_map}")
        return ConfoundEncoder(
            name=spec.label,
            source="label_int",
            value_to_class=confound_map,
            num_classes=NUM_DISC_BASES,
        )

    if spec.mapping == "identity":
        if not source_values:
            logger.warning(
                "confound '%s' (identity on '%s') found no source values; "
                "disabling adversarial training",
                spec.label,
                spec.source,
            )
            return None
        unique = sorted({v for v in source_values if v not in (None, "")})
        value_to_class = {v: i for i, v in enumerate(unique)}
        logger.info(
            "Adversarial confound '%s': identity on '%s', %d classes",
            spec.label,
            spec.source,
            len(value_to_class),
        )
        return ConfoundEncoder(spec.label, spec.source, value_to_class, len(value_to_class))

    # mapping == "lookup"
    assert spec.table is not None
    table = _load_lookup_table(spec.table, data_dir)
    categories = sorted(set(table.values()))
    cat_to_int = {c: i for i, c in enumerate(categories)}
    value_to_class = {value: cat_to_int[cat] for value, cat in table.items()}
    logger.info(
        "Adversarial confound '%s': lookup on '%s' via %s, %d classes",
        spec.label,
        spec.source,
        spec.table,
        len(categories),
    )
    return ConfoundEncoder(spec.label, spec.source, value_to_class, len(categories))
