"""Leakage canary (issue #407): does the evaluation protocol let a run shortcut score?

The fixture is built so that the *only* thing separating the classes is which run
a read came from: every read's signal is noise around a per-run offset, and no
read carries any real class information. Two classes with two runs each, runs
sitting at distinct, well-separated offsets.

A model that can only memorise run signatures -- which is what a high-capacity
network does when a run effect is label-predictive -- scores:

* ~1.0 AUROC when train and test share runs (a read-level split), and
* exactly 0.5 when the test runs were never seen in training (the pipeline
  default: a whole run held out), because an unseen signature carries no
  information.

So the second number is the honest one, and the first is the leak. The test
that matters is `test_pipeline_default_split_gives_chance_cross_run_auroc`:
it takes the split flags from the pipeline's own config and rules, so a change
that flips the default back to read-level splitting fails it. The read-level
test next to it proves the canary can actually detect a leak.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from sklearn.metrics import roc_auc_score

from leech.chunking import ChunkTable, load_chunks, save_chunks
from leech.cli import cli

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"

SIGNAL_LEN = 32
KMER_LEN = 11
N_FEAT = 4
READS_PER_RUN = 40
CHUNKS_PER_READ = 2

#: (class, run) -> signal offset. Well separated, so a memoriser can tell runs
#: apart perfectly; arranged so that offset alone is label-predictive.
RUN_OFFSETS = {
    ("A", "runA1"): 0.0,
    ("A", "runA2"): 10.0,
    ("B", "runB1"): 20.0,
    ("B", "runB2"): 30.0,
}


def _chunks(label: str, run: str, offset: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    chunks = []
    for read in range(READS_PER_RUN):
        for j in range(CHUNKS_PER_READ):
            chunks.append(
                {
                    # Noise around the run's offset; nothing class-specific.
                    "signal": (offset + 0.1 * rng.standard_normal(SIGNAL_LEN)).astype(np.float32),
                    "sequence": "ACGTACGTACG",
                    "dwell": np.ones(KMER_LEN, dtype=np.float32),
                    "features": np.zeros((N_FEAT, KMER_LEN), dtype=np.float32),
                    "label": label,
                    "label_int": 0 if label == "A" else 1,
                    "read_id": f"{run}_{read:05d}",
                    "base_idx": 100 + j,
                    "source_group": label,
                    "reference_name": "tRNA",
                    "feature_start": -5,
                    "feature_end": 5,
                    "cl_value": None,
                    "run_id": run,
                    "flow_cell_id": f"FC_{run}",
                }
            )
    return chunks


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    """One input file per class, each holding both of that class's runs."""
    root = tmp_path_factory.mktemp("canary")
    paths = {}
    for i, label in enumerate(("A", "B")):
        chunks = []
        for k, ((cls, run), offset) in enumerate(RUN_OFFSETS.items()):
            if cls == label:
                chunks += _chunks(label, run, offset, seed=10 * i + k)
        path = root / label / "all.npz"
        save_chunks(chunks, path)
        paths[label] = path
    return paths


def _pipeline_default_split_flags() -> list[str]:
    """What the pipeline's merge rules append to `leech data merge` by default.

    Evaluates `merge_split_flags` from the real `common.smk` against the real
    `config.yaml`, rather than restating either: the point is to catch a
    default that drifts back to a read-level split.
    """
    config = yaml.safe_load((PIPELINE / "config" / "config.yaml").read_text())
    tree = ast.parse((PIPELINE / "workflow" / "rules" / "common.smk").read_text())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "merge_split_flags")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SPLIT_MODES" for t in node.targets)
        )
    ]
    assert len(wanted) == 2, "common.smk lost merge_split_flags / SPLIT_MODES"

    class _Quiet:
        def warning(self, *_a, **_k): ...

    namespace = {"config": config, "logger": _Quiet(), "WorkflowError": RuntimeError}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "common.smk", "exec"), namespace)
    return namespace["merge_split_flags"]().split()


def _merge(corpora, out: Path, extra: list[str]) -> None:
    args = [
        "data",
        "merge",
        "-i",
        f"A={corpora['A']}",
        "-i",
        f"B={corpora['B']}",
        "--output-dir",
        str(out),
        "--seed",
        "13",
        *extra,
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output + repr(result.exception)


def _fingerprint(signal: np.ndarray) -> int:
    """A run signature: the read's mean level, snapped to the offset grid."""
    return int(round(float(np.mean(signal)) / 10.0))


def _run_memoriser_auroc(split_dir: Path) -> float:
    """AUROC on test of a model that can only memorise run signatures.

    P(class B | signature) is the training frequency; a signature never seen in
    training gets the uninformative 0.5.
    """
    train = load_chunks(split_dir / "train.npz")
    test = load_chunks(split_dir / "test.npz")
    seen: dict[int, list[int]] = {}
    for chunk in train:
        seen.setdefault(_fingerprint(chunk["signal"]), []).append(int(chunk["label_int"]))
    p_b = {fp: float(np.mean(labels)) for fp, labels in seen.items()}

    labels = np.array([int(c["label_int"]) for c in test])
    scores = np.array([p_b.get(_fingerprint(c["signal"]), 0.5) for c in test])
    assert len(set(labels)) == 2, "test split must contain both classes to have an AUROC"
    return float(roc_auc_score(labels, scores))


def test_fixture_run_offset_alone_predicts_the_label(corpora):
    """The canary's premise: with runs shared, a run-signature memoriser is perfect."""
    chunks = load_chunks(corpora["A"]) + load_chunks(corpora["B"])
    by_fp: dict[int, set[int]] = {}
    for chunk in chunks:
        by_fp.setdefault(_fingerprint(chunk["signal"]), set()).add(int(chunk["label_int"]))
    assert len(by_fp) == len(RUN_OFFSETS)
    assert all(len(labels) == 1 for labels in by_fp.values())


def test_read_level_split_leaks_the_run_effect(corpora, tmp_path):
    """The canary has teeth: a read-level split scores the run shortcut as accuracy."""
    out = tmp_path / "read_level"
    _merge(corpora, out, [])
    assert _run_memoriser_auroc(out) > 0.95


def test_pipeline_default_split_gives_chance_cross_run_auroc(corpora, tmp_path):
    flags = _pipeline_default_split_flags()
    assert flags, (
        "the pipeline's default merge adds no split flags, i.e. it splits at read level "
        "(rnabioco/leech#394): run-level effects would be scored as accuracy"
    )
    out = tmp_path / "pipeline_default"
    _merge(corpora, out, flags)

    auroc = _run_memoriser_auroc(out)
    assert auroc == pytest.approx(0.5, abs=0.02), (
        f"cross-run AUROC {auroc:.3f} on a fixture with no class signal beyond the run offset"
    )

    # And the split really is cross-run: no test run was trained on.
    train_runs = set(ChunkTable.from_npz(out / "train.npz").require_values("run_id").astype(str))
    test_runs = set(ChunkTable.from_npz(out / "test.npz").require_values("run_id").astype(str))
    assert test_runs and not test_runs & train_runs


def test_pipeline_default_is_run_holdout():
    config = yaml.safe_load((PIPELINE / "config" / "config.yaml").read_text())
    assert config["split_mode"] == "run_holdout"


# ---------------------------------------------------------------------------
# Label divergence (issue #402): a corpus whose classes differ only at focus-
# relative offsets >= D. No model is trained; the scorer is the simplest one
# that can read the divergent region.
# ---------------------------------------------------------------------------

DIVERGENCE_P = 20  # from the start of the motif
DIVERGENCE_MOTIF_OFFSET = 2
DIVERGENCE_D = DIVERGENCE_P - DIVERGENCE_MOTIF_OFFSET  # 18


def _divergent_features(feature_end: int, n_per_class: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows over offsets 0..feature_end, class signal only at offsets >= D."""
    rng = np.random.default_rng(402)
    width = feature_end + 1
    labels = np.repeat([0, 1], n_per_class)
    feats = rng.standard_normal((len(labels), width))
    divergent = np.arange(width) >= DIVERGENCE_D
    feats[:, divergent] += np.where(labels[:, None] == 1, 5.0, -5.0)
    return feats, labels


def _score_divergent_region(feats: np.ndarray) -> np.ndarray:
    """Mean over the stored offsets >= D; an uninformative 0.5 when none are stored."""
    divergent = np.arange(feats.shape[1]) >= DIVERGENCE_D
    if not divergent.any():
        return np.full(len(feats), 0.5)
    return feats[:, divergent].mean(axis=1)


def _prepare_window(tmp_path, monkeypatch, feature_end: int) -> None:
    """Run `handle_prepare` (extraction faked) with the divergence declared."""
    from test_chunk_writer import make_chunks

    from leech.commands.prepare import handle_prepare

    def fake_parallel(*, chunk_sink, **_):
        chunk_sink(make_chunks(6))
        return [], {
            "total_reads": 6,
            "reads_with_motif": 6,
            "reads_without_motif": 0,
            "total_chunks": 6,
            "failed_reads": 0,
            "failed_batches": 0,
        }

    monkeypatch.setattr("leech.preparation.prepare_training_data_parallel", fake_parallel)
    handle_prepare(
        pod5=tmp_path / "x.pod5",
        bam=tmp_path / "x.bam",
        output_dir=tmp_path / f"out_{feature_end}",
        motif="CCAGGC",
        motif_offset=DIVERGENCE_MOTIF_OFFSET,
        motif_reference="bam",
        refine_signal_map=False,
        workers=1,
        no_split=True,
        feature_start=0,
        feature_end=feature_end,
        signal_context=(90, 200),
        base_justify="center",
        label_divergence_position=DIVERGENCE_P,
    )


def test_divergent_region_is_a_perfect_label_predictor_when_stored():
    stored, labels = _divergent_features(feature_end=20)
    assert roc_auc_score(labels, _score_divergent_region(stored)) == 1.0

    clipped, labels = _divergent_features(feature_end=DIVERGENCE_D - 1)
    assert roc_auc_score(labels, _score_divergent_region(clipped)) == pytest.approx(0.5)


def test_guard_refuses_the_window_that_stores_the_divergent_region(tmp_path, monkeypatch):
    # The window the scorer wins on is refused ...
    with pytest.raises(ValueError, match="label-divergence"):
        _prepare_window(tmp_path, monkeypatch, feature_end=20)
    # ... and the one it is blind on is accepted.
    _prepare_window(tmp_path, monkeypatch, feature_end=DIVERGENCE_D - 1)
    assert (tmp_path / f"out_{DIVERGENCE_D - 1}" / "all.npz").exists()
