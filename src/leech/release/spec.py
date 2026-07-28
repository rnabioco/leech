"""
YAML release-spec parser for model releases.

A release spec describes the editorial metadata for a model bundle
(description, organism, curated metrics).  Technical metadata (architecture,
pairs, config) lives inside the ``.pt`` bundle itself; the release-notes
renderer merges both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("leech.release_spec")


@dataclass
class TrainingProvenance:
    """Where and how the model was trained."""

    pipeline_commit: str | None = None
    leech_version: str | None = None
    config_file: str | None = None
    date: str | None = None


@dataclass
class MetricsSummary:
    """Curated evaluation metrics."""

    mean_accuracy: float | None = None
    mean_f1: float | None = None
    mean_auroc: float | None = None
    details: str | None = None
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class ReleaseSpec:
    """Parsed YAML release specification."""

    name: str
    version: str
    description: str
    bundle: str

    organism: str | None = None
    experiment: str | None = None
    data_version: str | None = None

    training: TrainingProvenance = field(default_factory=TrainingProvenance)
    metrics: MetricsSummary = field(default_factory=MetricsSummary)

    prerelease: bool = False
    extra_assets: list[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        return f"model-{self.name}-v{self.version}"


def load_release_spec(path: Path) -> ReleaseSpec:
    """Load and parse a YAML release-spec file."""
    import yaml

    text = path.read_text()
    raw: dict = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    # Required fields
    for key in ("name", "version", "description", "bundle"):
        if key not in raw:
            raise ValueError(f"Missing required field '{key}' in {path}")

    # Parse nested sections
    training_raw = raw.pop("training", {}) or {}
    training = TrainingProvenance(
        pipeline_commit=training_raw.get("pipeline_commit"),
        leech_version=training_raw.get("leech_version"),
        config_file=training_raw.get("config_file"),
        date=training_raw.get("date"),
    )

    metrics_raw = raw.pop("metrics", {}) or {}
    known_metric_keys = {"mean_accuracy", "mean_f1", "mean_auroc", "details"}
    extra_metrics = {k: v for k, v in metrics_raw.items() if k not in known_metric_keys}
    metrics = MetricsSummary(
        mean_accuracy=metrics_raw.get("mean_accuracy"),
        mean_f1=metrics_raw.get("mean_f1"),
        mean_auroc=metrics_raw.get("mean_auroc"),
        details=metrics_raw.get("details"),
        extra=extra_metrics,
    )

    return ReleaseSpec(
        name=raw["name"],
        version=raw["version"],
        description=raw["description"],
        bundle=raw["bundle"],
        organism=raw.get("organism"),
        experiment=raw.get("experiment"),
        data_version=raw.get("data_version"),
        training=training,
        metrics=metrics,
        prerelease=raw.get("prerelease", False),
        extra_assets=raw.get("extra_assets") or [],
    )


def validate_spec(spec: ReleaseSpec) -> list[str]:
    """Validate a parsed spec.  Returns a list of warnings (empty = OK)."""
    warnings: list[str] = []

    bundle_path = Path(spec.bundle)
    if not bundle_path.exists():
        warnings.append(f"Bundle file not found: {spec.bundle}")
    elif not bundle_path.suffix == ".pt":
        warnings.append(f"Bundle file does not have .pt extension: {spec.bundle}")

    if not spec.description.strip():
        warnings.append("Description is empty")

    for asset in spec.extra_assets:
        if not Path(asset).exists():
            warnings.append(f"Extra asset not found: {asset}")

    return warnings
