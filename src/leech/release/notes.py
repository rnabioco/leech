"""
Markdown release-notes renderer.

Merges editorial metadata from the YAML spec with technical metadata
extracted from the ``.pt`` bundle.
"""

from __future__ import annotations

from typing import Any

from leech.release.spec import ReleaseSpec


def render_release_notes(
    spec: ReleaseSpec,
    bundle_metadata: dict[str, Any] | None = None,
    bundle_size_mb: float | None = None,
) -> str:
    """Render Markdown release notes from spec + bundle metadata."""
    sections: list[str] = []

    # --- Description ---
    sections.append(spec.description.strip())

    # --- Bundle Details ---
    if bundle_metadata:
        lines = ["## Bundle Details", ""]
        lines.append(f"- **Architecture**: {bundle_metadata.get('architecture', 'unknown')}")
        lines.append(f"- **Comparison type**: {bundle_metadata.get('comparison_type', 'unknown')}")
        lines.append(f"- **Number of models**: {bundle_metadata.get('num_models', 'unknown')}")

        is_vmap = bundle_metadata.get("vmap", False)
        is_ts = bundle_metadata.get("torchscript", False)
        fmt = "vmap" if is_vmap else ("TorchScript" if is_ts else "state_dict")
        lines.append(f"- **Format**: {fmt}")

        if bundle_size_mb is not None:
            lines.append(f"- **File size**: {bundle_size_mb:.1f} MB")

        pairs = bundle_metadata.get("pairs", [])
        if pairs:
            if len(pairs) <= 20:
                pairs_str = ", ".join(f"`{p}`" for p in pairs)
            else:
                shown = ", ".join(f"`{p}`" for p in pairs[:20])
                pairs_str = f"{shown}, ... ({len(pairs)} total)"
            lines.append(f"- **Pairs**: {pairs_str}")

        sections.append("\n".join(lines))

    # --- Context ---
    context_lines: list[str] = []
    if spec.organism:
        context_lines.append(f"- **Organism**: {spec.organism}")
    if spec.experiment:
        context_lines.append(f"- **Experiment**: {spec.experiment}")
    if spec.data_version:
        context_lines.append(f"- **Data version**: {spec.data_version}")
    if context_lines:
        sections.append("## Context\n\n" + "\n".join(context_lines))

    # --- Performance ---
    m = spec.metrics
    metric_rows: list[str] = []
    if m.mean_accuracy is not None:
        metric_rows.append(f"| Mean accuracy | {m.mean_accuracy:.4f} |")
    if m.mean_f1 is not None:
        metric_rows.append(f"| Mean F1 | {m.mean_f1:.4f} |")
    if m.mean_auroc is not None:
        metric_rows.append(f"| Mean AUROC | {m.mean_auroc:.4f} |")
    for key, val in m.extra.items():
        metric_rows.append(f"| {key} | {val:.4f} |")

    if metric_rows:
        header = "| Metric | Value |\n| --- | --- |"
        sections.append("## Performance\n\n" + header + "\n" + "\n".join(metric_rows))

    # --- Training Provenance ---
    t = spec.training
    prov_lines: list[str] = []
    if t.leech_version:
        prov_lines.append(f"- **leech version**: {t.leech_version}")
    if t.pipeline_commit:
        prov_lines.append(f"- **Pipeline commit**: `{t.pipeline_commit}`")
    if t.config_file:
        prov_lines.append(f"- **Config file**: `{t.config_file}`")
    if t.date:
        prov_lines.append(f"- **Training date**: {t.date}")
    if prov_lines:
        sections.append("## Training Provenance\n\n" + "\n".join(prov_lines))

    # --- Download & Usage ---
    tag = spec.tag
    bundle_filename = spec.bundle.rsplit("/", 1)[-1] if "/" in spec.bundle else spec.bundle
    usage = f"""\
## Download & Usage

```bash
# Download with leech
leech model fetch --tag {tag}

# Or download with gh
gh release download {tag} --pattern "*.pt"

# Inspect
leech model bundle-info --bundle {bundle_filename}

# Run inference
leech predict --bundle {bundle_filename} --all --pod5 reads.pod5 --bam alignments.bam -o predictions.bam
```"""
    sections.append(usage)

    return "\n\n".join(sections) + "\n"
