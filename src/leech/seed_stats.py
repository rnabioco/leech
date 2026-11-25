"""Paired-seed comparison statistics (issue #405).

Torch-free and dependency-free so the Snakemake aggregation scripts can import
it. Two arms trained with the same ``--seed`` share a seed's data order and
initialisation RNG draw, so the per-seed *difference* is the quantity to
report, with its SD across seeds; comparing arm means hides that pairing.
"""

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence

# Seeds per arm when none are configured. Effects discussed in the review are
# +0.0005..+0.005 AUROC, below the noise three seeds can resolve.
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def paired_seed_differences(
    rows: Iterable[Mapping],
    baseline_arm: str,
    metrics: Sequence[str],
    group_keys: Sequence[str] = ("pair",),
) -> list[dict]:
    """Per-(group, arm) mean and SD of ``arm - baseline`` matched on seed.

    Seeds present for only one of the two arms are dropped from that
    comparison. ``<metric>_paired_diff_sd`` is the sample SD (ddof=1) and is
    NaN with fewer than two paired seeds. The baseline arm is not emitted.
    """
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        gkey = tuple(row[k] for k in group_keys)
        by_key.setdefault(gkey, {}).setdefault(row["arm"], {})[row["seed"]] = row

    out: list[dict] = []
    for gkey, arms in by_key.items():
        base = arms.get(baseline_arm)
        if base is None:
            continue
        for arm, per_seed in arms.items():
            if arm == baseline_arm:
                continue
            seeds = sorted(set(per_seed) & set(base))
            rec: dict = dict(zip(group_keys, gkey, strict=True))
            rec["arm"] = arm
            rec["n_paired_seeds"] = len(seeds)
            for metric in metrics:
                diffs = [
                    per_seed[s][metric] - base[s][metric]
                    for s in seeds
                    if metric in per_seed[s] and metric in base[s]
                ]
                rec[f"{metric}_paired_diff_mean"] = statistics.mean(diffs) if diffs else math.nan
                rec[f"{metric}_paired_diff_sd"] = (
                    statistics.stdev(diffs) if len(diffs) > 1 else math.nan
                )
            out.append(rec)
    return out
