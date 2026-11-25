"""Diagnostic: how much label noise do the scores imply? (issue #398)

``--label-noise-rate`` is typed by hand; nothing estimates or checks it. This
fits a two-component Gaussian mixture to held-out logits *per source group*. A
group is meant to be one class, so its logits should form one mode on the
labelled side of the decision boundary; mass in a mode on the other side is the
implied flip rate. It is reported beside the supplied rate and changes nothing.

The model's own errors also land in the wrong-side mode, so this is closer to an
upper bound than to a measurement.
"""

import numpy as np

MIN_GROUP_SIZE = 50


def _fit_two_gaussians(
    x: np.ndarray, iters: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EM for a 1-D two-component mixture; returns (weights, means, variances)."""
    lo, hi = np.quantile(x, [0.1, 0.9])
    means = np.array([lo, hi], dtype=np.float64)
    if means[0] == means[1]:
        means = np.array([x.min(), x.max()], dtype=np.float64)
    base_var = max(float(x.var()), 1e-6)
    var = np.full(2, base_var)
    floor = 1e-4 * base_var
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        logp = -0.5 * ((x[:, None] - means) ** 2 / var + np.log(2 * np.pi * var)) + np.log(w)
        logp -= logp.max(axis=1, keepdims=True)
        resp = np.exp(logp)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-12
        new_means = (resp * x[:, None]).sum(axis=0) / nk
        var = np.maximum((resp * (x[:, None] - new_means) ** 2).sum(axis=0) / nk, floor)
        w = nk / nk.sum()
        converged = np.allclose(new_means, means, atol=1e-6)
        means = new_means
        if converged:
            break
    return w, means, var


def implied_flip_rates(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    supplied: dict[str, float] | None = None,
    *,
    min_n: int = MIN_GROUP_SIZE,
) -> dict[str, dict]:
    """Per-group implied flip rate beside the supplied one.

    Args:
        logits: held-out binary logits (positive = class 1).
        labels: 0/1 labels the model was trained against.
        groups: ``source_group`` per chunk.
        supplied: ``{group: rate}`` as passed to ``--label-noise-rate``.
        min_n: groups smaller than this report ``implied_flip_rate=None``.

    The group's labelled class is its majority label. Its implied flip rate is
    the weight of the mixture component whose mean lies on the opposite side of
    0; ``0.0`` when both components sit on the labelled side.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups).astype(str)
    supplied = supplied or {}
    out: dict[str, dict] = {}
    for name in sorted(set(groups.tolist())):
        sel = groups == name
        n = int(sel.sum())
        entry: dict = {
            "n": n,
            "labelled_class": None,
            "implied_flip_rate": None,
            "supplied_rate": supplied.get(name),
        }
        out[name] = entry
        if n < min_n:
            continue
        positive = float((labels[sel] == 1).mean()) >= 0.5
        entry["labelled_class"] = int(positive)
        x = logits[sel] if positive else -logits[sel]  # labelled side is now positive
        w, means, _ = _fit_two_gaussians(x)
        wrong = means < 0
        entry["implied_flip_rate"] = float(w[wrong].sum()) if wrong.any() else 0.0
    return out
