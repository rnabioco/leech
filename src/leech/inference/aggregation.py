"""Aggregation functions for combining multi-model predictions."""

import logging

logger = logging.getLogger("leech.inference")

# Pair name -> [negative_class, positive_class], as recorded by the bundler.
PairLabels = dict[str, list[str]] | None

_warned_legacy_split = False


def resolve_pair_labels(pair: str, pair_labels: PairLabels) -> tuple[str, str]:
    """Return ``(negative_class, positive_class)`` for a pairwise model.

    Prefers the ``pair_labels`` map recorded at bundle time, which comes from
    each model's own ``label_map``. Falls back to splitting the pair name on the
    first underscore for bundles written before that map existed — a fallback
    that is wrong whenever a class name contains an underscore, hence the
    warning.
    """
    if pair_labels:
        labels = pair_labels.get(pair)
        if labels and len(labels) == 2:
            return labels[0], labels[1]

    global _warned_legacy_split
    if not _warned_legacy_split:
        _warned_legacy_split = True
        logger.warning(
            "Bundle has no pair_labels metadata; recovering class names by "
            "splitting pair names on '_'. This is incorrect if any class name "
            "contains an underscore. Re-bundle to record labels explicitly."
        )
    negative, _, positive = pair.partition("_")
    return negative, positive


def aggregate_pairwise(
    pairs: list[str], probs: list[float], pair_labels: PairLabels = None
) -> tuple[str, float, dict[str, float]]:
    """
    Aggregate pairwise model probabilities into a single class prediction.

    Label convention: for pair "A_B", A = label 0 and B = label 1.
    Sigmoid probability p: vote strength (1-p) goes to A, p goes to B.

    Args:
        pairs: List of pair names (e.g., ["Ala_Gly", "Ala_Ser", "Gly_Ser"])
        probs: List of sigmoid probabilities, one per pair
        pair_labels: Bundle metadata mapping each pair to its two class names

    Returns:
        Tuple of (predicted_label, confidence, vote_totals) where:
        - predicted_label: class with highest total vote strength
        - confidence: winner's vote fraction (winner_total / sum_all_totals)
        - vote_totals: dict mapping class -> total vote strength
    """
    votes: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        label_a, label_b = resolve_pair_labels(pair, pair_labels)
        votes[label_a] = votes.get(label_a, 0.0) + (1.0 - prob)
        votes[label_b] = votes.get(label_b, 0.0) + prob

    total = sum(votes.values())
    predicted_label = max(votes, key=votes.__getitem__)
    confidence = votes[predicted_label] / total if total > 0 else 0.0
    return predicted_label, confidence, votes


def aggregate_pairwise_weighted(
    pairs: list[str], probs: list[float], pair_labels: PairLabels = None
) -> tuple[str, float, dict[str, float]]:
    """Pairwise aggregation with confidence weighting.

    Models seeing OOD data produce p ~ 0.5 (uncertain). Weight each
    model's vote by its confidence: w = |2p - 1|, so uncertain models
    contribute near-zero and confident models dominate.
    """
    votes: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        label_a, label_b = resolve_pair_labels(pair, pair_labels)
        confidence = abs(2 * prob - 1)  # 0 at p=0.5, 1 at p=0 or p=1
        votes[label_a] = votes.get(label_a, 0.0) + (1.0 - prob) * confidence
        votes[label_b] = votes.get(label_b, 0.0) + prob * confidence
    total = sum(votes.values())
    predicted_label = max(votes, key=votes.__getitem__)
    confidence_score = votes[predicted_label] / total if total > 0 else 0.0
    return predicted_label, confidence_score, votes


def aggregate_pairwise_tournament(
    pairs: list[str], probs: list[float], pair_labels: PairLabels = None, top_k: int = 5
) -> tuple[str, float, dict[str, float]]:
    """Tournament-style aggregation.

    Round 1: naive vote to identify top-K candidates.
    Round 2: only aggregate models involving top-K classes.
    Eliminates 90%+ of OOD noise.
    """
    # Round 1: get initial rankings
    _, _, initial_votes = aggregate_pairwise(pairs, probs, pair_labels)
    top_labels = sorted(initial_votes, key=initial_votes.__getitem__, reverse=True)[:top_k]

    # Round 2: only use relevant models
    final_votes: dict[str, float] = dict.fromkeys(top_labels, 0.0)
    for pair, prob in zip(pairs, probs, strict=True):
        label_a, label_b = resolve_pair_labels(pair, pair_labels)
        if label_a in final_votes and label_b in final_votes:
            final_votes[label_a] += 1.0 - prob
            final_votes[label_b] += prob

    total = sum(final_votes.values())
    predicted_label = max(final_votes, key=final_votes.__getitem__)
    confidence_score = final_votes[predicted_label] / total if total > 0 else 0.0
    return predicted_label, confidence_score, final_votes


def aggregate_one_vs_all(
    pairs: list[str], probs: list[float], pair_labels: PairLabels = None
) -> tuple[str, float, dict[str, float]]:
    """
    Aggregate one-vs-all model probabilities into a single class prediction.

    Label convention: for pair "A_notA", A = label 0 and notA = label 1, so a
    low sigmoid probability means the read is more likely to be the target and
    the score for A is (1 - p).

    Args:
        pairs: List of pair names (e.g., ["Ala_notAla", "Gly_notGly"])
        probs: List of sigmoid probabilities, one per pair
        pair_labels: Bundle metadata mapping each pair to its two class names

    Returns:
        Tuple of (predicted_label, confidence, scores) where:
        - predicted_label: class with highest score
        - confidence: max score
        - scores: dict mapping class -> score
    """
    scores: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        target, _ = resolve_pair_labels(pair, pair_labels)
        scores[target] = 1.0 - prob

    predicted_label = max(scores, key=scores.__getitem__)
    confidence = scores[predicted_label]
    return predicted_label, confidence, scores
