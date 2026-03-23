"""Aggregation functions for combining multi-model predictions."""


def aggregate_pairwise(pairs: list[str], probs: list[float]) -> tuple[str, float, dict[str, float]]:
    """
    Aggregate pairwise model probabilities into a single amino acid prediction.

    Label convention: pair "A_B" (alphabetical) -> A = label 0, B = label 1.
    Sigmoid probability p: vote strength (1-p) goes to A, p goes to B.

    Args:
        pairs: List of pair names (e.g., ["Ala_Gly", "Ala_Ser", "Gly_Ser"])
        probs: List of sigmoid probabilities, one per pair

    Returns:
        Tuple of (predicted_aa, confidence, vote_totals) where:
        - predicted_aa: amino acid with highest total vote strength
        - confidence: winner's vote fraction (winner_total / sum_all_totals)
        - vote_totals: dict mapping amino acid -> total vote strength
    """
    votes: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        aa_a, aa_b = pair.split("_", 1)
        votes[aa_a] = votes.get(aa_a, 0.0) + (1.0 - prob)
        votes[aa_b] = votes.get(aa_b, 0.0) + prob

    total = sum(votes.values())
    predicted_aa = max(votes, key=votes.__getitem__)
    confidence = votes[predicted_aa] / total if total > 0 else 0.0
    return predicted_aa, confidence, votes


def aggregate_pairwise_weighted(
    pairs: list[str], probs: list[float]
) -> tuple[str, float, dict[str, float]]:
    """Pairwise aggregation with confidence weighting.

    Models seeing OOD data produce p ~ 0.5 (uncertain). Weight each
    model's vote by its confidence: w = |2p - 1|, so uncertain models
    contribute near-zero and confident models dominate.
    """
    votes: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        aa_a, aa_b = pair.split("_", 1)
        confidence = abs(2 * prob - 1)  # 0 at p=0.5, 1 at p=0 or p=1
        votes[aa_a] = votes.get(aa_a, 0.0) + (1.0 - prob) * confidence
        votes[aa_b] = votes.get(aa_b, 0.0) + prob * confidence
    total = sum(votes.values())
    predicted_aa = max(votes, key=votes.__getitem__)
    confidence_score = votes[predicted_aa] / total if total > 0 else 0.0
    return predicted_aa, confidence_score, votes


def aggregate_pairwise_tournament(
    pairs: list[str], probs: list[float], top_k: int = 5
) -> tuple[str, float, dict[str, float]]:
    """Tournament-style aggregation.

    Round 1: naive vote to identify top-K candidates.
    Round 2: only aggregate models involving top-K AAs.
    Eliminates 90%+ of OOD noise.
    """
    # Round 1: get initial rankings
    _, _, initial_votes = aggregate_pairwise(pairs, probs)
    top_aas = sorted(initial_votes, key=initial_votes.__getitem__, reverse=True)[:top_k]

    # Round 2: only use relevant models
    final_votes: dict[str, float] = dict.fromkeys(top_aas, 0.0)
    for pair, prob in zip(pairs, probs, strict=True):
        aa_a, aa_b = pair.split("_", 1)
        if aa_a in final_votes and aa_b in final_votes:
            final_votes[aa_a] += 1.0 - prob
            final_votes[aa_b] += prob

    total = sum(final_votes.values())
    predicted_aa = max(final_votes, key=final_votes.__getitem__)
    confidence_score = final_votes[predicted_aa] / total if total > 0 else 0.0
    return predicted_aa, confidence_score, final_votes


def aggregate_one_vs_all(
    pairs: list[str], probs: list[float]
) -> tuple[str, float, dict[str, float]]:
    """
    Aggregate one-vs-all model probabilities into a single amino acid prediction.

    Label convention: pair "A_notA" -> A = label 0, notA = label 1.
    Score for AA = (1 - p), since low probability means more likely to be the target.

    Args:
        pairs: List of pair names (e.g., ["Ala_notAla", "Gly_notGly"])
        probs: List of sigmoid probabilities, one per pair

    Returns:
        Tuple of (predicted_aa, confidence, scores) where:
        - predicted_aa: amino acid with highest score
        - confidence: max score
        - scores: dict mapping amino acid -> score
    """
    scores: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        aa = pair.split("_", 1)[0]
        scores[aa] = 1.0 - prob

    predicted_aa = max(scores, key=scores.__getitem__)
    confidence = scores[predicted_aa]
    return predicted_aa, confidence, scores
