"""Increment 2 — the metrics.

Every expected value here is worked out by hand and written into the assertion. That is the whole
point of building the ruler before anything to measure: a wrong metric does not look wrong once real
retrieval is running, it looks like mediocre retrieval.
"""

from __future__ import annotations

import uuid

import pytest

from src.eval.dataset import Question
from src.eval.metrics import (
    Scores,
    all_gold_at_k,
    recall_at_k,
    reciprocal_rank,
    score,
    score_by_type,
)


def pid(n: int) -> uuid.UUID:
    """Readable stand-in ids: pid(1), pid(2), ..."""
    return uuid.UUID(int=n)


def question(qid: str, gold: set[int], qtype: str = "compositional") -> Question:
    return Question(
        id=qid,
        text=f"q-{qid}",
        answer="a",
        gold_paragraph_ids=frozenset(pid(n) for n in gold),
        question_type=qtype,
        gold_triples=(),
    )


# --- recall@k ---------------------------------------------------------------------------------


def test_recall_counts_hits_over_gold_size() -> None:
    # gold = {1,2}; top-3 = [1,9,8] → 1 of 2 found → 1/2
    assert recall_at_k([pid(1), pid(9), pid(8)], {pid(1), pid(2)}, 3) == 0.5


def test_recall_denominator_is_gold_not_k() -> None:
    """The precision/recall swap. gold=2, k=10, both found → 2/2 = 1.0, NOT 2/10 = 0.2."""
    ranked = [pid(1), pid(2)] + [pid(n) for n in range(100, 108)]
    assert recall_at_k(ranked, {pid(1), pid(2)}, 10) == 1.0


def test_recall_ignores_results_below_the_cutoff() -> None:
    # gold = {5}; it sits at position 4, so k=3 must not see it
    ranked = [pid(1), pid(2), pid(3), pid(5)]
    assert recall_at_k(ranked, {pid(5)}, 3) == 0.0
    assert recall_at_k(ranked, {pid(5)}, 4) == 1.0


def test_duplicates_waste_their_slot() -> None:
    """A retriever returning the same paragraph 3 times has used 3 of its 3 slots, not 1."""
    ranked = [pid(1), pid(1), pid(1), pid(2)]
    # top-3 is [1,1,1] → only gold 1 found → 1/2
    assert recall_at_k(ranked, {pid(1), pid(2)}, 3) == 0.5


def test_recall_with_four_gold() -> None:
    # bridge_comparison shape: 4 gold, 3 found in top-5 → 3/4
    ranked = [pid(1), pid(2), pid(3), pid(90), pid(91)]
    assert recall_at_k(ranked, {pid(1), pid(2), pid(3), pid(4)}, 5) == 0.75


# --- all_gold@k -------------------------------------------------------------------------------


def test_all_gold_is_true_only_when_every_gold_is_present() -> None:
    ranked = [pid(1), pid(2), pid(3)]
    assert all_gold_at_k(ranked, {pid(1), pid(2)}, 3) is True
    assert all_gold_at_k(ranked, {pid(1), pid(4)}, 3) is False


def test_all_gold_is_false_when_recall_says_half() -> None:
    """The case the metric exists for: recall reads 0.5, the question is unanswerable."""
    ranked = [pid(1), pid(7), pid(8), pid(9), pid(10)]
    gold = {pid(1), pid(2)}
    assert recall_at_k(ranked, gold, 5) == 0.5
    assert all_gold_at_k(ranked, gold, 5) is False


# --- reciprocal rank --------------------------------------------------------------------------


def test_rank_is_one_based() -> None:
    """A hit at the top is 1.0, not infinity and not 0.5. Off-by-one here inflates every score."""
    assert reciprocal_rank([pid(1)], {pid(1)}) == 1.0
    assert reciprocal_rank([pid(9), pid(1)], {pid(1)}) == 0.5
    assert reciprocal_rank([pid(9), pid(8), pid(1)], {pid(1)}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_hit() -> None:
    # gold at positions 2 and 3 → 1/2, not 1/3 and not their mean
    assert reciprocal_rank([pid(9), pid(1), pid(2)], {pid(1), pid(2)}) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_is_found() -> None:
    assert reciprocal_rank([pid(7), pid(8)], {pid(1)}) == 0.0


# --- guards -----------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1])
def test_a_cutoff_below_one_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k([pid(1)], {pid(1)}, k)


def test_an_empty_gold_set_is_rejected() -> None:
    """Silently scoring 0/0 as 1.0 would be worse than failing."""
    with pytest.raises(ValueError, match="gold set is empty"):
        recall_at_k([pid(1)], set(), 5)


# --- aggregation ------------------------------------------------------------------------------


def test_averaging_is_macro_not_micro() -> None:
    """Constructed so the two disagree, and pinned to the macro answer.

    q1: 2 gold, 1 found  → 0.5
    q2: 4 gold, 4 found  → 1.0
    macro = (0.5 + 1.0) / 2       = 0.75
    micro = (1 + 4) / (2 + 4)     = 0.8333...   ← would over-weight the 4-gold question
    """
    qs = [question("q1", {1, 2}), question("q2", {3, 4, 5, 6})]
    retrieved = {
        "q1": [pid(1), pid(90), pid(91), pid(92), pid(93)],
        "q2": [pid(3), pid(4), pid(5), pid(6), pid(94)],
    }
    out = score(qs, retrieved, ks=(5,))
    assert out.recall_at[5] == 0.75


def test_scores_are_averaged_over_every_question() -> None:
    qs = [question("q1", {1}), question("q2", {2})]
    retrieved = {"q1": [pid(1)], "q2": [pid(99)]}  # one hit, one miss
    out = score(qs, retrieved, ks=(1,))
    assert out.questions == 2
    assert out.recall_at[1] == 0.5
    assert out.mrr == 0.5  # (1.0 + 0.0) / 2


def test_a_question_with_no_results_is_scored_not_skipped() -> None:
    """A retriever that silently drops questions must not get a higher average for it."""
    qs = [question("q1", {1}), question("q2", {2})]
    out = score(qs, {"q1": [pid(1)]}, ks=(1,))
    assert out.questions == 2
    assert out.recall_at[1] == 0.5


def test_all_gold_is_not_reported_at_k_equals_one() -> None:
    """With 2+ gold it is always 0 — a property of the cutoff, not of the retriever."""
    out = score([question("q1", {1, 2})], {"q1": [pid(1)]}, ks=(1, 5))
    assert 1 not in out.all_gold_at
    assert 5 in out.all_gold_at


def test_all_gold_aggregate_is_the_fraction_of_answerable_questions() -> None:
    qs = [question("q1", {1, 2}), question("q2", {3, 4})]
    retrieved = {"q1": [pid(1), pid(2)], "q2": [pid(3), pid(99)]}
    out = score(qs, retrieved, ks=(5,))
    assert out.all_gold_at[5] == 0.5  # one of two questions fully answerable


def test_score_by_type_splits_and_scores_independently() -> None:
    qs = [
        question("q1", {1}, "compositional"),
        question("q2", {2}, "compositional"),
        question("q3", {3, 4, 5, 6}, "bridge_comparison"),
    ]
    retrieved = {
        "q1": [pid(1)],
        "q2": [pid(99)],
        "q3": [pid(3), pid(4), pid(5), pid(6)],
    }
    by_type = score_by_type(qs, retrieved, ks=(5,))
    assert set(by_type) == {"compositional", "bridge_comparison"}
    assert by_type["compositional"].questions == 2
    assert by_type["compositional"].recall_at[5] == 0.5
    assert by_type["bridge_comparison"].recall_at[5] == 1.0


def test_scoring_no_questions_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="no questions"):
        score([], {})


def test_format_is_readable() -> None:
    line = Scores(questions=100, recall_at={5: 0.4}, all_gold_at={5: 0.2}, mrr=0.31).format()
    assert "n=100" in line and "R@5 0.400" in line and "MRR 0.310" in line
