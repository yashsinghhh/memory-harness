"""Increment 3 — the harness and the baseline.

The two ends of the chain are pinned here: a retriever that returns exactly the right answers must
score 1.0, and one that returns nothing must score 0.0. If either drifts, everything measured in
between is meaningless.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from src.eval.baselines import random_retriever
from src.eval.dataset import Question
from src.eval.harness import Report, run


def pid(n: int) -> uuid.UUID:
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


CORPUS = [pid(n) for n in range(1, 51)]


# --- the two ends of the chain ------------------------------------------------------------------


def test_a_perfect_retriever_scores_one() -> None:
    """Pins the whole chain. If this ever fails, no number the harness reports means anything."""
    qs = [question("q1", {1, 2}), question("q2", {3, 4, 5, 6})]
    report = run(qs, lambda q: list(q.gold_paragraph_ids), ks=(5, 10))
    assert report.overall.recall_at[5] == 1.0
    assert report.overall.all_gold_at[5] == 1.0
    assert report.overall.mrr == 1.0


def test_a_retriever_that_returns_nothing_scores_zero() -> None:
    qs = [question("q1", {1, 2})]
    report = run(qs, lambda _q: [], ks=(5,))
    assert report.overall.recall_at[5] == 0.0
    assert report.overall.all_gold_at[5] == 0.0
    assert report.overall.mrr == 0.0


# --- the baseline -------------------------------------------------------------------------------


def test_baseline_returns_exactly_k_results() -> None:
    retrieve = random_retriever(CORPUS, k=10)
    assert len(retrieve(question("q1", {1}))) == 10


def test_baseline_ignores_the_question_text() -> None:
    """It is supposed to know nothing — same id, same answer, whatever the question says."""
    retrieve = random_retriever(CORPUS, k=5)
    a = question("same-id", {1})
    b = Question(
        id="same-id",
        text="a completely different question",
        answer="z",
        gold_paragraph_ids=frozenset({pid(9)}),
        question_type="comparison",
        gold_triples=(),
    )
    assert list(retrieve(a)) == list(retrieve(b))


def test_baseline_is_stable_across_question_order() -> None:
    """Seeded per question, not per run — otherwise the floor moves when the question set does."""
    retrieve = random_retriever(CORPUS, k=5)
    q1, q2 = question("q1", {1}), question("q2", {2})
    forwards = [list(retrieve(q1)), list(retrieve(q2))]
    backwards_q2, backwards_q1 = list(retrieve(q2)), list(retrieve(q1))
    assert forwards == [backwards_q1, backwards_q2]


def test_baseline_gives_different_questions_different_results() -> None:
    retrieve = random_retriever(CORPUS, k=10)
    assert list(retrieve(question("q1", {1}))) != list(retrieve(question("q2", {1})))


def test_baseline_never_repeats_a_paragraph() -> None:
    results = random_retriever(CORPUS, k=20)(question("q1", {1}))
    assert len(set(results)) == 20


def test_baseline_refuses_to_draw_more_than_the_corpus_holds() -> None:
    with pytest.raises(ValueError, match="cannot draw"):
        random_retriever(CORPUS, k=len(CORPUS) + 1)


# --- the report ---------------------------------------------------------------------------------


def test_every_question_is_asked() -> None:
    seen: list[str] = []

    def spy(q: Question) -> Sequence[uuid.UUID]:
        seen.append(q.id)
        return []

    run([question("q1", {1}), question("q2", {2})], spy, ks=(5,))
    assert seen == ["q1", "q2"]


def test_report_breaks_down_by_question_type() -> None:
    qs = [question("q1", {1}, "compositional"), question("q2", {2}, "bridge_comparison")]
    report: Report = run(qs, lambda q: list(q.gold_paragraph_ids), ks=(5,))
    assert set(report.by_type) == {"compositional", "bridge_comparison"}
    assert "compositional" in report.format()
