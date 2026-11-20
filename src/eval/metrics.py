"""Scoring functions for retrieval.

Three metrics, because one is not enough to describe a multi-hop result honestly:

``recall_at_k``     the fraction of correct paragraphs that came back — moves in small steps, so it
                    shows gradual progress.
``all_gold_at_k``   whether *every* correct paragraph came back — whether the question is answerable
                    at all. A two-hop question answered from one paragraph is not half answered.
``reciprocal_rank`` how high the first correct paragraph appeared — ranking quality, which recall
                    cannot see.

The same result reads differently under each. Two correct paragraphs, one returned at position 1:
recall 0.50, all-gold 0, reciprocal rank 1.00. All three are true.

Everything here is pure: ids in, floats out. No I/O, no state, no store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from uuid import UUID

from src.eval.dataset import Question

#: Cutoffs reported by default. ``all_gold`` is not reported at k=1 — with two or more correct
#: paragraphs it is always 0, which is a property of the cutoff rather than of the retriever.
DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


def _top_k(retrieved: Sequence[UUID], k: int) -> Sequence[UUID]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return retrieved[:k]


def recall_at_k(retrieved: Sequence[UUID], gold: Set[UUID], k: int) -> float:
    """Fraction of the correct paragraphs found in the top k.

    The denominator is ``len(gold)``, not ``k``. Using ``k`` would be precision, and the two are
    easy to swap by accident — most of the reason these functions are written and tested before any
    retriever exists.

    Duplicates in ``retrieved`` waste their slot rather than scoring twice: the top-k slice is taken
    first, and only then reduced to a set.
    """
    if not gold:
        raise ValueError("gold set is empty; a question with no correct answer cannot be scored")
    return len(set(_top_k(retrieved, k)) & gold) / len(gold)


def all_gold_at_k(retrieved: Sequence[UUID], gold: Set[UUID], k: int) -> bool:
    """Whether *every* correct paragraph is in the top k.

    HotpotQA calls this shape supporting-fact exact match. It is the metric that says whether the
    question could actually be answered from what came back.
    """
    if not gold:
        raise ValueError("gold set is empty; a question with no correct answer cannot be scored")
    return gold <= set(_top_k(retrieved, k))


def reciprocal_rank(retrieved: Sequence[UUID], gold: Set[UUID]) -> float:
    """``1 / rank`` of the first correct paragraph, or 0.0 if none appears.

    **Ranks are 1-based**: the first item has rank 1, so a hit at the top scores 1.0. Enumerating
    from zero would inflate every score and look exactly like unusually good ranking.

    With several correct paragraphs this uses the *first* hit — the standard definition.
    Completeness is ``all_gold_at_k``'s job.
    """
    if not gold:
        raise ValueError("gold set is empty; a question with no correct answer cannot be scored")
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True, slots=True)
class Scores:
    """Macro-averaged scores over a set of questions.

    **Macro, not micro**: each question is scored, then the scores are averaged. Pooling every hit
    and every gold into one ratio would weight questions by how many correct paragraphs they have —
    and with gold sets of 2 and 4, the 4-gold questions would silently count double.
    """

    questions: int
    recall_at: Mapping[int, float]
    all_gold_at: Mapping[int, float]
    mrr: float

    def format(self) -> str:
        recall = "  ".join(f"R@{k} {v:.3f}" for k, v in sorted(self.recall_at.items()))
        allg = "  ".join(f"All@{k} {v:.3f}" for k, v in sorted(self.all_gold_at.items()))
        return f"n={self.questions:<4} {recall}  |  {allg}  |  MRR {self.mrr:.3f}"


def score(
    questions: Sequence[Question],
    retrieved: Mapping[str, Sequence[UUID]],
    ks: Sequence[int] = DEFAULT_KS,
) -> Scores:
    """Score a run. ``retrieved`` maps question id to that question's ranked paragraph ids.

    A question missing from ``retrieved`` is scored as having returned nothing, rather than being
    skipped — a retriever that silently drops questions would otherwise raise its own average.
    """
    if not questions:
        raise ValueError("no questions to score")

    # all_gold at k=1 is 0 whenever a question has 2+ correct paragraphs, which is always here.
    all_gold_ks = [k for k in ks if k > 1]

    recall_totals = dict.fromkeys(ks, 0.0)
    all_gold_totals = dict.fromkeys(all_gold_ks, 0.0)
    rr_total = 0.0

    for q in questions:
        ranked = retrieved.get(q.id, ())
        gold = q.gold_paragraph_ids
        for k in ks:
            recall_totals[k] += recall_at_k(ranked, gold, k)
        for k in all_gold_ks:
            all_gold_totals[k] += 1.0 if all_gold_at_k(ranked, gold, k) else 0.0
        rr_total += reciprocal_rank(ranked, gold)

    n = len(questions)
    return Scores(
        questions=n,
        recall_at={k: v / n for k, v in recall_totals.items()},
        all_gold_at={k: v / n for k, v in all_gold_totals.items()},
        mrr=rr_total / n,
    )


def score_by_type(
    questions: Sequence[Question],
    retrieved: Mapping[str, Sequence[UUID]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Scores]:
    """Score per question type.

    Question types differ in how many correct paragraphs they need — 2 for most, 4 for
    ``bridge_comparison`` — so they are genuinely different difficulties. An aggregate hides that,
    and the hardest type is exactly where structure is supposed to help most.
    """
    grouped: dict[str, list[Question]] = {}
    for q in questions:
        grouped.setdefault(q.question_type, []).append(q)
    return {t: score(qs, retrieved, ks) for t, qs in sorted(grouped.items())}
