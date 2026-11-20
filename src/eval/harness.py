"""Run a retriever over a question set and score what comes back.

A *retriever* here is just a callable: question in, ranked paragraph ids out. Not a ``Protocol`` — a
protocol with one method is a function type wearing a class, and the four retrievers this will
eventually accept all have the same shape.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from src.eval.dataset import Question
from src.eval.metrics import DEFAULT_KS, Scores, score, score_by_type

#: Question in, ranked paragraph ids out. Best first.
Retriever = Callable[[Question], Sequence[UUID]]


@dataclass(frozen=True, slots=True)
class Report:
    """One evaluation run: the aggregate, and the same split by question type."""

    name: str
    overall: Scores
    by_type: dict[str, Scores]

    def format(self) -> str:
        lines = [f"{self.name}", f"  overall   {self.overall.format()}"]
        for qtype, scores in sorted(self.by_type.items()):
            lines.append(f"  {qtype:<18}{scores.format()}")
        return "\n".join(lines)


def run(
    questions: Sequence[Question],
    retriever: Retriever,
    *,
    name: str = "run",
    ks: Sequence[int] = DEFAULT_KS,
) -> Report:
    """Ask the retriever every question, then score the lot.

    Results are collected before scoring rather than scored as they arrive, so the metrics see the
    whole run at once — macro averaging needs the count up front, and a retriever that dies halfway
    should fail rather than silently report a partial average.
    """
    retrieved: dict[str, Sequence[UUID]] = {q.id: retriever(q) for q in questions}
    return Report(
        name=name,
        overall=score(questions, retrieved, ks),
        by_type=score_by_type(questions, retrieved, ks),
    )
