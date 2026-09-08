"""Load a 2WikiMultihopQA slice into a corpus and a question set.

The benchmark ships each question with 10 paragraphs — a few gold, the rest topically-related
distractors — plus the titles of the gold paragraphs and gold ``(subject, relation, object)``
triples.

Two things this deliberately does differently from how benchmark loaders usually work:

1. **Paragraphs keep an identity.** The usual shape is a flat list of bare strings. That is fine
   for scoring generated *answers*, but it cannot answer "was paragraph X retrieved?" — which is
   exactly what recall@k asks. Every paragraph here carries a content-derived id, and gold is a
   *set of ids*.

2. **Identity is computed from normalised text.** 1,725 titles in the dev split carry two or more
   distinct texts, all of them tokenisation artifacts — ``"Anhalt- Zerbst"`` vs ``"Anhalt-Zerbst"``.
   Hashing raw text would split those into near-duplicate paragraphs and understate recall whenever
   the retriever found the variant that wasn't labelled gold. See ``identity_key``.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Fixed namespace: paragraph ids must be stable across machines and runs, forever.
_NAMESPACE: Final = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

_NON_ALNUM: Final = re.compile(r"[^a-z0-9]+")

#: Instances in the published 2WikiMultihopQA dev split. Asserted on load — a mismatch means the
#: wrong file or a truncated download, and every number measured afterwards would be against a
#: different corpus than the one recorded.
DEV_SPLIT_SIZE: Final = 12_576


def identity_key(text: str) -> str:
    """Reduce text to what makes it *the same paragraph*, discarding presentation.

    NFKC-fold, lowercase, drop every non-alphanumeric character. Aggressive on purpose: measured
    across the full dev split, this brings the number of titles carrying more than one distinct text
    from 1,725 down to **zero**, while never merging two genuinely different subjects.

    Used only to derive ids. The raw text is what gets stored and embedded.
    """
    return _NON_ALNUM.sub("", unicodedata.normalize("NFKC", text).lower())


def paragraph_id(text: str) -> uuid.UUID:
    """Content-derived, deterministic id. Same text, same id, on any machine."""
    return uuid.uuid5(_NAMESPACE, identity_key(text))


@dataclass(frozen=True, slots=True)
class Paragraph:
    """One retrievable unit of the corpus."""

    id: uuid.UUID
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Question:
    """One question, with everything needed to score an answer to it."""

    id: str
    text: str
    answer: str
    gold_paragraph_ids: frozenset[uuid.UUID]
    question_type: str
    gold_triples: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A corpus and the questions asked of it. Paragraphs are deduplicated."""

    paragraphs: tuple[Paragraph, ...]
    questions: tuple[Question, ...]

    @property
    def paragraph_slots(self) -> int:
        """Paragraphs before deduplication — 10 per question."""
        return len(self.questions) * 10

    @property
    def dedup_ratio(self) -> float:
        return self.paragraph_slots / len(self.paragraphs)

    def paragraph(self, paragraph_id: uuid.UUID) -> Paragraph:
        for p in self.paragraphs:
            if p.id == paragraph_id:
                return p
        raise KeyError(paragraph_id)


def _load_raw(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        raw: Any = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(raw).__name__}")
    instances: list[dict[str, Any]] = raw  # pyright: ignore[reportUnknownVariableType]
    if len(instances) != DEV_SPLIT_SIZE:
        raise ValueError(
            f"{path}: expected {DEV_SPLIT_SIZE:,} instances (published dev split), "
            f"found {len(instances):,}. Wrong file or truncated download."
        )
    return instances


def build_eval_set(instances: list[dict[str, Any]]) -> EvalSet:
    """Assemble an :class:`EvalSet` from raw benchmark instances."""
    paragraphs: dict[uuid.UUID, Paragraph] = {}
    questions: list[Question] = []

    for item in instances:
        # title -> id, for this instance only. A title repeating inside one context always carries
        # the same normalised text (verified across the full dev split), so overwriting is safe.
        local: dict[str, uuid.UUID] = {}
        for title, sentences in item["context"]:
            text = " ".join(sentences)
            pid = paragraph_id(text)
            # First occurrence wins. Iteration order is deterministic, so the stored text is too.
            paragraphs.setdefault(pid, Paragraph(id=pid, title=title, text=text))
            local[title] = pid

        gold = frozenset(local[title] for title, _ in item["supporting_facts"])
        if not gold:
            raise ValueError(f"{item['_id']}: no gold paragraphs resolved")

        questions.append(
            Question(
                id=item["_id"],
                text=item["question"],
                answer=item["answer"],
                gold_paragraph_ids=gold,
                question_type=item["type"],
                gold_triples=tuple(
                    (s, r, o) for s, r, o in item.get("evidences", []) if s and r and o
                ),
            )
        )

    return EvalSet(paragraphs=tuple(paragraphs.values()), questions=tuple(questions))


def load(
    path: Path,
    *,
    instances: int = 100,
    seed: int = 42,
    holdout: bool = False,
) -> EvalSet:
    """Load a reproducible slice of the dev split.

    Draws ``2 * instances`` and splits them in half: the first half is the working set, the
    second is the holdout. Sampling twice with different seeds would overlap — measured at 3 of
    100 — and a holdout sharing instances with the set you tuned on is not a holdout.

    The holdout exists to be opened **once**, at the end. Every number quoted from the working set
    has had design decisions tuned against it; the holdout's has not.
    """
    if instances < 1:
        raise ValueError("instances must be >= 1")

    pool = _load_raw(path)
    if 2 * instances > len(pool):
        raise ValueError(f"need {2 * instances:,} instances, split has {len(pool):,}")

    # A local Random keeps this reproducible without touching global random state.
    import random

    drawn = random.Random(seed).sample(pool, 2 * instances)
    chosen = drawn[instances:] if holdout else drawn[:instances]
    return build_eval_set(chosen)
