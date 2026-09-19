"""Retrievers that exist to be beaten.

A baseline is not a strategy anyone would use. It is a reference point — the score of knowing
nothing, which everything built afterwards has to clear. It lives under ``eval/`` rather than
``retrieval/`` for that reason: ``retrieval/`` is for the strategies actually being compared.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from uuid import UUID

from src.eval.dataset import Question
from src.eval.harness import Retriever


def random_retriever(paragraph_ids: Sequence[UUID], k: int = 10, seed: int = 42) -> Retriever:
    """Return ``k`` paragraphs chosen at random, ignoring the question entirely.

    Seeded **per question**, from the seed and the question id, so the result is identical no
    matter what order questions are asked in or how many are asked. Seeding once for the whole run
    would make the floor depend on iteration order, and a baseline that moves is not a baseline.
    """
    if k > len(paragraph_ids):
        raise ValueError(f"cannot draw {k} from a corpus of {len(paragraph_ids)}")

    def retrieve(question: Question) -> Sequence[UUID]:
        rng = random.Random(f"{seed}:{question.id}")
        return rng.sample(list(paragraph_ids), k)

    return retrieve
