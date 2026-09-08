"""Increment 1 — the 2Wiki loader.

Two tiers. Logic tests run on a synthetic fixture and always execute. Integration tests need the
real 53 MB dev split, which is gitignored, so they skip cleanly when it isn't there.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from src.eval.dataset import (
    DEV_SPLIT_SIZE,
    EvalSet,
    build_eval_set,
    identity_key,
    load,
    paragraph_id,
)

DATA = Path("data/2wikimultihop_dev.json")
needs_data = pytest.mark.skipif(
    not DATA.exists(), reason="run: curl -sL -o data/2wikimultihop_dev.json <hf-url>"
)


def _instance(iid: str, titles: list[str], gold: list[str]) -> dict[str, Any]:
    return {
        "_id": iid,
        "question": f"q-{iid}",
        "answer": f"a-{iid}",
        "type": "compositional",
        "context": [[t, [f"{t} sentence one.", f"{t} sentence two."]] for t in titles],
        "supporting_facts": [[t, 0] for t in gold],
        "evidences": [[gold[0], "related_to", gold[-1]]] if gold else [],
    }


# --- identity ---------------------------------------------------------------------------------


def testidentity_key_ignores_tokenisation_artifacts() -> None:
    """The exact failure found in the dev split: 1,725 titles differ only like this."""
    assert identity_key("Anhalt- Zerbst ( 17 March 1540)") == identity_key(
        "Anhalt-Zerbst (17 March 1540)"
    )


def testidentity_key_still_separates_different_text() -> None:
    assert identity_key("Bernhard VII") != identity_key("Bernhard VIII")


def test_paragraph_id_is_stable_across_processes() -> None:
    """Hard-coded on purpose. If this ever changes, every stored id is invalidated."""
    assert str(paragraph_id("Inception was directed by Nolan.")) == str(
        uuid.uuid5(uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), "inceptionwasdirectedbynolan")
    )


# --- build logic ------------------------------------------------------------------------------


def test_shared_paragraphs_deduplicate() -> None:
    built = build_eval_set([_instance("a", ["X", "Y"], ["X"]), _instance("b", ["X", "Z"], ["Z"])])
    assert len(built.paragraphs) == 3  # X, Y, Z — not 4
    assert len(built.questions) == 2


def test_gold_is_resolved_to_ids_not_titles() -> None:
    built = build_eval_set([_instance("a", ["X", "Y"], ["Y"])])
    (q,) = built.questions
    (gold,) = q.gold_paragraph_ids
    assert built.paragraph(gold).title == "Y"


def test_gold_triples_are_carried() -> None:
    built = build_eval_set([_instance("a", ["X", "Y"], ["X", "Y"])])
    assert built.questions[0].gold_triples == (("X", "related_to", "Y"),)


def test_unresolvable_gold_raises_rather_than_silently_dropping() -> None:
    """A silently-dropped gold paragraph caps recall below 1.0 with nothing to show why."""
    broken = _instance("a", ["X"], [])
    broken["supporting_facts"] = []
    with pytest.raises(ValueError, match="no gold paragraphs"):
        build_eval_set([broken])


# --- against the real split -------------------------------------------------------------------


@needs_data
def test_split_size_matches_the_published_figure() -> None:
    assert len(json.loads(DATA.read_text())) == DEV_SPLIT_SIZE


@needs_data
def test_same_seed_gives_an_identical_slice() -> None:
    a, b = load(DATA, instances=20), load(DATA, instances=20)
    assert [q.id for q in a.questions] == [q.id for q in b.questions]


@needs_data
def test_a_different_seed_gives_a_different_slice() -> None:
    """Guards against seeding that silently does nothing."""
    a, b = load(DATA, instances=20, seed=42), load(DATA, instances=20, seed=7)
    assert [q.id for q in a.questions] != [q.id for q in b.questions]


@needs_data
def test_holdout_shares_no_instances_with_the_working_set() -> None:
    """Both are drawn from one sample and split, precisely so this holds."""
    work = load(DATA, instances=100)
    held = load(DATA, instances=100, holdout=True)
    assert {q.id for q in work.questions} & {q.id for q in held.questions} == set()


@needs_data
def test_every_gold_id_is_present_in_the_corpus() -> None:
    """The one that matters. A gold paragraph missing from the corpus caps recall invisibly."""
    ev = load(DATA, instances=100)
    corpus = {p.id for p in ev.paragraphs}
    for q in ev.questions:
        assert q.gold_paragraph_ids <= corpus, q.id


@needs_data
def test_every_question_has_gold_and_the_counts_are_as_documented() -> None:
    ev = load(DATA, instances=100)
    sizes = {len(q.gold_paragraph_ids) for q in ev.questions}
    assert sizes <= {2, 4}, sizes  # 2 normally, 4 for bridge_comparison
    assert all(q.gold_paragraph_ids for q in ev.questions)


@needs_data
def test_deduplication_actually_fires() -> None:
    ev: EvalSet = load(DATA, instances=100)
    assert len(ev.paragraphs) < ev.paragraph_slots
    assert ev.dedup_ratio > 1.0


@needs_data
def test_a_truncated_file_is_rejected(tmp_path: Path) -> None:
    short = tmp_path / "short.json"
    short.write_text(json.dumps(json.loads(DATA.read_text())[:10]))
    with pytest.raises(ValueError, match="truncated"):
        load(short, instances=2)
