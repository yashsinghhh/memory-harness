"""Command line interface.

The CLI is the only surface this project has — no REST, no server, no UI. ``eval`` is a first-class
command rather than a script, because the discipline is to run it on every change that could move a
number, and a command gets run where a script gets forgotten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.eval.baselines import random_retriever
from src.eval.dataset import load
from src.eval.harness import run

app = typer.Typer(add_completion=False, help="A memory layer, built in increments.")

DEFAULT_DATA = Path("data/2wikimultihop_dev.json")


@app.callback()
def main() -> None:
    """Keeps subcommands addressable by name.

    Typer collapses a single-command app into a bare root command, so ``eval`` would only be
    reachable as the whole program. This holds the ``<tool> <command>`` shape steady while the other
    commands are still being built.
    """


@app.command("eval")
def eval_command(
    data: Annotated[Path, typer.Option(help="Benchmark file.")] = DEFAULT_DATA,
    instances: Annotated[int, typer.Option(help="Questions to evaluate.")] = 100,
    seed: Annotated[
        int, typer.Option(help="Sampling seed. Changing this re-baselines everything.")
    ] = 42,
    k: Annotated[int, typer.Option(help="Results each retriever returns.")] = 10,
    holdout: Annotated[
        bool,
        typer.Option("--holdout", help="Score the held-out set. Opened ONCE, at the end."),
    ] = False,
) -> None:
    """Score retrievers against the question set."""
    if not data.exists():
        typer.echo(f"error: {data} not found. Download the benchmark first.", err=True)
        raise typer.Exit(1)

    if holdout:
        typer.echo("!! holdout set — every number from it is single-use. !!\n")

    evalset = load(data, instances=instances, seed=seed, holdout=holdout)
    typer.echo(
        f"corpus     {len(evalset.paragraphs)} paragraphs "
        f"({evalset.paragraph_slots} slots, {evalset.dedup_ratio:.2f}x dedup)"
    )
    typer.echo(f"questions  {len(evalset.questions)}\n")

    paragraph_ids = [p.id for p in evalset.paragraphs]
    report = run(
        evalset.questions,
        random_retriever(paragraph_ids, k=k, seed=seed),
        name=f"random baseline (k={k}, seed={seed})",
    )
    typer.echo(report.format())


if __name__ == "__main__":
    app()
