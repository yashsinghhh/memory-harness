# memory-harness

A memory layer, built in increments: ingest a corpus, extract entities and relations into a knowledge
graph and a vector index, retrieve over both — and measure each capability so the next increment can be
shown to beat the last.

**Status: early.** Toolchain in place; the pipeline is being built increment by increment. Numbers land
in the eval log as each capability arrives. This README grows with them.

---

## Resources

The papers and sources this project is built on. Anything cited here was read, not recalled.

### Benchmarks and evaluation

- **2WikiMultihopQA** — Ho, Duong Nguyen, Sugawara, Aizawa (2020), *Constructing A Multi-hop QA Dataset
  for Comprehensive Evaluation of Reasoning Steps*, COLING 2020.
  [arXiv:2011.01060](https://arxiv.org/abs/2011.01060) ·
  [dataset](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)
  Supplies the corpus, the questions, gold supporting-paragraph labels, and gold `(subject, relation,
  object)` triples. 167,454 train / 12,576 dev / 12,576 test; 10 paragraphs per instance in the
  distractor setting; four question types — comparison, inference, compositional, bridge-comparison.

- **MuSiQue** — Trivedi, Balasubramanian, Khot, Sabharwal, *MuSiQue: Multihop Questions via Single-hop
  Question Composition*, TACL.
  [TACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996/MuSiQue-Multihop-Questions-via-Single-hop-Question)
  Argues earlier multi-hop datasets contain shortcuts that let a model answer without genuinely
  multi-hopping. Relevant as a caveat on any multi-hop result measured here.

### Long context vs. memory

- **Lost in the Middle** — Liu et al. (2023).
  [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
  Accuracy is U-shaped by position in the context window. Worth pairing with
  [Databricks' long-context RAG work](https://www.databricks.com/blog/long-context-rag-performance-llms),
  which finds the effect is weakening in newer models — so the case for a memory layer rests on
  *persistence being orthogonal to capacity*, not on long context retrieving badly.

---

## Development

```bash
uv sync          # Python 3.12, pinned
uv run pytest
uv run ruff check .
uv run pyright
```
