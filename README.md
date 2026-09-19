# KG-LLM-TUNE

**Fine-tune a sub-1B model to own the extraction path of a GraphRAG pipeline, so that indexing runs locally at speed instead of costing a large-model API call per chunk.**

That is the whole point of this repo. Everything below is detail.

## The split: extraction vs synthesis

A GraphRAG pipeline makes many LLM calls. They are not the same kind of call.

**Owned by the fine-tuned small model (this project):**

- entity extraction
- relation extraction
- entity descriptions and relation descriptions
- claim / covariate extraction
- reference-grounded question answering over a supplied chunk or retrieved context
- structured JSON output under the project schema
- source-grounded chunk/entity/relation summaries used by extraction and indexing
- query routing (local vs global vs direct)

**Stays with a larger model:**

- community report generation
- global-search map-reduce
- final answer synthesis

The dividing line is **extraction vs synthesis**, not index-time vs query-time. Query routing is index-agnostic but it is a short classification, so it comes to the small model. Community reports are index-time but they are long-form abstractive writing, so they stay with the big one.

This matters because extraction is the high-volume call — it is per-chunk, it is structured, and it is verifiable. Those three properties are exactly what makes it a good fine-tuning target and a bad place to spend API budget.

Training data must follow that boundary. Use datasets that exercise the GraphRAG flow directly: entities, relationships, claims, source-grounded descriptions/summaries, structured output enforcement, routing labels, and reference-grounded QA where the answer must be supported by the provided context. Generic public relation-extraction data is only a smoke-test and format-bootstrap tool; it is not the main training mixture.

## Two phases

**Phase 1 — the models.** Produce a fine-tuned extraction model and a selected embedding model, both validated against a hand-annotated gold set from Kartik's own corpus. Phase 1 is what this repo currently contains. It ends when the Gate 3 downstream check in [`docs/PHASE1_GOALS.md`](docs/PHASE1_GOALS.md) passes.

**Phase 2 — the pipeline.** Build the full GraphRAG architecture running locally on Phase 1's models: chunking, graph construction, community detection, embeddings, hybrid local/global search. Phase 2 does not start until Phase 1 has a packaged, gated model. Nothing in this repo commits to a Phase 2 design yet.

## Decisions already made

These came out of a completed feasibility assessment plus Kartik's explicit model-priority decision. They are settled; reopen them only with new evidence, not new opinion.

| Decision | Choice | Why |
| --- | --- | --- |
| **Model (primary)** | **SmolLM2-360M-Instruct** | Kartik's explicit primary choice: choose the smaller, faster model first, with the ~9 F1 gap from the prior feasibility assessment accepted as a known, managed risk |
| Benchmark comparison | Qwen3-0.6B, non-thinking mode (`enable_thinking=False`) | Run at Gate 0 alongside the 360M on the same pilot subset, then later as a full comparison |
| Fallback | Qwen2.5-0.5B-Instruct | De-risked — prior feasibility assessment cites published extraction F1 of 0.828 on this task class |
| Embedding | EmbeddingGemma-300M primary, potion-retrieval-32M as a serious A/B | potion is ~200× faster on CPU; graph traversal may carry enough retrieval load that the quality gap costs nothing measurable |
| Training | SFT → rejection-sampling self-distillation → constrained decoding at inference | Tasks are verifiable, so a verifier plus rejection sampling beats preference optimisation |
| Dropped | DPO, RLHF, GRPO | Verifiable tasks don't need preference optimisation; GRPO's ~5 GB floor does not fit 3.4 GB of VRAM anyway |
| Real training runs | Kaggle T4 (16 GB), full fine-tune | Gate numbers come from the shipping full-FT recipe. The 360M full-FT footprint is not measured yet; Kaggle stays the default until a committed memory test proves otherwise |
| Iteration training runs | **Local 1650 Ti, plain LoRA** | LoRA on 360M is expected around 1.0–1.5 GB; LoRA on 0.6B is ~2.0–3.0 GB. Pilots, debugging, resume tests, HP sanity checks belong here |
| Local hardware role | LoRA iteration, inference, evaluation, integration | Not a full-FT box, but genuinely a training box |
| Constrained decoding | XGrammar or Outlines | Guarantees parseable output |

Two consequences of that last row are load-bearing and appear throughout the docs:

1. **Schema validity becomes 100% by construction, so it measures nothing.** Never report parse rate as a result. Gate on semantic value accuracy — are the entity strings and relation endpoints *right*, not merely well-formed.
2. **T4 is SM 7.5**, so fp16 + `GradScaler`. No bf16. Any config or notebook that assumes bf16 is wrong for this project. (The local 1650 Ti is SM 7.5 too.)

And one local-training consequence worth knowing before the first run: on SM 7.5 there is **no FlashAttention-2**, so naive attention at sequence length 4096 costs roughly **537 MB per layer**. Since extraction prompts are long, *context length — not parameter count — is what will OOM the local card*. Local training must explicitly use PyTorch SDPA's memory-efficient backend or xformers. Prefer plain LoRA over QLoRA locally: 4-bit saves only ~540 MB on a 0.72 GB base and costs dequantisation overhead.

### Two risks that come with the 360M choice

Accepted, not ignored. Both are planning constraints:

1. **It may still need few-shot prompting in production.** The prior feasibility assessment records SmolLM2-360M at **0.527 F1 without few-shot** versus **0.735 with 2-shot** (external figures, not measured here). If the production prompt has to carry demonstrations, those tokens go into every chunk's context — which erodes the throughput advantage that motivated picking the smaller model. Measure the with- and without-demonstration throughput, not just the quality.
2. **8k context is a hard planning constraint.** It fits a ~1,200-token chunk plus 2-shot demonstrations, but leaves no headroom for wider chunks, more shots, or gleaning passes. Any design that wants larger chunks or multi-round gleaning has to fit inside 8k or change model.

Gate 0 includes **both** SmolLM2-360M and Qwen3-0.6B in the baseline, and the early local LoRA pilot runs them side by side on the same subset. That pilot validates and de-risks Kartik's primary-model choice; it does not make Qwen co-primary, and it is diagnostic rather than a gate number. See [`docs/PHASE1_GOALS.md`](docs/PHASE1_GOALS.md).

## Top risk

**Description hallucination poisoning the graph.** A wrong entity is one wrong node. A fabricated description is a plausible-sounding lie that propagates into community reports, into embeddings, and into every answer that touches that node — and it looks fine on inspection.

Mitigation is mandatory, not optional:

- train descriptions as **spans or near-spans of the source chunk**, not as free generation
- enforce **source-overlap at decode time**, so a description that drifts from the chunk cannot be emitted
- measure **description faithfulness** as a first-class gate metric (Gate 1 threshold ≥0.95, stop below 0.90)

## Status

Phase 1, pre-Gate-0. Scaffold plus local smoke plumbing.

Done: repo scaffold, local venv on `D:`, DocRED open-data format-bootstrap pull, one tiny SmolLM2-360M local LoRA smoke run, one tiny QLoRA rank sweep over r=8, r=16, and r=32, and one 5k-example QLoRA rank sweep over r=8, r=16, and r=32. The smoke run shows local CUDA training and loss movement on a 16-train / 4-val diagnostic subset; the recorded values live in [`runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json`](runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json) and [`runs/20260918-080000-smollm2-docred-lora-smoke/env.json`](runs/20260918-080000-smollm2-docred-lora-smoke/env.json). The 5k QLoRA sweep selected r=16 for the next local GraphRAG-specific pilot under this config, based on [`runs/20260919-003400-smollm2-docred5k-qlora-r16/metrics.json`](runs/20260919-003400-smollm2-docred5k-qlora-r16/metrics.json); r=32 is not selected because [`runs/20260919-042300-smollm2-docred5k-qlora-r32/metrics.json`](runs/20260919-042300-smollm2-docred5k-qlora-r32/metrics.json) records `train_loss_finite` false. The QLoRA rank sweeps are indexed in [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md).

Not done: no gold set, no Gate 0 baseline, no Qwen3 side-by-side pilot, no GraphRAG-specific SFT mixture, no Kaggle full fine-tune, and no gate metric.

The immediate critical path is the **gold set** — 200–500 hand-annotated examples from Kartik's own corpus. Everything else in Phase 1 is measured against it, so nothing downstream can start until it exists. See [`docs/DATA_STRATEGY.md`](docs/DATA_STRATEGY.md).

Known blockers are tracked in [`docs/BLOCKERS.md`](docs/BLOCKERS.md) rather than left implicit. Two of them are policy questions only Kartik can answer.

Benchmark process and committed run evidence are tracked in [`docs/BENCHMARKING_PROTOCOL.md`](docs/BENCHMARKING_PROTOCOL.md) and [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md).

## Getting started

```bash
git clone https://github.com/Kartik-A-1820/KG-LLM-TUNE.git
cd KG-LLM-TUNE
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e .                                   # pyproject not written yet
```

Then read, in this order:

1. [`docs/PHASE1_GOALS.md`](docs/PHASE1_GOALS.md) — what "done" means, with numeric stop conditions
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit and what runs where
3. [`docs/DATA_STRATEGY.md`](docs/DATA_STRATEGY.md) — where training data comes from and what it costs
4. [`AGENTS.md`](AGENTS.md) — the rules for working in this repo, human or agent

[`HANDOFF_PROMPT.md`](HANDOFF_PROMPT.md) is a self-contained briefing to paste when delegating work to another agent with no context on the project. Keep it current as decisions change.

## Layout

```
src/kg_llm_tune/   library code — anything reused lives here, not in a notebook
notebooks/         exploration and Kaggle training notebooks only
configs/           YAML run configs; every run is driven by one
data/              gitignored; see data/README.md for expected layout
eval/              gold set and evaluation harness — the ground truth of this project
scripts/           thin CLI entry points over src/
docs/              planning and design documents
```

## Licence

Not yet chosen. Note that some candidate training datasets are CC-BY-SA-4.0, which has share-alike implications for derived data — see [`docs/DATA_STRATEGY.md`](docs/DATA_STRATEGY.md) before picking one.
