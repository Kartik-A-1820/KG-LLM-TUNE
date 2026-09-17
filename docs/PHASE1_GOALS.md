# Phase 1 Goals and Stage Gates

Phase 1 succeeds when a sub-1B model, running locally, can build a knowledge graph good enough that answers over it are within 10% of answers over a teacher-built graph — at 5× the indexing throughput.

Everything below is how that gets measured, and where it stops.

## The measuring stick

**All gate metrics are measured against a hand-annotated gold set drawn from Kartik's own corpus** (200–500 examples — see `docs/DATA_STRATEGY.md`). Not against a public benchmark, not against teacher output, not against a held-out slice of the training data.

This matters more than any single threshold below. Public-benchmark F1 tells you how the model does on Wikipedia-style text with a fixed entity schema. It tells you very little about how it does on the documents this pipeline will actually index. A model that scores well on DocRED and badly on the gold set is a model that fails in production and looks fine in the metrics.

The gold set is versioned and hashed. Every metric records which version it was measured against. A number measured against gold-set v1 is not comparable to one measured against v2, and mixing them is a §5 violation of `AGENTS.md`.

## What we do *not* measure

**Schema validity / parse rate.** Constrained decoding (XGrammar or Outlines) makes output 100% parseable by construction. Reporting parse rate as a result is reporting that the constraint library works. It is not a model metric and it must not appear in any gate.

What replaces it is **semantic value accuracy**: given that the output parses, are the values right? Is the entity string the right string? Does the relation point at the right two entities? Is the type correct? That is the question constrained decoding does not answer for you.

---

## Gate 0 — Baseline, before any training

**Purpose:** establish that fine-tuning has headroom worth spending weeks on. This is the cheapest gate and the one most often skipped; skipping it is how a project spends a month recovering a gap that did not exist.

**Do:** measure all of these on the gold set, with the same prompts and the same constrained decoder:

| System | Role |
| --- | --- |
| Stock SmolLM2-360M-Instruct | **Kartik's chosen primary** — the starting point |
| Stock Qwen3-0.6B (non-thinking) | benchmark comparison, not co-primary |
| A 7B-class instruct model | the "just use a bigger local model" alternative |
| The teacher model | the ceiling, and the thing being distilled |

Report the same metric set as Gate 1 for each. Measure each of the two small models **both with and without few-shot demonstrations** — see the recommendation below for why that column matters.

### Recommended: run both small models through a pilot fine-tune, not just a stock baseline

**This is cheap de-risking, and it is worth doing before anything long is committed. It is diagnostic; it does not produce official gate numbers.**

SmolLM2-360M is Kartik's chosen primary. The prior feasibility assessment says it costs roughly 9 F1 points versus a 0.5B — but that evidence is about *stock* models on *other people's* data. The pilot does not decide whether Qwen becomes co-primary; it validates and de-risks the SmolLM2-first plan by asking a narrower question:

> **Does the 9-point gap survive fine-tuning on this corpus?**

Nothing in the literature answers that, because the literature did not fine-tune on this corpus.

**The experiment:** local LoRA on a 1–5k example subset, same data, same prompts, same eval — **SmolLM2-360M and Qwen3-0.6B side by side**. A few hours on the local card. Cost: an afternoon. Outcome: evidence about whether the SmolLM2-first choice is holding, not permission to quietly reverse the model priority.

**Why it is worth the afternoon, whichever way it lands:**

- **If 360M clears the pilot bar**, that is not a consolation result — it is a genuinely better outcome than the plan assumed. A smaller, faster model that looks viable under the same local pilot means lower latency, more headroom, and a cheaper Phase 2. Validated rather than hoped for.
- **If it does not**, that is known in **week one instead of week four**, before a full Kaggle training budget and a teacher-distillation spend have gone into the wrong base model.

The asymmetry is the whole argument: a few hours now against several weeks of potential rework. Run it.

### Two SmolLM2-360M-specific risks to carry forward

Neither is a reason to change the choice. Both are planning constraints that must show up in the design:

**1. It may still need few-shot prompting.** The prior feasibility assessment records SmolLM2-360M at **0.527 F1 without few-shot** versus **0.735 with 2-shot** (external figures, not measured here). That is a large dependence on demonstrations. If the production prompt has to carry them, those tokens enter every chunk's context — which **erodes the throughput advantage that motivated choosing the smaller model in the first place**.

So the Gate 1 throughput number must be measured with **the prompt that will actually ship**, demonstrations included. A throughput figure measured on a zero-shot prompt that production will not use is not a real number. Whether fine-tuning removes the few-shot dependence is itself a finding worth recording — that is part of what the pilot above answers.

**2. 8k context is a hard planning constraint.** A ~1,200-token chunk plus 2-shot demonstrations plus the schema fits in 8k. It does not leave headroom for:

- wider chunks
- more than about two demonstrations
- multi-round gleaning passes over the same chunk

Treat 8k as a fixed budget that every prompt design must fit inside. Any Phase 2 design wanting larger chunks or gleanings has to either fit the budget or change model — and discovering that at Phase 2 is the expensive ordering.

**Proceed only if fine-tuning has visible headroom** — that is, the gap between the stock primary and the teacher is large enough that closing most of it is worth the effort, *and* the 7B does not already solve the problem at acceptable local speed.

**Stop and rethink if:**

- The stock primary is already close to the teacher on this corpus → the corpus is easier than assumed; skip fine-tuning, go to Phase 2 with prompting alone.
- The teacher itself scores poorly on the gold set → the task definition or entity schema is the problem, not the model. Fix the schema before training anything.
- The 7B matches the teacher at acceptable throughput on the local box → the project's premise (needing a sub-1B) needs re-examination.

**Deliverable:** `runs/gate0-baseline/metrics.json` with one block per system (each with and without demonstrations), plus diagnostic pilot run directories such as `runs/pilot-lora-360m/` and `runs/pilot-lora-qwen06b/` from the side-by-side pilot, and a short written read of where the headroom is and whether the SmolLM2-first choice is holding.

---

## Gate 1 — SFT

**Purpose:** confirm supervised fine-tuning moved the model into usable range.

| Metric | Target | Hard stop |
| --- | --- | --- |
| Entity recall | ≥ 0.85 | < 0.70 |
| Entity F1 | ≥ 0.80 | < 0.65 |
| Strict relation-triple F1 | ≥ 0.60 | < 0.45 |
| Description faithfulness | ≥ 0.95 | < 0.90 |
| Batched throughput | ≥ 500 tok/s | < 300 tok/s |

**Definitions — fix these before measuring, not after:**

- **Entity recall is weighted above precision on purpose.** A missed entity is a node that can never be retrieved; a spurious entity is a node nothing links to, which is cheap. Recall has the higher floor for that reason.
- **Strict relation-triple F1** means all three of (head, relation type, tail) match. No partial credit. Strict is the honest measure because a triple with the wrong tail is not 2/3 correct — it is wrong, and it will be traversed.
- **Description faithfulness** is the share of generated descriptions that are supported by the source chunk. Operationalise it with the programmatic overlap check first (see `docs/DATA_STRATEGY.md`); an LLM judge is a secondary signal, never the primary number, because the judge is the same class of system being judged.
- **Throughput** is measured batched, on the local 1650 Ti, with constrained decoding on, at the batch size Phase 2 will actually use. Unbatched or unconstrained numbers are not comparable and must not be reported here.

**Between target and stop** is the fix-it band: more data, better data, different LoRA rank / full-FT choice, prompt revision. It is not a pass.

**Below any hard stop:** do not proceed to rejection sampling. Rejection sampling amplifies what SFT learned; it cannot install a capability that is not there. Go back to data.

---

## Gate 2 — Rejection-sampling self-distillation

**Purpose:** confirm the self-distillation round is adding signal rather than reinforcing the model's existing errors.

**Pass requires all three:**

1. Entity F1 improves by **≥3 points** over the Gate 1 model
2. Description faithfulness **improves** (not merely holds)
3. Entity recall is **not lost** — a precision-driven F1 gain that costs recall is a regression for this pipeline, per the Gate 1 reasoning

**Hard stop — the verifier pass rate:** if **fewer than 30% of sampled completions pass the verifier**, stop. Do not raise the sampling temperature, do not loosen the verifier, do not sample more.

A sub-30% pass rate means the base SFT model is not producing enough correct output for self-distillation to have material to work with. Sampling harder just collects a larger set of the same mistakes, and self-distilling on it bakes those mistakes in. **The correct response is to go back for more SFT data — backward, not forward.** This is the gate most likely to be rationalised past, so it is written as an absolute.

**Also worth watching:** verifier pass rate by task. If entity extraction passes at 70% and claim extraction at 12%, the answer is not a global decision — it is that claims need their own data.

---

## Gate 3 — Downstream utility

**This is the gate that matters. The others are leading indicators.**

**Purpose:** answer the only question that counts — does a graph built by the small model support answers as good as a graph built by the teacher?

**Protocol:**

1. Take a **held-out slice** of the corpus — not the gold set, not the training source.
2. Build **two graphs** over the same slice: one with the fine-tuned small model, one with the teacher. Identical chunking, identical prompts, identical everything downstream. The only variable is the extraction model.
3. Run the **same QA set** through both, with the same synthesis model on top.
4. Compare answer quality, and measure indexing wall-clock throughput for both.

**Outcomes:**

| Result | Decision |
| --- | --- |
| ≥ 90% of teacher-graph answer quality **and** ≥ 5× indexing throughput | **Ship.** Phase 1 complete. |
| 75–90% of teacher quality | **Selective escalation.** Route hard or high-value chunks to the teacher, keep the small model for the bulk. Define the routing rule from the error analysis, not by guessing. |
| < 75% | **Stop.** The small model is not carrying the extraction path. Re-scope. |

Both conditions are required for a ship. 95% of teacher quality at 2× throughput is not a win — the entire premise is that local indexing becomes cheap.

## Gate 1 can pass while Gate 3 fails

Expect this. Plan for it. It is not a sign that something went wrong.

Extraction F1 and retrieval utility are different quantities, and they come apart in specific, predictable ways:

- **F1 weights all entities equally; retrieval does not.** Missing ten rare entities that nobody queries costs ten F1 points and zero answers. Missing one hub entity costs one F1 point and breaks every multi-hop path through it.
- **Graph structure is not in the F1.** A graph can have excellent per-chunk triples and still fragment into disconnected components because entity *resolution* across chunks failed — the same entity written two ways becomes two nodes. Per-chunk F1 cannot see this at all.
- **Descriptions feed embeddings and community reports.** A faithful-but-thin description scores fine on faithfulness and retrieves badly.
- **Errors correlate.** The model's mistakes are systematic, not random, so they cluster on the same document types — which means they concentrate in the same region of the graph rather than averaging out.

If Gate 1 passes and Gate 3 fails, the diagnosis is not "the model is bad." It is: *which* of the above is it? Run the error analysis on the failing QA items, trace them back to the graph, and find whether the loss is in missing hubs, fragmentation, or description quality. Each has a different fix, and only one of them is "more SFT data."

## Where runs happen: local LoRA vs Kaggle full fine-tune

Gates are measured on Kaggle-trained full fine-tunes. But the local GPU is a training device too, and using it well is what keeps the gate cycle short.

**Memory reality on the local 1650 Ti (3.4 GB):**

| Setup | VRAM | Fits? |
| --- | --- | --- |
| LoRA on a 360M model | ~1.0–1.5 GB | yes |
| LoRA on Qwen3-0.6B | ~2.0–3.0 GB | yes |
| Full fine-tune of Qwen3-0.6B | ~6.5 GB | **no** |
| Full fine-tune of SmolLM2-360M | not measured | unknown — measure before assuming |

The constraint is not memory alone. It is **wall-clock** and **LoRA vs full fine-tune**.

**Rule of thumb:**

| Run | Where |
| --- | --- |
| Overfit-a-tiny-batch check (20 examples, loss must collapse) | Local LoRA |
| Pipeline debugging | Local LoRA |
| **Checkpoint-and-resume test** (kill a short run, resume it) | Local LoRA |
| Hyperparameter sanity checks | Local LoRA |
| Pilot on a 1–5k subset, end to end | Local LoRA |
| Full SFT on the real dataset | Kaggle, full FT |
| Rejection sampling at volume | Kaggle |
| **Any run producing a gate number** | **Kaggle, full FT** |

A 4–8 hour T4 run becomes **weeks** locally at an estimated 25–50× slowdown vs a 4090. The 360M full-FT footprint still has to be measured; until a committed memory test proves local full FT is viable, real runs go to Kaggle.

Getting the loop right locally before spending a Kaggle session is good practice. A session burned on a bug a 3-minute local run would have caught is the expensive mistake.

**Real runs are full fine-tune because this is task shift.** The LoRA-Learns-Less evidence favours full FT when the model is learning a new output format and a new behaviour, which is exactly this. LoRA is the iteration tool, not the shipping recipe — so do not report a local LoRA number as a gate result.

**Two local gotchas that will bite:**

1. **Context length, not parameters, is what OOMs the local card.** SM 7.5 has no FlashAttention-2 support. Naive attention at sequence length 4096 costs roughly **537 MB per layer**. GraphRAG extraction prompts are long — ~1,200-token chunks plus few-shot plus schema — so this is the binding constraint, not the adapter size. **Require PyTorch SDPA's memory-efficient backend or xformers explicitly, and assert it is active.** A silent fallback to the naive path presents as an OOM that looks like "model too big" and wastes a day.
2. **QLoRA buys almost nothing here.** 4-bit saves ~540 MB on a 0.72 GB base and costs dequantisation overhead every forward pass. Prefer **plain LoRA** locally; QLoRA earns its keep at 7B+, where base weights dominate the budget.

## Non-goals for Phase 1

- Community report generation, global-search reduce, final answer synthesis — these stay with the larger model by design.
- Any RLHF/DPO/GRPO stage.
- Any Phase 2 pipeline component beyond what is needed to run Gate 3's two-graph comparison.
- Serving infrastructure, API, or UI.
