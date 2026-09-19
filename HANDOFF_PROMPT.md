# Handoff Prompt

A self-contained briefing for delegating work on this project to another agent (Codex, or any assistant with zero prior context).

**How to use:** copy everything below the horizontal rule, to the end of this file, and paste it as the first message. It assumes no knowledge of the project.

**Keep it current.** When a decision changes, a gate is passed, or a blocker closes, update this file in the same commit. A stale handoff prompt is worse than none — it briefs an agent into acting on decisions that were already reversed.

---

You are working on **KG-LLM-TUNE**. Read this entire brief before doing anything. You have no prior context on this project; everything you need is here.

## 1. Identity and locations

The project exists in two places, and **they must stay in sync**:

- **Local:** `D:\KG-LLM-TUNE\` (Windows machine; moved from `F:\KG-LLM-TUNE\` on 2026-09-17 for space)
- **Remote:** `https://github.com/Kartik-A-1820/KG-LLM-TUNE` (public)

Pull before you start. Push when you finish. If the two have diverged, stop and report the divergence rather than resolving it by overwriting one side.

Owner: Kartik (GitHub `Kartik-A-1820`).

## 2. The goal

**Fine-tune a sub-1B model to own the *extraction* path of a GraphRAG pipeline, so that indexing runs locally at speed instead of costing a large-model API call per chunk.**

Success is defined downstream, not by extraction F1: a graph built by the small model must support answers within 10% of a teacher-built graph, at ≥5× indexing throughput.

## 3. The extraction / synthesis split

The dividing line is **extraction vs synthesis** — *not* index-time vs query-time. Do not restate it as index vs query; that is a different and wrong split.

**The fine-tuned small model owns:**

- entity extraction
- relation extraction
- entity descriptions and relation descriptions
- claim / covariate extraction
- reference-grounded question answering over a supplied chunk or retrieved context
- structured JSON output under the project schema
- source-grounded chunk/entity/relation summaries used by extraction and indexing
- query routing

**A larger model keeps:**

- community report generation
- global-search map-reduce
- final answer synthesis

Why the line falls there: extraction is high-volume, schema-structured, and verifiable. Synthesis is low-volume, free-form, and unverifiable — so there is no verifier, no rejection sampling, and a much worse feedback loop for fine-tuning it. Query routing joins the small model despite being query-time because it is a short classification with a fixed label set.

Training data follows this split. Do **not** train on broad available datasets merely because they exist. The SFT mixture should be built from tasks the GraphRAG extraction/indexing path actually needs: entity and relationship extraction, claims/covariates, source-grounded descriptions and summaries, strict structured output, routing labels, and reference-grounded QA where every answer is supported by the provided text. Generic public RE data such as DocRED is for format bootstrapping and smoke tests only.

## 4. Two phases

- **Phase 1 (current):** produce a fine-tuned extraction model and a selected embedding model, validated against a hand-annotated gold set. This is all the repo contains.
- **Phase 2 (not started):** the full GraphRAG pipeline running locally on Phase 1's models. Do not design or build Phase 2 components except the minimum needed for the Gate 3 comparison.

## 5. Decided technical choices — do not change these without flagging

These came out of a completed feasibility assessment plus Kartik's explicit model-priority decision. They are settled. If you believe one is wrong, **say so explicitly and wait** — do not quietly substitute an alternative.

**Model (primary): SmolLM2-360M-Instruct.** This is Kartik's explicit primary choice. It was chosen deliberately, with full knowledge that the prior feasibility assessment puts it ~9 F1 points behind a 0.5B. The size and speed are worth the risk to Kartik; the risk is managed, not ignored. Do not substitute a different base model.

**Benchmark comparison, not co-primary:** Qwen3-0.6B in non-thinking mode (`enable_thinking=False`) — thinking tokens are pure cost on a structured task. The early local LoRA pilot runs both models side by side on the same 1–5k subset to validate and de-risk Kartik's primary-model choice: does the 9-point external gap survive fine-tuning *on this corpus*? This pilot is diagnostic, not a gate number and not a model-priority vote.

**Fallback:** Qwen2.5-0.5B-Instruct — the prior feasibility assessment cites published extraction F1 of 0.828 on this task class (external figure, not measured here).

**Two 360M-specific risks that are planning constraints, not footnotes:**

1. **Few-shot dependence.** The prior feasibility assessment records 0.527 F1 zero-shot vs 0.735 with 2-shot (external figures, not measured here). If demonstrations must stay in the production prompt, their tokens enter every chunk's context and **erode the throughput advantage that motivated the smaller model**. Gate 1 throughput must therefore be measured with the prompt that actually ships, demonstrations included.
2. **8,192 tokens is the native context budget.** Hugging Face `AutoConfig` for `HuggingFaceTB/SmolLM2-360M-Instruct` reports `max_position_embeddings` 8192. A ~1,200-token chunk plus 2-shot plus schema fits. There is no headroom for wider chunks, more shots, or gleaning passes. Every prompt design must fit inside 8192, or the model choice changes. GraphRAG capability pilots should target the full native context for extraction, reference-grounded QA, and context-aware summaries; shorter local runs are smoke-only unless explicitly labelled otherwise.

**Embedding:** EmbeddingGemma-300M primary, with **potion-retrieval-32M as a serious A/B**, not a token alternative. potion is ~200× faster on CPU, and the hypothesis is that graph traversal carries enough of GraphRAG's retrieval load that the quality gap may not show up end to end. A/B is decided on end-to-end answer quality, not embedding-leaderboard scores.

**Training pipeline:** SFT → rejection-sampling self-distillation → constrained decoding at inference.

**DPO and RLHF are dropped, deliberately.** The tasks are verifiable, so a verifier plus rejection sampling beats preference optimisation. GRPO additionally has a ~5 GB floor that does not fit the 3.4 GB local card. Do not reintroduce a preference-optimisation stage.

**Constrained decoding:** XGrammar or Outlines. Two consequences that must be respected everywhere:

1. **Schema validity becomes 100% by construction, so it measures nothing.** Never report parse rate as a result. Gate on **semantic value accuracy** — are the entity strings and relation endpoints correct, not merely well-formed.
2. The grammar and prompt templates are **part of the model artifact**. A checkpoint without them is not a deliverable.

**Training hardware — both, with a rule:**

| Where | What | Why |
| --- | --- | --- |
| **Local 1650 Ti (3.4 GB), plain LoRA** | Overfit-a-tiny-batch checks, pipeline debugging, checkpoint-and-resume tests, HP sanity, 1–5k pilots, the Gate 0 model bake-off | LoRA on 360M is ~1.0–1.5 GB, on 0.6B ~2.0–3.0 GB — both fit. Getting the loop right before spending a Kaggle session is good practice |
| **Kaggle T4 (16 GB), full fine-tune** | Real SFT runs, rejection sampling at volume, **anything producing a gate number** | The 360M full-FT footprint is not measured yet; Kaggle is the default for the shipping recipe until a committed memory test proves local full FT is viable. The local card is an estimated 25–50× slower than a 4090, so a 4–8 h T4 run is weeks |

Real runs are **full fine-tune, not LoRA**, because this is **task shift** — a new output format and a new behaviour — which is the regime where the LoRA-Learns-Less evidence favours full FT. LoRA is the iteration tool, not the shipping recipe. **Never report a local LoRA number against a gate threshold.**

Two hardware facts that will bite you:

- **T4 and the 1650 Ti are both SM 7.5: fp16 + `GradScaler`, no bf16.** A bf16 config is a bug, not a preference. Blog-post recipes will get this wrong.
- **On the local card, context length — not parameter count — is what OOMs you.** SM 7.5 has no FlashAttention-2, so naive attention at sequence length 4096 costs ~537 MB per layer, and extraction prompts are long (~1,200-token chunk + few-shot + schema). **Require PyTorch SDPA's memory-efficient backend or xformers explicitly, and assert it is active** — a silent fallback presents as an OOM that looks like "model too big."
- **Prefer plain LoRA over QLoRA locally.** 4-bit saves only ~540 MB on a 0.72 GB base and costs dequantisation overhead every forward pass.

## 6. The top risk, and the mandatory mitigation

**Description hallucination poisoning the graph.** A wrong entity is one wrong node. A fabricated description is a plausible-sounding lie that propagates into community reports, into embeddings, and into every answer touching that node — and it looks fine on inspection.

**The mitigation is mandatory, not optional:**

- Train descriptions as **spans or near-spans of the source chunk**, not as free generation.
- **Enforce source-overlap at decode time** — a description below the overlap threshold with its chunk cannot be emitted.
- Measure **description faithfulness** as a gate metric (target ≥0.95, hard stop below 0.90).

Grammar constrains *shape*; overlap constrains *groundedness*. Both are required. A perfectly-shaped fabricated description is exactly the failure this project must not ship.

## 7. Stage gates — exact thresholds

All metrics are measured against a **hand-annotated gold set from Kartik's own corpus** (200–500 examples), not a public benchmark and not teacher output.

**Gate 0 — baseline, before any training.** Measure stock SmolLM2-360M, stock Qwen3-0.6B, a 7B-class model, and the teacher on the gold set with identical prompts and decoder — each small model **with and without demonstrations**. Then run the **side-by-side LoRA pilot** (360M vs 0.6B, same 1–5k subset) described in §5 to validate and de-risk the SmolLM2 primary choice; it is diagnostic, not a gate number, and it does not make Qwen co-primary. Proceed only if fine-tuning has visible headroom. Stop and rethink if the stock primary is already near the teacher, if the teacher itself scores poorly (the schema is the problem, not the model), or if the 7B already solves it at acceptable local speed.

**Gate 1 — SFT:**

| Metric | Target | Hard stop |
| --- | --- | --- |
| Entity recall | ≥ 0.85 | < 0.70 |
| Entity F1 | ≥ 0.80 | < 0.65 |
| Strict relation-triple F1 | ≥ 0.60 | < 0.45 |
| Description faithfulness | ≥ 0.95 | < 0.90 |
| Batched throughput | ≥ 500 tok/s | < 300 tok/s |

Recall is weighted above precision on purpose — a missed entity is an unreachable node, a spurious one is cheap. Strict triple F1 means all of (head, relation, tail) correct, no partial credit. Below any hard stop: do not proceed to rejection sampling.

**Gate 2 — rejection sampling.** Pass requires all three: entity F1 improves by **≥3 points**, description faithfulness improves, and entity recall is not lost.

**Hard stop: if fewer than 30% of sampled completions pass the verifier, stop.** Do not raise temperature, loosen the verifier, or sample more. A sub-30% pass rate means SFT did not give self-distillation enough correct material, and sampling harder just collects more of the same mistakes. **Go back for more SFT data — backward, not forward.** This gate is the one most likely to be rationalised past.

**Gate 3 — downstream. This is the gate that matters.** Build two graphs over the same held-out slice — one with the small model, one with the teacher — with everything else identical. Run the same QA set through both.

| Result | Decision |
| --- | --- |
| ≥90% of teacher-graph answer quality **and** ≥5× indexing throughput | **Ship** |
| 75–90% | **Selective escalation** — route hard chunks to the teacher, derive the routing rule from error analysis |
| <75% | **Stop and re-scope** |

Both conditions are required to ship. 95% quality at 2× throughput is not a win.

**Expect Gate 1 to pass while Gate 3 fails.** This is a realistic outcome, not a sign something broke. Extraction F1 and retrieval utility are different quantities: F1 weights all entities equally but missing one hub entity breaks every path through it; per-chunk F1 cannot see cross-chunk entity-resolution failure that fragments the graph; and model errors are systematic, so they concentrate in one region of the graph rather than averaging out.

## 8. Standards — enforceable, from AGENTS.md

Read `AGENTS.md` in full. The condensed version:

**The benchmark-number rule, which is absolute:** *no number enters any document, commit message, summary, or decision unless it came from a committed results file, and the citation names that file.* Write `entity F1 0.81 (runs/20260920-sft-v3/metrics.json)`. Never interpolate, never infer a number from a related metric, never carry one forward from another project. Mark estimates as estimates with their basis. **If you do not have a number, say you do not have it.** A plausible fabricated metric is more expensive than no metric, because it gets planned against.

**Reproducibility:** pinned versions; seed in the config, never a literal; every run is `script + config`; **no hard-coded paths** (not `D:\...`, not `F:\...`, not `/kaggle/input/...`); the config is *copied* into the run directory, not referenced. Every run writes `env.json` with commit SHA, dirty-tree flag, versions, GPU, CUDA.

**Run tracking:** `runs/<timestamp>-<name>/` with `config.yaml`, `env.json`, `metrics.json`, `log.txt`. Failed runs are **kept** and marked failed. `metrics.json` records the gold-set version/hash it scored against. Weights are gitignored; metrics/config/env are committed. Every completed, failed, or aborted stage run also gets a row in `docs/RUN_LEDGER.md` in the same commit.

**Notebooks:** exploration and Kaggle execution only. Training logic lives in `src/kg_llm_tune/`; the Kaggle notebook is a thin driver that clones a **pinned commit SHA**. Anything copied into a second notebook graduates to `src/`. Strip outputs. No notebook is ever the source of a benchmark number.

**Kaggle notebook requirements — designed in, not retrofitted:**

- The notebook is **generated from locally-validated code**, never written independently. Validate the loop locally with LoRA on a small subset first. Divergence between the local and Kaggle training paths is how these projects break — two separately-written loops drift on a tokenizer flag, and the bug surfaces as "Kaggle scored worse" rather than as an error.
- Sessions complete inside the limit **by design**: a graceful stop at **8–9 hours**, triggered by an **elapsed-time watchdog checked inside the training loop**, not by estimating up front that the run will fit.
- On stop, the sequence is fixed: finish the current step → save weights → optimizer state → scheduler state → RNG state → step/epoch counters → write a **resume manifest** → **exit cleanly** so `/kaggle/working` is preserved. An unclean exit loses the outputs and turns a graceful stop into a lost session.
- **Resume is first-class from day one, and must continue the LR schedule and data ordering — not silently restart them.** This is the classic bug: a resumed run that restarts warmup or reshuffles from epoch zero produces a plausible loss curve and an **invalid result**, without crashing or warning. Assert both continue.
- **Checkpoint periodically as well as at the time limit**, so an unexpected disconnect costs at most N steps.
- Save optimizer, scheduler, and RNG state — weights-only checkpoints make resume a lie. Test resume by deliberately killing a short run, locally, before the first long Kaggle run.

**Data provenance:** every dataset gets a row in `data/README.md` (URL, licence, date, SHA, use). `Babelscape/rebel-dataset` is **CC-BY-SA-4.0** — share-alike, with implications for derived data; do not include it until the repo licence is settled. Teacher ToS must be checked for output-use restrictions before generating training data.

**Coding style (Kartik's, follow it):** short, readable, human-traceable. **No docstrings.** Comments explain *why*, not *what*. Minimal defensive abstraction — no base class with one subclass, no factory for two options. Fail loudly: no bare `except`, no `except Exception: pass`; expected failures are caught specifically and **counted**. Type hints on module-level signatures only.

**No silent failure anywhere.** A pipeline that quietly drops 8% of its input and reports clean metrics is the worst outcome available. Every filter reports what it dropped and why.

## 9. Current status

**Phase 1, pre-Gate-0.** Scaffold plus local smoke plumbing.

Done: repo created, full document scaffold written (README, AGENTS.md, `docs/PHASE1_GOALS.md`, `docs/ARCHITECTURE.md`, `docs/DATA_STRATEGY.md`, `docs/BLOCKERS.md`, `docs/BENCHMARKING_PROTOCOL.md`, `docs/RUN_LEDGER.md`), directory structure with purpose READMEs, `.gitignore`, example configs for both the Kaggle full-FT run and the local LoRA run, DocRED open-data format-bootstrap pull, repo-local venv on `D:`, one tiny SmolLM2-360M local LoRA smoke run, one tiny QLoRA rank sweep over r=8, r=16, and r=32, one 5k-example QLoRA rank sweep over r=8, r=16, and r=32, and three context memory probes.

Committed run evidence: `runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json`, `runs/20260918-203600-smollm2-docred-qlora-r8/metrics.json`, `runs/20260918-203700-smollm2-docred-qlora-r16/metrics.json`, `runs/20260918-203800-smollm2-docred-qlora-r32/metrics.json`, `runs/20260918-210000-smollm2-docred5k-qlora-r8/metrics.json`, `runs/20260919-003400-smollm2-docred5k-qlora-r16/metrics.json`, `runs/20260919-042300-smollm2-docred5k-qlora-r32/metrics.json`, `runs/20260919-ctx4096-smollm2-qlora-r16-probe/metrics.json`, `runs/20260919-ctx6144-smollm2-qlora-r16-probe/metrics.json`, and `runs/20260919-ctx8192-smollm2-qlora-r16-probe/metrics.json`. These are diagnostic only, not gate metrics. They used DocRED format-bootstrap data or synthetic memory probing, not the target GraphRAG SFT mix. The current next short-context local QLoRA rank is r=16 per `docs/RUN_LEDGER.md` and `runs/20260919-003400-smollm2-docred5k-qlora-r16/metrics.json`; r=32 is not selected under this config because `runs/20260919-042300-smollm2-docred5k-qlora-r32/metrics.json` records `train_loss_finite` false. Native-context local QLoRA training at r=16 failed with `OutOfMemoryError` per `runs/20260919-ctx8192-smollm2-qlora-r16-probe/metrics.json`, while 4096-token and 6144-token synthetic probes completed per their metrics files. Run 8192-token GraphRAG capability pilots on Kaggle unless a shorter run is explicitly labelled smoke-only.

Not done: no gold set, no Gate 0 baseline, no Qwen3 side-by-side pilot, no GraphRAG-specific SFT mixture, no Kaggle full fine-tune, and no gate metric.

**Critical path — the gold set.** 200–500 hand-annotated examples from Kartik's own corpus, stratified for coverage, spans not just strings, with the annotation guideline written *before* annotating and a double-annotated slice to establish the noise floor. Nothing downstream can be measured until it exists. Protocol in `docs/DATA_STRATEGY.md` §1.

**Cost finding that shapes the plan:** free API tiers cannot produce the training set. Groq's caps are **per model per day**, making ~20k examples roughly 80 days; OpenRouter free is roughly 20 days — and that is for *one* teacher pass, while multi-teacher agreement needs three. Budget **$20–100 of paid credit**. Run a 500-chunk pilot before the full spend.

## 10. Open blockers

See `docs/BLOCKERS.md` for the full text. Summary:

- **B1 Kaggle not signed in.** Two separate problems: the browser's Google account is `instagramy1820@gmail.com` (the wrong identity — signing in would hit the wrong account or create a new one), and the Google OAuth popup opens outside the automatable tab group so the flow may not be drivable at all. Fix: sign in manually once at the machine, then obtain a Kaggle API token and stop depending on the browser.
- **B2 Local execution needs Kartik physically present** (~2 weeks). Terminals are click-only; folder prompts time out.
- **B3 Linux sandbox broken** (`vm_bundles is a symlink or junction`).
- **B4 POLICY, only Kartik can answer:** can corpus data leave Impetus for Kaggle? If no, the architecture changes — corpus becomes evaluation-only, teacher distillation over corpus chunks becomes questionable too, and the public repo cannot hold the gold set. **Answer this before collecting any data.**
- **B5 POLICY, only Kartik can answer:** what does the OmniRouter endpoint route to? Affects both teacher ToS compliance and whether the three "independent" teachers are actually independent — correlated models agree on their shared hallucinations, which defeats the agreement rule.

## 11. What you must NOT do

- **Do not change a decided architecture choice without flagging it and waiting.** The choices in §5 are settled. Raise an objection; do not act on it unilaterally.
- **Do not invent numbers.** No metric, benchmark, cost, timing, or citation that did not come from a committed results file or a named external source. This is the single most damaging thing you can do here.
- **Do not commit weights, checkpoints, datasets, or anything from the corpus.** `.gitignore` covers these; do not override it. The repo is public.
- **Do not push directly to `main` without the work being reviewable.** Use a branch and a PR for anything beyond a documentation fix. A large unreviewable commit is not acceptable regardless of how good it is.
- **Do not upload corpus data anywhere** — Kaggle, a teacher API, anywhere — until blocker B4 is answered.
- **Do not mark a task complete because code was written.** Complete means it ran and the output was inspected.
- **Do not work around an environment limitation silently.** Write it into `docs/BLOCKERS.md`. A silent workaround is a blocker discovered three weeks later.
- **Do not report parse rate or schema validity as a result.** Constrained decoding makes it 100%; it measures the library, not the model.

## 12. Where to start

1. Read `AGENTS.md`, `docs/PHASE1_GOALS.md`, `docs/ARCHITECTURE.md`, `docs/DATA_STRATEGY.md`.
2. Check `docs/BLOCKERS.md` — if the work depends on a blocker, say so and stop rather than improvising around it.
3. Pick work off the critical path: the gold-set protocol and `src/kg_llm_tune/` module skeletons are the highest-value things that need no Kaggle, no paid API, and no policy answer.
4. Report what you did, what ran, what it produced, and what you could not do — in that order.
