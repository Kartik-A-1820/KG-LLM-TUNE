# Data Strategy

Where training data comes from, what it costs, and what gets thrown away before it is trusted.

One finding up front, because it changes the plan rather than decorating it: **free API tiers cannot produce this training set.** Budget **$20–100 of paid credit**. Details in §4.

---

## 1. The gold set — the non-negotiable foundation

**200–500 hand-annotated examples from Kartik's own corpus.** Everything in Phase 1 is measured against this. It is the single highest-value artifact in the repo and the only one that cannot be regenerated.

It is not training data. It is the measuring stick. Nothing from the gold set enters SFT, rejection sampling, or the Gate 3 held-out slice.

### Protocol

1. **Sample for coverage, not randomly.** Stratify across document types, lengths, and density. A random sample over-represents the modal document and leaves the hard cases — tables, lists, long nested sentences, ambiguous coreference — measured by one or two examples each. Those are the cases that will break the pipeline.
2. **Write the annotation guideline before annotating.** Entity types, what counts as a relation, how to handle coreference, whether nested entities are split, span boundary conventions, what to do with a pronoun that is the only mention. Ambiguity resolved ad hoc during annotation is ambiguity baked into the metric.
3. **Annotate spans, not just strings.** Character offsets into the chunk. This is what makes the verbatim check (§5) and description-faithfulness scoring possible.
4. **Double-annotate a slice.** Re-annotate 30–50 examples after a gap of some days and measure self-agreement. This is the noise floor of the whole project: if self-agreement on strict triples is 0.85, a model scoring 0.85 has hit the ceiling of what the gold set can distinguish, and a 0.87 vs 0.84 comparison is measuring nothing.
5. **Version and hash it.** `eval/gold/v1/`, with a manifest recording the hash and the guideline version. Every metric records which version it scored against.
6. **Freeze it before training.** Editing the gold set after seeing model errors is how a project talks itself into a good score. If it must change, that is a new version, and prior numbers do not carry forward.

### Size

500 is better than 200. Below 200, confidence intervals on F1 are wide enough that gate thresholds stop separating outcomes — a 3-point Gate 2 improvement would not be distinguishable from noise. **Estimate:** at n=200 the 95% CI on an F1 near 0.80 is roughly ±0.055, so 200 is about the floor at which Gate 2's ±3-point criterion is meaningful at all.

It lives in `eval/gold/` and it is **committed**, subject to the corpus-export policy question in `docs/BLOCKERS.md`.

---

## 2. Candidate public datasets

Use these to bulk out the SFT mix and to pretrain the output format — **not** to define success. They are different text from a different domain with a different schema.

| Dataset | Licence | Use | Note |
| --- | --- | --- | --- |
| `Babelscape/rebel-dataset` | **CC-BY-SA-4.0** | Large-scale relation extraction | Share-alike. Derived training data — and arguably a model trained on it — carries obligations. Do not include until the repo licence is settled. |
| `thunlp/docred` | Check per-file; DocRED is generally MIT-adjacent with Wikipedia-derived CC-BY-SA text | Document-level RE, cross-sentence relations | Closest public proxy for multi-sentence extraction. Verify the licence on the specific HF copy used, not on the paper. |
| CoNLL04 | Research-use; verify | Small, clean joint entity+relation | Good for format bootstrapping and quick sanity runs |
| SciERC | Research-use; verify | Scientific-domain entities/relations | Include only if the corpus is technical; otherwise it teaches the wrong entity distribution |

**Licences are verified per HF repo at pull time and recorded in `data/README.md`** with the date. A licence stated in a paper is not the licence on the copy you downloaded.

**Domain gap is the real caveat.** Public RE datasets are mostly Wikipedia-style encyclopedic prose with a closed relation inventory. If Kartik's corpus is technical documents, reports, or internal writing, a model tuned hard on REBEL will learn REBEL's entity taxonomy and carry it into the graph. Treat public data as **format and structure pretraining**; teacher-distilled in-domain data is what teaches the actual task.

---

## 3. Teacher distillation with multi-teacher agreement

Public data teaches the shape. The teacher teaches the corpus.

### Plan

1. Sample chunks from the corpus — excluding gold-set chunks and the Gate 3 held-out slice.
2. Run each chunk through **three teacher models** from different families. Different families matter: two models from the same family share failure modes, so their agreement is much weaker evidence than it looks.
3. **Agreement rule: a triple is kept only if ≥2 of 3 teachers agree on it.** Agreement is on the normalised triple (head, relation, tail), not on the raw string.
4. Entities are kept if they appear in ≥2 of 3 outputs. Descriptions are taken from the teacher whose output best overlaps the source chunk — not concatenated, not averaged, because merging descriptions invents claims that no teacher made.
5. Record per-chunk agreement rate. Low-agreement chunks are diagnostic: they mark either genuinely hard text or an under-specified schema, and they are the right candidates to add to the gold set in v2.

**What agreement buys:** it filters teacher hallucination, which is the main risk in distillation. A hallucinated triple is usually idiosyncratic to one model, so it fails the 2-of-3 test.

**What agreement costs:** recall. Real-but-subtle relations that only the strongest teacher catches get dropped. That is an accepted trade — for this project a clean training signal beats a complete one, since a precision-poor training set teaches the student to hallucinate, and hallucination is the top project risk.

**ToS:** every teacher's terms are checked for restrictions on using outputs to train models, and the finding is recorded in `data/README.md` before generation starts. This is currently open for the OmniRouter endpoint — see `docs/BLOCKERS.md`.

---

## 4. Cost — free tiers cannot do this

Measured against a target of roughly 20,000 training examples.

| Route | Throughput reality | Time to 20k | Verdict |
| --- | --- | --- | --- |
| Groq free tier | Per-model **daily** caps, not just per-minute | **~80 days** | Unusable |
| OpenRouter free models | Free-tier daily request caps | **~20 days** | Unusable |
| Paid credit | Rate-limited by minute, not by day | Hours to days | **Use this** |

The Groq number is the one that kills the free plan. The cap is **per model per day**, so the usual workaround — rotating across models — does not compound the way it does with per-minute limits. It just spreads the same daily ceiling.

And note this is the cost for **one** teacher pass. Multi-teacher agreement multiplies it by three, which makes the free-tier timeline absurd rather than merely slow.

**Budget $20–100 of paid credit** for teacher distillation. Against 80 days of wall-clock — during which nothing downstream can start — this is the cheapest decision in the project.

**Spend control:**

- Start with a **500-chunk pilot** before the full run. Inspect the output, check agreement rates, confirm the prompts are right. Discovering a prompt bug after spending the whole budget is the failure mode here.
- Cache every teacher response to disk keyed by (model, prompt hash, chunk hash). Re-running the pipeline must not re-bill.
- Log token counts and running cost per batch; write them into the run directory.
- Set a hard spend cap at the provider.

---

## 5. Programmatic pre-checks — before any LLM judge

**Every generated example passes these deterministic checks first. Only survivors reach an LLM judge.**

Two reasons, and both are load-bearing:

1. **Cost.** The judge is an API call per example. Filtering deterministically first cuts the judge's input set, often by a large fraction. The cheap check runs first.
2. **Reliability.** An LLM judge is the same class of system being judged, with correlated blind spots. It will wave through a fabricated entity that reads plausibly. A string comparison will not. Where a deterministic check exists, it is strictly better evidence than a judge — so the judge is reserved for the genuinely subjective residue.

### The checks

1. **Schema validity.** Parses, required fields present, types correct, relation labels in the allowed inventory.
2. **Entity strings appear verbatim in the source chunk.** Exact substring match, after a documented normalisation (whitespace, case policy, unicode). If it is not in the text, it was invented — no judge needed to know that. This is the single most effective hallucination filter in the pipeline.
3. **Relation endpoints exist in the entity list.** Both head and tail resolve to entities extracted from the same chunk. A relation pointing at a non-existent node is a dangling edge, and dangling edges break traversal silently.
4. **Description overlap threshold.** Each description must share at least a configured fraction of content tokens with its source chunk. This is the training-side counterpart of the decode-time overlap constraint in `ARCHITECTURE.md`. The threshold is a config value, tuned once on the gold set and then fixed — and recorded, because changing it mid-project invalidates comparisons.

Every check records **counts of what it dropped and why**, into the run's metrics file. Those counts are diagnostics, not noise: a spike in check-2 failures means a teacher is drifting into paraphrase; a spike in check-3 means the entity and relation prompts have fallen out of sync.

### Then, and only then, the judge

The LLM judge handles what deterministic checks cannot: is the relation *type* right, is the description *informative* as well as grounded, is the extraction complete. Its output is a secondary signal. It never overrides a programmatic failure, and its scores are never the primary number in a gate.

---

## 6. Data splits

| Split | Source | Used for | Never used for |
| --- | --- | --- | --- |
| **Gold set** (200–500) | Hand-annotated, own corpus | All gate metrics | Training, sampling, tuning |
| **SFT train** | Public + teacher-distilled | SFT | Any reported metric |
| **SFT val** | Teacher-distilled, held out | Early stopping, LR selection | Gate numbers |
| **Rejection-sampling pool** | Corpus chunks, unlabelled | Self-distillation generation | Evaluation |
| **Gate 3 held-out slice** | Corpus chunks | Two-graph downstream comparison | Everything else |

The Gate 3 slice is held out from the start and touched once. Chunk-hash based, so a re-run cannot accidentally leak it into the training pool.

---

## 7. Storage and provenance

- `data/` is **gitignored**. Nothing from the corpus is committed. See `data/README.md` for the expected layout.
- `eval/gold/` **is** committed — small, hand-made, irreplaceable, and version-controlled on purpose.
- Every dataset gets a row in `data/README.md`: source URL, licence, date pulled, SHA, and what it is used for. A dataset with no row is not used.
- Teacher-response cache lives under `data/cache/` and is gitignored but **backed up** — it represents real spend.
- **Nothing from Kartik's corpus leaves the machine** — including to Kaggle — until the export-policy question in `docs/BLOCKERS.md` is answered. This is a hard constraint on the training plan, since training runs on Kaggle, and it may force a synthetic or public-only training set with the corpus used for evaluation alone.
