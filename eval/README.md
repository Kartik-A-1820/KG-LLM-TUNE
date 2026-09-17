# eval/

The gold set and the evaluation harness. This directory is the ground truth of the project.

Unlike `data/`, this **is committed** — the annotations are hand-made, small, and irreplaceable, and version control is the point.

(Caveat: the repo is public. If the corpus is sensitive, see `docs/BLOCKERS.md` B4 before committing any annotations.)

## Layout

```
eval/
  gold/
    v1/
      guideline.md      annotation guideline — written BEFORE annotating
      examples.jsonl    chunks + annotated entities, relations, descriptions (with char spans)
      manifest.json     hash, count, date, guideline version, annotator
      agreement.json    self-agreement from the double-annotated slice
  harness/              scoring code lives in src/kg_llm_tune/evaluate.py; this holds fixtures
```

## The gold set in one paragraph

200–500 hand-annotated examples from Kartik's own corpus, stratified for coverage rather than sampled randomly. It is the **measuring stick, never training data** — it does not enter SFT, rejection sampling, or the Gate 3 held-out slice. Every reported metric records which gold-set version it scored against. Full protocol in `docs/DATA_STRATEGY.md` §1.

## Rules

- **Freeze before training.** Editing the gold set after seeing model errors is how a project talks itself into a good score. Changes mean a new version; prior numbers do not carry forward.
- **Version and hash.** `v1`, `v2`, … each with a manifest. A metric measured against v1 is not comparable to one measured against v2, and mixing them violates `AGENTS.md` §5.
- **Self-agreement is the noise floor.** Re-annotate 30–50 examples after a gap and measure. If self-agreement on strict triples is 0.85, then a model at 0.85 has hit the ceiling of what this gold set can distinguish, and 0.87-vs-0.84 comparisons are measuring nothing. Know this number before arguing about small deltas.
- **Annotate spans, not strings.** Character offsets into the chunk — this is what makes the verbatim check and faithfulness scoring possible.

## Metrics computed here

- entity precision / recall / F1 — recall weighted higher on purpose; a missed entity is an unreachable node
- strict relation-triple F1 — (head, relation, tail) all correct, no partial credit
- description faithfulness — share of descriptions supported by the source chunk
- batched throughput (tok/s) — measured with constrained decoding on, at realistic batch size

**Not computed:** schema validity / parse rate. Constrained decoding makes it 100% by construction, so it measures the constraint library, not the model.
