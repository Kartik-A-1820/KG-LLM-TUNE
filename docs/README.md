# docs/

Planning and design documents for Phase 1. Read in this order:

| Document | What it answers |
| --- | --- |
| [`PHASE1_GOALS.md`](PHASE1_GOALS.md) | What "done" means — the four stage gates with numeric targets and hard stop conditions, plus the local-vs-Kaggle training split |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the pieces fit, what runs where, the two-model split, where constrained decoding sits, how the embedder is A/B'd |
| [`DATA_STRATEGY.md`](DATA_STRATEGY.md) | The gold set, candidate public datasets and their licences, teacher distillation with multi-teacher agreement, the programmatic pre-checks, and what the data actually costs |
| [`BENCHMARKING_PROTOCOL.md`](BENCHMARKING_PROTOCOL.md) | How every stage records configs, environment, metrics, logs, comparisons, and failures |
| [`RUN_LEDGER.md`](RUN_LEDGER.md) | Factual index of committed runs and the metrics each run proves |
| [`BLOCKERS.md`](BLOCKERS.md) | What is blocked, why, and which two questions only Kartik can answer |

`AGENTS.md` and `HANDOFF_PROMPT.md` live at the repo root because they are entry points, not reference material.

These are working documents. Update them when a decision changes — a doc that disagrees with the code is worse than no doc, because it gets planned against.
