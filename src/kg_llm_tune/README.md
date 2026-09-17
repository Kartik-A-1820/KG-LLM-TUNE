# src/kg_llm_tune

Library code. Anything reused lives here — not in a notebook.

The rule from `AGENTS.md` §2: the first time a cell is copied into a second notebook, it graduates to this package. Notebooks call into here; they do not contain logic.

## Planned modules

Flat files, not sub-packages. This is a small library and nesting it would be the kind of defensive abstraction `AGENTS.md` §8 rules out.

| Module | Responsibility |
| --- | --- |
| `data.py` | Corpus loading, chunking, normalisation, split management (chunk-hash based, so the Gate 3 slice cannot leak) |
| `teacher.py` | Teacher API calls, response caching keyed by (model, prompt hash, chunk hash), multi-teacher agreement |
| `filters.py` | The programmatic pre-checks — schema validity, verbatim entity strings, endpoint existence, description overlap. Returns drop counts by reason, not just a boolean |
| `train.py` | SFT loop. Config-driven, seeded, checkpoint-and-resume with optimizer + scheduler + RNG state |
| `sample.py` | Rejection sampling and self-distillation, including the verifier pass-rate accounting Gate 2 depends on |
| `decode.py` | Constrained decoding (XGrammar / Outlines) plus source-overlap enforcement |
| `evaluate.py` | Gold-set scoring — entity P/R/F1, strict relation-triple F1, description faithfulness, throughput |
| `graph.py` | Minimal graph construction, only as much as Gate 3's two-graph comparison needs. Not the Phase 2 pipeline |
| `embeddings.py` | Embedding A/B harness for EmbeddingGemma-300M vs potion-retrieval-32M |
| `runs.py` | Run directory creation, `env.json` capture, `metrics.json` writing |

Nothing here is written yet.

## Conventions

- No hard-coded paths. Paths come from config.
- No docstrings. Comments explain *why*.
- Type hints on module-level signatures.
- Fail loudly: no bare `except`, no `except Exception: pass`. Expected-and-tolerable failures are caught specifically and **counted**.
- Every function that drops data returns or records how much it dropped and why.
