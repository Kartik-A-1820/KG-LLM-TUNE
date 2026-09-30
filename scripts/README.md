# scripts/

Research utilities and CLI entry points for dataset preparation, local pilot training, and evaluation. Production training logic should move into `src/kg_llm_tune` before it is reused by Kaggle.

## Available

| Script | Does |
| --- | --- |
| `prepare_docred_sft.py` | Downloads and converts a seeded DocRED subset to the project extraction JSON shape |
| `prepare_kuzu_pilot.py` | Combines DocRED extraction and Neo4j Text2Cypher rows into a hashed local pilot split |
| `train_lora_smoke.py` | Local LoRA/QLoRA diagnostic trainer with progress metrics and optimizer/RNG/data-position checkpoints |
| `probe_context_memory.py` | Measures local context-length memory behavior |
| `evaluate_relation_smoke.py` | Scores relation triples on the small relation smoke fixture |
| `smoke_train_loss.py` | Minimal training-loss smoke check |

These local utilities are diagnostic. They do not produce Gate 0 or Gate 1 metrics.
Install the pinned packages from `requirements.txt` using the repository's `.venv`; do not install them into the global Python environment.

## Planned

| Script | Does |
| --- | --- |
| `prepare_corpus.py` | Chunk + normalise the corpus, assign stable chunk hashes, draw the Gate 3 holdout |
| `run_teacher.py` | Teacher distillation with caching and multi-teacher agreement. Supports `--limit` for the 500-chunk pilot |
| `build_sft.py` | Programmatic pre-checks → LLM judge on survivors → SFT train/val split. Reports drop counts by reason |
| `train_sft.py` | SFT. `--resume` flips it into resume-from-checkpoint |
| `rejection_sample.py` | Sampling + verifier + self-distillation set construction. Reports verifier pass rate overall and per task |
| `evaluate.py` | Score a model against a gold-set version; writes `metrics.json` |
| `gate3_compare.py` | Build two graphs over the holdout (small model vs teacher), run the QA set through both, report quality ratio and indexing throughput |
| `embedding_ab.py` | End-to-end embedding A/B over a fixed graph |

The planned production pipeline below remains unimplemented.

## Convention

```bash
python scripts/train_sft.py --config configs/phase1_sft_local_lora.yaml
python scripts/train_sft.py --config configs/phase1_sft_kaggle.yaml --resume runs/20260920-sft-v1
```

Same script, different config. If a script needs a code edit to change behaviour, the config is missing a field.
