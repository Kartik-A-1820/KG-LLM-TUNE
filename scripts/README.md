# scripts/

Thin CLI entry points over `src/kg_llm_tune`. Argument parsing, config loading, and a call into the library — nothing else. Logic that lives here instead of in `src/` cannot be imported, tested, or reused from the Kaggle notebook.

Every script takes `--config <path>` and writes into a timestamped run directory.

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

None written yet.

## Convention

```bash
python scripts/train_sft.py --config configs/phase1_sft_local_lora.yaml
python scripts/train_sft.py --config configs/phase1_sft_kaggle.yaml --resume runs/20260920-sft-v1
```

Same script, different config. If a script needs a code edit to change behaviour, the config is missing a field.
