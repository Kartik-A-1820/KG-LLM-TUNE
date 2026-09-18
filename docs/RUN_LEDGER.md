# Run Ledger

This is the factual index of committed runs. Every metric below cites the run
file that contains it. If a value is not in a committed run file, it does not
belong here.

## Ledger

| Stage | Run | Model | Data | Hardware | Status | Recorded metrics | Decision / next step | Caveats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open-data local LoRA smoke | `runs/20260918-080000-smollm2-docred-lora-smoke/` | `HuggingFaceTB/SmolLM2-360M-Instruct` | `thunlp/docred`, MIT; source hashes in `runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json` | NVIDIA GeForce GTX 1650 Ti, CUDA available, torch `2.11.0+cu128` in `runs/20260918-080000-smollm2-docred-lora-smoke/env.json` | complete | `train_examples` 16, `val_examples` 4, `initial_val_loss` 0.8313882797956467, `final_val_loss` 0.7843165993690491, `val_loss_delta` -0.047071680426597595, `elapsed_seconds` 94.892, `cuda_max_memory_allocated_bytes` 1076456448, all from `runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json` | Local path, venv, CUDA, tokenizer/model loading, LoRA adapter path, and loss movement are verified for a tiny diagnostic run. Next: build the GraphRAG-specific mix and gold/eval path before any gate claim. | Diagnostic only, not a gate metric. DocRED is format bootstrap data, not the target SFT mix. Run was made from a dirty tree and therefore is provisional per `runs/20260918-080000-smollm2-docred-lora-smoke/env.json`. It predates `docs/BENCHMARKING_PROTOCOL.md`, so no `log.txt` exists. |

## Current Evidence State

- There is one committed local GPU training record.
- There are no Gate 0, Gate 1, Gate 2, or Gate 3 metrics yet.
- There is no committed gold set yet.
- There is no committed Qwen3-0.6B side-by-side pilot yet.
- There is no committed full fine-tune memory test for SmolLM2-360M yet.
