# configs/

Every run is `script + config file`. No editing code to change a hyperparameter.

## Rules

- **The config used is copied into the run directory**, not referenced by path. The file on disk will change; the copy is the record.
- **No hard-coded paths anywhere.** Paths live in the `paths:` block, and the same config must run locally and on Kaggle with only that block changed.
- **The seed is a config field.** Never a literal in code.
- **`precision: fp16`** for anything targeting Kaggle. The T4 is SM 7.5 and has no bf16. A config specifying bf16 is a bug.
- Configs are committed. A run whose config is not in the repo is not reproducible.

## Files

| File | Purpose |
| --- | --- |
| `phase1_sft.example.yaml` | Kaggle full fine-tune — the real SFT run. Gate numbers come from this shape of config. |
| `phase1_sft_local_lora.example.yaml` | Local LoRA iteration on the 1650 Ti — pilots, debugging, resume tests, HP sanity. Diagnostic only; not a gate config. |

Both are templates. Copy, do not edit in place.

Two things the local config carries that the Kaggle one does not, and that must not be dropped:

- **`attention.impl`** — SM 7.5 has no FlashAttention-2, and naive attention at seq 4096 is ~537 MB per layer. SDPA's memory-efficient backend or xformers is required, and the run asserts the backend is actually active.
- **`quantization: none`** — plain LoRA, not QLoRA. 4-bit saves ~540 MB on a 0.72 GB base and costs dequant overhead; not worth it at this size.

More will be added as stages are built: `gate0_baseline.yaml`, `teacher_distill.yaml`, `rejection_sampling.yaml`, `eval_gold.yaml`, `embedding_ab.yaml`.
