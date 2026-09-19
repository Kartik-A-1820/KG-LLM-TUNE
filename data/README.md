# data/

**Gitignored.** Nothing from the corpus is committed. This file is the only thing in here that is tracked.

## Expected layout

```
data/
  raw/            original corpus documents, untouched
  chunks/         chunked + normalised, with stable chunk hashes
  teacher/        raw per-teacher responses (one file per teacher)
  cache/          teacher response cache, keyed (model, prompt_hash, chunk_hash)
  sft/
    train.jsonl
    val.jsonl
  rejection/      sampled completions + verifier verdicts
  holdout/        Gate 3 slice — chunk-hash selected, touched once
```

The gold set is **not** here. It lives in `eval/gold/` and it is committed — see `eval/README.md`.

## Provenance table

Every dataset used gets a row. A dataset with no row is not used. Fill in at pull time, not from memory.

| Dataset | Source URL | Licence | Date pulled | SHA | Used for |
| --- | --- | --- | --- | --- | --- |
| `thunlp/docred` | https://huggingface.co/datasets/thunlp/docred | MIT, per Hugging Face dataset tag | 2026-09-18 | train source `0d01cd07cabd7f9077db6ea8628832cf60281b4228ec1ba54647f836a3b17d02`; val source `6ae4d7f5b0b9d2cbe74b9634ed43b35b7cb5b7c0dc3a16226dbe343139a4ae05` in `runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json` | Open-data SFT smoke / format bootstrap only; not a gate metric |
| `thunlp/docred` distant split | https://huggingface.co/datasets/thunlp/docred | MIT, per Hugging Face dataset tag | 2026-09-18 | train distant source `c420c0429310583cfc9459f7daa26b1f4c11ff5c7a1481aa64ab9db2b296b905`; val source `6ae4d7f5b0b9d2cbe74b9634ed43b35b7cb5b7c0dc3a16226dbe343139a4ae05` in `data/open_pilots/docred_sft_5k_distant_v1/manifest.json` | 5k local QLoRA rank speed/VRAM benchmark only; distantly supervised and not a gate metric |

Licence is verified against the actual copy downloaded, not against the paper.

## Teacher ToS findings

Before any teacher generates training data, its terms are checked for restrictions on using outputs to train models, and the finding is recorded here.

| Teacher | Provider | Output-use terms | Checked on | Verdict |
| --- | --- | --- | --- | --- |
| *(none yet)* | | | | |

Open question: what the OmniRouter endpoint actually routes to. See `docs/BLOCKERS.md` B5.

## Hard constraints

- **Nothing from the corpus leaves the machine** — including to Kaggle or to a teacher API — until the export-policy question in `docs/BLOCKERS.md` B4 is answered.
- `data/cache/` is gitignored but should be **backed up**. It represents real money spent.
- The Gate 3 holdout is selected by chunk hash at the start and never re-drawn, so a re-run cannot leak it into training.
