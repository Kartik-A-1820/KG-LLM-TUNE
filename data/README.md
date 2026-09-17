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
| *(none yet)* | | | | | |

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
