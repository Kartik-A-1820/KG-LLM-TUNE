# notebooks/

Notebooks exist here for exactly two reasons.

**1. Exploration.** Looking at data, sanity-checking a prompt, plotting a distribution, poking at a failing example. Throwaway by nature.

**2. Kaggle execution.** Kaggle runs notebooks, so the training notebook is the deployment surface for Kaggle. That is a platform constraint, not a design choice.

## Anything reusable graduates to `src/`

The trigger is the **first copy**, not the third. The moment a cell is pasted into a second notebook, it belongs in `src/kg_llm_tune/`.

The Kaggle training notebook is a **thin driver**:

```
clone repo at a pinned commit SHA  ->  install  ->  load config  ->
call into kg_llm_tune.train  ->  write checkpoints to persistent output
```

Training logic does not live in notebook cells. A notebook that contains the training loop cannot be reviewed, diffed, tested, or resumed reliably — and it cannot be reused for the next run without copy-paste drift.

## Rules

- **Strip outputs before committing**, except where an output is the evidence for something referenced in a doc.
- **No notebook is ever the source of a benchmark number.** Numbers come from committed results files (`AGENTS.md` §5). A notebook cell showing `f1: 0.83` is not a result.
- **Pin the repo to a commit SHA** in the Kaggle notebook, never a branch. A run that cloned `main` at an unknown time is not reproducible.
- **fp16 + `GradScaler`.** The Kaggle T4 is SM 7.5 — no bf16. Copying a bf16 recipe from a blog post is the most likely way this breaks.
- **Checkpoint-and-resume gets tested before the first long run**, by deliberately killing a short run and resuming it. Not after losing a session.

## Expected notebooks

| Notebook | Purpose | Runs on |
| --- | --- | --- |
| `00_corpus_explore.ipynb` | Look at the corpus, chunk-length distribution, document types | Local |
| `01_gold_set_annotation.ipynb` | Annotation helper — display chunk, capture spans | Local |
| `02_teacher_pilot.ipynb` | 500-chunk teacher pilot, inspect agreement rates before spending the budget | Local |
| `10_kaggle_sft.ipynb` | SFT training driver | Kaggle T4 |
| `11_kaggle_rejection_sampling.ipynb` | Rejection sampling + self-distillation driver | Kaggle T4 |
| `20_eval_gold.ipynb` | Gold-set evaluation and error analysis | Local |
| `21_embedding_ab.ipynb` | EmbeddingGemma vs potion-retrieval end-to-end A/B | Local |

None written yet.
