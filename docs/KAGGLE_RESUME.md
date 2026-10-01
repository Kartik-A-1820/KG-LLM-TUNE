# Kaggle Resume Package

The local diagnostic run is paused at optimizer step 400. Use
`notebooks/kuzu_graphrag_qlora_resume.ipynb` to continue it on Kaggle. This
package contains prepared data and resumable adapter state; it does not contain
the base SmolLM2 weights or any raw source datasets.

## Upload Once

Create one **private** Kaggle Dataset. The three JSON files are already
uploaded to yours; add `kg-llm-tune-resume-step-00000400.zip` from
`D:\KG-LLM-TUNE-KAGGLE-UPLOAD\`. That archive contains this resume structure
(the dataset itself keeps `train.jsonl`, `val.jsonl`, and `manifest.json` at its
top level):

```text
kg-llm-tune-resume-step-00000400.zip
  resume_bundle/
    config.yaml
    env.json
    metrics.json
    log.txt
    checkpoints/
      step-00000400/
        state.pt
        manifest.json
        adapter/
          adapter_config.json
          adapter_model.safetensors
          README.md
```

The data files come from `data/open_pilots/kuzu_graphrag_v1/`. The resume files
come from `runs/20260930-224836-smollm2-kuzu-5k-r16-4096-qlora/`. The archive
contains every file under `step-00000400/adapter/`. The
essential training state is `state.pt`
(optimizer, scaler, RNG, epoch, batch position, and counters), together with
the adapter and both manifests. Config, environment, metrics, and log files are
required by the trainer's resume validation and preserve provenance.

Do **not** upload:

- SmolLM2 base model weights: the notebook fetches
  `HuggingFaceTB/SmolLM2-360M-Instruct` at revision
  `a10cc1512eabd3dde888204e902eca88bddb4951` from Hugging Face.
- Raw DocRED or Text2Cypher downloads, local virtual environment, pip cache,
  or the entire `runs/` directory.
- Kaggle credentials or any access token.

Enable Internet in the Kaggle notebook settings for cloning the pinned project
commit and fetching the base model. Select a GPU accelerator.

## Later Sessions

After each session pauses, the final notebook cell writes exactly one portable
artifact, `/kaggle/working/kg-llm-tune-resume-step-NNNNNNNN.zip`, where the
number is the saved optimizer step. It contains the run configuration,
environment, metrics and logs; adapter weights; checkpoint manifest; and
`state.pt` (optimizer/scaler, RNG states, epoch, next batch and counters). An
artifact manifest records every payload file's size and SHA-256, plus model and
source-commit identity. The notebook verifies the ZIP CRCs, required members,
and all payload hashes before reporting success. The ZIP is written atomically;
an incomplete archive is not published at the target path.

Use Kaggle **Save Version** after the notebook run to retain that ZIP in the
notebook output. For continuity independent of the notebook version, download
that single ZIP and add it to a new **private Kaggle Dataset**. On the next run,
attach that private resume dataset and the original private data dataset. The
notebook safely extracts every attached `kg-llm-tune-resume-*.zip`, validates
the train/validation hashes against the original dataset, and resumes from the
highest complete matching optimizer step. The newly produced ZIP is the sole
resume artifact to carry forward; no model weights or datasets need to be
committed to GitHub.

Kaggle documents a 12-hour maximum for CPU/GPU notebook sessions and up to 20
GB of saved `/kaggle/working` output. The notebook enforces a shorter 10-hour
budget, reserves time for startup, and checkpoints on the first optimizer step
at or after its deadline. The watchdog starts on the first executed notebook
cell, so select the GPU and start running promptly. Save Version must complete
after the final ZIP verification; check that the ZIP appears in notebook output
before ending the session. See the [Kaggle notebook
runtime documentation](https://www.kaggle.com/docs/notebooks).

This checkpoint is a diagnostic run, not a production-ready Kuzu model. Its
5,000 training rows combine 3,500 DocRED graph-extraction rows and 1,500
Neo4j Text2Cypher rows; validation has 350 and 150 rows respectively. The
Cypher examples have not been validated against Kuzu, and this mix does not
cover reference-grounded QA, summarization, or every GraphRAG capability. Do
not interpret language-model loss as a task-quality or gate result.
