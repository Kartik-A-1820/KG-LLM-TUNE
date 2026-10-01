# Kaggle Resume Package

The local diagnostic run is paused at optimizer step 400. Use
`notebooks/kuzu_graphrag_qlora_resume.ipynb` to continue it on Kaggle. This
package contains prepared data and resumable adapter state; it does not contain
the base SmolLM2 weights or any raw source datasets.

## Upload Once

Create one **private** Kaggle Dataset and upload this directory structure:

```text
kg-llm-tune-pilot/
  kuzu_graphrag_v1/
    train.jsonl
    val.jsonl
    manifest.json
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
  README.md (optional)
```

The data files come from `data/open_pilots/kuzu_graphrag_v1/`. The resume files
come from `runs/20260930-224836-smollm2-kuzu-5k-r16-4096-qlora/`. Upload every
file under `step-00000400/adapter/` if that exact file list differs. The
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

After a session pauses, save a Kaggle notebook version so its
`/kaggle/working/kaggle_resume_bundle` output is retained. Attach that notebook
output as an input on the next session, alongside the original private data
dataset. The notebook searches attached inputs for complete checkpoints,
validates the checkpoint's training and validation data hashes, and resumes
from the highest complete matching optimizer step. Each session writes the
next bundle under `/kaggle/working/kaggle_resume_bundle`.

Kaggle documents a 12-hour maximum for CPU/GPU notebook sessions and up to 20
GB of saved `/kaggle/working` output. The notebook enforces a shorter 10-hour
budget, reserves time for startup, and checkpoints on the first optimizer step
at or after its deadline. The watchdog starts on the first executed notebook
cell, so select the GPU and start running promptly. See the [Kaggle notebook
runtime documentation](https://www.kaggle.com/docs/notebooks).

This checkpoint is a diagnostic run, not a production-ready Kuzu model. Its
5,000 training rows combine 3,500 DocRED graph-extraction rows and 1,500
Neo4j Text2Cypher rows; validation has 350 and 150 rows respectively. The
Cypher examples have not been validated against Kuzu, and this mix does not
cover reference-grounded QA, summarization, or every GraphRAG capability. Do
not interpret language-model loss as a task-quality or gate result.
