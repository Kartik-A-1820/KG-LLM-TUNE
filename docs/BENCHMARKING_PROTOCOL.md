# Benchmarking Protocol

This project treats benchmark records as product evidence. A run is not evidence
until its configuration, environment, and metrics are written to disk and
committed.

## Required files

Every stage run writes a directory:

```text
runs/<YYYYMMDD-HHMMSS>-<short-name>/
  config.yaml
  env.json
  metrics.json
  log.txt
```

Predictions, verifier verdicts, and error-analysis files may also be written
there. Weights, checkpoints, datasets, caches, and corpus material are not
committed.

The three committed files are mandatory:

- `config.yaml`: exact run configuration copied into the run directory.
- `env.json`: git SHA, dirty-tree flag, Python, package versions, GPU, CUDA,
  and hardware memory facts available at run start.
- `metrics.json`: flat machine-readable results, including `status`, examples
  counted, source data hashes, elapsed time, memory metrics, and task metrics.

`log.txt` is also required for new runs. If an old run predates this protocol,
mark that caveat in `docs/RUN_LEDGER.md` instead of reconstructing a log after
the fact.

## Benchmark-number rule

No number enters a README, doc, commit message, chat summary, or decision unless
it cites a committed results file. The citation must name the file.

Use this form:

```text
final_val_loss 0.7843165993690491
(runs/20260918-080000-smollm2-docred-lora-smoke/metrics.json)
```

Do not round a metric differently from the source file unless the rounded value
is clearly marked as a display rounding. Do not infer one metric from another.
Do not compare runs unless both runs have committed metrics.

External literature numbers are allowed only when marked as external and kept
separate from this project's measured results.

## Local GPU records

Local GPU runs must record:

- GPU name and CUDA availability in `env.json`
- CUDA and PyTorch versions in `env.json`
- total and free CUDA memory at start when available
- peak CUDA memory allocated in `metrics.json`
- elapsed wall time in `metrics.json`
- train/validation example counts in `metrics.json`
- seed, batch size, accumulation steps, context length, and model id

Local LoRA results are diagnostics. They can prove that the data path, memory
envelope, and loss movement are sane. They cannot be reported against Gate 0,
Gate 1, Gate 2, or Gate 3 thresholds.

## Stage documentation

After every stage run, update `docs/RUN_LEDGER.md` in the same commit as the
run files. The ledger entry must state:

- stage and purpose
- run directory
- model
- dataset/source and source hashes
- hardware
- key metrics
- decision or next action
- caveats, including dirty tree, missing log, non-gate status, or dataset limits

For an improvement claim, cite both the baseline and candidate run directories
and name the comparison axis. Examples:

- valid: same model, same data, same seed, different max length
- valid: same data and eval, SmolLM2-360M vs Qwen3-0.6B
- invalid: DocRED smoke loss compared to gold-set F1
- invalid: local LoRA loss compared to a Kaggle full-FT gate threshold

## Failure records

Failed, aborted, or OOM runs are evidence too. Keep their directories and write
`status: failed` or `status: aborted` in `metrics.json`, plus the failure mode
and the count of examples processed before failure.

Deleting failed runs biases the project toward lucky seeds and hides capacity
limits.
