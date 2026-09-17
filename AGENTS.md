# AGENTS.md

Rules for anyone — human or agent — working in this repo. These are requirements, not suggestions. If a rule blocks you, say so and stop; do not route around it.

---

## 1. Production quality

This is a research project that produces a deployed artifact. Research sloppiness is allowed in exploration and nowhere else.

- Code that produces a number anyone will act on is production code. It lives in `src/`, it is committed, and it runs from a config.
- No silent failure. If a chunk fails to parse, a teacher call errors, or a file is missing, the run records it and the count appears in the results file. A pipeline that quietly drops 8% of its input and reports clean metrics is the worst possible outcome.
- No partial merges. A change that leaves the repo in a state where `scripts/` entry points don't run is not done.

## 2. Notebooks vs modules

Notebooks are for two things only:

1. **Exploration** — looking at data, sanity-checking a prompt, plotting a distribution.
2. **Kaggle execution** — the training notebook is the deployment surface for Kaggle, because Kaggle runs notebooks.

Everything else is a module.

- The Kaggle training notebook is a **thin driver**: clone the repo, install, load a config, call into `src/kg_llm_tune`, write checkpoints. Training logic does not live in notebook cells.
- The moment a cell is copied into a second notebook, it graduates to `src/`. First copy is the trigger, not the third.
- Notebooks are committed with **outputs stripped**, except where an output is the evidence (a plot referenced in a doc). Strip them before commit; `.gitignore` covers checkpoints but not outputs.
- No notebook is ever cited as the source of a benchmark number. See §5.

## 3. Reproducibility

A result that cannot be regenerated is not a result.

- **Pin versions.** `requirements.txt` / `pyproject.toml` pin exact versions, including `transformers`, `torch`, `peft`, `trl`, and the constrained-decoding library. Kaggle's base image moves; pinning is what stops a silent behaviour change from being read as a training improvement.
- **Seed everything.** `random`, `numpy`, `torch`, and the dataloader. The seed is a config field, never a literal in code. Note in the results file that full determinism on GPU is not guaranteed even so.
- **Config-driven runs.** Every run is `script + config file`. No editing code to change a hyperparameter. The exact config used is copied into the run's output directory — not referenced by path, copied, because the file on disk will change.
- **No hard-coded paths.** Not `F:\...`, not `/kaggle/input/...`, not `C:\Users\...`. Paths come from config or environment. The same config must run locally and on Kaggle with only the path block changed.
- **Record the environment.** Every run writes `env.json`: git commit SHA, dirty-tree flag, python version, key package versions, GPU name, CUDA version. A run from a dirty tree is marked dirty and its numbers are provisional.

## 4. Experiment tracking

- Every run gets a directory: `runs/<YYYYMMDD-HHMMSS>-<short-name>/` containing `config.yaml`, `env.json`, `metrics.json`, `log.txt`, and any predictions dumped for error analysis.
- `metrics.json` is machine-readable and flat. It carries the metric values, the gold-set version/hash it was measured against, and the number of examples scored.
- Local tracking is files on disk first. If a tracker (W&B, TensorBoard) is added, it is a *mirror* — the files stay authoritative, because the tracker is an account that can be lost and the files are in the repo.
- Failed and aborted runs are kept and marked `status: failed` in `metrics.json`. Deleting failed runs is how a project accidentally reports only its lucky seeds.
- Run directories with weights are gitignored; `metrics.json`, `config.yaml`, and `env.json` are **committed**.

## 5. The benchmark-number rule

**No number enters a README, a doc, a commit message, a chat summary, or a decision unless it came from a committed results file, and the citation names that file.**

- Write `entity F1 0.81 (runs/20260920-sft-v3/metrics.json)`, not `entity F1 ~0.81`.
- Numbers quoted from outside literature are marked as such with their source, and are never mixed into a table of this project's own measurements without a column saying where each row came from.
- An estimate is written as an estimate, with the word "estimate" in it and the basis stated. "~80 days at Groq free tier (estimate: daily cap ÷ 20k examples)" is acceptable. "About 80 days" is not.
- If you do not have the number, say you do not have the number. Do not interpolate, do not infer it from a related metric, do not carry it forward from a previous project.

This rule exists because a plausible fabricated metric is more expensive than no metric — it gets planned against.

## 6. Data provenance and licensing

- Every dataset used gets a row in `data/README.md`: source URL, licence, date pulled, and what it is used for.
- **CC-BY-SA-4.0 is share-alike.** `Babelscape/rebel-dataset` is CC-BY-SA-4.0. Training data derived from it, and arguably a model trained on it, carries obligations. Do not put it in the SFT mix until the repo licence question is settled.
- Teacher-generated data inherits the teacher's terms of service. Before a teacher model is used to generate training data, its ToS must be checked for output-use restrictions, and the finding recorded in `data/README.md`. This is an open question for the OmniRouter endpoint — see `docs/BLOCKERS.md`.
- Kartik's own corpus is the gold set's source and may be subject to employer restrictions. Do not upload any of it anywhere — including Kaggle — until that policy question is answered.
- Gold-set annotations are committed. They are hand-made, they are small, and they are the most valuable artifact in the repo.

## 7. Training workflow — local first, Kaggle for real

**Local LoRA on the 1650 Ti is a first-class part of the loop, not a fallback.** LoRA on SmolLM2-360M is expected around 1.0–1.5 GB, and LoRA on Qwen3-0.6B is ~2.0–3.0 GB; both fit the 3.4 GB card. Use it for: overfit-a-tiny-batch checks, pipeline debugging, the checkpoint-and-resume test, hyperparameter sanity, and 1–5k pilot runs.

**Kaggle is for the real runs** — full fine-tune and anything that produces a gate number. The 360M full-FT footprint is not measured yet; Kaggle remains the default until a committed memory test proves local full FT is viable. The local card is an estimated 25–50× slower than a 4090, so a 4–8 h T4 run is weeks locally.

Rules that follow from this:

- **Never report a local LoRA result against a gate threshold.** Gate numbers come from the Kaggle full fine-tune. A LoRA pilot number is a diagnostic and is labelled as one.
- **Real runs are full fine-tune because this is task shift** — new output format, new behaviour — which is the regime where the LoRA-Learns-Less evidence favours full FT. LoRA is the iteration tool, not the shipping recipe.
- **Local training must set the attention backend explicitly** — PyTorch SDPA memory-efficient or xformers — and assert it is active. SM 7.5 has no FlashAttention-2; naive attention at seq 4096 is ~537 MB per layer, and extraction prompts are long. A silent fallback presents as an OOM that looks like "model too big."
- **Plain LoRA, not QLoRA, locally.** 4-bit saves ~540 MB on a 0.72 GB base and costs dequant overhead. Not worth it below ~7B.
- **Exercise the resume path locally** before the first Kaggle long run.

### Kaggle specifics

Kaggle sessions are time-boxed and can die. Design for that from the first run, not after losing one.

**The notebook is generated from locally-validated code, never written independently.** Validate the training loop locally with LoRA on a small subset first; the Kaggle notebook is then produced from that same code. Divergence between the local and Kaggle training paths is how these projects break — two separately-written loops drift on a tokenizer flag or a collator, and the bug surfaces as "the Kaggle run scored worse" rather than as an error.

**Sessions must complete inside the limit by design.** Target a graceful stop at **8–9 hours** against Kaggle's cap, triggered by an **elapsed-time watchdog checked inside the training loop** — not by estimating up front that the run will fit.

**The shutdown sequence on a time-limit stop is fixed:** finish the current step → save weights → save optimizer state → save scheduler state → save RNG state → save step/epoch counters → write a **resume manifest** → **exit cleanly**, so `/kaggle/working` is preserved. An unclean exit can lose the outputs, which turns a graceful stop into a lost session anyway.

- **Checkpoint-and-resume is a first-class requirement from day one, not a nice-to-have.** A run must resume from the last checkpoint with a single config flag. Tested by deliberately killing a short run and resuming it — locally, before the first long Kaggle run.
- **Resume must continue the LR schedule and the data ordering, not silently restart them.** This is the classic bug and it deserves naming: a resumed run that restarts warmup or reshuffles from epoch zero produces a plausible loss curve and an **invalid result**. It does not crash and it does not warn — it quietly trains a different recipe than the one you believe you ran. Verify both the LR value and the data position continue, explicitly.
- **Checkpoint periodically as well as at the time limit**, so an unexpected disconnect costs at most N steps. The time-limit checkpoint handles the expected ending; periodic ones handle the unexpected.
- Save optimizer + scheduler + RNG state, not just weights. Weights-only checkpoints make resume a lie.
- Checkpoints go to Kaggle's persistent output, and are pulled down and stored deliberately. Nothing important survives only inside a running session.
- **fp16 + `GradScaler`.** The T4 is SM 7.5 and has no bf16. A config specifying bf16 is a bug, not a preference.
- The notebook pins the repo to a commit SHA, not a branch. A run that cloned `main` at an unknown time is not reproducible.
- Record the Kaggle GPU actually allocated in `env.json` — T4 vs P100 changes both speed and numerics.

## 8. Coding style

Kartik's stated preferences. Follow them.

- **Short, readable, human-traceable.** A reader should be able to follow a file top to bottom without jumping through four layers of indirection.
- **No docstrings.** Names carry the meaning. If a function needs a paragraph to explain it, it is the wrong function.
- **Comments explain *why*, not *what*.** `# T4 has no bf16` is a good comment. `# loop over chunks` is noise — delete it.
- **Minimal defensive abstraction.** No base classes with one subclass, no factory for two options, no config abstraction layer over the config file. Write the second implementation before generalising, not before the first.
- Fail loudly and early. No bare `except:`, no `except Exception: pass`. If a failure is expected and tolerable, catch that specific exception and *count* it.
- Type hints on module-level function signatures. Not inside function bodies.

## 9. Review bar

Before anything is called done:

- [ ] It runs end to end from a committed config, on a clean clone.
- [ ] Every number in the accompanying write-up cites a committed results file (§5).
- [ ] The run directory is committed (metrics/config/env), weights excluded.
- [ ] No hard-coded paths, no unpinned new dependency, seed is in the config.
- [ ] If it touches training: the resume path was exercised, not just written.
- [ ] If it touches evaluation: it was run against the gold set, and the gold-set hash is recorded.
- [ ] `docs/BLOCKERS.md` updated if the work opened or closed a blocker.
- [ ] Failure modes stated. "It worked" is not a result; "it worked, and here is what it got wrong" is.

## 10. For agents specifically

- Do not fabricate a metric, a file path, a citation, or a library API. If you need a number you do not have, say so.
- Do not mark a task complete because the code was written. Complete means it ran and the output was inspected.
- If an instruction in this repo conflicts with a request, surface the conflict rather than silently picking one.
- Prefer deleting code to commenting it out. Git remembers.
- When you cannot do something because of an environment limitation, write it into `docs/BLOCKERS.md` rather than working around it silently. A silent workaround is a blocker that gets discovered three weeks later.
