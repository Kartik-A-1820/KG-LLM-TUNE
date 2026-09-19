# Blockers

Living document. Anything that stopped work goes here rather than being silently worked around. Update it when a blocker opens or closes — that is a line item on the `AGENTS.md` review bar.

Status as of **2026-09-17**. Scaffold created remotely; Kartik is away from the machine for ~2 weeks.

---

## B1 — Kaggle: not signed in, and the sign-in path has two separate problems

**Status:** open. Blocks all training.
**Severity:** high — Kaggle T4 is the only training environment in the plan.

This is not simply "not logged in". There are two distinct problems, and fixing one does not fix the other.

### B1a — Wrong Google account in the browser

The Google account signed into the browser is **`instagramy1820@gmail.com`** — the Instagram/YouTube account, not the primary one.

Signing into Kaggle with it would have either landed on the wrong Kaggle account or **created a new Kaggle account** under that address. Either outcome is worse than being blocked: a new account starts with no notebooks, no datasets, and a separate GPU quota, and an accidental second account is awkward to unwind later. Sign-in was therefore deliberately not attempted.

**Needs from Kartik:** confirm which Google identity the Kaggle account is actually under, and sign in with that one.

### B1b — The Google OAuth popup is not reachable by automation

"Sign in with Google" opens an **OAuth popup in a window outside the reachable tab group**. Browser automation here operates on a specific tab group; a popup window created outside it cannot be seen or driven.

So even with the correct account signed in, **the Google sign-in flow may not be drivable remotely at all**. This is a structural limitation, not a permissions issue, and it will recur every time the session expires.

**Possible routes when Kartik is back — in rough order of robustness:**

1. Sign into Kaggle **manually once** at the machine, in the browser used for automation, and let the session cookie persist. Simplest and most likely to work.
2. Use Kaggle's **email/password** login rather than Google OAuth, if the account has one set — that form is in-tab and drivable.
3. Use the **Kaggle API token** (`kaggle.json`) for notebook push/pull instead of the web UI. This sidesteps the browser entirely and is the right long-term answer for automated training runs — but obtaining the token requires a signed-in session once.

**Recommendation:** route 3, bootstrapped by route 1. Get the API token while at the machine; after that, training runs can be pushed without a browser at all.

---

## B2 — Local execution requires physical presence

**Status:** open until Kartik returns (~2 weeks).
**Severity:** medium — blocks execution, not planning.

- **Terminals are click-tier only** — clickable, but text cannot be typed into them. No git, no pip, no python.
- **Folder-access prompts time out** when Kartik is not at the machine. `F:` was declined twice earlier today before access was eventually granted; the prompt is interactive and will fail again if it lapses.

**Consequence:** no local run of anything. Gate 0 baselines, the embedding A/B, and all local evaluation wait for his return. Planning, documentation, and repo work proceed — which is what this scaffold is.

**Note:** F: access *did* succeed in the session that created the scaffold. **The project has since moved to `D:\KG-LLM-TUNE\` (2026-09-17, F: space constraints).** The D: copy is a fresh clone of `open-data-docred-pilot` and matches the remote. `F:\KG-LLM-TUNE\` was deleted after the D: checkout and venv were verified. Folder access is per-session and not a permanent grant; expect to re-request it.

---

## B3 — Linux sandbox broken

**Status:** open.
**Severity:** low for now, medium once data work starts.

The sandboxed Linux environment fails with `vm_bundles is a symlink or junction`. No shell, so no git operations, no scripted file generation, no local package installs from the assistant side.

**Consequence:** repo files are created through the GitHub web interface one at a time rather than with a single `git push`. Workable but slow, and it caps how much can be batched in one sitting.

**Fix:** unknown cause; likely environment-side. Worth retrying in a later session before assuming it is permanent.

---

## B4 — POLICY: can corpus data leave Impetus for Kaggle?

**Status:** open. **Only Kartik can answer this.**
**Severity:** high — this one can reshape the architecture, not just delay it.

Training runs on Kaggle. Kaggle is a third-party cloud service. The training data is derived from Kartik's own corpus. **If that corpus carries Impetus confidentiality or data-residency obligations, uploading it — or anything derived from it — to Kaggle may not be permissible.**

This is not a small compliance footnote. The whole Phase 1 plan assumes corpus-derived training data reaching a T4 on Kaggle.

**If the answer is no**, the plan changes in specific ways:

- Training data becomes public + synthetic only; the corpus is used for **evaluation only**, locally.
- Teacher distillation over corpus chunks becomes problematic too — the corpus would be going to a third-party API, which is arguably a bigger exposure than Kaggle.
- The gold set can stay local, but committing it to a **public** GitHub repo becomes a separate question with the same answer.
- Training hardware needs a rethink: a rented GPU under Kartik's control, or a much smaller local-feasible approach.

**Action:** answer this before any data collection begins. Answering it after the SFT set is built is the expensive ordering.

**Related, and worth deciding at the same time:** this repo is **public**. Anything committed to `eval/gold/` is world-readable. If the corpus is sensitive at all, either the gold set stays out of the repo or the repo goes private.

---

## B5 — POLICY: what does the OmniRouter endpoint route to?

**Status:** open. **Only Kartik can answer this.**
**Severity:** medium — affects teacher selection and ToS compliance.

The teacher-distillation plan needs to know which underlying providers the OmniRouter endpoint actually reaches. Two things depend on it:

1. **Terms of service.** Some providers restrict using model outputs to train competing models. Distillation is exactly that pattern. The restriction lives in the *underlying provider's* terms, which a router does not remove — so "we used a router" is not a defence. `AGENTS.md` §6 requires this finding be recorded in `data/README.md` before generation starts.
2. **Multi-teacher independence.** The agreement rule assumes three teachers **from different families**. If the router silently resolves two of the three to the same underlying model, or to two models from the same family, agreement becomes near-worthless — correlated models agree on their shared hallucinations, which is precisely what the rule is meant to filter.

**Action:** enumerate what OmniRouter actually routes to, check each provider's output-use terms, and record the findings. If routing is opaque or non-deterministic, use named providers directly for distillation instead — verifiable teacher identity is worth more here than routing convenience.

---

## Closed

*(nothing yet)*
