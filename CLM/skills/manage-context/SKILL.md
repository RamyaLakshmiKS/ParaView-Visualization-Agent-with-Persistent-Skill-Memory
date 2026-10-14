---
name: manage-context
description: Maintain a live working-context FILE during a long visualization task — the Context Language Models "context as a file" method. The agent freely rewrites it to keep only what matters. Use DURING a task (after each inspect/render step), not at start/end.
---
# Manage Context (Context-as-a-File)

Task context: $ARGUMENTS

Working file: `CLM/context/working.md`

This implements the core idea of **Context Language Models** (Shao et al.,
arXiv:2609.37725): instead of only *appending* to your context, treat your
working context as a FILE you may freely rewrite — delete stale lines, compress
raw tool dumps into one-line facts, reorder by importance — so your working
memory stays small and sharp as the task gets long.

This file is your SHORT-LIVED working memory for the CURRENT task. It is NOT the
long-term lesson store:
- `memory/memory.json`      = durable lessons across tasks (use recall/reflect).
- `CLM/context/working.md`  = disposable working memory for THIS task (this skill).

## When to run
Run this after each meaningful step: after inspecting the `.raw` file, and after
each render-and-look cycle. Each time, EDIT the file so it reflects only what you
still need.

## Procedure
1. If `CLM/context/working.md` does not exist, initialize it:
   `python CLM/context_store.py init --goal "$ARGUMENTS"`
2. Read the current file (`python CLM/context_store.py show`).
3. REWRITE it to keep only what matters now. Maintain these sections:
   - **## Goal** — the task in one line.
   - **## Resolved facts** — measured scalar range, dtype/shape, chosen background
     cutoff fraction, which band is dense. These are decisions, not raw dumps.
   - **## Current TF** — the transfer-function shape you are using right now
     (opacity/color control points as FRACTIONS of the measured range).
   - **## Open issues** — what the last PNG looked wrong, to fix next.
   - **## Discard log** — one line each: what you removed and why.
4. Check size: `python CLM/context_store.py size`. If it reports OVER BUDGET,
   compress further before continuing.

## Rules (the CLM behavior to exhibit)
- **Prefer REWRITING over appending.** A raw histogram dump becomes
  `range [2,4095], 12-bit, background below ~5%`. A long tool transcript becomes
  one resolved fact.
- **Compress, don't accumulate.** Keep the file under ~40 lines. When it grows,
  collapse resolved observations into the Resolved facts section and clear the
  raw material (log it in Discard log).
- **Keep only task-relevant state.** Delete anything that no longer affects the
  next render decision.
- **This file is disposable.** Durable, transferable lessons still go to
  `memory/memory.json` via `reflect-and-save` at the END of the task. Do not put
  cross-task lessons here, and do not put this task's scratch state there.
- **Never store secrets, API keys, full scripts, or raw byte dumps.**

## Why this helps (what the paper claims)
By managing its own context as a file, the agent holds a smaller, more relevant
working context — the paper reports better task outcomes at lower compute. Here
we track the analogue with `python CLM/clm_metrics.py` (peak/final working-context
size, retained vs discarded). Token counts are REAL (tiktoken `cl100k_base`, with
a chars/4 fallback when tiktoken is absent); working-context SIZE is a local proxy
for the paper's FLOPs axis — see CLM/README.md.
