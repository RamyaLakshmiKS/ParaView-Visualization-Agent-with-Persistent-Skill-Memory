# CLM — Context Language Models in this project

A self-contained implementation of the **"context as a file"** mechanism from
**Context Language Models (CLMs)**, Shao et al., *arXiv:2609.37725* (Meta
Superintelligence Labs + University of Washington), applied to this ParaView
visualization agent.

Everything here lives under `CLM/` and does **not** modify the rest of the
project. It is standard-library-only Python (no VTK, no data files needed to run
the demo).

---

## 1. The idea in one paragraph

A normal language model can only **append** to its context — it accumulates every
tool dump and never cleans up. A **CLM** instead treats its context as a **file it
may freely rewrite**: delete stale lines, compress a raw histogram dump into a
one-line fact, reorder by importance. The paper shows this yields **better task
outcomes at lower compute**, because the model carries a smaller, sharper working
context. This folder reproduces that mechanism for a volume-rendering task and
measures the size reduction.

---

## 2. How this maps to the paper

| Paper concept (arXiv:2609.37725) | Here |
|---|---|
| Context as a file the model edits (§4.1) | `context_store.py` — a working-context Markdown file with `set`/`append`/`discard`/`compress` operations |
| "Unrestricted update" `c_{t+1} = f(c_t)` (rewrite, not just append) | `WorkingContext.set_section` / `discard` / `compress_fact` (rewrite) vs `append` (the naive-only op) |
| Agent natively manages its own context | `skills/manage-context/SKILL.md` — a skill the agent runs *during* a task |
| Efficiency: better results at fewer FLOPs | `clm_metrics.py` — peak/final working-context size (a **local proxy** for FLOPs; see §5) |
| Qualitative "creative compaction" behavior | `demo.py` — raw inspect + histogram dumps collapse into resolved facts |

### What is intentionally **out of scope**

The paper has two further contributions this project **cannot** faithfully
implement, and we do not fake them:

- **Online reinforcement learning (stepwise GRPO).** Requires updating model
  weights. This project runs on a hosted agent CLI with no weight access.
- **Suffix Cache Reuse serving.** Requires modifying an inference server (SGLang).
  This project has no custom serving layer.

Both are legitimate scope boundaries, stated the same way the main project flags
its heuristic-metric limits.

---

## 3. CLM working context vs the project's `memory/memory.json`

These solve **different** problems and both are kept:

| | `memory/memory.json` (existing) | `CLM/context/working.md` (this folder) |
|---|---|---|
| Lifetime | **Durable**, across tasks and restarts | **Disposable**, one task |
| Holds | Distilled, transferable *lessons* | Live *working state* for the current render |
| Written by | `reflect-and-save` at task END | `manage-context` *during* the task |
| Read by | `recall-memory` at task START | the agent every turn |

In short: `memory.json` is long-term memory; the CLM file is short-term working
memory. The CLM skill explicitly forbids putting cross-task lessons in the working
file, and vice versa.

---

## 4. Usage

Run the demo (simulates the `mri_woman` render task twice — naive-append vs
context-as-a-file — no VTK/data needed):

```bash
cd CLM
python3 demo.py
python3 clm_metrics.py --compare context/naive.json context/clm.json
```

Drive the working-context file directly (what the `manage-context` skill does):

```bash
cd CLM
python3 context_store.py init --goal "render engine.raw, faint shell"
python3 context_store.py append "Resolved facts" "range [0,255] uint8, cutoff ~16%"
python3 context_store.py size      # -> lines/chars/est_tokens and budget status
python3 context_store.py show
```

Use the skill from an agent CLI (Kiro CLI / Claude Code), *during* a task, after
each inspect/render step:

```
/manage-context render mri_woman.raw, dense tissue stands out
```

> Note: `demo.py` imports `context_store` directly, so run it from inside `CLM/`.

---

## 5. Verified result (from `demo.py` + `clm_metrics.py`)

On the simulated `mri_woman` task, both agents do identical work; only context
management differs:

```
Per-step working-context size (est. tokens):
  step                           naive     CLM
  inspect .raw                     201      78
  histogram                        309     104
  render #1 + look                 345     164
  render #2 + look                 390     170
```

```
== CLM vs naive-append ==
  peak size   : naive 390 tok -> CLM 170 tok (56.4% smaller)
  final size  : naive 390 tok -> CLM 170 tok (56.4% smaller)
  CLM: 62 of 71 written lines discarded; retention ratio 0.141
```

The CLM agent holds a **~56% smaller** working context while keeping every fact
that drives the render (resolved dtype/shape, scalar range, background cutoff,
current transfer function). That mirrors the paper's "smaller working context →
less compute" claim.

### Honest proxy caveat

Working-context **size** is a *local proxy* for compute, **not** a FLOPs
measurement. We cannot count real server FLOPs here (no weight access, no custom
serving). The proxy is directionally meaningful — Transformer attention cost grows
super-linearly with context length, so a smaller working context does imply less
compute — but exact FLOPs/latency depend on the model and server and are out of
scope. This is the same spirit of disclosure as the main project's note that its
transcript metrics are weak heuristics.

---

## 6. Long-horizon test on real data (`longhorizon.py`)

The CLM benefit is strongest on **long-horizon** work. `longhorizon.py` processes
**every** file in `data/*.raw` in **one continuous session**, two ways — naive
(one context that grows across all files) vs CLM (compress each file to a resolved
fact and discard the raw material) — and shows the gap **widening as the horizon
grows**. It does REAL inspection of each `.raw` (infers dtype/dims from the byte
count, measures the real scalar range) and can optionally write real VTK renders.

```bash
cd CLM
python3 longhorizon.py            # all data/*.raw (no VTK needed)
python3 longhorizon.py --render   # also write results/<name>_longhorizon.png where dims known
python3 clm_metrics.py --compare \
    context/longhorizon_naive.json context/longhorizon_clm.json
```

**Verified result on the 3 shipped volumes** (real byte sizes + measured ranges:
bonsai uint8 256³ [0,255], engine uint8 256×256×128 [0,255], mri_woman uint16-le
256×256×109 [2,4095]):

```
Working-context size after each file (est. tokens) — the growth curve:
  after file           naive       CLM   gap x
  bonsai                 452        82    5.5x
  engine                 871       112    7.8x
  mri_woman             1339       145    9.2x

End of horizon: naive 1339 tok vs CLM 145 tok -> CLM is 89.2% smaller, 9.2x less
```

The key point: **naive grows ~linearly with the number of files while CLM stays
nearly flat.** After bonsai the CLM context is 5.5× smaller; after three files it
is 9.2× smaller. **Add more `.raw` files to `data/` and re-run — the gap keeps
widening**, which is exactly the long-horizon behavior the paper reports.

Any new file is handled automatically: if its `(dtype, dims)` is not in
`KNOWN_LAYOUTS`, the runner infers candidates from the byte count (assuming
256×256×depth) and measures the real range; `--render` is skipped only when the
dims are unknown or VTK is unavailable.

---

## 7. Files

```
CLM/
├── context_store.py          # the context-as-a-file mechanism (API + CLI)
├── clm_metrics.py            # working-context size metrics (efficiency proxy)
├── demo.py                   # naive-append vs CLM on a simulated render task
├── longhorizon.py            # LONG-HORIZON test: all data/*.raw in one session
├── skills/
│   └── manage-context/
│       └── SKILL.md          # the "manage your own context" agent skill
├── context/                  # generated working files + traces (demo output)
└── README.md                 # this file
```
