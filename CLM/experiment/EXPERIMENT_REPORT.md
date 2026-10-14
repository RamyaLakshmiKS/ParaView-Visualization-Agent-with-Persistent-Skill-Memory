# Does Context-as-a-File Work? A Behavioral Experiment on a Volume-Visualization Agent

**An empirical test of the "context as a file" method from *Context Language Models*
(Shao et al., arXiv:2609.37725), applied to our ParaView volume-rendering use case.**

---

## TL;DR

We gave an LLM agent its working context as an **editable file it manages with
arbitrary Bash**, exactly as the CLM paper's Section 4.1 describes, and tested it
against a plain **prompt-only** agent on tasks built from our real `data/` volumes.
With a deliberately small 2000-token live-context budget:

- When the evidence **fit** the budget, both agents scored **100%** — context
  management was unnecessary and made no difference.
- When the evidence **overflowed** the budget (all 11 volumes at once), the
  Context-as-a-File agent scored **100%** while the prompt-only agent scored
  **0%** — because the baseline must truncate the evidence and literally cannot
  see the facts it needs.
- The agent's management behavior was **adaptive, not scripted**: it edited its
  context file only after it went over budget, digested 11 bulky inspection dumps
  into a handful of resolved facts, **rejected planted distractor lines**, and
  **preserved the required "needle" facts** (3/3).

**Verdict for our use case: the method works, and its benefit appears precisely
where the paper says it should — on long/over-budget tasks.** It is not magic on
tasks that already fit.

---

## 1. What the paper's method actually is

From §4.1 of the paper (quoted mechanism): the model's live context is *mirrored
to an editable file whose path is in the system prompt*; the model *edits that
file using general Bash commands*; *every edit is synchronized back into the live
context for the next turn*; and *when the model does not edit the file, its output
is appended by default*. Formally the paper writes the context transition as
`c_{t+1} = f_θ(c_t)` — an **arbitrary, model-controlled** update, in contrast to a
standard LM that can only append (`c_{t+1} = c_t ⊕ f_θ(c_t)`).

The crucial property — and the paper's central claim — is **autonomy**: the model
decides *what* to keep, *when* to compact, and *how* to organize. The paper
explicitly positions this against prior systems that expose a *fixed* set of
context operations.

**What we reproduced:** the zero-shot context-as-a-file mechanism with a real
model in the loop (Groq `qwen/qwen3.8-27b`, a substitute for the paper's unreleased
Qwen3.6-27B). **What we did not do** (and do not claim): the paper's reinforcement
learning and Suffix-Cache-Reuse serving contributions, and any FLOPs/efficiency
measurement — out of scope for this behavioral study.

---

## 2. Experimental design

### 2.1 The agent (faithful to §4.1)

- The agent's live context each turn = system prompt + the current contents of a
  file `CTX.md` + a short rolling window of recent raw tool output.
- The agent acts by emitting **one Bash block** (read evidence, edit `CTX.md`,
  anything) **or** a `FINAL:` answer.
- Any edit to `CTX.md` is re-injected as context next turn (auto-sync).
- The system prompt describes **only the mechanism and the budget** — never a
  procedure. We never tell it to summarize, compact, or retrieve. That preserves
  the autonomy the research question is about.
- Baseline = **prompt-only**: all evidence concatenated into the prompt, truncated
  to the same 2000-token budget, no file, no tools. This is the "just put the
  context in the prompt" comparison.

### 2.2 Tasks — built from our real volume data

Every task uses **real, measured metadata** from `data/*.raw` (dtype, dimensions,
byte count, and scalar range read from the actual bytes), wrapped in bulky,
realistic inspection dumps. Difficulty rises with context pressure:

![Context pressure by task: T1 fits the 2000-token budget; T2, T3, and A_native overflow it.](figs/budget_pressure.png)

*Figure 1 — Context pressure per task. Only T1 fits the live-context budget; the
other three exceed it, forcing the agent to manage information rather than hold it
all at once. (Sizes measured from the task evidence files.)*

| Task | Files | Over budget? | What it tests |
|---|---|---|---|
| **T1_single** | 1 | No | Control — can it answer; does it correctly *not* over-manage? |
| **T2_multi** | 4 | Borderline | Mild multi-source combination near the budget edge |
| **T3_revisit_distractor** | 11 | Yes (~4×) | Over budget **+ planted distractors to reject + forced revisit** of an early fact |
| **A_native_metadata** | 11 | Yes (~4×) | The **real pipeline question** ("which volumes are 16-bit, with ranges"), clean |

Each task has a **deterministic answer key** derived from the real bytes (e.g. the
true number of 16-bit volumes is **4**: backpack, beechnut, bunny, mri_woman).

### 2.3 Protocol

- Model: `qwen/qwen3.8-27b` via Groq. Live-context budget: **2000 tokens** (chosen
  so the 11-file tasks genuinely overflow). Max 16 turns.
- **Seeds:** 3 for T1/T2/T3; **2 for A_native** (third seed intentionally skipped
  to reduce scope — the two agree and T3 is the primary overflow test).
- Exact live-context size measured each turn from Groq's `usage.prompt_tokens`.

---

## 3. Results

### 3.1 Correctness — the headline

![Correctness by task: CLM and baseline both 100% on T1 and T2; on T3 and A_native (shaded, overflow region) CLM is 100% and baseline is 0%.](figs/correctness_by_task.png)

*Figure 2 — Answer correctness, CLM vs prompt-only, by task. On tasks that fit the
budget (T1, T2) the two methods tie at 100%. On tasks that overflow the budget (T3,
A_native — shaded) CLM holds at 100% while the prompt-only baseline collapses to 0%.*

| Task | n | CLM correct | Baseline correct | Baseline had to truncate |
|---|---|---|---|---|
| T1_single (fits) | 3 | **3/3** | 3/3 | 0/3 |
| T2_multi (near) | 3 | **3/3** | 3/3 | 3/3 |
| T3_revisit_distractor (overflow) | 3 | **3/3** | **0/3** | 3/3 |
| A_native_metadata (overflow) | 2 | **2/2** | **0/2** | 2/2 |

### 3.2 Why the baseline fails — a diagnostic failure mode

On the overflow tasks, the prompt-only baseline must truncate the evidence to fit
2000 tokens. It then answers from the fragment it can see and reports
**`count_16bit=2`** — because truncation cuts off the later 16-bit volumes
(bunny, mri_woman) before it ever reads them. The true answer is **4**. The CLM
agent, reading files on demand instead of cramming them in, reports **4** every
time. This is not a subtle quality gap; the baseline is *structurally blind* to
evidence it never loaded.

### 3.3 The context-management behavior, turn by turn

![Live-context token trajectory per turn for T3 and A_native, with the 2000-token budget line and context-edit markers.](figs/context_trajectory.png)

*Figure 3 — CLM live-context size per turn on the two overflow tasks (exact tokens
from Groq usage). Every run climbs past the 2000-token budget (red dashed line) as
it reads evidence. Circled points are turns where the agent edited `CTX.md`. In
T3 seed 2, the edit at turn 4 visibly drops the live context from ~3800 back to
~2680 — the agent compacting under pressure.*

### 3.4 Observable behavior matrix

![Heatmap of observable behaviors per task: over-budget, edited CTX.md, token-reducing edit, evidence retrieval, needle survival.](figs/behavior_matrix.png)

*Figure 4 — Observable context-management behavior (averaged over seeds). Darker =
higher rate. Reading across the rows shows the behavior scaling with task pressure.*

Reading Figure 4 honestly:

- **"went over budget"** is 0 on T1 (fits) and 1.0 on every overflow task — the
  pressure is real and measured.
- **"edited CTX.md"** is **0 on T1** (it correctly didn't bother) and **0.67 on
  T3** (it edited when pressured). This is the adaptive signal: *the behavior
  tracks the task state.*
- **"evidence retrieval"** is 1.0 on the overflow tasks (the agent globbed over
  all evidence files) and lower on the trivial ones (it only needed one file).
- **"needle survival"** is 3/3 on T3 — when it compacted, the required facts
  survived.
- **Honest nuance:** on A_native the agent went over budget but **did not edit
  CTX.md** (0.0). It still answered correctly by reading on demand within the
  rolling window. So "over budget" does **not** force an edit every time — the
  agent sometimes solves the task without rewriting its file. We report this
  rather than smooth it over.

### 3.5 A concrete artifact: the agent's own compaction

This is the final `CTX.md` the agent wrote on T3 seed 2 — it digested 11 bulky
inspection dumps into four lines and **explicitly rejected the distractors**:

```markdown
## Task
Count 16-bit volumes; report resolved dtype/dims/voxels of aneurism_256x256x256_uint8.

## Findings (from RESOLVED lines, ignoring UNVERIFIED/DEPRECATED)
16-bit volumes: backpack, beechnut, bunny, mri_woman => 4
aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216

## Answer
count_16bit=4; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216
```

The phrase *"ignoring UNVERIFIED/DEPRECATED"* is the agent's own — we never told it
distractors existed in those words. That is emergent, task-appropriate
context curation (compact + organize + forget-selectively + preserve-critical),
not a hardcoded step.

---

## 4. Answering the research questions

| Research question | Finding | Evidence |
|---|---|---|
| **Q1. Solve tasks without all info in-window?** | **Yes.** Correct on both over-budget tasks where evidence can't all fit. | Fig 2; T3 3/3, A_native 2/2 |
| **Q2. How does it decide what/when to retrieve?** | Reads on demand (globs/greps evidence), not exhaustively; answers once it has the needed facts in 2–6 turns. | Traces; Fig 4 retrieval |
| **Q3. Does it independently summarize/compact/organize?** | **Yes, when pressured.** It rewrote CTX.md into a digest with sections, unprompted. | §3.5 artifact; Fig 4 "edited CTX.md" |
| **Q4. Preserve critical details across steps?** | **Yes.** Needle survival 3/3 on T3; distractors rejected. | Fig 4; §3.5 |
| **Q5. Strategy changes with complexity?** | **Yes.** 0 edits on the task that fits, edits on the task that overflows. | Fig 4 (T1 vs T3 "edited" row) |
| **Q6. Better than context-in-prompt?** | **Yes, when it overflows; tie when it fits.** | Fig 2 |

---

## 5. Did it work? — interpretation

**Yes, for our use case, and in exactly the regime the paper predicts.** The
value of treating context as a file is **not** that it makes easy tasks better —
on T1/T2 it tied the baseline. The value shows up the moment the task's evidence
exceeds what the model can hold at once: the prompt-only agent is forced to throw
information away and answers wrong; the context-as-a-file agent chooses what to
read and keep, and answers right.

Crucially, we did not hand it a strategy. The agent **decided** to compact only
under budget pressure, **chose** to read evidence on demand, and **curated** out
the distractors on its own. That is the "model natively manages its own context"
behavior the paper is about — observed here on a different domain (volume
metadata) and a different, smaller model than the paper used.

---

## 6. Honest limitations

1. **Substitute model.** `qwen/qwen3.8-27b` stands in for the paper's unreleased
   Qwen3.6-27B. Results are model-dependent; the paper itself notes CLM gains grow
   with model capability.
2. **Small scale.** 4 tasks, 2–3 seeds each. A demonstration of the effect, not a
   statistically powered benchmark.
3. **Domain reframing.** Volume rendering has no large text corpus to manage, so we
   tested the method on *metadata QA + planted structure* over the real volumes —
   faithful to the method's shape (ContextBench-style), but an extension of the
   paper's own task set, not a literal reproduction of it.
4. **Heuristic behavior metrics.** `evidence_retrieval` credits a glob over
   `evidence/*` as full access; `token-reducing edit` counts an edit followed by a
   lower measured live-context size. The **trustworthy** evidence is answer
   correctness and the `CTX.md` diffs/snapshots, not these derived counters.
5. **Behavior is not "learning."** The model's weights never change. We observed
   adaptive *in-context* behavior, not cross-task learning. We make no learning
   claim; that is the paper's separate RL contribution, which we did not reproduce.
6. **No efficiency/FLOPs claim** of any kind, by design.

---

## 7. Reproduce it

```bash
# 1. build tasks from the real data/ volumes (deterministic)
python3 CLM/experiment/make_tasks.py

# 2. run one condition on one task (needs GROQ_API_KEY in repo-root .env)
python3 CLM/experiment/runner.py --task T3_revisit_distractor --condition clm   --seed 0 --budget 2000
python3 CLM/experiment/runner.py --task T3_revisit_distractor --condition baseline --seed 0 --budget 2000

# 3. full A/B sweep + grading + behavior
python3 CLM/experiment/evaluate.py --seeds 0 1 2 --budget 2000

# 4. (offline) report + charts from completed runs — no API calls
python3 CLM/experiment/report_from_disk.py
python3 CLM/experiment/make_charts.py
```

Key per-run evidence (correctness, a full turn-by-turn trace, and the agent's own
`CTX.md`) is inlined in [`RESULTS.md`](RESULTS.md), so the findings are
self-contained. The raw `runs/` output can be regenerated with the reproduce
steps above and is not required to read the results.

---

*Model: Groq `qwen/qwen3.8-27b`. Method: Context-as-a-File (Shao et al.,
arXiv:2609.37725, §4.1). This is an independent experiment; it reproduces the
zero-shot context-as-a-file mechanism only, on our own data, and makes no FLOPs,
efficiency, or learning claims.*
