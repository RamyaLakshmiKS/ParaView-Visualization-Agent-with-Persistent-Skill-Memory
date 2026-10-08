# ParaView Visualization Agent with Persistent Skill Memory

A scientific-visualization agent that **learns across tasks**. After rendering one
volume dataset, the agent writes down what it learned into a persistent store on disk.
On a new, unseen dataset, it recalls those lessons and applies them — rendering with
fewer steps and without repeating past mistakes.

The key property: the memory is a **file on disk**, so it survives a full process
restart. Lessons learned in one session carry over to the next. This is cross-task,
persistent memory — not a within-session scratchpad.

---

## Why this is interesting

Most agents treat every task as brand new. This one accumulates experience:

- It **inspects** each dataset at runtime (actual scalar range, shape) instead of
  assuming fixed values, so it works on data it has never seen.
- It stores **transferable procedures**, not dataset-specific constants. A lesson like
  *"build the opacity ramp relative to the measured scalar range"* generalizes to any
  volume; *"opacity 0 at value 40"* would not, so it is never promoted to a reusable rule.
- It separates three kinds of memory, because "memory" is not one thing:
  - **Episodic** — concrete records of past attempts (what worked, reference params).
  - **Semantic** — distilled, reusable procedures generalized from episodes.
  - **Corrective** — mistakes not to repeat.

The payoff is measurable: run a new task with memory ON vs OFF and compare the number
of steps and errors.

---

## How it works

```
                 ┌──────────────────────────────┐
                 │  memory/memory.json          │  persistent store (survives restart)
                 │  { episodic, semantic,       │
                 │    corrective }              │
                 └──────────┬───────────────────┘
                   recall   │          │  reflect
                 ┌──────────▼──────────▼─────────┐
 task prompt ──▶ │  agent (Kiro CLI OR Claude    │ ──▶ results/<name>.png
                 │  Code + the SKILL.md skills)  │     results/<name>.py
                 │  /recall-memory               │
                 │  /reflect-and-save            │
                 └───────────────────────────────┘
```

The agent is driven by a small set of `SKILL.md` skills (reusable prompts invoked as
slash commands). They are plain Markdown + YAML frontmatter, so the **same files run
under Kiro CLI (`.kiro/skills/`) and under Claude Code (`.claude/skills/`)** — see
[Using these skills with Claude Code](#using-these-skills-with-claude-code).

| Skill | When | What it does |
|-------|------|--------------|
| `/recall-memory <task>`     | start of a task | Loads relevant lessons from `memory/memory.json`, prints "Lessons I'm applying", and requires inspecting the data before building a transfer function. |
| `/reflect-and-save <task>`  | end of a task   | Extracts lessons and appends them to `memory/memory.json`, keeping only transferable procedures. |
| `/inspect-memory`           | anytime         | Pretty-prints the current store (read-only). |
| `/run-experiment <dataset>` | anytime         | Runs the full memory **ON vs OFF** A/B experiment for any dataset and auto-writes the comparison into `RESULTS.md`. Dataset-generic — works on new datasets with no edits. |

The agent writes its own render script per task (`results/<name>.py`) using VTK — the
rendering code is not pre-supplied, so figuring out how to render correctly is part of
what gets learned and remembered.

---

## Repository layout

```
.kiro/skills/
  recall-memory/SKILL.md       # load relevant lessons at task start
  reflect-and-save/SKILL.md    # write lessons at task end (transferable only)
  inspect-memory/SKILL.md      # print the store (read-only)
  run-experiment/SKILL.md      # run the memory ON-vs-OFF A/B experiment for any dataset
memory/memory.json             # the persistent store (starts empty)
tasks/
  bonsai.txt                   # 256x256x256 uint8 volume
  engine.txt                   # 256x256x128 uint8 volume
  mri_woman.txt                # MRI volume (dtype/shape resolved by inspection)
data/                          # the .raw volumes (see "Getting the data")
experiment.py                  # harness that enforces memory ON/OFF and compares phases
experiment/phases/             # per-run snapshots (memory before/after, transcript)
results/                       # agent-written render scripts + PNGs (incl. <name>_on/_off)
RESULTS.md                     # auto-generated ON-vs-OFF comparison (one section per dataset)
smoke_test.py                  # headless-render environment check (no data needed)
metrics.py                     # turn a run transcript into a before/after table
requirements.txt
```

---

## Prerequisites

- Python 3.9+
- An agent CLI that reads `SKILL.md` skills. Either:
  - [Kiro CLI](https://github.com/aws/kiro-cli) — reads `.kiro/skills/*/SKILL.md` (used here), or
  - [Claude Code](https://docs.claude.com/en/docs/claude-code) — reads `.claude/skills/*/SKILL.md`
    (the same files; see [Using these skills with Claude Code](#using-these-skills-with-claude-code)).
- The Python packages in `requirements.txt` (`vtk`, `numpy`)

---

## Setup (reproducible from scratch)

```bash
# 1. clone and enter the repo
git clone <this-repo-url>
cd ParaView-Visualization-Agent-with-Persistent-Skill-Memory

# 2. create an isolated environment and install deps
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. verify headless rendering works (no dataset required)
python smoke_test.py                 # expect: "RENDER OK" and smoke_test.png
```

If `smoke_test.py` prints `RENDER OK` and writes `smoke_test.png`, your environment can
render volumes offscreen and you are ready to run the agent.

### Getting the data

The `.raw` volumes live in `data/`. `bonsai` and `engine` are classic open volume
datasets; `mri_woman` is an MRI volume. If `data/` is empty in your clone, obtain the
three volumes and place them as:

```
data/bonsai.raw       # 256 x 256 x 256, uint8   (16,777,216 bytes)
data/engine.raw       # 256 x 256 x 128, uint8   (8,388,608 bytes)
data/mri_woman.raw    # 256 x 256 x 109 uint16 OR 256 x 256 x 218 uint8 (14,286,848 bytes)
```

`.raw` files are headerless (raw bytes, no embedded dimensions or type), which is why
each task prompt states the shape/dtype or asks the agent to resolve it by inspection.

---

## Running the agent

Launch your agent CLI from the project root (so `.kiro/skills/` is picked up). Confirm
the skills are loaded — in Kiro CLI, `/context show` should list `recall-memory`,
`reflect-and-save`, and `inspect-memory`.

A single task looks like:

```
/recall-memory bonsai volume rendering
Do the task in tasks/bonsai.txt
/reflect-and-save bonsai
/inspect-memory
```

### Demonstrating cross-task memory

1. **Start empty.** `memory/memory.json` is `{"episodic":[],"semantic":[],"corrective":[]}`.
2. **Task A (memory ON).** Run `bonsai` as above. Open `memory/memory.json` — it now
   contains lessons.
3. **Restart the process.** Quit and relaunch the CLI (proves the memory is persistent,
   not in-session state).
4. **Task B (memory ON).** Run `engine`. `/recall-memory` surfaces the Task-A lessons;
   the agent applies them and avoids known mistakes.
5. **Task B (memory OFF, control).** Temporarily move `memory/memory.json` aside (or skip
   `/recall-memory`) and run `engine` cold.
6. **Compare.** Use `metrics.py` on each run's transcript to produce the table:

   ```bash
   python metrics.py --label "engine OFF" --transcript run_off.txt
   python metrics.py --label "engine ON"  --transcript run_on.txt
   ```

### Showing it generalizes (not overfit)

Run `mri_woman` with memory ON. It is a different modality with a different scalar range
and (likely) a different data type than bonsai/engine. Because the stored rules are
range-relative procedures and the agent inspects the data at runtime, the lessons still
help — evidence that the memory transfers rather than memorizes.

---

## Run your own experiment (`/run-experiment`)

`/run-experiment` runs the whole **memory ON vs OFF** A/B test for any dataset in one
command and writes the comparison into `RESULTS.md`. A Python harness (`experiment.py`)
enforces the memory state, so the OFF run is *genuinely cold* (your learned memory is
parked out of reach and restored afterward — the cold run can never read or pollute it).

The skill is **dataset-generic** — to test a brand-new dataset of your own, you only add
two files, no skill edits:

```bash
# 1. drop your headerless volume in data/
cp /path/to/your_volume.raw data/myvol.raw

# 2. write a short, goal-only task prompt (what to reveal, where to save)
cat > tasks/myvol.txt <<'EOF'
Task:
Render a volume visualization of data/myvol.raw.

Goal:
<describe what structure to reveal and which colors make it readable>

Save the image to results/myvol.png and the render script to results/myvol.py.
EOF
```

Then, from your agent CLI at the project root:

```
/run-experiment myvol
```

The skill will (all orchestrated for you):

1. Check `data/myvol.raw` + `tasks/myvol.txt` exist and print the current memory counts.
2. **Phase ON** — `/recall-memory`, solve the task (inspect → render → look → fix),
   `/reflect-and-save`, save `results/myvol_on.png`.
3. **Phase OFF** — solve it cold with no recalled lessons, save `results/myvol_off.png`.
4. **Compare + export** — regenerate `RESULTS.md` with a section for every ON/OFF pair,
   embedding the two images and the memory-growth evidence.

The hard evidence is the two PNGs side by side (does the ON render avoid a defect the
cold OFF render makes?) and the memory store; the transcript counts are heuristic
proxies.

If you don't pass a dataset name, the default TEST dataset is `engine`.

---

## Using these skills with Claude Code

The skills are plain `SKILL.md` files (YAML frontmatter + Markdown, using `$ARGUMENTS`),
which is exactly the format [Claude Code](https://docs.claude.com/en/docs/claude-code)
expects under `.claude/skills/`. To reuse the **same** skills in Claude Code, copy them
into that directory once:

```bash
# from the project root — mirror the Kiro skills into the Claude Code location
mkdir -p .claude/skills
cp -R .kiro/skills/* .claude/skills/
```

(macOS/Linux shown; on Windows PowerShell use
`Copy-Item -Recurse .kiro\skills\* .claude\skills\`.)

Then launch Claude Code from the project root and invoke them exactly as under Kiro CLI:

```
/recall-memory bonsai volume rendering
Do the task in tasks/bonsai.txt
/reflect-and-save bonsai
/inspect-memory
/run-experiment engine
```

Everything else — `memory/memory.json`, `experiment.py`, `tasks/`, `data/`, `results/`
— is agent-agnostic, so the persistent memory, the ON/OFF harness, and `RESULTS.md`
behave identically regardless of which CLI you drive the skills with. The repository
ships the `.kiro/skills/` copy; the one `cp` above is all that's needed to run the exact
same experiment under Claude Code.

---

## Resetting

To start fresh, reset the store and clear generated outputs:

```bash
echo '{"episodic":[],"semantic":[],"corrective":[]}' > memory/memory.json
rm -rf results/
```

---

## Notes

- Skills use the `.kiro/skills/*/SKILL.md` format. The same files also work with Claude
  Code under `.claude/skills/` — see
  [Using these skills with Claude Code](#using-these-skills-with-claude-code).
- Rendering is CPU/offscreen via VTK; no GPU is required.
- The agent, not this repo, writes each `results/<name>.py` render script — that code is
  part of what the agent learns to do well over time.
