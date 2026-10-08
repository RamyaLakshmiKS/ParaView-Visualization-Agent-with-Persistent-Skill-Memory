---
name: run-experiment
description: Run the persistent-memory A/B experiment for ANY dataset — solve it with memory ON vs OFF — and auto-print the comparison. Works for new datasets without editing this skill.
---
# Run Experiment: Persistent Memory ON vs OFF

Which dataset to TEST: $ARGUMENTS
If no argument is given, default TEST dataset = `engine`.

Goal: prove that lessons already in `memory/memory.json` (learned on earlier datasets)
help the agent on the TEST dataset, by solving it twice — once with memory, once cold —
and comparing. A Python harness (`experiment.py`) enforces the memory state so the OFF
run is genuinely cold.

This skill is dataset-generic. To test a NEW dataset, just make sure:
- `data/<name>.raw` exists, and
- `tasks/<name>.txt` exists (a short goal-only prompt).
Then invoke `/run-experiment <name>`. Nothing else needs editing.

Let T = the TEST dataset name from the argument (e.g. `mri_woman`).

## 0. Preconditions (run via shell)
```
test -f data/<T>.raw && test -f tasks/<T>.txt && echo "OK: data + task present" || echo "MISSING data/<T>.raw or tasks/<T>.txt"
```
Show the current memory so we know what lessons already exist:
```
python experiment.py status
```
Do NOT reset memory — the whole point is to use lessons learned on earlier datasets.

## 1. Phase T-ON — solve T WITH memory
```
python experiment.py phase-begin --name B_<T>_on --mode on
```
Then, as the agent:
1. Invoke `/recall-memory <T>` — report the "Lessons I'm applying" it surfaces.
2. Do the task in `tasks/<T>.txt`. Inspect the data at runtime, render, LOOK at the
   PNG, fix issues, finish. Save the render to `results/<T>_on.png` (copy
   `results/<T>.png` to `results/<T>_on.png` when done).
3. Invoke `/reflect-and-save <T>`.
Save the session transcript to `experiment/phases/B_<T>_on/transcript.txt`.
```
python experiment.py phase-end --name B_<T>_on
```

## 2. Phase T-OFF — solve T COLD (control)
```
python experiment.py phase-begin --name B_<T>_off --mode off
```
Then — keep it genuinely cold:
- Do NOT invoke `/recall-memory` or `/reflect-and-save`.
- Just do `tasks/<T>.txt` as a fresh agent with no prior lessons. Save the render to
  `results/<T>_off.png`.
Save the transcript to `experiment/phases/B_<T>_off/transcript.txt`.
```
python experiment.py phase-end --name B_<T>_off
```

## 3. Auto-compare and export
```
python experiment.py compare --on B_<T>_on --off B_<T>_off
python experiment.py export-results
```
`export-results` regenerates `RESULTS.md` with a section for EVERY completed
ON/OFF dataset pair it finds under `experiment/phases/`, embedding the
`results/<T>_on.png` vs `results/<T>_off.png` images and the memory-growth evidence.

## 4. Report
Summarize: which recalled lesson helped on T, whether the OFF run showed a mistake the
ON run avoided (confirm by eye in the two PNGs), and the memory counts before/after.
Be honest: the image contrast + memory store are the hard evidence; transcript counts
are heuristic proxies.
