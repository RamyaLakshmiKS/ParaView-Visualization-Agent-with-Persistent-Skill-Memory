---
name: recall-memory
description: Load relevant past lessons from memory/memory.json before starting a visualization task. Use at the START of every task.
---
# Recall Memory

Task context: $ARGUMENTS

Before doing the task, read `memory/memory.json`.

Select and surface ONLY the memories relevant to this task:
- Match `dataset_type` and task keywords in $ARGUMENTS against episodic and semantic entries.
- Always include ALL `corrective` entries (mistakes are cheap to re-check).

Then print a short section titled **"Lessons I'm applying"** that lists:
1. Any semantic rules you will follow.
2. Any corrective mistakes you will avoid.
3. Any episodic parameters worth reusing (e.g. transfer-function shape, camera hints).

After that, proceed with the task using those lessons.

Reliability requirement (applies to ANY dataset):
Before building a transfer function, INSPECT the data at runtime — read its actual
scalar min/max (and shape if headerless). Derive opacity/color control points relative
to that measured range. Never assume a fixed range, dtype, or value just because a past
task used it. Recalled lessons are procedures to apply, not constants to copy.

If `memory/memory.json` is empty or missing, print **"No prior memory — solving from scratch"** and continue.
