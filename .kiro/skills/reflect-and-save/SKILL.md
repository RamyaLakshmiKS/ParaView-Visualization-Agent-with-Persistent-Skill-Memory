---
name: reflect-and-save
description: After finishing a visualization task, extract lessons and append them to memory/memory.json. Use at the END of every task.
---
# Reflect and Save

Review what just happened on this task and update `memory/memory.json`.

1. Add ONE **episodic** record:
   `{ id, task, dataset_type, what_worked, params, outcome, timestamp }`
2. If a reusable rule emerged, add or strengthen a **semantic** record:
   `{ id, rule, evidence_ids, confidence }`. If the rule already exists, bump its
   `confidence` and append the new evidence id instead of duplicating it.
3. If a mistake occurred and was fixed, add or strengthen a **corrective** record:
   `{ id, mistake, fix, times_seen }`. If it already exists, increment `times_seen`.

Curation rules (do NOT bloat memory):
- Never duplicate an existing rule — strengthen it instead.
- Keep each record to one or two sentences.
- Store the LESSON, not transcripts, full scripts, or file dumps.
- Never store secrets or API keys.

Write valid JSON back to `memory/memory.json`, preserving the
`{"episodic": [...], "semantic": [...], "corrective": [...]}` structure.
