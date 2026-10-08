---
name: inspect-memory
description: Pretty-print the current contents of memory/memory.json for the live demo. Use to show the audience what the agent has learned.
---
# Inspect Memory

Read `memory/memory.json` and print a human-readable summary:

- **Episodic** (N records): one line each — task, outcome, key params.
- **Semantic** (N rules): one line each — the rule and its confidence.
- **Corrective** (N mistakes): one line each — the mistake and its fix.

If the store is empty, say so clearly. Do not modify the file.
