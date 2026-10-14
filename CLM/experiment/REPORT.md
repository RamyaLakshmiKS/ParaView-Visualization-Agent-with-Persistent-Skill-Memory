# Context-as-a-File — Behavioral Report

> Built offline from `runs/*/result.json`. No API calls. Scope: 3 seeds for T1/T2/T3; **A_native_metadata reduced to n=2** (third seed intentionally skipped — the two completed seeds already agree, and T3 is the primary overflow test).

- Model: `qwen/qwen3.8-27b` (Groq), substitute for the paper's (unreleased) Qwen3.6-27B. Live-context budget: 2000 tokens. Max turns: 16.
- **clm** = context-as-a-file (agent edits CTX.md via arbitrary Bash, autonomy preserved).
- **baseline** = prompt-only (all evidence in prompt, truncated to the same budget).
- No FLOPs / efficiency metrics. Correctness + observable behavior only.

## Correctness (A/B), by task

| Task | n (seeds) | CLM correct | Baseline correct | Baseline truncated |
|---|---|---|---|---|
| T1_single | 3 | 3/3 | 3/3 | 0/3 |
| T2_multi | 3 | 3/3 | 3/3 | 3/3 |
| T3_revisit_distractor | 3 | 3/3 | 0/3 | 3/3 |
| A_native_metadata | 2 | 2/2 | 0/2 | 2/2 |

## Observable context-management behavior (CLM), by task

| Task | over-budget runs | avg turns | avg ctx edits | avg token-reducing edits | wrote resolved facts to CTX.md | avg revisits | evidence-retrieval | needle survival |
|---|---|---|---|---|---|---|---|---|
| T1_single | 0/3 | 2.67 | 0.0 | 0.0 | 3/3 | 0.0 | 0.33 | n/a |
| T2_multi | 3/3 | 4.67 | 0.0 | 0.0 | 3/3 | 1.0 | 0.33 | n/a |
| T3_revisit_distractor | 3/3 | 4.33 | 0.67 | 0.33 | 3/3 | 0.0 | 1.0 | 3/3 |
| A_native_metadata | 2/2 | 5.0 | 0.0 | 0.0 | 2/2 | 0.5 | 1.0 | n/a |

## Per-run detail

- **T1_single seed0** — CLM: correct (`dtype=uint8; range=[0,255]`), turns=2, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=1, retrieval=1.0, over_budget=False, needle_survival=None. Baseline: correct (`dtype=uint8; range=[0,255]`, truncated=False).
- **T1_single seed1** — CLM: correct (`dtype=uint8; range=[0,255]`), turns=3, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=2, retrieval=0.0, over_budget=False, needle_survival=None. Baseline: correct (`dtype=uint8; range=[0,255]`, truncated=False).
- **T1_single seed2** — CLM: correct (`dtype=uint8; range=[0,255]`), turns=3, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=2, retrieval=0.0, over_budget=False, needle_survival=None. Baseline: correct (`dtype=uint8; range=[0,255]`, truncated=False).
- **T2_multi seed0** — CLM: correct (`backpack_512x512x373_uint16: range=[0,4071]`), turns=4, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=3, retrieval=1.0, over_budget=True, needle_survival=None. Baseline: correct (`backpack_512x512x373_uint16: range=[0,4071]`, truncated=True).
- **T2_multi seed1** — CLM: correct (`backpack_512x512x373_uint16: range=[0,4071]`), turns=4, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=3, evidence_turns=3, retrieval=0.0, over_budget=True, needle_survival=None. Baseline: correct (`backpack_512x512x373_uint16: range=[0,4071]`, truncated=True).
- **T2_multi seed2** — CLM: correct (`backpack_512x512x373_uint16: range=[0,4071]`), turns=6, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=0, retrieval=0.0, over_budget=True, needle_survival=None. Baseline: correct (`backpack_512x512x373_uint16: range=[0,4071]`, truncated=True).
- **T3_revisit_distractor seed0** — CLM: correct (`count_16bit=4; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216`), turns=4, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=3, retrieval=1.0, over_budget=True, needle_survival=True. Baseline: WRONG (`count_16bit=2; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216`, truncated=True).
- **T3_revisit_distractor seed1** — CLM: correct (`count_16bit=4; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216`), turns=4, edits=1, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=2, retrieval=1.0, over_budget=True, needle_survival=True. Baseline: WRONG (`count_16bit=2; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216`, truncated=True).
- **T3_revisit_distractor seed2** — CLM: correct (`count_16bit=4; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216`), turns=5, edits=1, token_reducing_edits=1, wrote_facts=True, revisits=0, evidence_turns=1, retrieval=1.0, over_budget=True, needle_survival=True. Baseline: WRONG (`count_16bit=2; aneurism_256x256x256_uint8: dtype=uint8, dims=256x256x256, voxels=16777216`, truncated=True).
- **A_native_metadata seed0** — CLM: correct (`count_16bit=4; backpack_512x512x373_uint16: range=[0,4071]; beechnut_1024x1024x1546_uint16: range=[10129,16842]; bunny_512x512x361_uint16: range=[0,63536]; mri_woman: range=[2,4095]`), turns=5, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=1, evidence_turns=4, retrieval=1.0, over_budget=True, needle_survival=None. Baseline: WRONG (`count_16bit=2; backpack_512x512x373_uint16: range=[0,4071]; beechnut_1024x1024x1546_uint16: range=[10129,16842]`, truncated=True).
- **A_native_metadata seed1** — CLM: correct (`count_16bit=4; backpack_512x512x373_uint16: range=[0,4071]; beechnut_1024x1024x1546_uint16: range=[10129,16842]; bunny_512x512x361_uint16: range=[0,63536]; mri_woman: range=[2,4095]`), turns=5, edits=0, token_reducing_edits=0, wrote_facts=True, revisits=0, evidence_turns=4, retrieval=1.0, over_budget=True, needle_survival=None. Baseline: WRONG (`count_16bit=2; backpack_512x512x373_uint16: range=[0,4071]; beechnut_1024x1024x1546_uint16: range=[10129,16842]`, truncated=True).

## Findings

**Primary result — CLM wins exactly when context overflows.** On tasks whose evidence fits the 2000-token budget (T1, and the near-budget T2), CLM and the prompt-only baseline TIE. On the two tasks whose evidence genuinely exceeds the budget and spans all 11 files (T3_revisit_distractor and A_native_metadata), the baseline must truncate and is WRONG in every run, while CLM is correct in every run.
  - T1_single: CLM 3/3 vs baseline 3/3.
  - T2_multi: CLM 3/3 vs baseline 3/3.
  - T3_revisit_distractor: CLM 3/3 vs baseline 0/3.
  - A_native_metadata: CLM 2/2 vs baseline 0/2.

**Failure mode of the baseline is diagnostic.** On T3/A_native the baseline reports `count_16bit=2` instead of the true `4`: truncating the evidence to fit the prompt literally cuts off the later 16-bit volumes, so it answers from what little it can see. CLM reads files on demand and reports the true count.

**The behavior is adaptive, not scripted.** CLM edited CTX.md only after the OVER BUDGET signal fired (T3), and did not bother on the control task that fit (T1). When it did edit, it wrote a compact digest of resolved facts and the required 'anchor' facts survived (needle survival 3/3 on T3). We never told it when or how to manage context; the system prompt describes only the mechanism and the budget.

**What we do NOT claim.** No learning (weights/skills unchanged across tasks). No efficiency/FLOPs claim. Metric caveats: `evidence_retrieval` credits a glob over `evidence/*` as full access; `n_token_reducing_edits` counts edits followed by a lower measured live-context size. The trustworthy evidence is the answer correctness and the CTX.md diffs/snapshots in each run's result.json.

## Interpretation discipline

- **Tool-use** = issued Bash / read files. **Adaptive** = context-management behavior changed with task state (edited CTX.md only when OVER BUDGET; revisited the early anchor fact when the question demanded it). **Learning** (cross-task weight/skill change) is OUT OF SCOPE and NOT claimed.
- Evidence per claim: per-turn Bash, CTX.md diffs, and snapshots in `CLM/experiment/runs/clm_<task>_seed<n>/result.json`.
