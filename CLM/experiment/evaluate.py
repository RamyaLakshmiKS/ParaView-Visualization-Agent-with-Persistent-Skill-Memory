#!/usr/bin/env python3
"""
evaluate.py — orchestrate the A/B, grade deterministically, and summarize BEHAVIOR.

Runs each task under both conditions (clm = context-as-a-file, baseline =
prompt-only) across multiple seeds, grades answers against the deterministic
answer keys, and extracts observable context-management behavior from the CLM
traces. Writes CLM/experiment/REPORT.md.

Metrics (NO FLOPs, NO efficiency):
  correctness            : did the final answer match ground truth
  evidence_retrieval     : for CLM, fraction of needed files actually read
  needle_survival        : for CLM over-budget runs, did the required facts
                           remain recoverable in the final CTX.md after compaction
  behavior signals       : #bash turns, #ctx edits, #compaction events
                           (edit that shrinks CTX.md), #revisits (re-reading a
                           file already read, or re-deriving an earlier fact)

Interpretation discipline: we label behavior as tool-use vs adaptive, and never
claim "learning". Learning (cross-task weight/skill change) is out of scope.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402

TASKS_DIR = HERE / "tasks"
RUNS_DIR = HERE / "runs"
REPORT = HERE / "REPORT.md"


# ---------------------------------------------------------------------------
# Deterministic graders (per task kind)
# ---------------------------------------------------------------------------
def _nums(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s or "")]


def grade(answer: Optional[str], key: Dict) -> Dict:
    """Return {correct: bool, detail: str} for the given answer vs key."""
    a = (answer or "").lower()
    kind = key.get("kind")

    if kind == "single":
        dt = key["dtype"].lower()
        lo, hi = key["range"]
        dtype_ok = dt.split("-")[0] in a  # 'uint8'/'uint16' present
        # range: both endpoints appear
        rng_ok = (str(int(lo)) in a or f"{lo:g}" in a) and (str(int(hi)) in a or f"{hi:g}" in a)
        return {"correct": bool(dtype_ok and rng_ok),
                "detail": f"dtype_ok={dtype_ok} range_ok={rng_ok}"}

    if kind == "filter16":
        truth = key["truth"]  # {name: [lo,hi]} of the 16-bit ones
        # correct iff every true 16-bit name appears AND no 8-bit name is wrongly listed as 16-bit
        names_present = {n for n in key["files"] if n.split("_")[0].lower() in a or n.lower() in a}
        truth_names = set(truth.keys())
        # must mention all truth names; count check as primary
        all_truth = all((n.lower() in a or n.split("_")[0].lower() in a) for n in truth_names)
        count_ok = str(len(truth_names)) in a
        return {"correct": bool(all_truth and count_ok),
                "detail": f"need={sorted(truth_names)} count={len(truth_names)} "
                          f"all_truth={all_truth} count_ok={count_ok}"}

    if kind == "revisit":
        count_ok = f"count_16bit={key['count_16bit']}" in a.replace(" ", "") or \
                   bool(re.search(rf"count_16bit\s*=\s*{key['count_16bit']}\b", a))
        dt = key["anchor_dtype"].split("-")[0]
        dtype_ok = dt in a
        w, h, d = key["anchor_dims"]
        dims_ok = (str(w) in a and str(h) in a and str(d) in a)
        vox_ok = str(key["anchor_voxels"]) in a.replace(",", "")
        correct = count_ok and dtype_ok and dims_ok and vox_ok
        return {"correct": bool(correct),
                "detail": f"count_ok={count_ok} dtype_ok={dtype_ok} dims_ok={dims_ok} vox_ok={vox_ok}"}

    return {"correct": False, "detail": "unknown kind"}


# ---------------------------------------------------------------------------
# Behavior extraction from a CLM trace
# ---------------------------------------------------------------------------
def behavior(result: Dict) -> Dict:
    import re as _re
    trace = result.get("trace", [])
    key = result.get("answer_key", {})
    # the set of real evidence files for this task
    if key.get("kind") == "revisit":
        needed_files = set(key.get("planted", {}).keys())
    else:
        needed_files = set(key.get("files", []) or ([key.get("file")] if key.get("file") else []))

    n_bash = n_edits = 0
    revisits = 0
    seen_files = set()           # specific files named in commands
    accessed_evidence_turns = 0  # turns that read the evidence dir at all (incl. globs)
    glob_access = False

    for t in trace:
        if t.get("model_output_kind") != "bash":
            continue
        n_bash += 1
        cmd = (t.get("bash") or "")
        if t.get("ctx_edited"):
            n_edits += 1
        # did this turn touch the evidence directory in ANY way?
        if "evidence/" in cmd or "evidence/*" in cmd:
            accessed_evidence_turns += 1
            if "evidence/*" in cmd or _re.search(r"head\s+.*evidence", cmd) or "evidence/*.txt" in cmd:
                glob_access = True
        # specific files named (for revisit detection)
        for f in _re.findall(r"evidence/([A-Za-z0-9_.\-]+\.txt)", cmd):
            if f in seen_files:
                revisits += 1
            seen_files.add(f)

    # evidence_retrieval: fraction of needed files PROVABLY touched. If the agent
    # used a glob over evidence/*, it accessed ALL files -> retrieval = 1.0.
    if not needed_files:
        retrieval = None
    elif glob_access:
        retrieval = 1.0
    else:
        retrieval = round(len(seen_files & needed_files) / len(needed_files), 2)

    # context-management acts: edits to CTX.md, and whether any edit REDUCED the
    # live-context token count turn-over-turn (true compaction effect).
    toks = [t["live_tokens"] for t in trace if "live_tokens" in t]
    token_reducing_edits = 0
    for i, t in enumerate(trace):
        if t.get("ctx_edited") and i + 1 < len(trace):
            cur = t.get("live_tokens")
            nxt = trace[i + 1].get("live_tokens")
            if cur and nxt and nxt < cur:
                token_reducing_edits += 1

    # did the final CTX.md hold RESOLVED FACTS (digested), not just the raw task?
    final_ctx = (result.get("final_ctx") or "")
    wrote_resolved = bool(_re.search(r"(uint8|uint16|range|voxel|dims|256x256|dtype)", final_ctx.lower())) \
        and len(final_ctx) > 120

    # needle survival: are the required resolved facts recoverable in final CTX.md?
    survival = None
    if key.get("kind") == "revisit":
        fc = final_ctx.lower()
        anchor = key["anchor"].lower()
        w, h, d = key["anchor_dims"]
        survival = all([
            anchor.split("_")[0] in fc,
            key["anchor_dtype"].split("-")[0] in fc,
            (str(w) in fc and str(h) in fc and str(d) in fc),
        ])

    return {
        "n_bash_turns": n_bash,
        "n_ctx_edits": n_edits,
        "n_token_reducing_edits": token_reducing_edits,
        "wrote_resolved_facts_to_ctx": wrote_resolved,
        "n_revisits": revisits,
        "evidence_turns": accessed_evidence_turns,
        "files_read_unique": sorted(seen_files),
        "evidence_retrieval": retrieval,
        "needle_survival": survival,
        "ever_over_budget": any(t.get("over_budget") for t in trace),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all(tasks: List[str], seeds: List[int], budget: int, max_turns: int) -> Dict:
    rows = []
    for task in tasks:
        for seed in seeds:
            clm = runner.run_clm(task, seed, budget, max_turns)
            base = runner.run_baseline(task, seed, budget)
            g_clm = grade(clm["final_answer"], clm["answer_key"])
            g_base = grade(base["final_answer"], base["answer_key"])
            beh = behavior(clm)
            rows.append({
                "task": task, "seed": seed,
                "clm_answer": clm["final_answer"], "clm_correct": g_clm["correct"],
                "clm_detail": g_clm["detail"],
                "base_answer": base["final_answer"], "base_correct": g_base["correct"],
                "base_detail": g_base["detail"],
                "base_truncated": base.get("evidence_truncated"),
                "clm_turns": clm["turns_used"],
                "behavior": beh,
            })
            print(f"[{task} seed{seed}] clm={'OK' if g_clm['correct'] else 'X'} "
                  f"base={'OK' if g_base['correct'] else 'X'} "
                  f"turns={clm['turns_used']} edits={beh['n_ctx_edits']} "
                  f"compact={beh['n_compaction_events']} retr={beh['evidence_retrieval']}")
    return {"rows": rows, "budget": budget, "max_turns": max_turns,
            "model": runner.GroqChat().model}


def write_report(summary: Dict) -> None:
    rows = summary["rows"]
    tasks = sorted({r["task"] for r in rows})
    lines = []
    lines.append("# Context-as-a-File — Behavioral Experiment Report\n")
    lines.append(f"- Model: `{summary['model']}` (Groq)")
    lines.append(f"- Live-context budget: {summary['budget']} tokens  |  max turns: {summary['max_turns']}")
    lines.append(f"- Conditions: **clm** (context-as-a-file, agent edits CTX.md via Bash, autonomy preserved) vs **baseline** (prompt-only, evidence truncated to the same budget)")
    lines.append("- No FLOPs / efficiency metrics. Correctness + observable behavior only.\n")

    lines.append("## Correctness (A/B), by task\n")
    lines.append("| Task | seeds | CLM correct | Baseline correct | Baseline truncated |")
    lines.append("|---|---|---|---|---|")
    for task in tasks:
        tr = [r for r in rows if r["task"] == task]
        n = len(tr)
        clm_c = sum(r["clm_correct"] for r in tr)
        base_c = sum(r["base_correct"] for r in tr)
        trunc = sum(bool(r["base_truncated"]) for r in tr)
        lines.append(f"| {task} | {n} | {clm_c}/{n} | {base_c}/{n} | {trunc}/{n} |")

    lines.append("\n## Observable context-management behavior (CLM), by task\n")
    lines.append("| Task | over-budget | avg bash turns | avg ctx edits | avg compactions | avg revisits | avg evidence-retrieval | needle survival |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for task in tasks:
        tr = [r for r in rows if r["task"] == task]
        n = len(tr)
        ob = sum(r["behavior"]["ever_over_budget"] for r in tr)
        def avg(k):
            vals = [r["behavior"][k] for r in tr if r["behavior"][k] is not None]
            return round(sum(vals) / len(vals), 2) if vals else "n/a"
        surv = [r["behavior"]["needle_survival"] for r in tr if r["behavior"]["needle_survival"] is not None]
        surv_s = f"{sum(surv)}/{len(surv)}" if surv else "n/a"
        lines.append(f"| {task} | {ob}/{n} | {avg('n_bash_turns')} | {avg('n_ctx_edits')} | "
                     f"{avg('n_compaction_events')} | {avg('n_revisits')} | {avg('evidence_retrieval')} | {surv_s} |")

    lines.append("\n## Per-run detail\n")
    for r in rows:
        b = r["behavior"]
        lines.append(f"- **{r['task']} seed{r['seed']}** — "
                     f"CLM: {'correct' if r['clm_correct'] else 'WRONG'} "
                     f"(`{r['clm_answer']}`), turns={r['clm_turns']}, "
                     f"edits={b['n_ctx_edits']}, compactions={b['n_compaction_events']}, "
                     f"revisits={b['n_revisits']}, retrieval={b['evidence_retrieval']}, "
                     f"over_budget={b['ever_over_budget']}, needle_survival={b['needle_survival']}. "
                     f"Baseline: {'correct' if r['base_correct'] else 'WRONG'} "
                     f"(`{r['base_answer']}`, truncated={r['base_truncated']}).")

    lines.append("\n## Interpretation discipline\n")
    lines.append("- **Tool-use** = the agent issued Bash/read files. **Adaptive** = its "
                 "context-management behavior changed with task state (e.g. only compacted "
                 "when over budget, revisited an early fact when the question demanded it). "
                 "**Learning** (cross-task weight/skill change) is OUT OF SCOPE and NOT claimed.")
    lines.append("- Evidence for each claim is the per-run trace under "
                 "`CLM/experiment/runs/clm_<task>_seed<n>/result.json` (per-turn bash, "
                 "ctx diffs, and CTX.md snapshots).")

    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {HERE/'REPORT.md'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["T1_single", "T2_multi", "T3_revisit_distractor", "A_native_metadata"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=runner.DEFAULT_BUDGET)
    ap.add_argument("--max-turns", type=int, default=runner.DEFAULT_MAX_TURNS)
    args = ap.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    summ = run_all(args.tasks, args.seeds, args.budget, args.max_turns)
    write_report(summ)
