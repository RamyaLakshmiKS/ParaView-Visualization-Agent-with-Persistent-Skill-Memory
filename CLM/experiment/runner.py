#!/usr/bin/env python3
"""
runner.py — the Context-as-a-File agent loop (paper arXiv:2609.37725, Section 4.1)
plus a prompt-only baseline, with full per-turn trace capture.

FAITHFUL MECHANISM (what the paper specifies in 4.1):
  * The agent's LIVE CONTEXT is mirrored to an editable file, CTX.md, whose path
    is given in the system prompt.
  * The agent edits CTX.md with GENERAL BASH commands (not a fixed op set).
  * Every edit is AUTO-SYNCHRONIZED back into the live context for the next turn:
    here, the current contents of CTX.md are re-injected as a system message each
    turn. This is the "mirror the context into an editable file ... each
    modification immediately synchronized to the LM's context" design.
  * When the agent does NOT edit CTX.md, its generated text is APPENDED by default
    (standard LM behavior).

AUTONOMY (the user's hard requirement, and the paper's core claim):
  We expose CAPABILITY, not procedure. The agent gets a single `bash` tool and an
  editable file. We NEVER tell it to summarize/compact/retrieve in any order. What
  it reads, whether/when it compacts CTX.md, how it organizes it, and when it
  answers are all its decisions. The system prompt describes the mechanism and the
  budget, nothing about strategy.

BUDGET (to force management, NOT to measure efficiency):
  A small live-context budget (default 6000 tokens) is enforced. Each turn we
  measure the EXACT live-context token size via the Groq usage field. If CTX.md +
  recent turns exceed the budget, the agent is told it is over budget (an
  environmental hint, exactly as the paper does: "CLM receives its editing
  reminder N tokens before the budget", Appendix E). We never auto-compact for it.

BASELINE (prompt-only):
  Same model, same task, ALL evidence concatenated into the prompt. If it exceeds
  the budget it is TRUNCATED (what you must do when you can't manage context).
  No file, no tools. This is the "put available context directly in the prompt"
  comparison the user asked for.

No FLOPs. No efficiency metrics. Traces + answers only.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from groq_client import GroqChat, count_prompt_tokens  # noqa: E402

TASKS_DIR = HERE / "tasks"
RUNS_DIR = HERE / "runs"
DEFAULT_BUDGET = 6000
DEFAULT_MAX_TURNS = 16


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
CLM_SYSTEM = """\
You are an agent solving a task. Your LIVE WORKING CONTEXT is mirrored to a file:

    {ctx_path}

This file IS your memory between turns. Its current contents are shown to you each
turn under "=== CTX.md (your live context) ===". You may change it using ANY shell
command (cat, grep, sed, python3 heredocs, redirects, etc.). Any edit you make is
automatically loaded back as your context on the next turn. If you do not edit it,
your reply is simply added to the conversation.

External evidence for the task lives on disk under:

    {evidence_dir}/

These files are NOT in your context. Read whatever you need with shell commands.

Your live context budget is about {budget} tokens. If you are told you are OVER
BUDGET, you must shrink CTX.md before continuing or you will lose the ability to
work. HOW you manage CTX.md — what to read, what to keep, whether and when to
summarize or reorganize, what to discard — is entirely your choice.

HOW TO ACT each turn, output EXACTLY ONE of:
  (a) a shell action, as a fenced block:
        ```bash
        <commands>
        ```
     To change your context, write to CTX.md inside that block (e.g. a
     `python3 - <<'PY' ... PY` heredoc or `cat > {ctx_path} <<'EOF' ... EOF`).
  (b) your final answer, as:
        FINAL: <your answer on one line, in the exact format the task requests>

Do not output both. Think briefly, then act. Answer only when confident.
"""

BASELINE_SYSTEM = """\
You are an agent solving a task. All available evidence is included below in this
prompt. Read it and answer. Output your final answer as:
  FINAL: <your answer on one line, in the exact format the task requests>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_bash(text: str) -> Optional[str]:
    m = re.search(r"```bash\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # tolerate a bare ``` fence
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m and ("CTX.md" in m.group(1) or any(c in m.group(1) for c in ("cat", "grep", "python3", "sed", "echo"))):
        return m.group(1).strip()
    return None


def extract_final(text: str) -> Optional[str]:
    m = re.search(r"FINAL:\s*(.+)", text)
    return m.group(1).strip() if m else None


def run_bash(cmd: str, cwd: pathlib.Path, timeout: int = 30) -> Tuple[str, str, int]:
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                           text=True, timeout=timeout)
        return p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", 124


def diff_text(a: str, b: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                        "CTX.md.before", "CTX.md.after"))[:4000]


# ---------------------------------------------------------------------------
# Context-as-a-File run
# ---------------------------------------------------------------------------
def run_clm(task_id: str, seed: int, budget: int = DEFAULT_BUDGET,
            max_turns: int = DEFAULT_MAX_TURNS) -> Dict:
    task_dir = TASKS_DIR / task_id
    question = (task_dir / "question.txt").read_text().strip()
    key = json.loads((task_dir / "answer_key.json").read_text())

    # per-run sandbox workspace (so bash edits are isolated + reproducible)
    ws = RUNS_DIR / f"clm_{task_id}_seed{seed}"
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "evidence").mkdir(parents=True)
    for f in (task_dir / "evidence").glob("*.txt"):
        shutil.copy(f, ws / "evidence" / f.name)
    ctx_path = ws / "CTX.md"
    ctx_path.write_text(f"# Working context\n\n## Task\n{question}\n")

    chat = GroqChat(temperature=0.3 + 0.1 * seed)  # vary a bit per seed
    sys_prompt = CLM_SYSTEM.format(ctx_path="CTX.md", evidence_dir="evidence",
                                   budget=budget)

    trace: List[Dict] = []
    ctx_snapshots: List[str] = [ctx_path.read_text()]
    conversation_tail: List[Dict] = []  # recent raw turns (ephemeral rolling window)
    WINDOW = 6  # keep only the last 6 raw messages; older material must be saved to CTX.md
    final_answer = None
    _ratio = 1.0  # adaptive correction: actual_tokens / (chars/2.6), updated each turn

    for turn in range(1, max_turns + 1):
        ctx_now = ctx_path.read_text()
        # Build the live context sent to the model: system + CTX.md mirror +
        # the appended (non-edit) tail + the standing user question.
        live_ctx_block = f"=== CTX.md (your live context) ===\n{ctx_now}\n=== end CTX.md ==="
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "system", "content": live_ctx_block}]
        messages += conversation_tail
        messages.append({"role": "user", "content":
                         f"TASK: {question}\n(Turn {turn}/{max_turns}. Act now.)"})

        # Estimate live-context size locally to decide the over-budget ENV hint
        # BEFORE the call (we avoid a separate probe call that doubles requests
        # and trips rate limits). This dense, numeric evidence tokenizes heavier
        # than prose: measured ~chars/2.3 on this content, so we use chars/2.6 as
        # a slightly conservative divisor. The EXACT size is recorded after the
        # call from usage.prompt_tokens; we also back-correct the hint using the
        # previous turn's actual/estimate ratio.
        est_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = max(1, round(est_chars / 2.6 * _ratio))
        over = est_tokens > budget
        if over:
            messages.append({"role": "system", "content":
                             f"[ENV] OVER BUDGET: live context is ~{est_tokens} tokens "
                             f"(> {budget}). Shrink CTX.md before continuing."})

        reply, usage = chat.complete(messages)
        live_tokens = usage.get("prompt_tokens", est_tokens)  # EXACT, from the API
        # back-correct the estimator for next turn (clamped to a sane band)
        base_est = max(1, round(est_chars / 2.6))
        _ratio = min(2.0, max(0.6, live_tokens / base_est))
        bash = extract_bash(reply)
        final = extract_final(reply)

        rec: Dict = {"turn": turn, "live_tokens": live_tokens, "est_tokens": est_tokens,
                     "over_budget": over,
                     "model_output_kind": "answer" if final else ("bash" if bash else "text"),
                     "reply_preview": reply[:300]}

        if final:
            final_answer = final
            rec["final"] = final
            trace.append(rec)
            break

        if bash:
            before = ctx_path.read_text()
            out, err, code = run_bash(bash, cwd=ws)
            after = ctx_path.read_text()
            edited = (after != before)
            files_read = re.findall(r"evidence/([A-Za-z0-9_.\-]+\.txt)", bash)
            rec.update({
                "bash": bash[:800],
                "stdout_preview": out[:600],
                "stderr_preview": err[:300],
                "exit": code,
                "ctx_edited": edited,
                "files_read": sorted(set(files_read)),
                "ctx_diff": diff_text(before, after) if edited else "",
                "ctx_size_chars_after": len(after),
            })
            # feed the command output back as a tool message. We keep only a
            # SHORT rolling window of raw turns (below): old raw observations age
            # out of the live context, so if the agent wants to keep a fact it
            # must write it into CTX.md. This is the paper's design — CTX.md is
            # the durable memory; raw turns are ephemeral. We never compact for
            # the agent (autonomy preserved); we only age out raw material.
            tool_msg = (f"[shell exit={code}]\nSTDOUT:\n{out[:2200]}\n"
                        f"STDERR:\n{err[:400]}")
            conversation_tail.append({"role": "assistant", "content": reply[:1200]})
            conversation_tail.append({"role": "user", "content": tool_msg})
            conversation_tail[:] = conversation_tail[-WINDOW:]
            if edited:
                ctx_snapshots.append(after)
        else:
            # no edit, no final -> append-by-default (paper's standard LM fallback)
            conversation_tail.append({"role": "assistant", "content": reply[:1200]})
            conversation_tail[:] = conversation_tail[-WINDOW:]

        trace.append(rec)

    result = {
        "condition": "clm",
        "task_id": task_id,
        "seed": seed,
        "budget": budget,
        "turns_used": len(trace),
        "final_answer": final_answer,
        "answer_key": key,
        "trace": trace,
        "ctx_snapshots": ctx_snapshots,
        "final_ctx": ctx_path.read_text(),
        "tokens": {"prompt": chat.total_prompt_tokens,
                   "completion": chat.total_completion_tokens,
                   "calls": chat.n_calls},
    }
    _save(result, ws)
    return result


# ---------------------------------------------------------------------------
# Prompt-only baseline
# ---------------------------------------------------------------------------
def run_baseline(task_id: str, seed: int, budget: int = DEFAULT_BUDGET) -> Dict:
    task_dir = TASKS_DIR / task_id
    question = (task_dir / "question.txt").read_text().strip()
    key = json.loads((task_dir / "answer_key.json").read_text())

    # concatenate ALL evidence into the prompt
    evidence_parts = []
    for f in sorted((task_dir / "evidence").glob("*.txt")):
        evidence_parts.append(f"----- {f.name} -----\n{f.read_text()}")
    evidence = "\n".join(evidence_parts)

    chat = GroqChat(temperature=0.3 + 0.1 * seed)
    # truncate evidence to fit the SAME budget (what you must do without mgmt).
    # Use a local chars/4 estimate (no extra probe call) for the truncation
    # decision; the exact prompt size is recorded from the real call's usage.
    def est(msgs):
        return max(1, round(sum(len(m.get("content", "")) for m in msgs) / 4))
    base_msgs = [{"role": "system", "content": BASELINE_SYSTEM},
                 {"role": "user", "content": f"TASK: {question}\n\nEVIDENCE:\n{evidence}"}]
    full_tokens = est(base_msgs)
    truncated = False
    if full_tokens > budget:
        truncated = True
        approx_keep = int(len(evidence) * (budget / full_tokens) * 0.90)
        evidence_trunc = evidence[:approx_keep] + "\n...[EVIDENCE TRUNCATED TO FIT BUDGET]..."
        base_msgs = [{"role": "system", "content": BASELINE_SYSTEM},
                     {"role": "user", "content": f"TASK: {question}\n\nEVIDENCE:\n{evidence_trunc}"}]

    reply, usage = chat.complete(base_msgs)
    live_tokens = usage.get("prompt_tokens", est(base_msgs))  # EXACT, from the API
    final = extract_final(reply) or (reply.strip().splitlines()[-1] if reply.strip() else "")

    result = {
        "condition": "baseline",
        "task_id": task_id,
        "seed": seed,
        "budget": budget,
        "live_tokens": live_tokens,
        "evidence_truncated": truncated,
        "full_evidence_tokens": full_tokens,
        "final_answer": final,
        "answer_key": key,
        "reply_preview": reply[:600],
        "tokens": {"prompt": chat.total_prompt_tokens,
                   "completion": chat.total_completion_tokens,
                   "calls": chat.n_calls},
    }
    ws = RUNS_DIR / f"baseline_{task_id}_seed{seed}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    _save(result, ws)
    return result


def _save(result: Dict, ws: pathlib.Path) -> None:
    (ws / "result.json").write_text(json.dumps(result, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Context-as-a-File runner (CLM + baseline)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--condition", choices=["clm", "baseline"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    args = ap.parse_args(argv)
    RUNS_DIR.mkdir(exist_ok=True)
    if args.condition == "clm":
        res = run_clm(args.task, args.seed, args.budget, args.max_turns)
    else:
        res = run_baseline(args.task, args.seed, args.budget)
    print(f"[{res['condition']}] task={res['task_id']} seed={res['seed']} "
          f"answer={res.get('final_answer')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
