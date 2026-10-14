#!/usr/bin/env python3
"""
make_charts.py — generate the comparison visuals for the Context-as-a-File report.

Reads completed runs/*/result.json and writes PNG charts into
CLM/experiment/figs/. No API calls. Charts:
  1. correctness_by_task.png     — CLM vs baseline correctness, grouped bars
  2. context_trajectory.png      — live-context tokens per turn (CLM) with the
                                   budget line and ctx-edit markers (overflow story)
  3. budget_pressure.png         — fits vs overflows: evidence size vs budget
  4. behavior_matrix.png         — heatmap of observable behaviors by task
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import grade, behavior  # reuse graders

RUNS = HERE / "runs"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

TASKS = ["T1_single", "T2_multi", "T3_revisit_distractor", "A_native_metadata"]
SHORT = {"T1_single": "T1\n(fits)", "T2_multi": "T2\n(near)",
         "T3_revisit_distractor": "T3\n(overflow+\ndistractor)",
         "A_native_metadata": "A_native\n(overflow)"}
SEEDS = [0, 1, 2]
BUDGET = 2000

CLM_C = "#16a34a"
BASE_C = "#dc2626"
GRID = "#e5e7eb"


def load(cond, task, seed):
    f = RUNS / f"{cond}_{task}_seed{seed}" / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def collect():
    data = {}
    for task in TASKS:
        clm_runs, base_runs = [], []
        for s in SEEDS:
            c = load("clm", task, s)
            b = load("baseline", task, s)
            if c:
                c["_grade"] = grade(c["final_answer"], c["answer_key"])["correct"]
                c["_beh"] = behavior(c)
                clm_runs.append(c)
            if b:
                b["_grade"] = grade(b["final_answer"], b["answer_key"])["correct"]
                base_runs.append(b)
        data[task] = {"clm": clm_runs, "base": base_runs}
    return data


# ---------------------------------------------------------------------------
def chart_correctness(data):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(TASKS))
    clm_rate, base_rate, labels = [], [], []
    for t in TASKS:
        c = data[t]["clm"]; b = data[t]["base"]
        clm_rate.append(100 * sum(r["_grade"] for r in c) / len(c) if c else 0)
        base_rate.append(100 * sum(r["_grade"] for r in b) / len(b) if b else 0)
        labels.append(SHORT[t])
    w = 0.38
    b1 = ax.bar([i - w/2 for i in x], clm_rate, w, label="Context-as-a-File (CLM)", color=CLM_C)
    b2 = ax.bar([i + w/2 for i in x], base_rate, w, label="Prompt-only baseline", color=BASE_C)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 2,
                    f"{int(r.get_height())}%", ha="center", fontsize=9, fontweight="bold")
    ax.axvspan(1.5, 3.5, color="#fef2f2", alpha=0.6, zorder=0)
    ax.text(2.5, 108, "context OVERFLOWS the budget here", ha="center",
            fontsize=9, color="#991b1b", style="italic")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Answer correctness (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Task correctness: CLM vs prompt-only, by context pressure",
                 fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "correctness_by_task.png", dpi=150)
    plt.close(fig)


def chart_trajectory(data):
    """Live-context tokens per turn for the two overflow tasks, with budget + edits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, task in zip(axes, ["T3_revisit_distractor", "A_native_metadata"]):
        for r in data[task]["clm"]:
            toks = [t["live_tokens"] for t in r["trace"]]
            turns = list(range(1, len(toks) + 1))
            ax.plot(turns, toks, marker="o", linewidth=2, label=f"seed {r['seed']}")
            # mark ctx edits
            for i, t in enumerate(r["trace"]):
                if t.get("ctx_edited"):
                    ax.scatter([i + 1], [t["live_tokens"]], s=180, facecolors="none",
                               edgecolors="#111827", linewidths=2, zorder=5)
        ax.axhline(BUDGET, color=BASE_C, linestyle="--", linewidth=1.8)
        ax.text(1, BUDGET + 60, f"budget = {BUDGET} tok", color=BASE_C, fontsize=9)
        ax.set_title(SHORT[task].replace("\n", " "), fontweight="bold")
        ax.set_xlabel("turn")
        ax.grid(color=GRID); ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("live-context size (exact tokens, Groq usage)")
    fig.suptitle("CLM live-context grows past the budget, then the agent edits CTX.md "
                 "(⬤ = context edit)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "context_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_budget_pressure():
    """Evidence size per task vs the budget — which tasks overflow."""
    # evidence sizes (approx tokens) measured from the task evidence files
    sizes = {}
    for t in TASKS:
        ed = HERE / "tasks" / t / "evidence"
        chars = sum(len(p.read_text()) for p in ed.glob("*.txt"))
        sizes[t] = chars / 2.6  # same dense-content divisor used by the runner
    fig, ax = plt.subplots(figsize=(8.5, 5))
    vals = [sizes[t] for t in TASKS]
    colors = [CLM_C if v <= BUDGET else BASE_C for v in vals]
    bars = ax.bar([SHORT[t] for t in TASKS], vals, color=colors)
    ax.axhline(BUDGET, color="#111827", linestyle="--", linewidth=1.8)
    ax.text(0.02, BUDGET + 150, f"live-context budget = {BUDGET} tokens",
            color="#111827", fontsize=9)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 120, f"~{int(v)}", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("total evidence size (~tokens)")
    ax.set_title("Context pressure by task: green fits the budget, red overflows",
                 fontweight="bold")
    ax.legend(handles=[Patch(color=CLM_C, label="fits budget"),
                       Patch(color=BASE_C, label="overflows budget")], loc="upper left")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "budget_pressure.png", dpi=150)
    plt.close(fig)


def chart_behavior_matrix(data):
    """Heatmap: observable behavior signals (averaged over seeds) per task."""
    import numpy as np
    metrics = ["ever_over_budget", "n_ctx_edits", "n_token_reducing_edits",
               "evidence_retrieval", "needle_survival"]
    mlabels = ["went over\nbudget", "edited\nCTX.md", "token-reducing\nedit",
               "evidence\nretrieval", "needle\nsurvival"]
    M = np.full((len(TASKS), len(metrics)), np.nan)
    for i, t in enumerate(TASKS):
        runs = data[t]["clm"]
        for j, m in enumerate(metrics):
            vals = []
            for r in runs:
                v = r["_beh"].get(m)
                if v is None:
                    continue
                vals.append(float(v) if not isinstance(v, bool) else (1.0 if v else 0.0))
            if vals:
                # normalize edit counts to 0..1 for the heatmap color
                mv = sum(vals) / len(vals)
                if m in ("n_ctx_edits", "n_token_reducing_edits"):
                    mv = min(1.0, mv)
                M[i, j] = mv
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(mlabels, fontsize=9)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([SHORT[t].replace("\n", " ") for t in TASKS], fontsize=9)
    for i in range(len(TASKS)):
        for j in range(len(metrics)):
            v = M[i, j]
            txt = "n/a" if (v != v) else (f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center",
                    color="#111827" if (v != v or v < 0.6) else "white", fontsize=9)
    ax.set_title("Observable context-management behavior (CLM, averaged over seeds)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="fraction / normalized rate")
    fig.tight_layout()
    fig.savefig(FIGS / "behavior_matrix.png", dpi=150)
    plt.close(fig)


def main():
    data = collect()
    chart_correctness(data)
    chart_trajectory(data)
    chart_budget_pressure()
    chart_behavior_matrix(data)
    print("wrote charts to", FIGS)
    for p in sorted(FIGS.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
