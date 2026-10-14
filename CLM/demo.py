#!/usr/bin/env python3
"""
demo.py — a runnable, dependency-free demonstration of Context Language Models
(context-as-a-file) on a simulated volume-rendering task.

It plays the SAME task through two agents:
  (A) naive-append  — can only add to its context (what a standard LM does).
  (B) context-as-a-file (CLM) — freely rewrites its working context, compressing
      raw tool dumps into one-line resolved facts and discarding stale material.

Both agents "do" the same work (inspect a headerless .raw, resolve dtype/shape,
build a range-relative transfer function, render, look, fix). The task steps mirror
the real mri_woman lesson already in memory/memory.json. The difference is purely
how each manages its working context — and that is what we measure.

No VTK or data files are needed; the render steps are simulated so the demo runs
anywhere. It writes two trace files consumed by clm_metrics.py and prints a
side-by-side size comparison.

Run:
  python CLM/demo.py
  python CLM/clm_metrics.py --compare CLM/context/naive.json CLM/context/clm.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from context_store import WorkingContext  # local import (run from CLM/ or via -m)

HERE = os.path.dirname(os.path.abspath(__file__))
CTX_DIR = os.path.join(HERE, "context")

GOAL = "volume render mri_woman.raw so dense tissue stands out and soft tissue is faint"

# A realistic stream of per-step "tool output" the agent would receive. The raw
# dumps are intentionally bulky — that is exactly what a naive agent keeps and a
# CLM compresses.
RAW_INSPECT_DUMP: List[str] = [
    f"byte[{i}]=0x{(i * 37) % 256:02x}" for i in range(24)
] + [
    "file size = 14,286,848 bytes",
    "candidate A: uint8 256x256x218  (fits byte count)",
    "candidate B: uint16-LE 256x256x109 (fits byte count)",
    "adjacent-slice correlation A = 0.46",
    "adjacent-slice correlation B = 0.88",
    "scalar range if B (uint16-LE) = [2, 4095]  (looks 12-bit)",
]

RAW_HISTOGRAM_DUMP: List[str] = [f"bin[{b}]={ (4096 - b*41) % 1000 }" for b in range(32)]

RENDER1_LOG: List[str] = [
    "wrote results/mri_woman.py (142 lines)",
    "vtk offscreen render -> results/mri_woman.png",
    "LOOK: skin shell too opaque, hides interior anatomy",
]

RENDER2_LOG: List[str] = [
    "edited transfer function: ghost soft-tissue band 0.08-0.34",
    "vtk offscreen render -> results/mri_woman.png",
    "LOOK: skull + sutures read clearly, soft tissue translucent — good",
]


def _snapshot(step: int, label: str, wc: WorkingContext) -> Dict:
    return {
        "step": step,
        "label": label,
        "lines": wc.line_count(),
        "chars": wc.char_count(),
        "est_tokens": wc.est_tokens(),
    }


def run_naive() -> Dict:
    """Naive agent: append everything, never rewrite. Context only grows."""
    wc = WorkingContext.new(os.path.join(CTX_DIR, "naive_working.md"), goal=GOAL)
    trace: List[Dict] = []
    total_written = 1  # the goal line

    # Step 1: dump the entire raw inspection into context
    for line in RAW_INSPECT_DUMP:
        wc.append("Resolved facts", line)
    total_written += len(RAW_INSPECT_DUMP)
    trace.append(_snapshot(1, "inspect .raw", wc))

    # Step 2: dump the whole histogram
    for line in RAW_HISTOGRAM_DUMP:
        wc.append("Resolved facts", line)
    total_written += len(RAW_HISTOGRAM_DUMP)
    trace.append(_snapshot(2, "histogram", wc))

    # Step 3: first render log appended
    for line in RENDER1_LOG:
        wc.append("Open issues", line)
    total_written += len(RENDER1_LOG)
    trace.append(_snapshot(3, "render #1 + look", wc))

    # Step 4: second render log appended on top of everything
    for line in RENDER2_LOG:
        wc.append("Open issues", line)
    total_written += len(RENDER2_LOG)
    trace.append(_snapshot(4, "render #2 + look", wc))

    wc.save()
    return {
        "name": "naive-append",
        "trace": trace,
        "total_lines_written": total_written,
        "lines_discarded": 0,
    }


def run_clm() -> Dict:
    """CLM agent: context-as-a-file. Compress raw dumps into resolved facts,
    discard stale material, keep the working context small."""
    wc = WorkingContext.new(os.path.join(CTX_DIR, "clm_working.md"), goal=GOAL)
    trace: List[Dict] = []
    total_written = 1
    discarded = 0

    # Step 1: inspect, then COMPRESS the whole dump into two resolved facts.
    # (The agent read RAW_INSPECT_DUMP but keeps only the conclusions.)
    total_written += len(RAW_INSPECT_DUMP)
    wc.append("Resolved facts", "dtype/shape = uint16-LE 256x256x109 (corr 0.88 > 0.46)")
    wc.append("Resolved facts", "scalar range [2,4095], 12-bit; background below ~5%")
    wc.append("Discard log", f"compressed {len(RAW_INSPECT_DUMP)} raw inspect lines -> 2 facts")
    discarded += len(RAW_INSPECT_DUMP) - 2
    trace.append(_snapshot(1, "inspect .raw (compressed)", wc))

    # Step 2: read histogram, keep ONE conclusion, discard the 32 bins.
    total_written += len(RAW_HISTOGRAM_DUMP)
    wc.append("Resolved facts", "histogram: low-value air/noise dominates -> hard cutoff at ~5%")
    wc.append("Discard log", f"compressed {len(RAW_HISTOGRAM_DUMP)} histogram bins -> 1 fact")
    discarded += len(RAW_HISTOGRAM_DUMP) - 1
    trace.append(_snapshot(2, "histogram (compressed)", wc))

    # Step 3: set the current transfer function; render #1; record only the
    # actionable issue, not the full render log.
    total_written += len(RENDER1_LOG)
    wc.set_section("Current TF", [
        "cutoff 0; ghost soft band 0.08-0.34 (~1-8%); dense ramp >0.46 to 0.75-0.95",
        "color: cool blue-grey soft -> warm gold/white dense",
    ])
    wc.set_section("Open issues", ["skin shell too opaque, hides interior — lower soft-band opacity"])
    wc.append("Discard log", f"kept 1 issue from {len(RENDER1_LOG)} render-log lines")
    discarded += len(RENDER1_LOG) - 1
    total_written += 2  # the TF lines
    trace.append(_snapshot(3, "render #1 + look (rewrite)", wc))

    # Step 4: render #2 fixes it; the issue is RESOLVED, so clear Open issues and
    # record the final state compactly.
    total_written += len(RENDER2_LOG)
    wc.discard("Open issues", "resolved: skull+sutures read, soft tissue translucent")
    discarded += 1
    trace.append(_snapshot(4, "render #2 + look (resolved)", wc))

    wc.save()
    return {
        "name": "context-as-a-file (CLM)",
        "trace": trace,
        "total_lines_written": total_written,
        "lines_discarded": discarded,
    }


def main() -> int:
    os.makedirs(CTX_DIR, exist_ok=True)
    naive = run_naive()
    clm = run_clm()

    with open(os.path.join(CTX_DIR, "naive.json"), "w", encoding="utf-8") as fh:
        json.dump(naive, fh, indent=2)
    with open(os.path.join(CTX_DIR, "clm.json"), "w", encoding="utf-8") as fh:
        json.dump(clm, fh, indent=2)

    print("Simulated task:", GOAL)
    print()
    print("Per-step working-context size (est. tokens):")
    print(f"  {'step':<28}{'naive':>8}{'CLM':>8}")
    for n, c in zip(naive["trace"], clm["trace"]):
        print(f"  {n['label'][:26]:<28}{n['est_tokens']:>8}{c['est_tokens']:>8}")
    print()
    print("Wrote traces:")
    print("  CLM/context/naive.json")
    print("  CLM/context/clm.json")
    print("  CLM/context/naive_working.md   (grew — everything kept)")
    print("  CLM/context/clm_working.md     (small — compressed + discarded)")
    print()
    print("Now compare:")
    print("  python CLM/clm_metrics.py --compare CLM/context/naive.json CLM/context/clm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
