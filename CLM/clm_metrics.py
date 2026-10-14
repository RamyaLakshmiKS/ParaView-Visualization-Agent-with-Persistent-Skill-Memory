#!/usr/bin/env python3
"""
clm_metrics.py — measure the Context Language Models efficiency claim locally.

The CLM paper's headline is "better results AND lower compute (FLOPs)" because
the model keeps a smaller, more relevant working context. We cannot count real
server FLOPs in this project (no weight access, no custom serving layer), so we
measure the honest local analogue:

  working-context SIZE over the course of a task
    - peak size      (the worst moment — what would blow a context budget)
    - final size     (what you carry to the end)
    - retained vs discarded lines (how much the agent compressed away)

A naive-append agent only grows; a CLM (context-as-a-file) agent rewrites and
compresses, so its peak and final sizes stay small while still holding the facts
that matter. This module quantifies that gap.

PROXY CAVEAT (stated plainly, like the project's transcript-metric caveat):
context size is a PROXY for compute, not a FLOPs measurement. Attention cost
grows super-linearly with context length, so a smaller working context does
imply less compute — but the exact FLOPs/latency depend on the model and server,
which are out of scope here. See CLM/README.md.

Usage:
  # from a trace produced by demo.py (list of per-step sizes)
  python CLM/clm_metrics.py --trace CLM/context/last_run.json

  # or point at two trace files to print an A/B comparison
  python CLM/clm_metrics.py --compare CLM/context/naive.json CLM/context/clm.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class StepSize:
    """Working-context size at one step of a task."""
    step: int
    label: str
    lines: int
    chars: int
    est_tokens: int


@dataclass
class RunSummary:
    name: str
    steps: int
    peak_lines: int
    final_lines: int
    peak_tokens: int
    final_tokens: int
    total_lines_written: int
    lines_discarded: int

    def retention_ratio(self) -> float:
        if self.total_lines_written == 0:
            return 1.0
        return round(self.final_lines / self.total_lines_written, 3)


def summarize(name: str, trace: List[StepSize], total_written: int, discarded: int) -> RunSummary:
    peak_lines = max((s.lines for s in trace), default=0)
    peak_tokens = max((s.est_tokens for s in trace), default=0)
    final = trace[-1] if trace else StepSize(0, "empty", 0, 0, 0)
    return RunSummary(
        name=name,
        steps=len(trace),
        peak_lines=peak_lines,
        final_lines=final.lines,
        peak_tokens=peak_tokens,
        final_tokens=final.est_tokens,
        total_lines_written=total_written,
        lines_discarded=discarded,
    )


def load_trace(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _summary_from_trace_file(path: str) -> RunSummary:
    data = load_trace(path)
    steps = [StepSize(**s) for s in data["trace"]]
    return summarize(
        name=data.get("name", os.path.basename(path)),
        trace=steps,
        total_written=data.get("total_lines_written", sum(s.lines for s in steps)),
        discarded=data.get("lines_discarded", 0),
    )


def print_summary(s: RunSummary) -> None:
    print(f"== {s.name} ==")
    print(f"  steps                 : {s.steps}")
    print(f"  peak working context  : {s.peak_lines} lines (~{s.peak_tokens} tokens)")
    print(f"  final working context : {s.final_lines} lines (~{s.final_tokens} tokens)")
    print(f"  total lines written   : {s.total_lines_written}")
    print(f"  lines discarded        : {s.lines_discarded}")
    print(f"  retention ratio        : {s.retention_ratio()} (final / total written)")


def print_comparison(naive: RunSummary, clm: RunSummary) -> None:
    print_summary(naive)
    print()
    print_summary(clm)
    print()
    print("== CLM vs naive-append (the paper's 'smaller working context') ==")

    def pct_drop(a: int, b: int) -> str:
        if a == 0:
            return "n/a"
        return f"{round((a - b) / a * 100, 1)}% smaller"

    print(f"  peak size   : naive {naive.peak_tokens} tok -> CLM {clm.peak_tokens} tok "
          f"({pct_drop(naive.peak_tokens, clm.peak_tokens)})")
    print(f"  final size  : naive {naive.final_tokens} tok -> CLM {clm.final_tokens} tok "
          f"({pct_drop(naive.final_tokens, clm.final_tokens)})")
    print()
    print("  NOTE: size is a local PROXY for compute/FLOPs, not a FLOPs measurement.")
    print("        See CLM/README.md for why and for the paper's actual metric.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clm_metrics",
        description="Measure working-context size (CLM efficiency proxy).",
    )
    p.add_argument("--trace", help="a single run trace JSON (from demo.py)")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("NAIVE_JSON", "CLM_JSON"),
        help="two trace JSONs to compare (naive-append vs context-as-a-file)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        naive = _summary_from_trace_file(args.compare[0])
        clm = _summary_from_trace_file(args.compare[1])
        print_comparison(naive, clm)
        return 0
    if args.trace:
        print_summary(_summary_from_trace_file(args.trace))
        return 0
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
