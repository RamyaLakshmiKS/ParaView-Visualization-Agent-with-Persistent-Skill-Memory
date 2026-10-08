#!/usr/bin/env python3
"""
metrics.py — turn a Claude Code run transcript into the before/after table.

The demo's whole thesis is a measurable delta: with memory ON vs OFF, how many
steps/tool-calls and errors/retries did the agent need, and did it repeat a known
mistake?

Usage:
    python metrics.py --label "engine OFF" --transcript run_off.txt
    python metrics.py --label "engine ON"  --transcript run_on.txt

Then collect the printed rows into the README table.

This is intentionally simple and heuristic — it counts signals in a saved transcript.
Adjust the patterns to match whatever transcript format you capture (Claude Code
--output-format stream-json, or a plain copy-paste).
"""

import argparse
import re
import sys


# Heuristic signals. Tune these to your captured transcript format.
TOOL_CALL_PATTERNS = [
    r"\bRunning tool\b",
    r"\btool_use\b",
    r"\bBash\b",
    r"\bWrite\b",
    r"\bEdit\b",
]
ERROR_PATTERNS = [
    r"\bTraceback\b",
    r"\bError\b",
    r"\bfailed\b",
    r"\bretry(ing)?\b",
    r"\bExcept(ion)?\b",
]
# A known mistake we want to detect being avoided (the color-bar / axes slip).
MISTAKE_PATTERNS = [
    r"color\s*bar",
    r"scalar\s*bar",
    r"orientation\s*axes",
]


def count(patterns, text):
    total = 0
    for p in patterns:
        total += len(re.findall(p, text, flags=re.IGNORECASE))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Row label, e.g. 'engine ON'")
    ap.add_argument("--transcript", required=True, help="Path to saved run transcript")
    args = ap.parse_args()

    try:
        with open(args.transcript, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Transcript not found: {args.transcript}")
        return 1

    tool_calls = count(TOOL_CALL_PATTERNS, text)
    errors = count(ERROR_PATTERNS, text)
    mistake_mentions = count(MISTAKE_PATTERNS, text)

    print(f"\n| Run | Tool calls (proxy for steps) | Errors/retries | Mistake signal |")
    print(f"|-----|------------------------------|----------------|----------------|")
    print(f"| {args.label} | {tool_calls} | {errors} | {mistake_mentions} |\n")
    print("Note: these are heuristic counts from the transcript. Verify by eye for the demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
