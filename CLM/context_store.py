#!/usr/bin/env python3
"""
context_store.py — the core "context as a file" mechanism from Context Language
Models (CLMs, Shao et al., arXiv:2609.37725).

Paper idea (plain words): a normal LM can only APPEND to its context. A CLM
instead mirrors its context into a FILE it may freely rewrite — delete stale
lines, compress raw tool dumps into one-line facts, reorder by importance — and
every edit is synced straight back into the live context for the next turn.

This module is that mechanism, scoped to one visualization task and expressed as
plain file operations an agent (or a human) can drive with ordinary tools. It is
deliberately small, dependency-free (standard library only), and self-contained
inside the CLM/ folder.

Scope boundary (honest):
  - This implements the "context as a file" working-memory mechanism (the paper's
    Section 4.1) and lets us MEASURE working-context size (the paper's efficiency
    axis).
  - It does NOT implement the RL (stepwise GRPO) or the Suffix Cache Reuse serving
    work — those require weight access and a custom inference server, which this
    project does not have. See CLM/README.md.

Relationship to the existing project:
  - memory/memory.json  = long-term, cross-task LESSONS (persists across tasks).
  - CLM working context  = short-lived, within-task WORKING MEMORY (disposable).
  They are complementary; this module never touches memory/memory.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The canonical sections of a working-context file. Keeping a fixed, small set of
# sections is what lets the agent compress aggressively without losing the facts
# that actually drive the next render.
DEFAULT_SECTIONS: List[str] = [
    "Goal",
    "Resolved facts",
    "Current TF",
    "Open issues",
    "Discard log",
]

# A soft cap. The CLM behavior we want to encourage is: when the working context
# grows past this, compress rather than keep appending.
DEFAULT_MAX_LINES = 40


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class WorkingContext:
    """A mirror of the agent's working context, backed by a Markdown file.

    The file IS the context. Any edit here is meant to be synced back into the
    model's live context on the next turn (that syncing is done by the agent
    harness / CLI; this class just owns the file and the edit operations).
    """

    path: str
    sections: Dict[str, List[str]] = field(default_factory=dict)
    max_lines: int = DEFAULT_MAX_LINES

    # ---- construction -----------------------------------------------------
    @classmethod
    def new(cls, path: str, goal: str = "", max_lines: int = DEFAULT_MAX_LINES) -> "WorkingContext":
        wc = cls(path=path, sections={s: [] for s in DEFAULT_SECTIONS}, max_lines=max_lines)
        if goal:
            wc.sections["Goal"] = [goal.strip()]
        return wc

    @classmethod
    def load(cls, path: str) -> "WorkingContext":
        """Parse an existing working-context Markdown file back into sections."""
        if not os.path.exists(path):
            return cls.new(path)
        sections: Dict[str, List[str]] = {}
        current: Optional[str] = None
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh.read().splitlines():
                line = raw.rstrip("\n")
                if line.startswith("## "):
                    current = line[3:].strip()
                    sections.setdefault(current, [])
                elif current is not None:
                    stripped = line.strip()
                    # keep content lines (drop blank separators and bullet dashes)
                    if stripped and stripped != "-":
                        sections[current].append(stripped.lstrip("- ").rstrip())
        # ensure canonical sections exist
        for s in DEFAULT_SECTIONS:
            sections.setdefault(s, [])
        return cls(path=path, sections=sections)

    # ---- edit operations (the "unrestricted update" of a CLM) -------------
    def set_section(self, section: str, lines: List[str]) -> None:
        """Replace a whole section. This is the REWRITE primitive — the heart of
        CLM: prefer rewriting over appending."""
        self.sections[section] = [l.strip() for l in lines if l.strip()]

    def append(self, section: str, line: str) -> None:
        """Append one line to a section (the ONLY thing a plain LM can do)."""
        self.sections.setdefault(section, []).append(line.strip())

    def discard(self, section: str, reason: str) -> None:
        """Clear a section and log one line about why — the paper's 'creative
        compaction': a raw dump collapses into a single conclusion."""
        dropped = len(self.sections.get(section, []))
        self.sections[section] = []
        self.append("Discard log", f"cleared '{section}' ({dropped} lines) — {reason}")

    def compress_fact(self, raw_lines: List[str], one_liner: str) -> None:
        """Replace many raw observation lines with a single resolved fact.
        Example: a 300-line histogram dump -> 'range [2,4095], 12-bit, bg < ~5%'."""
        self.append("Resolved facts", one_liner)
        self.append("Discard log", f"compressed {len(raw_lines)} raw lines -> 1 fact")

    # ---- measurement (the paper's efficiency axis) ------------------------
    def line_count(self) -> int:
        return sum(len(v) for v in self.sections.values())

    def char_count(self) -> int:
        return len(self.render())

    def est_tokens(self) -> int:
        """Rough token estimate (~4 chars/token). A local stand-in for the
        paper's FLOPs axis; see README for why this is a proxy, not real FLOPs."""
        return max(1, round(self.char_count() / 4))

    def over_budget(self) -> bool:
        return self.line_count() > self.max_lines

    # ---- serialization ----------------------------------------------------
    def render(self) -> str:
        out: List[str] = []
        for s in DEFAULT_SECTIONS + [x for x in self.sections if x not in DEFAULT_SECTIONS]:
            if s not in self.sections:
                continue
            out.append(f"## {s}")
            if self.sections[s]:
                out.extend(f"- {line}" for line in self.sections[s])
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(self.render())


# ---------------------------------------------------------------------------
# CLI — lets a human or an agent harness drive the file with ordinary commands.
# ---------------------------------------------------------------------------
def _cmd_init(args: argparse.Namespace) -> int:
    wc = WorkingContext.new(args.path, goal=args.goal or "", max_lines=args.max_lines)
    wc.save()
    print(f"initialized working context at {args.path}")
    print(f"  sections: {', '.join(DEFAULT_SECTIONS)}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    wc = WorkingContext.load(args.path)
    print(wc.render())
    return 0


def _cmd_set(args: argparse.Namespace) -> int:
    wc = WorkingContext.load(args.path)
    wc.set_section(args.section, args.line)
    wc.save()
    print(f"set section '{args.section}' ({len(args.line)} line(s))")
    return 0


def _cmd_append(args: argparse.Namespace) -> int:
    wc = WorkingContext.load(args.path)
    wc.append(args.section, args.text)
    wc.save()
    print(f"appended to '{args.section}'")
    return 0


def _cmd_discard(args: argparse.Namespace) -> int:
    wc = WorkingContext.load(args.path)
    wc.discard(args.section, args.reason or "no longer relevant")
    wc.save()
    print(f"discarded '{args.section}'")
    return 0


def _cmd_size(args: argparse.Namespace) -> int:
    wc = WorkingContext.load(args.path)
    status = "OVER BUDGET" if wc.over_budget() else "ok"
    print(
        f"lines={wc.line_count()} chars={wc.char_count()} "
        f"est_tokens={wc.est_tokens()} budget={wc.max_lines} [{status}]"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="context_store",
        description="Context-as-a-file working-context manager (CLM mechanism).",
    )
    p.add_argument(
        "--path",
        default=os.path.join(os.path.dirname(__file__), "context", "working.md"),
        help="path to the working-context file (default: CLM/context/working.md)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="create a fresh working-context file")
    sp.add_argument("--goal", help="one-line task goal")
    sp.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    sp.set_defaults(func=_cmd_init)

    sp = sub.add_parser("show", help="print the current working context")
    sp.set_defaults(func=_cmd_show)

    sp = sub.add_parser("set", help="REWRITE a whole section (CLM primitive)")
    sp.add_argument("section")
    sp.add_argument("line", nargs="+", help="replacement line(s)")
    sp.set_defaults(func=_cmd_set)

    sp = sub.add_parser("append", help="append one line to a section")
    sp.add_argument("section")
    sp.add_argument("text")
    sp.set_defaults(func=_cmd_append)

    sp = sub.add_parser("discard", help="clear a section and log why")
    sp.add_argument("section")
    sp.add_argument("--reason")
    sp.set_defaults(func=_cmd_discard)

    sp = sub.add_parser("size", help="measure working-context size (efficiency axis)")
    sp.set_defaults(func=_cmd_size)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
