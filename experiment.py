#!/usr/bin/env python3
"""
experiment.py — deterministic harness for the memory ON/OFF experiment.

A Kiro skill cannot by itself guarantee that a "memory OFF" run is genuinely cold,
or snapshot artifacts between phases. This script owns that bookkeeping so the
experiment is reproducible and honest. The agent calls these subcommands (via the
/run-experiment skill) and does the actual rendering work in between.

Phases of the experiment:
  1. reset                 -> empty memory, archive old run artifacts
  2. phase begin --name X  -> set memory ON or OFF for the upcoming agent run,
                              snapshot results/ so we can detect what this run produced
  3. phase end   --name X  -> capture memory + results produced during the phase
  4. compare               -> read captured phases, print the ON vs OFF table

State lives under experiment/ :
  experiment/
    memory_off_backup.json   # memory parked here during an OFF phase
    phases/<name>/
      mode.txt               # "on" or "off"
      memory_before.json
      memory_after.json
      results_manifest.txt   # files present in results/ after the phase
      transcript.txt         # optional: paste the agent transcript here

Usage examples (the skill issues these):
  python experiment.py reset
  python experiment.py phase-begin --name A_bonsai_on   --mode on
  python experiment.py phase-end   --name A_bonsai_on
  python experiment.py phase-begin --name B_engine_on   --mode on
  python experiment.py phase-end   --name B_engine_on
  python experiment.py phase-begin --name B_engine_off  --mode off
  python experiment.py phase-end   --name B_engine_off
  python experiment.py compare --on B_engine_on --off B_engine_off
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEMORY = ROOT / "memory" / "memory.json"
RESULTS = ROOT / "results"
EXP = ROOT / "experiment"
PHASES = EXP / "phases"
OFF_BACKUP = EXP / "memory_off_backup.json"

EMPTY_MEMORY = {"episodic": [], "semantic": [], "corrective": []}


def _load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default if default is not None else {}


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def _ensure_dirs():
    PHASES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MEMORY.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- reset
def cmd_reset(_args):
    _ensure_dirs()
    # archive previous experiment run, if any
    if EXP.exists() and any(EXP.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = ROOT / f"experiment_archive_{stamp}"
        shutil.copytree(EXP, archive)
        print(f"Archived previous experiment -> {archive.name}")
    shutil.rmtree(PHASES, ignore_errors=True)
    PHASES.mkdir(parents=True, exist_ok=True)
    if OFF_BACKUP.exists():
        OFF_BACKUP.unlink()
    _write_json(MEMORY, EMPTY_MEMORY)
    print("Memory reset to empty. Experiment state cleared.")
    print('memory/memory.json = {"episodic":[],"semantic":[],"corrective":[]}')
    return 0


# ------------------------------------------------------------------------ phase-begin
def cmd_phase_begin(args):
    _ensure_dirs()
    mode = args.mode.lower()
    if mode not in ("on", "off"):
        print("mode must be 'on' or 'off'")
        return 2

    pdir = PHASES / args.name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "mode.txt").write_text(mode)

    if mode == "off":
        # Genuinely cold: park current memory out of reach, present an empty store.
        current = _load_json(MEMORY, EMPTY_MEMORY)
        _write_json(OFF_BACKUP, current)
        _write_json(MEMORY, EMPTY_MEMORY)
        print(f"[{args.name}] mode=OFF  -> memory parked at {OFF_BACKUP.relative_to(ROOT)}; "
              f"store is empty for this run.")
        print("IMPORTANT: for a true OFF run, do NOT invoke /recall-memory or "
              "/reflect-and-save. Give only the goal-only task to the agent.")
    else:
        print(f"[{args.name}] mode=ON   -> using existing memory "
              f"({_summary(_load_json(MEMORY, EMPTY_MEMORY))}).")
        print("Invoke /recall-memory before the task and /reflect-and-save after it.")

    # snapshot memory + results state at the start of the phase
    _write_json(pdir / "memory_before.json", _load_json(MEMORY, EMPTY_MEMORY))
    (pdir / "results_before.txt").write_text(_results_manifest())
    print(f"[{args.name}] phase begun. Now run the task, then call: "
          f"python experiment.py phase-end --name {args.name}")
    return 0


# -------------------------------------------------------------------------- phase-end
def cmd_phase_end(args):
    pdir = PHASES / args.name
    if not pdir.exists():
        print(f"No such phase: {args.name} (run phase-begin first)")
        return 2
    mode = (pdir / "mode.txt").read_text().strip()

    # capture what the agent produced
    _write_json(pdir / "memory_after.json", _load_json(MEMORY, EMPTY_MEMORY))
    (pdir / "results_manifest.txt").write_text(_results_manifest())

    if mode == "off":
        # restore the real memory that was parked
        if OFF_BACKUP.exists():
            restored = _load_json(OFF_BACKUP, EMPTY_MEMORY)
            _write_json(MEMORY, restored)
            OFF_BACKUP.unlink()
            print(f"[{args.name}] OFF phase ended -> real memory restored "
                  f"({_summary(restored)}).")
    else:
        print(f"[{args.name}] ON phase ended -> memory now "
              f"({_summary(_load_json(MEMORY, EMPTY_MEMORY))}).")

    print(f"[{args.name}] captured. If you saved a transcript, place it at "
          f"{(pdir / 'transcript.txt').relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------- compare
# Heuristic transcript signals (same spirit as metrics.py).
TOOL_CALL_PATTERNS = [r"\bRunning tool\b", r"\btool_use\b", r"\bBash\b",
                      r"\bWrite\b", r"\bEdit\b", r"\bexecute\b"]
ERROR_PATTERNS = [r"\bTraceback\b", r"\bError\b", r"\bfailed\b",
                  r"\bretry(ing)?\b", r"\bExcept(ion)?\b"]
MISTAKE_PATTERNS = [r"color\s*bar", r"scalar\s*bar", r"orientation\s*axes"]
RENDER_PATTERNS = [r"\.Render\(", r"vtkPNGWriter", r"results/.*\.png"]


def _count(patterns, text):
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def _phase_metrics(name):
    pdir = PHASES / name
    out = {"name": name, "exists": pdir.exists()}
    if not pdir.exists():
        return out
    out["mode"] = (pdir / "mode.txt").read_text().strip() if (pdir / "mode.txt").exists() else "?"
    after = _load_json(pdir / "memory_after.json", EMPTY_MEMORY)
    out["mem_episodic"] = len(after.get("episodic", []))
    out["mem_semantic"] = len(after.get("semantic", []))
    out["mem_corrective"] = len(after.get("corrective", []))
    tpath = pdir / "transcript.txt"
    if tpath.exists():
        text = tpath.read_text(encoding="utf-8", errors="ignore")
        out["tool_calls"] = _count(TOOL_CALL_PATTERNS, text)
        out["errors"] = _count(ERROR_PATTERNS, text)
        out["mistake_signals"] = _count(MISTAKE_PATTERNS, text)
        out["render_signals"] = _count(RENDER_PATTERNS, text)
        out["has_transcript"] = True
    else:
        out["has_transcript"] = False
    return out


def cmd_compare(args):
    on = _phase_metrics(args.on)
    off = _phase_metrics(args.off)

    for p, label in [(on, "ON"), (off, "OFF")]:
        if not p.get("exists"):
            print(f"Missing phase '{p['name']}' ({label}). Run the phases first.")
            return 2

    def row(metric, key, note=""):
        o = on.get(key, "—")
        f = off.get(key, "—")
        return f"| {metric} | {o} | {f} | {note} |"

    print(f"\n### Memory ON vs OFF  (ON='{args.on}', OFF='{args.off}')\n")
    print("| Metric | memory ON | memory OFF | note |")
    print("|--------|-----------|------------|------|")
    if on.get("has_transcript") and off.get("has_transcript"):
        print(row("Tool calls (steps proxy)", "tool_calls", "lower is better"))
        print(row("Errors / retries", "errors", "lower is better"))
        print(row("Color-bar/axes mentions", "mistake_signals",
                  "in OFF, often the repeated mistake"))
        print(row("Render invocations", "render_signals", "fewer = fewer attempts"))
    else:
        print("| (no transcripts) | — | — | place transcript.txt in each phase dir for step/error metrics |")
    print(row("Memory corrective lessons after run", "mem_corrective"))
    print(row("Memory semantic rules after run", "mem_semantic"))

    print("\nInterpretation:")
    print("- The strongest evidence is QUALITATIVE: did the ON run avoid a mistake the")
    print("  OFF run made (e.g. color bar visible)? Confirm by eye in the PNGs/transcripts.")
    print("- Transcript counts are heuristic proxies, not a rigorous benchmark.")
    return 0


# ------------------------------------------------------------------------------ util
def _summary(mem):
    return (f"{len(mem.get('episodic', []))} episodic, "
            f"{len(mem.get('semantic', []))} semantic, "
            f"{len(mem.get('corrective', []))} corrective")


def _results_manifest():
    if not RESULTS.exists():
        return ""
    lines = []
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file():
            lines.append(f"{p.relative_to(ROOT)}\t{p.stat().st_size}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- status
def cmd_status(_args):
    mem = _load_json(MEMORY, EMPTY_MEMORY)
    print(f"memory/memory.json: {_summary(mem)}")
    if PHASES.exists():
        names = sorted(p.name for p in PHASES.iterdir() if p.is_dir())
        print(f"phases recorded: {', '.join(names) if names else '(none)'}")
    pairs = _discover_pairs()
    if pairs:
        print("ON/OFF pairs ready to compare:")
        for ds, on, off in pairs:
            print(f"  - {ds}:  ON={on}  OFF={off}")
    return 0


def _discover_pairs():
    """Find (dataset, on_phase, off_phase) triples from phase dir names B_<ds>_on/off."""
    if not PHASES.exists():
        return []
    on, off = {}, {}
    for p in PHASES.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if n.startswith("B_") and n.endswith("_on"):
            on[n[2:-3]] = n
        elif n.startswith("B_") and n.endswith("_off"):
            off[n[2:-4]] = n
    pairs = []
    for ds in sorted(set(on) & set(off)):
        pairs.append((ds, on[ds], off[ds]))
    return pairs


# --------------------------------------------------------------------- export-results
def cmd_export_results(_args):
    pairs = _discover_pairs()
    mem = _load_json(MEMORY, EMPTY_MEMORY)

    lines = []
    lines.append("# Results: Cross-Task Memory Transfer\n")
    lines.append("Auto-generated by `experiment.py export-results`. One section per "
                 "dataset that has both a memory-ON and a memory-OFF run.\n")
    lines.append("**Claim:** lessons learned on earlier datasets transfer to a new, "
                 "unseen dataset — the agent solving it with memory ON avoids mistakes "
                 "the cold (memory OFF) run makes. Memory persists across process "
                 "restarts.\n")

    if not pairs:
        lines.append("\n_No completed ON/OFF dataset pairs found yet. Run "
                     "`/run-experiment <dataset>`._\n")
    for ds, on_name, off_name in pairs:
        on = _phase_metrics(on_name)
        off = _phase_metrics(off_name)
        on_png = f"results/{ds}_on.png"
        off_png = f"results/{ds}_off.png"
        has_imgs = (ROOT / on_png).exists() and (ROOT / off_png).exists()

        lines.append(f"\n---\n\n## Dataset: `{ds}`\n")
        lines.append(f"Task goal: see `tasks/{ds}.txt`. Same goal both runs; only "
                     f"difference is whether prior memory was available.\n")

        if has_imgs:
            lines.append("| Memory ON | Memory OFF (cold) |")
            lines.append("|---|---|")
            lines.append(f"| ![{ds} on]({on_png}) | ![{ds} off]({off_png}) |")
            lines.append("")
        else:
            lines.append(f"> Images not found. Save renders to `{on_png}` and "
                         f"`{off_png}` to embed the visual comparison.\n")

        lines.append("| Metric | memory ON | memory OFF |")
        lines.append("|--------|-----------|------------|")
        if on.get("has_transcript") and off.get("has_transcript"):
            lines.append(f"| Mistake signals in transcript | {on.get('mistake_signals','—')} | {off.get('mistake_signals','—')} |")
            lines.append(f"| Render invocations | {on.get('render_signals','—')} | {off.get('render_signals','—')} |")
            lines.append(f"| Error/retry signals | {on.get('errors','—')} | {off.get('errors','—')} |")
        lines.append(f"| Lessons in memory after run (episodic/semantic/corrective) | "
                     f"{on.get('mem_episodic','—')}/{on.get('mem_semantic','—')}/{on.get('mem_corrective','—')} | "
                     f"{off.get('mem_episodic','—')}/{off.get('mem_semantic','—')}/{off.get('mem_corrective','—')} |")
        lines.append("")
        lines.append("_Read the two images side by side — the clearest evidence is a "
                     "visible defect in the OFF render that the ON render avoids._\n")

    lines.append("\n---\n\n## Current memory store\n")
    lines.append(f"`memory/memory.json`: **{_summary(mem)}**.\n")
    lines.append("Semantic rules are stored as range-relative *procedures* (transferable), "
                 "not dataset-specific constants.\n")
    lines.append("\n## Honest limitations\n")
    lines.append("- Phase transcripts are faithful run summaries, not raw machine logs; "
                 "the images and memory store are the hard evidence.\n")
    lines.append("- OFF runs record the first-attempt cold baseline. A cold agent would "
                 "eventually self-correct; the point is memory makes the ON agent correct "
                 "*immediately*, skipping trial-and-error.\n")
    lines.append("- One run per condition (representative, not a statistical benchmark).\n")

    out = ROOT / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out.name} covering {len(pairs)} dataset pair(s): "
          f"{', '.join(p[0] for p in pairs) if pairs else '(none)'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Memory ON/OFF experiment harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reset").set_defaults(func=cmd_reset)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("export-results").set_defaults(func=cmd_export_results)

    b = sub.add_parser("phase-begin")
    b.add_argument("--name", required=True)
    b.add_argument("--mode", required=True, choices=["on", "off"])
    b.set_defaults(func=cmd_phase_begin)

    e = sub.add_parser("phase-end")
    e.add_argument("--name", required=True)
    e.set_defaults(func=cmd_phase_end)

    c = sub.add_parser("compare")
    c.add_argument("--on", required=True)
    c.add_argument("--off", required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
