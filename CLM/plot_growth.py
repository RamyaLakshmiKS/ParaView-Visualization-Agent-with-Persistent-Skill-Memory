#!/usr/bin/env python3
"""
plot_growth.py — plot the long-horizon working-context growth curves.

Reads the two traces produced by longhorizon.py and draws naive-append vs
context-as-a-file (CLM) working-context size (estimated tokens) after each
dataset. The chart is the visual companion to the blog write-up.

Usage:
  python CLM/plot_growth.py
  python CLM/plot_growth.py --naive CLM/context/longhorizon_naive.json \
                            --clm   CLM/context/longhorizon_clm.json \
                            --out   CLM/context/longhorizon_growth.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CTX_DIR = os.path.join(HERE, "context")


def load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _short(label: str) -> str:
    # trace labels look like "after <name>"; strip the prefix and any _dims_dtype
    name = label[6:] if label.startswith("after ") else label
    for marker in ("_", " "):
        pass
    # keep it compact for the x axis
    name = name.split("_")[0]
    return name


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot long-horizon context growth curves.")
    ap.add_argument("--naive", default=os.path.join(CTX_DIR, "longhorizon_naive.json"))
    ap.add_argument("--clm", default=os.path.join(CTX_DIR, "longhorizon_clm.json"))
    ap.add_argument("--out", default=os.path.join(CTX_DIR, "longhorizon_growth.png"))
    args = ap.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}")
        print("Install with: pip install matplotlib")
        return 1

    naive = load(args.naive)
    clm = load(args.clm)

    n_tokens = [s["est_tokens"] for s in naive["trace"]]
    c_tokens = [s["est_tokens"] for s in clm["trace"]]
    labels = [_short(s["label"]) for s in naive["trace"]]
    x = list(range(1, len(n_tokens) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, n_tokens, marker="o", linewidth=2.4, color="#dc2626",
            label="naive-append (keeps everything)")
    ax.plot(x, c_tokens, marker="s", linewidth=2.4, color="#16a34a",
            label="context-as-a-file / CLM (compress + discard)")

    # annotate the end-of-horizon gap
    if n_tokens and c_tokens:
        gap = n_tokens[-1] / c_tokens[-1] if c_tokens[-1] else 0
        pct = (n_tokens[-1] - c_tokens[-1]) / n_tokens[-1] * 100 if n_tokens[-1] else 0
        ax.annotate(
            f"{gap:.1f}× smaller\n({pct:.0f}% less)",
            xy=(x[-1], c_tokens[-1]),
            xytext=(x[-1] - 2.2, (n_tokens[-1] + c_tokens[-1]) / 2),
            fontsize=11, fontweight="bold", color="#111827",
            arrowprops=dict(arrowstyle="->", color="#6b7280"),
        )

    ax.fill_between(x, c_tokens, n_tokens, color="#fca5a5", alpha=0.18)

    ax.set_title("Long-horizon working-context growth: naive vs Context Language Model",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("datasets processed in one session (horizon →)")
    ax.set_ylabel("working-context size (tokens, tiktoken cl100k_base)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    ax.margins(x=0.02)

    # caption with the honest proxy note
    fig.text(0.5, 0.005,
             "Real token counts (tiktoken cl100k_base). Working-context size is a local "
             "proxy for compute, not a FLOPs measurement. Same tool output fed to both; "
             "only context management differs.",
             ha="center", fontsize=8, color="#6b7280")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  naive: {n_tokens[0]} -> {n_tokens[-1]} est. tokens over {len(x)} datasets")
    print(f"  CLM  : {c_tokens[0]} -> {c_tokens[-1]} est. tokens over {len(x)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
