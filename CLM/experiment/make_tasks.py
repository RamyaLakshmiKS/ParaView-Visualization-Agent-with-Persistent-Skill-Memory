#!/usr/bin/env python3
"""
make_tasks.py — build Context-as-a-File stress tasks from the REAL data/ volumes.

This turns the project's actual volume data into a context-management testbed
faithful in SHAPE to the paper's ContextBench diagnostic (needle retention,
multi-source lookup, distractor rejection) — Appendix D of arXiv:2609.37725 —
but built entirely from OUR data's real metadata. Nothing here is invented: the
dtype, dimensions, byte count and measured scalar range of each volume are read
from the real bytes. We then wrap each volume's facts in a bulky, realistic
"inspection dump" (the kind of tool output an agent actually receives) and plant
DISTRACTOR lines the agent must reject.

Outputs, per task, under CLM/experiment/tasks/<task_id>/:
  - evidence/<name>.txt   : one bulky dump per relevant volume (the external info)
  - question.txt          : the task prompt (goal only; no procedure prescribed)
  - answer_key.json       : deterministic ground truth + the planted "needle"
                            facts that must survive compaction

Tiers (increasing context pressure / reasoning):
  T1  single file, fits the budget           -> control
  T2  several files, near the budget         -> must choose what to keep
  T3  many files + distractors, over budget, -> must offload/compact AND revisit
      answer needs a fact seen early

NO FLOPs, no efficiency metrics. This only prepares inputs + ground truth.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
TASKS_DIR = HERE / "tasks"

DTYPE_BYTES = {"uint8": 1, "uint16-le": 2, "uint16-be": 2, "float32": 4}
KNOWN_LAYOUTS = {
    "bonsai": ("uint8", (256, 256, 256)),
    "engine": ("uint8", (256, 256, 128)),
    "mri_woman": ("uint16-le", (256, 256, 109)),
}


def parse_name_layout(name: str, nbytes: int):
    import re
    m = re.search(r"(\d+)x(\d+)x(\d+)", name)
    if not m:
        return None
    w, h, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    low = name.lower()
    if "uint16" in low or "u16" in low or "int16" in low:
        dtype = "uint16-be" if ("_be" in low or "bigend" in low) else "uint16-le"
    elif "float32" in low or "f32" in low:
        dtype = "float32"
    else:
        dtype = "uint8"
    if w * h * d * DTYPE_BYTES.get(dtype, 1) != nbytes:
        return None
    return dtype, (w, h, d)


def measure_range(path: pathlib.Path, dtype: str) -> Tuple[float, float]:
    """Real scalar min/max from the bytes (sampled for very large files)."""
    import numpy as np
    npd = {"uint8": "<u1", "uint16-le": "<u2", "uint16-be": ">u2", "float32": "<f4"}[dtype]
    width = DTYPE_BYTES[dtype]
    nbytes = path.stat().st_size
    # sample up to ~64MB for speed on multi-GB files; range is stable under sampling here
    cap = min(nbytes, 64_000_000)
    cap -= cap % width
    with open(path, "rb") as fh:
        buf = fh.read(cap)
    arr = np.frombuffer(buf, dtype=npd)
    if arr.size == 0:
        return (0.0, 0.0)
    return (float(arr.min()), float(arr.max()))


def resolve(path: pathlib.Path) -> Dict:
    name = path.stem
    nbytes = path.stat().st_size
    if name in KNOWN_LAYOUTS:
        dtype, dims = KNOWN_LAYOUTS[name]
        src = "known-layout"
    else:
        nl = parse_name_layout(name, nbytes)
        if nl:
            dtype, dims, src = nl[0], nl[1], "filename"
        else:
            dtype, dims, src = "uint8", (256, 256, nbytes // (256 * 256)), "inference"
    vmin, vmax = measure_range(path, dtype)
    return {"name": name, "nbytes": nbytes, "dtype": dtype, "dims": dims,
            "range": [vmin, vmax], "bits": 16 if "16" in dtype else (32 if "32" in dtype else 8),
            "layout_source": src}


def dump_for(info: Dict, with_distractor: bool, seed: int) -> Tuple[str, Dict]:
    """A bulky, realistic inspection dump for one volume.

    Returns (text, planted) where `planted` records the needle facts that MUST
    be preserved. The true facts are real; distractors are clearly-wrong extra
    lines the agent must not believe.
    """
    import random
    rng = random.Random(f"{info['name']}-{seed}")
    name = info["name"]
    nb = info["nbytes"]
    dtype = info["dtype"]
    w, h, d = info["dims"]
    vmin, vmax = info["range"]

    lines: List[str] = []
    lines.append(f"=== INSPECTION REPORT: {name}.raw ===")
    lines.append(f"path: data/{name}.raw")
    lines.append(f"file_size_bytes: {nb}")
    # bulky hex preview (realistic noise)
    for i in range(40):
        lines.append(f"byte[{i:03d}]=0x{(nb // (i + 3)) % 256:02x}")
    # candidate interpretations (the real resolution + noise)
    lines.append(f"candidate: {dtype} {w}x{h}x{d} (fits {nb} bytes) [SELECTED]")
    lines.append(f"candidate: uint8 256x256x{nb // (256*256)} (byte-count guess)")
    # a long pseudo-histogram (the bulk a naive agent would keep)
    for b in range(48):
        lines.append(f"hist_bin[{b:02d}]={ (nb // 101 - b * 777) % 1000 }")
    # THE KEY FACTS (needles) — real, must survive compaction
    key_dtype = f"RESOLVED dtype of {name} = {dtype} ({info['bits']}-bit)"
    key_range = f"RESOLVED scalar range of {name} = [{vmin:g}, {vmax:g}]"
    key_dims = f"RESOLVED dimensions of {name} = {w}x{h}x{d}"
    lines += [key_dtype, key_range, key_dims]
    lines.append(f"note: {name} voxels = {w*h*d}")

    planted = {"dtype": dtype, "bits": info["bits"], "range": [vmin, vmax],
               "dims": [w, h, d], "voxels": w * h * d}

    if with_distractor:
        # plausible-but-false lines the agent must reject (clearly marked as
        # superseded/unverified so a careful reader rejects them)
        fake_dtype = "uint8" if "16" in dtype else "uint16-le"
        fake_hi = int(vmax * 2 + 7)
        lines.append(f"UNVERIFIED rumor: {name} might be {fake_dtype} (OLD note, superseded)")
        lines.append(f"DEPRECATED: earlier run reported range [0,{fake_hi}] for {name} (WRONG, pre-fix)")
        rng.shuffle  # keep deterministic structure; distractors appended at end on purpose

    return "\n".join(lines) + "\n", planted


def build_tasks(seed: int = 0) -> List[Dict]:
    infos = [resolve(p) for p in sorted(DATA_DIR.glob("*.raw"))]
    by_name = {i["name"]: i for i in infos}
    tasks: List[Dict] = []

    # choose some real volumes known to be 16-bit vs 8-bit for gradable questions
    uint16 = [i["name"] for i in infos if i["bits"] == 16]
    uint8 = [i["name"] for i in infos if i["bits"] == 8]

    # ---- T1: single file, fits budget (control) ----
    t1_file = uint8[0] if uint8 else infos[0]["name"]
    tasks.append({
        "task_id": "T1_single",
        "tier": 1,
        "files": [t1_file],
        "distractor": False,
        "question": (f"Using only the inspection evidence provided on disk, state the "
                     f"resolved dtype and the measured scalar range of the volume "
                     f"'{t1_file}'. Answer with a single line: "
                     f"dtype=<...>; range=[<min>,<max>]."),
        "answer_key": {"kind": "single", "file": t1_file,
                       "dtype": by_name[t1_file]["dtype"],
                       "range": by_name[t1_file]["range"]},
    })

    # ---- T2: multi-file, near budget (choose what to keep) ----
    t2_files = (uint16[:1] + uint8[:3]) if uint16 else uint8[:4]
    tasks.append({
        "task_id": "T2_multi",
        "tier": 2,
        "files": t2_files,
        "distractor": False,
        "question": (f"Across the volumes {t2_files}, list every one whose resolved "
                     f"dtype is 16-bit, with its measured scalar range. If none are "
                     f"16-bit, say 'none'. Answer as lines 'name: range=[min,max]'."),
        "answer_key": {"kind": "filter16", "files": t2_files,
                       "truth": {n: by_name[n]["range"] for n in t2_files
                                 if by_name[n]["bits"] == 16}},
    })

    # ---- T3: many files + distractors, over budget, requires revisit ----
    # Put the needed-first fact early, then pile on more files + distractors so
    # the agent must offload/compact and later revisit the early fact.
    t3_files = infos  # ALL volumes -> guaranteed to exceed a 6K live budget
    t3_names = [i["name"] for i in t3_files]
    anchor = t3_names[0]  # the "early" fact we will ask about last
    tasks.append({
        "task_id": "T3_revisit_distractor",
        "tier": 3,
        "files": t3_names,
        "distractor": True,
        "question": (
            "You are given inspection dumps for MANY volumes, some containing "
            "UNVERIFIED/DEPRECATED distractor lines you must ignore. "
            "Step 1: determine how many volumes are 16-bit. "
            f"Step 2: THEN report the resolved dtype, dimensions, and voxel count "
            f"of the FIRST volume you were given, '{anchor}'. "
            "Answer with: count_16bit=<N>; "
            f"{anchor}: dtype=<...>, dims=<WxHxD>, voxels=<N>."),
        "answer_key": {
            "kind": "revisit",
            "count_16bit": sum(1 for i in t3_files if i["bits"] == 16),
            "anchor": anchor,
            "anchor_dtype": by_name[anchor]["dtype"],
            "anchor_dims": by_name[anchor]["dims"],
            "anchor_voxels": by_name[anchor]["dims"][0] * by_name[anchor]["dims"][1] * by_name[anchor]["dims"][2],
        },
    })

    # ---- Track A native-use-case task (sanity): metadata QA over all files ----
    tasks.append({
        "task_id": "A_native_metadata",
        "tier": 2,
        "files": [i["name"] for i in infos],
        "distractor": False,
        "question": ("For the volume-rendering pipeline, you must pick a background "
                     "opacity cutoff per volume as a FRACTION of its measured range. "
                     "First identify which volumes are 16-bit (they need special dtype "
                     "handling). Report: count_16bit=<N>; and list each 16-bit volume "
                     "as 'name: range=[min,max]'."),
        "answer_key": {"kind": "filter16", "files": [i["name"] for i in infos],
                       "truth": {i["name"]: i["range"] for i in infos if i["bits"] == 16}},
    })

    return tasks, by_name


def write_tasks(seed: int = 0) -> None:
    tasks, by_name = build_tasks(seed)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for t in tasks:
        tdir = TASKS_DIR / t["task_id"]
        edir = tdir / "evidence"
        edir.mkdir(parents=True, exist_ok=True)
        planted_all = {}
        for fname in t["files"]:
            text, planted = dump_for(by_name[fname], t["distractor"], seed)
            (edir / f"{fname}.txt").write_text(text)
            planted_all[fname] = planted
        (tdir / "question.txt").write_text(t["question"] + "\n")
        key = dict(t["answer_key"])
        key["planted"] = planted_all
        (tdir / "answer_key.json").write_text(json.dumps(key, indent=2))
        manifest.append({"task_id": t["task_id"], "tier": t["tier"],
                         "n_files": len(t["files"]), "distractor": t["distractor"]})
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(tasks)} tasks to {TASKS_DIR}")
    for m in manifest:
        print(" ", m)


if __name__ == "__main__":
    write_tasks()
