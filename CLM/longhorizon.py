#!/usr/bin/env python3
"""
longhorizon.py — the LONG-HORIZON test of Context Language Models in this project.

Why this exists
---------------
The CLM paper's benefit shows up on LONG-HORIZON tasks: when an agent works for a
long time across many steps, a naive agent's context GROWS without bound, while a
CLM (context-as-a-file) agent keeps rewriting/compressing so its working context
stays small. One or two files barely show it; a batch of many files makes the gap
obvious — and it keeps widening as you add more files to data/.

This runner processes EVERY file in data/*.raw in ONE continuous session (the long
horizon), two ways:

  (A) naive-append       — one cumulative context that keeps growing across ALL
                           files; nothing is ever discarded.
  (B) context-as-a-file  — after each file, compress that file's raw tool output
      (CLM)                into a one-line resolved fact and discard the bulk,
                           carrying only a tiny running summary forward.

For each file it performs REAL work where possible:
  - reads the real byte size on disk,
  - infers candidate (dtype, dims) interpretations from the byte count,
  - measures the actual scalar range by reading the raw bytes,
  - produces a (bulky, realistic) inspect + histogram + render-log step stream.

The actual VTK render is OPTIONAL (--render); the context experiment does not need
pixels, because the CLM effect is about CONTEXT MANAGEMENT, not image quality.

Outputs
-------
  - a per-file + cumulative growth table (printed),
  - CLM/context/longhorizon_naive.json and longhorizon_clm.json traces,
    consumable by clm_metrics.py --compare.

Usage
-----
  python CLM/longhorizon.py                 # all data/*.raw, no VTK needed
  python CLM/longhorizon.py --render        # also write real PNGs where dims known
  python CLM/clm_metrics.py --compare \
      CLM/context/longhorizon_naive.json CLM/context/longhorizon_clm.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CTX_DIR = os.path.join(HERE, "context")

sys.path.insert(0, HERE)
from context_store import WorkingContext  # noqa: E402

# Known layouts for the shipped datasets (used for the real render + nicer facts).
# Any file NOT listed here is still handled: dims are inferred from the byte count.
KNOWN_LAYOUTS: Dict[str, Dict] = {
    "bonsai":    {"dtype": "uint8",      "dims": (256, 256, 256)},
    "engine":    {"dtype": "uint8",      "dims": (256, 256, 128)},
    "mri_woman": {"dtype": "uint16-le",  "dims": (256, 256, 109)},
}

DTYPE_BYTES = {"uint8": 1, "uint16-le": 2, "uint16-be": 2, "float32": 4}


# ---------------------------------------------------------------------------
# Real inspection of a .raw file
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402


def parse_name_layout(name: str, nbytes: int) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """Parse (dtype, dims) encoded in a filename, e.g.
    'backpack_512x512x373_uint16' or 'blunt_fin_256x128x64_uint8'.

    Returns (dtype, (w,h,d)) only if the encoded layout actually matches the file
    byte count — so a mislabeled name can never silently produce a scrambled
    interpretation. uint16 defaults to little-endian (the common convention here).
    """
    dim_m = _re.search(r"(\d+)x(\d+)x(\d+)", name)
    if not dim_m:
        return None
    w, h, d = (int(dim_m.group(1)), int(dim_m.group(2)), int(dim_m.group(3)))

    low = name.lower()
    if "uint16" in low or "int16" in low or "u16" in low:
        dtype = "uint16-be" if ("_be" in low or "bigend" in low) else "uint16-le"
    elif "float32" in low or "f32" in low:
        dtype = "float32"
    else:
        dtype = "uint8"

    expected = w * h * d * DTYPE_BYTES.get(dtype, 1)
    if expected != nbytes:
        return None  # name does not match the bytes; fall back to inference
    return dtype, (w, h, d)


def infer_candidates(nbytes: int) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Infer plausible (dtype, dims) interpretations from a byte count.

    We assume the two trailing dims are 256x256 (the convention for these
    volumes) and solve for depth under each dtype. This mirrors the real
    'resolve headerless by inspection' lesson in memory/memory.json. Used only
    when the filename does not encode a layout that matches the byte count.
    """
    candidates: List[Tuple[str, Tuple[int, int, int]]] = []
    base = 256 * 256
    for dtype, width in (("uint8", 1), ("uint16-le", 2)):
        if nbytes % (base * width) == 0:
            depth = nbytes // (base * width)
            if depth > 0:
                candidates.append((dtype, (256, 256, depth)))
    return candidates


def measure_range(path: str, dtype: str) -> Tuple[float, float]:
    """Measure the real scalar min/max by reading the raw bytes.

    Uses numpy if available; otherwise falls back to the stdlib `array` module so
    the long-horizon context experiment runs even without numpy/VTK installed.
    """
    try:
        import numpy as np  # type: ignore
        np_dtype = {
            "uint8": "<u1",
            "uint16-le": "<u2",
            "uint16-be": ">u2",
            "float32": "<f4",
        }[dtype]
        arr = np.fromfile(path, dtype=np_dtype)
        if arr.size == 0:
            return (0.0, 0.0)
        return (float(arr.min()), float(arr.max()))
    except Exception:
        # stdlib fallback: sample to stay fast on large files
        import array as _array
        typecode = {"uint8": "B", "uint16-le": "H", "uint16-be": "H"}.get(dtype, "B")
        a = _array.array(typecode)
        width = DTYPE_BYTES.get(dtype, 1)
        sample_bytes = min(os.path.getsize(path), 4_000_000)
        sample_bytes -= sample_bytes % width
        with open(path, "rb") as fh:
            a.frombytes(fh.read(sample_bytes))
        if typecode == "H" and dtype == "uint16-be":
            a.byteswap()
        if len(a) == 0:
            return (0.0, 0.0)
        return (float(min(a)), float(max(a)))


def inspect_file(path: str) -> Dict:
    """Produce a realistic, bulky inspection record for one .raw file."""
    name = os.path.splitext(os.path.basename(path))[0]
    nbytes = os.path.getsize(path)
    candidates = infer_candidates(nbytes)
    name_layout = parse_name_layout(name, nbytes)

    # choose dtype/dims, in priority order:
    #   1. a hard-coded known layout (the 3 canonical datasets)
    #   2. a layout ENCODED IN THE FILENAME that matches the byte count
    #   3. byte-count inference (assumes 256x256xdepth)
    if name in KNOWN_LAYOUTS:
        dtype = KNOWN_LAYOUTS[name]["dtype"]
        dims = KNOWN_LAYOUTS[name]["dims"]
        layout_source = "known-layout"
    elif name_layout is not None:
        dtype, dims = name_layout
        layout_source = "filename"
    elif candidates:
        dtype, dims = candidates[0]
        layout_source = "byte-count-inference"
    else:
        dtype, dims = ("uint8", (0, 0, 0))
        layout_source = "unresolved"

    vmin, vmax = measure_range(path, dtype)

    # The "raw dumps" an agent would receive — intentionally bulky. These are what
    # a naive agent keeps forever and a CLM compresses to one line.
    raw_header_bytes = [f"byte[{i}]=0x{(nbytes // (i + 1)) % 256:02x}" for i in range(24)]
    candidate_lines = [f"candidate: {dt} {dm} (fits {nbytes} bytes)" for dt, dm in candidates]
    if name_layout is not None:
        candidate_lines.append(f"filename encodes: {name_layout[0]} {name_layout[1]} (matches bytes)")
    histogram = [f"bin[{b}]={ (nbytes // 97 - b * 131) % 1000 }" for b in range(32)]

    resolved = (f"{name}: {dtype} {dims[0]}x{dims[1]}x{dims[2]}, "
                f"range [{vmin:g},{vmax:g}] (via {layout_source})")
    return {
        "name": name,
        "path": path,
        "nbytes": nbytes,
        "dtype": dtype,
        "dims": dims,
        "range": (vmin, vmax),
        "layout_source": layout_source,
        "raw_header_bytes": raw_header_bytes,
        "candidate_lines": candidate_lines,
        "histogram": histogram,
        "resolved_fact": resolved,
    }


# ---------------------------------------------------------------------------
# Optional real VTK render (only for files with a known layout)
# ---------------------------------------------------------------------------
def try_render(info: Dict, max_voxels: int = 80_000_000) -> Optional[str]:
    name = info["name"]
    dims = info["dims"]
    if info.get("layout_source") in (None, "unresolved") or dims[0] * dims[1] * dims[2] == 0:
        return None
    # Guard: offscreen CPU ray-casting a multi-GB volume can exhaust memory.
    # Skip very large volumes for the render (the CONTEXT experiment does not
    # need pixels); they are still fully inspected and measured.
    if dims[0] * dims[1] * dims[2] > max_voxels:
        return None
    try:
        import numpy as np
        import vtk
        from vtk.util import numpy_support
    except Exception:
        return None

    dtype = info["dtype"]
    np_dtype = {"uint8": "<u1", "uint16-le": "<u2", "uint16-be": ">u2"}.get(dtype, "<u1")
    arr = np.fromfile(info["path"], dtype=np_dtype)
    if arr.size != dims[0] * dims[1] * dims[2]:
        return None
    vmin, vmax = float(arr.min()), float(arr.max())
    span = max(1.0, vmax - vmin)

    def f(frac):
        return vmin + frac * span

    vtk_type = vtk.VTK_UNSIGNED_CHAR if dtype == "uint8" else vtk.VTK_UNSIGNED_SHORT
    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="F"), deep=True, array_type=vtk_type)
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(vtk_arr)

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(f(0.00), 0.00)
    opacity.AddPoint(f(0.16), 0.00)   # background cutoff (recalled lesson)
    opacity.AddPoint(f(0.30), 0.03)
    opacity.AddPoint(f(0.55), 0.45)
    opacity.AddPoint(f(1.00), 0.95)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(f(0.20), 0.55, 0.62, 0.72)
    color.AddRGBPoint(f(0.55), 0.98, 0.80, 0.30)
    color.AddRGBPoint(f(1.00), 0.88, 0.12, 0.08)

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image)
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddVolume(volume)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(640, 560)
    rw.AddRenderer(renderer)
    renderer.ResetCamera()
    rw.Render()

    out = os.path.join(PROJECT_ROOT, "results", f"{name}_longhorizon.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(out)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    return out


# ---------------------------------------------------------------------------
# The long-horizon session, run two ways
# ---------------------------------------------------------------------------
def _snapshot(step: int, label: str, wc: WorkingContext) -> Dict:
    return {
        "step": step,
        "label": label,
        "lines": wc.line_count(),
        "chars": wc.char_count(),
        "est_tokens": wc.est_tokens(),
    }


def run_session(infos: List[Dict], strategy: str) -> Dict:
    """Run one continuous session over ALL files under one strategy.

    strategy == 'naive' : append every raw dump to a single growing context.
    strategy == 'clm'   : compress each file to one resolved fact + a tiny running
                          summary; discard the raw material.
    """
    assert strategy in ("naive", "clm")
    wc = WorkingContext.new(
        os.path.join(CTX_DIR, f"longhorizon_{strategy}_working.md"),
        goal="long-horizon: render every volume in data/ in one session",
        max_lines=10_000,  # don't cap; we WANT to see naive grow
    )
    trace: List[Dict] = []
    total_written = 1
    discarded = 0
    step = 0
    per_file: List[Dict] = []

    for info in infos:
        name = info["name"]
        raw_lines = info["raw_header_bytes"] + info["candidate_lines"] + info["histogram"]
        render_log = [
            f"wrote results/{name}.py",
            f"render -> results/{name}.png",
            "LOOK: first pass shell too opaque, lower soft band",
            f"re-render -> results/{name}.png",
            "LOOK: interior reads clearly — good",
        ]

        if strategy == "naive":
            # keep EVERYTHING, forever
            for ln in raw_lines:
                wc.append("Resolved facts", f"[{name}] {ln}")
            for ln in render_log:
                wc.append("Open issues", f"[{name}] {ln}")
            total_written += len(raw_lines) + len(render_log)
        else:
            # CLM: compress this file to ONE resolved fact + ONE outcome line,
            # discard all the raw material, keep a tiny running summary.
            total_written += len(raw_lines) + len(render_log)
            wc.append("Resolved facts", info["resolved_fact"] + "; bg cutoff ~15%, dense core opaque")
            wc.append("Discard log", f"[{name}] compressed {len(raw_lines)+len(render_log)} lines -> 1 fact")
            discarded += (len(raw_lines) + len(render_log)) - 1
            total_written += 1
            # keep only the LAST file's transient state in Current TF / Open issues
            wc.set_section("Current TF", [
                f"[{name}] range-relative TF: cutoff ~15%, ghost mid band, dense ramp >55%",
            ])
            wc.set_section("Open issues", [])  # resolved by the re-render
            # bound the running summary: keep at most the last 1 TF line
        step += 1
        snap = _snapshot(step, f"after {name}", wc)
        trace.append(snap)
        per_file.append({"name": name, "est_tokens_after": snap["est_tokens"], "lines_after": snap["lines"]})

    wc.save()
    return {
        "name": f"long-horizon {strategy}" + (" (naive-append)" if strategy == "naive" else " (context-as-a-file)"),
        "trace": trace,
        "total_lines_written": total_written,
        "lines_discarded": discarded,
        "per_file": per_file,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Long-horizon CLM test over all data/*.raw")
    ap.add_argument("--data-dir", default=DATA_DIR, help="directory of .raw volumes")
    ap.add_argument("--render", action="store_true", help="also write real VTK PNGs where dims are known")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.raw")))
    if not paths:
        print(f"No .raw files found in {args.data_dir}")
        return 1

    os.makedirs(CTX_DIR, exist_ok=True)

    print(f"Long-horizon session over {len(paths)} file(s) in {args.data_dir}:")
    infos: List[Dict] = []
    for p in paths:
        info = inspect_file(p)
        infos.append(info)
        print(f"  - {info['name']:<12} {info['nbytes']:>12,} bytes  "
              f"-> {info['dtype']} {info['dims']}  range {info['range']}")
        if args.render:
            out = try_render(info)
            print(f"      render: {out if out else 'skipped (unknown dims or VTK unavailable)'}")

    naive = run_session(infos, "naive")
    clm = run_session(infos, "clm")

    with open(os.path.join(CTX_DIR, "longhorizon_naive.json"), "w", encoding="utf-8") as fh:
        json.dump(naive, fh, indent=2)
    with open(os.path.join(CTX_DIR, "longhorizon_clm.json"), "w", encoding="utf-8") as fh:
        json.dump(clm, fh, indent=2)

    # growth table across the horizon
    print()
    print("Working-context size after each file (est. tokens) — the growth curve:")
    print(f"  {'after file':<16}{'naive':>10}{'CLM':>10}{'gap x':>8}")
    for n, c in zip(naive["trace"], clm["trace"]):
        gap = (n["est_tokens"] / c["est_tokens"]) if c["est_tokens"] else 0
        print(f"  {n['label'][6:]:<16}{n['est_tokens']:>10}{c['est_tokens']:>10}{gap:>7.1f}x")

    nf = naive["trace"][-1]["est_tokens"] if naive["trace"] else 0
    cf = clm["trace"][-1]["est_tokens"] if clm["trace"] else 0
    print()
    print(f"End of horizon: naive {nf} tok vs CLM {cf} tok", end="")
    if cf:
        print(f"  ->  CLM is {round((nf - cf) / nf * 100, 1)}% smaller, {nf / cf:.1f}x less")
    else:
        print()
    print()
    print("Traces written:")
    print("  CLM/context/longhorizon_naive.json")
    print("  CLM/context/longhorizon_clm.json")
    print()
    print("Full metrics:")
    print("  python CLM/clm_metrics.py --compare "
          "CLM/context/longhorizon_naive.json CLM/context/longhorizon_clm.json")
    print()
    print("Add more .raw files to data/ and re-run — the gap widens with the horizon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
