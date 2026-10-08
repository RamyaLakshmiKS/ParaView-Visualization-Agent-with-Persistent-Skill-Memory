#!/usr/bin/env python3
"""
mri_woman.py — volume render of data/mri_woman.raw (MRI scalar volume).

Goal: reveal internal anatomy — denser tissue stands out, softer tissue faint,
with readable colors.

Dtype/shape RESOLVED BY INSPECTION (headerless .raw, 14,286,848 bytes):
  * uint16 little-endian, 256 x 256 x 109, measured range [2, 4095] (12-bit MRI).
  * Confirmed against the uint8/256x256x218 alternative by adjacent-slice
    correlation: uint16-LE = 0.88 (coherent anatomy) vs uint8 = 0.46 (garbled).

Lessons applied (recalled from memory, ON run):
  * sem-range-relative-tf : measured the real scalar range and placed every
    opacity/color control point as a FRACTION of it (MRI dtype/range differs
    from bonsai/engine, so fixed values would have failed).
  * sem-background-cutoff / cor-haze : ~75% of voxels sit below frac 0.05
    (air/background). Opacity held at 0 up to frac 0.05 so no speckle haze.
  * sem-faint-shell-opaque-core : soft tissue (low-mid band) ghosted at very low
    opacity; denser tissue (high band) ramped opaque so it stands out.
  * cor-density-color-assumption : inspected the value distribution first
    (p50 frac 0.014, tissue spans ~0.05-0.5) and placed bands from that, not
    from an assumed ordering.
  * sem-verify-by-eye : render, look at the PNG, fix, finish.
"""
import sys
import numpy as np
import vtk
from vtk.util import numpy_support

RAW = "data/mri_woman.raw"
OUT = "results/mri_woman.png"
DIMS = (256, 256, 109)          # resolved by inspection (uint16-LE)


def main():
    # INSPECT: load as uint16-LE and measure the real range (do not assume).
    arr = np.fromfile(RAW, dtype="<u2")
    assert arr.size == DIMS[0] * DIMS[1] * DIMS[2], f"unexpected size {arr.size}"
    vmin, vmax = float(arr.min()), float(arr.max())
    span = vmax - vmin
    print(f"[inspect] mri_woman dims={DIMS} dtype=uint16-LE "
          f"scalar range=[{vmin},{vmax}]")

    def f(frac):   # fraction of measured range -> absolute scalar
        return vmin + frac * span

    # VTK wants the scalars as a contiguous array; keep uint16.
    vtk_arr = numpy_support.numpy_to_vtk(
        arr.astype(np.uint16).ravel(order="F"), deep=True,
        array_type=vtk.VTK_UNSIGNED_SHORT)
    image = vtk.vtkImageData()
    image.SetDimensions(*DIMS)
    image.SetSpacing(1.0, 1.0, 1.3)   # MRI slice spacing a touch thicker than in-plane
    image.GetPointData().SetScalars(vtk_arr)

    # OPACITY: hard background cutoff (kills haze), soft tissue FAINT, dense tissue
    # opaque. All points are fractions of the MEASURED range.
    # Fix after verify-by-eye (iter 1 made the skin shell too opaque, hiding the
    # interior): pull the broad soft/mid tissue band DOWN to a ghost so denser
    # interior structure reads through it; reserve real opacity for the high band.
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(f(0.00), 0.00)
    opacity.AddPoint(f(0.05), 0.00)   # cutoff: ~75% of voxels (air/noise) below here
    opacity.AddPoint(f(0.08), 0.010)  # soft tissue (skin): barely-there ghost
    opacity.AddPoint(f(0.22), 0.025)  # soft tissue stays very faint
    opacity.AddPoint(f(0.34), 0.08)   # mid tissue: still see-through
    opacity.AddPoint(f(0.46), 0.35)   # denser tissue starts to stand out
    opacity.AddPoint(f(0.60), 0.75)   # dense structure clearly opaque
    opacity.AddPoint(f(1.00), 0.95)   # densest cores fully opaque

    # COLOR: cool/dim for the faint soft tissue -> warm/bright reserved for the
    # dense high band, so the interior anatomy pops against a ghosted exterior.
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(f(0.06), 0.16, 0.20, 0.32)   # soft tissue: dim cool blue-grey
    color.AddRGBPoint(f(0.30), 0.40, 0.48, 0.60)   # soft->mid: pale cool grey
    color.AddRGBPoint(f(0.44), 0.70, 0.72, 0.70)   # transition: neutral
    color.AddRGBPoint(f(0.52), 0.92, 0.76, 0.40)   # dense: warm gold
    color.AddRGBPoint(f(0.66), 0.97, 0.52, 0.20)   # denser: orange
    color.AddRGBPoint(f(0.80), 0.95, 0.28, 0.16)   # red
    color.AddRGBPoint(f(1.00), 1.00, 0.96, 0.90)   # densest cores: bright near-white

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    prop.SetAmbient(0.30)
    prop.SetDiffuse(0.70)
    prop.SetSpecular(0.30)
    prop.SetInterpolationTypeToLinear()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddVolume(volume)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(-15)
    renderer.ResetCameraClippingRange()

    render_window.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(OUT)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"[render] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
