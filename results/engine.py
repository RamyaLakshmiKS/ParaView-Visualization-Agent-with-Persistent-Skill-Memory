#!/usr/bin/env python3
"""
engine.py — volume render of data/engine.raw (256 x 256 x 128 uint8).

Goal: show internal structure — dense inner parts stand out, outer shell faint,
readable colors.

Lessons applied (recalled from the bonsai task, memory ON):
  * sem-range-relative-tf : measure the real scalar range, build the transfer
    function from FRACTIONS of it (not fixed scalar values).
  * sem-background-cutoff / cor-haze : keep opacity 0 below a cutoff fraction so
    the low-value speckle "haze" never appears.
  * cor-density-color-assumption : inspected the value distribution first — the
    dense inner metal sits in a high band (~0.5 of range), the shell is a lower,
    broad band. Map opacity/color from that, don't assume.
  * sem-verify-by-eye : render, look, fix.
"""
import sys
import numpy as np
import vtk
from vtk.util import numpy_support

DIMS = (256, 256, 128)      # engine: 8,388,608 bytes = 256*256*128 uint8
RAW = "data/engine.raw"
OUT = "results/engine.png"


def main():
    # INSPECT: measure the real range (do not assume) — recalled lesson.
    arr = np.fromfile(RAW, dtype=np.uint8)
    assert arr.size == DIMS[0] * DIMS[1] * DIMS[2], f"unexpected size {arr.size}"
    vmin, vmax = float(arr.min()), float(arr.max())
    span = vmax - vmin
    print(f"[inspect] engine dims={DIMS} dtype=uint8 scalar range=[{vmin},{vmax}]")

    def f(frac):   # fraction of measured range -> absolute scalar
        return vmin + frac * span

    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="F"), deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    image = vtk.vtkImageData()
    image.SetDimensions(*DIMS)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(vtk_arr)

    # OPACITY: hard background cutoff (kills haze), outer shell FAINT, dense inner
    # parts opaque. All points are fractions of the measured range.
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(f(0.00), 0.00)
    opacity.AddPoint(f(0.16), 0.00)   # cutoff: below here = background haze
    opacity.AddPoint(f(0.22), 0.015)  # outer shell: ghosted, very faint
    opacity.AddPoint(f(0.48), 0.04)   # shell / transition stays faint
    opacity.AddPoint(f(0.52), 0.45)   # dense inner metal starts to stand out
    opacity.AddPoint(f(0.65), 0.80)   # dense inner: clearly opaque
    opacity.AddPoint(f(1.00), 0.98)   # densest cores fully opaque

    # COLOR: cool/dim for faint shell, warm/bright for dense inner structure
    # so the interior reads clearly against the ghosted shell.
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(f(0.20), 0.55, 0.62, 0.72)   # shell: pale cool grey (faint)
    color.AddRGBPoint(f(0.48), 0.60, 0.66, 0.74)   # shell->inner
    color.AddRGBPoint(f(0.54), 0.98, 0.80, 0.30)   # dense inner: warm gold
    color.AddRGBPoint(f(0.72), 0.95, 0.45, 0.12)   # hotter orange
    color.AddRGBPoint(f(1.00), 0.88, 0.12, 0.08)   # densest: red

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    prop.SetAmbient(0.3)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.3)
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
    render_window.SetSize(800, 700)
    render_window.AddRenderer(renderer)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(35)
    cam.Elevation(20)
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
