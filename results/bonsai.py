#!/usr/bin/env python3
"""
bonsai.py — volume render of data/bonsai.raw (256^3 uint8).

Goal: a potted tree — brown pot, silver branch, golden leaves.

Approach (transferable procedure):
  1. INSPECT the data at runtime: measure actual scalar min/max; do not assume.
  2. Build opacity/color control points RELATIVE to the measured range
     (fractions of min..max), not hard-coded scalar values.
  3. Keep background (lowest values) transparent so the shell/air drops out.
"""
import sys
import numpy as np
import vtk
from vtk.util import numpy_support

DIMS = (256, 256, 256)      # bonsai is a 256^3 volume (16,777,216 bytes)
RAW = "data/bonsai.raw"
OUT = "results/bonsai.png"


def main():
    # 1. Load + INSPECT (measure real range, never assume).
    arr = np.fromfile(RAW, dtype=np.uint8)
    assert arr.size == DIMS[0] * DIMS[1] * DIMS[2], f"unexpected size {arr.size}"
    vmin, vmax = float(arr.min()), float(arr.max())
    span = vmax - vmin
    print(f"[inspect] bonsai dims={DIMS} dtype=uint8 scalar range=[{vmin},{vmax}]")

    # Helper: map a fraction of the measured range to an absolute scalar value.
    def f(frac):
        return vmin + frac * span

    # 2. Wrap as vtkImageData (Fortran order flat scalars).
    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="F"), deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    image = vtk.vtkImageData()
    image.SetDimensions(*DIMS)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(vtk_arr)

    # 3. Opacity: a HARD background cutoff kills the low-value speckle "haze".
    #    Inspection showed structure is value > ~0.25 of range; everything below
    #    that is noise, so hold opacity at 0 until then. Points are fractions of
    #    the MEASURED range (transferable), not fixed scalar values.
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(f(0.00), 0.00)
    opacity.AddPoint(f(0.22), 0.00)   # cutoff: below here is background haze
    opacity.AddPoint(f(0.26), 0.06)   # thin foliage specks fade in
    opacity.AddPoint(f(0.45), 0.14)   # branch / woody
    opacity.AddPoint(f(0.70), 0.40)   # dense wood / pot wall
    opacity.AddPoint(f(1.00), 0.90)   # densest (pot)

    # 4. Color by scalar band. In bonsai the pot is the DENSEST material (high
    #    values), the woody branch is mid-high, and the thin leaves are the
    #    lower-density specks. Map accordingly to hit the goal palette.
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(f(0.26), 0.95, 0.80, 0.20)   # golden leaves (thin, low density)
    color.AddRGBPoint(f(0.45), 0.88, 0.72, 0.28)   # leaf->branch
    color.AddRGBPoint(f(0.58), 0.78, 0.78, 0.82)   # silver branch (mid density)
    color.AddRGBPoint(f(0.72), 0.60, 0.42, 0.22)   # branch->pot
    color.AddRGBPoint(f(0.88), 0.42, 0.27, 0.13)   # brown pot (densest)
    color.AddRGBPoint(f(1.00), 0.36, 0.22, 0.10)   # darkest brown

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()                       # shading gives the tree 3D form
    prop.SetAmbient(0.3)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.2)
    prop.SetInterpolationTypeToLinear()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)   # white backdrop
    renderer.AddVolume(volume)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(700, 800)
    render_window.AddRenderer(renderer)

    # Camera: look at the tree from slightly above the front.
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(-15)      # tilt down a touch to see the pot
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
