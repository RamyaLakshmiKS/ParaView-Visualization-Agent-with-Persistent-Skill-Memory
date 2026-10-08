#!/usr/bin/env python3
"""
mri_woman_off.py — COLD (memory OFF) volume render of data/mri_woman.raw.

No prior lessons available (fresh agent). Taking the common first-attempt
approach for a headerless raw MRI volume:
  - Assume 8-bit (uint8) data, which is the usual default for .raw volume demos.
  - The file is 14,286,848 bytes; as uint8 that factors as 256 x 256 x 218.
  - Build a straightforward opacity ramp over the 0..255 range and a standard
    grey/warm colormap. (No background-cutoff or faint-shell technique known.)
"""
import sys
import numpy as np
import vtk
from vtk.util import numpy_support

RAW = "data/mri_woman.raw"
OUT = "results/mri_woman_off.png"
DIMS = (256, 256, 218)   # 14,286,848 bytes / (256*256) = 218 slices, assuming uint8


def main():
    arr = np.fromfile(RAW, dtype=np.uint8)
    print(f"[off] assuming uint8 dims={DIMS}, size={arr.size}")

    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="F"), deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    image = vtk.vtkImageData()
    image.SetDimensions(*DIMS)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(vtk_arr)

    # Naive opacity ramp over the full 0..255 range, starting low at 0 (no hard
    # background cutoff — the haze lesson is not known here).
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0,   0.00)
    opacity.AddPoint(20,  0.05)   # low values still get some opacity -> haze
    opacity.AddPoint(80,  0.20)
    opacity.AddPoint(150, 0.45)
    opacity.AddPoint(255, 0.85)

    # Standard warm colormap over 0..255 (no faint-shell / opaque-core idea).
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0,   0.10, 0.10, 0.10)
    color.AddRGBPoint(80,  0.60, 0.45, 0.30)
    color.AddRGBPoint(150, 0.85, 0.65, 0.40)
    color.AddRGBPoint(255, 0.95, 0.90, 0.80)

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
    print(f"[off] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
