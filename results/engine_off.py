#!/usr/bin/env python3
"""
engine_off.py — COLD render of data/engine.raw (memory OFF control).

No prior lessons. This is the naive first attempt a fresh agent typically makes:
assume the standard 0..255 range for a uint8 volume and build a simple LINEAR
opacity ramp straight from the bottom of the range. Render once.

(This is intentionally the un-informed baseline. It is expected to show the
low-value "haze" that the memory-ON run avoided via a background cutoff.)
"""
import sys
import numpy as np
import vtk
from vtk.util import numpy_support

DIMS = (256, 256, 128)
RAW = "data/engine.raw"
OUT = "results/engine.png"


def main():
    arr = np.fromfile(RAW, dtype=np.uint8)
    image = vtk.vtkImageData()
    image.SetDimensions(*DIMS)
    image.SetSpacing(1.0, 1.0, 1.0)
    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="F"), deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(vtk_arr)

    # Naive opacity: simple linear ramp over the assumed 0..255 range.
    # Low values get nonzero opacity -> this is what produces the haze.
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0.0, 0.0)
    opacity.AddPoint(255.0, 0.6)

    # Generic rainbow-ish color map over the assumed range.
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0.0, 0.0, 0.0, 1.0)
    color.AddRGBPoint(128.0, 0.0, 1.0, 0.0)
    color.AddRGBPoint(255.0, 1.0, 0.0, 0.0)

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOff()
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
