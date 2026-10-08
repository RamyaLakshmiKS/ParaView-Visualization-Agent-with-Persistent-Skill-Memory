#!/usr/bin/env python3
"""
smoke_test.py — Step 0 environment de-risk.

Generates a synthetic 64^3 scalar volume (a fuzzy sphere) and renders it to a PNG
using VTK's offscreen volume renderer. No dataset download required.

Run:
    python smoke_test.py

Success = it prints "RENDER OK" and writes smoke_test.png.
If this works, the whole project is unblocked: the agent can render volumes headlessly.
If VTK is missing:  pip install vtk numpy
"""

import sys

def main():
    try:
        import numpy as np
        import vtk
        from vtk.util import numpy_support
    except Exception as e:
        print(f"IMPORT FAILED: {e}")
        print("Fix: pip install vtk numpy   (in a venv)")
        return 1

    # 1. Synthetic volume: a smooth sphere scalar field in a 64^3 grid.
    n = 64
    lin = np.linspace(-1.0, 1.0, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    field = np.clip(1.0 - r, 0.0, 1.0).astype(np.float32)  # 1 at center -> 0 at edge

    # 2. Wrap as a vtkImageData.
    flat = field.ravel(order="F")  # VTK expects Fortran order for flat scalars
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    image = vtk.vtkImageData()
    image.SetDimensions(n, n, n)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(vtk_arr)

    # 3. Transfer functions: opacity ramp 0->1, cool->warm color map.
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0.0, 0.0)
    opacity.AddPoint(1.0, 0.9)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0.0, 0.23, 0.29, 0.75)   # low = cool blue
    color.AddRGBPoint(0.5, 0.87, 0.87, 0.87)   # mid = grey
    color.AddRGBPoint(1.0, 0.71, 0.02, 0.15)   # high = warm red

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOff()
    volume_property.SetInterpolationTypeToLinear()

    mapper = vtk.vtkGPUVolumeRayCastMapper()
    try:
        mapper.SetInputData(image)
    except Exception:
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(image)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    # 4. Offscreen render.
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # white
    renderer.AddVolume(volume)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)   # headless
    render_window.SetSize(512, 512)
    render_window.AddRenderer(renderer)

    renderer.ResetCamera()
    render_window.Render()

    # 5. Write PNG.
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName("smoke_test.png")
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    print(f"RENDER OK  (vtk {vtk.VTK_VERSION}) -> smoke_test.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
