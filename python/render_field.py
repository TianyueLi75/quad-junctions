"""Volume-render the Y-bifurcation scalar field to field_volume.png (pvpython)."""
from paraview.simple import *

reader = XMLImageDataReader(FileName=['field.vti'])
reader.PointArrayStatus = ['field']
UpdatePipeline()
di = reader.GetDataInformation().GetPointDataInformation().GetArrayInformation('field')
fmin, fmax = di.GetComponentRange(0)
print("field range", fmin, fmax)

view = CreateView('RenderView')
view.ViewSize = [1280, 960]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

disp = Show(reader, view)
disp.SetRepresentationType('Volume')
ColorBy(disp, ('POINTS', 'field'))

ctf = GetColorTransferFunction('field')
otf = GetOpacityTransferFunction('field')
ctf.ApplyPreset('Viridis (matplotlib)', True)
ctf.RescaleTransferFunction(0.0, fmax)

# transparent low field (background), progressively opaque toward the tube core
otf.RescaleTransferFunction(0.0, fmax)
otf.Points = [
    0.6,  0.00, 0.5, 0.0,
    1.5,  0.03, 0.5, 0.0,
    3.5,  0.12, 0.5, 0.0,
    6.0,  0.30, 0.5, 0.0,
    fmax, 0.55, 0.5, 0.0,
]
disp.SetScalarBarVisibility(view, False)

# tilted top-down view so the Y branching is clearly legible
view.ResetCamera()
cam = GetActiveCamera()
cam.SetFocalPoint(0.0, -0.15, 0.0)
cam.SetPosition(0.9, -1.7, 3.4)
cam.SetViewUp(0.0, 1.0, 0.0)
view.ResetCamera()          # keep everything in frame, preserve direction
cam.Dolly(1.15)
view.CameraParallelProjection = 0
Render()

SaveScreenshot('field_volume.png', view, ImageResolution=[1280, 960])
print("wrote field_volume.png")
