"""Generic mesh render: pvbatch render_mesh.py surf.vtu wire.vtp out.png [view]

view: 'iso' (default), 'top', or 'front'.
Renders the surface (solid, light) plus a faint wireframe overlay.
"""
import sys
from paraview.simple import *

surf_f = sys.argv[1]
wire_f = sys.argv[2]
out_f  = sys.argv[3]
view_kind = sys.argv[4] if len(sys.argv) > 4 else 'iso'

view = CreateView('RenderView')
view.ViewSize = [1280, 960]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

surf = XMLUnstructuredGridReader(FileName=[surf_f])
UpdatePipeline()
sd = Show(surf, view)
sd.SetRepresentationType('Surface')
ColorBy(sd, None)                      # solid shading, not by field
sd.DiffuseColor = [0.30, 0.55, 0.85]
sd.Ambient = 0.25
sd.Specular = 0.3
sd.SpecularPower = 30

wire = XMLPolyDataReader(FileName=[wire_f])
UpdatePipeline()
wd = Show(wire, view)
wd.SetRepresentationType('Wireframe')
wd.AmbientColor = [0.1, 0.1, 0.15]
wd.DiffuseColor = [0.1, 0.1, 0.15]
wd.LineWidth = 0.75
wd.Opacity = 0.35

view.ResetCamera()
cam = GetActiveCamera()
cam.SetFocalPoint(0.0, -0.15, 0.0)
if view_kind == 'top':
    cam.SetPosition(0.0, -0.15, 4.0); cam.SetViewUp(0.0, 1.0, 0.0)
elif view_kind == 'front':
    cam.SetPosition(0.0, -0.15, 4.0); cam.SetViewUp(0.0, 1.0, 0.0)
else:  # iso
    cam.SetPosition(2.2, -2.4, 2.6); cam.SetViewUp(0.0, 0.0, 1.0)
view.ResetCamera()
cam.Dolly(1.25)
Render()
SaveScreenshot(out_f, view, ImageResolution=[1280, 960])
print("wrote", out_f)
