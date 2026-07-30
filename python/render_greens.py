"""Render the Y-bifurcation surface (junction + arms pieces) colored by the SCTL 'value' point
field (log scale) -- used for the Green's-identity error dump.
  pvbatch render_greens.py out.png vmin vmax in1.vtu[.pvtu] [in2 ...]
"""
import sys
from paraview.simple import *

outf = sys.argv[1]
vmin = float(sys.argv[2]); vmax = float(sys.argv[3])
ins = sys.argv[4:]

view = CreateView('RenderView'); view.ViewSize = [1500, 1050]
view.UseColorPaletteForBackground = 0; view.Background = [1, 1, 1]; view.OrientationAxesVisibility = 0

disps = []
B = [1e30, -1e30, 1e30, -1e30, 1e30, -1e30]   # combined bounds xmin,xmax,ymin,ymax,zmin,zmax
for f in ins:
    r = XMLUnstructuredGridReader(FileName=[f]); UpdatePipeline()
    d = Show(r, view); d.SetRepresentationType('Surface')
    ColorBy(d, ('POINTS', 'value'))
    disps.append(d)
    b = r.GetDataInformation().GetBounds()
    for k in (0, 2, 4): B[k] = min(B[k], b[k])
    for k in (1, 3, 5): B[k] = max(B[k], b[k])

ctf = GetColorTransferFunction('value'); ctf.ApplyPreset('Inferno (matplotlib)', True)
ctf.MapControlPointsToLogSpace(); ctf.UseLogScale = 1
ctf.RescaleTransferFunction(vmin, vmax)

bar = GetScalarBar(ctf, view); bar.Title = "Green's rel err"; bar.ComponentTitle = ''
bar.TitleColor = [0, 0, 0]; bar.LabelColor = [0, 0, 0]
bar.ScalarBarLength = 0.5
disps[0].SetScalarBarVisibility(view, True)

# Oblique camera from the (+x,-y,+z) octant so view-up=z is never parallel to the view direction.
cx, cy, cz = (B[0]+B[1])/2, (B[2]+B[3])/2, (B[4]+B[5])/2
L = max(B[1]-B[0], B[3]-B[2], B[5]-B[4], 1e-6)
cam = GetActiveCamera()
cam.SetFocalPoint(cx, cy, cz)
cam.SetPosition(cx + 1.8*L, cy - 2.0*L, cz + 1.4*L)
cam.SetViewUp(0, 0, 1)
view.ResetCamera()   # keep direction, zoom to fit
Render(); SaveScreenshot(outf, view, ImageResolution=[1500, 1050]); print("wrote", outf, "bounds", B)
