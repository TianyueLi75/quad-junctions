"""Render an SCTL WriteVTK .vtu (surface + feature-edge wireframe): pvbatch render_vtu.py in.vtu out.png [az el]"""
import sys
from paraview.simple import *
inf, outf = sys.argv[1], sys.argv[2]
view = CreateView('RenderView'); view.ViewSize=[1400,1000]
view.UseColorPaletteForBackground=0; view.Background=[1,1,1]; view.OrientationAxesVisibility=0
r = XMLUnstructuredGridReader(FileName=[inf]); UpdatePipeline()
d = Show(r, view); d.SetRepresentationType('Surface')
try: ColorBy(d, None)
except Exception: pass
d.DiffuseColor=[0.35,0.6,0.85]; d.Ambient=0.25; d.Specular=0.3
e = ExtractEdges(Input=r); UpdatePipeline()
de = Show(e, view); de.SetRepresentationType('Wireframe')
de.AmbientColor=[0.08,0.08,0.12]; de.DiffuseColor=[0.08,0.08,0.12]; de.LineWidth=0.6; de.Opacity=0.35
view.ResetCamera(); cam=GetActiveCamera(); cam.SetFocalPoint(0,-0.15,0)
cam.SetPosition(2.2,-2.4,2.6); cam.SetViewUp(0,0,1); view.ResetCamera(); cam.Dolly(1.25)
Render(); SaveScreenshot(outf, view, ImageResolution=[1400,1000]); print("wrote",outf)
