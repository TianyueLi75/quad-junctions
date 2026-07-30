"""Render a surface VTU colored by dl_err (log scale) + colorbar: pvbatch render_error.py in.vtu out.png"""
import sys
from paraview.simple import *
inf, outf = sys.argv[1], sys.argv[2]
view = CreateView('RenderView'); view.ViewSize=[1400,1000]
view.UseColorPaletteForBackground=0; view.Background=[1,1,1]; view.OrientationAxesVisibility=0
r = XMLUnstructuredGridReader(FileName=[inf]); UpdatePipeline()
d = Show(r, view); d.SetRepresentationType('Surface')
ColorBy(d, ('POINTS','dl_err'))
ctf = GetColorTransferFunction('dl_err'); ctf.ApplyPreset('Inferno (matplotlib)', True)
ctf.MapControlPointsToLogSpace(); ctf.UseLogScale = 1
ctf.RescaleTransferFunction(1e-5, 2e-1)
bar = GetScalarBar(ctf, view); bar.Title='DL error'; bar.ComponentTitle=''
bar.TitleColor=[0,0,0]; bar.LabelColor=[0,0,0]; d.SetScalarBarVisibility(view, True)
view.ResetCamera(); cam=GetActiveCamera(); cam.SetFocalPoint(0,-0.15,0)
cam.SetPosition(2.2,-2.4,2.6); cam.SetViewUp(0,0,1); view.ResetCamera(); cam.Dolly(1.25)
Render(); SaveScreenshot(outf, view, ImageResolution=[1400,1000]); print("wrote",outf)
