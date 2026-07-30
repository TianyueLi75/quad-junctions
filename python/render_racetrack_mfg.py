"""Render the manufactured-solution z=0 slice from the racetrack (lens) mfg test.

Reads the CSV written by `ybifurc-channel-bie mfg` (columns x,y,umag,errmag over a regular Nx x Ny grid,
tube-interior points zeroed by the exterior mask) and draws two panels: represented speed |u| and the
pointwise error |u_bie - u_exact| (log scale) across the exterior fluid, incl. the lens "eye".

   Usage: python render_racetrack_mfg.py vis/ybifurc-channel-mfg-ordN-nrefM-slice.csv [out.png]
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

csv = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else csv.replace(".csv", ".png")

d = np.genfromtxt(csv, delimiter=",", names=True)
x, y, umag, err = d["x"], d["y"], d["umag"], d["errmag"]

# Regular grid: recover Nx, Ny from the unique coordinates (CSV order is ix outer, iy inner).
xs, ys = np.unique(x), np.unique(y)
Nx, Ny = len(xs), len(ys)
U = umag.reshape(Nx, Ny).T          # -> (Ny, Nx) for imshow/pcolormesh with x across, y up
E = err.reshape(Nx, Ny).T
extent = [xs.min(), xs.max(), ys.min(), ys.max()]

# Mask the zeroed tube interior (masked points show as a neutral outline of the racetrack).
Um = np.ma.masked_where(U == 0, U)
Em = np.ma.masked_where(E == 0, E)
emin = max(Em.min() if Em.count() else 1e-16, 1e-16)
emax = Em.max() if Em.count() else 1e-12

fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
im0 = ax[0].imshow(Um, origin="lower", extent=extent, aspect="equal", cmap="viridis")
ax[0].set_title("represented speed |u|  (exterior Stokes, single Stokeslet inside the tube)")
fig.colorbar(im0, ax=ax[0], shrink=0.85)

im1 = ax[1].imshow(Em, origin="lower", extent=extent, aspect="equal", cmap="magma",
                   norm=LogNorm(vmin=emin, vmax=emax))
ax[1].set_title("pointwise error |u_bie - u_exact|  (max %.2e)" % emax)
fig.colorbar(im1, ax=ax[1], shrink=0.85)
for a in ax:
    a.set_xlabel("x"); a.set_ylabel("y")
    a.set_facecolor("#dddddd")   # masked tube interior

fig.savefig(out, dpi=130)
print("wrote", out, " grid %dx%d" % (Nx, Ny), " max|err| %.3e" % emax)
