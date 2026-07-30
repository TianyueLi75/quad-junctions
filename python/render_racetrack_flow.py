"""Render the interior pressure-drop flow z=0 slice from the racetrack (lens) flow test.

Reads the CSV written by `ybifurc-channel-bie flow` (x,y,ux,uy,uz,umag over a regular Nx x Ny grid,
exterior points zeroed by the interior mask) and draws speed |u| with velocity streamlines/quiver over
the z=0 plane -- the flow entering the left stem, splitting around the two racetrack walls, and exiting
the right stem.

   Usage: python render_racetrack_flow.py vis/ybifurc-channel-flow-ordN-nrefM-flow-slice.csv [out.png]
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

csv = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else csv.replace(".csv", ".png")

d = np.genfromtxt(csv, delimiter=",", names=True)
x, y, ux, uy, umag = d["x"], d["y"], d["ux"], d["uy"], d["umag"]

xs, ys = np.unique(x), np.unique(y)
Nx, Ny = len(xs), len(ys)
X = x.reshape(Nx, Ny).T; Y = y.reshape(Nx, Ny).T      # (Ny, Nx)
U = ux.reshape(Nx, Ny).T; V = uy.reshape(Nx, Ny).T
S = umag.reshape(Nx, Ny).T
Sm = np.ma.masked_where(S == 0, S)                     # zeroed exterior -> masked (racetrack outline)

fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
pc = ax.pcolormesh(X, Y, Sm, cmap="viridis", shading="auto")
fig.colorbar(pc, ax=ax, shrink=0.85, label="speed |u|")

# Direction quiver at INTERIOR grid points only (speeds span stems ~88 vs walls ~40, so normalize each
# arrow to unit length -> shows flow DIRECTION cleanly; exterior/masked points carry no arrow).
step = max(1, Nx // 55)
Xs2, Ys2 = X[::step, ::step], Y[::step, ::step]
Us2, Vs2, Ss2 = U[::step, ::step], V[::step, ::step], S[::step, ::step]
mask = Ss2 > 0
mag = np.where(mask, np.hypot(Us2, Vs2), 1.0)
ax.quiver(Xs2[mask], Ys2[mask], (Us2/mag)[mask], (Vs2/mag)[mask], color="w",
          pivot="mid", scale=45, width=0.0022, headwidth=4)

ax.set_aspect("equal"); ax.set_facecolor("#dddddd")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("interior Stokes pressure-drop flow (1 inlet left, 1 outlet right)  z=0 slice\n"
             "max|u| = %.4g" % np.nanmax(S))
fig.savefig(out, dpi=130)
print("wrote", out, " grid %dx%d" % (Nx, Ny), " max|u| %.4g" % np.nanmax(S))
