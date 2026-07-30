#!/usr/bin/env python3
"""Render the C++ twirling-cilia (flagella) centerlines to a PNG for a quick shape check against
python/flagellum.png. Reads flagella_centerline.csv (finger,x,y,z,r) written by
`STUD_DUMP_CENTERLINE=1 ./bin/stud_sphere-hybrid-bie flagella`. Draws each finger's centerline as a
colored 3D curve inside a faint unit sphere. This checks the CENTERLINES only (the base foot + tip cap
adhere to the good mesh separately)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = "/mnt/home/tli10/quad-junctions"
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{WORK}/flagella_centerline.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/python/flagella_centerlines.png"

a = np.genfromtxt(CSV, delimiter=",", names=True)
fid = a["finger"].astype(int)
fingers = sorted(set(fid.tolist()))
cmap = plt.get_cmap("tab10")

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111, projection="3d")

# faint unit sphere for reference
us, vs = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
ax.plot_surface(np.cos(us)*np.sin(vs), np.sin(us)*np.sin(vs), np.cos(vs),
                color="0.5", alpha=0.06, linewidth=0, shade=False)

for i, f in enumerate(fingers):
    m = fid == f
    x, y, z = a["x"][m], a["y"][m], a["z"][m]
    c = cmap(i % 10)
    ax.plot(x, y, z, color=c, lw=3, label=f"finger {f}")
    ax.scatter(x[0], y[0], z[0], color=c, s=30, marker="o")   # base (on surface)
    ax.scatter(x[-1], y[-1], z[-1], color=c, s=30, marker="^")  # tip (inside)

ax.set_title("flagella centerlines (o = base on surface, ^ = tip inside)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_box_aspect((1, 1, 1))
for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
    lim(-1.05, 1.05)
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}  ({len(fingers)} fingers, {len(fid)} points)")

# quick numeric sanity, echoing flagella.py's checks
pts = {f: np.c_[a["x"][fid == f], a["y"][fid == f], a["z"][fid == f]] for f in fingers}
best = np.inf
for i, fi in enumerate(fingers):
    for fj in fingers[i+1:]:
        d = np.linalg.norm(pts[fi][:, None, :] - pts[fj][None, :, :], axis=-1).min()
        best = min(best, d)
tip_r = {f: np.linalg.norm(pts[f][-1]) for f in fingers}
print(f"min centerline separation: {best:.3f}")
print(f"tip radii (|tip|, should be <1, inside): {[f'{tip_r[f]:.3f}' for f in fingers]}")
