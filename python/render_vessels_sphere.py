"""Three views (top-down x-y, side x-z, front y-z) of the vessels geometry surface nodes, parsed straight
from SCTL's appended-raw .vtu. For the sphere-draped network (ybifurc-vessels-bie ... sphere_deg>0) the
side/front panels reveal the spherical-cap curvature; for the planar network z=0 they are flat lines.
   Usage: python render_vessels_sphere.py junc.vtu arms.vtu out.png
"""
import sys, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def positions(fname):
    raw = open(fname, "rb").read()
    hdr = raw[:raw.find(b"<AppendedData")].decode("latin1")
    npts = int(re.search(r'NumberOfPoints="(\d+)"', hdr).group(1))
    off = int(re.search(r'Name="Position"[^>]*offset="(\d+)"', hdr).group(1))
    m = re.search(rb'<AppendedData encoding="raw">\s*_', raw)
    base = m.end()
    nbytes = np.frombuffer(raw, np.uint32, 1, base + off)[0]
    data = np.frombuffer(raw, np.float32, nbytes // 4, base + off + 4)
    return data.reshape(-1, 3)[:npts]

junc, arms, out = sys.argv[1], sys.argv[2], sys.argv[3]
Pj, Pa = positions(junc), positions(arms)
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
views = [(0, 1, "top-down (x-y)"), (0, 2, "side (x-z)"), (1, 2, "front (y-z)")]
for ax, (i, j, title) in zip(axes, views):
    ax.scatter(Pa[:, i], Pa[:, j], s=0.5, c="#4E86B5", lw=0)
    ax.scatter(Pj[:, i], Pj[:, j], s=0.5, c="#B03A2E", lw=0)
    ax.set_aspect("equal"); ax.set_title(title)
fig.suptitle("vessels geometry: arterial (red) + venous, arms (blue).  z-range [%.3f, %.3f]"
             % (float(min(Pj[:, 2].min(), Pa[:, 2].min())), float(max(Pj[:, 2].max(), Pa[:, 2].max()))))
fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)
