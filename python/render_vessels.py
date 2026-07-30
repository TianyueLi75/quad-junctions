"""Top-down scatter of the vessels geometry surface nodes (planar z=0), parsed straight from SCTL's
appended-raw .vtu (ParaView's XML reader can't parse SCTL's raw blob). Directly comparable to
arterial_venous_smoothed_nolabels.svg.
   Usage: python render_vessels.py junc.vtu arms.vtu out.png
"""
import sys, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def positions(fname):
    raw = open(fname, "rb").read()
    # header (before the raw blob) gives NumberOfPoints and the Position offset
    hdr = raw[:raw.find(b"<AppendedData")].decode("latin1")
    npts = int(re.search(r'NumberOfPoints="(\d+)"', hdr).group(1))
    off = int(re.search(r'Name="Position"[^>]*offset="(\d+)"', hdr).group(1))
    m = re.search(rb'<AppendedData encoding="raw">\s*_', raw)
    base = m.end()                                   # first byte after the underscore
    nbytes = np.frombuffer(raw, np.uint32, 1, base + off)[0]
    data = np.frombuffer(raw, np.float32, nbytes // 4, base + off + 4)
    return data.reshape(-1, 3)[:npts]

junc, arms, out = sys.argv[1], sys.argv[2], sys.argv[3]
Pj, Pa = positions(junc), positions(arms)
fig, ax = plt.subplots(figsize=(13, 10))
ax.scatter(Pa[:, 0], Pa[:, 1], s=0.6, c="#4E86B5", lw=0, label="arms")
ax.scatter(Pj[:, 0], Pj[:, 1], s=0.6, c="#B03A2E", lw=0, label="junctions")
ax.set_aspect("equal")
ax.set_title("vessels geometry (top-down, z=0) -- compare to arterial_venous_smoothed_nolabels.svg\n"
             "arterial (red tree) left->right, venous right->left; %d junc + %d arm nodes"
             % (len(Pj), len(Pa)))
ax.legend(markerscale=12, loc="upper right")
fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out, "junc z-range",
      float(Pj[:, 2].min()), float(Pj[:, 2].max()))
