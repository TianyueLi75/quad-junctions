"""Parse SCTL WriteVTK appended-raw .vtu piece(s) directly (bypassing ParaView) and scatter-plot the
surface nodes colored by the 'value' field (Green's rel error, log scale).
  python3 plot_greens_vtu.py out.png vmin vmax piece1.vtu [piece2.vtu ...]
"""
import sys, struct, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_vtu(fn):
    raw = open(fn, 'rb').read()
    m = re.search(rb'<AppendedData\s+encoding="raw">\s*_', raw)
    base = m.end()                          # first byte after the '_' marker
    hdr = raw[:m.start()].decode('latin1')
    def offset(name):
        mm = re.search(r'Name="%s"[^>]*offset="(\d+)"' % name, hdr)
        return int(mm.group(1))
    def arr(off, dt):
        n = struct.unpack_from('<I', raw, base + off)[0]         # uint32 byte-count header
        return np.frombuffer(raw[base+off+4 : base+off+4+n], dtype=dt)
    pos = arr(offset('Position'), '<f4').reshape(-1, 3)
    val = arr(offset('value'),    '<f4')
    return pos, val

outf, vmin, vmax = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
P = []; V = []
for f in sys.argv[4:]:
    p, v = read_vtu(f); P.append(p); V.append(v)
    print("  %s: %d nodes, err range [%.2e, %.2e]" % (f.split('/')[-1], len(v), v.min(), v.max()))
P = np.vstack(P); V = np.concatenate(V)
print("total nodes=%d  max Green's rel err=%.3e" % (len(V), V.max()))

c = np.log10(np.clip(V, vmin, vmax))
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection='3d')
s = ax.scatter(P[:,0], P[:,1], P[:,2], c=c, cmap='inferno',
               vmin=np.log10(vmin), vmax=np.log10(vmax), s=4, linewidths=0, depthshade=False)
# equal aspect
mid = P.mean(0); r = (P.max(0)-P.min(0)).max()/2
for setlim, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
    setlim(m-r, m+r)
ax.view_init(elev=22, azim=-60); ax.set_box_aspect((1,1,1))
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title("Stokes Green's-identity rel. error over the surface\n(order 8, nref 1, CSBQ slender arms, tol=1e-7)")
cb = fig.colorbar(s, ax=ax, shrink=0.6, pad=0.02)
cb.set_label(r"$\log_{10}$ Green's rel. error")
fig.savefig(outf, dpi=130, bbox_inches='tight')
print("wrote", outf)
