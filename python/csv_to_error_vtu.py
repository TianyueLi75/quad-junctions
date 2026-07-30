"""Convert the C++ DL-error CSV (x,y,z,r_xy,z_abs,dl_err at each GL node) into a clean
ASCII .vtu surface colored by dl_err, readable in any ParaView.

  run_py.sh csv_to_error_vtu.py <dlerr.csv> <order> [out_prefix]

Uses the CSV's own node coordinates (the projected iso-surface nodes) reshaped to
[panels, q, q] and Lagrange-interpolated (through the GL nodes) to a gap-free
tessellation, so it matches the C++ surface exactly.
"""
import sys
import numpy as np
from bifurcation import PanelSet, panels_to_tessellation, write_vtu_quadgrid, lagrange_matrix, gl_nodes

csv = sys.argv[1]
q   = int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else csv.rsplit('.', 1)[0]

data = np.loadtxt(csv, delimiter=',', skiprows=1)
XYZ = data[:, 0:3]
err = data[:, 5]
N = len(XYZ)
P = N // (q*q)
assert P*q*q == N, f"row count {N} not divisible by q*q={q*q}"
nodes = XYZ.reshape(P, q, q, 3)
enode = err.reshape(P, q, q)

panels = PanelSet(nodes, np.zeros((P, 4, 3)), q)   # corners unused by tessellate()

# geometry tessellation (gap-free), plus error interpolated the same way
m = q + 1
pts, _ = panels.tessellate(m)                      # (P,m,m,3)
x, _ = gl_nodes(q); u = np.linspace(-1, 1, m); L = lagrange_matrix(x, u)   # (m,q)
e1 = np.einsum("mi,pij->pmj", L, enode)
etess = np.einsum("nj,pmj->pmn", L, e1)            # (P,m,m)

points = pts.reshape(-1, 3)
scal = etess.reshape(-1)
quads = []
for pi in range(P):
    base = pi*m*m
    for i in range(m-1):
        for j in range(m-1):
            a = base + i*m + j
            quads.append([a, a+1, a+m+1, a+m])
quads = np.array(quads, np.int64)
write_vtu_quadgrid(out + "_surface.vtu", points, quads, point_scalar=scal, name="dl_err")
print(f"panels={P} q={q}  dl_err range [{err.min():.2e}, {err.max():.2e}]  median={np.median(err):.2e}")
print(f"wrote {out}_surface.vtu (colored by dl_err)")
