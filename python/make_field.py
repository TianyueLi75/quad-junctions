"""Sample the Y-bifurcation field on a regular volume grid and write .vti."""
import numpy as np
from bifurcation import Field, CFG, write_vti

fld = Field()

pad = CFG.grid_pad
# domain generously covers the arms (length ~gauss_len) and the ~0.3 tube half-thickness
zhalf = 0.4
xlo, xhi = -1.30 - pad, 1.30 + pad
ylo, yhi = -1.30 - pad, 1.00 + pad
zlo, zhi = -(zhalf + pad), (zhalf + pad)
nx, ny, nz = CFG.grid_n

xs = np.linspace(xlo, xhi, nx)
ys = np.linspace(ylo, yhi, ny)
zs = np.linspace(zlo, zhi, nz)
origin  = (xs[0], ys[0], zs[0])
spacing = (xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])

# x fastest, then y, then z
Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")   # shape (nz,ny,nx)
pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
vals = fld.f(pts)                                   # already flat, x-fastest

write_vti("field.vti", origin, spacing, (nx, ny, nz), vals, name="field")
print(f"wrote field.vti  dims={nx}x{ny}x{nz}  "
      f"range=[{vals.min():.4f},{vals.max():.4f}]")
print(f"domain x[{xlo:.2f},{xhi:.2f}] y[{ylo:.2f},{yhi:.2f}] z[{zlo:.2f},{zhi:.2f}]")
