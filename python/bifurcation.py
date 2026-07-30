"""
Y-bifurcation shared library: field, GL machinery, panels, VTK IO.

Contents
--------
* Field        : sum of isotropic Gaussians placed along three radial arms,
                 with an exact analytic gradient (matches C++ YField).
* GL machinery : Gauss-Legendre nodes/weights and a barycentric-Lagrange
                 interpolation operator (GL nodes -> arbitrary sample points).
* PanelSet     : a collection of q x q tensor GL panels (Nystrom convention),
                 with a gap-free tessellation for rendering.
* IO           : hand-written ASCII VTK writers (.vti ImageData, .vtu
                 UnstructuredGrid, .vtp PolyData) so no VTK python bindings
                 are required on the compute side.

The SURFACE MESH is produced by ybifurc.py (swept-O-grid; the source of truth for
src/test-ybifurc-geom.cpp). The old box-slab shrink-wrap builder and the gradient-flow
projection that used to live here were removed 2026-07-16 -- they were superseded by the
swept-O-grid (box-slab creases floored BIE at ~4e-2; gradient flow caustics at tips).

Arms meet at the origin at 120 degrees: one inlet pointing -y, two symmetric
branches up-left and up-right.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Configuration (shared by every script so the pieces stay consistent)
# ----------------------------------------------------------------------------

class Config:
    # --- field / arms (the only inputs to the iso-surface; see ybifurc.py) ---
    arm_angles_deg = (-90.0, 30.0, 150.0)   # inlet down, two branches up
    sigma          = 0.15                    # Gaussian width (tube radius scale)
    amp            = 1.0                     # Gaussian amplitude
    gauss_ds       = 0.05                    # spacing of Gaussians along an arm
    gauss_len      = 0.95                    # arm length covered by Gaussians

    # --- field sampling grid for the volume render (make_field.py) ---
    grid_pad       = 0.15
    grid_n         = (140, 140, 56)

CFG = Config()


# ----------------------------------------------------------------------------
# Field
# ----------------------------------------------------------------------------

class Field:
    """f(x) = sum_i amp * exp(-|x-c_i|^2 / (2 sigma^2)); exact gradient."""

    def __init__(self, cfg=CFG):
        self.sigma = cfg.sigma
        self.amp   = cfg.amp
        self.inv2s2 = 1.0 / (2.0 * cfg.sigma**2)
        self.invs2  = 1.0 / (cfg.sigma**2)
        self.centers = self._build_centers(cfg)

    @staticmethod
    def _build_centers(cfg):
        pts = [np.zeros(3)]                              # one Gaussian at the junction
        s = np.arange(cfg.gauss_ds, cfg.gauss_len + 1e-12, cfg.gauss_ds)
        for ang in np.deg2rad(cfg.arm_angles_deg):
            u = np.array([np.cos(ang), np.sin(ang), 0.0])
            for si in s:
                pts.append(si * u)
        return np.array(pts)

    def f(self, X, chunk=200000):
        """Field value at points X (..,3) -> (..,)."""
        X = np.asarray(X, float)
        flat = X.reshape(-1, 3)
        out = np.empty(flat.shape[0])
        C = self.centers
        for a in range(0, flat.shape[0], chunk):
            b = min(a + chunk, flat.shape[0])
            d2 = ((flat[a:b, None, :] - C[None, :, :])**2).sum(-1)
            out[a:b] = self.amp * np.exp(-d2 * self.inv2s2).sum(1)
        return out.reshape(X.shape[:-1])

    def grad(self, X, chunk=200000):
        """Gradient at points X (..,3) -> (..,3)."""
        X = np.asarray(X, float)
        flat = X.reshape(-1, 3)
        out = np.empty_like(flat)
        C = self.centers
        for a in range(0, flat.shape[0], chunk):
            b = min(a + chunk, flat.shape[0])
            diff = flat[a:b, None, :] - C[None, :, :]        # (n,Nc,3)
            d2 = (diff**2).sum(-1)                            # (n,Nc)
            E = self.amp * np.exp(-d2 * self.inv2s2)          # (n,Nc)
            out[a:b] = -(E[:, :, None] * diff).sum(1) * self.invs2
        return out.reshape(X.shape)

    def f_and_grad(self, X):
        return self.f(X), self.grad(X)


# ----------------------------------------------------------------------------
# Gauss-Legendre nodes and barycentric-Lagrange interpolation
# ----------------------------------------------------------------------------

def gl_nodes(q):
    """GL nodes and weights on [-1,1] (strictly interior)."""
    x, w = np.polynomial.legendre.leggauss(q)
    return x, w


def bary_weights(x):
    """Barycentric weights for interpolation nodes x."""
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        d = x[j] - x
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    return w


def lagrange_matrix(src, dst):
    """Row-stochastic matrix L (len(dst) x len(src)) with values(dst)=L@values(src)
    for the polynomial interpolant through nodes `src`."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = bary_weights(src)
    L = np.zeros((len(dst), len(src)))
    for i, xt in enumerate(dst):
        diff = xt - src
        exact = np.where(np.abs(diff) < 1e-14)[0]
        if exact.size:
            L[i, exact[0]] = 1.0
        else:
            num = w / diff
            L[i, :] = num / num.sum()
    return L


# ----------------------------------------------------------------------------
# Panels (tensor-product GL, Nystrom convention)
# ----------------------------------------------------------------------------

class PanelSet:
    """A collection of q x q GL panels.

    nodes : (P, q, q, 3) node coordinates.
    corners : (P, 4, 3) the quad corners the panel was built from
              (order c00, c10, c11, c01), kept for reference / wireframe.
    """

    def __init__(self, nodes, corners, q):
        self.nodes = np.asarray(nodes, float)
        self.corners = np.asarray(corners, float)
        self.q = q

    @property
    def n_panels(self):
        return self.nodes.shape[0]

    @staticmethod
    def from_quads(quad_corners, q):
        """Build panels by bilinear placement of GL nodes on each quad.

        quad_corners : (P,4,3) with corners ordered c00,c10,c11,c01.
        """
        quad_corners = np.asarray(quad_corners, float)
        x, _ = gl_nodes(q)
        a = 0.5 * (x + 1.0)                     # GL params on [0,1]
        A, B = np.meshgrid(a, a, indexing="ij") # (q,q)
        c00 = quad_corners[:, 0][:, None, None, :]
        c10 = quad_corners[:, 1][:, None, None, :]
        c11 = quad_corners[:, 2][:, None, None, :]
        c01 = quad_corners[:, 3][:, None, None, :]
        A = A[None, :, :, None]
        B = B[None, :, :, None]
        nodes = ((1-A)*(1-B)*c00 + A*(1-B)*c10 + A*B*c11 + (1-A)*B*c01)
        return PanelSet(nodes, quad_corners, q)

    # --- dense evaluation for a watertight, gap-free tessellation ---------
    def tessellate(self, m):
        """Evaluate the panel Lagrange interpolant on an m x m uniform grid
        (INCLUDING the panel boundary at +/-1) so adjacent panels visually meet.

        Returns points (P,m,m,3) and the 1D uniform param in [-1,1].
        """
        x, _ = gl_nodes(self.q)
        u = np.linspace(-1.0, 1.0, m)
        L = lagrange_matrix(x, u)               # (m,q)
        # nodes: (P,q,q,3) -> interp in first param then second
        # step 1: along axis-1 (i): (P,m,q,3)
        t1 = np.einsum("mi,piqc->pmqc", L, self.nodes)
        # step 2: along axis-2 (j): (P,m,m,3)
        t2 = np.einsum("nj,pmjc->pmnc", L, t1)
        return t2, u


# ----------------------------------------------------------------------------
# ASCII VTK writers  (no vtk bindings required)
# ----------------------------------------------------------------------------

def write_vti(path, origin, spacing, dims, scalars, name="field"):
    """Write a scalar field as ASCII VTK XML ImageData (.vti).
    dims = (nx,ny,nz); scalars flattened with x fastest, then y, then z."""
    nx, ny, nz = dims
    ext = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n')
        fh.write(f'  <ImageData WholeExtent="{ext}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
                 f'Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">\n')
        fh.write(f'    <Piece Extent="{ext}">\n')
        fh.write(f'      <PointData Scalars="{name}">\n')
        fh.write(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
        s = np.asarray(scalars, np.float32).ravel()
        fh.write("          " + " ".join(f"{v:.6g}" for v in s) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('      </PointData>\n')
        fh.write('    </Piece>\n')
        fh.write('  </ImageData>\n')
        fh.write('</VTKFile>\n')


def write_vtu_quadgrid(path, points, quads, point_scalar=None, name="field"):
    """Write an unstructured grid of quads (VTK_QUAD=9) as ASCII .vtu.
    points (Np,3); quads (Nc,4) index arrays; optional point scalar (Np,)."""
    points = np.asarray(points, float)
    quads = np.asarray(quads, np.int64)
    np_, nc = len(points), len(quads)
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
        fh.write('  <UnstructuredGrid>\n')
        fh.write(f'    <Piece NumberOfPoints="{np_}" NumberOfCells="{nc}">\n')
        if point_scalar is not None:
            fh.write(f'      <PointData Scalars="{name}">\n')
            fh.write(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
            fh.write("          " + " ".join(f"{v:.7g}" for v in np.asarray(point_scalar).ravel()) + "\n")
            fh.write('        </DataArray>\n')
            fh.write('      </PointData>\n')
        fh.write('      <Points>\n')
        fh.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        fh.write("\n".join(f"          {p[0]:.10g} {p[1]:.10g} {p[2]:.10g}" for p in points) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('      </Points>\n')
        fh.write('      <Cells>\n')
        fh.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        fh.write("\n".join("          " + " ".join(map(str, q)) for q in quads) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        fh.write("          " + " ".join(str(4*(i+1)) for i in range(nc)) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        fh.write("          " + " ".join(["9"]*nc) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('      </Cells>\n')
        fh.write('    </Piece>\n')
        fh.write('  </UnstructuredGrid>\n')
        fh.write('</VTKFile>\n')


def write_vtp_lines(path, points, polylines):
    """Write polylines as ASCII .vtp PolyData.
    points (Np,3); polylines = list of index lists."""
    points = np.asarray(points, float)
    np_ = len(points)
    nl = len(polylines)
    conn = []
    offs = []
    tot = 0
    for pl in polylines:
        conn.extend(pl)
        tot += len(pl)
        offs.append(tot)
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n')
        fh.write('  <PolyData>\n')
        fh.write(f'    <Piece NumberOfPoints="{np_}" NumberOfVerts="0" NumberOfLines="{nl}" '
                 f'NumberOfStrips="0" NumberOfPolys="0">\n')
        fh.write('      <Points>\n')
        fh.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        fh.write("\n".join(f"          {p[0]:.10g} {p[1]:.10g} {p[2]:.10g}" for p in points) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('      </Points>\n')
        fh.write('      <Lines>\n')
        fh.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        fh.write("          " + " ".join(map(str, conn)) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        fh.write("          " + " ".join(map(str, offs)) + "\n")
        fh.write('        </DataArray>\n')
        fh.write('      </Lines>\n')
        fh.write('    </Piece>\n')
        fh.write('  </PolyData>\n')
        fh.write('</VTKFile>\n')


# ----------------------------------------------------------------------------
# Panels -> VTK tessellation + wireframe helpers
# ----------------------------------------------------------------------------

def panels_to_tessellation(panels, field=None, m=None):
    """Build a gap-free quad tessellation of all panels.
    Returns (points (Np,3), quads (Nc,4), scalar (Np,) or None)."""
    if m is None:
        m = panels.q + 1
    pts, _ = panels.tessellate(m)                  # (P,m,m,3)
    P = pts.shape[0]
    points = pts.reshape(-1, 3)
    quads = []
    for pi in range(P):
        base = pi * m * m
        for i in range(m - 1):
            for j in range(m - 1):
                a = base + i*m + j
                quads.append([a, a+1, a+m+1, a+m])
    quads = np.array(quads, np.int64)
    scal = field.f(points) if field is not None else None
    return points, quads, scal


def panels_to_wireframe(panels, m=None):
    """Panel-boundary polylines (the 4 curved edges of each panel).
    Returns (points (Np,3), polylines list)."""
    if m is None:
        m = panels.q + 1
    pts, _ = panels.tessellate(m)                  # (P,m,m,3)
    P = pts.shape[0]
    points = pts.reshape(-1, 3)
    lines = []
    for pi in range(P):
        base = pi * m * m
        idx = lambda i, j: base + i*m + j
        top    = [idx(0, j) for j in range(m)]
        right  = [idx(i, m-1) for i in range(m)]
        bottom = [idx(m-1, j) for j in range(m-1, -1, -1)]
        left   = [idx(i, 0) for i in range(m-1, -1, -1)]
        loop = top + right[1:] + bottom[1:] + left[1:]
        lines.append(loop)
    return points, lines
