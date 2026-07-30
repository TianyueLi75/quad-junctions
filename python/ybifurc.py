"""
Y-bifurcation swept-O-grid surface mesh generator (Python reference / source of truth
for src/test-ybifurc-geom.cpp).

The surface is the iso-surface f = LEVEL of a sum-of-Gaussians Y field (three arms at
-90/30/150 deg, see bifurcation.Field). It is meshed as three block types that share
edge CURVES (=> watertight) and are each locally STAR-SHAPED so their parameter->surface
map is smooth (=> spectral area convergence and BIE-grade normals):

  * junction : the central blob as a sphere-with-3-holes; 3 azimuthal sectors, each a
               bigon (over the +-z poles) with the arm hole in the middle, meshed as an
               Nr x Na O-grid annulus. Nodes = rays from the ORIGIN.
  * arm tube : Ns x Na sliding-center-ray tube; the ray center slides origin(base)->axis
               (seam), so the base edge exactly matches the junction hole ring and the
               seam is a clean axis cross-section.
  * arm cap  : a BUTTERFLY dome at each tip -- a central GNOMONIC square (non-degenerate
               everywhere) + 4 Coons arc blocks fairing out to the circular seam ring
               (= the tube seam curve, so watertight). Center-rays from a point inside.
               NB the plain FG-squircle cap (add_cap) has a DEGENERATE Jacobian at its 4
               corners -> ~1e-6 normal error at the seam and is NOT BIE-grade; do not use.

Projection is a fixed-RAY root solve f(c0 + s*d) = level (star-shaped about each block's
own center), NOT gradient-flow Newton -- gradient flow caustics at convex tips and floors
area/normals at ~1e-5.

CLI:  run_py.sh ybifurc.py            # build at defaults, verify, write ybifurc_mesh.vtu(+wire)
"""
import numpy as np
from scipy.optimize import brentq
from bifurcation import Field, CFG, PanelSet, panels_to_tessellation, \
    panels_to_wireframe, write_vtu_quadgrid, write_vtp_lines

FLD = Field(CFG)                       # the Gaussian Y field (canonical; matches C++ YField)
THETAS = np.deg2rad(CFG.arm_angles_deg)


# ----------------------------------------------------------------------------
# Parameters (production defaults -- match src/test-ybifurc-geom.cpp YSwept)
# ----------------------------------------------------------------------------
class YParams:
    level     = 1.5        # iso-surface level
    alpha_deg = 38.0       # junction hole half-angle
    Lseam     = 0.88       # arm-axis arc-length of the tube->cap seam (arm on-axis pole ~1.10)
    zc_off    = 0.10       # cap center offset inside the seam (along -u_k)
    core_frac = 0.45       # butterfly-cap core half-size (tangent units)
    Nr        = 2          # junction radial rings per sector       } base panel counts,
    Na        = 16         # azimuthal panels (MUST be a mult of 4)  } chosen for ~square
    Ns        = 4          # arm axial panels                        } panels (aspect < 1.85);
    Ncap      = 2          # butterfly core & arc panels per dir      } scale together to h-refine


# ----------------------------------------------------------------------------
# Gauss-Legendre nodes/weights on [0,1] and the on-node derivative matrix
# ----------------------------------------------------------------------------
def gl01(q):
    x, w = np.polynomial.legendre.leggauss(q)     # [-1,1]
    return 0.5*(x + 1.0), 0.5*w                   # -> [0,1]

def diff_matrix(nodes):
    """D[i,j] = l_j'(nodes[i]) for the Lagrange basis on `nodes` (barycentric)."""
    n = len(nodes); x = nodes
    w = np.ones(n)
    for j in range(n):
        d = x[j] - x; d[j] = 1.0; w[j] = 1.0/np.prod(d)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: D[i, j] = (w[j]/w[i])/(x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return D


# ----------------------------------------------------------------------------
# Star-shaped ray projection + small geometry helpers
# ----------------------------------------------------------------------------
def ray_root(c0, d, level=YParams.level, smax=3.0):
    """From interior center c0 along direction d, find the surface point f=level."""
    d = d/np.linalg.norm(d)
    return c0 + brentq(lambda s: FLD.f((c0 + s*d)[None])[0] - level, 1e-5, smax)*d

def slerp(a, b, w):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    dot = np.clip(a@b, -1, 1); om = np.arccos(dot)
    return a if om < 1e-12 else (np.sin((1-w)*om)*a + np.sin(w*om)*b)/np.sin(om)

def frame(k):
    """Arm k frame: axis u (in xy-plane), e1='up'(+z), e2=e1 x u (+phi tangent)."""
    u = np.array([np.cos(THETAS[k]), np.sin(THETAS[k]), 0.0])
    e1 = np.array([0.0, 0.0, 1.0])
    return u, e1, np.cross(e1, u)


# ----------------------------------------------------------------------------
# Block 1: junction (sphere-with-3-holes, rays from the origin)
# ----------------------------------------------------------------------------
def junction_dir(k, t, s, p=YParams):
    """Direction for sector k at annulus params t (radial, 0=hole ..1=bigon loop), s (around)."""
    u, e1, e2 = frame(k)
    alpha = np.deg2rad(p.alpha_deg)
    Ptop = np.array([0, 0, 1.0]); Pbot = np.array([0, 0, -1.0])
    phiR = THETAS[k] + np.deg2rad(60); phiL = THETAS[k] - np.deg2rad(60)
    eR = np.array([np.cos(phiR), np.sin(phiR), 0.0]); eL = np.array([np.cos(phiL), np.sin(phiL), 0.0])
    beta = 2*np.pi*s
    rad = np.cos(beta)*e1 + np.sin(beta)*e2
    inner = np.cos(alpha)*u + np.sin(alpha)*rad; inner /= np.linalg.norm(inner)
    seg = s*4.0
    if   seg < 1: A, B, w = Ptop, eR, seg-0     # 4-sided outer bigon loop (great-circle arcs)
    elif seg < 2: A, B, w = eR, Pbot, seg-1
    elif seg < 3: A, B, w = Pbot, eL, seg-2
    else:         A, B, w = eL, Ptop, seg-3
    return slerp(inner, slerp(A, B, w), t)

def junction_panels(q, p=YParams):
    a01, _ = gl01(q); P = []
    for k in range(3):
        for ir in range(p.Nr):
            for ia in range(p.Na):
                t0, t1 = ir/p.Nr, (ir+1)/p.Nr; s0, s1 = ia/p.Na, (ia+1)/p.Na
                nd = np.empty((q, q, 3))
                for i in range(q):
                    t = t0 + a01[i]*(t1-t0)
                    for j in range(q):
                        s = s0 + a01[j]*(s1-s0)
                        nd[i, j] = ray_root(np.zeros(3), junction_dir(k, t, s, p), p.level)
                P.append(nd)
    return P


# ----------------------------------------------------------------------------
# Block 2: arm tube (sliding-center ray; base edge = junction hole ring)
# ----------------------------------------------------------------------------
def arm_point(k, eta, beta, p=YParams):
    """eta in [0,1] base(=junction hole)->seam; beta azimuth around the arm."""
    u, e1, e2 = frame(k)
    alpha = np.deg2rad(p.alpha_deg)
    c = eta*p.Lseam*u                                       # center slides origin -> axis
    rad = np.cos(beta)*e1 + np.sin(beta)*e2
    d = (1-eta)*np.cos(alpha)*u + ((1-eta)*np.sin(alpha) + eta)*rad
    return ray_root(c, d, p.level)

def arm_panels(q, p=YParams):
    a01, _ = gl01(q); P = []
    for k in range(3):
        for l in range(p.Ns):
            e0, e1 = l/p.Ns, (l+1)/p.Ns
            for ia in range(p.Na):
                b0, b1 = ia/p.Na, (ia+1)/p.Na
                nd = np.empty((q, q, 3))
                for i in range(q):
                    eta = e0 + a01[i]*(e1-e0)
                    for j in range(q):
                        beta = 2*np.pi*(b0 + a01[j]*(b1-b0))
                        nd[i, j] = arm_point(k, eta, beta, p)
                P.append(nd)
    return P


# ----------------------------------------------------------------------------
# Block 3: arm cap (butterfly dome: gnomonic core + 4 Coons arc blocks)
# ----------------------------------------------------------------------------
def cap_panels(q, p=YParams):
    a01, _ = gl01(q); P = []; h = p.core_frac
    for k in range(3):
        u, e1, e2 = frame(k)
        c0 = (p.Lseam - p.zc_off)*u
        gdir = lambda x, y: u + x*e1 + y*e2                 # gnomonic direction (non-degenerate)
        # central gnomonic square [-h,h]^2
        for ic in range(p.Ncap):
            for jc in range(p.Ncap):
                x0 = -h+2*h*ic/p.Ncap; x1 = -h+2*h*(ic+1)/p.Ncap
                y0 = -h+2*h*jc/p.Ncap; y1 = -h+2*h*(jc+1)/p.Ncap
                nd = np.empty((q, q, 3))
                for i in range(q):
                    yy = y0 + a01[i]*(y1-y0)
                    for j in range(q):
                        xx = x0 + a01[j]*(x1-x0); nd[i, j] = ray_root(c0, gdir(xx, yy), p.level)
                P.append(nd)
        # 4 Coons arc blocks: eta 0=core edge -> 1=seam ring; xi along; rotate kk*90deg
        for kk in range(4):
            rot = kk*np.pi/2; cr, sr = np.cos(rot), np.sin(rot)
            for ir in range(p.Ncap):
                for ia in range(p.Ncap):
                    e0 = ir/p.Ncap; e1r = (ir+1)/p.Ncap; x0 = ia/p.Ncap; x1 = (ia+1)/p.Ncap
                    nd = np.empty((q, q, 3))
                    for i in range(q):
                        xi = x0 + a01[i]*(x1-x0)
                        for j in range(q):
                            eta = e0 + a01[j]*(e1r-e0)
                            din = gdir(cr*h - sr*h*(2*xi-1), sr*h + cr*h*(2*xi-1))   # rotated core edge
                            beta = rot + (-np.pi/4 + xi*np.pi/2)
                            dout = arm_point(k, 1.0, beta, p) - c0                    # seam ring (= tube seam)
                            nd[i, j] = ray_root(c0, slerp(din, dout, eta), p.level)
                    P.append(nd)
    return P


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
def build_ybifurcation(q, p=YParams):
    """Full swept-O-grid Y-bifurcation as a list of (q,q,3) GL-node panels."""
    return junction_panels(q, p) + arm_panels(q, p) + cap_panels(q, p)


# ----------------------------------------------------------------------------
# Verification (curved-surface flux/area, panel squareness) + self-convergence
# ----------------------------------------------------------------------------
def flux_area(panels, wq, D):
    """Return (area, |flux|=|int n dA| with n oriented outward via -grad f, min surface speed)."""
    A = 0.0; F = np.zeros(3); minspd = 1e9
    for nd in panels:
        dXa = np.einsum("ik,kjc->ijc", D, nd); dXb = np.einsum("jk,ikc->ijc", D, nd)
        nrm = np.cross(dXa, dXb)
        g = FLD.grad(nd.reshape(-1, 3)).reshape(nd.shape)
        sign = np.sign(-np.sum(nrm*g, axis=-1))[..., None]
        F += np.sum((wq[:, None]*wq[None, :])[..., None]*sign*nrm, axis=(0, 1))
        spd = np.linalg.norm(nrm, axis=-1)
        A += np.sum((wq[:, None]*wq[None, :])*spd); minspd = min(minspd, spd.min())
    return A, np.linalg.norm(F), minspd

def aspect(panels, q):
    """Per-panel aspect = max/min of the two param-direction GL-node polyline lengths."""
    r = []
    for nd in panels:
        Lu = np.mean([np.sum(np.linalg.norm(np.diff(nd[:, j, :], axis=0), axis=1)) for j in range(q)])
        Lv = np.mean([np.sum(np.linalg.norm(np.diff(nd[i, :, :], axis=0), axis=1)) for i in range(q)])
        r.append(max(Lu, Lv)/max(min(Lu, Lv), 1e-30))
    return np.array(r)

def report(p=YParams, orders=(4, 8, 12, 16)):
    """Build at several orders; print area self-convergence, watertightness, squareness."""
    print(f"Y-bifurcation swept-O-grid  Nr={p.Nr} Na={p.Na} Ns={p.Ns} Ncap={p.Ncap} "
          f"alpha={p.alpha_deg} Lseam={p.Lseam} level={p.level}")
    prev = None
    for q in orders:
        a01, wq = gl01(q); D = diff_matrix(a01)
        panels = build_ybifurcation(q, p)
        A, F, mn = flux_area(panels, wq, D); asp = aspect(panels, q)
        line = (f"  order={q:2d} panels={len(panels):3d} area={A:.13f} |flux|/A={F/A:.1e} "
                f"minspd={mn:.1e} aspect(max={asp.max():.2f},#>2={int((asp>2).sum())})")
        if prev is not None: line += f"  |dA|={abs(A-prev):.2e}"
        print(line); prev = A


# ----------------------------------------------------------------------------
# VTK export (surface tessellation + panel-edge wireframe)
# ----------------------------------------------------------------------------
def write_mesh(panels, q, prefix="ybifurc_mesh"):
    ps = PanelSet(np.array(panels), np.zeros((len(panels), 4, 3)), q)
    pts, quads, scal = panels_to_tessellation(ps, field=FLD, m=q+1)
    write_vtu_quadgrid(f"{prefix}.vtu", pts, quads, point_scalar=scal, name="f")
    wp, wl = panels_to_wireframe(ps, m=q+1)
    write_vtp_lines(f"{prefix}_wire.vtp", wp, wl)
    print(f"wrote {prefix}.vtu (+_wire.vtp)  [{len(panels)} panels]")


if __name__ == "__main__":
    report()
    q = 10
    write_mesh(build_ybifurcation(q, YParams), q)
