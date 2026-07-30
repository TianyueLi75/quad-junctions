"""
M2 hybrid-geometry prototype: junction-only QuadElemList (blob with 3 holes relocated to a
planar seam circle at s_seam) + slender-equivalent arms (straight centerline along u_k, circular
cross-section of the iso-surface radius r(s), tapering to a point at the tip).

Goal here is a FAST watertightness / geometry check of the hybrid union before porting to C++:
the junction hole ring at s_seam must exactly equal the slender arm base circle (node-conforming),
so junction-with-3-holes + 3 closed slender arms = one closed surface (flux int n dA -> 0).

The slender arm here mimics CSBQ SlenderElemList: centerline = straight axis u_k, cross-section =
planar circle of radius r(s) in the plane perpendicular to u_k. (Measured: the true arm is
axisymmetric to ~1e-16 and self-tapers, so this is essentially the exact iso-surface arm for
s >~ 0.7; the seam circle pinning error is ~ non-circularity(s_seam).)

CLI:  run_py.sh ybifurc_hybrid.py
"""
import numpy as np
from bifurcation import Field, CFG
from ybifurc import gl01, diff_matrix, frame, ray_root, slerp, flux_area, aspect

FLD = Field(CFG)
THETAS = np.deg2rad(CFG.arm_angles_deg)


class HParams:
    level   = 1.5
    s_seam  = 0.62      # junction<->arm seam plane (arclength along arm axis); non-circ ~1e-5 here
    Na      = 16        # azimuthal panels (mult of 4), shared by junction holes and arm base
    Nr      = 3         # junction radial rings (seam-circle -> pole loop)
    Ns_arm  = 6         # slender arm axial panels (s_seam -> tip)
    r_tip   = 3e-3      # stop the arm where the iso radius drops below this (near-closed tip)


def arm_radius(k, s, level=HParams.level):
    """Iso-surface cross-section radius of arm k in the plane perpendicular to u_k at axial s.
    Returns 0.0 once the axis point s*u is outside the surface (arm has closed)."""
    u, e1, e2 = frame(k)
    p = s * u
    if FLD.f(p[None])[0] <= level:   # axis point outside the surface -> arm closed here
        return 0.0
    # axisymmetric -> one azimuth suffices; average a few for robustness near the junction
    rs = [np.linalg.norm(ray_root(p, np.cos(b)*e1 + np.sin(b)*e2, level) - p)
          for b in np.linspace(0, 2*np.pi, 8, endpoint=False)]
    return float(np.mean(rs))


def seam_circle(k, beta, s_seam=HParams.s_seam, level=HParams.level):
    """Point on the shared seam circle: planar, perpendicular to u_k, radius = iso radius at s_seam."""
    u, e1, e2 = frame(k)
    R = arm_radius(k, s_seam, level)
    return s_seam * u + R * (np.cos(beta) * e1 + np.sin(beta) * e2)


# ---------------------------------------------------------------------------
# Junction: blob with 3 holes, holes relocated to the planar seam circle at s_seam.
# t=0 ring is pinned EXACTLY to the seam circle (matches the arm base); t>0 is ray-rooted
# from the origin onto the iso-surface up to the +-z pole bigon loop (t=1), as in ybifurc.py.
# ---------------------------------------------------------------------------
def junction_hole_dir(k, s):
    """Direction from origin to the seam-circle point at azimuth s (fractional, beta=2 pi s)."""
    d = seam_circle(k, 2*np.pi*s)
    return d / np.linalg.norm(d)

def junction_pole_dir(k, s):
    """The +-z pole bigon loop direction (t=1), identical to ybifurc.junction_dir at t=1."""
    Ptop = np.array([0, 0, 1.0]); Pbot = np.array([0, 0, -1.0])
    phiR = THETAS[k] + np.deg2rad(60); phiL = THETAS[k] - np.deg2rad(60)
    eR = np.array([np.cos(phiR), np.sin(phiR), 0.0]); eL = np.array([np.cos(phiL), np.sin(phiL), 0.0])
    seg = s * 4.0
    if   seg < 1: A, B, w = Ptop, eR, seg - 0
    elif seg < 2: A, B, w = eR, Pbot, seg - 1
    elif seg < 3: A, B, w = Pbot, eL, seg - 2
    else:         A, B, w = eL, Ptop, seg - 3
    return slerp(A, B, w)

def junction_point(k, t, s, p=HParams):
    """Junction geometry map, ray-rooted star-shaped from the origin onto the iso-surface, PLUS a
    decaying correction vector that pins t=0 exactly onto the seam circle. The correction is the
    (tiny, ~non-circularity) gap vector circle-iso(t=0); it decays as (1-t/tb)^2 across the first
    radial ring, so the whole junction stays within ~gap of the iso-surface AND t=0 == arm base."""
    dh = junction_hole_dir(k, s); dp = junction_pole_dir(k, s)
    iso = ray_root(np.zeros(3), slerp(dh, dp, t), p.level)
    tb = 1.0/p.Nr                     # correction confined to the first radial ring
    if t >= tb:
        return iso
    iso0 = ray_root(np.zeros(3), dh, p.level)             # iso point in the circle's direction (t=0)
    circ = seam_circle(k, 2*np.pi*s, p.s_seam, p.level)   # exact arm base point
    decay = (1.0 - t/tb)**2                                # 1 at t=0, C1-zero at t=tb
    return iso + decay*(circ - iso0)

def junction_panels(q, p=HParams):
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
                        nd[i, j] = junction_point(k, t, s, p)
                P.append(nd)
    return P

def junction_offiso(p=HParams, q=12):
    """Max |f-level| over the junction (measures the on-surface deviation introduced by pinning)."""
    a01, _ = gl01(q); m = 0.0
    for k in range(3):
        for ir in range(p.Nr):
            for ia in range(p.Na):
                t0, t1 = ir/p.Nr, (ir+1)/p.Nr; s0, s1 = ia/p.Na, (ia+1)/p.Na
                for i in range(q):
                    t = t0 + a01[i]*(t1-t0)
                    for j in range(q):
                        s = s0 + a01[j]*(s1-s0)
                        P = junction_point(k, t, s, p)
                        m = max(m, abs(FLD.f(P[None])[0] - p.level))
    return m


# ---------------------------------------------------------------------------
# Slender-equivalent arm: straight centerline along u_k, planar circular cross-sections of
# radius r(s), from s_seam to the tip (r -> ~0). Base ring == junction seam circle (conforming).
# ---------------------------------------------------------------------------
def arm_tip_s(k, p=HParams):
    """Largest s with iso radius > r_tip (bisection); the arm closes just beyond this."""
    lo, hi = p.s_seam, 1.30
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if arm_radius(k, mid, p.level) > p.r_tip: lo = mid
        else: hi = mid
    return lo

def arm_panels(q, p=HParams):
    """Tessellate the slender arm surface as (q,q,3) panels for the flux/area check."""
    a01, _ = gl01(q); P = []
    for k in range(3):
        u, e1, e2 = frame(k)
        s_tip = arm_tip_s(k, p)
        for l in range(p.Ns_arm):
            sa = p.s_seam + (s_tip - p.s_seam)*l/p.Ns_arm
            sb = p.s_seam + (s_tip - p.s_seam)*(l+1)/p.Ns_arm
            for ia in range(p.Na):
                b0, b1 = ia/p.Na, (ia+1)/p.Na
                nd = np.empty((q, q, 3))
                for i in range(q):
                    s = sa + a01[i]*(sb-sa); R = arm_radius(k, s, p.level)
                    for j in range(q):
                        beta = 2*np.pi*(b0 + a01[j]*(b1-b0))
                        nd[i, j] = s*u + R*(np.cos(beta)*e1 + np.sin(beta)*e2)
                P.append(nd)
    return P


def build_hybrid(q, p=HParams):
    return junction_panels(q, p), arm_panels(q, p)


def seam_gap(k, p=HParams, nbeta=400):
    """Max distance between the junction hole EDGE CURVE (t->0, ray-rooted from origin onto the
    iso-surface) and the exact planar seam circle (the arm base). This is the true seam mismatch
    the hybrid pays; ~ non-circularity(s_seam) * R."""
    gmax = 0.0
    for s in np.linspace(0, 1, nbeta, endpoint=False):
        j = ray_root(np.zeros(3), junction_hole_dir(k, s), p.level)  # junction t=0 curve point
        a = seam_circle(k, 2*np.pi*s, p.s_seam, p.level)             # exact seam circle point
        gmax = max(gmax, np.linalg.norm(j - a))
    return gmax


def report(p=HParams, orders=(6, 10)):
    print(f"Hybrid Y-bifurcation  s_seam={p.s_seam}  Na={p.Na} Nr={p.Nr} Ns_arm={p.Ns_arm} level={p.level}")
    for k in range(3):
        print(f"  arm {k}: R(s_seam)={arm_radius(k,p.s_seam):.5f}  s_tip={arm_tip_s(k,p):.4f}  "
              f"seam_gap(unpinned)={seam_gap(k,p):.2e}")
    print(f"  junction pinned to seam circle: max|f-level| on junction = {junction_offiso(p):.2e} "
          f"(pinning turns the seam gap into this smooth on-surface deviation)")
    for q in orders:
        a01, wq = gl01(q); D = diff_matrix(a01)
        jP, aP = build_hybrid(q, p); panels = jP + aP
        A, F, mn = flux_area(panels, wq, D)
        jA, jF, jmn = flux_area(jP, wq, D)          # junction-only (open surface)
        asp_j = aspect(jP, q)                        # only the JUNCTION aspect matters (it's the QuadElemList)
        print(f"  order={q:2d} junc={len(jP)} arm={len(aP)} union_panels={len(panels):3d} "
              f"union_area={A:.10f} union|flux|/A={F/A:.2e} "
              f"junc_minspd={jmn:.1e} junc_aspect(max={asp_j.max():.2f},#>2={int((asp_j>2).sum())})")


if __name__ == "__main__":
    report()
