"""
poly-fit centerline core (proof-of-concept, standalone numerics).

Pipeline target (new 'pf' pipeline), replacing the downsample C2-cubic body:
  (reused upstream)  vmtk polyline
    -> Taubin low-pass (lam=0.5, mu=-0.53, 64 passes)
    -> curvature/turn guard
    -> junction mouths (pos + axis tangent) via canonical junction
    -> quintic Hermite mouth->centerline (straight lead + corner)
  (NEW here)         GLOBAL POLYNOMIAL FIT of the body point set
    -> uniform-arclength degree-10 Chebyshev panels along the fitted polynomial

Key idea: a single global polynomial is analytic (C-inf). If its degree d <= 10,
then its restriction to ANY subinterval is still degree <= 10, so every degree-10
Chebyshev panel reproduces the curve EXACTLY -- no knot straddle, no panel-snap,
spectral surface resolution regardless of panel boundaries.

This module is pure numpy; it takes arrays so it is independent of the .graph loader.
"""
import numpy as np
from numpy.polynomial import chebyshev as C


# ----------------------------------------------------------------------------
# reused upstream stages (reimplemented in python to match the C++ constants)
# ----------------------------------------------------------------------------
def taubin_smooth(P, lam=0.5, mu=-0.53, passes=64, fix_ends=True):
    """Taubin lambda/mu low-pass. P: (N,3). Pins endpoints (ends anchored to junctions)."""
    P = np.asarray(P, float).copy()
    N = len(P)
    if N < 3:
        return P

    def one_pass(X, w):
        Y = X.copy()
        lap = 0.5 * (X[:-2] + X[2:]) - X[1:-1]       # discrete Laplacian, interior
        Y[1:-1] = X[1:-1] + w * lap
        return Y

    for _ in range(passes):
        P = one_pass(P, lam)
        P = one_pass(P, mu)
    # endpoints already untouched (interior-only update) -> fix_ends is implicit
    return P


def arclength(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def resample_uniform(P, K):
    """resample polyline P to K points uniform in arclength."""
    s = arclength(P)
    L = s[-1]
    sk = np.linspace(0, L, K)
    out = np.empty((K, 3))
    for j in range(3):
        out[:, j] = np.interp(sk, s, P[:, j])
    return out, L


def menger_rho(P):
    """three-point (Menger) circumradius at each interior vertex. returns rho (N-2,)."""
    a = P[:-2]; b = P[1:-1]; c = P[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = 0.5 * cross
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = (ab * bc * ca) / (4.0 * area)
    rho[area < 1e-14] = np.inf
    return rho


def curvature_guard(P, r, rho_over_r_min=1.5, extra_taubin=200):
    """
    'smooth out turns too sharp for the radius': if any built Menger rho/r drops
    below the self-fold threshold, apply extra local Taubin passes until it clears
    (or give up and report). Returns (P_fixed, worst_rho_over_r).
    """
    P = P.copy()
    for _ in range(extra_taubin):
        rho = menger_rho(P)
        worst = np.nanmin(rho) / r
        if worst >= rho_over_r_min:
            break
        P = taubin_smooth(P, passes=1)
    rho = menger_rho(P)
    return P, np.nanmin(rho) / r


# ----------------------------------------------------------------------------
# quintic Hermite (mouth stub -> body start): position+tangent+curvature at both ends
# ----------------------------------------------------------------------------
def quintic_hermite(p0, t0, a0, p1, t1, a1, n=40):
    """
    quintic Hermite on xi in [0,1] matching:
      H(0)=p0, H'(0)=t0, H''(0)=a0 ; H(1)=p1, H'(1)=t1, H''(1)=a1.
    (a0=a1=0 => curvature-free corner, the downsample 'natural' choice.)
    """
    xi = np.linspace(0, 1, n)[:, None]
    h0 = 1 - 10*xi**3 + 15*xi**4 - 6*xi**5
    h1 = xi - 6*xi**3 + 8*xi**4 - 3*xi**5
    h2 = 0.5*xi**2 - 1.5*xi**3 + 1.5*xi**4 - 0.5*xi**5
    h3 = 10*xi**3 - 15*xi**4 + 6*xi**5
    h4 = -4*xi**3 + 7*xi**4 - 3*xi**5
    h5 = 0.5*xi**3 - xi**4 + 0.5*xi**5
    return (h0*p0 + h1*t0 + h2*a0 + h3*p1 + h4*t1 + h5*a1)


# ----------------------------------------------------------------------------
# NEW: constrained global polynomial fit (Chebyshev basis, KKT constraints)
# ----------------------------------------------------------------------------
def _cheb_design(t, deg, dom=(0.0, 1.0)):
    """Vandermonde of Chebyshev basis T_0..T_deg evaluated at t (mapped to [-1,1])."""
    t = np.atleast_1d(np.asarray(t, float))
    V = np.empty((len(t), deg + 1))
    for k in range(deg + 1):
        c = np.zeros(deg + 1); c[k] = 1.0
        V[:, k] = C.chebval(_map(t, dom), c)
    return V


def _map(t, dom):
    a, b = dom
    return (2.0 * t - (a + b)) / (b - a)


def _cheb_basis_at(t, deg, order, dom=(0.0, 1.0)):
    """row vector of the `order`-th derivative of each Cheb basis fn at scalar t."""
    a, b = dom
    scale = (2.0 / (b - a)) ** order
    row = np.empty(deg + 1)
    for k in range(deg + 1):
        c = np.zeros(deg + 1); c[k] = 1.0
        if order > 0:
            c = C.chebder(c, m=order)
        row[k] = C.chebval(_map(np.array([t]), dom), c)[0] * (scale if order else 1.0)
    return row


def fit_coord_constrained(t, y, deg, constraints, dom=(0.0, 1.0)):
    """
    min ||V c - y||^2 s.t. B c = d, one scalar coordinate.
    constraints: list of (t_node, order, value). Solve KKT.
    """
    V = _cheb_design(t, deg, dom)
    A = V.T @ V
    g = V.T @ y
    if constraints:
        B = np.array([_cheb_basis_at(tn, deg, order, dom) for (tn, order, _) in constraints])
        d = np.array([val for (_, _, val) in constraints])
        m = B.shape[0]
        K = np.block([[2 * A, B.T], [B, np.zeros((m, m))]])
        rhs = np.concatenate([2 * g, d])
        sol = np.linalg.solve(K, rhs)
        return sol[:deg + 1]
    return np.linalg.solve(A, g)


class PolyArm:
    """a fitted arm centerline: 3 Chebyshev polynomials of degree `deg` over t in [0,1]."""
    def __init__(self, coeffs, deg, dom=(0.0, 1.0)):
        self.coeffs = coeffs            # (3, deg+1) Cheb coeffs
        self.deg = deg
        self.dom = dom

    def eval(self, t):
        t = np.atleast_1d(t)
        V = _cheb_design(t, self.deg, self.dom)
        return V @ self.coeffs.T        # (len(t),3)

    def deriv(self, t, order=1):
        t = np.atleast_1d(t)
        a, b = self.dom
        scale = (2.0 / (b - a)) ** order
        out = np.empty((len(t), 3))
        for j in range(3):
            c = C.chebder(self.coeffs[j], m=order)
            out[:, j] = C.chebval(_map(t, self.dom), c) * scale
        return out

    def rho_over_r(self, r, n=400):
        t = np.linspace(0, 1, n)
        d1 = self.deriv(t, 1); d2 = self.deriv(t, 2)
        num = np.linalg.norm(np.cross(d1, d2), axis=1)
        den = np.linalg.norm(d1, axis=1) ** 3
        kappa = np.where(num > 1e-14, num / den, 0.0)
        rho = np.where(kappa > 1e-14, 1.0 / kappa, np.inf)
        return rho / r


# ----------------------------------------------------------------------------
# panel resolvability: CSBQ lays `cheb`=10 Chebyshev-Gauss nodes per panel
# (degree-9 interpolant). A global poly of degree <= 9 is reproduced EXACTLY on
# every panel regardless of boundary placement -> no knot straddle.
# ----------------------------------------------------------------------------
def cheb_gauss_nodes(N):
    i = np.arange(N)
    return (1.0 - np.cos((2 * i + 1) * np.pi / (2 * N))) / 2.0   # on [0,1]


def panel_interp_error(eval_fn, panel_edges, cheb=10, oversample=60):
    """
    For each panel [ta,tb], sample eval_fn at `cheb` Cheb-Gauss nodes, build the
    degree-(cheb-1) interpolant, and measure its max error vs eval_fn on a dense
    in-panel grid. Returns (max over panels, per-panel array). This is exactly the
    geometry-formation error CSBQ's per-panel toroidal quadrature reads.
    """
    nds = cheb_gauss_nodes(cheb)
    per = []
    for ta, tb in zip(panel_edges[:-1], panel_edges[1:]):
        tn = ta + nds * (tb - ta)
        Yn = eval_fn(tn)                       # (cheb,3) sample values
        # interpolate each coord by a degree-(cheb-1) poly in local xi in [-1,1]
        xi = 2 * nds - 1
        tg = np.linspace(0, 1, oversample)
        xg = 2 * tg - 1
        Vn = np.vander(xi, cheb, increasing=True)
        Vg = np.vander(xg, cheb, increasing=True)
        err = 0.0
        true = eval_fn(ta + tg * (tb - ta))
        for j in range(3):
            c = np.linalg.solve(Vn, Yn[:, j])
            recon = Vg @ c
            err = max(err, np.abs(recon - true[:, j]).max())
        per.append(err)
    per = np.array(per)
    return per.max(), per


def natural_cubic(knots):
    """natural cubic spline through knots (K,3) in arclength param; returns eval_fn(tau in [0,1])
    plus the knot arclength fractions (for the downsample-body comparison / knot straddle)."""
    K = len(knots)
    s = arclength(knots); L = s[-1]; frac = s / L
    h = np.diff(s)
    M = np.zeros((3, K))                  # second derivatives per coord
    for j in range(3):
        A = np.zeros((K, K)); b = np.zeros(K)
        A[0, 0] = 1.0; A[-1, -1] = 1.0    # natural
        for i in range(1, K - 1):
            A[i, i - 1] = h[i - 1]; A[i, i] = 2 * (h[i - 1] + h[i]); A[i, i + 1] = h[i]
            b[i] = 6 * ((knots[i + 1, j] - knots[i, j]) / h[i] - (knots[i, j] - knots[i - 1, j]) / h[i - 1])
        M[j] = np.linalg.solve(A, b)

    def ev(tau):
        tau = np.atleast_1d(tau); sq = tau * L
        idx = np.clip(np.searchsorted(s, sq) - 1, 0, K - 2)
        out = np.empty((len(tau), 3))
        for n in range(len(tau)):
            i = idx[n]; hi = h[i]; a = (s[i + 1] - sq[n]) / hi; bb = (sq[n] - s[i]) / hi
            for j in range(3):
                out[n, j] = (a * knots[i, j] + bb * knots[i + 1, j]
                             + (a ** 3 - a) * hi ** 2 / 6 * M[j][i]
                             + (bb ** 3 - bb) * hi ** 2 / 6 * M[j][i + 1])
        return out
    return ev, frac


def pick_degree(report, rel_impr=0.15, prefer_max=9, floor_rel=3.0):
    """knee selector: 'as low as possible while staying close'. Walk the ladder; stop at
    the smallest degree beyond which each extra degree cuts maxdev by < rel_impr, OR once
    maxdev is within floor_rel x of the best-on-ladder floor. Prefer deg<=prefer_max (panel-exact)."""
    devs = [row['maxdev_chk'] for row in report]
    degs = [row['deg'] for row in report]
    best = min(devs)
    for k in range(len(report)):
        at_floor = devs[k] <= floor_rel * best
        stalled = (k + 1 < len(report) and devs[k + 1] > (1 - rel_impr) * devs[k])
        if at_floor and (stalled or degs[k] >= prefer_max):
            return degs[k]
    return degs[int(np.argmin(devs))]


# ----------------------------------------------------------------------------
# NEW: piecewise global-polynomial fit (wide knots, high per-segment order,
# C^cont continuity at joints). S=1 => a single global polynomial (short arms).
# Each segment is analytic (degree<=9 => panel-exact); knots are wide so panels
# snapped to them have no slivers.
# ----------------------------------------------------------------------------
class SegPolyArm:
    def __init__(self, coeffs, deg, fknots):
        self.coeffs = coeffs          # (S,3,deg+1) Cheb coeffs in local u per segment
        self.deg = deg
        self.f = np.asarray(fknots)   # segment boundaries in t in [0,1], len S+1
        self.S = len(fknots) - 1

    def _piece(self, t):
        return np.clip(np.searchsorted(self.f, t, side='right') - 1, 0, self.S - 1)

    def eval(self, t):
        t = np.atleast_1d(np.asarray(t, float)); out = np.empty((len(t), 3))
        pc = self._piece(t)
        for i in range(self.S):
            m = pc == i
            if not m.any():
                continue
            h = self.f[i + 1] - self.f[i]; u = (t[m] - self.f[i]) / h
            V = _cheb_design(u, self.deg)
            out[m] = V @ self.coeffs[i].T
        return out

    def deriv(self, t, order=1):
        t = np.atleast_1d(np.asarray(t, float)); out = np.empty((len(t), 3))
        pc = self._piece(t)
        for i in range(self.S):
            m = pc == i
            if not m.any():
                continue
            h = self.f[i + 1] - self.f[i]; u = (t[m] - self.f[i]) / h; sc = (1.0 / h) ** order
            for j in range(3):
                c = C.chebder(self.coeffs[i][j], m=order)
                out[m, j] = C.chebval(_map(u, (0, 1)), c) * sc
        return out

    def rho_over_r(self, r, n=1200):
        t = np.linspace(0, 1, n); d1 = self.deriv(t, 1); d2 = self.deriv(t, 2)
        num = np.linalg.norm(np.cross(d1, d2), axis=1); den = np.linalg.norm(d1, axis=1) ** 3
        kappa = np.where(num > 1e-14, num / np.maximum(den, 1e-30), 0.0)
        rho = np.where(kappa > 1e-12, 1.0 / kappa, np.inf)
        return rho / r


def _locbasis(u, deg, order):
    """row of order-th derivative of each Cheb basis fn at scalar u in [0,1] (local, unscaled)."""
    row = np.empty(deg + 1)
    for k in range(deg + 1):
        c = np.zeros(deg + 1); c[k] = 1.0
        if order:
            c = C.chebder(c, m=order)
        row[k] = C.chebval(_map(np.array([u]), (0, 1)), c)[0] * ((2.0) ** order if order else 1.0)
    return row


def adaptive_body_window(sm, r, Llead=None, base_k=2.0, step_k=2.0, cap_k=12.0,
                         rho_tol=1.6, end_frac=0.12, deg=9, cont=4, Lseg=45.0, turn_seg_deg=35.0):
    """
    Choose the body window [room_A, L-room_B] adaptively: push an end's room deeper (so the
    sharp junction-entry hook is carried by the straight lead + quintic corner, not the body
    polynomial) while that end's fitted rho/r folds below rho_tol. room = Llead + k*r.
    Cures the Class-A boundary overshoot. Returns (body, arm, info, kA, kB).
    """
    if Llead is None:
        Llead = 0.5 * r
    s = arclength(sm); L = s[-1]
    kA, kB = base_k, base_k

    def fit_with(kA, kB):
        iA = max(int(np.searchsorted(s, Llead + kA * r)), 2)
        iB = min(int(np.searchsorted(s, L - (Llead + kB * r))), len(sm) - 3)
        if iB - iA < max(8, deg + 3):
            return None
        body = sm[iA:iB + 1]
        arm, info = fit_arm_segmented(body, r, body[0], body[-1], Lseg=Lseg,
                                      turn_seg_deg=turn_seg_deg, deg=deg, cont=cont)
        return body, arm, info

    res = fit_with(kA, kB)
    for _ in range(12):
        if res is None:
            break
        body, arm, info = res
        tt = np.linspace(0, 1, 1200); rr = arm.rho_over_r(r, 1200)
        loA = rr[tt < end_frac].min() if (tt < end_frac).any() else np.inf
        loB = rr[tt > 1 - end_frac].min() if (tt > 1 - end_frac).any() else np.inf
        okA = loA >= rho_tol or kA >= cap_k
        okB = loB >= rho_tol or kB >= cap_k
        if okA and okB:
            break
        if not okA:
            kA = min(cap_k, kA + step_k)
        if not okB:
            kB = min(cap_k, kB + step_k)
        nxt = fit_with(kA, kB)
        if nxt is None:
            break
        res = nxt
    body, arm, info = res
    info = dict(info); info['kA'] = kA; info['kB'] = kB
    return body, arm, info, kA, kB


def segment_knots(body, Lseg=45.0, turn_seg_deg=35.0, knot_mode='turn'):
    """
    Choose body segment boundaries (in t in [0,1]). S from BOTH a length rule (L/Lseg)
    and a turning rule (total turn / turn_seg_deg), so long OR sharply-bending arms get more
    segments. 'turn' places knots at equal cumulative turning (knots cluster where it bends);
    'arclength' places them uniformly in arclength.
    """
    s = arclength(body); L = s[-1]
    tv = np.diff(body, axis=0); tn = tv / np.linalg.norm(tv, axis=1, keepdims=True)
    ang = np.arccos(np.clip((tn[:-1] * tn[1:]).sum(1), -1, 1))
    cturn = np.concatenate([[0.0], np.cumsum(ang)])
    total_turn_deg = float(np.degrees(cturn[-1]))
    S = max(int(np.ceil(L / Lseg)), int(np.ceil(total_turn_deg / turn_seg_deg)), 1)
    if knot_mode == 'turn' and cturn[-1] > 1e-9:
        targ = np.linspace(0, cturn[-1], S + 1)
        svtx = s[1:]                                   # arclength at vertices where cturn is defined
        f = np.interp(targ, cturn, svtx) / L
    else:
        f = np.linspace(0, 1, S + 1)
    f[0] = 0.0; f[-1] = 1.0
    f = np.unique(np.clip(np.sort(f), 0, 1))
    return f, total_turn_deg


def fit_arm_segmented(body, r, p0, p1, Lseg=45.0, turn_seg_deg=35.0, knot_mode='turn',
                      deg=9, cont=2, n_fit_per_seg=24, match_tangent=False, u0=None, u1=None):
    """
    Piecewise-polynomial body fit. Segments chosen by segment_knots (length + turning);
    per-segment degree-`deg` Cheb poly; C^cont continuity at interior joints; endpoints fixed
    to p0,p1 (optional end-tangent match). S=1 => a single global polynomial. Returns (arm, info).
    """
    body = np.asarray(body, float)
    s = arclength(body); L = s[-1]; t = s / L
    f, total_turn = segment_knots(body, Lseg, turn_seg_deg, knot_mode)
    S = len(f) - 1
    nc = deg + 1
    pc = np.clip(np.searchsorted(f, t, side='right') - 1, 0, S - 1)

    coeffs = np.empty((S, 3, nc)); report = {}
    for j in range(3):                                # independent per coordinate
        ncol = S * nc
        rows_A = []; rhs_b = []
        # data rows (downsample per segment for the fit; keep all for checks later)
        for i in range(S):
            idx = np.where(pc == i)[0]
            if len(idx) < deg + 2:                    # borrow nearest points for thin segments
                mid = 0.5 * (f[i] + f[i + 1]); idx = np.argsort(np.abs(t - mid))[:deg + 3]
            if n_fit_per_seg and len(idx) > n_fit_per_seg:
                idx = idx[np.unique(np.linspace(0, len(idx) - 1, n_fit_per_seg).astype(int))]
            h = f[i + 1] - f[i]; u = (t[idx] - f[i]) / h
            Vi = _cheb_design(u, deg)
            for q, ii in enumerate(idx):
                row = np.zeros(ncol); row[i * nc:(i + 1) * nc] = Vi[q]
                rows_A.append(row); rhs_b.append(body[ii, j])
        A = np.array(rows_A); bvec = np.array(rhs_b)
        # constraints
        Brows = []; dvals = []
        # endpoints (position)
        r0 = np.zeros(ncol); r0[0:nc] = _locbasis(0.0, deg, 0); Brows.append(r0); dvals.append(p0[j])
        r1 = np.zeros(ncol); r1[(S - 1) * nc:S * nc] = _locbasis(1.0, deg, 0); Brows.append(r1); dvals.append(p1[j])
        if match_tangent and u0 is not None:
            h0 = f[1] - f[0]
            rr = np.zeros(ncol); rr[0:nc] = _locbasis(0.0, deg, 1) / h0
            Brows.append(rr); dvals.append(L * u0[j])
            hL = f[S] - f[S - 1]
            rr = np.zeros(ncol); rr[(S - 1) * nc:S * nc] = _locbasis(1.0, deg, 1) / hL
            Brows.append(rr); dvals.append(L * u1[j])
        # C^cont continuity at interior joints (derivative orders 0..cont)
        for i in range(1, S):
            hL = f[i] - f[i - 1]; hR = f[i + 1] - f[i]
            for m in range(cont + 1):
                row = np.zeros(ncol)
                row[(i - 1) * nc:i * nc] = _locbasis(1.0, deg, m) / hL ** m
                row[i * nc:(i + 1) * nc] = -_locbasis(0.0, deg, m) / hR ** m
                Brows.append(row); dvals.append(0.0)
        B = np.array(Brows); dv = np.array(dvals)
        # KKT
        G = A.T @ A; g = A.T @ bvec; mB = B.shape[0]
        K = np.block([[2 * G, B.T], [B, np.zeros((mB, mB))]])
        rhs = np.concatenate([2 * g, dv])
        sol = np.linalg.solve(K, rhs)[:ncol]
        coeffs[:, j, :] = sol.reshape(S, nc)
    arm = SegPolyArm(coeffs, deg, f)
    # fidelity: geometric nearest-point deviation vs ALL body points
    tt = np.linspace(0, 1, max(1500, 40 * S)); Cc = arm.eval(tt)
    dgeo = np.sqrt(((body[:, None, :] - Cc[None, :, :]) ** 2).sum(-1)).min(1)
    rr = arm.rho_over_r(r)
    info = dict(S=S, deg=deg, cont=cont, L=L, fknots=f, total_turn_deg=total_turn,
                maxdev=float(dgeo.max() / r), meandev=float(dgeo.mean() / r),
                min_rho_r=float(rr.min()))
    return arm, info


def fit_arm(points, r, p0, u0, p1, u1,
            deg_min=3, deg_max=14, tol_over_r=1e-3,
            match_tangent=True, match_curv0=False, n_fit=None):
    """
    Fit the body point set with a GLOBAL polynomial.
      points : (N,3) smoothed body centerline (endpoints ~ p0,p1).
      r      : tube radius (for relative tolerances).
      p0,p1  : endpoint positions (corner/lead ends). u0,u1: unit end tangents.
    Downsample to n_fit points for the fit; hold out the rest as fidelity checks.
    Raise degree until max held-out deviation/r < tol_over_r (or deg_max; then table).
    """
    P = np.asarray(points, float)
    s = arclength(P); L = s[-1]; t = s / L            # chord param in [0,1]
    # end-tangent targets: arclength-normalized speed ~ L
    spd = L
    def build_constraints():
        cons = [[], [], []]   # per coordinate
        for j in range(3):
            cons[j].append((0.0, 0, p0[j]))
            cons[j].append((1.0, 0, p1[j]))
            if match_tangent:
                cons[j].append((0.0, 1, spd * u0[j]))
                cons[j].append((1.0, 1, spd * u1[j]))
            if match_curv0:
                cons[j].append((0.0, 2, 0.0))
                cons[j].append((1.0, 2, 0.0))
        return cons

    if n_fit is None:
        n_fit = min(len(P), max(2 * (deg_max + 1), 24))
    fit_idx = np.unique(np.linspace(0, len(P) - 1, n_fit).astype(int))
    chk_idx = np.setdiff1d(np.arange(len(P)), fit_idx)
    cons = build_constraints()

    report = []
    for deg in range(deg_min, deg_max + 1):
        coeffs = np.empty((3, deg + 1))
        for j in range(3):
            coeffs[j] = fit_coord_constrained(t[fit_idx], P[fit_idx, j], deg, cons[j])
        arm = PolyArm(coeffs, deg)
        # deviation on ALL points at their chord param (held-out check)
        fit_pts = arm.eval(t)
        dev = np.linalg.norm(fit_pts - P, axis=1)
        maxdev_chk = dev[chk_idx].max() / r if len(chk_idx) else dev.max() / r
        maxdev_all = dev.max() / r
        rr = arm.rho_over_r(r)
        report.append(dict(deg=deg, maxdev_chk=maxdev_chk, maxdev_all=maxdev_all,
                           min_rho_r=float(np.min(rr))))
        if maxdev_chk < tol_over_r:
            return arm, dict(deg=deg, L=L, tol_met=True, maxdev_chk=maxdev_chk,
                             maxdev_all=maxdev_all, min_rho_r=float(np.min(rr)),
                             fit_idx=fit_idx, chk_idx=chk_idx, report=report)
    # did not meet tol: return best (deg_max) and flag to table
    return arm, dict(deg=deg_max, L=L, tol_met=False, maxdev_chk=maxdev_chk,
                     maxdev_all=maxdev_all, min_rho_r=float(np.min(rr)),
                     fit_idx=fit_idx, chk_idx=chk_idx, report=report)
