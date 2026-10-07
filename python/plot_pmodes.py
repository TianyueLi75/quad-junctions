#!/usr/bin/env python3
"""
Near-null MODE visualization for the completed interior-Stokes capsule operator.

Reads the compact near-null basis dumped by `capsule-robin-bie pspectrum` with QJ_PSPECTRUM_MODES=k
(see include/quad_junctions/operator_spectrum.hpp::write_nearnull_basis). For each aspect-ratio case
(ar17 / ar167 / ar1670) it renders, for the up-to-5 smallest-|lambda| modes:

  * SURFACE  -- each mode's density magnitude painted on the WHOLE capsule (caps + shaft), unrolled to
                (axial s, azimuth theta) with axial=0 at the A-cap tip,
                plus the ONE rotation-invariant aggregate  rho(x) = sqrt(sum_i ||v_i(x)||^2)  (the only
                basis-independent single picture of the near-null subspace -- a signed sum sum_i v_i is
                basis-arbitrary and prone to cancellation, so it is deliberately NOT produced).
  * FLOW     -- the interior velocity each mode drives, on the axial-e1 midplane over the WHOLE capsule
                (both caps + shaft), read straight from the C++ StokesBIO evaluation (correct near-quadrature
                -- a naive python far-field sum is invalid inside a slender lumen, where dist_far ~ panel
                length >> tube radius so no interior point is far from the wall). Only points strictly inside
                the capsule wall (minus an inward buffer, larger at the caps) are evaluated; the grid stays
                regular so the exterior renders blank -> a filled capsule cross-section. Plus the invariant
                flow-energy aggregate sqrt(sum_i ||u_i(x)||^2).
  * MONTAGE  -- ar17 | ar167 | ar1670 side by side for mode-0 surface, surface energy, flow energy: the
                direct "how the near-null modes change with aspect ratio" figure.

All heavy compute (eigendecomposition AND the interior flow field) is done once in C++ and dumped; this
script only reads the compact bundle and draws -- changing a plot never re-runs the operator.

Usage:  /mnt/home/tli10/venv/bin/python python/plot_pmodes.py [base_dir=vis/capsule-pspectrum/comp-aspect]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Only the LOW-Nelem aspect runs (spectra stable); skip ar17-smallr (smaller same-aspect) and *-n60.
CASES = ["ar17", "ar167", "ar1670"]
PREFIX = "straight-comp-int-none.modes"     # QJ_COMPLETION=1, interior, variant none

# Near-null cutoff (matches plot_pspectrum.py's nearzero-tail THRESH). The count of |lambda|<NULL_THRESH
# grows with aspect ratio; the surface/flow ENERGY aggregates sum over exactly that many modes. The number
# of per-mode eigenvectors VISUALIZED per aspect is held fixed at N_VIS.
NULL_THRESH = 1e-2
N_VIS = 5


def read_mat(fname):
    """sctl::Matrix::Write reader: 2x uint64 dims then row-major float64 -> (n0,n1) ndarray."""
    with open(fname, "rb") as f:
        dims = np.fromfile(f, dtype=np.uint64, count=2)
        n0, n1 = int(dims[0]), int(dims[1])
        data = np.fromfile(f, dtype=np.float64, count=n0 * n1)
    if data.size != n0 * n1:
        raise ValueError(f"{fname}: expected {n0*n1} doubles, got {data.size}")
    return data.reshape(n0, n1)


def read_meta(path):
    m = {}
    for line in open(path):
        t = line.split()
        if not t:
            continue
        m[t[0]] = [float(x) for x in t[1:]] if len(t) > 2 else float(t[1])
    return m


def load_case(cdir):
    """Load one case's near-null basis bundle. Returns a dict or None if absent."""
    p = os.path.join(cdir, PREFIX)
    if not os.path.exists(p + ".meta"):
        return None
    meta = read_meta(p + ".meta")
    N, Nb, Ns, k = int(meta["N"]), int(meta["Nb"]), int(meta["Ns"]), int(meta["k"])
    nnode = Nb + Ns
    d = dict(meta=meta, N=N, Nb=Nb, Ns=Ns, k=k, nnode=nnode,
             SL=float(meta["SL_scal"]), DL=float(meta["DL_scal"]),
             axis=np.array(meta["axis"]), R=float(meta["R_shaft"]), L=float(meta["length"]),
             # axial coord of the A-cap tip; subtract so axial=0 sits at the tip (0 for old bundles).
             s0=float(meta.get("flow_s0", 0.0)))
    d["X0"] = read_mat(p + "-X0.mat").reshape(nnode, 3)
    d["Xn0"] = read_mat(p + "-Xn0.mat").reshape(nnode, 3)
    d["mre"] = read_mat(p + "-modes-re.mat").reshape(k, nnode, 3)   # per-node density, real part
    d["mim"] = read_mat(p + "-modes-im.mat").reshape(k, nnode, 3)
    ev = np.loadtxt(p + ".eigval.csv", delimiter=",", comments="#", ndmin=2)
    d["eig"] = ev[:, 1] + 1j * ev[:, 2]
    # Actual near-null count: prefer the value C++ wrote to meta; else count the full spectrum eig.csv
    # (|lambda|<NULL_THRESH). The energy aggregates sum over this many modes; visualization uses n_vis.
    if "n_null" in meta:
        n_null = int(meta["n_null"])
    else:
        eig_all = os.path.join(cdir, PREFIX[:-len(".modes")] + ".eig.csv")
        n_null = k
        if os.path.exists(eig_all):
            ea = np.loadtxt(eig_all, delimiter=",", comments="#", ndmin=2)
            n_null = int((np.hypot(ea[:, 1], ea[:, 2]) < NULL_THRESH).sum())
    if n_null > k:
        print(f"    [warn] {cdir}: near-null count {n_null} > dumped modes {k}; "
              f"energy summed over {k} only -- rerun so C++ auto-grows k to the near-null count.")
    d["n_null"] = max(1, min(n_null, k))          # >=1, and can't exceed the modes actually dumped
    d["n_vis"] = min(N_VIS, k)
    # Interior flow grid, evaluated in C++ (correct near-quadrature). Grid order [ix][ir].
    nx, nr = int(meta["flow_nx"]), int(meta["flow_nr"])
    d["nx"], d["nr"] = nx, nr
    d["flowX"] = read_mat(p + "-flowX.mat").reshape(nx, nr, 3)
    d["flowU"] = (read_mat(p + "-flowU-re.mat") + 1j * read_mat(p + "-flowU-im.mat")).reshape(k, nx, nr, 3)
    return d


# ---- midplane frame: e0 = axis, e1 = the perpendicular the C++ flow grid was built along --------------
def midplane_frame(axis):
    e0 = axis / np.linalg.norm(axis)
    t0 = np.zeros(3); t0[int(np.argmin(np.abs(e0)))] = 1.0
    e1 = np.cross(t0, e0); e1 /= np.linalg.norm(e1)
    return e0, e1


# ---- surface parametrization: axis-aligned (s = X.axis, theta about axis) -----------------------------
def unroll(X, axis, L):
    e0 = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0]) if abs(e0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(e0, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    s = X @ e0
    a = X @ e1
    b = X @ e2
    theta = np.arctan2(b, a)
    return s, theta


def mag(v):   # per-node velocity magnitude ||v|| = sqrt(sum_i |v_i|^2), v shape (nnode,3), possibly complex
    return np.sqrt((np.abs(v) ** 2).sum(-1))


# ---- plots --------------------------------------------------------------------------------------------
def plot_surface_modes(d, cdir, case):
    X = d["X0"]                                  # ALL nodes: caps (0:Nb) + shaft (Nb:) -- whole capsule
    s, th = unroll(X, d["axis"], d["L"]); s = s - d["s0"]   # axial=0 at the A-cap tip
    k = d["n_vis"]                               # visualize the N_VIS smallest-|lambda| modes only
    fig, axes = plt.subplots(k, 1, figsize=(9, 2.2 * k), squeeze=False)
    for m in range(k):
        vm = mag(d["mre"][m] + 1j * d["mim"][m])
        ax = axes[m][0]
        sc = ax.scatter(s, th, c=vm, s=8, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="|v|")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(f"mode {m}  |$\\lambda$|={abs(d['eig'][m]):.2e}")
    axes[-1][0].set_xlabel("axial s")
    fig.suptitle(f"Near-null density modes on capsule (caps+shaft)  --  {case} (aspect {d['L']/d['R']:.0f})")
    fig.tight_layout()
    out = os.path.join(cdir, f"pmodes-{case}-surface.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def surface_energy(d):
    """rho(x) = sqrt(sum_m ||v_m(x)||^2) per node over the n_null NEAR-NULL modes -- the rotation-invariant
    near-null subspace picture (summed over the actual near-null count, not the visualized N_VIS)."""
    e = np.zeros(d["nnode"])
    for m in range(d["n_null"]):
        vm = d["mre"][m] + 1j * d["mim"][m]
        e += (np.abs(vm) ** 2).sum(-1)
    return np.sqrt(e)


def plot_surface_energy(d, cdir, case):
    X = d["X0"]                                   # whole capsule (caps + shaft)
    s, th = unroll(X, d["axis"], d["L"]); s = s - d["s0"]
    rho = surface_energy(d)
    fig, ax = plt.subplots(figsize=(9, 3))
    sc = ax.scatter(s, th, c=rho, s=10, cmap="magma")
    fig.colorbar(sc, ax=ax, label=r"$\rho=\sqrt{\sum_i\|v_i\|^2}$")
    ax.set_xlabel("axial s"); ax.set_ylabel(r"$\theta$")
    ax.set_title(f"Near-null subspace energy on capsule ({d['n_null']} near-null modes)  --  "
                 f"{case} (aspect {d['L']/d['R']:.0f})")
    fig.tight_layout()
    out = os.path.join(cdir, f"pmodes-{case}-surface-energy.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def flow_axes(d):
    """(S, RR) meshes for the C++ flow grid, in (axial s, radial r) coordinates. Grid order [ix][ir]."""
    e0, e1 = midplane_frame(d["axis"])
    X = d["flowX"]                       # (nx, nr, 3) -- full regular grid over the whole capsule
    S = X @ e0 - d["s0"]                 # axial coordinate (nx, nr), 0 = A-cap tip
    RR = X @ e1                          # signed radial coordinate on the e1 line (nx, nr)
    return S, RR, e0, e1


def plot_flow_modes(d, cdir, case):
    """Plot the C++-evaluated interior flow of the n_vis smallest modes on the midplane, + the invariant
    flow energy summed over the n_null NEAR-NULL modes."""
    S, RR, e0, e1 = flow_axes(d)
    k = d["n_vis"]                       # per-mode panels: visualize the N_VIS smallest only
    fig, axes = plt.subplots(k, 1, figsize=(9, 2.4 * k), squeeze=False)
    for m in range(k):
        U = d["flowU"][m].real          # (nx, nr, 3) -- real part of the represented field
        us = U @ e0; ur = U @ e1
        spd = np.sqrt(us ** 2 + ur ** 2)
        ax = axes[m][0]
        pc = ax.pcolormesh(S, RR, spd, cmap="viridis", shading="auto")
        fig.colorbar(pc, ax=ax, label="|u|")
        st = max(1, S.shape[0] // 30)
        ax.quiver(S[::st, ::2], RR[::st, ::2], us[::st, ::2], ur[::st, ::2],
                  color="w", scale_units="width", width=0.0018)
        ax.set_ylabel("r"); ax.set_title(f"flow of mode {m}  |$\\lambda$|={abs(d['eig'][m]):.2e}")
    axes[-1][0].set_xlabel("axial s")
    fig.suptitle(f"Interior flow driven by each near-null mode  --  {case} (aspect {d['L']/d['R']:.0f})")
    fig.tight_layout()
    out = os.path.join(cdir, f"pmodes-{case}-flow.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    # invariant flow energy sqrt(sum_i ||u_i||^2), summed over the ACTUAL near-null modes (n_null).
    Uacc = np.zeros(S.shape)
    for m in range(d["n_null"]):
        Uacc = Uacc + (np.abs(d["flowU"][m]) ** 2).sum(-1)   # invariant: sum over comps AND near-null modes
    en = np.sqrt(Uacc)
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    pc = ax2.pcolormesh(S, RR, en, cmap="magma", shading="auto")
    fig2.colorbar(pc, ax=ax2, label=r"$\sqrt{\sum_i\|u_i\|^2}$")
    ax2.set_xlabel("axial s"); ax2.set_ylabel("r")
    ax2.set_title(f"Near-null flow energy ({d['n_null']} near-null modes)  --  "
                  f"{case} (aspect {d['L']/d['R']:.0f})")
    fig2.tight_layout()
    out2 = os.path.join(cdir, f"pmodes-{case}-flow-energy.png")
    fig2.savefig(out2, dpi=130); plt.close(fig2)
    return out, out2, en, S, RR


def main():
    global PREFIX
    # usage: plot_pmodes.py [base_dir] [geom=straight-comp]  -- same call signature as plot_pmodes_dl.py
    #   geom is the operator gtag (straight-comp = dl+sl+completion); it overrides the modes-bundle prefix.
    base = sys.argv[1] if len(sys.argv) > 1 else "vis/capsule-pspectrum/comp-aspect"
    geom = sys.argv[2] if len(sys.argv) > 2 else "straight-comp"
    PREFIX = geom + "-int-none.modes"
    loaded = []
    for case in CASES:
        cdir = os.path.join(base, case)
        d = load_case(cdir)
        if d is None:
            print(f"  [skip] {case}: no {PREFIX}.meta in {cdir} (run pspectrum with QJ_PSPECTRUM_MODES=5)")
            continue
        # SURFACE plots need only X0 + the eigenvectors (no kernel eval). FLOW uses the C++-evaluated
        # midplane grid (correct near-quadrature; a naive python sum is invalid inside a slender lumen).
        print(f"  [{case}] N={d['N']} k={d['k']} aspect={d['L']/d['R']:.0f}")
        print("    surface :", plot_surface_modes(d, cdir, case))
        print("    surf-E  :", plot_surface_energy(d, cdir, case))
        fo, fe, en, S, RR = plot_flow_modes(d, cdir, case)
        print("    flow    :", fo)
        print("    flow-E  :", fe)
        loaded.append((case, d, en, S, RR))

    if len(loaded) >= 2:
        have_flow = any(en is not None for _, _, en, _, _ in loaded)
        nrow = 3 if have_flow else 2
        fig, axes = plt.subplots(nrow, len(loaded), figsize=(4.2 * len(loaded), 3 * nrow), squeeze=False)
        # Each panel autoscales its own colorbar (no fixed/shared ranges) -- per-geometry contrast only.
        for j, (case, d, en, S, RR) in enumerate(loaded):
            X = d["X0"]; s, th = unroll(X, d["axis"], d["L"]); s = s - d["s0"]   # whole capsule
            m0 = mag(d["mre"][0] + 1j * d["mim"][0])
            rho = surface_energy(d)
            a0, a1 = axes[0][j], axes[1][j]
            sc = a0.scatter(s, th, c=m0, s=6, cmap="viridis"); fig.colorbar(sc, ax=a0)
            a0.set_title(f"{case} (a={d['L']/d['R']:.0f})\nmode 0 |v|"); a0.set_ylabel(r"$\theta$")
            sc = a1.scatter(s, th, c=rho, s=6, cmap="magma"); fig.colorbar(sc, ax=a1)
            a1.set_title(r"surface energy $\rho$"); a1.set_ylabel(r"$\theta$"); a1.set_xlabel("axial s")
            if have_flow:
                a2 = axes[2][j]
                if en is not None:
                    pc = a2.pcolormesh(S, RR, en, cmap="magma", shading="auto"); fig.colorbar(pc, ax=a2)
                else:
                    a2.text(0.5, 0.5, "flow n/a\n(near-field)", ha="center", va="center")
                a2.set_title("flow energy"); a2.set_xlabel("axial s"); a2.set_ylabel("r")
        fig.suptitle("Near-null modes vs aspect ratio")
        fig.tight_layout()
        out = os.path.join(base, "pmodes-aspect-montage.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print("  montage :", out)

        # ---- Companion montage: surface energy + flow energy, each divided by the near-null count k=n_null
        #      (per-mode-averaged aggregate, so geometries with very different near-null counts compare
        #      fairly). Same data, autoscaled per panel. Written as <montage>_normalized.png. ----
        nrow2 = 2 if have_flow else 1
        fig2, ax2 = plt.subplots(nrow2, len(loaded), figsize=(4.2 * len(loaded), 3 * nrow2), squeeze=False)
        for j, (case, d, en, S, RR) in enumerate(loaded):
            kk = d["n_null"]
            X = d["X0"]; s, th = unroll(X, d["axis"], d["L"]); s = s - d["s0"]
            rho = surface_energy(d) / kk
            b0 = ax2[0][j]
            sc = b0.scatter(s, th, c=rho, s=6, cmap="magma"); fig2.colorbar(sc, ax=b0)
            b0.set_title(f"{case} (a={d['L']/d['R']:.0f}, k={kk})\n" r"surface energy $\rho/k$")
            b0.set_ylabel(r"$\theta$"); b0.set_xlabel("axial s")
            if have_flow:
                b1 = ax2[1][j]
                if en is not None:
                    pc = b1.pcolormesh(S, RR, en / kk, cmap="magma", shading="auto"); fig2.colorbar(pc, ax=b1)
                else:
                    b1.text(0.5, 0.5, "flow n/a\n(near-field)", ha="center", va="center")
                b1.set_title("flow energy /k"); b1.set_xlabel("axial s"); b1.set_ylabel("r")
        fig2.suptitle("Near-null energy (1/k normalized) vs aspect ratio")
        fig2.tight_layout()
        out2 = os.path.join(base, "pmodes-aspect-montage_normalized.png")
        fig2.savefig(out2, dpi=130); plt.close(fig2)
        print("  montage :", out2)


if __name__ == "__main__":
    main()
