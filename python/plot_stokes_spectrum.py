#!/usr/bin/env python3
"""
Stokes slender-body BIE spectrum study -- Figure 7 / Figure 8 analogs of Malhotra & Barnett 2024 (CSBQ).

Inputs: raw dense operator matrices dumped by two drivers into a directory (default vis/stokes-spectrum):
    <case>s   = pure single-layer  S
    <case>d   = double-layer        -- for a `torus` case this is (1/2 I + D) (exterior jump baked in by
                CSBQ test/spectrum.cpp's CombinedFieldOp); for every other (our) case it is pure D.
Each file is sctl::Matrix::Write format: 2x uint64 dims (n0,n1) then n0*n1 float64, ROW-major.

`<case>` must contain `eps<value>` (e.g. torus_eps1e-2, cap_eps1e-2); the slenderness eps sets the
predicted optimum  eta* = 1/(2 eps log(1/eps)).

Combined-field operators built here (eta>0 the admixture, I the identity on the vector DOF):
    S,   c/2 I + D,   K(eta) = c/2 I + D + eta S .
The 1/2-jump sign c = +/-1 is fixed by the mesh's OUTWARD-normal convention (D's principal value and the
jump must reinforce for a well-conditioned CFIE -- the paper's Eq.33 exterior form is +1/2 I + D + eta S);
it is auto-detected per case (empirically +1 for both the CSBQ torus and our capsule). This is the same
combined-field operator our flow solves invert (up to that orientation sign). K(eta=0) = c/2 I + D is the
ill-conditioned double layer that S regularizes.

Outputs (in <dir>/plots):
    <geom>_fig7_svd.png   rows {S, c/2 I+D, K(eta*)} x cols eps : sorted singular values (Fig-7 analog)
    <geom>_fig7_eig.png   same grid : eigenvalues in the complex plane
    stokes_cond_scaling.png          kappa(K(eta)) vs eta per eps (Fig-8 analog) + measured-optimum scaling
    scaling_summary.txt              argmin_eta kappa vs eta* = 1/(2 eps log 1/eps), per case

Usage:  python python/plot_stokes_spectrum.py [dir=vis/stokes-spectrum]
"""
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------
def read_mat(fname):
    """sctl::Matrix::Write reader: 2x uint64 dims then row-major float64 -> (n0,n1) ndarray."""
    with open(fname, "rb") as f:
        dims = np.fromfile(f, dtype=np.uint64, count=2)
        n0, n1 = int(dims[0]), int(dims[1])
        data = np.fromfile(f, dtype=np.float64, count=n0 * n1)
    if data.size != n0 * n1:
        raise ValueError(f"{fname}: expected {n0*n1} doubles, got {data.size}")
    return data.reshape(n0, n1)   # sctl data_ptr is row-major (M[i][j]) -> C-order reshape


def parse_eps(case):
    m = re.search(r"eps([0-9eE.+-]+)", case)
    if not m:
        raise ValueError(f"cannot parse eps from case name '{case}'")
    return float(m.group(1))


def geom_of(case):
    """geometry family = leading token before _eps (torus, cap, ...)."""
    return case.split("_eps")[0]


def eta_star(eps):
    return 1.0 / (2.0 * eps * np.log(1.0 / eps))


def load_case(dirpath, case):
    """Return dict with pure S, pure D, eps for one case, reconciling the torus half-jump convention."""
    S = read_mat(os.path.join(dirpath, case + "s"))
    Dfile = read_mat(os.path.join(dirpath, case + "d"))
    N = S.shape[0]
    assert S.shape == (N, N) and Dfile.shape == (N, N), f"{case}: non-square/mismatched matrices"
    # Torus 'd' = 1/2 I + D (exterior jump baked in); recover pure D. Our drivers dump pure D already.
    D = Dfile - 0.5 * np.eye(N) if "torus" in case else Dfile
    d = dict(case=case, geom=geom_of(case), eps=parse_eps(case), N=N, S=S, D=D)
    d["c"] = jump_sign(d)   # outward-normal 1/2-jump sign for the well-conditioned CFIE
    return d


# ----------------------------------------------------------------------------------------------------
# Combined-field operator  K(eta) = c/2 I + D + eta S ,  where the sign c = +/-1 of the 1/2-jump is set
# by the mesh's OUTWARD-normal convention (the double-layer principal value D and the jump must reinforce
# for the CFIE to be well-conditioned -- the paper's exterior form is +1/2 I + D + eta S). We auto-detect
# c per case (the sign giving the smaller condition number at eta*), so a mesh with either orientation is
# handled; empirically c = +1 for both the CSBQ torus and our capsule. This is the SAME operator the flow
# solves invert (up to that orientation sign); its eta* = 1/(2 eps log 1/eps) is the Stokes scaling law.
def jump_sign(d):
    es = eta_star(d["eps"])
    I = np.eye(d["N"])
    kp = cond_trim(svdvals(0.5 * I + d["D"] + es * d["S"]))
    km = cond_trim(svdvals(-0.5 * I + d["D"] + es * d["S"]))
    return 1.0 if kp <= km else -1.0


def K_op(d, eta):
    """Combined field  K(eta) = c/2 I + D + eta S  (c = d['c'], the outward-normal jump sign)."""
    return d["c"] * 0.5 * np.eye(d["N"]) + d["D"] + eta * d["S"]


def D_op(d):
    """Double-layer combined operator at eta=0: c/2 I + D (ill-conditioned; S regularizes it)."""
    return d["c"] * 0.5 * np.eye(d["N"]) + d["D"]


def svdvals(M):
    return np.linalg.svd(M, compute_uv=False)


def cond_trim(sv, rel=1e-12):
    """kappa over singular values above rel*smax (drop numerical-null modes, as CSBQ spectrum.cpp does)."""
    smax = sv[0]
    keep = sv[sv > rel * smax]
    return smax / keep[-1]


# ----------------------------------------------------------------------------------------------------
def fig7(dirpath, geom, cases, outdir):
    """rows {S, D_int, K(eta*)} x cols eps : singular values (svd) and eigenvalues (eig)."""
    cases = sorted(cases, key=lambda d: -d["eps"])   # large eps -> small eps, left to right
    ncol = len(cases)
    op_names = ["S (single layer)", "c/2 I + D  (double-layer combined)",
                "K(eta*) = c/2 I + D + eta* S  (combined field)"]

    def build_ops(d):
        return [d["S"], D_op(d), K_op(d, eta_star(d["eps"]))]

    # ---- singular values ----
    fig, ax = plt.subplots(3, ncol, figsize=(3.6 * ncol, 9.2), squeeze=False)
    for j, d in enumerate(cases):
        ops = build_ops(d)
        for i, M in enumerate(ops):
            sv = svdvals(M)
            a = ax[i][j]
            a.semilogy(np.arange(sv.size), sv, ".", ms=3)
            a.set_title(f"{op_names[i]}\neps={d['eps']:g}  kappa={cond_trim(sv):.2e}", fontsize=9)
            a.grid(True, which="both", alpha=0.25)
            if j == 0:
                a.set_ylabel("singular value")
            if i == 2:
                a.set_xlabel("index (descending)")
    fig.suptitle(f"Stokes BIE singular-value spectra -- {geom}  (Fig. 7 analog, CFIE convention)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, f"{geom}_fig7_svd.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)

    # ---- eigenvalues (complex plane) ----
    fig, ax = plt.subplots(3, ncol, figsize=(3.6 * ncol, 9.2), squeeze=False)
    for j, d in enumerate(cases):
        ops = build_ops(d)
        for i, M in enumerate(ops):
            ev = np.linalg.eigvals(M)
            a = ax[i][j]
            a.plot(ev.real, ev.imag, ".", ms=3)
            a.axhline(0, color="k", lw=0.4); a.axvline(0, color="k", lw=0.4)
            a.set_title(f"{op_names[i]}\neps={d['eps']:g}", fontsize=9)
            a.grid(True, alpha=0.25)
            if j == 0:
                a.set_ylabel("Im")
            if i == 2:
                a.set_xlabel("Re")
    fig.suptitle(f"Stokes BIE eigenvalues -- {geom}  (Fig. 7 analog, CFIE convention)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, f"{geom}_fig7_eig.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


def fig8(all_cases, outdir, summary_lines):
    """kappa(K(eta)) vs eta per (geom, eps) with the eta* marker + measured-optimum scaling law."""
    geoms = sorted({d["geom"] for d in all_cases})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    color = {}
    for d in sorted(all_cases, key=lambda d: (d["geom"], -d["eps"])):
        es = eta_star(d["eps"])
        etas = np.logspace(np.log10(es) - 2.5, np.log10(es) + 2.5, 25)
        # precompute c/2 I + D once; K(eta) = base + eta S
        base = D_op(d)
        kap = np.array([cond_trim(svdvals(base + eta * d["S"])) for eta in etas])
        imin = int(np.argmin(kap))
        eta_opt = etas[imin]
        key = (d["geom"], d["eps"])
        ls = {"torus": "-", "cap": "--"}.get(d["geom"], ":")
        c = color.setdefault(d["eps"], f"C{len(color)}")
        ax[0].loglog(etas, kap, ls, color=c, label=f"{d['geom']} eps={d['eps']:g}")
        ax[0].axvline(es, color=c, ls=":", lw=0.8)
        summary_lines.append(
            f"{d['geom']:8s} eps={d['eps']:.1e}  c={int(d['c']):+d}  eta*=1/(2eps log 1/eps)={es:.3e}  "
            f"argmin_eta kappa={eta_opt:.3e}  ratio(meas/pred)={eta_opt/es:.2f}  "
            f"kappa_min={kap[imin]:.2e}  kappa(eta=0)={cond_trim(svdvals(base)):.2e}")
        # scaling scatter (right panel): measured optimum vs eps
        ax[1].loglog(d["eps"], eta_opt, {"torus": "o", "cap": "s"}.get(d["geom"], "^"),
                     color=c, ms=8, label=f"{d['geom']} eps={d['eps']:g}")

    ax[0].set_xlabel("eta"); ax[0].set_ylabel("kappa(K(eta))")
    ax[0].set_title("Condition number of Stokes combined field K = c/2 I + D + eta S\n"
                    "(dotted verticals = predicted eta* = 1/(2 eps log 1/eps))", fontsize=10)
    ax[0].grid(True, which="both", alpha=0.25); ax[0].legend(fontsize=7, ncol=2)

    # reference law curve on the right panel
    epss = np.array(sorted({d["eps"] for d in all_cases}))
    ee = np.logspace(np.log10(epss.min()) - 0.3, np.log10(epss.max()) + 0.3, 50)
    ax[1].loglog(ee, eta_star(ee), "k-", lw=1, label="eta* = 1/(2 eps log 1/eps)")
    ax[1].set_xlabel("eps"); ax[1].set_ylabel("argmin_eta kappa (measured)")
    ax[1].set_title("Empirical Stokes scaling law: optimal admixture vs slenderness", fontsize=10)
    ax[1].grid(True, which="both", alpha=0.25); ax[1].legend(fontsize=7)

    fig.tight_layout()
    p = os.path.join(outdir, "stokes_cond_scaling.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


# ----------------------------------------------------------------------------------------------------
def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else "vis/stokes-spectrum"
    outdir = os.path.join(dirpath, "plots")
    os.makedirs(outdir, exist_ok=True)

    # discover cases: every <case>s that has a matching <case>d
    cases = []
    for sfile in sorted(glob.glob(os.path.join(dirpath, "*s"))):
        case = os.path.basename(sfile)[:-1]
        if os.path.exists(os.path.join(dirpath, case + "d")):
            cases.append(case)
    if not cases:
        print(f"no <case>s/<case>d matrix pairs found in {dirpath}"); sys.exit(1)
    print("cases:", cases)

    loaded = [load_case(dirpath, c) for c in cases]
    for d in loaded:
        print(f"  {d['case']}: N={d['N']} eps={d['eps']:g} eta*={eta_star(d['eps']):.3e}")

    # Fig-7 analog per geometry
    by_geom = {}
    for d in loaded:
        by_geom.setdefault(d["geom"], []).append(d)
    for geom, cs in by_geom.items():
        fig7(dirpath, geom, cs, outdir)

    # Fig-8 analog + scaling law across everything
    summary = ["# Stokes combined-field conditioning: measured optimum vs predicted eta*=1/(2 eps log 1/eps)"]
    fig8(loaded, outdir, summary)
    sp = os.path.join(outdir, "scaling_summary.txt")
    with open(sp, "w") as f:
        f.write("\n".join(summary) + "\n")
    print("wrote", sp)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
