#!/usr/bin/env python3
"""
plot_pspectrum.py -- plot the preconditioned singular-value spectra written by
src/capsule-robin-bie.cpp `pspectrum` mode.

For each geometry (straight|spiral) it reads the 8 CSVs
  vis/capsule-pspectrum/<geom>-<prob>-<variant>.csv,  prob in {int,ext},
  variant in {none, seam, armonly, caparm},
and makes a 2-panel figure (interior | exterior), each overlaying the four
sorted singular-value curves sigma_i vs index i (log-y). The point of the plot:

  - none    : the un-preconditioned operator A (one 2nd-kind cluster + a near-null
              tail on the interior).
  - caparm  : whole-surface block-Jacobi -> singular values re-clustered near 1.
  - armonly : arm blocks rescaled, caps left at identity -> the spectrum SPLITS
              into two clusters (partial-rescale mismatch) -> harder for GMRES.
  - seam    : cap<->arm-seam coupled block.

Usage:  python3 python/plot_pspectrum.py [straight|spiral]   (default: straight)
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "vis/capsule-pspectrum"
VARIANTS = ["none", "caparm", "seam", "armonly", "caparmneg", "armonlyneg"]
COLORS = {"none": "0.35", "caparm": "tab:green", "seam": "tab:blue", "armonly": "tab:red",
          "caparmneg": "tab:purple", "armonlyneg": "tab:orange"}
LABELS = {"none": "none (A)", "caparm": "cap+arm (whole block-Jacobi, ->+I)",
          "seam": "seam (cap+ring coupled)", "armonly": "arm-only (caps=I, ->+I)",
          "caparmneg": "cap+arm neg (->-I)", "armonlyneg": "arm-only neg (arm->-I, caps=-1/2)"}


def read_csv(path):
    cond = smax = smin = float("nan")
    with open(path) as f:
        for line in f:
            if line.startswith("# cond="):
                for tok in line[1:].split():
                    k, _, v = tok.partition("=")
                    if k == "cond": cond = float(v)
                    elif k == "smax": smax = float(v)
                    elif k == "smin": smin = float(v)
    data = np.loadtxt(path, delimiter=",", comments="#")
    sigma = data[:, 1]
    return sigma, cond, smax, smin


def read_eig(path):
    """Read a .eig.csv (index,re,im) + its # min_abs_lambda / dep_normality_F / froF header.
    Also returns the (0,1)-normalized departure from normality dep_F/||A||_F = sqrt(1 - sum|lambda|^2/
    ||A||_F^2): 0 = normal (eigenvalues carry all the Frobenius mass, so the eig scatter fully predicts
    GMRES), ->1 = maximally non-normal (trust the scatter less)."""
    minmag = depF = froF = float("nan")
    with open(path) as f:
        for line in f:
            if line.startswith("# min_abs_lambda="):
                for tok in line[1:].split():
                    k, _, v = tok.partition("=")
                    if k == "min_abs_lambda": minmag = float(v)
                    elif k == "dep_normality_F": depF = float(v)
                    elif k == "froF": froF = float(v)
    data = np.loadtxt(path, delimiter=",", comments="#")
    depF_norm = depF / froF if (froF and np.isfinite(froF) and froF != 0) else float("nan")
    return data[:, 1], data[:, 2], minmag, depF, depF_norm  # re, im, min|lambda|, dep_F, dep_F/||A||_F


def plot_geom(geom):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for ax, prob, title in zip(axes, ["int", "ext"],
                               ["Interior  (SL=-1, DL=+1, jump=-1/2)",
                                "Exterior  (SL=+1, DL=+1, jump=+1/2)"]):
        any_curve = False
        for v in VARIANTS:
            path = os.path.join(DIR, f"{geom}-{prob}-{v}.csv")
            if not os.path.exists(path):
                continue
            sigma, cond, smax, smin = read_csv(path)
            idx = np.arange(sigma.size)
            ax.semilogy(idx, sigma, color=COLORS[v], lw=1.6,
                        label=f"{LABELS[v]}   (κ={cond:.1f})")
            any_curve = True
        if not any_curve:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("index (sorted descending)")
        ax.set_ylabel(r"singular value $\sigma_i$ of $M^{-1}A$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Preconditioned spectra  --  {geom} capsule "
                 f"(none / cap+arm / seam / arm-only)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(DIR, f"pspectrum-{geom}.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # Companion: zoom on the small-sigma tail (conditioning end) on a rank-fraction x-axis.
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, prob, title in zip(axes2, ["int", "ext"], ["Interior tail", "Exterior tail"]):
        for v in VARIANTS:
            path = os.path.join(DIR, f"{geom}-{prob}-{v}.csv")
            if not os.path.exists(path):
                continue
            sigma, cond, smax, smin = read_csv(path)
            n = sigma.size
            k = max(1, n // 5)                      # smallest 20%
            xs = np.arange(n - k, n) / n
            ax.semilogy(xs, sigma[-k:], color=COLORS[v], lw=1.6, label=LABELS[v])
        ax.set_title(title)
        ax.set_xlabel("index / N  (tail)")
        ax.set_ylabel(r"$\sigma_i$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
    fig2.suptitle(f"Small-singular-value tail  --  {geom} capsule", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(DIR, f"pspectrum-{geom}-tail.png")
    fig2.savefig(out2, dpi=130)
    print("wrote", out2)


def plot_eig(geom):
    """Complex-plane eigenvalue scatter of M^{-1}A -- the diagnostic that actually tracks GMRES.
    Singular values (plot_geom) fix the condition number but NOT the iteration count; for these
    near-normal 2nd-kind operators GMRES tracks the EIGENVALUE clustering. Expectation:
      none/caparm -> one tight cluster (fast GMRES);  armonly -> split/scattered (slow), even though
      its condition number is the lowest;  interior clusters carry one lambda ~ 0 (the flux null mode)."""
    paths = {(p, v): os.path.join(DIR, f"{geom}-{p}-{v}.eig.csv")
             for p in ("int", "ext") for v in VARIANTS}
    if not any(os.path.exists(p) for p in paths.values()):
        print("no .eig.csv files for", geom, "-- run pspectrum with QJ_SPECTRUM_EIG=1 (default)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, prob, title in zip(axes, ["int", "ext"],
                               ["Interior  (SL=-1, DL=+1, jump=-1/2)",
                                "Exterior  (SL=+1, DL=+1, jump=+1/2)"]):
        for v in VARIANTS:
            path = paths[(prob, v)]
            if not os.path.exists(path):
                continue
            re, im, minmag, depF, depFn = read_eig(path)
            ax.scatter(re, im, s=6, color=COLORS[v], alpha=0.55, edgecolors="none",
                       label=f"{LABELS[v]}   (min|λ|={minmag:.1e}, "
                             rf"dep$_F$/‖A‖$_F$={depFn:.3f})")
        ax.axvline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(r"Re $\lambda$")
        ax.set_ylabel(r"Im $\lambda$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"Eigenvalue spectrum of $M^{{-1}}A$  --  {geom} capsule "
                 f"(this, not the singular values, is what GMRES sees)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(DIR, f"pspectrum-{geom}-eig.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # Companion: zoom near the origin (the flux null mode + the tightest cluster) on the interior.
    fig2, ax = plt.subplots(1, 1, figsize=(6.8, 6.2))
    for v in VARIANTS:
        path = paths[("int", v)]
        if not os.path.exists(path):
            continue
        re, im, minmag, depF, depFn = read_eig(path)
        ax.scatter(re, im, s=8, color=COLORS[v], alpha=0.6, edgecolors="none", label=LABELS[v])
    ax.axvline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 0.6)
    ax.axvspan(-1e9, 0, color="0.9", zorder=0)   # shade Re<0 -- a variant with mass on BOTH sides straddles 0
    ax.set_title(f"Interior eigenvalues on the real axis  --  {geom} (does the cluster straddle 0?)")
    ax.set_xlabel(r"Re $\lambda$"); ax.set_ylabel(r"Im $\lambda$")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="upper right")
    fig2.tight_layout()
    out2 = os.path.join(DIR, f"pspectrum-{geom}-eig-zoom.png")
    fig2.savefig(out2, dpi=130)
    print("wrote", out2)


def plot_eig_completed(geom):
    """SEPARATE eigenvalue scatter for the NET-FLUX-COMPLETED interior operator (QJ_COMPLETION=1),
    interior only, straight only, {none, caparmneg} = {off, block-Jacobi-flipped} only. Its CSVs are
    written by the driver under the "<geom>-comp" gtag so they never clobber the plain-operator run.
    The point of THIS plot: the plain interior operator carries one isolated lambda ~ 0 (the flux null
    mode); the rank-1 completion A' = A + (w/area) n (x) ∮n·(.) dA moves that eigenvalue off the origin
    to O(1), so the completed 'off' scatter should show NO near-zero eigenvalue, and the block-Jacobi-
    flipped variant a single clean cluster. (Compatible-RHS GMRES is nonetheless unchanged -- see the
    iteration study -- because the mode is never excited; this plot shows the mode is genuinely removed
    from the spectrum, which is what the completion is for.)"""
    ctag = f"{geom}-comp"
    variants = ["none", "caparmneg"]
    paths = {v: os.path.join(DIR, f"{ctag}-int-{v}.eig.csv") for v in variants}
    if not any(os.path.exists(p) for p in paths.values()):
        print(f"no completed-operator .eig.csv for {geom} -- run: "
              f"QJ_COMPLETION=1 ... capsule-robin-bie pspectrum (straight, interior)")
        return
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(12, 5.6))
    for v in variants:
        p = paths[v]
        if not os.path.exists(p):
            continue
        re, im, minmag, depF, depFn = read_eig(p)
        lab = f"{LABELS[v]}   (min|λ|={minmag:.1e}, dep$_F$/‖A‖$_F$={depFn:.3f})"
        ax.scatter(re, im, s=8, color=COLORS[v], alpha=0.6, edgecolors="none", label=lab)
        axz.scatter(re, im, s=10, color=COLORS[v], alpha=0.65, edgecolors="none", label=LABELS[v])
    for a in (ax, axz):
        a.axvline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        a.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        a.set_xlabel(r"Re $\lambda$"); a.set_ylabel(r"Im $\lambda$")
        a.grid(True, alpha=0.25)
    ax.set_title("Full spectrum")
    ax.legend(fontsize=8, loc="upper left")
    axz.set_xlim(-1.3, 1.3); axz.set_ylim(-0.6, 0.6)
    axz.axvspan(-1e9, 0, color="0.9", zorder=0)
    axz.set_title("Zoom on the origin (is the flux null mode gone?)")
    axz.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Eigenvalues of the NET-FLUX-COMPLETED interior operator  --  {geom} capsule "
                 f"(off vs block-Jacobi-flipped)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(DIR, f"pspectrum-{geom}-comp-eig.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def plot_eig_laplace(geom):
    """Interior LAPLACE combined-field operator (-1/2 I + D - S, Laplace3D scalar kernels), plotted BY
    ITSELF as a 1x2: LEFT the complex-plane eigenvalue scatter, RIGHT the near-zero |lambda| tail (the
    conditioning end). This is the scalar analog of the Stokes operator in plot_eig; kept separate
    because it has 1 DOF/node and its own file <geom>-int-laplace.eig.csv (written by capsule-robin-bie
    pspectrum). The tail shows how close the smallest eigenvalue sits to 0 (the interior-Dirichlet
    constant-mode near-null)."""
    path = os.path.join(DIR, f"{geom}-int-laplace.eig.csv")
    if not os.path.exists(path):
        print(f"no interior-Laplace eig csv for {geom} -- run pspectrum "
              f"(writes {os.path.basename(path)})")
        return
    re, im, minmag, depF, depFn = read_eig(path)
    mag = np.hypot(re, im)
    col = "tab:purple"

    fig, (axe, axt) = plt.subplots(1, 2, figsize=(12, 5.2))
    # LEFT: eigenvalue scatter in the complex plane
    axe.scatter(re, im, s=8, color=col, alpha=0.6, edgecolors="none",
                label=f"interior Laplace  (min|λ|={minmag:.1e}, "
                      rf"dep$_F$/‖A‖$_F$={depFn:.3f}, N={re.size})")
    axe.axvline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    axe.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    axe.set_title("Eigenvalues of  -1/2 I + D - S  (Laplace)")
    axe.set_xlabel(r"Re $\lambda$"); axe.set_ylabel(r"Im $\lambda$")
    axe.grid(True, alpha=0.25); axe.legend(fontsize=8, loc="upper left")

    # RIGHT: near-zero tail -- smallest |lambda| first (log-y)
    k = min(60, mag.size)
    tail_mag = np.sort(mag)[:k]
    axt.semilogy(np.arange(1, k + 1), tail_mag, marker=".", ms=5, lw=1.2, color=col,
                 label=f"smallest {k} |λ|  (min|λ|={minmag:.1e})")
    axt.set_title("Near-zero tail  (smallest |λ|)")
    axt.set_xlabel("rank from the smallest eigenvalue")
    axt.set_ylabel(r"$|\lambda|$")
    axt.grid(True, which="both", alpha=0.25); axt.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Interior Laplace combined-field operator  --  {geom} capsule "
                 f"(eigenvalues + near-zero tail)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(DIR, f"pspectrum-{geom}-laplace-eig.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------
# Near-null tail of the NET-FLUX-COMPLETED interior operator, across an aspect/resolution sweep.
# Three runs (each QJ_COMPLETION=1 pspectrum, straight, interior, variant "none"), written by
# scripts/submit-capsule-pspectrum-completion-aspect.sh into its own QJ_PSPECTRUM_OUTDIR subdir so the
# shared "straight-comp-int-none.eig.csv" name does not collide. For each run we take |lambda_i|, keep
# only the near-null modes |lambda_i| < THRESH, sort ascending, and plot on a log-y axis against the
# raw (UNNORMALIZED) index i -- so a run with more sub-threshold eigenvalues simply extends further
# right. This is the picture of whether the completion has pulled the flux null mode off zero, and how
# that near-null tail moves as the arm gets slenderer (a.r. 16.7 -> 167) and as the arm resolution is
# then restored (Nelem x10 at fixed Fourier/order).
THRESH = 1e-2
# Each run is one line on the near-null tail plot. To add/remove a run, just edit this list -- the
# plot reads <base>/<subdir>/straight-comp-int-none.eig.csv and skips any subdir with no CSV yet.
# Fields (subdir + the per-run geometry, shown in the legend so same-aspect runs stay distinguishable):
#   aspect = L/R,  R = R_shaft,  nelem = n_axial,  fourier/order fixed across the sweep.
# Aspect examples: base R=0.3,L=5 -> 16.7; 0.1*R -> 167; 0.01*R -> 1670; R=0.03,L=0.5 -> 16.7 (0.1x body).
TAIL_RUNS = [
    # original sweep (regenerate by uncommenting the matching lines in the submit script)
    {"subdir": "ar17",       "aspect": 16.7, "R": 0.3,   "nelem": 6,  "fourier": 8, "order": 4},
    {"subdir": "ar167",      "aspect": 167,  "R": 0.03,  "nelem": 6,  "fourier": 8, "order": 4},
    {"subdir": "ar167-n60",  "aspect": 167,  "R": 0.03,  "nelem": 60, "fourier": 8, "order": 4},
    # new runs
    {"subdir": "ar1670",     "aspect": 1670, "R": 0.003, "nelem": 6,  "fourier": 8, "order": 4},
    {"subdir": "ar1670-n60", "aspect": 1670, "R": 0.003, "nelem": 60, "fourier": 8, "order": 4},
    {"subdir": "ar17-smallr","aspect": 16.7, "R": 0.03,  "nelem": 6,  "fourier": 8, "order": 4},
]


def plot_nearzero_tail(base=DIR, geom="straight-comp", prob="int", variant="none"):
    """Near-null tail of the completed interior operator, one line per TAIL_RUNS entry that has data:
    log-y |lambda_i| < THRESH vs the raw (unnormalized) index, sorted ascending. A run with a single
    sub-threshold eigenvalue shows as one marker (drawn large enough to be visible, with axis margins
    so it never sits on the spine)."""
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    any_curve = False
    for k, run in enumerate(TAIL_RUNS):
        path = os.path.join(base, run["subdir"], f"{geom}-{prob}-{variant}.eig.csv")
        if not os.path.exists(path):
            print("skip (no data):", path)
            continue
        re, im, minmag, depF, depFn = read_eig(path)
        mag = np.hypot(re, im)
        sel = np.sort(mag[mag < THRESH])
        if sel.size == 0:
            print(f"no |lambda|<{THRESH:g} in {path}")
            continue
        # marker 'o' (ms=7) so a lone point is clearly visible; a thin line joins multi-point curves.
        ax.semilogy(np.arange(sel.size), sel, marker="o", ms=7, mew=0.6, mec="white", lw=1.3,
                    color=cmap(k % 10),
                    label=f"a.r.={run['aspect']:g}, R={run['R']:g}, Nelem={run['nelem']}, "
                          f"Fourier={run['fourier']}, order={run['order']}  "
                          f"($n_{{<{THRESH:g}}}$={sel.size}, min$|\\lambda|$={sel[0]:.2e})")
        any_curve = True
    if not any_curve:
        ax.text(0.5, 0.5, "no data -- run submit-capsule-pspectrum-completion-aspect.sh",
                ha="center", transform=ax.transAxes)
    ax.margins(x=0.04, y=0.10)                 # keep a lone point off the spine so it always shows
    ax.set_xlabel("index $i$  (near-null eigenvalues, sorted ascending, unnormalized)")
    ax.set_ylabel(r"$|\lambda_i|$   (log scale)")
    ax.set_title("Near-null spectrum of the net-flux-completed interior operator\n"
                 rf"$|\lambda_i| < {THRESH:g}$  --  straight capsule (aspect / arm-resolution / scale sweep)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    out = os.path.join(base, "pspectrum-comp-nearzero-tail.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "straight"
    if arg == "tail":
        # Near-null tail sweep:  python3 python/plot_pspectrum.py tail [sweep_base_dir]
        base = sys.argv[2] if len(sys.argv) > 2 else DIR
        plot_nearzero_tail(base)
    else:
        geom = arg
        plot_geom(geom)                      # singular-value spectra (kept)
        # Complex-plane eigenvalue CLUSTERING scatters are OFF by default (per the near-null-tail
        # study); set PSPEC_CLUSTER=1 to restore them.
        if os.environ.get("PSPEC_CLUSTER", "0") != "0":
            plot_eig(geom)
            plot_eig_completed(geom)
            plot_eig_laplace(geom)
