#!/usr/bin/env python3
"""
plot_pspectrum_dl.py -- spectrum plots for the S-FREE rank-7 DL-completion interior operator.

Parallel to plot_pspectrum.py, but for the operator built by `capsule-robin-bie pspectrum` with
QJ_DL_COMPLETION=1 (dl_completion.hpp):

    A_dl = (-1/2 I + D)  +  w_flux flux-proj      (NO single layer S; rank-1 flux completion).

Its CSVs live under the "straight-dlflux" (flux completion) or "straight-dlonly" (no completion) gtag,
one aspect run per subdir (ar17_dl / ar167_dl / ar1670_dl), written by
scripts/submit-capsule-pspectrum-dlcompletion-aspect.sh. Two plot families (per gtag):

  (1) CLUSTER  -- complex-plane eigenvalue scatter of A_dl per aspect (full spectrum + origin zoom):
                  where does the flux completion move the near-null eigenvalue?  ->  pspectrum-<tag>-eig.png
  (2) DECAY    -- near-null tail |lambda_i| < THRESH vs aspect ratio (one line per aspect, log-y),
                  the "how the near-null modes decay as a.r. increases" figure  ->
                  pspectrum-<tag>-nearzero-tail.png

Usage:
  python3 python/plot_pspectrum_dl.py [cluster|tail|all] [base_dir] [geom=straight-dlflux]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "vis/capsule-pspectrum/comp-aspect"
# gtag of the S-free DL operator variant to plot (set on the command line, 3rd positional):
#   straight-dlflux = rank-1 flux completion (default),  straight-dlonly = pure DL (no completion).
GEOM = "straight-dlflux"
PROB, VARIANT = "int", "none"
THRESH = 1e-2


def _tag(geom):
    """Short label for filenames/titles, e.g. 'straight-dlflux' -> 'dlflux'."""
    return geom.split("-", 1)[1] if "-" in geom else geom

# One entry per aspect run (subdir + geometry for the legend). Aspect = L/R with R shrunk 10x each step.
RUNS = [
    {"subdir": "ar17_dl",   "aspect": 16.7, "R": 0.3,   "nelem": 6, "fourier": 8, "order": 4},
    {"subdir": "ar167_dl",  "aspect": 167,  "R": 0.03,  "nelem": 6, "fourier": 8, "order": 4},
    {"subdir": "ar1670_dl", "aspect": 1670, "R": 0.003, "nelem": 6, "fourier": 8, "order": 4},
]


def read_eig(path):
    """Read a .eig.csv (index,re,im) + its header (min_abs_lambda / dep_normality_F / froF).
    Returns re, im, min|lambda|, dep_F, dep_F/||A||_F."""
    minmag = depF = froF = float("nan")
    with open(path) as f:
        for line in f:
            if line.startswith("# min_abs_lambda="):
                for tok in line[1:].split():
                    k, _, v = tok.partition("=")
                    if k == "min_abs_lambda": minmag = float(v)
                    elif k == "dep_normality_F": depF = float(v)
                    elif k == "froF": froF = float(v)
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    depF_norm = depF / froF if (froF and np.isfinite(froF) and froF != 0) else float("nan")
    return data[:, 1], data[:, 2], minmag, depF, depF_norm


def _path(base, run, geom):
    return os.path.join(base, run["subdir"], f"{geom}-{PROB}-{VARIANT}.eig.csv")


def plot_clusters(base, geom=GEOM):
    """Complex-plane eigenvalue scatter of A_dl, one column per aspect: full spectrum (top) + origin
    zoom (bottom). The zoom shades Re<0 so a straddle of the origin is obvious."""
    runs = [r for r in RUNS if os.path.exists(_path(base, r, geom))]
    if not runs:
        print(f"no {geom} .eig.csv found under {base} -- run the dlcompletion sweep first")
        return
    tag = _tag(geom)
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.4), squeeze=False)
    for j, run in enumerate(runs):
        re, im, minmag, depF, depFn = read_eig(_path(base, run, geom))
        col = plt.get_cmap("tab10")(j % 10)
        lab = (f"a.r.={run['aspect']:g}\nmin|λ|={minmag:.1e}\n"
               rf"dep$_F$/‖A‖$_F$={depFn:.3f}")
        af, az = axes[0][j], axes[1][j]
        af.scatter(re, im, s=7, color=col, alpha=0.55, edgecolors="none")
        af.set_title(f"{run['subdir']}  (a.r.={run['aspect']:g})")
        af.text(0.02, 0.98, lab, transform=af.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        az.scatter(re, im, s=10, color=col, alpha=0.65, edgecolors="none")
        az.set_xlim(-1.3, 1.3); az.set_ylim(-0.6, 0.6)
        az.axvspan(-1e9, 0, color="0.9", zorder=0)
        az.set_title("origin zoom")
        for a in (af, az):
            a.axvline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
            a.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
            a.set_xlabel(r"Re $\lambda$"); a.set_ylabel(r"Im $\lambda$")
            a.grid(True, alpha=0.25)
    fig.suptitle(f"Eigenvalue clustering of the interior DL operator [{tag}]  "
                 r"$-\frac{1}{2} I+D$ (+ flux completion)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(base, f"pspectrum-{tag}-eig.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def plot_nearzero_tail(base, geom=GEOM):
    """Near-null tail of A_dl as aspect ratio increases: one line per aspect, log-y |lambda_i|<THRESH
    sorted ascending against the raw (unnormalized) index. Shows how many eigenvalues stay near zero
    (and how small) once S is replaced by the 7-mode completion, and how that moves with a.r."""
    tag = _tag(geom)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    any_curve = False
    for k, run in enumerate(RUNS):
        path = _path(base, run, geom)
        if not os.path.exists(path):
            print("skip (no data):", path); continue
        re, im, minmag, depF, depFn = read_eig(path)
        mag = np.hypot(re, im)
        sel = np.sort(mag[mag < THRESH])
        if sel.size == 0:
            print(f"no |lambda|<{THRESH:g} in {path} (min|λ|={minmag:.2e})")
            continue
        ax.semilogy(np.arange(sel.size), sel, marker="o", ms=7, mew=0.6, mec="white", lw=1.3,
                    color=cmap(k % 10),
                    label=f"a.r.={run['aspect']:g}, R={run['R']:g}, Nelem={run['nelem']} "
                          f"($n_{{<{THRESH:g}}}$={sel.size}, min$|\\lambda|$={sel[0]:.2e})")
        any_curve = True
    if not any_curve:
        ax.text(0.5, 0.5, "no data -- run submit-capsule-pspectrum-dlcompletion-aspect.sh",
                ha="center", transform=ax.transAxes)
    ax.margins(x=0.04, y=0.10)
    ax.set_xlabel("index $i$  (near-null eigenvalues, sorted ascending, unnormalized)")
    ax.set_ylabel(r"$|\lambda_i|$   (log scale)")
    ax.set_title(f"Near-null decay of the interior DL operator [{tag}]\n"
                 rf"$|\lambda_i| < {THRESH:g}$  vs aspect ratio (straight capsule)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(base, f"pspectrum-{tag}-nearzero-tail.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    # usage: plot_pspectrum_dl.py [cluster|tail|all] [base_dir] [geom]
    #   geom = straight-dlflux (default) | straight-dlonly
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("cluster", "tail", "all"):
        base = sys.argv[2] if len(sys.argv) > 2 else BASE
        geom = sys.argv[3] if len(sys.argv) > 3 else GEOM
    else:                                  # first arg was the base dir
        base, arg = arg, "all"
        geom = sys.argv[2] if len(sys.argv) > 2 else GEOM
    if arg in ("cluster", "all"):
        plot_clusters(base, geom)
    if arg in ("tail", "all"):
        plot_nearzero_tail(base, geom)
