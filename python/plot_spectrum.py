#!/usr/bin/env python3
"""
Plot + qualitatively compare the singular-value spectra of the coupled SL+DL BIE self operator
produced by the QJ_SPECTRUM gate (scripts/run-operator-spectrum.sh -> vis/spectrum/*.csv).

Each CSV: two header comment lines
    # label=... N=... interior=... SL_scal=... DL_scal=... jump=...
    # cond=... smax=... smin=...
then rows  index,sigma  (sigma sorted descending).

We produce three comparison figures + a cond-number table, grouped by the physical lever:
  (a) LENGTH   : capsule L=1 vs L=10 (same n_axial), and racetrack L~5 vs L~10; overlay the
                 same-count and refined/auto-panel variants to show the small-sigma tail is
                 mesh-independent (2nd-kind).
  (b) CURVATURE: straight capsule L=10 vs spiral amp0.5 (same L/R/n_axial).
  (c) LUMEN    : capsule (open genus-0) vs ybifurc-hybrid (branched genus-0) vs racetrack
                 (genus-1 closed-loop lumen) -- the small-sigma tail that drives GMRES.

Usage:  python python/plot_spectrum.py vis/spectrum [out_dir]
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    """Return (label, N, cond, smax, smin, sigma[])."""
    meta = {}
    sig = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.lstrip("#").split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
                continue
            if line.lower().startswith("index"):
                continue
            parts = line.split(",")
            if len(parts) == 2:
                sig.append(float(parts[1]))
    sig = np.array(sorted(sig, reverse=True))
    label = meta.get("label", os.path.splitext(os.path.basename(path))[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    smax = float(meta.get("smax", sig.max() if len(sig) else np.nan))
    smin = float(meta.get("smin", sig.min() if len(sig) else np.nan))
    # A closed interior lumen carries a physical flux/pressure NULL MODE per independent enclosed
    # channel (net-flux compatibility of interior Stokes-Dirichlet); those sigma ~ 1e-9 swamp the raw
    # cond and make length/curvature invisible. Count them (sigma < RELNULL*smax) and report a TRIMMED
    # cond = smax / (smallest sigma ABOVE the null floor) -- the conditioning the GMRES actually sees
    # once the (compatible) RHS is projected off the null space. n_null itself is the LUMEN signal.
    RELNULL = 1e-6
    nz = sig[sig > RELNULL * smax] if len(sig) else sig
    n_null = int(len(sig) - len(nz))
    smin_trim = float(nz.min()) if len(nz) else np.nan
    cond_trim = smax / smin_trim if smin_trim and np.isfinite(smin_trim) else np.nan
    return dict(stem=stem, label=label, N=int(meta.get("N", len(sig))),
                cond=float(meta.get("cond", (smax / smin if smin else np.nan))),
                smax=smax, smin=smin, sig=sig,
                n_null=n_null, cond_trim=cond_trim, smin_trim=smin_trim)


def overlay(ax, cases, title, tail_frac=0.25):
    """Overlay sorted-sigma curves (log-y) vs normalized index, tail zoom inset-free."""
    for c in cases:
        s = c["sig"]
        if len(s) == 0:
            continue
        x = np.arange(len(s)) / max(len(s) - 1, 1)
        ax.semilogy(x, s, lw=1.6,
                    label=f"{c['stem']}  (N={c['N']}, n_null={c['n_null']}, "
                          f"cond_trim={c['cond_trim']:.1e})")
    ax.set_title(title)
    ax.set_xlabel("normalized singular-value index  (0=largest)")
    ax.set_ylabel(r"$\sigma$")
    ax.axhline(0.5, color="0.6", ls=":", lw=1, label=r"$|jump|=0.5$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")


def tail(ax, cases, title, ntail=60):
    """Zoom on the SMALLEST singular values -- the conditioning-determining tail."""
    for c in cases:
        s = c["sig"]
        if len(s) == 0:
            continue
        k = min(ntail, len(s))
        smin_tail = s[-k:][::-1]  # ascending: smallest first
        ax.semilogy(np.arange(1, k + 1), smin_tail, marker=".", ms=4, lw=1.2,
                    label=f"{c['stem']} (n_null={c['n_null']}, smin_trim={c['smin_trim']:.1e})")
    ax.set_title(title + "  -- small-sigma tail")
    ax.set_xlabel("rank from the smallest singular value")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else d
    os.makedirs(out, exist_ok=True)

    cases = {c["stem"]: c for c in (load(p) for p in sorted(glob.glob(os.path.join(d, "*.csv"))))}
    if not cases:
        print(f"no CSVs in {d}")
        sys.exit(1)

    # ---- cond-number table (raw cond, plus null-mode count + trimmed cond) ----
    print(f"\n{'case':22s} {'N':>7s} {'cond_raw':>11s} {'n_null':>6s} {'cond_trim':>11s} "
          f"{'smax':>11s} {'smin_trim':>11s}")
    print("-" * 86)
    for k in sorted(cases):
        c = cases[k]
        print(f"{c['stem']:22s} {c['N']:7d} {c['cond']:11.3e} {c['n_null']:6d} "
              f"{c['cond_trim']:11.3e} {c['smax']:11.3e} {c['smin_trim']:11.3e}")
    with open(os.path.join(out, "cond_table.csv"), "w") as f:
        f.write("case,N,cond_raw,n_null,cond_trim,smax,smin_raw,smin_trim\n")
        for k in sorted(cases):
            c = cases[k]
            f.write(f"{c['stem']},{c['N']},{c['cond']:.8e},{c['n_null']},{c['cond_trim']:.8e},"
                    f"{c['smax']:.8e},{c['smin']:.8e},{c['smin_trim']:.8e}\n")

    def pick(*stems):
        return [cases[s] for s in stems if s in cases]

    # ---- (a) LENGTH + panel study ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cap_len = pick("cap_L1_na8", "cap_L10_na8", "cap_L10_na16", "cap_L10_na24")
    overlay(ax[0, 0], cap_len, "Capsule length & panel refinement (R=0.3)")
    tail(ax[0, 1], cap_len, "Capsule length & panels")
    race_len = pick("race_L5_auto", "race_L10_auto", "race_L5_fixed", "race_L10_fixed")
    overlay(ax[1, 0], race_len, "Racetrack length: auto (scaled) vs fixed panels")
    tail(ax[1, 1], race_len, "Racetrack length")
    fig.suptitle("(a) LENGTH: does body length move the spectrum?  Same-count vs scaled/refined panels",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out, "spectrum_length.png"), dpi=130)

    # ---- (b) CURVATURE ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    curv = pick("cap_L10_na12", "cap_spiral_amp0p5")
    overlay(ax[0], curv, "Straight vs spiral (L=10, R=0.3, n_axial=12)")
    tail(ax[1], curv, "Straight vs spiral")
    fig.suptitle("(b) CURVATURE: straight tube vs spiral amp0.5", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, "spectrum_curvature.png"), dpi=130)

    # ---- (c) LUMEN / topology ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lumen = pick("cap_L10_na8", "hybrid", "race_L10_auto")
    overlay(ax[0], lumen, "Open tube vs branched junction vs genus-1 loop")
    tail(ax[1], lumen, "Lumen topology")
    fig.suptitle("(c) LUMEN: open genus-0 (capsule) / branched genus-0 (ybifurc-hybrid) / "
                 "genus-1 closed loop (racetrack)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, "spectrum_lumen.png"), dpi=130)

    print(f"\nwrote spectrum_length.png, spectrum_curvature.png, spectrum_lumen.png, cond_table.csv -> {out}")


if __name__ == "__main__":
    main()
