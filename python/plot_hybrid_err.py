#!/usr/bin/env python3
"""Plot the per-node error distribution for the hybrid all-finger Stokes DL and on-surface Green's tests.
Reads hybrid_DL_err.csv / hybrid_green_err.csv (x,y,z,r,region,err); region 0=base(quad foot/sphere), 1=slender."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = "/mnt/home/tli10/quad-junctions"

def load(fn):
    a = np.genfromtxt(fn, delimiter=",", names=True)
    return a

def panel(ax_r, ax_3d, a, title):
    r = a["r"]; err = a["err"]; reg = a["region"].astype(int)
    x, y, z = a["x"], a["y"], a["z"]
    err = np.maximum(err, 1e-16)  # floor for log
    # err vs radius, split by region
    for rv, lab, c in [(0, "base (quad foot / sphere)", "#3b6fb0"), (1, "slender shaft", "#d1682a")]:
        m = reg == rv
        if m.any():
            ax_r.scatter(r[m], err[m], s=6, alpha=0.5, c=c, label=lab, edgecolors="none")
    ax_r.set_yscale("log"); ax_r.set_xlabel("radius  r = |x|"); ax_r.set_ylabel("|error|")
    ax_r.set_title(title + "  —  error vs radius"); ax_r.legend(loc="upper left", fontsize=8)
    ax_r.grid(True, which="both", alpha=0.25)
    # 3d spatial, colored by log10 err (viridis, perceptually uniform); subsample for legibility
    le = np.log10(err)
    n = len(x)
    if n > 40000:
        rng = np.random.default_rng(0); idx = rng.choice(n, 40000, replace=False)
    else:
        idx = np.arange(n)
    sc = ax_3d.scatter(x[idx], y[idx], z[idx], c=le[idx], s=4, cmap="viridis", alpha=0.7, edgecolors="none")
    ax_3d.set_title(title + "  —  spatial  (color = log10 |error|)")
    ax_3d.set_xlabel("x"); ax_3d.set_ylabel("y"); ax_3d.set_zlabel("z")
    cb = plt.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.08); cb.set_label("log10 |error|")
    # report where the max is
    im = np.argmax(err)
    print(f"  {title}: max |err|={err[im]:.3e} at (r={r[im]:.3f}, region={'slender' if reg[im] else 'base'}, xyz=({x[im]:.3f},{y[im]:.3f},{z[im]:.3f}))")

def main():
    tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""   # e.g. "ppf2" -> hybrid_DL_err_ppf2.csv/.png
    ttl = f"  (PPF={sys.argv[1][3:]})" if (len(sys.argv) > 1 and sys.argv[1].startswith("ppf")) else ""
    for fn, title, out in [
        (f"{WORK}/hybrid_DL_err{tag}.csv",   "Stokes DL identity"+ttl,       f"{WORK}/hybrid_DL_err{tag}.png"),
        (f"{WORK}/hybrid_green_err{tag}.csv","Stokes on-surface Green's"+ttl, f"{WORK}/hybrid_green_err{tag}.png"),
    ]:
        try:
            a = load(fn)
        except Exception as e:
            print(f"  skip {fn}: {e}"); continue
        fig = plt.figure(figsize=(13, 5.2))
        ax_r = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        panel(ax_r, ax_3d, a, title)
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  wrote {out}")

if __name__ == "__main__":
    main()
