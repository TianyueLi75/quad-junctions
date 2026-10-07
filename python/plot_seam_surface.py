#!/usr/bin/env python3
"""plot_seam_surface.py -- zoomed centerline + reconstructed SURFACE of a lead-corner seam.

Renders, for each (owner-junction, other-node) arm, a tight view of the coaxial-lead corner at the arm
MOUTH where the tube self-folds (curvature radius rho(s) < tube radius r(s)). Three columns per seam:
  (1) 3D: junction-body node cloud (grey) + reconstructed arm tube surface near the mouth, tube rings
      colored by rho/r (red = rho<r = inner-wall self-fold);
  (2) centerline x-y (zoom) with the meshed arm (bold), the vmtk reference (dotted), the seam mouth;
  (3) rho(s) and r(s) vs arclength: where the red rho dips below the grey r band the tube folds.

Surface is reconstructed from the per-node frame in <prefix>-jNNN.arms: each station has center (x,y,z),
radius r, and e1=(ox,oy,oz); e2 = t x e1 (t = centerline tangent); ring = center + r(cosθ e1 + sinθ e2)
over `fourier` azimuthal samples -- the same tube SlenderElemList meshes.

Usage:
    python python/plot_seam_surface.py <bundle-prefix> <approx> <out.png> owner:other[,owner:other,...]
    e.g. python python/plot_seam_surface.py geom/n298_s05 vis/n298_s05.approx vis/seam.png 311:312,210:220
"""
import os
import sys
import numpy as np


def parse_arms_bundle(path):
    d = [l.split() for l in open(path) if l.strip() and not l.lstrip().startswith("#")]
    i = 0
    order, narm, cheb, fourier = map(int, d[i]); i += 1
    arms = {}
    for _ in range(narm):
        other, is_cap, npanel = map(int, d[i]); i += 1
        C, R, E1 = [], [], []
        for _p in range(npanel):
            c, fo = map(int, d[i]); i += 1
            for _k in range(c):
                r = d[i]; i += 1
                C.append([float(r[0]), float(r[1]), float(r[2])])
                R.append(float(r[3])); E1.append([float(r[4]), float(r[5]), float(r[6])])
        arms[other] = dict(C=np.array(C), R=np.array(R), E1=np.array(E1), is_cap=is_cap, fourier=fourier)
    return arms


def parse_mesh_nodes(path, near=None, half=None):
    """Flat X Y Z node list (order in header). Optionally keep only nodes within `half` of `near`."""
    pts = []
    for l in open(path):
        if l.lstrip().startswith("#") or not l.strip():
            continue
        t = l.split()
        p = np.array([float(t[0]), float(t[1]), float(t[2])])
        if near is None or np.linalg.norm(p - near) <= half:
            pts.append(p)
    return np.array(pts) if pts else np.zeros((0, 3))


def tube_surface(C, R, E1, fourier):
    """Return X,Y,Z grids (Nstation x fourier) of the tube surface + per-station curvature radius rho."""
    n = len(C)
    T = np.zeros_like(C)
    T[1:-1] = C[2:] - C[:-2]
    T[0] = C[1] - C[0]; T[-1] = C[-1] - C[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-30
    th = np.linspace(0, 2 * np.pi, fourier + 1)
    X = np.zeros((n, fourier + 1)); Y = np.zeros_like(X); Z = np.zeros_like(X)
    for k in range(n):
        e1 = E1[k] - np.dot(E1[k], T[k]) * T[k]
        e1 /= np.linalg.norm(e1) + 1e-30
        e2 = np.cross(T[k], e1)
        ring = C[k][None, :] + R[k] * (np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :])
        X[k], Y[k], Z[k] = ring[:, 0], ring[:, 1], ring[:, 2]
    # curvature radius via circumradius of consecutive centerline triples
    rho = np.full(n, 1e30)
    for k in range(1, n - 1):
        a, b, c = C[k - 1], C[k], C[k + 1]
        cr = np.cross(b - a, b - c); area2 = np.linalg.norm(cr)
        if area2 > 1e-14:
            rho[k] = (np.linalg.norm(b - a) * np.linalg.norm(b - c) * np.linalg.norm(c - a)) / (2 * area2)
    rho[0], rho[-1] = rho[1], rho[-2]
    return X, Y, Z, rho, T


def main():
    prefix, approx_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seams = [tuple(int(x) for x in s.split(":")) for s in sys.argv[4].split(",")]

    # approx: node positions of junctions + arm centerlines (for the vmtk-ref overlay)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_network_approx import parse_approx
    from plot_network_maxdev import read_pairs_and_ref
    junc, arms_c = parse_approx(approx_path)
    pairs, ref = read_pairs_and_ref(approx_path)
    arm_by_pair = {frozenset(p): (ai, arms_c[ai]) for ai, p in enumerate(pairs)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    nrow = len(seams)
    fig = plt.figure(figsize=(19, 5.4 * nrow))

    for r, (owner, other) in enumerate(seams):
        bundle = parse_arms_bundle("%s-j%d.arms" % (prefix, owner))
        if other not in bundle:
            print("  [skip] arm %d->%d not in bundle" % (owner, other)); continue
        A = bundle[other]
        C, R, E1, fourier = A["C"], A["R"], A["E1"], A["fourier"]
        X, Y, Z, rho, T = tube_surface(C, R, E1, fourier)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        sN = s / s[-1]
        mouth = C[0]
        # zoom the tube to the corner region: first 35% of arclength (the coaxial lead + corner)
        klead = max(6, int(0.35 * len(C)))
        # box around the folding region
        kmin = int(np.argmin(rho))
        ctrp = C[min(kmin, len(C) - 1)]
        half = float(max(6.0 * R.max(), 3.5 * np.linalg.norm(C[klead] - mouth), 12.0))

        jn = parse_mesh_nodes("%s-j%d.mesh" % (prefix, owner), near=mouth, half=half)

        # ---- (1) 3D surface + centerline ----
        ax = fig.add_subplot(nrow, 3, 3 * r + 1, projection="3d")
        if len(jn):
            ax.scatter(jn[:, 0], jn[:, 1], jn[:, 2], s=1.2, c="#C2C8CE", alpha=0.25, zorder=1)
        rr = rho / np.maximum(R, 1e-9)                       # rho/r; <1 => inner-wall fold
        fc = plt.cm.RdYlGn(np.clip((rr - 0.5) / 1.5, 0, 1))  # red rho<r, green rho>=2r
        ax.plot_surface(X[:klead], Y[:klead], Z[:klead], facecolors=fc[:klead, None, :].repeat(fourier + 1, 1),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.9)
        ax.plot(C[:klead, 0], C[:klead, 1], C[:klead, 2], "-", color="#222", lw=1.8, zorder=6)
        ax.scatter([mouth[0]], [mouth[1]], [mouth[2]], c="#1F8A4C", s=40, marker="s")
        ax.set_title("j%d–j%d mouth surface (red = rho<r fold)\nrho_min=%.2g  r=%.2g" %
                     (owner, other, rho.min(), R.max()), fontsize=9)
        ax.set_xlim(mouth[0] - half, mouth[0] + half); ax.set_ylim(mouth[1] - half, mouth[1] + half)
        ax.set_zlim(mouth[2] - half, mouth[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # ---- (2) centerline x-y zoom ----
        ax2 = fig.add_subplot(nrow, 3, 3 * r + 2)
        i, j = 0, 1
        ax2.plot(C[:, i], C[:, j], "-", color="#B03A2E", lw=2.0, label="meshed arm")
        ai_c = arm_by_pair.get(frozenset((owner, other)))
        if ai_c is not None and ai_c[0] in ref:
            rp = ref[ai_c[0]]
            ax2.plot(rp[:, i], rp[:, j], ":", color="#333", lw=1.3, label="vmtk ref")
        ax2.scatter([mouth[i]], [mouth[j]], c="#1F8A4C", s=45, marker="s", label="seam mouth", zorder=6)
        ax2.scatter([C[kmin, i]], [C[kmin, j]], c="#7D3C98", s=45, marker="x", zorder=7, label="rho_min")
        ax2.set_xlim(mouth[i] - half, mouth[i] + half); ax2.set_ylim(mouth[j] - half, mouth[j] + half)
        ax2.set_aspect("equal"); ax2.grid(alpha=.3); ax2.legend(fontsize=8, loc="best")
        ax2.set_title("centerline x-y (zoom to lead corner)", fontsize=9)

        # ---- (3) rho(s) vs r(s) ----
        ax3 = fig.add_subplot(nrow, 3, 3 * r + 3)
        ax3.plot(sN, np.minimum(rho, 5 * R.max()), "-", color="#B03A2E", lw=1.6, label="curvature radius rho(s)")
        ax3.plot(sN, R, "-", color="#888", lw=1.6, label="tube radius r(s)")
        ax3.fill_between(sN, 0, R, color="#ccc", alpha=.4)
        bad = rho < R
        ax3.scatter(sN[bad], rho[bad], s=10, c="#7D3C98", zorder=5, label="rho<r (fold)")
        ax3.set_yscale("log"); ax3.set_xlabel("normalized arclength (0=mouth)")
        ax3.set_ylabel("radius (log)"); ax3.grid(alpha=.3, which="both"); ax3.legend(fontsize=8)
        ax3.set_title("rho(s) vs r(s): fold where red < grey", fontsize=9)

    fig.suptitle("scale=0.5 lead-corner seam folds (centerline + reconstructed tube surface)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=135)
    print("wrote", out)


if __name__ == "__main__":
    main()
