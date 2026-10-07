#!/usr/bin/env python3
"""plot_network_blend.py -- verify the DISPLACEMENT-BLEND network arms (QJ_ARM_BLEND) against vmtk.

The blend scheme keeps the raw vmtk centerline v(l) and adds a LOCALIZED correction that pulls each
arm mouth onto the junction-dictated seam point:  c(l) = v(l) + a_A(l)*delta_A + a_B(l)*delta_B, with
a smooth ramp a=1 at the mouth decaying to 0 a window inward.  So the meshed arm should (1) HUG the grey
vmtk skeleton and (2) deviate from it only in a SHORT band near each junction seam.

This script overlays the meshed arms on the vmtk graph AND plots, per arm, the displacement
|c(l) - v(l)| (nearest distance from each meshed centerline node to its vmtk edge polyline) vs the
normalized arclength along the arm -- the direct check that the displacement is small and localized.

Usage:
    python python/plot_network_blend.py <graph> <approx-dump> [out.png]

Regenerate the dump:
    QJ_ARM_BLEND=1 QJ_DUMP_APPROX=/tmp/net.approx \
        ./bin/bifurc-network-assemble /tmp/subnet.graph geom/scratch 12 1
"""
import os
import sys
import numpy as np

# reuse the parsers from the sibling script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_network_approx import parse_graph, parse_approx


def seg_dist(p, a, b):
    """distance from point p to segment ab."""
    ab = b - a
    d2 = float(ab @ ab)
    w = 0.0 if d2 < 1e-30 else max(0.0, min(1.0, float((p - a) @ ab) / d2))
    return float(np.linalg.norm(p - (a + w * ab)))


def poly_dist(p, poly):
    """nearest distance from point p to a polyline (Nx3)."""
    return min(seg_dist(p, poly[k], poly[k + 1]) for k in range(len(poly) - 1))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    graph_path, approx_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(approx_path)[0] + "_blend.png"

    nodes, edges = parse_graph(graph_path)
    junc, arms = parse_approx(approx_path)

    # index vmtk edges by unordered node-id pair, so each meshed arm (owner, other) finds its polyline.
    edge_by_pair = {}
    for e in edges:
        edge_by_pair[frozenset((e["n0"], e["n1"]))] = e["pts"]

    # per-arm displacement (nearest distance from each meshed node to the arm's own vmtk polyline).
    # the approx dump carries owner/other in NARM cols 0/1 (see plot_network_approx.parse_approx: it drops
    # them, so re-read here to get the pairing).
    with open(approx_path) as f:
        toks = [ln.split() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    pairs, i = [], 0
    while i < len(toks):
        if toks[i][0] == "NARM":
            na = int(toks[i][1]); i += 1
            for _ in range(na):
                t = toks[i]; pairs.append((int(t[0]), int(t[1]))); i += 1 + int(t[9])
        elif toks[i][0] == "NJUNC":
            i += 1 + int(toks[i][1])
        else:
            i += 1

    # EXACT reference v(l) (blend mode writes <approx>.ref: same stations as the meshed arm). When present
    # the displacement is |c(l)-v(l)| pointwise (the true blend correction); else fall back to nearest-to-
    # polyline distance, which on curved edges also picks up the CR-vs-chord interpolation gap.
    ref_by_arm = {}
    ref_path = approx_path + ".ref"
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            rtoks = [ln.split() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        k = 0
        while k < len(rtoks):
            if rtoks[k][0] == "NARM":
                na = int(rtoks[k][1]); k += 1
                for a in range(na):
                    t = rtoks[k]; owner, other, npr = int(t[0]), int(t[1]), int(t[2]); k += 1
                    rp = np.array([[float(x) for x in rtoks[k + m][:3]] for m in range(npr)])
                    k += npr
                    ref_by_arm[a] = rp
            else:
                k += 1
    exact = len(ref_by_arm) > 0

    devs, all_max, all_mean = [], [], []
    for ai, (A, (owner, other)) in enumerate(zip(arms, pairs)):
        pts = A["pts"]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        s = s / s[-1] if s[-1] > 0 else s
        if ai in ref_by_arm and len(ref_by_arm[ai]) == len(pts):
            d = np.linalg.norm(pts - ref_by_arm[ai], axis=1)          # exact |c(l) - v(l)|
        else:
            poly = edge_by_pair.get(frozenset((owner, other)))
            if poly is None or len(poly) < 2:
                devs.append(None); continue
            d = np.array([poly_dist(p, poly) for p in pts])           # nearest-to-polyline fallback
        devs.append((s, d, A["is_cap"]))
        all_max.append(d.max()); all_mean.append(d.mean())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_dev = fig.add_subplot(gs[1, :])

    for a, (i, j, ttl) in zip((ax_xy, ax_xz), [(0, 1, "x-y"), (0, 2, "x-z")]):
        for e in edges:
            a.plot(e["pts"][:, i], e["pts"][:, j], "-", color="#B8C0C8", lw=0.9, zorder=1)
        for A in arms:
            col = "#3E7CAE" if A["is_cap"] else "#B03A2E"
            a.plot(A["pts"][:, i], A["pts"][:, j], "-", color=col,
                   lw=1.5 if not A["is_cap"] else 0.9, alpha=0.9, zorder=2)
            a.scatter(A["cA"][i], A["cA"][j], s=10, c="#1F8A4C", marker="s", zorder=3)
        if junc.size:
            a.scatter(junc[:, i], junc[:, j], s=28, c="#C0392B", zorder=4)
        a.set_aspect("equal"); a.set_title("blend arm vs vmtk  (%s)" % ttl)
    handles = [Line2D([], [], color="#B8C0C8", lw=1.2, label="vmtk centerline"),
               Line2D([], [], color="#B03A2E", lw=1.6, label="meshed interior arm (blend)"),
               Line2D([], [], color="#3E7CAE", lw=1.2, label="meshed leaf arm (blend)"),
               Line2D([], [], color="#1F8A4C", marker="s", ls="", label="seam mouth"),
               Line2D([], [], color="#C0392B", marker="o", ls="", label="junction")]
    ax_xy.legend(handles=handles, loc="best", fontsize=9)

    # displacement vs normalized arclength (0=mouth A, 1=mouth B). Should peak at the ends, ~0 in middle.
    for dv in devs:
        if dv is None:
            continue
        s, d, is_cap = dv
        ax_dev.plot(s, d, "-", color=("#3E7CAE" if is_cap else "#B03A2E"),
                    lw=0.8, alpha=0.55)
    ax_dev.set_xlabel("normalized arclength along arm  (0 = mouth A,  1 = mouth B)")
    ax_dev.set_ylabel("displacement  |c(l) - v(l)|  (world units)")
    ax_dev.grid(True, alpha=0.3)
    if all_max:
        ax_dev.set_title("arm-centerline displacement from vmtk %s   "
                         "(median-max %.3g,  max %.3g,  median-mean %.3g,  over %d arms)"
                         % ("[exact |c-v|]" if exact else "[nearest-to-polyline]",
                            np.median(all_max), np.max(all_max), np.median(all_mean), len(all_max)))
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)
    if all_max:
        print("  displacement over %d arms:  max=%.4g  median-of-per-arm-max=%.4g  "
              "mean-of-per-arm-mean=%.4g" % (len(all_max), np.max(all_max),
                                             np.median(all_max), np.mean(all_mean)))


if __name__ == "__main__":
    main()
