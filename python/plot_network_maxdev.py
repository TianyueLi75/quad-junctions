#!/usr/bin/env python3
"""plot_network_maxdev.py -- network recreation + per-EDGE max displacement (subnet_blend.png layout).

Same figure layout as plot_network_blend.py (top = vmtk-vs-meshed overlay in two projections), but the
BOTTOM panel is the MAX displacement |c(l) - v(l)| of each meshed arm plotted BY EDGE INDEX (a stem per
arm), rather than displacement-vs-arclength curves overlaid. This makes the worst-recreated edges pop out
directly by index -- e.g. the j298 interior arm.

Displacement uses the EXACT per-arm reference the assembler writes to <approx>.ref (raw vmtk centerline
v(l) sampled at the meshed stations); falls back to nearest-distance-to-vmtk-polyline if no .ref present.

Usage:
    python python/plot_network_maxdev.py <graph> <approx-dump> [out.png]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_network_approx import parse_graph, parse_approx
from plot_network_blend import poly_dist


def read_pairs_and_ref(approx_path):
    """arm (owner, other) pairs in file order, and the exact per-arm reference polylines from <approx>.ref."""
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
                    t = rtoks[k]; npr = int(t[2]); k += 1
                    rp = np.array([[float(x) for x in rtoks[k + m][:3]] for m in range(npr)]); k += npr
                    ref_by_arm[a] = rp
            else:
                k += 1
    return pairs, ref_by_arm


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    graph_path, approx_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(approx_path)[0] + "_maxdev.png"

    nodes, edges = parse_graph(graph_path)
    junc, arms = parse_approx(approx_path)
    pairs, ref_by_arm = read_pairs_and_ref(approx_path)
    exact = len(ref_by_arm) > 0

    edge_by_pair = {frozenset((e["n0"], e["n1"])): e["pts"] for e in edges}

    # per-arm max & mean displacement, tagged interior/leaf, keyed by arm (edge) index.
    idx, maxdev, meandev, is_cap, worst = [], [], [], [], []
    for ai, (A, (owner, other)) in enumerate(zip(arms, pairs)):
        pts = A["pts"]
        if ai in ref_by_arm and len(ref_by_arm[ai]) == len(pts):
            d = np.linalg.norm(pts - ref_by_arm[ai], axis=1)
        else:
            poly = edge_by_pair.get(frozenset((owner, other)))
            if poly is None or len(poly) < 2:
                continue
            d = np.array([poly_dist(p, poly) for p in pts])
        idx.append(ai); maxdev.append(float(d.max())); meandev.append(float(d.mean()))
        is_cap.append(bool(A["is_cap"])); worst.append((float(d.max()), owner, other, bool(A["is_cap"])))
    idx = np.array(idx); maxdev = np.array(maxdev); meandev = np.array(meandev)
    is_cap = np.array(is_cap)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_dev = fig.add_subplot(gs[1, :])

    # ---- top: recreation overlay (grey vmtk skeleton + colored meshed arms), two projections ----
    for a, (i, j, ttl) in zip((ax_xy, ax_xz), [(0, 1, "x-y"), (0, 2, "x-z")]):
        for e in edges:
            a.plot(e["pts"][:, i], e["pts"][:, j], "-", color="#B8C0C8", lw=0.7, zorder=1)
        for A in arms:
            col = "#3E7CAE" if A["is_cap"] else "#B03A2E"
            a.plot(A["pts"][:, i], A["pts"][:, j], "-", color=col,
                   lw=1.3 if not A["is_cap"] else 0.8, alpha=0.9, zorder=2)
        if junc.size:
            a.scatter(junc[:, i], junc[:, j], s=10, c="#C0392B", zorder=4)
        a.set_aspect("equal"); a.set_title("network recreation: blend arms vs vmtk  (%s)" % ttl)
    handles = [Line2D([], [], color="#B8C0C8", lw=1.2, label="vmtk centerline"),
               Line2D([], [], color="#B03A2E", lw=1.6, label="meshed interior arm (blend)"),
               Line2D([], [], color="#3E7CAE", lw=1.2, label="meshed leaf arm (blend)"),
               Line2D([], [], color="#C0392B", marker="o", ls="", label="junction")]
    ax_xy.legend(handles=handles, loc="best", fontsize=9)

    # ---- bottom: MAX displacement per arm, by edge index (stem) ----
    ci = ~is_cap
    ax_dev.vlines(idx[ci], 0, maxdev[ci], color="#B03A2E", lw=1.1, alpha=0.85, label="interior arm")
    ax_dev.vlines(idx[~ci], 0, maxdev[~ci], color="#3E7CAE", lw=1.1, alpha=0.85, label="leaf arm")
    ax_dev.scatter(idx[ci], maxdev[ci], s=14, color="#B03A2E", zorder=3)
    ax_dev.scatter(idx[~ci], maxdev[~ci], s=12, color="#3E7CAE", zorder=3)
    med = float(np.median(maxdev)) if maxdev.size else 0.0
    ax_dev.axhline(med, color="#555", ls="--", lw=1.0, label="median-of-per-arm-max = %.3g" % med)
    # annotate the few worst edges by index
    for mx, ow, ot, cap in sorted(worst, reverse=True)[:4]:
        ai = idx[np.argmin(np.abs(maxdev - mx))]
        ax_dev.annotate("edge %d  (j%d-j%d)" % (ai, ow, ot), xy=(ai, mx),
                        xytext=(ai, mx * 1.02), fontsize=8, ha="center",
                        color="#7B241C" if not cap else "#1A5276")
    ax_dev.set_xlabel("arm (edge) index")
    ax_dev.set_ylabel("max displacement  max_l |c(l) - v(l)|  (world units)")
    ax_dev.grid(True, axis="y", alpha=0.3)
    ax_dev.legend(loc="upper right", fontsize=9)
    ax_dev.set_title("per-edge max recreation displacement %s   (median %.3g,  max %.3g,  over %d arms)"
                     % ("[exact |c-v|]" if exact else "[nearest-to-polyline]",
                        med, float(maxdev.max()) if maxdev.size else 0.0, len(idx)))

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)
    if maxdev.size:
        print("  per-edge max displacement:  median=%.4g  max=%.4g  mean-of-per-arm-mean=%.4g  (%d arms)"
              % (med, float(maxdev.max()), float(meandev.mean()), len(idx)))
        print("  worst 6 edges (max-dev : owner-other : kind):")
        for mx, ow, ot, cap in sorted(worst, reverse=True)[:6]:
            print("    %.4g  :  j%d-j%d  :  %s" % (mx, ow, ot, "leaf" if cap else "interior"))


if __name__ == "__main__":
    main()
