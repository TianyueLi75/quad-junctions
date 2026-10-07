#!/usr/bin/env python3
"""plot_network_zoom.py -- zoomed vmtk-vs-meshed centerline overlays around chosen junctions/edges.

For each focus node (a junction or the junction end of a leaf), draw a local x-y and x-z overlay of the
grey vmtk skeleton vs the colored meshed (blend) arms in a tight box around that node, exactly like the
whole-network recreation plot but zoomed. Meshed arms incident to the focus node are drawn bold; the seam
mouths, junction centers, and the per-arm max displacement |c(l)-v(l)| (from <approx>.ref) are annotated.

Usage:
    python python/plot_network_zoom.py <graph> <approx-dump> [out.png] [--nodes a,b,c | --worst N]
Default: --worst 6  (auto-pick the N worst-recreated edges' endpoint junctions).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_network_approx import parse_graph, parse_approx
from plot_network_maxdev import read_pairs_and_ref
from plot_network_blend import poly_dist


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    graph_path, approx_path = args[0], args[1]
    out = args[2] if len(args) > 2 else os.path.splitext(approx_path)[0] + "_zoom.png"
    nodes_opt = None
    worst_n = 6
    for k, a in enumerate(sys.argv):
        if a == "--nodes":
            nodes_opt = [int(x) for x in sys.argv[k + 1].split(",")]
        if a == "--worst":
            worst_n = int(sys.argv[k + 1])

    nodes, edges = parse_graph(graph_path)
    junc, arms = parse_approx(approx_path)
    pairs, ref = read_pairs_and_ref(approx_path)
    edge_by_pair = {frozenset((e["n0"], e["n1"])): e["pts"] for e in edges}

    # per-arm max displacement + endpoints
    arm_dev = []
    for ai, (A, (ow, ot)) in enumerate(zip(arms, pairs)):
        pts = A["pts"]
        if ai in ref and len(ref[ai]) == len(pts):
            d = np.linalg.norm(pts - ref[ai], axis=1)
        else:
            poly = edge_by_pair.get(frozenset((ow, ot)))
            d = np.array([poly_dist(p, poly) for p in pts]) if poly is not None and len(poly) >= 2 else np.array([np.nan])
        arm_dev.append((ai, ow, ot, float(np.nanmax(d)), bool(A["is_cap"])))

    # choose focus nodes
    if nodes_opt is None:
        worst = sorted(arm_dev, key=lambda t: -t[3])[:worst_n]
        focus = []
        for ai, ow, ot, mx, cap in worst:
            # the "problem" end is the JUNCTION whose seam sits off vmtk; for a leaf that's the junction (min-ish),
            # for interior pick the owner (min id). Use whichever endpoint is a junction & carries a placed center.
            fj = ow if nodes.get(ow, {}).get("typ", 1) == 0 else ot
            if fj not in focus:
                focus.append(fj)
        focus = focus[:worst_n]
    else:
        focus = nodes_opt

    # arms incident to a node (as owner or other)
    def incident_arms(nid):
        return [ai for ai, ow, ot, mx, cap in arm_dev if ow == nid or ot == nid]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ncol = 2  # x-y and x-z
    nrow = len(focus)
    fig, ax = plt.subplots(nrow, ncol, figsize=(13, 5.2 * nrow))
    if nrow == 1:
        ax = ax.reshape(1, 2)

    for r, fj in enumerate(focus):
        c0 = nodes[fj]["pos"]
        inc = incident_arms(fj)
        maxd_here = max([arm_dev[ai][3] for ai in inc], default=0.0)
        # tight box around the seam region: the deviation lives near the mouths (seam-center offset), not
        # out along the long cap tubes. Size from the seam-mouth offsets + the max c-v gap, clamped.
        seamoff = 0.0
        for ai in inc:
            seamoff = max(seamoff, float(np.linalg.norm(arms[ai]["cA"] - c0)))
        half = float(np.clip(max(7.0 * maxd_here, 4.5 * seamoff, 25.0), 30.0, 95.0))

        for cc, (i, j, ttl) in zip(range(2), [(0, 1, "x-y"), (0, 2, "x-z")]):
            a = ax[r, cc]
            # all vmtk edges (grey), but only those passing near the box matter visually
            for e in edges:
                p = e["pts"]
                if np.min(np.linalg.norm(p[:, [i, j]] - c0[[i, j]], axis=1)) < 1.6 * half:
                    a.plot(p[:, i], p[:, j], "-", color="#C2C8CE", lw=1.0, zorder=1)
            # meshed arms: bold if incident to focus node, faint otherwise
            for ai, (ow, ot, mx, cap) in [(t[0], (t[1], t[2], t[3], t[4])) for t in arm_dev]:
                pts = arms[ai]["pts"]
                if np.min(np.linalg.norm(pts[:, [i, j]] - c0[[i, j]], axis=1)) > 1.6 * half:
                    continue
                bold = ai in inc
                col = "#3E7CAE" if cap else "#B03A2E"
                a.plot(pts[:, i], pts[:, j], "-", color=col, lw=2.2 if bold else 0.8,
                       alpha=0.95 if bold else 0.4, zorder=3 if bold else 2)
                a.scatter(arms[ai]["cA"][i], arms[ai]["cA"][j], s=26 if bold else 8,
                          c="#1F8A4C", marker="s", zorder=4)
            # exact vmtk reference for incident arms (dotted dark) + a marker/segment at the max c-v gap
            for ai in inc:
                if ai in ref and len(ref[ai]) == len(arms[ai]["pts"]):
                    rp = ref[ai]; mp = arms[ai]["pts"]
                    a.plot(rp[:, i], rp[:, j], ":", color="#333", lw=1.1, zorder=2)
                    km = int(np.argmax(np.linalg.norm(mp - rp, axis=1)))
                    a.plot([mp[km, i], rp[km, i]], [mp[km, j], rp[km, j]], "-", color="#7D3C98", lw=1.6, zorder=5)
                    a.scatter([mp[km, i]], [mp[km, j]], s=30, c="#7D3C98", marker="x", zorder=6)
            a.scatter([c0[i]], [c0[j]], s=60, c="#C0392B", marker="o", zorder=6)
            for nb in [nodes[n] for n in nodes if np.linalg.norm(nodes[n]["pos"] - c0) < half and n != fj]:
                mk = "o" if nb["typ"] == 0 else "^"
                a.scatter([nb["pos"][i]], [nb["pos"][j]], s=30, c="#E67E22", marker=mk, zorder=5)
            a.set_xlim(c0[i] - half, c0[i] + half)
            a.set_ylim(c0[j] - half, c0[j] + half)
            a.set_aspect("equal")
            a.set_title("j%d  (%s)   max|c-v| here = %.3g" % (fj, ttl, maxd_here))

    handles = [Line2D([], [], color="#C2C8CE", lw=1.4, label="vmtk centerline"),
               Line2D([], [], color="#333", lw=1.1, ls=":", label="vmtk ref of focus arm (c-v gap)"),
               Line2D([], [], color="#B03A2E", lw=2.2, label="meshed interior arm (focus)"),
               Line2D([], [], color="#3E7CAE", lw=2.2, label="meshed leaf arm (focus)"),
               Line2D([], [], color="#1F8A4C", marker="s", ls="", label="seam mouth"),
               Line2D([], [], color="#C0392B", marker="o", ls="", label="focus junction"),
               Line2D([], [], color="#E67E22", marker="o", ls="", label="neighbour node")]
    ax[0, 0].legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle("zoomed vmtk vs new (n298 blend) centerline -- worst-recreated junctions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=140)
    print("wrote", out, "  focus nodes:", focus)


if __name__ == "__main__":
    main()
