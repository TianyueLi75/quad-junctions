#!/usr/bin/env python3
"""plot_network_approx.py -- verify the junction-approximated network against the vmtk graph.

Overlays the ORIGINAL vmtk centerlines (grey, from <prefix>.graph) with the APPROXIMATION that
bifurc-network-assemble actually meshes (colored, from a QJ_DUMP_APPROX dump): placed junction
centers, per-arm seam mouths, and the meshed arm centerlines. With QJ_ARM_FOLLOW=1 the colored arm
curves should HUG the grey vmtk lines; the straight racetrack (default / QJ_STRAIGHT) cuts chords.

Usage:
    python python/plot_network_approx.py <graph> <approx-dump> [out.png]

Regenerate the dump:
    QJ_ARM_FOLLOW=1 QJ_DUMP_APPROX=/tmp/net.approx \
        ./bin/bifurc-network-assemble data/vmtk/vessels_fixed.graph geom/scratch 12 1 ...
"""
import os
import sys
import numpy as np


def parse_graph(path):
    """Return (nodes, edges): nodes[id]->dict(pos,typ); edges->list of dict(n0,n1,pts[N,3])."""
    with open(path) as f:
        toks = [ln.split() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    i, nodes, edges = 0, {}, []
    while i < len(toks):
        head = toks[i][0]
        if head == "NCLUSTER":
            i += 1 + int(toks[i][1])                       # skip cluster spec lines
        elif head == "NNODE":
            nn = int(toks[i][1]); i += 1
            for _ in range(nn):
                t = toks[i]; i += 1
                nid = int(t[0])
                nodes[nid] = dict(pos=np.array([float(t[1]), float(t[2]), float(t[3])]),
                                  typ=int(t[5]))            # 0=junction, 1=cap
        elif head == "NEDGE":
            ne = int(toks[i][1]); i += 1
            for _ in range(ne):
                t = toks[i]; i += 1
                n0, n1, npts = int(t[1]), int(t[2]), int(t[5])
                pts = np.array([[float(x) for x in toks[i + k][:3]] for k in range(npts)])
                i += npts
                edges.append(dict(n0=n0, n1=n1, pts=pts))
        else:
            i += 1
    return nodes, edges


def parse_approx(path):
    """Return (junc[Nj,3], arms): arms->list of dict(is_cap, cA[3], uA[3], pts[N,3])."""
    with open(path) as f:
        toks = [ln.split() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    i, junc, arms = 0, [], []
    while i < len(toks):
        head = toks[i][0]
        if head == "NJUNC":
            nj = int(toks[i][1]); i += 1
            for _ in range(nj):
                t = toks[i]; i += 1
                junc.append([float(t[1]), float(t[2]), float(t[3])])
        elif head == "NARM":
            na = int(toks[i][1]); i += 1
            for _ in range(na):
                t = toks[i]; i += 1
                is_cap = int(t[2]); cA = [float(t[3]), float(t[4]), float(t[5])]
                uA = [float(t[6]), float(t[7]), float(t[8])]; npts = int(t[9])
                pts = np.array([[float(x) for x in toks[i + k][:3]] for k in range(npts)])
                i += npts
                arms.append(dict(is_cap=is_cap, cA=np.array(cA), uA=np.array(uA), pts=pts))
        else:
            i += 1
    return np.array(junc).reshape(-1, 3), arms


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    graph_path, approx_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(approx_path)[0] + "_approx.png"

    nodes, edges = parse_graph(graph_path)
    junc, arms = parse_approx(approx_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1, 2, figsize=(22, 11))
    for a, (i, j, ttl) in zip(ax, [(0, 1, "x-y"), (0, 2, "x-z")]):
        # original vmtk centerlines (grey, underneath)
        for e in edges:
            a.plot(e["pts"][:, i], e["pts"][:, j], "-", color="#B8C0C8", lw=0.8, zorder=1)
        # short seam-axis (canonical junction angle) arrows: the direction each arm leaves its junction,
        # i.e. the "vertex replaced by the junction angles". Length = 12% of the median arm chord.
        alen = 0.12 * (np.median([np.linalg.norm(A["pts"][-1] - A["pts"][0]) for A in arms]) if arms else 1.0)
        # approximated (meshed) arm centerlines
        for A in arms:
            col = "#3E7CAE" if A["is_cap"] else "#B03A2E"
            a.plot(A["pts"][:, i], A["pts"][:, j], "-", color=col,
                   lw=1.4 if not A["is_cap"] else 0.9, alpha=0.9, zorder=2)
            a.scatter(A["cA"][i], A["cA"][j], s=8, c="#1F8A4C", marker="s", zorder=3)  # seam mouth
            u = A["uA"]
            a.annotate("", xytext=(A["cA"][i], A["cA"][j]),
                       xy=(A["cA"][i] + alen * u[i], A["cA"][j] + alen * u[j]),
                       arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.1), zorder=3)
        # placed junction centers
        if junc.size:
            a.scatter(junc[:, i], junc[:, j], s=26, c="#C0392B", zorder=4)
        a.set_aspect("equal"); a.set_title("approx vs vmtk  (%s)" % ttl)
    handles = [Line2D([], [], color="#B8C0C8", lw=1.2, label="vmtk centerline"),
               Line2D([], [], color="#B03A2E", lw=1.6, label="meshed interior arm"),
               Line2D([], [], color="#3E7CAE", lw=1.2, label="meshed leaf arm"),
               Line2D([], [], color="#1F8A4C", marker="s", ls="", label="seam mouth"),
               Line2D([], [], color="#E67E22", lw=1.1, label="seam axis (junction angle)"),
               Line2D([], [], color="#C0392B", marker="o", ls="", label="junction")]
    ax[0].legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print("wrote", out, "(%d junctions, %d arms, %d vmtk edges)" % (len(junc), len(arms), len(edges)))


if __name__ == "__main__":
    main()
