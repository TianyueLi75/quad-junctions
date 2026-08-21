#!/usr/bin/env python3
"""
Post-process a vmtk-derived `<in>.graph` (python/vessels_vmtk_graph.py) into a FIXED-shape / FIXED-size,
COORDINATE-SCALED graph for the C++ assembler (`bin/bifurc-network-assemble`). The output `.graph` is the
same plain-text format, so the assembler reproduces the geometry from it WITHOUT re-running vmtk.

Transforms:
  * SHAPE fixed  -> one SYMMETRIC canonical junction per DEGREE (all degree-3 identical, all degree-4
    identical, ...). Replaces the per-angle clustering with one cluster per degree whose `dirs:` spec is a
    maximally-symmetric arrangement (120-deg Y / tetrahedron / trigonal-bipyramid / Fibonacci sphere).
    The arms bend to reconcile these fixed junction seams with the true (scaled) centerlines.
  * SIZE fixed   -> every junction node radius set to FIXED_R (uniform junction size, independent of local
    vessel radius), and every edge radius set to FIXED_R (uniform tube radius). Junctions no longer shrink
    with the vessel taper, so they are all the same size and clearly separated after the spread.
  * COORDS scaled -> all node positions and all edge centerlines multiplied by SCALE (default 10) so the
    whole network spreads out and junctions/arms do not crowd or overlap.

Run:  bash python/run_py.sh python/vessels_graph_transform.py [in.graph] [out.graph] [scale] [fixed_R]
Defaults: in=data/vmtk/vessels.graph  out=data/vmtk/vessels_fixed.graph  scale=10  fixed_R=auto
(auto = 0.15 * the shortest node-to-node chord AFTER scaling, so tubes stay well inside the gaps).
Also writes <out>_connectivity.png.
"""
import math
import os
import sys

import numpy as np


# --------------------------------------------------------------------------------------------------
def parse_graph(path):
    with open(path) as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    it = iter(lines)
    def nxt(): return next(it).split()
    ncl = int(nxt()[1])
    for _ in range(ncl):
        nxt()
    nn = int(nxt()[1])
    nodes = []
    for _ in range(nn):
        t = nxt()
        nodes.append(dict(id=int(t[0]), x=float(t[1]), y=float(t[2]), z=float(t[3]),
                          deg=int(t[4]), typ=int(t[5]), r=float(t[6]), cl=int(t[7])))
    ne = int(nxt()[1])
    edges = []
    for _ in range(ne):
        t = nxt()
        n0, n1, np_ = int(t[1]), int(t[2]), int(t[5])
        pts = np.array([list(map(float, nxt()[:4])) for _ in range(np_)])   # x y z r
        edges.append(dict(n0=n0, n1=n1, pts=pts))
    return nodes, edges


# --------------------------------------------------------------------------------------------------
def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n > 1e-12 else v


def dir_at(pts, at_start, window):
    """Unit direction of an edge leaving a node, over a short arc `window` (in the ORIGINAL, unscaled
    coords -- angles are scale-invariant). `pts` is the edge centerline (Nx3); at_start picks which end."""
    seq = pts if at_start else pts[::-1]
    base = seq[0]
    acc = 0.0
    tip = seq[-1]
    for k in range(1, len(seq)):
        acc += float(np.linalg.norm(seq[k] - seq[k - 1]))
        if acc >= window:
            tip = seq[k]
            break
    return unit(tuple(tip - base))


def junction_true_dirs(nodes, edges, window):
    """Per-junction list of TRUE incident edge unit directions (exact vessel branch angles), measured over
    the SAME short arc `window` (in ORIGINAL coords) that the C++ driver uses when it re-measures edge
    directions for placement (driver window = 1.5*node.radius in SCALED coords => window = 1.5*fixedR/scale
    here). Matching windows makes the saved canonical dirs equal the driver's placement targets, so the
    Kabsch placement residual is ~0 and the junction holes point exactly along the arms."""
    inc = {nd['id']: [] for nd in nodes}
    for e in edges:
        inc[e['n0']].append((e, True))
        inc[e['n1']].append((e, False))
    dirs = {}
    for nd in nodes:
        if nd['typ'] != 0:
            continue
        ds = [dir_at(e['pts'][:, :3], at_start, window) for (e, at_start) in inc[nd['id']]]
        dirs[nd['id']] = ds
    return dirs


# --------------------------------------------------------------------------------------------------
def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/vmtk/vessels.graph"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/vmtk/vessels_fixed.graph"
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    fixedR = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0

    nodes, edges = parse_graph(inp)
    P = {nd['id']: np.array([nd['x'], nd['y'], nd['z']]) for nd in nodes}

    # auto fixed radius = 0.15 * shortest node-node chord AFTER scaling
    if fixedR <= 0:
        dmin = min(np.linalg.norm(P[e['n0']] - P[e['n1']]) for e in edges if e['n0'] != e['n1'])
        fixedR = 0.15 * dmin * scale
    print("scale=%g  fixed junction/tube radius=%.4f" % (scale, fixedR))

    # ONE cluster PER JUNCTION, using its TRUE incident branch directions (angles unchanged -- only the
    # SIZE is fixed and the coordinates are scaled). No symmetric substitution, no angle approximation.
    win = 1.5 * fixedR / scale                  # match the driver's placement window (1.5*radius, scaled)
    tdirs = junction_true_dirs(nodes, edges, win)
    jids = [nd['id'] for nd in nodes if nd['typ'] == 0]
    id2cid = {jid: i for i, jid in enumerate(jids)}
    print("%d junctions -> %d true-angle clusters (one per junction; SIZE fixed, coords x%g)"
          % (len(jids), len(jids), scale))

    with open(out, "w") as f:
        f.write("# vessels-vmtk graph v1   (world coords)  [true-angle shape, FIXED size, coords x%g]\n" % scale)
        f.write("NCLUSTER %d\n" % len(jids))
        f.write("# cid degree spec\n")
        for jid in jids:
            d = tdirs[jid]
            spec = "dirs:" + ";".join("%.9f,%.9f,%.9f" % v for v in d)
            f.write("%d %d %s\n" % (id2cid[jid], len(d), spec))
        f.write("NNODE %d\n" % len(nodes))
        f.write("# id x y z degree type radius cluster\n")
        for nd in nodes:
            p = P[nd['id']] * scale
            if nd['typ'] == 0:                      # junction: fixed size, its own true-angle cluster
                r, cid = fixedR, id2cid[nd['id']]
            else:                                   # cap: fixed tube radius, no cluster
                r, cid = fixedR, -1
            f.write("%d %.9f %.9f %.9f %d %d %.9f %d\n"
                    % (nd['id'], p[0], p[1], p[2], nd['deg'], nd['typ'], r, cid))
        f.write("NEDGE %d\n" % len(edges))
        f.write("# id n0 n1 r0 r1 npts ; then npts lines: x y z r\n")
        for ei, e in enumerate(edges):
            pts = e['pts'].copy()
            pts[:, :3] *= scale                     # scale centerline coords
            pts[:, 3] = fixedR                      # uniform tube radius
            f.write("%d %d %d %.9f %.9f %d\n" % (ei, e['n0'], e['n1'], fixedR, fixedR, len(pts)))
            for k in range(len(pts)):
                f.write("%.9f %.9f %.9f %.9f\n" % (pts[k, 0], pts[k, 1], pts[k, 2], pts[k, 3]))
    print("wrote", out)

    # connectivity png of the NEW config
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        isj = np.array([nd['typ'] == 0 for nd in nodes])
        Q = np.array([P[nd['id']] * scale for nd in nodes])
        fig, ax = plt.subplots(1, 2, figsize=(22, 11))
        for a, (i, j, t) in zip(ax, [(0, 1, "x-y"), (0, 2, "x-z")]):
            for e in edges:
                interior = nodes[e['n0']]['typ'] == 0 and nodes[e['n1']]['typ'] == 0
                cl = e['pts'][:, :3] * scale
                a.plot(cl[:, i], cl[:, j], '-', color="#B03A2E" if interior else "#8A96A3",
                       lw=1.0 if interior else 0.6, alpha=0.8)
            a.scatter(Q[isj, i], Q[isj, j], s=20, c="#C0392B", zorder=3)
            a.scatter(Q[~isj, i], Q[~isj, j], s=10, c="#3E7CAE", zorder=2)
            a.set_aspect("equal"); a.set_title("fixed/scaled connectivity %s" % t)
        png = os.path.splitext(out)[0] + "_connectivity.png"
        fig.tight_layout(); fig.savefig(png, dpi=140); print("wrote", png)
    except Exception as ex:
        print("(png skipped: %s)" % ex)


if __name__ == "__main__":
    main()
