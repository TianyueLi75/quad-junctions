#!/usr/bin/env python3
"""build_n298_graph.py -- productionize the de-kinked j298 junction into the network graph.

The in-network junction j298 (node 298, cluster 150; the 173.7-deg near-antiparallel 302/303 pair)
floored network watertightness at ~1.4e-4 because its 297-298 centerline was KINKED. The de-kinked
centerline lives in `data/vmtk/vessels_fixed_j298fix.graph`, but that file still carries the OLD
NCLUSTER `dirs:` for cluster 150 -- the junction shape was never recomputed to match the new arm
angle. This script closes that gap on the GRAPH-BUILDER side (not the mesher): it recomputes every
incident-arm leaving tangent at each *junction* node straight from the (de-kinked) centerlines, using
the SAME arc-window tangent the vmtk grapher uses (`edge_dir_at`, window = 1.5 * node radius), and
rewrites the affected clusters' `dirs:` spec. Only clusters whose centerline actually changed move --
for the j298fix graph that is exactly cluster 150 (its 297 dir 0.4668,-0.0609,0.8823 ->
0.4998,0.1449,0.8539, the de-kinked leaving tangent), everything else byte-stable.

The result is a graph whose junction SHAPE and centerline agree, so `bifurc-network-assemble` places a
junction that matches the arm it actually meshes (the C variant from the network-blend study).

Usage:
    python python/build_n298_graph.py [in.graph] [out.graph]
Defaults: in = data/vmtk/vessels_fixed_j298fix.graph
          out = data/vmtk/vessels_fixed_n298.graph
"""
import os
import sys
import numpy as np


def read_lines(path):
    """Return the raw file lines split into (comments-preserved) tokens plus a flat data-token stream."""
    with open(path) as f:
        return f.readlines()


def parse(path):
    with open(path) as f:
        raw = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    it = iter(raw)

    def nxt():
        return next(it).split()

    ncl = int(nxt()[1])
    clusters = []
    for _ in range(ncl):
        t = nxt()
        clusters.append(dict(cid=int(t[0]), deg=int(t[1]), spec=t[2]))
    nn = int(nxt()[1])
    nodes = {}
    for _ in range(nn):
        t = nxt()
        nodes[int(t[0])] = dict(pos=np.array(list(map(float, t[1:4]))), deg=int(t[4]),
                                typ=int(t[5]), r=float(t[6]), cl=int(t[7]))
    ne = int(nxt()[1])
    edges = []
    for _ in range(ne):
        t = nxt()
        n0, n1, npt = int(t[1]), int(t[2]), int(t[5])
        pts = np.array([list(map(float, nxt()[:4])) for _ in range(npt)])
        edges.append(dict(n0=n0, n1=n1, pts=pts))
    return clusters, nodes, edges


def edge_dir_at(edge, node_id, window):
    """Unit tangent of `edge` leaving `node_id`, averaged over an arc `window` (world length).

    Verbatim port of vessels_vmtk_graph.edge_dir_at -- the same rule that produced the original dirs.
    """
    pts = edge["pts"][:, :3]
    if edge["n1"] == node_id:
        pts = pts[::-1]
    base = pts[0]
    acc, tip = 0.0, pts[-1]
    for k in range(1, len(pts)):
        acc += np.linalg.norm(pts[k] - pts[k - 1])
        if acc >= window:
            tip = pts[k]
            break
    d = tip - base
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def spec_from_dirs(dirs):
    return "dirs:" + ";".join("%.9f,%.9f,%.9f" % (v[0], v[1], v[2]) for v in dirs)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/vmtk/vessels_fixed_j298fix.graph"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/vmtk/vessels_fixed_n298.graph"

    clusters, nodes, edges = parse(src)
    # incident-edge index per node
    inc = {i: [] for i in nodes}
    for k, e in enumerate(edges):
        inc[e["n0"]].append(k)
        inc[e["n1"]].append(k)

    # For each cluster, find the (unique here) junction node carrying it, and recompute its dirs from
    # the current (de-kinked) centerlines with the grapher's arc-window tangent. The dir ORDER must match
    # the incident-edge order the mesher reads: NetworkGraph stores dirs in node-incident-edge order, so
    # we walk inc[node] in file order (== the order the arms were emitted at graph-build time).
    changed = []
    for c in clusters:
        members = [i for i, d in nodes.items() if d["cl"] == c["cid"] and d["typ"] == 0]
        if len(members) != 1:
            # multi-member clusters would need the medoid; none in this graph, so flag and skip.
            if members:
                print("  [skip] cluster %d has %d members %s (needs medoid) -- left unchanged"
                      % (c["cid"], len(members), members))
            continue
        node = members[0]
        win = 1.5 * nodes[node]["r"]
        dirs = [edge_dir_at(edges[k], node, win) for k in inc[node]]
        newspec = spec_from_dirs(dirs)
        if newspec != c["spec"]:
            changed.append((c["cid"], node, c["spec"], newspec))
            c["spec"] = newspec

    print("recomputed junction dirs from de-kinked centerlines: %d cluster(s) changed" % len(changed))
    for cid, node, old, new in changed:
        print("  cluster %d (node %d):" % (cid, node))
        print("    old %s" % old)
        print("    new %s" % new)

    # Re-emit: copy the source file verbatim, replacing ONLY the changed NCLUSTER spec lines. This keeps
    # the header comment, all node rows, and all 337 edge point-blocks byte-identical.
    lines = read_lines(src)
    # locate the cluster block: after the "NCLUSTER <n>" line, the next <n> non-comment lines are specs.
    cid_to_spec = {cid: new for cid, _, _, new in changed}
    out_lines, i = [], 0
    while i < len(lines):
        ln = lines[i]
        out_lines.append(ln)
        if ln.split()[:1] == ["NCLUSTER"]:
            ncl = int(ln.split()[1])
            i += 1
            done = 0
            while done < ncl and i < len(lines):
                cl_ln = lines[i]
                if cl_ln.strip() and not cl_ln.lstrip().startswith("#"):
                    cid = int(cl_ln.split()[0])
                    if cid in cid_to_spec:
                        deg = cl_ln.split()[1]
                        out_lines.append("%d %s %s\n" % (cid, deg, cid_to_spec[cid]))
                    else:
                        out_lines.append(cl_ln)
                    done += 1
                else:
                    out_lines.append(cl_ln)  # preserve comment lines inside the block
                i += 1
            continue
        i += 1

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.writelines(out_lines)
    print("wrote", out)


if __name__ == "__main__":
    main()
