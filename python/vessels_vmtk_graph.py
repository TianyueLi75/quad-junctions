#!/usr/bin/env python3
"""
STL -> vmtk centerline skeleton -> bifurcation GRAPH -> clustered `.graph` file for the C++
network-assembly driver (`bin/bifurc-network-assemble`).

Pipeline (all headless, no interactive seed picking):
  1. read the STL surface, triangulate.
  2. vmtkNetworkExtraction needs the surface to have >=1 OPENING; a watertight vessel STL has none,
     so we punch ONE hole at an extreme (max-x) vessel tip -- a real degree-1 cap -- which seeds the
     advancement; the extractor then traverses the whole closed tree from that single opening.
  3. vmtkNetworkExtraction -> a network vtkPolyData: polyline cells + a per-point `Radius` array
     (max inscribed sphere radius).
  4. build the abstract graph: cluster the polyline ENDPOINTS into graph nodes (proximity-merge),
     count incident polylines = node DEGREE; degree>=3 = junction, degree==1 = cap, degree==2 chains
     are dissolved into their through-edge (an "arm" then spans junction<->junction / junction<->cap).
  5. per junction: incident unit directions from a short arc window of each incident edge (robust to
     the raw skeleton kink at the node); per edge: resampled world centerline + radius profile.
  6. cluster the junctions by (degree, sorted pairwise-angle signature) with a tiny numpy Lloyd loop
     (no sklearn dep); the MEDOID node's incident dirs become that cluster's canonical `dirs:` spec.
  7. emit `<prefix>.graph` (plain text, parsed by include/quad_junctions/gen_network_geom.hpp) + a
     top-down QC scatter PNG (compare against the STL, like build_vessels_topology.py's plot).

Run (inside the vmtk venv -- NOT python/run_py.sh, which lacks vmtk):
    module load python/3.12.9 && source ~/venvs/vmtk/bin/activate
    python python/vessels_vmtk_graph.py [stl] [out-prefix] [advratance_ratio] [cluster_tol_deg]
Defaults: stl=vessels_quad_smooth_0.5.stl, prefix=data/vmtk/vessels, adv=1.05, cluster_tol_deg=12.
"""
import os
import sys
import time

import numpy as np
import vtk
from vmtk import vmtkscripts


# ---------------------------------------------------------------------------------------------------
# 1-2.  read + open the surface (punch one hole if closed) so network extraction can seed.
# ---------------------------------------------------------------------------------------------------
def load_and_open(stl_path):
    r = vmtkscripts.vmtkSurfaceReader(); r.InputFileName = stl_path; r.Execute()
    tri = vmtkscripts.vmtkSurfaceTriangle(); tri.Surface = r.Surface; tri.Execute()
    s = tri.Surface
    fe = vtk.vtkFeatureEdges(); fe.SetInputData(s)
    fe.BoundaryEdgesOn(); fe.FeatureEdgesOff(); fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff(); fe.Update()
    nbound = fe.GetOutput().GetNumberOfCells()
    b = s.GetBounds()
    diag = np.linalg.norm(np.array(b[1::2]) - np.array(b[0::2]))
    print("  boundary edges: %d   bbox-diag %.3f" % (nbound, diag), flush=True)
    if nbound > 0:
        return s, diag                                  # already open -> extractor seeds itself
    # closed: delete a small cell patch at the max-x tip to open exactly one hole.
    N = s.GetNumberOfPoints()
    xyz = np.asarray([s.GetPoint(i) for i in range(N)])
    tip = xyz[np.argmax(xyz[:, 0])]
    R = 0.02 * diag
    s.BuildLinks()
    s2 = vtk.vtkPolyData(); s2.DeepCopy(s); s2.BuildLinks()
    ndel = 0
    for c in range(s.GetNumberOfCells()):
        cell = s.GetCell(c)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        cc = xyz[ids].mean(axis=0)
        if np.linalg.norm(cc - tip) < R:
            s2.DeleteCell(c); ndel += 1
    s2.RemoveDeletedCells()
    cl = vtk.vtkCleanPolyData(); cl.SetInputData(s2); cl.Update()
    print("  punched 1 hole at tip [%.2f %.2f %.2f] (deleted %d cells)" % (tip[0], tip[1], tip[2], ndel), flush=True)
    return cl.GetOutput(), diag


# ---------------------------------------------------------------------------------------------------
# 3.  vmtk network extraction.
# ---------------------------------------------------------------------------------------------------
def extract_network(surface, adv):
    ne = vmtkscripts.vmtkNetworkExtraction()
    ne.Surface = surface
    ne.AdvancementRatio = adv
    ne.Execute()
    net = ne.Network
    if net.GetNumberOfCells() == 0:
        raise RuntimeError("empty network -- surface may still be closed or too coarse")
    return net


def cell_polylines(net):
    """Return list of (coords[n,3], radius[n]) per network polyline cell (ordered)."""
    rad = net.GetPointData().GetArray("Radius")
    out = []
    for c in range(net.GetNumberOfCells()):
        cell = net.GetCell(c)
        n = cell.GetNumberOfPoints()
        ids = [cell.GetPointId(j) for j in range(n)]
        P = np.asarray([net.GetPoint(i) for i in ids])
        Rr = np.asarray([rad.GetTuple1(i) for i in ids]) if rad else np.full(n, np.nan)
        out.append((P, Rr))
    return out


# ---------------------------------------------------------------------------------------------------
# 4.  endpoints -> graph nodes (proximity merge), edges carry the polyline + radius profile.
# ---------------------------------------------------------------------------------------------------
class Node:
    __slots__ = ("pos", "rads", "edges")
    def __init__(self, pos):
        self.pos = np.array(pos, float); self.rads = []; self.edges = []

class Edge:
    __slots__ = ("n0", "n1", "pts", "rad")
    def __init__(self, n0, n1, pts, rad):
        self.n0 = n0; self.n1 = n1; self.pts = pts; self.rad = rad


def build_graph(polylines, merge_tol):
    nodes = []
    def find_or_add(p, rr):
        for i, nd in enumerate(nodes):
            if np.linalg.norm(nd.pos - p) < merge_tol:
                nd.rads.append(rr); return i
        nodes.append(Node(p)); nodes[-1].rads.append(rr); return len(nodes) - 1
    edges = []
    for (P, R) in polylines:
        if len(P) < 2:
            continue
        i0 = find_or_add(P[0], R[0]); i1 = find_or_add(P[-1], R[-1])
        if i0 == i1:
            continue                                    # a loop segment -> skip (rare)
        e = Edge(i0, i1, P, R); edges.append(e)
        nodes[i0].edges.append(len(edges) - 1); nodes[i1].edges.append(len(edges) - 1)
    return nodes, edges


def drop_isolated(nodes, edges):
    """Drop degree-0 nodes (merge artifacts) and remap node ids so ids stay contiguous."""
    keep = [i for i, nd in enumerate(nodes) if len(nd.edges) > 0]
    remap = {old: new for new, old in enumerate(keep)}
    nodes2 = [nodes[i] for i in keep]
    for e in edges:
        e.n0 = remap[e.n0]; e.n1 = remap[e.n1]
    return nodes2, edges


def dissolve_degree2(nodes, edges):
    """Merge chains through degree-2 nodes into single edges (an arm spans junction/cap to junction/cap)."""
    alive_e = [True] * len(edges)
    def deg(i): return sum(alive_e[e] for e in nodes[i].edges)
    changed = True
    while changed:
        changed = False
        for i, nd in enumerate(nodes):
            live = [e for e in nd.edges if alive_e[e]]
            if len(live) != 2:
                continue
            ea, eb = edges[live[0]], edges[live[1]]
            # orient both polylines to START at node i, then splice (drop the duplicated node point).
            def oriented(e):
                pts, rad = e.pts, e.rad
                if e.n0 == i:
                    return pts, rad, e.n1
                return pts[::-1], rad[::-1], e.n0
            pa, ra, oa = oriented(ea)
            pb, rb, ob = oriented(eb)
            if oa == ob:
                continue                                # would form a loop; leave as-is
            newpts = np.vstack([pa[::-1], pb[1:]])       # ...oa -> i -> ob...
            newrad = np.concatenate([ra[::-1], rb[1:]])
            ne = Edge(oa, ob, newpts, newrad)
            edges.append(ne); alive_e.append(True)
            alive_e[live[0]] = False; alive_e[live[1]] = False
            nodes[oa].edges.append(len(edges) - 1); nodes[ob].edges.append(len(edges) - 1)
            changed = True
    # compact
    keep = [i for i, a in enumerate(alive_e) if a]
    remap = {old: new for new, old in enumerate(keep)}
    edges2 = [edges[i] for i in keep]
    for nd in nodes:
        nd.edges = [remap[e] for e in nd.edges if e in remap]
    return nodes, edges2


# ---------------------------------------------------------------------------------------------------
# 5.  per-junction incident directions (short arc window) + per-node radius.
# ---------------------------------------------------------------------------------------------------
def edge_dir_at(edge, node_id, window):
    """Unit tangent of `edge` leaving `node_id`, averaged over an arc `window` (world length)."""
    pts = edge.pts if edge.n0 == node_id else edge.pts[::-1]
    base = pts[0]
    acc = 0.0; tip = pts[-1]
    for k in range(1, len(pts)):
        acc += np.linalg.norm(pts[k] - pts[k - 1])
        if acc >= window:
            tip = pts[k]; break
    d = tip - base; n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def node_radius(nd):
    return float(np.nanmean(nd.rads)) if nd.rads else 0.0


# ---------------------------------------------------------------------------------------------------
# 6.  cluster junctions by (degree, sorted pairwise-angle signature).  tiny numpy Lloyd, no sklearn.
# ---------------------------------------------------------------------------------------------------
def angle_signature(dirs):
    """Sorted vector of pairwise angles (deg) between incident directions -- rotation invariant."""
    ang = []
    for a in range(len(dirs)):
        for b in range(a + 1, len(dirs)):
            c = float(np.clip(np.dot(dirs[a], dirs[b]), -1, 1))
            ang.append(np.degrees(np.arccos(c)))
    return np.sort(np.array(ang))


def cluster_by_degree(junc_ids, sigs, degrees, tol_deg):
    """Return cluster id per junction + medoid junction per cluster. Cluster within each degree by a
    greedy furthest-point seeding + Lloyd, splitting whenever a member exceeds tol_deg from its center."""
    cid = np.full(len(junc_ids), -1, int)
    medoids = {}
    ncl = 0
    for d in sorted(set(degrees)):
        idx = [i for i in range(len(junc_ids)) if degrees[i] == d]
        S = np.array([sigs[i] for i in idx])            # (m, C(d,2))
        assigned = np.full(len(idx), -1, int)
        centers = []
        for j in range(len(idx)):
            if assigned[j] >= 0:
                continue
            # start a new cluster at j, absorb all within tol
            centers.append(S[j].copy()); k = len(centers) - 1
            for jj in range(len(idx)):
                if assigned[jj] < 0 and np.linalg.norm(S[jj] - centers[k]) <= tol_deg * np.sqrt(len(S[jj])):
                    assigned[jj] = k
            # Lloyd refine center
            mem = [t for t in range(len(idx)) if assigned[t] == k]
            centers[k] = S[mem].mean(axis=0)
        # medoid per local cluster
        for k in range(len(centers)):
            mem = [t for t in range(len(idx)) if assigned[t] == k]
            dists = [np.linalg.norm(S[t] - centers[k]) for t in mem]
            med_local = mem[int(np.argmin(dists))]
            gid = ncl + k
            medoids[gid] = junc_ids[idx[med_local]]
            for t in mem:
                cid[idx[t]] = gid
        ncl += len(centers)
    return cid, medoids, ncl


# ---------------------------------------------------------------------------------------------------
# 7.  emit .graph  +  QC png.
# ---------------------------------------------------------------------------------------------------
def resample_polyline(pts, rad, npts):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)]); total = L[-1]
    if total < 1e-12:
        return pts[:1].repeat(npts, 0), np.repeat(rad[:1], npts)
    tt = np.linspace(0, total, npts)
    out = np.empty((npts, 3)); outr = np.empty(npts)
    for k in range(3):
        out[:, k] = np.interp(tt, L, pts[:, k])
    outr = np.interp(tt, L, rad)
    return out, outr


def write_graph(prefix, nodes, edges, is_junc, dirs_at, cid, medoids, ncl, node_dirs_for_medoid, npts_edge):
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    path = prefix + ".graph"
    with open(path, "w") as f:
        f.write("# vessels-vmtk graph v1   (world coords)\n")
        # clusters: cid degree spec(dirs:...)
        f.write("NCLUSTER %d\n" % ncl)
        f.write("# cid degree spec\n")
        for c in range(ncl):
            mj = medoids[c]
            md = node_dirs_for_medoid[mj]
            spec = "dirs:" + ";".join("%.9f,%.9f,%.9f" % (v[0], v[1], v[2]) for v in md)
            f.write("%d %d %s\n" % (c, len(md), spec))
        # nodes: id x y z degree type(0=junc,1=cap) radius cluster
        f.write("NNODE %d\n" % len(nodes))
        f.write("# id x y z degree type radius cluster\n")
        for i, nd in enumerate(nodes):
            deg = len(nd.edges)
            typ = 0 if is_junc[i] else 1
            f.write("%d %.9f %.9f %.9f %d %d %.9f %d\n"
                    % (i, nd.pos[0], nd.pos[1], nd.pos[2], deg, typ, node_radius(nd),
                       cid[i] if is_junc[i] else -1))
        # edges: id n0 n1 r0 r1 npts  then npts lines: x y z r
        f.write("NEDGE %d\n" % len(edges))
        f.write("# id n0 n1 r0 r1 npts ; then npts lines: x y z r\n")
        for ei, e in enumerate(edges):
            P, R = resample_polyline(e.pts, e.rad, npts_edge)
            f.write("%d %d %d %.9f %.9f %d\n" % (ei, e.n0, e.n1, R[0], R[-1], len(P)))
            for k in range(len(P)):
                f.write("%.9f %.9f %.9f %.9f\n" % (P[k, 0], P[k, 1], P[k, 2], R[k]))
    print("  wrote", path, flush=True)
    return path


def write_qc_png(prefix, nodes, edges, is_junc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print("  (skipping QC png: %s)" % ex, flush=True); return
    fig, ax = plt.subplots(figsize=(10, 9))
    for e in edges:
        ax.plot(e.pts[:, 0], e.pts[:, 1], "-", color="#8A96A3", lw=0.8, zorder=1)
    jx = [nd.pos[0] for i, nd in enumerate(nodes) if is_junc[i]]
    jy = [nd.pos[1] for i, nd in enumerate(nodes) if is_junc[i]]
    cx = [nd.pos[0] for i, nd in enumerate(nodes) if not is_junc[i]]
    cy = [nd.pos[1] for i, nd in enumerate(nodes) if not is_junc[i]]
    ax.scatter(jx, jy, c="#B03A2E", s=28, zorder=3, label="junction (deg>=3)")
    ax.scatter(cx, cy, c="#3E7CAE", s=14, zorder=2, label="cap (deg 1)")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8)
    ax.set_title("vmtk vessels graph (top-down x-y)")
    fig.tight_layout(); fig.savefig(prefix + "_qc.png", dpi=130)
    print("  wrote", prefix + "_qc.png", flush=True)


def main():
    stl = sys.argv[1] if len(sys.argv) > 1 else "vessels_quad_smooth_0.5.stl"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "data/vmtk/vessels"
    adv = float(sys.argv[3]) if len(sys.argv) > 3 else 1.05
    tol_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    t0 = time.time()
    print("[1/6] load + open", stl, flush=True)
    surf, diag = load_and_open(stl)
    print("[2/6] network extraction (adv=%.3f)" % adv, flush=True)
    net = extract_network(surf, adv)
    print("      network cells", net.GetNumberOfCells(), "points", net.GetNumberOfPoints(), flush=True)
    polys = cell_polylines(net)
    print("[3/6] build graph", flush=True)
    merge_tol = max(1e-6, 0.004 * diag)
    nodes, edges = build_graph(polys, merge_tol)
    nodes, edges = dissolve_degree2(nodes, edges)
    nodes, edges = drop_isolated(nodes, edges)          # remove degree-0 orphans, remap ids
    # recompute incidence-degree, classify
    for nd in nodes:
        nd.edges = list(nd.edges)
    degrees_all = [len(nd.edges) for nd in nodes]
    is_junc = [d >= 3 for d in degrees_all]
    njunc = sum(is_junc); ncap = sum(1 for d in degrees_all if d == 1)
    print("      nodes %d  junctions(deg>=3) %d  caps(deg1) %d" % (len(nodes), njunc, ncap), flush=True)
    from collections import Counter
    print("      degree histogram", dict(sorted(Counter(degrees_all).items())), flush=True)
    print("[4/6] incident directions", flush=True)
    node_dirs = {}
    for i, nd in enumerate(nodes):
        if not is_junc[i]:
            continue
        win = max(1.5 * node_radius(nd), 2.0 * merge_tol)
        node_dirs[i] = [edge_dir_at(edges[e], i, win) for e in nd.edges]
    print("[5/6] cluster junctions (tol %.1f deg)" % tol_deg, flush=True)
    junc_ids = [i for i in range(len(nodes)) if is_junc[i]]
    sigs = {i: angle_signature(node_dirs[i]) for i in junc_ids}
    cidj, medoids, ncl = cluster_by_degree(junc_ids, [sigs[i] for i in junc_ids],
                                           [len(node_dirs[i]) for i in junc_ids], tol_deg)
    cid = np.full(len(nodes), -1, int)
    for k, i in enumerate(junc_ids):
        cid[i] = cidj[k]
    print("      %d representative junction clusters" % ncl, flush=True)
    for c in range(ncl):
        members = [junc_ids[k] for k in range(len(junc_ids)) if cidj[k] == c]
        print("        cluster %d  degree %d  members %d  medoid-node %d"
              % (c, len(node_dirs[medoids[c]]), len(members), medoids[c]), flush=True)
    print("[6/6] emit graph + QC png", flush=True)
    write_graph(prefix, nodes, edges, is_junc, node_dirs, cid, medoids, ncl, node_dirs, npts_edge=64)
    write_qc_png(prefix, nodes, edges, is_junc)
    print("done  %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
