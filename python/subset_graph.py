#!/usr/bin/env python3
"""Extract a valid SUBNET .graph from a vessels-vmtk network graph.

Usage:
    python3 python/subset_graph.py <seed-node-ids-csv> <out.graph> [--hops N] [--graph PATH]

<seed-node-ids-csv>  comma-separated node ids to seed the subnet on (junctions OR caps/leaves;
                     a leaf seed pulls in the junction on the other end of its edge).
--hops N             grow the kept-junction set by N hops along interior (junction-junction)
                     edges, to include healthy neighbours as controls (default 0).
--graph PATH         source graph (default data/vmtk/vessels_fixed_n298.graph).

The subnet keeps, for every KEPT junction, ALL its incident edges (so node degree == cluster
arm_dir count, which the mesher asserts). Any far endpoint that would otherwise be shared by more
than one kept junction is itself promoted to a kept junction (closure), so every remaining far
endpoint appears in exactly ONE included edge and becomes a valid degree-1 CAP. Nodes and clusters
are re-indexed to a contiguous 0..N-1; cluster specs and edge centerline points are copied
byte-for-byte (no float re-serialization), so a preset arm bundle stays bit-identical.
"""
import sys, os

def read_lines(path):
    """Return the significant (non-blank, non-#) lines of the graph file, in order."""
    out = []
    with open(path) as f:
        for l in f:
            s = l.strip()
            if not s or s.startswith('#'):
                continue
            out.append(l.rstrip('\n'))
    return out

def parse(path):
    L = read_lines(path); i = 0
    def head(key):
        nonlocal i
        tok = L[i].split(); i += 1
        assert tok[0] == key, f"expected {key} got {tok[0]}"
        return int(tok[1])
    ncl = head("NCLUSTER")
    clusters = {}                       # cid -> (deg, spec_string)
    for _ in range(ncl):
        p = L[i].split(); i += 1
        cid, deg, spec = int(p[0]), int(p[1]), p[2]
        clusters[cid] = (deg, spec)
    nn = head("NNODE")
    nodes = {}                          # id -> dict
    for _ in range(nn):
        p = L[i].split(); i += 1
        nid = int(p[0])
        nodes[nid] = dict(raw=p, x=p[1], y=p[2], z=p[3], deg=int(p[4]),
                          typ=int(p[5]), r=p[6], cl=int(p[7]))
    ne = head("NEDGE")
    edges = {}                          # id -> dict (with verbatim point lines)
    for _ in range(ne):
        p = L[i].split(); i += 1
        eid, n0, n1, r0, r1, npt = int(p[0]), int(p[1]), int(p[2]), p[3], p[4], int(p[5])
        pts = L[i:i+npt]; i += npt
        edges[eid] = dict(n0=n0, n1=n1, r0=r0, r1=r1, npt=npt, pts=pts)
    return clusters, nodes, edges

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__); sys.exit(1)
    hops = 0; graph = "data/vmtk/vessels_fixed_n298.graph"
    seeds_csv, out = args[0], args[1]
    j = 2
    while j < len(args):
        if args[j] == "--hops": hops = int(args[j+1]); j += 2
        elif args[j] == "--graph": graph = args[j+1]; j += 2
        else: j += 1
    seeds = [int(s) for s in seeds_csv.split(',') if s.strip() != ""]

    clusters, nodes, edges = parse(graph)
    # incidence: node -> list of edge ids
    inc = {nid: [] for nid in nodes}
    for eid, e in edges.items():
        inc[e['n0']].append(eid); inc[e['n1']].append(eid)

    def is_junc(nid): return nodes[nid]['typ'] == 0
    def other(eid, nid): e = edges[eid]; return e['n1'] if e['n0'] == nid else e['n0']

    # seed -> kept junctions (a leaf seed pulls in its owner junction)
    kept = set()
    for s in seeds:
        if is_junc(s): kept.add(s)
        else:
            for eid in inc[s]:
                o = other(eid, s)
                if is_junc(o): kept.add(o)
    # hop expansion along interior (junction-junction) edges
    for _ in range(hops):
        frontier = set()
        for nid in kept:
            for eid in inc[nid]:
                o = other(eid, nid)
                if is_junc(o): frontier.add(o)
        kept |= frontier
    # CLOSURE: promote any non-kept node shared by >1 kept-junction edge to a kept junction,
    # so every remaining far endpoint sits in exactly one included edge (a valid degree-1 cap).
    while True:
        incident = set()
        for nid in kept:
            incident |= set(inc[nid])
        share = {}
        for eid in incident:
            for endp in (edges[eid]['n0'], edges[eid]['n1']):
                if endp not in kept:
                    share[endp] = share.get(endp, 0) + 1
        promote = [n for n, c in share.items() if c > 1 and is_junc(n)]
        if not promote:
            break
        kept |= set(promote)

    incident = sorted({eid for nid in kept for eid in inc[nid]})
    # every node that appears in an included edge
    used_nodes = set()
    for eid in incident:
        used_nodes.add(edges[eid]['n0']); used_nodes.add(edges[eid]['n1'])
    caps = sorted(used_nodes - kept)          # far endpoints -> caps (leaves)
    kept_sorted = sorted(kept)

    # re-index clusters (only those of kept junctions) and nodes
    new_cid = {}
    for cid in sorted({nodes[n]['cl'] for n in kept_sorted}):
        new_cid[cid] = len(new_cid)
    new_nid = {}
    for nid in kept_sorted + caps:            # junctions first, then caps (order is arbitrary but stable)
        new_nid[nid] = len(new_nid)

    # subnet degree of each kept junction == number of its incident edges (all kept)
    with open(out, 'w') as f:
        f.write(f"# subnet of {os.path.basename(graph)}  seeds={seeds_csv} hops={hops}\n")
        f.write(f"NCLUSTER {len(new_cid)}\n")
        f.write("# cid degree spec\n")
        for cid in sorted(new_cid, key=lambda c: new_cid[c]):
            deg, spec = clusters[cid]
            f.write(f"{new_cid[cid]} {deg} {spec}\n")
        f.write(f"NNODE {len(new_nid)}\n")
        f.write("# id x y z degree typ radius cluster\n")
        deg_sub = {n: sum(1 for eid in inc[n] if eid in set(incident)) for n in used_nodes}
        for nid in sorted(new_nid, key=lambda n: new_nid[n]):
            nd = nodes[nid]
            if nid in kept:
                f.write(f"{new_nid[nid]} {nd['x']} {nd['y']} {nd['z']} {deg_sub[nid]} 0 {nd['r']} {new_cid[nd['cl']]}\n")
            else:                              # cap / leaf: typ=1, degree 1, no cluster
                f.write(f"{new_nid[nid]} {nd['x']} {nd['y']} {nd['z']} 1 1 {nd['r']} -1\n")
        f.write(f"NEDGE {len(incident)}\n")
        f.write("# id n0 n1 r0 r1 npt ; then npt lines: x y z r\n")
        for k, eid in enumerate(incident):
            e = edges[eid]
            f.write(f"{k} {new_nid[e['n0']]} {new_nid[e['n1']]} {e['r0']} {e['r1']} {e['npt']}\n")
            for pl in e['pts']:
                f.write(pl + "\n")
    njunc = len(kept_sorted)
    print(f"[subset] kept {njunc} junctions, {len(caps)} caps, {len(incident)} edges "
          f"({sum(1 for eid in incident if edges[eid]['n0'] in kept and edges[eid]['n1'] in kept)} interior) -> {out}")

if __name__ == "__main__":
    main()
