#!/usr/bin/env python3
"""Per-arm geometric census of a network bundle, read from the ACTUAL meshed Chebyshev nodes.

Usage:  python3 python/census_arms.py <bundle-prefix> [--sep S] [--top N]

Reads every <prefix>-jNNN.arms and, per owned arm, reconstructs the meshed centerline (the CSBQ
slender panel Chebyshev nodes, radius, orientation e1) and reports the metrics that actually predict
a self-crossing tube (the "open gaps / overlapping surface"):

  reverse   max angle (deg) between consecutive node-chord tangents -- a >90 deg step is the Hermite
            corner DOUBLING BACK (the "hairpin"/overshoot), visible on the meshed nodes, not just base().
  rho/r     min curvature radius / tube radius on a UNIFORM-ARCLENGTH resample (cheb clustering at panel
            ends over-tightens the raw 3-pt circumradius -- always resample; see network-greens memory).
  dist/r    min geometric surface gap (|Pi-Pj|-ri-rj)/mean_r over node pairs separated by >SEP panel
            arclengths -- <0 means the tube overlaps ITSELF (a fold or a doubled-back hairpin). THE
            decisive metric.

Flags an arm BROKEN if reverse>90 or rho/r<1 or dist/r<0. --sep sets the self-contact arclength mask
in panel-lengths (default 1.5). --top ranks the worst N (default 25).
"""
import sys, glob, math

def parse_bundle(path):
    arms = []
    with open(path) as f:
        L = [l for l in f if l.strip() and not l.lstrip().startswith('#')]
    it = iter(L)
    order, narm, cheb, fourier = it.__next__().split()[:4]
    narm, cheb = int(narm), int(cheb)
    for _ in range(narm):
        other, iscap, npan = it.__next__().split()[:3]
        npan = int(npan)
        pts = []           # (x,y,z,r) per node, concatenated across panels
        pan_nnode = []
        for _ in range(npan):
            eo, fo = it.__next__().split()[:2]; eo = int(eo)
            pan_nnode.append(eo)
            for _ in range(eo):
                v = it.__next__().split()
                pts.append((float(v[0]), float(v[1]), float(v[2]), float(v[3])))
        arms.append(dict(other=int(other), iscap=int(iscap), npan=npan,
                         pts=pts, pan_nnode=pan_nnode))
    return arms

def d3(a, b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def analyse(arm, sep):
    P = arm['pts']; n = len(P)
    if n < 4: return None
    # cumulative arclength (node chords)
    s = [0.0]*n
    for i in range(1, n): s[i] = s[i-1] + d3(P[i], P[i-1])
    Ltot = s[-1]
    rmean = sum(p[3] for p in P)/n
    mean_pan_arc = Ltot/max(1, arm['npan'])
    # reverse: max angle between consecutive chord tangents
    def unit(u):
        m = math.sqrt(u[0]**2+u[1]**2+u[2]**2); return (u[0]/m,u[1]/m,u[2]/m) if m>1e-30 else None
    ch = []
    for i in range(1, n):
        u = unit((P[i][0]-P[i-1][0], P[i][1]-P[i-1][1], P[i][2]-P[i-1][2]))
        ch.append((u, d3(P[i], P[i-1])))
    thr = 0.02*mean_pan_arc          # skip ~coincident chords (roundoff direction)
    reverse = 0.0; prev = None
    for u, l in ch:
        if u is None or l < thr: continue
        if prev is not None:
            c = max(-1.0, min(1.0, u[0]*prev[0]+u[1]*prev[1]+u[2]*prev[2]))
            reverse = max(reverse, math.degrees(math.acos(c)))
        prev = u
    # rho/r on a uniform-arclength resample
    M = max(100, 4*arm['npan'])
    xs = []
    for k in range(M+1):
        sk = Ltot*k/M
        # locate segment
        lo, hi = 0, n-1
        while lo+1 < hi:
            mid = (lo+hi)//2
            if s[mid] <= sk: lo = mid
            else: hi = mid
        seg = s[hi]-s[lo]; w = (sk-s[lo])/seg if seg > 1e-30 else 0.0
        pt = tuple(P[lo][d]*(1-w)+P[hi][d]*w for d in range(3)) + \
             (P[lo][3]*(1-w)+P[hi][3]*w,)
        xs.append(pt)
    rho_r = 1e30; rho_loc = 0.0
    for i in range(1, M):
        a, b, c = xs[i-1], xs[i], xs[i+1]
        ab, bc, ca = d3(a,b), d3(b,c), d3(c,a)
        # triangle area via cross product
        v1 = (b[0]-a[0], b[1]-a[1], b[2]-a[2]); v2 = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        cr = (v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0])
        area2 = math.sqrt(cr[0]**2+cr[1]**2+cr[2]**2)
        if area2 < 1e-30: continue
        rho = ab*bc*ca/(2*area2)
        if rho/max(1e-30, b[3]) < rho_r: rho_r = rho/max(1e-30, b[3]); rho_loc = i/M
    # dist/r self-contact over arclength-separated node pairs
    # self-contact mask (PER PAIR): a pair registers as overlap only if their arclength separation exceeds
    # BOTH sep panel-lengths AND ~a LOCAL tube diameter 1.3*(ri+rj). The local-diameter floor (not a global
    # 2.5*rmean) is essential on TAPERED arms: near the fat end two nearly-parallel sections a bit over
    # rmean apart falsely read as overlap (tube-diameter adjacency), which the per-pair radii exclude.
    dmin = 1e30; dloc = (0.0, 0.0)
    for i in range(n):
        for j in range(i+1, n):
            if s[j]-s[i] < max(sep*mean_pan_arc, 1.3*(P[i][3]+P[j][3])): continue
            g = (d3(P[i], P[j]) - P[i][3] - P[j][3])/max(1e-30, rmean)
            if g < dmin: dmin = g; dloc = (s[i]/Ltot, s[j]/Ltot)
    return dict(reverse=reverse, rho_r=rho_r, rho_loc=rho_loc, dist_r=dmin, dloc=dloc,
                npan=arm['npan'], L=Ltot, rmean=rmean)

def main():
    if len(sys.argv) < 2: print(__doc__); sys.exit(1)
    prefix = sys.argv[1]; sep = 1.5; top = 25
    a = 2
    while a < len(sys.argv):
        if sys.argv[a] == '--sep': sep = float(sys.argv[a+1]); a += 2
        elif sys.argv[a] == '--top': top = int(sys.argv[a+1]); a += 2
        else: a += 1
    rows = []
    for path in sorted(glob.glob(prefix + "-j*.arms")):
        jid = path.split("-j")[-1].split(".arms")[0]
        for k, arm in enumerate(parse_bundle(path)):
            m = analyse(arm, sep)
            if m is None: continue
            m['owner'] = jid; m['other'] = arm['other']; m['cap'] = arm['iscap']
            rows.append(m)
    if not rows: print("no arms found for prefix", prefix); return
    broken = [r for r in rows if r['reverse'] > 90 or r['rho_r'] < 1 or r['dist_r'] < 0]
    print(f"[census] {len(rows)} arms ; BROKEN (reverse>90 | rho/r<1 | dist/r<0): {len(broken)}"
          f"  ({sum(1 for r in broken if r['cap'])} caps)")
    print(f"  {'owner->other':>14} {'cap':>3} {'npan':>5} {'reverse':>8} {'rho/r':>8} {'@loc':>6} {'dist/r':>8} {'@loc(i,j)':>14}")
    key = lambda r: (min(r['dist_r'], 1e9), r['rho_r'], -r['reverse'])
    for r in sorted(rows, key=key)[:top]:
        flag = ' '.join(f for f, c in [('REV', r['reverse']>90), ('FOLD', r['rho_r']<1), ('OVLP', r['dist_r']<0)] if c)
        print(f"  j{r['owner']}->{r['other']:<4} {'Y' if r['cap'] else ' ':>3} {r['npan']:>5} "
              f"{r['reverse']:8.1f} {r['rho_r']:8.3f} {r['rho_loc']:6.2f} {r['dist_r']:8.3f} "
              f"  {r['dloc'][0]:.2f},{r['dloc'][1]:.2f}  {flag}")
    # summary stats
    import statistics as st
    print(f"  min dist/r = {min(r['dist_r'] for r in rows):.3f} ; "
          f"min rho/r = {min(r['rho_r'] for r in rows):.3f} ; "
          f"max reverse = {max(r['reverse'] for r in rows):.1f} deg")

if __name__ == "__main__":
    main()
