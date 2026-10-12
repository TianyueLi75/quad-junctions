#!/usr/bin/env python3
"""Write the interior Green's sources the driver would pick for a bundle (stride over concatenated arm centerline
nodes, 36 = Nlon*Nlat) to a file, for QJ_GREENS_INT_XYZ so two bundles are compared with IDENTICAL sources.
usage: greens_int_sources.py <bundle_prefix> <out.xyz> [M=36]"""
import sys, glob
pref, out = sys.argv[1], sys.argv[2]; M = int(sys.argv[3]) if len(sys.argv) > 3 else 36
pts = []
for fn in sorted(glob.glob(pref + '-j???.arms')):
    L = [l for l in open(fn) if l.strip() and not l.startswith('#')]
    order, narm, cheb, four = map(int, L[0].split()); i = 1
    for a in range(narm):
        oth, cap, npan = map(int, L[i].split()); i += 1
        for p in range(npan):
            i += 1
            for k in range(cheb): pts.append(L[i].split()[:3]); i += 1
stride = max(1, len(pts) // M)
sel = [pts[i] for i in range(0, len(pts), stride)][:M]
open(out, 'w').write('\n'.join(' '.join(s) for s in sel) + '\n')
print(len(pts), 'centerline nodes; stride', stride, '->', len(sel), 'sources ->', out)
