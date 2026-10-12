# Next avenue: arm panel-SIZE GRADING (QJ_ARM_PANEL_GRADE)

Branch `debug`. Knob: `QJ_ARM_PANEL_GRADE=<R>` (R>1; unset/0 = OFF, bundle byte-identical; verified 320/320 files of the
full n298 network identical to geom/n298_seamsettle_tr6). `QJ_ARM_PANEL_GRADE_MAX` bounds sweeps (default 8).
Support knob: `QJ_L10=<len>` pins the reference panel arclength so a SUBSET graph reproduces the FULL network's
panelization (n298: 31.564552682809513). The assembler now also prints `L10 exact`.

## Finding (the localization, resolved one level deeper)
Located the 5 production DL-hotspot sites (log lines 47-71) on the bundle's arm panels (script: node -> nearest arm
centerline node -> panel index, length, bend). ALL FIVE sit on an abrupt LONG|SHORT panel-length knot created by turn-refine:

| site (x,y,z)            | arm            | hot panel | panel lengths around it           |
|-------------------------|----------------|-----------|-----------------------------------|
| (-201, 90.7, -306)      | j70 -> 82 cap  | p7        | p6 = 2.61 | p7.. = 0.37  (7.0x)   |
| (113, -675, 254)        | j213 -> 225 cap| p16       | p16 = 2.50 | p17 = 0.37  (6.8x)   |
| (-426, 25.2, -93)       | j150 -> 165 cap| p6        | p5 = 2.90 | p6.. = 0.36  (8.0x)   |
| (-259, 127, -316)       | j92 -> 101     | p12       | p12 = 0.42 | p13 = 2.54  (6.0x)   |
| (138, -731, 228)        | j224 -> 233 cap| p5        | p4 = 2.45 | p5.. = 0.41  (6.0x)   |

They are NOT mouths (3-6 radii in; mouths are exact by construction) and NOT the max-curvature panels (p13 of site 1 has
rho/r = 2.3 and is not hot; the hot p7 has rho/r = 13.6). Mechanism: turn-refine bisects only panels with bend > 6 deg
into ceil(bend/6) EQUAL sub-panels and leaves straight neighbours at h_panel, so arc length jumps up to 7x (cap 24x)
across one knot. The long neighbour carries a steep curvature RAMP (kappa 0.002 -> 0.074 /unit across the knot) that violates
CSBQ's toroidal rule assumption of near-constant panel curvature, and its near-quad sees a 7x smaller element. Network
census: 179 of 18636 panel-pairs have size ratio >= 4 (283 in 2.5-4); this is where the band lives.

## Lever
2:1-style balancing (octree-like) of the arm panel knots: repeatedly bisect (in t) any panel longer than R x its shorter
neighbour. Same curve, same radius, same surface (closure-neutral), terminal knots kept, rings exact. R=2: +5.1% arm
panels network-wide (18973 -> ~19950). R=3: +2.2%. R<2 is degenerate for bisection (R=1.5 = 2x the panels network-wide
since halves always differ by exactly 2).

## Ranking vs the other candidates (tied to the ARM-only localization)
1. PANEL GRADING (this, done). Directly hits the measured mechanism; new (not ROLE_REFINE, which refines only at
   lead|corner|run role knots; not n_axial a16, which is uniform and regressed; not TURN_REFINE, which created the jumps).
2. Cross-list near-quad coverage / seam overlap / node-conforming (cand. 1-3): all junction<->mouth hypotheses. Evidence
   against them for THIS band: top-20 are arm nodes 3-6 radii from any mouth; junction-side max in the subset is 7e-5
   (the post-grading floor). Keep as the next lever once the arm band is down.
3. Hermite corner-cut in network (4), per-arm low-pass nudge (6): smoothing levers; the hot panels are bend 1.6-4 deg,
   already smooth; the defect is discretization-size jump, not curve quality. Cheap follow-up only.
4. RADIUS_SCALE 0.3 / 0.1 (5): scales ALL error ~s^2 (units), a size lever not a fix; combine after grading.
5. From-mouth hole-cone (7): out of scope.

## Measured (4 cores)
FAITHFUL SUBSET: `python/subset_graph.py 82,225,165,101,233` (5 junctions, 14 caps) + `QJ_L10=31.564552682809513`.
Anchor check: the BASE subset DL gate reproduces the full-network hotspots at the SAME coordinates and value: max rel err
1.32625e-3 vs full 1.32833e-3 (site 4 / j92->101 is not faithful: it is interior in the full net, cap in the subset).
Laplace DL const-density (tol 1e-6, order 12, fourier 24, cheb 10), max over plot points within 3 units of each site:

| config            | arm panels | DL max rel err | site1  | site2  | site3  | site5  | junc max | COMBINED |
|-------------------|-----------:|---------------:|-------:|-------:|-------:|-------:|---------:|---------:|
| base (grade off)  | 441        | 1.326e-3       | 8.2e-4 | 6.8e-4 | 6.6e-4 | 5.7e-4 | 7.2e-5   | 3.82e-3  |
| GRADE=3           | 464 (+5%)  | 3.90e-4        | 7.5e-5 | 9.2e-5 | 8.3e-5 | 3.3e-5 | 7.3e-5   | 3.49e-3  |
| GRADE=2           | 486 (+10%) | 2.48e-4 (5.3x) | 7.4e-5 | 9.2e-5 | 8.3e-5 | 3.3e-5 | 7.3e-5   | 3.27e-3  |
| GRADE=1.5         | 843 (+91%) | 1.89e-4 (7.0x) | 7.4e-5 | 9.1e-5 | 8.8e-5 | 3.1e-5 | 7.3e-5   | 1.47e-3  |

Site values are plot-point DL errors (|U/q+1/2|) so they read slightly above node values. Hotspots drop 8-17x; the
residual max moves to a new spot 4 radii off site 2 (112,-681,268) at ~2e-4, and the junction floor 7e-5 is untouched.
Laplace Green's (subset, abs error ext/int): base 6.0e-4 / 5.6e-4 -> GRADE=2 9.2e-5 / 9.8e-5 (6x). Subset Green's
sources differ from the full gate, so only the ratio is meaningful.
FULL BUNDLE (built, geom/n298_tr6_grade2): COMBINED 0.0179 -> 0.0146 (rel 6.67e-8 -> ~5.5e-8); junc/arms minWt unchanged
(5.08e-6 / 1.04e-4, Jac>0). Full DL/Green's gate RAN (job 7223686) — see "full-network gate result" below.
Order-independence: not re-tested; no order/fourier/tol was changed anywhere, geometry only.
Caveats: subset numbers are from 4 of 5 faithful sites; R=2 is recommended (R=3 nearly as good, cheaper); the next
floor (~2e-4 residual + junction 7e-5) will need cand. 1-3 or RADIUS_SCALE.

## Prepared full gate (NOT submitted)
Bundle already built (serial, ~10 min):
`QJ_ARM_PANEL_GRADE=2 OMP_NUM_THREADS=4 ./bin/bifurc-network-assemble data/vmtk/vessels_fixed_n298.graph geom/n298_tr6_grade2 12 1 1.5 0.4 3 12 10 24`
Gate: `sbatch scripts/submit-network-greens-grade2.sh` (DL only), `--export=ALL,FULL=1` adds Green's; equivalent to
`OMP_NUM_THREADS=8 mpirun -n 2 env QJ_NET_MODE=greens QJ_GREENS_DLONLY=1 QJ_GREENS_KER=laplace ./bin/bifurc-network-flow-bie geom/n298_tr6_grade2 1e-6`.
Note: this checkout's test_DLIdentity prints only the argmax node (no top-20 [map]); set QJ_GREENS_DUMP=vis/g2 for the VTU map.

## Full-network gate result (2026-10-12, out/network-greens-7223686.log, geom/n298_tr6_grade2, GRADE=2)

GRADE=2 was validated on the FULL network (job 7223686), not just the subset:

| metric | production (7158497) | grade2 (7223686) | verdict |
|---|---|---|---|
| arm panels (slender) | 18973 | 20093 (+5.9%) | cost |
| COMBINED / rel closure | 0.0179 / 6.67e-8 | 0.01464 / ~5.5e-8 | slightly better |
| DL const-density max rel err (max abs) | 1.328e-3 (6.64e-4) | 4.296e-4 (2.15e-4) | IMPROVED 3.1x (subset predicted 5.3x) |
| Green's ext ("identity error", like-for-like) | 5.636e-4 | 1.64e-4 | improved 3.4x |
| Green's int ("identity error") | 5.899e-5 | 7.90e-4 | NOT COMPARABLE (artifact, see below) |

Note on format: this checkout's driver prints a "Green's identity error" (max-err/max|u|), not the rel-L2 the production
4.6e-6/8.7e-6 came from (different build); the columns above are the like-for-like quantity, identical in format across
logs 7158497 / 7198067 / 7202023 / 7223686.

**Interior Green's is inconclusive, NOT a real regression.** The interior sources are stride-sampled arm-centerline NODES
(stride = Ncen/36); grading changes Ncen, so the 36 sources physically MOVED between the two runs (the two source lists were
verified to differ point by point) — the same pitfall the a16 run hit (int 5.8e-3). A one-off fixed-source cross-check on the
hot5 subset (identical source points for both bundles) gave base 6.03e-4/5.58e-4 -> grade2 9.2e-5/6.0e-5 (interior ~9x BETTER),
i.e. the full-net "13x worsening" is a source-set artifact. (That fixed-source tooling was a probe only and has been reverted;
this route is not being pursued. A clean confirmation would require a full run with pinned interior sources.)

**Localization (full net).** The driver prints only the argmax node (no top-20, no VTU dumped). The full max is 2.15e-4 abs at
(-329,-382,369); the five baseline hotspot sites were 4.2e-4..6.6e-4 abs, so all five are now BELOW 2.15e-4 (subset: 3e-5..9e-5)
— grade2 cleared them as intended. The new argmax is arm j296->300 panel 10/24 (a 2:1 knot, len 2.69|1.34, bends 5.1|3.9 deg),
UNTOUCHED by grading (identical panels in both bundles) — a pre-existing site just below the baseline top-20 cutoff; a 1-junction
faithful subset reproduces it exactly (base 4.2957e-4, GRADE=2 4.2876e-4). No new fold (junc/arms minWt unchanged 5.08e-6/1.04e-4)
and no new long|short jump (all neighbour ratios <= 2 by construction). The C1 mismatch there is only ~0.03 deg (as at non-hot
knots), so it is not a panel-join defect: the centerline curvature still oscillates over ~2-3 radii despite the 64 Taubin passes.

**Verdict: GRADE=2 (R=2) is a KEEPER** — it removes the five size-jump hotspots (confirmed at the real sites and by the full-net
max falling 3.1x to a different, pre-existing site), costs +5.9% panels, improves closure, and creates no fold. R=3 leaves ~1.5x
more error; R<2 is degenerate for bisection. It does NOT reach the subset's 5.3x on the full net because the new floor (4.3e-4) is
a DIFFERENT mechanism — curvature (dkappa/ds) noise on bent arms, independent of panel size. That next floor is diagnosed but not
pursued here; seam coverage/overlap (cand. 1-3) remain not indicated since the remaining site is not at a mouth.

## Memory proposal (one line)
"GRADE=2 full gate (7223686): DL 1.33e-3 -> 4.3e-4 (3.1x), closure ~5.5e-8, ext Green's 3.4x better, five baseline hotspots cleared, no new fold; int Green's NOT comparable (stride-sampled sources shift with node count); new floor = j296 curvature-noise knot (not panel size)."
