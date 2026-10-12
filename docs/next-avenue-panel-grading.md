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
(5.08e-6 / 1.04e-4, Jac>0). Full DL/Green's NOT run (needs Slurm).
Order-independence: not re-tested; no order/fourier/tol was changed anywhere, geometry only.
Caveats: subset numbers are from 4 of 5 faithful sites; R=2 is recommended (R=3 nearly as good, cheaper); the next
floor (~2e-4 residual + junction 7e-5) will need cand. 1-3 or RADIUS_SCALE.

## Prepared full gate (NOT submitted)
Bundle already built (serial, ~10 min):
`QJ_ARM_PANEL_GRADE=2 OMP_NUM_THREADS=4 ./bin/bifurc-network-assemble data/vmtk/vessels_fixed_n298.graph geom/n298_tr6_grade2 12 1 1.5 0.4 3 12 10 24`
Gate: `sbatch scripts/submit-network-greens-grade2.sh` (DL only), `--export=ALL,FULL=1` adds Green's; equivalent to
`OMP_NUM_THREADS=8 mpirun -n 2 env QJ_NET_MODE=greens QJ_GREENS_DLONLY=1 QJ_GREENS_KER=laplace ./bin/bifurc-network-flow-bie geom/n298_tr6_grade2 1e-6`.
Note: this checkout's test_DLIdentity prints only the argmax node (no top-20 [map]); set QJ_GREENS_DUMP=vis/g2 for the VTU map.

## Memory proposal (one line)
"network panel grading: DL hotspots = turn-refine long|short panel-size knots (6-8x); QJ_ARM_PANEL_GRADE=2 (+5% panels) cut hot5 subset DL 1.33e-3 -> 2.5e-4; subset faithful via QJ_L10; full gate pending."

---
# UPDATE 2026-10-12: full-network gate result (out/network-greens-7223686.log, geom/n298_tr6_grade2, GRADE=2)

| metric | production (7158497 / f36 7198067 same-format) | grade2 (7223686) | verdict |
|---|---|---|---|
| arm panels (slender) | 18973 | 20093 (+5.9%) | |
| COMBINED / rel closure | 0.0179 / 6.67e-8 | 0.01464 / ~5.5e-8 | slightly better |
| DL const-density max rel err (max abs) | 1.328e-3 (6.64e-4) | 4.296e-4 (2.15e-4) | IMPROVED 3.1x (subset predicted 5.3x) |
| Green's ext ("error", same format) | 5.636e-4 | 1.64e-4 | improved 3.4x |
| Green's int ("error") | 5.899e-5 | 7.90e-4 | NOT COMPARABLE (see below) |

Rel-L2 is not printed by this checkout's driver (the production log came from a build that printed it); the "Green's identity
error" column above is the like-for-like quantity (max-err/max|u|), identical in format across 7158497, 7198067, 7202023, 7223686.

**Interior Green's is inconclusive, not shown regressed.** Interior sources are stride-sampled arm centerline NODES
(`stride = Ncen/36`); grading changes Ncen (18973*10 -> 20093*10 panels) so the 36 sources MOVED (verified: the two source
lists differ point by point). Same pitfall hit a16 (int 5.8e-3). Test with identical sources on the hot5 subset
(new `QJ_GREENS_INT_XYZ`, `python/greens_int_sources.py`): base 6.03e-4 / 5.58e-4 -> grade2 9.2e-5 / 6.0e-5 (int improves 9x).
So the 13x "worsening" is most likely a source-set artifact; a fixed-source full run decides (script below).

**Localization.** This checkout prints only the argmax node (no top-20), and no VTU was dumped. Rigorous inference:
the full max is 2.15e-4 abs at (-329,-382,369); the five baseline sites were 4.2e-4..6.6e-4 abs, so all five are now < 2.15e-4
(subset: 3e-5..9e-5). The new argmax is arm j296->300 (cap) panel 10/24 = the p9|p10 2:1 knot (len 2.69|1.34, bends 5.1|3.9 deg),
UNTOUCHED by grading (identical panels in both bundles), i.e. a pre-existing site that was just below the baseline top-20 cutoff.
A 1-junction faithful subset (seed 300, QJ_L10) reproduces it EXACTLY: base 4.2957e-4, GRADE=2 4.2876e-4 (same value as the full net).
No new fold: junc/arms minWt unchanged (5.08e-6 / 1.04e-4). No new long|short jump is created by construction (all neighbour
ratios <= 2). Knot C1 mismatch there is only ~0.03 deg (as at non-hot knots), pos gap 1e-6: it is not a panel-join mismatch.
Cause at this site: the centerline curvature still oscillates strongly over ~2-3 radii (kappa 0.012 -> 0.026 -> 0.054 -> 0.016 -> 0.065
across p8..p12) despite 64 Taubin passes: the toroidal near-constant-curvature assumption again.

**Probes on the j296 subset (1 junction, 3 caps; DL max rel err, base = 4.296e-4):**
| variant | arm panels | DL max rel | note |
|---|---|---|---|
| GRADE=2 | 165 (+4%) | 4.288e-4 | no effect here (not a size-jump site) |
| GRADE=2 + TURN_REFINE=4 | 181 | 3.87e-4 | -10% |
| GRADE=2 + TURN_REFINE=3 | 196 | ~3.9e-4 | -10% |
| GRADE=2 + LOWPASS_PASSES=256 | 165 | 2.85e-4 | 1.5x; centerline moves <= 0.69 r (mean 0.06-0.23 r) |
| GRADE=2 + LOWPASS_PASSES=1024 | 165 | 1.64e-4 | 2.6x; but centerline moves up to 4.2 r (mean 1.5 r): too much |
(`QJ_ARM_CL_LOWPASS_PASSES` was baked at 64; it is env-readable again, default 64 = byte-identical.)

## Verdict and decision
- GRADE=2 is a KEEPER: removes the five size-jump hotspots (verified at the real sites in faithful subsets and by the full-net
  max falling 3.1x to a different site), +5.9% panels, closure better, minWt unchanged. R=2 is right (R=3 leaves ~1.5x more, R<2 is degenerate for bisection).
- It does NOT reach the subset's 5.3x on the full net because the new floor (4.3e-4) is the NEXT mechanism: curvature
  noise on bent arms, independent of panel size. Next-ranked lever that actually moves it: more centerline low-pass (candidate 6),
  PASSES=256 (1.5x at that site, <= 0.7 r drift), possibly combined with graded refinement keyed on d(kappa)/ds (not built).
  TURN_REFINE<6 is a weak lever (-10%), RADIUS_SCALE shrink stays the generic size lever. Seam coverage/overlap (cand. 1-3) are still
  not indicated: neither remaining site is at a mouth.
- Caveat on lowpass: 256 passes alters the centerline (tortuosity) up to 0.7 radius; check fold/minWt on the full bundle (built:
  geom/n298_tr6_grade2_lp256) before trusting it.

## Prepared next gate (NOT submitted)
`sbatch scripts/submit-network-greens-grade2-fixsrc.sh` : production, grade2, grade2+lp256 with IDENTICAL interior sources
(`geom/greens_int_prod.xyz`, via QJ_GREENS_INT_XYZ), full Laplace Green's + DL. `--export=ALL,DLONLY=1` for the cheap DL-only pass.
Add `QJ_GREENS_DUMP=vis/<tag>` to get the spatial error map (VTU) since this build has no top-20 printout.

Memory proposal: "GRADE=2 full gate: DL 1.33e-3 -> 4.3e-4, closure 5.5e-8, ext Green's 3.4x better; int Green's NOT comparable (stride-sampled sources shift with node count; use QJ_GREENS_INT_XYZ); new floor = j296 curvature-noise knot, lowpass 256 gives 1.5x there."

Full bundle geom/n298_tr6_grade2_lp256 (GRADE=2 + LOWPASS_PASSES=256) built: COMBINED 0.0107 (rel ~4.0e-8), junc/arms minWt 5.08e-6 / 1.06e-4 (Jac>0, no fold),
but arm area changed 2.615e5 -> 2.658e5 (+1.6%), i.e. the centerlines genuinely moved; full DL/Green's pending the fix-source gate.
