# Network arm-centerline pipeline visuals

Truthfulness: every plotted point is read from the C++ per-stage dump (stages_all.txt) or data/vmtk/vessels_fixed_n298.graph; no Taubin/spline/Hermite math is re-implemented. Only labelled finite-difference diagnostics (rho(s), |dx|) are computed from dumped points.

Script: `python/plot_network_pipeline_stages.py STAGES GRAPH vis --seed 20261012` (random chain picker, seed 20261012, avoids node 6; or pass `--chains "199,209,243,331;20,27,33,39"`).

- pairA = nodes 199 -> 209 -> 243 -> 331(cap): arms j199-209-0, j209-243-0 (interior), j243-331-1 (leaf).
- pairB = nodes 20 -> 27 -> 33 -> 39(cap): arms j20-27-0, j27-33-0 (interior), j33-39-1 (leaf).

Panels (columns = arms in chain order):
- 3D, x-y, x-z: STAGE raw (grey o), STAGE lowpass (green dots, 2D only), STAGE final (red interior / blue leaf), CONTROL leads A0->QA and endC->QB (black).
- ZOOM (top right): arm 1 end A in plane (uA, lateral); raw/curv_limited/lowpass/final, A0, QA, knot nearest bse, uA handle of length corner_kr*|QA-knot| (derived). Lateral axis exaggerated, not equal-scaled.
- Row 2: |curv_limited-raw| and |lowpass-curv_limited| per knot vs spline_knots s (sec:smooth, eq:taubin1/2).
- Row 3: diagnostic (finite-diff of dumped points) curvature radius of raw / lowpass / final vs rA.
- Row 4: raw per-knot Radius (.graph col 4) vs STAGE final r= vs linear taper rA->rB from params (sec:taper). Raw x-axis is a uniform stand-in (index scaled to S).
- Row 5: |M_i| from STAGE spline_knots (natural cubic, eq:tridiag). s is chord arclength.

Spot check: for all 6 arms of both chains, STAGE raw == .graph edge points (reversed when edge n0 != owner), max abs diff 0.0 over all points (first 3 also 0.0), printed by the script.

Findings: curvature limiter changed knots only in j199-209, j243-331, j27-33 (a few knots); it is a no-op (identical to raw) in j209-243, j20-27, j33-39. Taubin low-pass moves knots by up to ~0.1 units everywhere but is visually indistinguishable from raw at figure scale in xy/xz. Radius: raw 2.0 (uniform R_fix) -> final 1.0 (QJ_RADIUS_SCALE 0.5); a visible taper only in j199-209 (~0.65->1.0, shrunken seam rA); no radius dip visible. Final r tracks the rA->rB line.

Row 5 titles now carry the real dumped values (n knots, max|M_i|, |M_0|,|M_n-1| = 0 natural, and the max residual of the tridiagonal system h_(i-1)M_(i-1)+2(h_(i-1)+h_i)M_i+h_iM_(i+1)=6(...) evaluated on the dumped knots/s/M; ~1e-16 for all arms).
