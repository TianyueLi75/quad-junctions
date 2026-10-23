# Network arm panelization — historical levers & durable pitfalls (tombstone)

> **Status 2026-10-23.** This doc previously tracked two BLEND-path (attempt 1) arm panelization levers —
> `QJ_ARM_PANEL_GRADE` (2:1 panel-size balancing) and `QJ_ARM_KNOT_SNAP` (insert C2 knots as extra panel
> boundaries). Both were removed with the BLEND arm builder when the network pipeline was standardized on the
> **poly-fit** arm pipeline (`append_polyfit_fiber` in `include/quad_junctions/gen_network_geom.hpp`). The
> content is reduced here to the pitfalls that still constrain any future panelization work.

## Attempts, one line each
- **Attempt 1 (BLEND, removed):** dense vmtk polyline + C2 overlay; `QJ_ARM_PANEL_GRADE` balanced long|short
  panels, `QJ_ARM_KNOT_SNAP` inserted C2 knots as panel boundaries. Former baseline `n298_seamsettle_tr6`.
- **Attempt 2 (DOWNSAMPLE, removed as a user path; kept only as poly-fit's short-arm fallback):** sparse
  uniform downsample + natural-C2 cubic + panel-snap by **SHIFT**. Best gate DL 9.63e-4 / closure 2.28e-8.
- **Attempt 3 (POLY-FIT, current / default):** wide-knot piecewise degree-9 Chebyshev body, panel boundaries
  on the segment knots → panel-exact by construction. See the LaTeX writeup / `polyfit_centerline.md`.

## Durable pitfalls (keep these)
1. **Never create slivers that leave CSBQ's cached toroidal aspect range.** Knot-snap by *inserting* knots as
   extra boundaries nearly doubled the slender panel count (18973→35000), drove min quadrature weight to
   9.2e-7, and tripped 16 `aspect-ratio outside the range of precomputed quadratures` warnings → the DL gate
   regressed **~1550×** (DL 1.33e-3 → 2.06). Only `data/special_quad_q10` is cached ([[csbq-elemorder-10]]).
   This is *the* reason both later alignments **shift** panels (keeping count/size ~constant) or make panels
   analytic within one segment, rather than splitting.
2. **Proxy ≠ gate.** Build-time geometry proxies (watertight closure `|∫n·dA|`, seam `dpos`/`dtang`) improved
   under knot-snap even as the actual DL quadrature gate regressed catastrophically. Always run the real
   DL/Green's gate, never trust the seam-check alone ([[network-build-test-order]]).
3. **The C2-kink straddle floor.** A degree-10 (cheb=10) panel reaches ~1e-13 only for an *analytic* curve.
   A piecewise-cubic (C2) centerline has a 3rd-derivative jump at every resample knot; a panel straddling a
   knot interpolates a `(s−s_k)_+^3` kink and converges only algebraically (~1e-7 floor). Poly-fit removes
   this by making the body a degree-9 polynomial within each wide segment (reproduced exactly on any
   subinterval), so panel-exactness is intrinsic, not a consequence of snapping.
4. **Turn-refinement is inert on the knot-straddle floor** (bisecting by tangent-bend does not align panel
   edges to centerline knots). It remains a valid safeguard for localized tight bends (`QJ_ARM_TURN_REFINE`).
5. **Interior Green's with stride-sampled sources is not A/B-comparable** across builds that change the node
   count: the sources physically move, so a "regression" can be a source-set artifact. Pin sources to compare.
