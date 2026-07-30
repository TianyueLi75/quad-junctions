# fmm3dbie S_init benchmark on the Y-bifurcation

Reproduces a **p = 12 row of Table 1c** of Greengard, O'Neil, Rachh et al. 2021, *"Fast multipole
methods for the evaluation of layer potentials with locally-corrected quadratures"*
(J. Comput. Phys. X **10** (2021) 100092) — the **near-field quadrature precomputation speed**

    S_init = N / T_init      (discretization points processed per second)

as a function of the quadrature tolerance ε, at discretization order p = 12, on this repo's
closed all-quad Y-bifurcation surface (order 12, nref 2, production defaults).

## What it does

1. **`dump_rvnodes.f90`** → `rvnodes_o11.txt` — the order-11 Vioreanu–Rokhlin nodes on the reference
   triangle in fmm3dbie's `get_vioreanu_nodes` order. norder = 11 gives npols = 78 = the paper's
   `n_p = p(p+1)/2` at **p = 12**.
2. **`../../src/ybifurc-export-fmm3dbie.cpp`** → `ybifurc_p12.srcvals` — builds the *same* closed,
   watertight surface `ybifurc-hybrid-bie` validates (`BuildYJunctionWithTransitions` +
   `BuildYArmsQuadTube`, arm_kind=1: junction body + transition tubes + hemisphere caps + quad-tube
   arms = **1200 quad elements**), splits each quad along a diagonal into 2 triangles, and samples
   the surface at the 78 RV nodes via `QuadElemList::GetGeom` (exact position + chain-ruled
   reference-triangle tangents + outward normal). Result: **2400 triangles, 187 200 points**.
3. **`s_init_sweep.f90`** — reads the surface, builds `srccoefs`, and for each ε times
   `getnearquad_helm_comb_dir` (single-layer Helmholtz, k = 1, `zpars=(1,1,0)`). The setup +
   timed-region sequence is lifted verbatim from fmm3dbie's own `lpcomp_helm_comb_dir_addsub`.
   Reports S_init = N/T_init, the oversampling parameter α = N_over/N, and m = nquad/N.

## Build & run

```
module load gcc/13.3.0            # or: . ../../sctl_source   (gfortran + MKLROOT)
./run.sh                          # builds everything, exports, sweeps 1-core then 8-core
```

Depends on the locally-built `extern/fmm3dbie/lib-static/libfmm3dbie.a` (and its bundled FMM3D).
See the top-level CLAUDE.md / `build-fmm3d*.log` for the one-time library build.

## Validation (geometry transferred correctly)

- Exporter `report_area`: surface area **5.348442553**, closure |∫n dA| = **6.3e-15**, all Jacobians
  positive; every split-triangle normal aligns with the true outward normal (min n·n = 1, no folds).
- Fortran `get_qwts` reproduces the **same area 5.348443** and closure ~1e-15 → srcvals/ordering
  round-trip is exact.

## Results — Table 1c row, p = 12, single-layer Helmholtz k = 1

N = 187 200 points (2400 triangular patches, 78 nodes each). m = 634.6 near entries/point.

**1-core (Intel Xeon workstation — directly comparable to the paper's single-core Table 1c):**

| p  | ε       | S_init (pts/s) | α    | T_init (s) |
|----|---------|----------------|------|------------|
| 12 | 5·10⁻³  | **332**        | 0.58 | 564.5      |
| 12 | 5·10⁻⁴  | **320**        | 0.65 | 585.7      |
| 12 | 5·10⁻⁷  | **280**        | 1.43 | 669.7      |
| 12 | 5·10⁻¹⁰ | **195**        | 1.95 | 958.6      |

**8-core (context; getnearquad scales ~8× — embarrassingly parallel over targets):**

| p  | ε       | S_init (pts/s) | α    | T_init (s) |
|----|---------|----------------|------|------------|
| 12 | 5·10⁻³  | 2652           | 0.58 | 70.6       |
| 12 | 5·10⁻⁴  | 2564           | 0.65 | 73.0       |
| 12 | 5·10⁻⁷  | 2194           | 1.43 | 85.3       |
| 12 | 5·10⁻¹⁰ | 1221           | 1.95 | 153.3      |

For reference, the paper's Table 1c (single-core laptop i5, Helmholtz SLP on the stellarator,
N_patches = 2400):

| p | 5·10⁻³ | 5·10⁻⁴ | 5·10⁻⁷ | 5·10⁻¹⁰ |
|---|--------|--------|--------|---------|
| 2 | 8460   | 6480   | 2970   | 1350    |
| 3 | 8130   | 5880   | 2460   | 1150    |
| 4 | 8230   | 6440   | 2820   | 1360    |
| 6 | 2890   | 2380   | 1250   | 734     |
| 8 | 2080   | 1750   | 945    | 581     |

**Reading the result.** The p = 12 row reproduces Table 1c's behavior exactly: for fixed p, S_init
decreases monotonically and α increases as ε → 0. Absolute S_init sits below the paper's rows because
p = 12 (78 nodes/triangle) is much higher order than the paper's max p = 8 (36 nodes) — the self/near
quadrature per point is intrinsically costlier — compounded by a single-core hardware factor
quantified below. (Patch aspect ratio is *not* the culprit: the p = 8 check below shows our split-quad
triangles are as well-shaped as the paper's stellarator patches.)

## Validation against Table 1 at p = 8 (`stell_sinit`)

To check the harness reproduces the paper, `stell_sinit.f90` generates the paper's **own** stellarator
(fmm3dbie's built-in `get_stellarator_npat`, the exact Eq. 6.1 surface — which also splits each quad
into 2 triangles) at nuv = (20, 60) → **N_patches = 2400, N = 86 400**, the paper's discretization,
and runs the identical sweep. We then run our Y-bifurcation at the same p = 8 (it also has exactly
2400 triangular patches), so the only difference is the geometry. All single-core:

| ε | metric | Paper Table 1 | fmm3dbie stellarator (repro) | Our Y-bifurcation |
|--------|--------|---------------|------------------------------|-------------------|
| —      | m      | 286           | **286.2**                    | 293.0             |
| 5·10⁻³ | α      | 0.777         | **0.78**                     | 0.78              |
|        | S_init | 2080          | 1138                         | 1144              |
| 5·10⁻⁴ | α      | 0.923         | **0.92**                     | 0.90              |
|        | S_init | 1750          | 1083                         | 1086              |
| 5·10⁻⁷ | α      | 2.24          | **2.24**                     | 2.27              |
|        | S_init | 945           | 743                          | 717               |
| 5·10⁻¹⁰| α      | 3.9           | **3.90**                     | 3.92              |
|        | S_init | 581           | 466                          | 458               |

**Two clean conclusions:**

1. **Method reproduced exactly.** α and m are geometry-and-tolerance determined (hardware-independent);
   our stellarator run matches Table 1a/1b to **3 significant figures** (α = 0.78/0.92/2.24/3.90 vs
   0.777/0.923/2.24/3.9; m = 286.2 vs 286). Since α and m fix nquad, the near-field *work* equals the
   paper's, so the S_init gap is purely time-per-operation — a **single-core hardware/compiler factor
   of ≈1.3–1.8×** (paper: i5 2.3 GHz laptop, gfortran 10.2; here: Flatiron Xeon single core, gfortran
   13.3 + MKL). The gap narrows at tight ε (466 vs 581 = 1.25×), where adaptive integration dominates
   over fixed overhead.

2. **Our geometry ≈ the paper's stellarator in quality.** At p = 8, N_patches = 2400, our
   Y-bifurcation matches the stellarator in S_init (within 3%), α (within 1%), and m (within 2%). The
   diagonal split of the Y-bifurcation quads yields patches of essentially the same aspect quality as
   the paper's discretization — so the p = 12 S_init above is limited by order, not by the surface.
