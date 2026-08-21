# Scheme-comparison benchmark

Ported from the `SCTL_quad_element` fork (branch `2-performance`). Compares the four singular
self/near quadrature schemes — **RP (RectPolar), Adaptive, Hybrid, Duffy** — of `QuadElemList` on a
twisted cubed sphere, reporting the on-surface Green's-identity error and the single-layer setup
throughput per scheme (the analogue of the original cubed-sphere bench's per-machine columns).

Self-contained: everything lives under `bench/scheme-compare/` and nothing in the top-level Makefile
changes. The driver `#include`s `<sctl/experimental/quad_element.cpp>` directly, so it also exercises
the ported near/self quadrature.

## Files
- `bench-scheme-compare.cpp` — the driver. `conv`/`omp` modes, one `(kernel, scheme)` per run,
  emits `@@ROW kernel=… scheme=… thr=… twist=… tol=… error=… pps=… setup=…` lines.
- `build.sh` — compiles the driver with the quad-junctions flags into `bin.<TAG>/`.
- `bench-scheme-compare.sh` — SLURM: (kernel × scheme) × (conv, omp) across concurrent core-pinned slots.
- `bench-scheme-compare-omp.sh` — SLURM: full-node OpenMP strong-scaling, per scheme, sequential.
- `parse-scheme-compare.sh` / `parse-scheme-compare-omp.sh` — parse `@@ROW` output → text + LaTeX tables.

## Build & run locally
```
. ./sctl_source                       # from the repo root
bash bench/scheme-compare/build.sh    # -> bench/scheme-compare/bin.native/bench-scheme-compare
BIN=bench/scheme-compare/bin.native/bench-scheme-compare
OMP_NUM_THREADS=8 $BIN conv laplace Duffy      # one (mode, kernel, scheme)
```
Schemes: `RP | Adaptive | Hybrid | Duffy`. Modes: `conv` (tol/twist sweep) and `omp` (thread scaling).

## On the cluster
```
sbatch --ntasks-per-node=2 --cpus-per-task=32 bench/scheme-compare/bench-scheme-compare.sh
sbatch bench/scheme-compare/bench-scheme-compare-omp.sh
```
The SLURM scripts are Flatiron/Icelake-specific (partition, `-march=native` → per-`TAG` `bin.<TAG>/`
to avoid SIGILL on Zen nodes sharing the filesystem); adjust the `#SBATCH` headers for other sites.
Outputs land in `doc/data/`; feed the merged conv/omp files to the `parse-*` scripts.
