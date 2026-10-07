#!/bin/bash
# =============================================================================
# run-capsule-mfs-precond.sh
#
# LOCAL sequential runner (login node / workstation) for the manufactured-Stokeslet Dirichlet solve
# (src/capsule-robin-bie.cpp `mfs` mode) as a GMRES-iteration x preconditioner probe. For each of 3
# capsule geometries and each of 4 preconditioners, run BOTH the exterior solve (mfs-ext, interior
# source) and the interior solve (mfs-int, exterior source).
#
# 4 precond x 3 geom x 2 problem = 24 single-process tests, run ONE AT A TIME (sequential), each with
# 8 OpenMP threads pinned to physical cores 0-7. Each test's stdout+stderr is written to a log named
# like a job step output (vis/capsule-mfs/<problem>-<geom>-<precond>.log) so scripts/parse-capsule-mfs.sh
# still works.
#
# After the run:  scripts/parse-capsule-mfs.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

. ./sctl_source

# Serial build (single process, OpenMP only -- no mpirun, so MPI=0). Object files are not keyed by
# build flags, so if the tree was last built MPI=1 this will rebuild; otherwise it is a no-op.
make MPI=0 bin/capsule-robin-bie

BIN=./bin/capsule-robin-bie
OUT=vis/capsule-mfs
mkdir -p "$OUT"

# ---- fixed mesh/solve params (match scripts/run-capsule-arm-precond.sh) --------------------------
ORDER=12; FOURIER=16; NCAP=2; TOL=1e-8

# ---- OpenMP pinning: 8 threads on physical cores 0-7, one test at a time ------------------------
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ---- preconditioner rows (label -> env prefix) ---------------------------------------------------
#   none       : no preconditioner
#   panelExact : exact per-panel block-Jacobi (QJ_JACOBI=block overrides arm groups -> arm off)
#   panelCyl   : per-panel straight-cylinder proxy (group=1, strict per-panel)
#   armWhole   : whole-arm straight-cylinder block
declare -a PRECOND_LABELS=( none panelExact panelCyl armWhole )
declare -A PRECOND_ENV=(
  [none]="QJ_ARM_PRECOND=off"
  [panelExact]="QJ_JACOBI=block QJ_ARM_PRECOND=off"
  [panelCyl]="QJ_ARM_PRECOND=panelcyl QJ_ARM_PRECOND_GEOM=straight QJ_ARM_PRECOND_GROUP=1"
  [armWhole]="QJ_ARM_PRECOND=whole QJ_ARM_PRECOND_GEOM=straight"
)

# ---- geometry columns (label -> "R L n_axial" ; spiral env) --------------------------------------
declare -a GEOM_LABELS=( straightL1 straightL10 spiralL10 )
declare -A GEOM_ARGS=(
  [straightL1]="0.3 1 8"
  [straightL10]="0.3 10 8"
  [spiralL10]="0.3 10 12"
)
declare -A GEOM_ENV=(
  [straightL1]=""
  [straightL10]=""
  [spiralL10]="QJ_SPIRAL_AMP=0.5 QJ_SPIRAL_TURNS=1"
)

echo "=== capsule mfs precond probe (LOCAL, sequential, 8 OMP threads on cores 0-7) ==="
echo "    order=$ORDER fourier=$FOURIER Ncap=$NCAP tol=$TOL"

n=0
for prob in int ext; do
  for g in "${GEOM_LABELS[@]}"; do
    read -r R L NA <<<"${GEOM_ARGS[$g]}"
    for p in "${PRECOND_LABELS[@]}"; do
      n=$(( n + 1 ))
      log="$OUT/${prob}-${g}-${p}.log"
      echo ">>> [$n/24] prob=$prob geom=$g precond=$p (R=$R L=$L n_axial=$NA)  -> $log"
      env ${GEOM_ENV[$g]} ${PRECOND_ENV[$p]} QJ_MFS_ONLY="$prob" \
          taskset -c 0-7 "$BIN" mfs "$ORDER" "$R" "$L" "$NA" "$FOURIER" "$NCAP" "$TOL" \
          > "$log" 2>&1
    done
  done
done

echo
echo "=== SUMMARY : GMRES iters (grep) ==="
for f in "$OUT"/*.log; do
  [ -e "$f" ] || continue
  it=$(grep -oE 'StokesBIO GMRES iters = [0-9]+' "$f" | grep -oE '[0-9]+$' | head -1 || true)
  printf "  %-40s iters=%s\n" "$(basename "$f")" "${it:-?}"
done

echo
echo "Now run:  scripts/parse-capsule-mfs.sh"
