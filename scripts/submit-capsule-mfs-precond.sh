#!/bin/bash
#SBATCH --job-name=capsule-mfs-precond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --time=04:00:00
#SBATCH --output=vis/capsule-mfs/slurm-%j.out
# =============================================================================
# submit-capsule-mfs-precond.sh
#
# Manufactured Stokeslet Dirichlet solve (src/capsule-robin-bie.cpp `mfs` mode) as a GMRES-iteration
# x preconditioner probe. For each of 3 capsule geometries and each of 4 preconditioners, run BOTH
# the exterior solve (mfs-ext, interior source) and the interior solve (mfs-int, exterior source).
#
# 4 precond x 3 geom x 2 problem = 24 single-rank tests. Each test is `mpirun -n 1` pinned to 16
# physical cores (taskset), so with the 64 allocated cores we run 4 tests concurrently (batches of 4).
#
# After the job:  python scripts/parse-capsule-mfs.py
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

. ./sctl_source

# Object files are not keyed by build flags -> clean before switching to MPI=1 (each test is mpirun).
make clean >/dev/null 2>&1 || true
make MPI=1 bin/capsule-robin-bie

BIN=./bin/capsule-robin-bie
OUT=vis/capsule-mfs
mkdir -p "$OUT"

# ---- fixed mesh/solve params (match scripts/run-capsule-arm-precond.sh) --------------------------
ORDER=12; FOURIER=16; NCAP=2; TOL=1e-8

# ---- core-mapping knobs --------------------------------------------------------------------------
CORES_PER_TEST=16
NCORES=${SLURM_CPUS_PER_TASK:-64}
SLOTS=$(( NCORES / CORES_PER_TEST ))     # concurrent tests (= 4 for 64/16)
[ "$SLOTS" -ge 1 ] || SLOTS=1

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

# ---- enumerate the 24 (problem, geom, precond) triples -------------------------------------------
declare -a JOBS=()
for prob in int ext; do
  for g in "${GEOM_LABELS[@]}"; do
    for p in "${PRECOND_LABELS[@]}"; do
      JOBS+=( "$prob|$g|$p" )
    done
  done
done

echo "=== capsule mfs precond probe : ${#JOBS[@]} tests, $SLOTS concurrent (${CORES_PER_TEST} cores each) ==="
echo "    order=$ORDER fourier=$FOURIER Ncap=$NCAP tol=$TOL"

run_one() {   # $1=slot  $2="prob|geom|precond"
  local slot="$1" spec="$2"
  local prob g p; IFS='|' read -r prob g p <<<"$spec"
  local lo=$(( slot * CORES_PER_TEST )) hi=$(( slot * CORES_PER_TEST + CORES_PER_TEST - 1 ))
  read -r R L NA <<<"${GEOM_ARGS[$g]}"
  local log="$OUT/${prob}-${g}-${p}.log"
  echo ">>> slot=$slot cores=$lo-$hi  prob=$prob geom=$g precond=$p (R=$R L=$L n_axial=$NA)  -> $log"
  env ${GEOM_ENV[$g]} ${PRECOND_ENV[$p]} QJ_MFS_ONLY="$prob" \
      OMP_NUM_THREADS="$CORES_PER_TEST" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "${lo}-${hi}" mpirun -n 1 --bind-to none \
      "$BIN" mfs "$ORDER" "$R" "$L" "$NA" "$FOURIER" "$NCAP" "$TOL" \
      > "$log" 2>&1 &
}

# ---- launch in batches of $SLOTS ----------------------------------------------------------------
i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  for (( s=0; s<SLOTS && i<${#JOBS[@]}; s++, i++ )); do
    run_one "$s" "${JOBS[$i]}"
  done
  wait
done

echo
echo "=== SUMMARY : GMRES iters (grep) ==="
for f in "$OUT"/*.log; do
  [ -e "$f" ] || continue
  it=$(grep -oE 'StokesBIO GMRES iters = [0-9]+' "$f" | grep -oE '[0-9]+$' | head -1 || true)
  printf "  %-40s iters=%s\n" "$(basename "$f")" "${it:-?}"
done

echo
echo "Now run:  python scripts/parse-capsule-mfs.py"
