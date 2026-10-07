#!/bin/bash
#SBATCH --job-name=capsule-arm-precond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=vis/capsule-precond/slurm-%j.out
# =============================================================================
# run-capsule-arm-precond.sh
#
# The CSBQ whole-arm EXACT block preconditioner (junction_precond.hpp: ArmWhole) on the capsule
# flow solve (src/capsule-robin-bie.cpp `flow` mode). Only the whole-arm exact block is kept -- the
# per-panel and length-normalized/additive variants were removed as ineffective.
#
# Part 1 (straight, retained baseline): 3 straight cases x precond {off, whole}. `whole` builds the
#         block at the arm's true straight geometry; it should help most on the slender case.
# Part 2 (SPIRAL study): a mild helix per case, comparing three preconditioners --
#         off             : no preconditioner (baseline),
#         wholeStraight   : block built from a STRAIGHT cylinder of the arm's ARC length + radius
#                           (the cheap, cacheable proxy; QJ_ARM_PRECOND_GEOM=straight),
#         wholeActual     : block built from the ACTUAL spiral geometry (QJ_ARM_PRECOND_GEOM=actual).
#         Question: does the straight proxy precondition a spiral as well as its exact block?
#
# A LEFT preconditioner must not change the solution, so the interior flux Phi(x) must match `off`
# for every variant (correctness oracle); the GMRES iteration count is the effect being measured.
#
# Serial single process; OpenMP pinned to 10 physical cores.
# =============================================================================
set -euo pipefail
# Resolve the project root. Under `sbatch`, Slurm copies this script into the spool dir, so $0's dirname
# points there (the "./sctl_source: No such file" failure, job 7081720). Prefer SLURM_SUBMIT_DIR (the dir
# sbatch was run from), else the script's own dir; then climb until sctl_source is found, so BOTH
# `bash scripts/run-capsule-arm-precond.sh` and `sbatch scripts/run-capsule-arm-precond.sh` land in root.
root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$root"
while [ ! -f sctl_source ] && [ "$PWD" != "/" ]; do cd ..; done
[ -f sctl_source ] || { echo "ERROR: cannot locate project root (no sctl_source found from '$root')"; exit 1; }
echo "project root: $PWD"

. ./sctl_source

make MPI=0 bin/capsule-robin-bie

BIN=./bin/capsule-robin-bie
OUT=vis/capsule-precond
mkdir -p "$OUT"

# fixed mesh/solve params
ORDER=12; FOURIER=16; NCAP=2; TOL=1e-8

# physical-core pinning for OpenMP (single process, no mpirun). Under sbatch, match the allocation
# (--cpus-per-task); a direct `bash` run falls back to 10.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=== capsule whole-arm-exact precond : order=$ORDER fourier=$FOURIER Ncap=$NCAP tol=$TOL ==="

# ---- Part 1: straight cases, off vs whole(exact) -------------------------------------------------
echo
echo "--- Part 1: straight arms (off vs whole-exact) ---"
STRAIGHT_CASES=(
  "L1R0p3   1  0.3"     # aspect  3.3  (short-thin)
  "L10R3    10 3"       # aspect  3.3  (long-fat)
  "L10R0p3  10 0.3"     # aspect 33    (slender)
)
for c in "${STRAIGHT_CASES[@]}"; do
  read -r label L R <<<"$c"
  for tag in off whole; do
    log="$OUT/straight-${label}-${tag}.log"
    echo ">>> straight case=$label L=$L R=$R variant=$tag  ->  $log"
    QJ_ARM_PRECOND=$tag \
      "$BIN" flow "$ORDER" "$R" "$L" 8 "$FOURIER" "$NCAP" "$TOL" 2>&1 | tee "$log"
  done
done

# ---- Part 2: spiral study, off vs straight-proxy vs actual-geometry ------------------------------
echo
echo "--- Part 2: spiral arms (off vs whole-straight-proxy vs whole-actual-geom) ---"
# "label length R_shaft n_axial spiral_amp spiral_turns"
SPIRAL_CASES=(
  "S_L10R0p3  10 0.3 12 0.5 1"   # slender helix (aspect 33), excursion ~ radius
  "S_L10R1    10 1   10 1.0 1"   # moderate helix (aspect 10)
)
# "tag  QJ_ARM_PRECOND  QJ_ARM_PRECOND_GEOM"
SPIRAL_VARIANTS=(
  "off            off       straight"
  "wholeStraight  whole     straight"
  "wholeActual    whole     actual"
)
for c in "${SPIRAL_CASES[@]}"; do
  read -r label L R NA AMP TURNS <<<"$c"
  for v in "${SPIRAL_VARIANTS[@]}"; do
    read -r tag pc geom <<<"$v"
    log="$OUT/${label}-${tag}.log"
    echo ">>> spiral case=$label L=$L R=$R n_axial=$NA amp=$AMP turns=$TURNS variant=$tag  ->  $log"
    QJ_SPIRAL_AMP=$AMP QJ_SPIRAL_TURNS=$TURNS QJ_ARM_PRECOND=$pc QJ_ARM_PRECOND_GEOM=$geom \
      "$BIN" flow "$ORDER" "$R" "$L" "$NA" "$FOURIER" "$NCAP" "$TOL" 2>&1 | tee "$log"
  done
done

# ---- Part 3: GROUPED-panel study on the amp=0.5 case ONLY (S_L10R0p3) -----------------------------
# Block-diagonal over groups of 2 ADJACENT panels (vs the per-panel proxy that dropped ALL inter-panel
# coupling and HURT: 54 vs off 42). group2Straight = straight-cylinder groups; group2Arced = the ACTUAL
# arced sub-tubes. Question: does keeping the adjacent-panel coupling recover the whole-arm gain (off 42,
# wholeActual 19)? group1Straight reproduces the old per-panel proxy (regression anchor, expect ~54).
echo
echo "--- Part 3: grouped-panel proxy on S_L10R0p3 (amp=0.5) : off vs group2 straight/arced ---"
GL=10; GR=0.3; GNA=12; GAMP=0.5; GTURNS=1
# "tag  QJ_ARM_PRECOND_GROUP  QJ_ARM_PRECOND_GEOM"
GROUP_VARIANTS=(
  "group1Straight  1  straight"
  "group2Straight  2  straight"
  "group2Arced     2  actual"
)
for v in "${GROUP_VARIANTS[@]}"; do
  read -r tag grp geom <<<"$v"
  log="$OUT/S_L10R0p3-${tag}.log"
  echo ">>> spiral case=S_L10R0p3 variant=$tag group=$grp geom=$geom  ->  $log"
  QJ_SPIRAL_AMP=$GAMP QJ_SPIRAL_TURNS=$GTURNS QJ_ARM_PRECOND=panelcyl \
    QJ_ARM_PRECOND_GROUP=$grp QJ_ARM_PRECOND_GEOM=$geom \
    "$BIN" flow "$ORDER" "$GR" "$GL" "$GNA" "$FOURIER" "$NCAP" "$TOL" 2>&1 | tee "$log"
done

# ---- Part 4: the STRADDLING (problematic) interior preconditioners -- ARM-ONLY and SEAM-ONLY --------
# The spectrum study flagged the TWO preconditioners whose interior eigenvalues STRADDLE the origin: the
# inverted arm/seam blocks are pushed to +1 while the un-preconditioned caps stay at the raw jump -1/2, so
# mass sits on BOTH sides of 0 -- the layout the analysis blames for the arm-only stall (invisible to the
# condition number). Those two are:
#   arm-only : QJ_ARM_PRECOND=whole  (whole-arm block inverted, caps + junction left at identity)
#   seam     : QJ_ARM_PRECOND=seam   (cap<->arm-seam coupled block, middle arm + interior left at identity)
# Run each with the default +I target and with QJ_PRECOND_SIGN=jump (sign(jump)*I = -I here -> inverted
# blocks land on the SAME negative side as the caps, no straddle), to test whether the consistent-sign fix
# helps the cases the spectra say are worst. Uses `mfs` mode with QJ_MFS_ONLY=int: it is INTERIOR-ONLY
# (the operator this whole study is about), it is the purpose-built GMRES x precond probe (reports iters
# AND rel-L2 accuracy), and -- unlike `flow` -- it is the mode `seam` is actually wired into. A left
# preconditioner cannot change the solution, so rel-L2 must match across all variants (correctness oracle).
#
# NB: every mfs-int solve runs the interior RHS net-flux compatibility check BEFORE GMRES
# (assert_rhs_flux_compatible): the Stokeslet trace from an EXTERIOR source is net-flux-free, so it prints
# "[mfs-int] RHS net flux Phi = ..." ~ 0 (in range). QJ_FLUX_TOL / QJ_FLUX_CHECK=0 adjust/relax it.
echo
echo "--- Part 4: problematic interior preconds (arm-only, seam) x sign {+I, jump=-I}, mfs-int ---"
P4L=10; P4R=0.3; P4NA=8
# "tag           QJ_ARM_PRECOND  QJ_PRECOND_SIGN"
PROB_VARIANTS=(
  "off           off    plus"
  "armOnlyPlus   whole  plus"
  "armOnlyJump   whole  jump"
  "seamPlus      seam   plus"
  "seamJump      seam   jump"
)
for v in "${PROB_VARIANTS[@]}"; do
  read -r tag pc sgn <<<"$v"
  log="$OUT/mfsProblem-${tag}.log"
  echo ">>> mfs-int slender L=$P4L R=$P4R variant=$tag precond=$pc sign=$sgn  ->  $log"
  QJ_MFS_ONLY=int QJ_ARM_PRECOND=$pc QJ_PRECOND_SIGN=$sgn \
    "$BIN" mfs "$ORDER" "$P4R" "$P4L" "$P4NA" "$FOURIER" "$NCAP" "$TOL" 2>&1 | tee "$log"
done

# ---- Part 5: ARM-ONLY sign flip on a SPIRAL, straight-proxy precond in two flavors ------------------
# Extends the Part-4 arm-only flip to a BENT (spiral) arm, where the straight proxy no longer matches the
# geometry exactly. Two straight-cylinder proxies, each arm-only (caps=identity), each +I vs jump(-I):
#   wholeStraight : ONE straight cylinder of the arm's ARC length (so the SAME aspect ratio as the spiral
#                   arm), QJ_ARM_PRECOND=whole QJ_ARM_PRECOND_GEOM=straight. Matches length/aspect but not
#                   the curvature (whole-arm axial coupling built straight).
#   cyl1Straight  : per-panel straight 1-panel cylinders (QJ_ARM_PRECOND=panelcyl group=1, straight): each
#                   block is a SHORTER straight cylinder (different, smaller aspect) approximating one
#                   curved spiral panel -- a self-to-self block with a slight geometric mismatch per panel.
# Question: on the bent arm (where these proxies mis-precondition -- flow showed wholeStraight/group1 HURT
# vs off), does the consistent-sign flip (-I, no origin straddle) still help, or does the geometric
# mismatch dominate? mfs-int (interior, generic Stokeslet RHS -- the RHS that exposes the straddle).
echo
echo "--- Part 5: arm-only sign flip on SPIRAL S_L10R0p3 (whole-straight & cyl-1-straight, +I vs -I) ---"
P5L=10; P5R=0.3; P5NA=12; P5AMP=0.5; P5TURNS=1
# "tag                QJ_ARM_PRECOND  QJ_ARM_PRECOND_GROUP  QJ_PRECOND_SIGN"
SPIRAL_FLIP_VARIANTS=(
  "off                off       1  plus"
  "wholeStraightPlus  whole     1  plus"
  "wholeStraightJump  whole     1  jump"
  "cyl1StraightPlus   panelcyl  1  plus"
  "cyl1StraightJump   panelcyl  1  jump"
)
for v in "${SPIRAL_FLIP_VARIANTS[@]}"; do
  read -r tag pc grp sgn <<<"$v"
  log="$OUT/spiralArmFlip-${tag}.log"
  echo ">>> spiral mfs-int L=$P5L R=$P5R na=$P5NA amp=$P5AMP variant=$tag precond=$pc group=$grp sign=$sgn  ->  $log"
  QJ_SPIRAL_AMP=$P5AMP QJ_SPIRAL_TURNS=$P5TURNS QJ_MFS_ONLY=int \
    QJ_ARM_PRECOND=$pc QJ_ARM_PRECOND_GROUP=$grp QJ_ARM_PRECOND_GEOM=straight QJ_PRECOND_SIGN=$sgn \
    "$BIN" mfs "$ORDER" "$P5R" "$P5L" "$P5NA" "$FOURIER" "$NCAP" "$TOL" 2>&1 | tee "$log"
done

echo
echo "=== SUMMARY : GMRES iters + rel-L2 + RHS net-flux check (grep) ==="
for f in "$OUT"/straight-*.log "$OUT"/S_*.log "$OUT"/mfsProblem-*.log "$OUT"/spiralArmFlip-*.log; do
  [ -e "$f" ] || continue
  it=$(grep -oE 'StokesBIO GMRES iters = [0-9]+' "$f" | grep -oE '[0-9]+$' || echo '?')
  rl=$(grep -oE 'rel-L2 = [0-9.e+-]+' "$f" | grep -oE '[0-9.e+-]+$' | head -1 || echo '-')
  flux=$(grep -oE 'RHS net flux  Phi = int b.n dA = [^ ]+' "$f" | grep -oE '[^ ]+$' | head -1 || echo '-')
  compat=$(grep -oE 'range/compatibility (OK|VIOLATED)' "$f" | grep -oE '(OK|VIOLATED)$' | head -1 || echo '-')
  printf "  %-34s iters=%-4s relL2=%-11s  Phi=%-12s  %s\n" "$(basename "$f")" "$it" "$rl" "$flux" "$compat"
done
