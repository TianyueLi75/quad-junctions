#!/bin/bash
# Scheme-comparison sweep + OpenMP strong scaling for bench-scheme-compare on one Icelake node.
#
# A copy of scripts/bench-cubed-sphere.sbatch specialised to compare the four singular-quadrature
# schemes {RP, Adaptive, Hybrid, Duffy} on a single machine. The convergence table's per-scheme
# columns are the analogue of the original bench's {Rome, Genoa, Icelake} machine columns.
#
# The (kernel x scheme) x (conv, omp) jobs are dispatched across SLURM_NTASKS_PER_NODE concurrent
# slots; each slot runs one job at a time with OMP_NUM_THREADS=SLURM_CPUS_PER_TASK pinned to its
# own disjoint set of PHYSICAL cores (taskset mask + OMP_PLACES=cores). Per-job outputs are written
# separately and then merged into one conv file and one omp file for the parser.
#
#   sbatch --ntasks-per-node=2 --cpus-per-task=32 scripts/bench-scheme-compare.sh
#
# Watch with `squeue -u $USER`; check the achieved efficiency afterwards with `seff <jobid>`.
#
# NOTE ON -march=native. The Makefile compiles with -march=native, so an Icelake binary carries
# AVX-512 and dies with SIGILL on the Zen 2 workstations, which share this home filesystem. Each
# architecture therefore builds into its own bin.<tag>/ and obj.<tag>/ and nothing writes ./bin.

#SBATCH --job-name=scheme-compare
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=doc/data/slurm-%j.out
#SBATCH --error=doc/data/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p doc/data

source ./sctl_source

NT="${SLURM_CPUS_PER_TASK:-$(nproc)}"
TAG="${SLURM_JOB_CONSTRAINTS:-icelake}"

echo "host       : $(hostname)"
echo "cpu        : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "sockets    : $(lscpu | awk -F: '/^Socket\(s\)/{print $2}' | tr -d ' ')"
echo "avx512f    : $(grep -c -m1 avx512f /proc/cpuinfo || true)"
echo "tag        : $TAG"
echo "threads    : $NT"
echo

bash bench/scheme-compare/build.sh "$TAG"
BIN="bench/scheme-compare/bin.$TAG/bench-scheme-compare"

# Bind OpenMP threads to physical cores; NT = SLURM_CPUS_PER_TASK threads per job.
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS="$NT"

KERNELS="laplace stokes"
SCHEMES="RP Adaptive Hybrid Duffy"

# Concurrency = tasks-per-node; each slot gets its own NT physical cores.
NSLOTS="${SLURM_NTASKS_PER_NODE:-1}"

# Physical-core list that respects the Slurm cgroup: expand the logical CPUs this step is actually
# allowed to use (/proc/self/status), then keep one logical CPU per physical core (dedup by CORE).
expand_list() { local x; for x in ${1//,/ }; do case $x in *-*) seq "${x%-*}" "${x#*-}";; *) echo "$x";; esac; done; }
ALLOWED_RAW="$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
declare -A CORE_OF
while IFS=, read -r cpu core _; do CORE_OF[$cpu]=$core; done \
  < <(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#')
PCPUS=(); declare -A SEEN_CORE
for cpu in $(expand_list "$ALLOWED_RAW" | sort -n); do
  core="${CORE_OF[$cpu]:-$cpu}"
  if [ -z "${SEEN_CORE[$core]:-}" ]; then SEEN_CORE[$core]=1; PCPUS+=("$cpu"); fi
done
NPHYS=${#PCPUS[@]}
echo "phys cores  : $NPHYS available (allowed=$ALLOWED_RAW)"
echo "dispatch    : $NSLOTS slots x $NT threads/job"
if [ $((NSLOTS * NT)) -gt "$NPHYS" ]; then
  echo "*** WARNING: NSLOTS*NT = $((NSLOTS * NT)) > $NPHYS physical cores -- slots may oversubscribe, pts/s meaningless"
fi
echo

# Contiguous NT-physical-core taskset list for a given slot (empty if we run out of cores).
slot_cpulist() {
  local s=$1 i idx start=$(( $1 * NT )) list=""
  for ((i = 0; i < NT; i++)); do
    idx=$(( start + i )); [ "$idx" -lt "$NPHYS" ] || break
    list+="${list:+,}${PCPUS[$idx]}"
  done
  echo "$list"
}

# Job list: every (kernel, scheme) for conv, then for omp.
JOBS=()
for kernel in $KERNELS; do for scheme in $SCHEMES; do JOBS+=("conv $kernel $scheme"); done; done
for kernel in $KERNELS; do for scheme in $SCHEMES; do JOBS+=("omp $kernel $scheme");  done; done
NJOBS=${#JOBS[@]}

PARTS="doc/data/parts-${TAG}-${SLURM_JOB_ID:-local}"
rm -rf "$PARTS"; mkdir -p "$PARTS"

# Dynamic work queue: a shared counter (flock-serialised) hands out the next job index. Each slot
# pulls the next job as soon as it frees up, so a slow job (e.g. RectPolar at large Nbeta) on one
# slot never blocks the others -- unlike a static round-robin stripe, which can pile all the
# expensive schemes onto one slot.
CTR="$PARTS/.next"; echo 0 > "$CTR"
LOCK="$PARTS/.lock"; : > "$LOCK"

next_job() {                 # echo the next unclaimed index, or nothing when the queue is drained
  local n
  exec 9>"$LOCK"; flock 9
  n=$(<"$CTR")
  [ "$n" -lt "$NJOBS" ] && echo $((n + 1)) > "$CTR"
  flock -u 9
  [ "$n" -lt "$NJOBS" ] && echo "$n"
}

run_slot() {
  local slot=$1 cpulist k mode kernel scheme out t0 t1
  cpulist="$(slot_cpulist "$slot")"
  while k="$(next_job)"; [ -n "$k" ]; do
    read -r mode kernel scheme <<< "${JOBS[$k]}"
    out="$PARTS/${mode}.${kernel}.${scheme}"
    t0=$SECONDS
    echo "[slot $slot cpus=${cpulist:-<all>}] START $mode $kernel $scheme"
    if [ -n "$cpulist" ]; then
      taskset -c "$cpulist" "$BIN" "$mode" "$kernel" "$scheme" "$NT" > "$out.txt" 2> "$out.err" || echo "  !! $mode $kernel $scheme exited nonzero (see $out.err)"
    else
      "$BIN" "$mode" "$kernel" "$scheme" "$NT" > "$out.txt" 2> "$out.err" || echo "  !! $mode $kernel $scheme exited nonzero (see $out.err)"
    fi
    t1=$SECONDS
    echo "[slot $slot] DONE  $mode $kernel $scheme  ($((t1 - t0)) s)"
  done
}

for ((s = 0; s < NSLOTS; s++)); do run_slot "$s" & done
wait

# --- merge per-job parts into one conv file and one omp file, then parse ---
CONV_OUT="doc/data/scheme-compare-conv-${TAG}.txt"
OMP_OUT="doc/data/scheme-compare-omp-${TAG}.txt"
cat "$PARTS"/conv.*.txt > "$CONV_OUT" 2>/dev/null || : > "$CONV_OUT"
cat "$PARTS"/omp.*.txt  > "$OMP_OUT"  2>/dev/null || : > "$OMP_OUT"

bash scripts/parse-scheme-compare.sh "$CONV_OUT" "$OMP_OUT" "doc/data/scheme-compare"
echo
echo "# per-job logs   : $PARTS/{conv,omp}.<kernel>.<scheme>.{txt,err}"
echo "# merged inputs  : $CONV_OUT , $OMP_OUT"
echo "# wrote          : doc/data/scheme-compare-{conv,omp}.tex"
