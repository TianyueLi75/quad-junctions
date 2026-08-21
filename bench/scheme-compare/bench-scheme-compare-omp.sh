#!/usr/bin/env bash
#
# OMP THREAD-SCALING sub-study for bench-scheme-compare, run for EACH of the four singular-quadrature
# schemes {RP, Adaptive, Hybrid, Duffy} on a FULL Icelake node.
#
# This is a copy of the OMP portion of ~/quad-junctions/scripts/twisted_sphere_run.sh, specialised
# to the scheme comparison. It differs from scripts/bench-scheme-compare.sh (the combined conv+omp
# sweep) in one important way: that script splits the node into SLURM_NTASKS_PER_NODE concurrent
# slots, so its OMP scaling is capped at SLURM_CPUS_PER_TASK threads. Here we take the WHOLE node
# with a single task (ntasks-per-node=1, cpus-per-task=<all phys cores>) so the strong-scaling study
# reaches the full core count.
#
# For each (kernel x scheme) the bench-scheme-compare binary's `omp` mode is invoked once; the binary
# internally descends the OpenMP width over the full node then powers of two down to 1 (order 12,
# ppf 8, twist pi/6, tol 1e-9) and emits one `@@ROW ... thr=<n> ...` line per width. Threads bind to
# distinct physical cores via OMP_PLACES=cores / OMP_PROC_BIND=close; the process is taskset-pinned
# to one logical CPU per physical core (cgroup-aware) so no width oversubscribes.
#
# The (kernel x scheme) jobs run SEQUENTIALLY -- each wants the whole node -- unlike the parallel
# slot dispatch in the combined script. Per-scheme output is kept in its own part file and then
# merged into one omp file for the parser.
#
#   sbatch scripts/bench-scheme-compare-omp.sh
#
# Watch with `squeue -u $USER`; check achieved efficiency afterwards with `seff <jobid>`.
#
# NOTE ON -march=native: the Makefile compiles with -march=native, so an Icelake binary carries
# AVX-512 and SIGILLs on the Zen 2 workstations sharing this home filesystem. Each architecture
# therefore builds into its own bin.<tag>/ and obj.<tag>/.

#SBATCH --job-name=scheme-compare-omp
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:30:00
#SBATCH --output=doc/data/slurm-omp-%j.out
#SBATCH --error=doc/data/slurm-omp-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p doc/data

source ./sctl_source

TAG="${SLURM_JOB_CONSTRAINTS:-icelake}"

echo "host       : $(hostname)"
echo "cpu        : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "sockets    : $(lscpu | awk -F: '/^Socket\(s\)/{print $2}' | tr -d ' ')"
echo "avx512f    : $(grep -c -m1 avx512f /proc/cpuinfo || true)"
echo "tag        : $TAG"
echo

bash bench/scheme-compare/build.sh "$TAG"
BIN="bench/scheme-compare/bin.$TAG/bench-scheme-compare"

# --- physical-core list that respects the Slurm cgroup: expand the logical CPUs this step is
#     actually allowed to use (/proc/self/status), then keep one logical CPU per physical core. ---
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
CPULIST="$(IFS=,; echo "${PCPUS[*]}")"     # one logical CPU per physical core, the whole node
echo "phys cores : $NPHYS available (allowed=$ALLOWED_RAW)"
echo "cpulist    : $CPULIST"
echo

# Bind OpenMP threads to physical cores; the binary descends the width from NT down to 1.
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS="$NPHYS"
NT="$NPHYS"

KERNELS="laplace stokes"
SCHEMES="RP Adaptive Hybrid Duffy"

PARTS="doc/data/parts-omp-${TAG}-${SLURM_JOB_ID:-local}"
rm -rf "$PARTS"; mkdir -p "$PARTS"

# --- run each (kernel, scheme) SEQUENTIALLY on the full node; the binary sweeps the OMP widths. ---
for kernel in $KERNELS; do
  for scheme in $SCHEMES; do
    out="$PARTS/omp.${kernel}.${scheme}"
    t0=$SECONDS
    echo "########################################################################"
    echo "# omp-study  kernel=$kernel  scheme=$scheme  nt_max=$NT  cores={$CPULIST}"
    echo "########################################################################"
    taskset -c "$CPULIST" "$BIN" omp "$kernel" "$scheme" "$NT" > "$out.txt" 2> "$out.err" \
      || echo "  !! omp $kernel $scheme exited nonzero (see $out.err)"
    cat "$out.txt"
    echo "# DONE $kernel $scheme ($(( SECONDS - t0 )) s)"
    echo
  done
done

# --- merge per-scheme parts into one omp file, then parse into per-scheme sections. ---
OMP_OUT="doc/data/scheme-compare-omp-fullnode-${TAG}.txt"
cat "$PARTS"/omp.*.txt > "$OMP_OUT" 2>/dev/null || : > "$OMP_OUT"

bash scripts/parse-scheme-compare-omp.sh "$OMP_OUT" "doc/data/scheme-compare-omp-fullnode"
echo
echo "# per-scheme logs : $PARTS/omp.<kernel>.<scheme>.{txt,err}"
echo "# merged input    : $OMP_OUT"
echo "# wrote           : doc/data/scheme-compare-omp-fullnode.tex"
