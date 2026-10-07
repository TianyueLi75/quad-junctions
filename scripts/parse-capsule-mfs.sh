#!/bin/bash
# =============================================================================
# parse-capsule-mfs.sh -- print two GMRES-iteration tables (rows = preconditioner, cols = geometry)
# from the logs produced by scripts/run-capsule-mfs-precond.sh: one for the EXTERIOR solve (mfs-ext)
# and one for the INTERIOR solve (mfs-int).
#
# Log filenames are vis/capsule-mfs/<problem>-<geom>-<precond>.log  (problem in {int,ext}). The
# iteration count and rel-L2 are read from the driver line:
#   [mfs-int|mfs-ext] StokesBIO GMRES iters = <n>  rel-L2 = <e> ...
#
# Usage:
#   scripts/parse-capsule-mfs.sh                 # default log dir vis/capsule-mfs
#   scripts/parse-capsule-mfs.sh <logdir>
# =============================================================================
set -uo pipefail

LOGDIR="${1:-$(dirname "$0")/../vis/capsule-mfs}"
if [ ! -d "$LOGDIR" ]; then
  echo "no log directory $LOGDIR -- run scripts/run-capsule-mfs-precond.sh first" >&2
  exit 1
fi

PRECONDS=( none panelExact panelCyl armWhole )        # rows (fixed order)
GEOMS=( straightL1 straightL10 spiralL10 )            # cols (fixed order)

# cell(problem, precond, geom) -> "iters" (or "?"); also records rel-L2 into REL[...]
declare -A ITER REL
for prob in ext int; do
  for pc in "${PRECONDS[@]}"; do
    for g in "${GEOMS[@]}"; do
      f="$LOGDIR/${prob}-${g}-${pc}.log"
      it="?"; rl=""
      if [ -e "$f" ]; then
        line=$(grep -oE 'StokesBIO GMRES iters = [0-9]+[^)]*' "$f" | head -1 || true)
        if [ -n "$line" ]; then
          it=$(printf '%s\n' "$line" | grep -oE 'iters = [0-9]+' | grep -oE '[0-9]+')
          rl=$(printf '%s\n' "$line" | grep -oE 'rel-L2 = [0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$' || true)
        fi
      fi
      ITER["$prob|$pc|$g"]="${it:-?}"
      REL["$prob|$pc|$g"]="$rl"
    done
  done
done

# column width: widest geometry label + padding
colw=0
for g in "${GEOMS[@]}"; do [ "${#g}" -gt "$colw" ] && colw=${#g}; done
colw=$(( colw + 3 ))
roww=0
for pc in "${PRECONDS[@]}"; do [ "${#pc}" -gt "$roww" ] && roww=${#pc}; done

print_table() {   # $1=problem key  $2=title
  local prob="$1" title="$2"
  printf '\n=== %s : GMRES iterations ===\n' "$title"
  printf "%-${roww}s  " ""
  for g in "${GEOMS[@]}"; do printf "%${colw}s" "$g"; done
  printf '\n'
  local width=$(( roww + 2 + colw * ${#GEOMS[@]} ))
  printf '%*s\n' "$width" '' | tr ' ' '-'
  for pc in "${PRECONDS[@]}"; do
    printf "%-${roww}s  " "$pc"
    for g in "${GEOMS[@]}"; do printf "%${colw}s" "${ITER[$prob|$pc|$g]}"; done
    printf '\n'
  done
}

print_table ext "EXTERIOR (mfs-ext)"
print_table int "INTERIOR (mfs-int)"

# rel-L2 correctness sanity dump (small = valid manufactured solve)
printf '\n--- rel-L2 (correctness oracle; small = valid manufactured solve) ---\n'
for prob in ext int; do
  for pc in "${PRECONDS[@]}"; do
    for g in "${GEOMS[@]}"; do
      rl="${REL[$prob|$pc|$g]}"
      [ -n "$rl" ] && printf '  %3s %-12s %-11s iters=%4s  rel-L2=%s\n' \
        "$prob" "$g" "$pc" "${ITER[$prob|$pc|$g]}" "$rl"
    done
  done
done
