#!/bin/bash
# Laplace single-layer SETUP-TIME sweep over the project's canonical near-eval parameter sets:
#   tol=1e-5 (Nbeta=48,md=4) | 1e-7 (100,8) | 1e-9 (200,12) | 1e-11 (400,30)   [cov_q=6]
# fmm3dbie eps = the same tol at each level. Orders 8 and 12, SINGLE-THREADED.
# Speed = Nnodes / (SetupSingular + SetupNear)  [SCTL]  =  Nnodes / (near-list + getnearquad) [fmm3dbie].
set -uo pipefail
cd "$(dirname "$0")"; HERE="$(pwd)"; ROOT="$(cd ../.. && pwd)"
. "$ROOT/sctl_source" 2>/dev/null || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=true OMP_PLACES=cores

TOLS=(1e-5 1e-7 1e-9 1e-11)
NBS=( 48   100  200  400  )
MDS=( 4    8    12   30   )
COVQ=6; NREF=2

for o in 8 12; do
  mesh="ybifurc_p${o}.srcvals"
  for k in 0 1 2 3; do
    tol=${TOLS[$k]}; nb=${NBS[$k]}; md=${MDS[$k]}
    echo "## order $o  tol $tol (Nb=$nb md=$md)  $(date +%T)"
    ( cd "$ROOT" && ./bin/ybifurc-laplace-selfsetup "$o" "$NREF" "$tol" "$nb" "$md" "$COVQ" ) \
        > "$HERE/lap_sctl_o${o}_${tol}.txt" 2>&1
    ./lap_total_setup "$mesh" "$tol" > "$HERE/lap_fmm_o${o}_${tol}.txt" 2>&1
    echo "   done order $o tol $tol"
  done
done
echo "ALL_LAP_SWEEP_DONE $(date +%T)"

# ---------------- tabulate ----------------
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }
speed_sctl(){ strip <"$1" | awk '
  /Nnodes=/{ for(i=1;i<=NF;i++) if($i ~ /^Nnodes=/){split($i,a,"="); N=a[2]} }
  /\+-SetupSingular/{s=$NF} /\+-SetupNear/{r=$NF}
  END{ if(s!=""&&r!=""&&N!="") printf "%.1f", N/(s+r); else printf "NA" }'; }
speed_fmm(){ strip <"$1" | awk '
  /Nnodes=/{ for(i=1;i<=NF;i++) if($i ~ /^Nnodes=/){split($i,a,"="); N=a[2]} }
  /near-list build/{nl=$NF} /getnearquad self\+near/{gq=$NF}
  END{ if(nl!=""&&gq!=""&&N!="") printf "%.1f", N/(nl+gq); else printf "NA" }'; }

echo
echo "============ LAPLACE-SL SETUP SPEED (nodes/s, single-thread, sing+near only) ============"
for o in 8 12; do
  echo "  --- order $o ---"
  printf "   %-8s | %-12s | %-12s | %s\n" "tol" "SCTL" "fmm3dbie" "SCTL/fmm3dbie"
  for tol in "${TOLS[@]}"; do
    ss=$(speed_sctl "lap_sctl_o${o}_${tol}.txt"); sf=$(speed_fmm "lap_fmm_o${o}_${tol}.txt")
    ratio=$(awk -v a="$ss" -v b="$sf" 'BEGIN{ if(a!="NA"&&b!="NA"&&b>0) printf "%.1fx",a/b; else printf "-" }')
    printf "   %-8s | %-12s | %-12s | %s\n" "$tol" "$ss" "$sf" "$ratio"
  done
done
echo "=========================================================================================="
echo "done $(date)"
