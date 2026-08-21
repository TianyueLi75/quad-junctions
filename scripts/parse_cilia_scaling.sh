#!/bin/bash
# =============================================================================
# parse_cilia_scaling.sh -- parse cilia-carpet strong/weak scaling logs (bash/awk).
#
# Same output as parse_cilia_scaling.py, per srun step:
#   SLsing SLnear DLsing DLnear = SetupSingular/SetupNear of the SL (1st) then
#                                 DL (2nd) "+-Setup" block in the setup summary
#   Setup(s)                    = the "+-cilia_carpet_setup" total node
#   iters solve(s) s/iter       = from the "flow: GMRES converged ..." line
#   geom(s)                     = the "+-cilia_geometry_build" total node (always present)
#   tClose/tSubp/tCell cpIt fbFrac fbStall fbMax fbDist allDist elevF mOrd cellsT elevCF
#                               = from the "[nearbench]" line (BENCH=1 builds ONLY; "-" otherwise).
#                                 fbStall/fbMax = fraction of fallbacks that were early line-search
#                                 stalls vs iteration-exhaustion; fbDist/allDist = mean foot distance
#                                 of fallback targets vs all near targets (fbDist<<allDist + stall-
#                                 dominated = spurious floor-limited fallback).
#                                 Sub-SetupNear attribution: closest-point / subpanel-interp /
#                                 cell-integrate seconds (summed over threads*ranks), mean CP iters,
#                                 grid-fallback fraction, angle-elevated-order fraction, mean near GL
#                                 order, cells per target, elevated-order cell fraction.
# All times are t_avg (the first of the profiler's t_avg/t_max pair). Runs are
# split at the driver's "[strong]"/"[weak]" header lines (ranks=, Npatch=).
# SetupSingular/SetupNear appear as "+-" tree rows ONLY in the setup summary
# (the GMRES summary has SetupFMM/EvalNear instead), so counting occurrences
# per run yields exactly SL then DL.
#
# Usage:
#   scripts/parse_cilia_scaling.sh out/cilia_weak-<jobid>.log     
#   scripts/parse_cilia_scaling.sh out/cilia_weak-*.log          # multiple files ok
#   scripts/parse_cilia_scaling.sh --csv out/cilia_strong-*.log > strong.csv
# =============================================================================
set -euo pipefail

CSV=0
if [ "${1:-}" = "--csv" ]; then CSV=1; shift; fi
if [ "$#" -lt 1 ]; then
  echo "usage: $0 [--csv] LOG [LOG ...]" >&2
  exit 2
fi

awk -v csv="$CSV" '
function flush() {
  if (have) {
    rows[nr] = sprintf("%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
                       kind, ranks, npatch,
                       f(sing1), f(near1), f(sing2), f(near2), f(total),
                       (iters=="" ? "-" : iters), f(solve), f(periter),
                       f(geom), f(nb_tclose), f(nb_tsubp), f(nb_tcell), f(nb_cpit), f(nb_fb),
                       f(nb_fbstall), f(nb_fbmax), f(nb_fbdist), f(nb_alldist),
                       f(nb_elevf), f(nb_mord), f(nb_cellst), f(nb_elevc))
    nr++
  }
}
function f(v) { return (v=="" ? "-" : v) }
# first numeric token after "+-<name>" on the current (ANSI-stripped) line
function numafter(name,   idx, rest, a, n, i) {
  idx = index(line, "+-" name)
  if (idx == 0) return ""
  rest = substr(line, idx + length(name) + 2)
  n = split(rest, a, /[ \t]+/)
  for (i = 1; i <= n; i++) if (a[i] ~ /^[0-9.]+$/) return a[i]
  return ""
}
# value of "key=<val>" on the [nearbench] line (val = up to next space)
function kv(key,   idx, rest, a, n) {
  idx = index(line, key "=")
  if (idx == 0) return ""
  rest = substr(line, idx + length(key) + 1)
  n = split(rest, a, /[ \t]+/)
  return a[1]
}
BEGIN { have=0; nr=0 }
{
  line = $0
  gsub(/\033\[[0-9;]*m/, "", line)   # strip ANSI colour codes

  # A RUN header is a "[strong]"/"[weak]" line that carries ranks= (the driver header). The submit script
  # also emits secondary "[weak]  VSCALE=... z_plate=..." annotation lines with NO ranks=; those must NOT
  # start a new run (else they flush a half-empty row and reset ranks/Npatch to 0).
  if (line ~ /^\[(strong|weak)\]/ && line ~ /ranks=/) {
    flush()
    have=1
    kind = (line ~ /^\[strong\]/) ? "strong" : "weak"
    ranks=0; npatch=0
    sing1=sing2=near1=near2=total=iters=solve=periter=""
    geom=nb_tclose=nb_tsubp=nb_tcell=nb_cpit=nb_fb=nb_elevf=nb_mord=nb_cellst=nb_elevc=""
    nb_fbstall=nb_fbmax=nb_fbdist=nb_alldist=""
    nsing=0; nnear=0
    if (match(line, /ranks=[0-9]+/))  { s=substr(line,RSTART,RLENGTH); sub(/ranks=/,"",s);  ranks=s }
    if (match(line, /Npatch=[0-9]+/)) { s=substr(line,RSTART,RLENGTH); sub(/Npatch=/,"",s); npatch=s }
    next
  }
  if (!have) next

  if (line ~ /\+-cilia_carpet_setup[ \t]/)       { total = numafter("cilia_carpet_setup") }
  else if (line ~ /\+-cilia_geometry_build[ \t]/) { geom  = numafter("cilia_geometry_build") }
  else if (line ~ /\+-SetupSingular[ \t]/) { v=numafter("SetupSingular"); if(++nsing==1) sing1=v; else sing2=v }
  else if (line ~ /\+-SetupNear[ \t]/)     { v=numafter("SetupNear");     if(++nnear==1) near1=v; else near2=v }

  # Optional per-target near-setup breakdown (BENCH=1 builds only); absent runs keep "-".
  if (line ~ /\[nearbench\]/) {
    nb_tclose=kv("t_closest"); nb_tsubp=kv("t_subpanel"); nb_tcell=kv("t_cellint")
    nb_cpit=kv("cp_iters");    nb_fb=kv("fallback");      nb_elevf=kv("elev_frac")
    nb_mord=kv("mean_order");  nb_cellst=kv("cells_per_tgt"); nb_elevc=kv("elev_cell_frac")
    nb_fbstall=kv("fb_stall_frac"); nb_fbmax=kv("fb_max_frac")
    nb_fbdist=kv("mean_fb_dist");    nb_alldist=kv("mean_all_dist")
  }

  if (line ~ /GMRES converged in/) {
    if (match(line, /converged in [0-9]+ iters/)) { s=substr(line,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); iters=s }
    if (match(line, /avg solve[ \t]*=[ \t]*[0-9.]+/))    { s=substr(line,RSTART,RLENGTH); sub(/.*=[ \t]*/,"",s); solve=s }
    if (match(line, /avg per-iter[ \t]*=[ \t]*[0-9.]+/)) { s=substr(line,RSTART,RLENGTH); sub(/.*=[ \t]*/,"",s); periter=s }
  }
}
END {
  flush()
  if (nr == 0) { print "no scaling runs found in the given log(s)" > "/dev/stderr"; exit 1 }

  split("kind ranks Npatch SLsing(s) SLnear(s) DLsing(s) DLnear(s) Setup(s) iters solve(s) s/iter geom(s) tClose(s) tSubp(s) tCell(s) cpIt fbFrac fbStall fbMax fbDist allDist elevF mOrd cellsT elevCF", hdr, " ")
  ncol = 25

  if (csv == 1) {
    line = hdr[1]; for (c=2;c<=ncol;c++) line = line "," hdr[c]; print line
    for (r=0;r<nr;r++) { n=split(rows[r], a, "\t"); line=a[1]; for(c=2;c<=ncol;c++) line=line","a[c]; print line }
    exit 0
  }

  # column widths for a right-justified table
  for (c=1;c<=ncol;c++) w[c]=length(hdr[c])
  for (r=0;r<nr;r++) { n=split(rows[r], a, "\t"); for(c=1;c<=ncol;c++) if(length(a[c])>w[c]) w[c]=length(a[c]) }

  out=""; for(c=1;c<=ncol;c++) out=out (c>1?"  ":"") sprintf("%*s", w[c], hdr[c]); print out
  out=""; for(c=1;c<=ncol;c++){ d=""; for(i=0;i<w[c];i++) d=d"-"; out=out (c>1?"  ":"") d } print out
  for (r=0;r<nr;r++) { n=split(rows[r], a, "\t"); out=""; for(c=1;c<=ncol;c++) out=out (c>1?"  ":"") sprintf("%*s", w[c], a[c]); print out }
}
' "$@"
