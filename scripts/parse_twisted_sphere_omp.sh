#!/usr/bin/env bash
#
# Parse the OMP THREAD-SCALING sub-study log from twisted_sphere_run.sh (out/twisted-sphere-omp-*.log)
# and print the same setup table as parse_twisted_sphere.sh, but keyed on the OpenMP thread count.
#
# The sub-study runs one `stokes_greens` process per OMP thread count {1,2,4,8,16,32} at a FIXED parameter set
# (tol=1e-9, Nbeta=200, max_depth=12, twist=pi, order 12/ppf 8), each profiled with a cold Setup. 
# Per run this pulls, from the profiler tree + summary lines:
#
#     SL SetupSingular / SL SetupNear   (first  Setup() = the SL operator)   t_avg (s) + f/s_avg (GFLOP/s)
#     DL SetupSingular / DL SetupNear   (second Setup() = the DL operator)   t_avg (s) + f/s_avg (GFLOP/s)
#     Nnodes                            (STOKES-GREENS SETUP line)
#     Green's-identity max_rel_err      (STOKES-GREENS-IDENTITY line)
#
# and adds nodes/s = Nnodes / setup_tot and a speedup column (setup_tot at omp=1 / setup_tot). The
# fixed tol/Nbeta/max_depth/twist are printed once in a note line (they should be identical across rows).
#
# NB: this is a SEPARATE parser because the OMP log anchors on `# omp=` block headers, whereas
# parse_twisted_sphere.sh anchors on `# core=` (the twist-sweep blocks) and carries no omp column.
#
# Usage:  scripts/parse_twisted_sphere_omp.sh [logfile]
#         (default: newest out/twisted-sphere-omp-*.log)

set -euo pipefail

LOG=${1:-}
if [ -z "$LOG" ]; then
  LOG=$(ls -t out/twisted-sphere-omp-*.log 2>/dev/null | head -1 || true)
  [ -z "$LOG" ] && { echo "no logfile given and no out/twisted-sphere-omp-*.log found" >&2; exit 1; }
fi
[ -r "$LOG" ] || { echo "cannot read log: $LOG" >&2; exit 1; }
echo "# parsing: $LOG"
echo "# times SL_sing..setup_tot in seconds (t_avg); *_f columns in GFLOP/s (f/s_avg); speedup = setup_tot(omp=1)/setup_tot"

sed -e 's/\x1b\[[0-9;]*m//g' "$LOG" | awk '
  function twlabel(t,   x) {
    x = t + 0;
    if (x < 1e-9)             return "0";
    if (x > 1.50 && x < 1.60) return "pi/2";
    if (x > 3.10 && x < 3.20) return "pi";
    return sprintf("%.4g", x);
  }
  # Two numeric columns after scope name `pat`: t_avg (s) then f/s_avg (GFLOP/s) -> globals _t, _fs.
  function grabpair(pat,   i, found, cnt) {
    found = 0; cnt = 0; _t = ""; _fs = "";
    for (i = 1; i <= NF; i++) {
      if (!found) { if ($i ~ pat) found = 1; continue; }
      if ($i ~ /^[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$/) {
        if (cnt == 0) { _t = $i + 0; cnt = 1; }
        else          { _fs = $i + 0; return; }
      }
    }
  }
  function fsfmt(v) { return (v == "") ? "     NA" : sprintf("%7.2f", v + 0); }
  function flush(   tot, thru, spd) {
    if (!have) return;
    tot = ssing[0] + snear[0] + ssing[1] + snear[1];
    thru = (tot > 0 && nnodes != "") ? nnodes / tot : 0;
    if (base_tot == "" && tot > 0) base_tot = tot;              # first row (omp=1) sets the baseline
    spd = (base_tot != "" && tot > 0) ? base_tot / tot : 0;
    printf "%4s  %8.3f %8.3f %8.3f %8.3f %9.3f  %7s %7s %7s %7s  %8s %11.3e %10.3e  %7.2f\n",
      omp,
      ssing[0]+0, snear[0]+0, ssing[1]+0, snear[1]+0, tot,
      fsfmt(ssing_fs[0]), fsfmt(snear_fs[0]), fsfmt(ssing_fs[1]), fsfmt(snear_fs[1]),
      (nnodes==""?"NA":nnodes), thru, (gerr==""?0:gerr) + 0, spd;
  }
  BEGIN { have = 0; nsing = 0; nnear = 0; base_tot = ""; noted = 0; }
  # --- block header: flush previous, reset, capture omp + the (fixed) tol/Nbeta/max_depth/twist ---
  /^# omp=/ {
    flush();
    have = 1; nsing = 0; nnear = 0;
    omp = tol = nbeta = md = twist = nnodes = gerr = "";
    ssing[0]=ssing[1]=snear[0]=snear[1]="";
    ssing_fs[0]=ssing_fs[1]=snear_fs[0]=snear_fs[1]="";
    line = $0; gsub(/[()]/, " ", line);
    ntok = split(line, tok, /[ \t]+/);
    for (i = 1; i <= ntok; i++) {
      if      (tok[i] ~ /^omp=/)       { omp   = substr(tok[i], 5); }
      else if (tok[i] ~ /^tol=/)       { tol   = substr(tok[i], 5); }
      else if (tok[i] ~ /^Nbeta=/)     { nbeta = substr(tok[i], 7); }
      else if (tok[i] ~ /^max_depth=/) { md    = substr(tok[i], 11); }
      else if (tok[i] ~ /^twist=/)     { twist = substr(tok[i], 7); }
    }
    if (!noted) {                                              # emit the fixed-parameter note + header once
      printf "# fixed: tol=%s Nbeta=%s max_depth=%s twist=%s\n", tol, nbeta, md, twlabel(twist);
      printf "%4s  %8s %8s %8s %8s %9s  %7s %7s %7s %7s  %8s %11s %10s  %7s\n",
        "omp","SL_sing","SL_near","DL_sing","DL_near","setup_tot",
        "SLsng_f","SLnr_f","DLsng_f","DLnr_f","Nnodes","nodes/s","greens_err","speedup";
      printf "%4s  %8s %8s %8s %8s %9s  %7s %7s %7s %7s  %8s %11s %10s  %7s\n",
        "----","--------","--------","--------","--------","---------",
        "-------","-------","-------","-------","--------","-----------","----------","-------";
      noted = 1;
    }
    next;
  }
  # --- profiler tree rows (1st Setup = SL, 2nd = DL); +- excludes VERBOSE {}/} traces; two cols each ---
  /\+-SetupSingular/ { grabpair("SetupSingular"); if (_t != "" && nsing < 2) { ssing[nsing] = _t; ssing_fs[nsing] = _fs; nsing++; } next; }
  /\+-SetupNear/     { grabpair("SetupNear");     if (_t != "" && nnear < 2) { snear[nnear] = _t; snear_fs[nnear] = _fs; nnear++; } next; }
  # --- summary lines ---
  /STOKES-GREENS SETUP/ {
    if (match($0, /Nnodes=[0-9]+/)) nnodes = substr($0, RSTART+7, RLENGTH-7);
    next;
  }
  /STOKES-GREENS-IDENTITY/ {
    if (match($0, /max_rel_err=[0-9.eE+-]+/)) gerr = substr($0, RSTART+12, RLENGTH-12);
    next;
  }
  END { flush(); }
'
