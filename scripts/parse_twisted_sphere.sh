#!/usr/bin/env bash
#
# Parse a twisted_sphere_run.sh sweep log and print a per-swept-point SETUP table.
#
# The `stokes_greens` mode profiles a single cold Setup() of the SL and DL Stokes BoundaryIntegralOps
# (after a warm-up Setup + ClearSetup), then reports the on-surface Green's-identity error. For each
# swept point (tol tier x twist) this script pulls, from the profiler tree and summary lines:
#
#     SL SetupSingular / SL SetupNear   (first  Setup() = the SL operator)
#     DL SetupSingular / DL SetupNear   (second Setup() = the DL operator)
#     Nnodes                            (STOKES-GREENS SETUP line)
#     Green's-identity max_rel_err      (STOKES-GREENS-IDENTITY line)
#
# and prints one row per point plus the setup throughput = Nnodes / (SL+DL SetupSingular+SetupNear),
# the headline metric this sweep exists to measure (cf. the driver: "speed = Nnodes / t_avg(...)").
#
# NOTE: Green's identity is a DIRECT test (two SL/DL applies), NOT an iterative solve -- there is no
# GMRES phase, iteration count, or per-solve time in this log, so those columns are intentionally absent.
#
# Anchors used (robust to the parallel-batch interleaving, since each task's output is buffered and
# printed as a contiguous block):
#   - block header  "# core=.. tol=X (Nbeta=N max_depth=M) twist=T order=O ppf=P"   (params; I emit this)
#   - profiler rows "...+-SetupSingular <t_avg>" / "...+-SetupNear <t_avg>"          (the +- = tree rows,
#                                                                                     not the VERBOSE {})
#   - "STOKES-GREENS SETUP ... Nnodes=NN ..."
#   - "STOKES-GREENS-IDENTITY ... max_rel_err=E"
#
# Usage:  scripts/parse_twisted_sphere.sh [logfile]
#         (default: newest out/twisted-sphere-sweep-*.log)

set -euo pipefail

LOG=${1:-}
if [ -z "$LOG" ]; then
  LOG=$(ls -t out/twisted-sphere-sweep-*.log 2>/dev/null | head -1 || true)
  [ -z "$LOG" ] && { echo "no logfile given and no out/twisted-sphere-sweep-*.log found" >&2; exit 1; }
fi
[ -r "$LOG" ] || { echo "cannot read log: $LOG" >&2; exit 1; }
echo "# parsing: $LOG"
echo "# times SL_sing..setup_tot in seconds (t_avg); *_f columns in GFLOP/s (f/s_avg); NA = column absent in log"

# Strip ANSI color codes (SCTL_VERBOSE/profiler emit them) before parsing, then drive an awk state machine.
sed -e 's/\x1b\[[0-9;]*m//g' "$LOG" | awk '
  function twlabel(t,   x) {                       # friendly twist label
    x = t + 0;
    if (x < 1e-9)                return "0";
    if (x > 1.50 && x < 1.60)    return "pi/2";
    if (x > 3.10 && x < 3.20)    return "pi";
    return sprintf("%.4g", x);
  }
  # Scan a profiler-tree row for the two numeric columns after the scope name `pat`:
  #   +-SetupSingular  <t_avg>  <f/s_avg>   -> sets globals _t (seconds) and _fs (GFLOP/s).
  # Backward compatible with t_avg-only logs: if only one number follows, _fs stays "".
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
  function flush(   tot, thru) {
    if (!have) return;
    tot = ssing[0] + snear[0] + ssing[1] + snear[1];
    thru = (tot > 0 && nnodes != "") ? nnodes / tot : 0;
    printf "%-6s %5s %3s %-5s  %8.3f %8.3f %8.3f %8.3f %9.3f  %7s %7s %7s %7s  %8s %11.3e %10.3e\n",
      tol, nbeta, md, twlabel(twist),
      ssing[0]+0, snear[0]+0, ssing[1]+0, snear[1]+0, tot,
      fsfmt(ssing_fs[0]), fsfmt(snear_fs[0]), fsfmt(ssing_fs[1]), fsfmt(snear_fs[1]),
      (nnodes==""?"NA":nnodes), thru, (gerr==""?0:gerr) + 0;
  }
  BEGIN {
    have = 0; nsing = 0; nnear = 0;
    # t_avg columns (SL_sing..DL_near, setup_tot) are seconds; the *_fs columns are f/s_avg in GFLOP/s.
    printf "%-6s %5s %3s %-5s  %8s %8s %8s %8s %9s  %7s %7s %7s %7s  %8s %11s %10s\n",
      "tol","Nbeta","md","twist","SL_sing","SL_near","DL_sing","DL_near","setup_tot",
      "SLsng_f","SLnr_f","DLsng_f","DLnr_f","Nnodes","nodes/s","greens_err";
    printf "%-6s %5s %3s %-5s  %8s %8s %8s %8s %9s  %7s %7s %7s %7s  %8s %11s %10s\n",
      "------","-----","---","-----","--------","--------","--------","--------","---------",
      "-------","-------","-------","-------","--------","-----------","----------";
  }
  # --- block header: flush the previous point, reset, capture the swept parameters ---
  /^# core=/ {
    flush();
    have = 1; nsing = 0; nnear = 0;
    tol = nbeta = md = twist = nnodes = gerr = "";
    ssing[0]=ssing[1]=snear[0]=snear[1]="";
    ssing_fs[0]=ssing_fs[1]=snear_fs[0]=snear_fs[1]="";
    line = $0; gsub(/[()]/, " ", line);            # drop parens around (Nbeta=.. max_depth=..)
    ntok = split(line, tok, /[ \t]+/);
    for (i = 1; i <= ntok; i++) {
      if (tok[i] ~ /^tol=/)       { tol   = substr(tok[i], 5); }
      else if (tok[i] ~ /^Nbeta=/){ nbeta = substr(tok[i], 7); }
      else if (tok[i] ~ /^max_depth=/){ md = substr(tok[i], 11); }
      else if (tok[i] ~ /^twist=/){ twist = substr(tok[i], 7); }
    }
    next;
  }
  # --- profiler tree rows (1st Setup = SL, 2nd = DL); the +- marker excludes the VERBOSE {}/} traces.
  #     Each row carries t_avg (s) then f/s_avg (GFLOP/s); grabpair() pulls both. ---
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
