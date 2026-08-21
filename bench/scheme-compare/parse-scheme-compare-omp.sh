#!/usr/bin/env bash
#
# Parse the FULL-NODE OMP thread-scaling log from bench-scheme-compare-omp.sh into a per-scheme
# strong-scaling table (text + compilable LaTeX). Pure bash + gawk.
#
# This is the scheme-comparison analogue of ~/quad-junctions/scripts/parse_twisted_sphere_omp.sh:
# that parser keys one setup table on the OpenMP thread count for a single run; here we emit ONE
# SECTION PER (kernel, scheme), each a full thread-scaling sweep, so the four schemes can be read
# side by side. It reads the `@@ROW ... thr=<n> ...` lines emitted by bench-scheme-compare's omp mode
# (order 12, ppf 8, twist pi/6, tol 1e-9) -- NOT a profiler tree -- so per width it has:
#
#     thr        OpenMP thread count
#     setup      single-layer BIOp Setup() wall time (s)
#     pts/s/core N_pts / setup / thr        (the binary's reported per-core throughput)
#     pts/s tot  pts/s/core * thr           (aggregate node throughput)
#     speedup    setup(1) / setup(thr)      (strong-scaling speedup vs the 1-thread baseline)
#     eff        speedup / thr              (parallel efficiency)
#     error      Green's-identity max rel err (should be ~constant across widths -- a sanity check)
#
# Usage:
#   parse-scheme-compare-omp.sh <omp.txt> [out-prefix]
# Prints the per-scheme sections to stdout and, if out-prefix is given, writes <out-prefix>.tex
# (standalone, pdflatex-ready) with one table block per scheme.
set -euo pipefail

OMP="${1:-}"
PREFIX="${2:-}"
SCHEMES="RP Adaptive Hybrid Duffy"

[ -n "$OMP" ] && [ -f "$OMP" ] || { echo "usage: $0 <omp.txt> [out-prefix]   (omp.txt not found)" >&2; exit 1; }

# ---------------------------------------------------------------- shared gawk library
read -r -d '' AWKLIB <<'AWK' || true
function sci_parts(x,   e,m) {
  if (x <= 0 || x != x) return ""
  e = int(log(x)/log(10)); m = x/(10^e)
  while (m >= 10) { m /= 10; e++ }
  while (m <  1) { m *= 10; e-- }
  return sprintf("%.2f|%d", m, e)
}
function sci_txt(x,   p,a) { p=sci_parts(x); if(p=="")return "--"; split(p,a,"|"); return sprintf("%.2fe%+03d",a[1],a[2]) }
function sci_tex(x,   p,a) { p=sci_parts(x); if(p=="")return "--"; split(p,a,"|"); return sprintf("$%.2f\\times10^{%d}$",a[1],a[2]) }
function parse_row(   i,kv) {
  delete d
  for (i=1;i<=NF;i++) if (split($i,kv,"=")==2) d[kv[1]]=kv[2]
}
# collect @@ROW lines into E/S/P/ERR keyed by kernel,scheme,thr; track kernel & thr order.
function ingest(   k,s,thr) {
  parse_row()
  k=d["kernel"]; s=d["scheme"]; thr=d["thr"]+0
  if (!(k in kseen))            { kseen[k]=++nk;  korder[nk]=k }
  ks=k SUBSEP s
  if (!(ks in ksseen))          { ksseen[ks]=1 }
  key=ks SUBSEP thr
  SET[key]=d["setup"]+0; PPS[key]=d["pps"]+0; ERR[key]=d["error"]+0; HAVE[key]=1
  if (!(thr in thrseen))        { thrseen[thr]=1; THR[++nthr]=thr }
}
# sort the collected thread widths ascending into TW[1..nw]
function sort_thr(   i,n) {
  n = asort(THR, TW); nw = n
}
BEGIN { ns = split(SCHEMELIST, S, " "); nk=0; nthr=0 }
AWK

# ---------------------------------------------------------------- text renderer
read -r -d '' AWK_TXT <<'AWK' || true
/^@@ROW/ { ingest() }
END {
  sort_thr()
  for (ki=1;ki<=nk;ki++) { k=korder[ki]
    for (si=1;si<=ns;si++) { s=S[si]; ks=k SUBSEP s
      if (!(ks in ksseen)) continue
      printf "=== %s / %s ===\n", k, s
      printf "%4s  %10s  %11s  %12s  %9s  %7s  %11s\n",
        "thr","setup(s)","pts/s/core","pts/s_tot","speedup","eff","error"
      printf "%4s  %10s  %11s  %12s  %9s  %7s  %11s\n",
        "----","----------","-----------","------------","---------","-------","-----------"
      base=""
      for (wi=1;wi<=nw;wi++) { thr=TW[wi]; key=ks SUBSEP thr
        if (!(key in HAVE)) continue
        st=SET[key]; pc=PPS[key]; tot=pc*thr; er=ERR[key]
        if (base=="" && st>0) base=st
        spd=(base!="" && st>0)?base/st:0
        eff=(thr>0)?spd/thr:0
        printf "%4d  %10.4f  %11.1f  %12.1f  %8.2fx  %6.2f%%  %11s\n",
          thr, st, pc, tot, spd, eff*100, sci_txt(er)
      }
      printf "\n"
    }
  }
}
AWK

# ---------------------------------------------------------------- latex renderer (one block/scheme)
read -r -d '' AWK_TEX <<'AWK' || true
/^@@ROW/ { ingest() }
END {
  sort_thr()
  print "\\documentclass[11pt]{article}"
  print "\\usepackage{booktabs}"
  print "\\usepackage[margin=1in]{geometry}"
  print "\\begin{document}"
  print "\\begin{table}[t]\\centering\\small"
  print "\\setlength{\\tabcolsep}{6pt}"
  print "\\begin{tabular}{l l r r r r r}"
  print "\\toprule"
  print "kernel, scheme & thr & setup (s) & pts/s/core & pts/s tot & speedup & eff \\\\"
  print "\\midrule"
  fb=1
  for (ki=1;ki<=nk;ki++) { k=korder[ki]
    for (si=1;si<=ns;si++) { s=S[si]; ks=k SUBSEP s
      if (!(ks in ksseen)) continue
      if (!fb) print "\\addlinespace"; fb=0
      base=""; first=1
      for (wi=1;wi<=nw;wi++) { thr=TW[wi]; key=ks SUBSEP thr
        if (!(key in HAVE)) continue
        st=SET[key]; pc=PPS[key]; tot=pc*thr
        if (base=="" && st>0) base=st
        spd=(base!="" && st>0)?base/st:0
        eff=(thr>0)?spd/thr:0
        lbl = first ? (k ", " s) : ""
        first=0
        printf "%s & %d & %.4f & %.0f & %.0f & %.2f$\\times$ & %.1f\\%% \\\\\n",
          lbl, thr, st, pc, tot, spd, eff*100
      }
    }
  }
  print "\\bottomrule"
  print "\\end{tabular}"
  print "\\caption{OpenMP strong scaling of the single-layer setup on a twisted cubed sphere (order 12, 8 patches/face, twist $\\pi/6$, tol $10^{-9}$), one block per quadrature scheme, on a full Icelake node. speedup $=$ setup(1 thread)$/$setup($n$), eff $=$ speedup$/n$.}"
  print "\\end{table}"
  print "\\end{document}"
}
AWK

echo "========================================================================"
echo "OPENMP STRONG SCALING -- full node, one section per (kernel, scheme)"
echo "========================================================================"
gawk -v SCHEMELIST="$SCHEMES" "$AWKLIB$AWK_TXT" "$OMP"

if [ -n "$PREFIX" ]; then
  gawk -v SCHEMELIST="$SCHEMES" "$AWKLIB$AWK_TEX" "$OMP" > "${PREFIX}.tex"
  echo "# wrote ${PREFIX}.tex"
fi
