#!/bin/bash
# Parse bench-scheme-compare output into text tables + compilable LaTeX (pure bash + gawk).
#
# Reads the @@ROW tagged lines emitted by bin/bench-scheme-compare:
#   @@ROW kernel=<k> scheme=<s> thr=<n> twist=<t> tol=<tol> error=<e> pps=<p> setup=<s>
#
# CONVERGENCE run -> a Table-2-style table where the {Rome, Genoa, Icelake} machine subcolumns are
# replaced by the four schemes {RP, Adaptive, Hybrid, Duffy} under BOTH an `error` group and a
# `pts/s/core` group. OMP run -> a per-scheme strong-scaling table (1-thread pts/s, full-node
# pts/s, speedup).
#
# Usage:
#   parse-scheme-compare.sh <conv.txt> <omp.txt> <out-prefix>
# Writes <out-prefix>-conv.tex and <out-prefix>-omp.tex (standalone, pdflatex-ready) and prints
# both tables to stdout. A missing / '-' input skips that table.
set -euo pipefail

CONV="${1:-}"
OMP="${2:-}"
PREFIX="${3:-scheme-compare}"

SCHEMES="RP Adaptive Hybrid Duffy"

# ---------------------------------------------------------------- shared gawk library (functions)
read -r -d '' AWKLIB <<'AWK' || true
function sci_parts(x,   e,m) {           # -> "mant|exp" for x>0, else ""
  if (x <= 0 || x != x) return ""
  e = int(log(x)/log(10)); m = x/(10^e)
  while (m >= 10) { m /= 10; e++ }
  while (m <  1) { m *= 10; e-- }
  return sprintf("%.2f|%d", m, e)
}
function sci_txt(x,   p,a) { p=sci_parts(x); if(p=="")return "--"; split(p,a,"|"); return sprintf("%.2fe%+03d",a[1],a[2]) }
function sci_tex(x,   p,a) { p=sci_parts(x); if(p=="")return "--"; split(p,a,"|"); return sprintf("$%.2f\\times10^{%d}$",a[1],a[2]) }
function twlabel(tw, latex,   j) {
  for (j=1; j<=ntw; j++) if (tw-tv[j] < 1e-3 && tv[j]-tw < 1e-3) return latex ? tx[j] : tl[j]
  return sprintf("%.3f", tw)
}
function twidx(tw,   j) { for (j=1;j<=ntw;j++) if (tw-tv[j] < 1e-3 && tv[j]-tw < 1e-3) return j; return 0 }
function parse_row(   i,kv) {            # fills global d[] from an @@ROW line ($0)
  delete d
  for (i=1;i<=NF;i++) if (split($i,kv,"=")==2) d[kv[1]]=kv[2]
}
BEGIN {
  ns = split(SCHEMELIST, S, " ")
  PI = atan2(0,-1)
  tv[1]=PI/6; tl[1]="pi/6"; tx[1]="\\pi/6"
  tv[2]=PI/2; tl[2]="pi/2"; tx[2]="\\pi/2"
  tv[3]=PI;   tl[3]="pi";   tx[3]="\\pi"
  ntw = 3
}
AWK

# ---------------------------------------------------------------- convergence gawk program
read -r -d '' AWK_CONV <<'AWK' || true
/^@@ROW/ {
  parse_row()
  k=d["kernel"]; s=d["scheme"]; tw=d["twist"]+0; tol=d["tol"]+0
  ti=twidx(tw); if (ti==0) next
  if (!(k in kseen)) { kseen[k]=++nk; korder[nk]=k }
  tolkey=sprintf("%.3e",tol)
  if (!(tolkey in tolseen)) { tolseen[tolkey]=tol; TVraw[++nt]=tol }
  twpresent[k SUBSEP ti]=1
  key=k SUBSEP ti SUBSEP tolkey SUBSEP s
  E[key]=d["error"]+0; P[key]=d["pps"]+0; H[key]=1
}
END {
  # tols descending
  n = asort(TVraw, TVs); for (i=1;i<=n;i++) TOL[i]=TVs[n-i+1]

  if (MODE=="text") {
    printf "%-14s %7s | %-5s", "kernel,twist", "tol", "error"
    for (j=1;j<=ns;j++) printf " %9s", S[j]
    printf " | %-10s", "pts/s/core"
    for (j=1;j<=ns;j++) printf " %8s", S[j]
    printf "\n"
    for (c=0;c<118;c++) printf "-"; printf "\n"
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (ti=1;ti<=ntw;ti++) { if (!((k SUBSEP ti) in twpresent)) continue
        first=1
        for (i=1;i<=n;i++) { tol=TOL[i]; tk=sprintf("%.3e",tol)
          lbl = first ? (k ", " twlabel(tv[ti],0)) : ""
          first=0
          printf "%-14s %7.0e | %5s", lbl, tol, ""
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " %9s", (key in H) ? sci_txt(E[key]) : "--" }
          printf " | %10s", ""
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " %8s", (key in H) ? sprintf("%.0f",P[key]) : "--" }
          printf "\n"
        }
        printf "\n"
      }
    }
  } else {
    print "\\documentclass[11pt]{article}"
    print "\\usepackage{booktabs}"
    print "\\usepackage[margin=1in,landscape]{geometry}"
    print "\\begin{document}"
    print "\\begin{table}[t]\\centering\\small"
    print "\\setlength{\\tabcolsep}{4pt}"
    spec="l l"; for (j=1;j<=ns;j++) spec=spec " r"; for (j=1;j<=ns;j++) spec=spec " r"
    print "\\begin{tabular}{" spec "}"
    print "\\toprule"
    printf "& & \\multicolumn{%d}{c}{error} & \\multicolumn{%d}{c}{pts/s/core} \\\\\n", ns, ns
    printf "\\cmidrule(lr){3-%d} \\cmidrule(lr){%d-%d}\n", 2+ns, 3+ns, 2+2*ns
    printf "kernel, twist & tol"
    for (j=1;j<=ns;j++) printf " & %s", S[j]
    for (j=1;j<=ns;j++) printf " & %s", S[j]
    print " \\\\"
    print "\\midrule"
    fb=1
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (ti=1;ti<=ntw;ti++) { if (!((k SUBSEP ti) in twpresent)) continue
        if (!fb) print "\\addlinespace"; fb=0
        first=1
        for (i=1;i<=n;i++) { tol=TOL[i]; tk=sprintf("%.3e",tol)
          lbl = first ? (k ", $" twlabel(tv[ti],1) "$") : ""
          first=0
          printf "%s & %s", lbl, sci_tex(tol)
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " & %s", (key in H) ? sci_tex(E[key]) : "--" }
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " & %s", (key in H) ? sprintf("%.0f",P[key]) : "--" }
          print " \\\\"
        }
      }
    }
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{On-surface Green's identity error and single-layer setup throughput on a twisted cubed sphere (order 12, 12 patches/face), by quadrature scheme. The machine columns of the original Table~2 are replaced by the four schemes.}"
    print "\\end{table}"
    print "\\end{document}"
  }
}
AWK

# ---------------------------------------------------------------- omp gawk program
read -r -d '' AWK_OMP <<'AWK' || true
/^@@ROW/ {
  parse_row()
  k=d["kernel"]; s=d["scheme"]; thr=d["thr"]+0; pps=d["pps"]+0
  if (!(k in kseen)) { kseen[k]=++nk; korder[nk]=k }
  key=k SUBSEP s
  if (!(key in seenks)) { seenks[key]=1 }
  P[key SUBSEP thr]=pps
  if (!((key) in tmin) || thr < tmin[key]) tmin[key]=thr
  if (!((key) in tmax) || thr > tmax[key]) tmax[key]=thr
}
END {
  if (MODE=="text") {
    printf "%-8s %-9s %11s %11s %8s %9s\n", "kernel","scheme","1thr pts/s","full pts/s","threads","speedup"
    for (c=0;c<60;c++) printf "-"; printf "\n"
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (j=1;j<=ns;j++) { s=S[j]; key=k SUBSEP s
        if (!(key in seenks)) continue
        lo=tmin[key]; hi=tmax[key]
        p1=P[key SUBSEP lo]; pn=P[key SUBSEP hi]     # per-core throughput
        spd=(p1*lo>0)?(pn*hi)/(p1*lo):0              # total(hi)/total(lo)
        printf "%-8s %-9s %11.1f %11.1f %8d %8.1fx\n", k, s, p1*lo, pn*hi, hi, spd
      }
      printf "\n"
    }
  } else {
    print "\\documentclass[11pt]{article}"
    print "\\usepackage{booktabs}"
    print "\\begin{document}"
    print "\\begin{table}[t]\\centering\\small"
    print "\\begin{tabular}{l l r r r r}"
    print "\\toprule"
    print "kernel & scheme & 1 thread pts/s & full-node pts/s & threads & speedup \\\\"
    print "\\midrule"
    fb=1
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      if (!fb) print "\\addlinespace"; fb=0
      for (j=1;j<=ns;j++) { s=S[j]; key=k SUBSEP s
        if (!(key in seenks)) continue
        lo=tmin[key]; hi=tmax[key]
        p1=P[key SUBSEP lo]; pn=P[key SUBSEP hi]
        spd=(p1*lo>0)?(pn*hi)/(p1*lo):0
        printf "%s & %s & %.0f & %.0f & %d & %.1f$\\times$ \\\\\n", k, s, p1*lo, pn*hi, hi, spd
      }
    }
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{OpenMP strong scaling of the single-layer setup (order 12, 8 patches/face, twist $\\pi/6$, tol $10^{-9}$), by quadrature scheme.}"
    print "\\end{table}"
    print "\\end{document}"
  }
}
AWK

have() { [ -n "$1" ] && [ "$1" != "-" ] && [ -f "$1" ]; }

echo "========================================================================"
echo "CONVERGENCE (error + pts/s/core per scheme)"
echo "========================================================================"
if have "$CONV"; then
  gawk -v SCHEMELIST="$SCHEMES" -v MODE=text "$AWKLIB$AWK_CONV" "$CONV"
  gawk -v SCHEMELIST="$SCHEMES" -v MODE=tex  "$AWKLIB$AWK_CONV" "$CONV" > "${PREFIX}-conv.tex"
  echo "# wrote ${PREFIX}-conv.tex"
else
  echo "# (no convergence input)"
fi

echo
echo "========================================================================"
echo "OPENMP STRONG SCALING (per scheme)"
echo "========================================================================"
if have "$OMP"; then
  gawk -v SCHEMELIST="$SCHEMES" -v MODE=text "$AWKLIB$AWK_OMP" "$OMP"
  gawk -v SCHEMELIST="$SCHEMES" -v MODE=tex  "$AWKLIB$AWK_OMP" "$OMP" > "${PREFIX}-omp.tex"
  echo "# wrote ${PREFIX}-omp.tex"
else
  echo "# (no omp input)"
fi
