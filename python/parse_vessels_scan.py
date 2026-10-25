#!/usr/bin/env python3
"""Extract {parameters, MFS error, net-flux error} from the vessels-scan logs (stdlib only).

usage: python3 -I python/parse_vessels_scan.py out/vessels-scan-*.log  (or out/vscan-*.log)
Per log, reads the [SCAN] header and the tagged result lines printed by ybifurc-vessels-flow-bie:
  watertight COMBINED, Stokes DL const-density err, Stokes Green's (ext sources) err,
  [MFS] RESULT rel_L2/iters, [FLOW] RESULT iters,
  [NETFLUX] RESULT Q_inlet / Q_outlet / Q_conn_sum (connector flux conservation check).
Missing entries (job unfinished/failed) print as '-'.
"""
import re, sys, glob

NUM = r"([-+0-9.eE]+|nan|inf)"
PAT = {
    "tol":    r"\[SCAN\] tol=(\S+)", "order": r"\[SCAN\] .*?order=(\d+)",
    "fourier": r"\[SCAN\] .*?fourier=(\d+)", "nax": r"\[SCAN\] .*?nax=(\d+)",
    "wt":     r"COMBINED=" + NUM,
    "dl":     r"DL const-density identity: max rel err = " + NUM,
    "green":  r"Green's identity error = " + NUM,
    "mfs":    r"\[MFS\] RESULT iters=(\d+) rel_L2=" + NUM,
    "flow":   r"\[FLOW\] RESULT iters=(\d+)",
    # Net-flux conservation: inlet arm vs outlet arm vs sum over connector arms (all = p_in).
    "nf_in":    r"\[NETFLUX\] RESULT .*?Q_inlet=" + NUM,
    "nf_out":   r"\[NETFLUX\] RESULT .*?Q_outlet=" + NUM,
    "nf_conn":  r"\[NETFLUX\] RESULT .*?Q_conn_sum=" + NUM,
    "nf_ivc":   r"\[NETFLUX\] RESULT .*?inlet_vs_conn=" + NUM,
    "nf_ovc":   r"\[NETFLUX\] RESULT .*?outlet_vs_conn=" + NUM,
    "nf_ivo":   r"\[NETFLUX\] RESULT .*?inlet_vs_outlet=" + NUM,
    "nf_worst": r"\[NETFLUX\] RESULT .*?worst_conn_rel_err=" + NUM,
}

def parse(path):
    t = open(path, errors="replace").read()
    m = {k: re.search(v, t) for k, v in PAT.items()}
    f = lambda k, i=1: m[k].group(i) if m[k] else "-"
    return dict(tol=f("tol"), order=f("order"), fourier=f("fourier"), nax=f("nax"),
                wt=f("wt"), dl=f("dl"), green=f("green"), mfs_it=f("mfs", 1), mfs=f("mfs", 2),
                flow_it=f("flow"), nf_in=f("nf_in"), nf_out=f("nf_out"), nf_conn=f("nf_conn"),
                nf_ivc=f("nf_ivc"), nf_ovc=f("nf_ovc"), nf_ivo=f("nf_ivo"), nf_worst=f("nf_worst"))

def main():
    paths = sorted(p for a in (sys.argv[1:] or ["out/vessels-scan-*.log", "out/vscan-*.log"]) for p in glob.glob(a))
    rows = [parse(p) for p in paths]
    rows = [r for r in rows if r["tol"] != "-"]
    key = lambda r: (float(r["tol"]), int(r["order"]), int(r["nax"]))
    rows.sort(key=key, reverse=False)
    cols = [("tol", "tol"), ("order", "ord"), ("fourier", "fou"), ("nax", "nax"), ("wt", "watertight"),
            ("dl", "DL-stk"), ("green", "Green-stk-ext"), ("mfs", "MFS relL2"), ("mfs_it", "MFS it"),
            ("flow_it", "flow it"), ("nf_in", "Q_inlet"), ("nf_out", "Q_outlet"), ("nf_conn", "Q_conn_sum"),
            ("nf_ivc", "inlet-vs-conn"), ("nf_ovc", "outlet-vs-conn"), ("nf_ivo", "inlet-vs-outlet"),
            ("nf_worst", "worst-conn")]
    print("  ".join("%-*s" % (max(len(h), 10), h) for _, h in cols))
    for r in rows:
        print("  ".join("%-*s" % (max(len(h), 10), r[k]) for k, h in cols))

if __name__ == "__main__":
    main()
