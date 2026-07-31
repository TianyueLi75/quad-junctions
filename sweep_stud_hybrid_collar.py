#!/usr/bin/env python3
"""
Collar mesh sweep for the PATCH-RELATIVE cilia base (R_shaft = 0.25*S, the thin cilium).

Geometry: bin/stud_sphere-hybrid-bie mode `centerfinger` at PatchPerFace=3, with the default shaft
sizing R_shaft = frac*S (frac=0.25 via QJ_RSHAFT_FRAC), r_fil = 0.1*R_shaft, H_shaft = 3*R_shaft -- so the
finger is self-similar to its patch and the collar is thin and scale-invariant.

Question: with the shaft now a fixed fraction of the patch, what collar (Nc rings, Naz azimuthal panels)
holds each accuracy level? DL-LAPLACE identity only (STUD_LAPLACE_ONLY + STUD_DL_ONLY). The cap is kept
DECOUPLED and FIXED at cap_Naz=8 (it just scales with R_shaft), so only the collar/foot Naz varies.

For each order in {12,16} and target T in {1e-5,1e-7,1e-9,1e-11} (tol=T, the near-tol expected accuracy),
walk a coarse->fine (Nc,Naz) ladder and report the COARSEST collar whose DL error is within 1-2 magnitudes
of T (PASS <=10*T, MARGINAL <=100*T). Early-stop at the first pass, or on a failing plateau.
"""
import os, re, math, csv, subprocess, sys, time

BIN   = os.environ.get("QJ_HYBRID_BIN", "./bin/stud_sphere-hybrid-bie")
MODE  = "centerfinger"
PPF   = 3
FRAC  = 0.25                # R_shaft = FRAC*S  (S = 1/PPF) -- the thin cilium
S     = 1.0 / PPF
CAP_NAZ = 8                 # cap azimuthal panels, FIXED (decoupled) while collar Naz varies
FOURIER, CHEB, INVERT, GRADE, CORE = 16, 10, 1, 1.0, 0.40

ORDERS  = [12, 16]
TARGETS = [1e-5, 1e-7, 1e-9, 1e-11]

# coarse -> fine collar ladder, ordered by collar panel count (Nc*Naz), tiebreak on Nc
LADDER = [(1, 8), (1, 12), (2, 8), (1, 16), (2, 12), (3, 8), (1, 24), (2, 16), (3, 12), (2, 24), (3, 16), (3, 24)]

# quadrature (near-eval) scaled to the target. HARD sets: Nbeta{48,100,200,300,400,512}, max_depth{4,8,12,30}.
QUAD = {1e-5: (100, 12), 1e-7: (200, 30), 1e-9: (400, 30), 1e-11: (512, 30)}

RUN_TIMEOUT = 5400
OMP = os.environ.get("OMP_NUM_THREADS", "8")
DL_RE = re.compile(r"DL const-density identity.*max rel err = ([0-9.eE+\-]+)")
RS_RE = re.compile(r"R_shaft=([0-9.eE+\-]+)")
WT_RE = re.compile(r"\|int n dA\|:.*COMBINED=([0-9.eE+\-]+)")


def collar_nc(naz):   # analytic auto ring count at R_foot=0.55*S
    r_foot = (FRAC + 0.1 * FRAC) * S
    return max(1, math.ceil(math.log(math.sqrt(2) * S / r_foot) / math.log(1 + 2 * math.pi / naz)))


def run(order, tol, nbeta, maxd, nc, naz):
    # centerfinger CLI: tol Nbeta max_depth R_shaft Nc Naz order r_fil n_axial fourier cheb PPF invert
    #                   H_shaft grade core cap_Naz  (R_shaft/r_fil/H_shaft = -1 => patch-relative default)
    args = [BIN, MODE, f"{tol:g}", str(nbeta), str(maxd), "-1", str(nc), str(naz), str(order),
            "-1", "-1", str(FOURIER), str(CHEB), str(PPF), str(INVERT), "-1",
            f"{GRADE:g}", f"{CORE:g}", str(CAP_NAZ)]
    env = dict(os.environ, STUD_LAPLACE_ONLY="1", STUD_DL_ONLY="1", OMP_NUM_THREADS=OMP,
               QJ_RSHAFT_FRAC=f"{FRAC:g}")
    t0 = time.time()
    try:
        p = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=RUN_TIMEOUT)
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or ""); rc = -9
    dlm, rsm, wt = DL_RE.search(out), RS_RE.search(out), WT_RE.search(out)
    return dict(order=order, tol=tol, nbeta=nbeta, maxd=maxd, nc=nc, naz=naz,
                dl=float(dlm.group(1)) if dlm else float("nan"),
                R_shaft=float(rsm.group(1)) if rsm else float("nan"),
                watertight=float(wt.group(1)) if wt else float("nan"),
                secs=round(time.time() - t0, 1), rc=rc)


def band(dl, T):
    if math.isnan(dl):
        return "ERR"
    if dl <= 10 * T:
        return "PASS"
    if dl <= 100 * T:
        return "MARGINAL"
    return "FAIL"


def main():
    rows, summary = [], {}
    for order in ORDERS:
        for T in TARGETS:
            nbeta, maxd = QUAD[T]
            print(f"\n===== order={order} target={T:g} [tol={T:g} Nbeta={nbeta} md={maxd}] "
                  f"(R_shaft={FRAC:g}*S={FRAC*S:.4f}) =====")
            reached, best = None, None
            for nc, naz in LADDER:   # full coarse->fine ladder; stop at first pass (no plateau break --
                r = run(order, T, nbeta, maxd, nc, naz)   # Nc is the real lever and must be explored)
                r.update(target=T, band=band(r["dl"], T), nc_auto=collar_nc(naz))
                rows.append(r)
                print(f"  Nc={nc} Naz={naz:2d} (auto {r['nc_auto']})  DL={r['dl']:.3e}  "
                      f"{r['band']:8s} R_shaft={r['R_shaft']:.4f} wt={r['watertight']:.1e} ({r['secs']}s)")
                if best is None or (not math.isnan(r["dl"]) and r["dl"] < best["dl"]):
                    best = r
                if r["band"] in ("PASS", "MARGINAL"):
                    reached = r
                    break
            if reached:
                summary[(order, T)] = dict(reached=True, row=reached)
                print(f"  -> coarsest ({reached['band']}) at Nc={reached['nc']} Naz={reached['naz']}: "
                      f"DL={reached['dl']:.3e}")
            else:
                summary[(order, T)] = dict(reached=False, floor=best)
                print(f"  -> UNREACHED. floor DL={best['dl']:.3e} at Nc={best['nc']} Naz={best['naz']}")

    csv_path = "stud-hybrid-collar-sweep-results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "target", "tol", "Nbeta", "max_depth", "Nc", "Naz", "Nc_auto",
                    "R_shaft", "watertight", "DL", "band", "secs", "rc"])
        for r in rows:
            w.writerow([r["order"], f'{r["target"]:g}', f'{r["tol"]:g}', r["nbeta"], r["maxd"],
                        r["nc"], r["naz"], r["nc_auto"], f'{r["R_shaft"]:.5f}',
                        f'{r["watertight"]:.2e}', f'{r["dl"]:.6e}', r["band"], r["secs"], r["rc"]])
    print(f"\nwrote {csv_path} ({len(rows)} runs)")

    print("\n" + "=" * 74)
    print(f"SUMMARY: coarsest collar (Nc,Naz) reaching each target (centerfinger PPF=3, R_shaft={FRAC:g}*S, DL-Laplace)")
    print("=" * 74)
    hdr = f'{"order":>5} {"target":>8} | {"Nc":>3} {"Naz":>4} | {"DL":>10} | {"band":>8}'
    print(hdr); print("-" * len(hdr))
    for order in ORDERS:
        for T in TARGETS:
            s = summary[(order, T)]
            if s["reached"]:
                r = s["row"]
                print(f'{order:>5} {T:>8g} | {r["nc"]:>3} {r["naz"]:>4} | {r["dl"]:>10.2e} | {s["row"]["band"]:>8}')
            else:
                fl = s["floor"]
                print(f'{order:>5} {T:>8g} | {"":>3} {"":>4} | {fl["dl"]:>10.2e} | {"floor":>8}')
        print("-" * len(hdr))


if __name__ == "__main__":
    if not os.path.exists(BIN):
        sys.exit(f"missing {BIN} -- build: . ./sctl_source && make bin/stud_sphere-hybrid-bie")
    main()
