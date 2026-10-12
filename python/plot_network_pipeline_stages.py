#!/usr/bin/env python3
"""Truthful per-stage visuals of the arm-centerline construction for a node chain.
All plotted curves/points are read from the C++ per-stage dump (STAGES) and the .graph;
nothing is re-implemented (no Taubin/spline/Hermite math here).  Only finite-difference
diagnostics of dumped points (labelled) are computed.  Theory: obj_to_quadrature_methods.tex
sec:smooth (curv-limit, Taubin eq:taubin1/2), sec:centerline (Hermite eq:hermite, tridiag
eq:tridiag, spline eq:spline), sec:taper.

usage: plot_network_pipeline_stages.py STAGES GRAPH OUTDIR [--chains "a,b,c,d;e,f,g,h"] [--seed N]
  chain = node ids junction,junction,...,cap  (consecutive nodes joined by a graph edge)
  without --chains, two chains (junc-junc-junc-cap) are drawn at random from --seed.
"""
import sys, os, random, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

RED, BLUE, GREY = "#C0392B", "#3E7CAE", "#8A96A3"
C_RAW, C_CURV, C_LOW, C_FIN = "#8A96A3", "#E08E0B", "#2E8B57", "#1B1B1B"


def parse_graph(path):
    toks = [l.split() for l in open(path) if l.strip() and not l.lstrip().startswith("#")]
    i, nodes, edges = 0, {}, []
    while i < len(toks):
        h = toks[i][0]
        if h == "NCLUSTER": i += 1 + int(toks[i][1])
        elif h == "NNODE":
            n = int(toks[i][1]); i += 1
            for _ in range(n):
                t = toks[i]; i += 1
                nodes[int(t[0])] = dict(pos=np.array(t[1:4], float), typ=int(t[5]))
        elif h == "NEDGE":
            n = int(toks[i][1]); i += 1
            for _ in range(n):
                t = toks[i]; i += 1
                n0, n1, npts = int(t[1]), int(t[2]), int(t[5])
                p = np.array([[float(x) for x in toks[i + k][:4]] for k in range(npts)]); i += npts
                edges.append(dict(n0=n0, n1=n1, xyz=p[:, :3], r=p[:, 3]))
        else: i += 1
    return nodes, edges


def arm_tag(a, b, nodes):
    """(tag, owner) per the dump's rule."""
    ta, tb = nodes[a]["typ"], nodes[b]["typ"]
    if ta == 0 and tb == 0: return "j%d-%d-0" % (min(a, b), max(a, b)), min(a, b)
    j, c = (a, b) if ta == 0 else (b, a)
    return "j%d-%d-1" % (j, c), j


def parse_stages(path, tags):
    want = set(tags); out = {}; cur = None; blk = None
    for ln in open(path):
        if ln.startswith("# ARM "):
            t = ln.split()[2]; cur = t if t in want else None
            if cur: out[cur] = dict(ctrl={}); blk = None
            continue
        if cur is None: continue
        if ln.startswith("# END ARM"): cur = None; continue
        s = ln.split()
        if s[0] == "STAGE":
            blk = s[1]; out[cur][blk] = []; continue
        if s[0] == "CONTROL": blk = None; continue
        if blk is None:
            if s[0] in ("A0", "uA", "endC", "uB"): out[cur]["ctrl"][s[0]] = np.array(s[1:4], float)
            elif s[0] in ("QA", "QB"):
                out[cur]["ctrl"][s[0]] = np.array(s[1:4], float)
                out[cur]["ctrl"][s[-1].split("=")[0]] = float(s[-1].split("=")[1])
            elif s[0] == "params":
                for kv in s[1:]: k, v = kv.split("="); out[cur]["ctrl"][k] = float(v)
            continue
        if blk == "spline_knots":
            out[cur][blk].append([float(x) for x in s[:3]] + [float(s[3].split("=")[1])] + [float(v) for v in s[4].split("=")[1].split(",")])
        elif blk == "final":
            out[cur][blk].append([float(x) for x in s[:3]] + [float(s[3].split("=")[1])])
        else: out[cur][blk].append([float(x) for x in s[:3]])
    for a in out.values():
        for k in list(a):
            if k != "ctrl": a[k] = np.array(a[k])
    return out


def rho_fd(P):
    """diagnostic (finite-diff of dumped points): circumradius of consecutive triples; arclength s."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.r_[0, np.cumsum(d)]
    a, b, c = d[:-1], d[1:], np.linalg.norm(P[2:] - P[:-2], axis=1)
    cr = np.linalg.norm(np.cross(P[1:-1] - P[:-2], P[2:] - P[1:-1]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"): rho = a * b * c / (2 * cr)
    return s[1:-1], rho


def pick_chains(nodes, edges, seed, avoid=(6,)):
    rng = random.Random(seed); adj = {}
    for e in edges:
        adj.setdefault(e["n0"], set()).add(e["n1"]); adj.setdefault(e["n1"], set()).add(e["n0"])
    J = [n for n in nodes if nodes[n]["typ"] == 0 and n not in avoid]
    chains, used = [], set()
    for _ in range(100000):
        a = rng.choice(J); b_ = [x for x in adj[a] if nodes[x]["typ"] == 0 and x not in avoid]
        if not b_: continue
        b = rng.choice(b_); c_ = [x for x in adj[b] if nodes[x]["typ"] == 0 and x not in avoid and x != a]
        if not c_: continue
        c = rng.choice(c_); d_ = [x for x in adj[c] if nodes[x]["typ"] == 1]
        if not d_: continue
        ch = (a, b, c, rng.choice(d_))
        nb = set(ch) | {y for x in ch[:3] for y in adj[x]}
        if used & nb: continue
        chains.append(ch); used |= nb
        if len(chains) == 2: return chains
    raise SystemExit("no chains")


def plot_chain(chain, nodes, edges, st, path, title):
    arms = []
    for a, b in zip(chain[:-1], chain[1:]):
        tag, owner = arm_tag(a, b, nodes); d = st[tag]
        e = next(e for e in edges if {e["n0"], e["n1"]} == {a, b})
        rev = e["n0"] != owner     # stage raw is oriented from the owner
        gp = e["xyz"][::-1] if rev else e["xyz"]; gr = e["r"][::-1] if rev else e["r"]
        arms.append(dict(tag=tag, d=d, gp=gp, gr=gr, cap=tag.endswith("-1")))
    fig = plt.figure(figsize=(21, 20))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.5, 1, 1, 1, 1], hspace=0.38, wspace=0.28)
    fig.suptitle(title, fontsize=13, y=0.995)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    axs = {"xy": fig.add_subplot(gs[0, 1]), "xz": fig.add_subplot(gs[0, 2])}
    axz = fig.add_subplot(gs[0, 3])
    for A in arms:
        d = A["d"]; col = BLUE if A["cap"] else RED; c = d["ctrl"]
        for ax, (i, j) in ((axs["xy"], (0, 1)), (axs["xz"], (0, 2))):
            ax.plot(d["raw"][:, i], d["raw"][:, j], "-", c=C_RAW, lw=0.8, marker="o", ms=2.5, mfc="none")
            ax.plot(d["lowpass"][:, i], d["lowpass"][:, j], ".", c=C_LOW, ms=3)
            ax.plot(d["final"][:, i], d["final"][:, j], "-", c=col, lw=1.4)
            ax.plot(*zip(c["A0"][[i, j]], c["QA"][[i, j]]), "-", c="k", lw=2.2)
            ax.plot(*zip(c["endC"][[i, j]], c["QB"][[i, j]]), "-", c="k", lw=2.2)
            ax.plot(c["QA"][i], c["QA"][j], "s", c="k", ms=4); ax.plot(c["QB"][i], c["QB"][j], "s", c="k", ms=4)
        ax3.plot(*d["raw"].T, "-", c=C_RAW, lw=0.7, marker="o", ms=2, mfc="none")
        ax3.plot(*d["final"][:, :3].T, "-", c=col, lw=1.5)
        ax3.plot(*np.c_[c["A0"], c["QA"]], "-", c="k", lw=2.2); ax3.plot(*np.c_[c["endC"], c["QB"]], "-", c="k", lw=2.2)
    for n in chain:
        p = nodes[n]["pos"]; mk = "^" if nodes[n]["typ"] else "o"
        ax3.scatter(*p, c=BLUE if nodes[n]["typ"] else RED, marker=mk, s=25)
        for ax, (i, j) in ((axs["xy"], (0, 1)), (axs["xz"], (0, 2))):
            ax.plot(p[i], p[j], mk, c=BLUE if nodes[n]["typ"] else RED, ms=7); ax.annotate(str(n), (p[i], p[j]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax3.set_title("3D: raw vmtk knots (grey o), final meshed centerline (red=interior/blue=leaf),\nblack = coaxial leads A0->QA, endC->QB   [STAGE raw / final / CONTROL]", fontsize=8)
    axs["xy"].set_title("x-y projection\nsame arrays as 3D; green dots = lowpass (C2) knots [STAGE lowpass]", fontsize=8)
    axs["xz"].set_title("x-z projection\nsame arrays; black segments = coaxial lead (uA/uB); squares = QA/QB", fontsize=8)
    for ax, l in ((axs["xy"], ("x", "y")), (axs["xz"], ("x", "z"))): ax.set_xlabel(l[0]); ax.set_ylabel(l[1]); ax.set_aspect("equal", "datalim")
    # zoom on corner of the first arm, end A, in its PCA plane
    A = arms[0]; d = A["d"]; c = d["ctrl"]; L = 6 * c["lead_len"] if "lead_len" in c else 15
    ctr = c["A0"]; mask = lambda P: np.linalg.norm(P[:, :3] - ctr, axis=1) < max(3 * c["bse"], 10)
    loc = np.vstack([d["raw"][mask(d["raw"])], d["lowpass"][mask(d["lowpass"])], d["final"][mask(d["final"])][:, :3]])
    far = d["lowpass"][mask(d["lowpass"])][-1] - ctr; e1 = c["uA"]; e2 = far - (far @ e1) * e1; e2 /= np.linalg.norm(e2)
    B = np.c_[e1, e2]; pr = lambda P: (P[:, :3] - ctr) @ B
    for key, st_, kw in (("raw", C_RAW, dict(marker="o", mfc="none", ms=4, lw=0.8)), ("curv_limited", C_CURV, dict(marker="x", ms=4, lw=0.6)),
                         ("lowpass", C_LOW, dict(marker=".", ms=6, lw=0.6))):
        P = d[key]; P = P[mask(P)]; q = pr(P); axz.plot(q[:, 0], q[:, 1], "-", c=st_, label=key, **kw)
    q = pr(d["final"][mask(d["final"])]); axz.plot(q[:, 0], q[:, 1], "-", c=RED if not A["cap"] else BLUE, lw=1.8, label="final (meshed)")
    for nm, mk in (("A0", "*"), ("QA", "s")):
        q = pr(c[nm][None]); axz.plot(q[0, 0], q[0, 1], mk, c="k", ms=9, label=nm)
    kn = d["spline_knots"]; k0 = kn[np.argmin(abs(kn[:, 3] - c["bse"]))]
    q = pr(k0[None, :3]); axz.plot(q[0, 0], q[0, 1], "D", c="m", ms=6, label="knot nearest bse (run entry)")
    h = c["corner_kr"] * np.linalg.norm(k0[:3] - c["QA"]); q0 = pr(c["QA"][None])[0]; dq = (c["uA"] @ B) * h
    axz.annotate("", xy=q0 + dq, xytext=q0, arrowprops=dict(arrowstyle="->", color="m", lw=1.5))
    axz.set_title("ZOOM: arm %s end A (plane = uA and the lateral direction to the far window knot;\naxes NOT equal-scaled, lateral exaggerated)\nHermite corner QA -> run entry; magenta arrow = uA handle,\nlength = corner_kr*|QA-knot(bse)| [CONTROL; eq:hermite]" % A["tag"], fontsize=8)
    axz.legend(fontsize=6, loc="best"); axz.set_xlabel("along uA"); axz.set_ylabel("lateral")
    # rows per arm
    for k, A in enumerate(arms):
        d = A["d"]; col = BLUE if A["cap"] else RED; c = d["ctrl"]; sk = d["spline_knots"][:, 3]
        a1 = fig.add_subplot(gs[1, k]); a2 = fig.add_subplot(gs[2, k]); a3 = fig.add_subplot(gs[3, k]); a4 = fig.add_subplot(gs[4, k])
        dr = np.linalg.norm(d["curv_limited"] - d["raw"], axis=1); dl = np.linalg.norm(d["lowpass"] - d["curv_limited"], axis=1)
        a1.semilogy(sk, dr + 1e-18, "-o", c=C_CURV, ms=2, lw=0.8, label="|curv_limited - raw|")
        a1.semilogy(sk, dl + 1e-18, "-o", c=C_LOW, ms=2, lw=0.8, label="|lowpass - curv_limited| (Taubin)")
        a1.set_title("%s  knot displacement per step, vs knot arclength s\n[STAGE raw->curv_limited->lowpass, same index]" % A["tag"], fontsize=8)
        a1.set_ylabel("|dx|"); a1.legend(fontsize=6)
        rr = {}
        for nm, P in (("raw", d["raw"]), ("lowpass", d["lowpass"]), ("final", d["final"][:, :3])):
            s, rho = rho_fd(P); rr[nm] = (s, rho)
        a2.semilogy(*rr["raw"], "-", c=C_RAW, lw=0.8, label="raw knots"); a2.semilogy(*rr["lowpass"], "-", c=C_LOW, lw=0.9, label="lowpass knots")
        a2.semilogy(*rr["final"], "-", c=col, lw=1.2, label="final meshed")
        a2.axhline(c["rA"], c="k", ls=":", lw=0.8, label="rA (tube radius)")
        a2.set_title("diagnostic (finite-diff of dumped points): curvature radius rho(s)\nsource: STAGE raw / lowpass / final", fontsize=8); a2.set_ylabel("rho"); a2.legend(fontsize=6)
        a3.plot(np.linspace(0, c["S"], len(d["raw"])), A["gr"], "-o", c=C_RAW, ms=2, lw=0.8, label="raw per-knot Radius (.graph, uniform R_fix)")
        sf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(d["final"][:, :3], axis=0), axis=1))]
        a3.plot(sf, d["final"][:, 3], "-", c=col, lw=1.4, label="final r (STAGE final r=)")
        a3.plot([0, c["S"]], [c["rA"], c["rB"]], ":", c="k", lw=0.8, label="linear taper rA->rB (params)")
        a3.set_title("radius: raw -> final (QJ_RADIUS_SCALE & taper, sec:taper)\nraw s axis = uniform stand-in (raw knot index scaled to S)", fontsize=8)
        a3.set_ylabel("r"); a3.legend(fontsize=6); a3.set_ylim(0, 1.15 * max(A["gr"].max(), d["final"][:, 3].max()))
        kn = d["spline_knots"]; a4.plot(sk, np.linalg.norm(kn[:, 4:7], axis=1), "-o", c=C_LOW, ms=2, lw=0.9)
        a4.set_ylabel("|M_i|"); a4.set_title("C2 natural-cubic knots: |M_i| (2nd-deriv coeff, M_0=M_n-1=0)\n[STAGE spline_knots M=; eq:tridiag]; s = chord arclength", fontsize=8)
        a4.set_xlabel("knot arclength s")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stages"); ap.add_argument("graph"); ap.add_argument("outdir")
    ap.add_argument("--chains", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", default="pairA,pairB"); a = ap.parse_args()
    nodes, edges = parse_graph(a.graph)
    chains = [tuple(int(x) for x in s.split(",")) for s in a.chains.split(";")] if a.chains else pick_chains(nodes, edges, a.seed)
    print("seed", a.seed, "chains", chains)
    for ch, nm in zip(chains, a.names.split(",")):
        tags = [arm_tag(x, y, nodes)[0] for x, y in zip(ch[:-1], ch[1:])]
        st = parse_stages(a.stages, tags); print(nm, ch, tags)
        out = os.path.join(a.outdir, "network_pipeline_%s.png" % nm)
        plot_chain(ch, nodes, edges, st, out, "Arm-centerline pipeline, chain nodes %s -> arms %s (seed %d)" % (ch, tags, a.seed))
        # honesty check
        for t, (x, y) in zip(tags, zip(ch[:-1], ch[1:])):
            e = next(e for e in edges if {e["n0"], e["n1"]} == {x, y}); owner = arm_tag(x, y, nodes)[1]
            g = e["xyz"][::-1] if e["n0"] != owner else e["xyz"]
            print("  spotcheck", t, "max|raw[:3]-graph[:3]| =", np.abs(st[t]["raw"][:3] - g[:3]).max(), "| all-pts:", np.abs(st[t]["raw"] - g).max())
        print("  wrote", out)

if __name__ == "__main__": main()
