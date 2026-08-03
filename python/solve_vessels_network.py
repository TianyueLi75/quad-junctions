#!/usr/bin/env python3
"""
Reduced Hagen-Poiseuille RESISTANCE-NETWORK solve for the 20-junction arterial/venous vessels network
-- the analytic reference that src/ybifurc-vessels-flow-bie.cpp checks its full boundary-integral Stokes
solve against at the mid-point of each connector arm.

Same topology and method as the in-driver C++ solve (quad_junctions/vessels_network_solve.hpp): every arm
is modelled as a Hagen-Poiseuille resistor R = 8*mu*L/(pi*r^4), the total volumetric flux p_in is injected
at the arterial-tree root (junction 0) and withdrawn at the venous-tree root (junction 10), and Kirchhoff's
laws (a graph-Laplacian system L p = b) are solved for the nodal pressures. This script reuses the topology
tables from build_vessels_topology.py so there is one source of truth.

Modelling choices (the classic "network method" idealization):
  * arm LENGTH  = straight world chord between the two junction centers (svgs=0.06 world units).
    The C++ driver uses the true BENT centerline length (single-corner / racetrack arms are a little
    longer than the chord), so absolute pressures differ from C++ by the arms' bending (~few %); the
    dimensionless FLUX SPLIT below is robust to this.
  * arm RADIUS  = R0 * 0.8^gen (the driver's generational taper; for an intra-tree arm the coarser
    parent generation wins, r = R0*0.8^min(gen0,gen1)).
  * FLUX FRACTIONS Q_k/p_in and dimensionless pressures are INDEPENDENT of R0 and of svgs (all radii
    scale as R0, all lengths as svgs, so conductance ratios cancel both) -- they are fixed by the
    dimensionless geometry alone. Absolute R (and thus absolute pressure for a given p_in) scales as
    R0^-4 * svgs; we report pressures per unit p_in with mu=1, R0=0.26931 (the driver's default seam
    radius) so the numbers are directly comparable to the C++ header line.

Run:   bash python/run_py.sh python/solve_vessels_network.py [p_in]
Outputs:
  * stdout table  -- per-junction pressures + per-connector flux/pressure-drop in units of p_in
  * python/vessels_pressures.png  -- vessels_skeleton.png annotated with pressures and connector fluxes
"""
import math
import os
import sys

import numpy as np

# Single source of truth for the topology (same tables the C++ driver's header is generated from).
import build_vessels_topology as topo

SVGS = 0.06        # model units per SVG pixel (the driver default)
R0 = 0.26931       # gen-0 seam radius at the driver defaults (world units); cancels for flux fractions
TAPER = 0.8        # radius ~ TAPER^gen (matches the driver's junction scale)
MU = 1.0           # dynamic viscosity (Stokes)
ROOT_IN = topo.NAME2ID["J1"]    # arterial root (id 0) -- inflow port
ROOT_OUT = topo.NAME2ID["VJ1"]  # venous root  (id 10) -- outflow port


def world(x, y):
    """SVG pixel (y-down) -> world (planar), matching the driver: X=s*(x-340), Y=s*(270-y)."""
    return SVGS * (x - 340.0), SVGS * (270.0 - y)


def radius_of_gen(gen):
    return R0 * (TAPER ** gen)


def build_edges():
    """List of resistive arms as (j0, j1, r, L, is_conn), mirroring the C++ segs (root caps excluded)."""
    edges = []
    # intra-tree parent->child arms (radius = coarser/parent generation).
    for jid, (_, x, y, parent, gen) in enumerate(topo.JUNCS):
        if parent < 0:
            continue
        px, py, pgen = topo.JUNCS[parent][1], topo.JUNCS[parent][2], topo.JUNCS[parent][4]
        (X0, Y0), (X1, Y1) = world(px, py), world(x, y)
        L = math.hypot(X1 - X0, Y1 - Y0)
        r = radius_of_gen(min(gen, pgen))
        edges.append((parent, jid, r, L, False))
    # arterial<->venous leaf connectors (both endpoints same generation).
    for (aL, vL, _is_lens) in topo.CONNECTORS:
        aj, ax_, ay = topo.LEAVES[aL]
        vj, vx, vy = topo.LEAVES[vL]
        (X0, Y0), (X1, Y1) = world(ax_, ay), world(vx, vy)
        L = math.hypot(X1 - X0, Y1 - Y0)
        r = radius_of_gen(topo.gen_of(aj))
        edges.append((aj, vj, r, L, True))
    return edges


def solve(edges, p_in):
    n = len(topo.JUNCS)
    Lap = np.zeros((n, n))
    conductance = []
    for (j0, j1, r, L, is_conn) in edges:
        g = math.pi * r ** 4 / (8.0 * MU * L)   # 1/R
        conductance.append(g)
        Lap[j0, j0] += g; Lap[j1, j1] += g
        Lap[j0, j1] -= g; Lap[j1, j0] -= g
    b = np.zeros(n)
    b[ROOT_IN] = p_in
    b[ROOT_OUT] = -p_in
    P = np.linalg.pinv(Lap) @ b        # min-norm (mean-zero) pressures
    out = []
    for (j0, j1, r, L, is_conn), g in zip(edges, conductance):
        dP = P[j0] - P[j1]
        out.append(dict(j0=j0, j1=j1, r=r, L=L, g=g, dP=dP, Q=g * dP, is_conn=is_conn))
    return P, out


def print_table(P, sol, p_in):
    print(f"\n=== reduced Hagen-Poiseuille resistance network (mu={MU}, R0={R0}, svgs={SVGS}, p_in={p_in}) ===")
    print("junction pressures (mean-zero, proportional to p_in):")
    for jid, (name, *_rest) in enumerate(topo.JUNCS):
        end = "\n" if jid % 5 == 4 else "   "
        print(f"  {name:>7}[{jid:2d}] = {P[jid]: .5g}", end=end)
    print("\narterial<->venous connectors (arm i:  jA->jV   r        L        dP         Q         Q/p_in):")
    conn_i, Qsum = 0, 0.0
    for e in sol:
        if not e["is_conn"]:
            continue
        Qsum += e["Q"]
        print(f"  conn {conn_i:2d}:  {e['j0']:2d}->{e['j1']:2d}  {e['r']:.4f}  {e['L']:7.4f}  "
              f"{e['dP']: .5g}  {e['Q']: .5g}  {e['Q']/p_in: .5f}")
        conn_i += 1
    print(f"  sum of connector fluxes Q = {Qsum:.6g}  (target p_in={p_in}; global conservation)")


def annotate(P, sol, p_in, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9))
    RED, BLUE, GRAY, LENSC = "#B03A2E", "#3E7CAE", "#8A96A3", "#D98A3D"

    # arms (same primitives as build_vessels_topology.plot, thin, as a backdrop).
    for jid, (name, x, y, parent, gen) in enumerate(topo.JUNCS):
        col = RED if jid < 10 else BLUE
        assign, tg = topo.assign_branches(jid)
        rj = topo.radius_px(gen)
        for kind, key, pos in tg:
            bd = assign[key]
            stub = (x + 2.2 * rj * bd[0], y + 2.2 * rj * bd[1])
            if kind == "child":
                cx, cy = pos
                lw = topo.radius_px(topo.gen_of(key))
                ax.plot([x, stub[0]], [y, stub[1]], color=col, lw=lw, solid_capstyle="round", alpha=0.5)
                ax.plot([stub[0], cx], [stub[1], cy], color=col, lw=lw, solid_capstyle="round", alpha=0.5)
            else:
                lx, ly = pos
                ax.plot([x, lx], [y, ly], color=col, lw=topo.radius_px(gen), solid_capstyle="round", alpha=0.5)

    # connectors annotated with flux fraction Q/p_in.
    conn_i = 0
    conns = [e for e in sol if e["is_conn"]]
    for ci, (aL, vL, is_lens) in enumerate(topo.CONNECTORS):
        aj, ax_, ay = topo.LEAVES[aL]
        vj, vx, vy = topo.LEAVES[vL]
        c = LENSC if is_lens else GRAY
        e = conns[ci]
        lw = topo.radius_px(topo.gen_of(aj)) * 0.9
        ax.plot([ax_, vx], [ay, vy], color=c, lw=lw, solid_capstyle="round", zorder=1, alpha=0.9)
        ax.text((ax_ + vx) / 2, (ay + vy) / 2 - 4, f"{ci}: Q={e['Q']/p_in:.3f}", color="k",
                fontsize=7, ha="center", va="bottom", zorder=5)

    # junction bodies + pressure labels.
    for jid, (name, x, y, parent, gen) in enumerate(topo.JUNCS):
        col = RED if jid < 10 else BLUE
        ax.add_patch(plt.Circle((x, y), topo.radius_px(gen) * 1.3, color=col, zorder=3))
        ax.text(x, y - topo.radius_px(gen) * 1.3 - 3, f"P={P[jid]:.3g}", color=col, fontsize=6.5,
                ha="center", va="bottom", zorder=4)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("vessels resistance-network solution (P at junctions, Q/p_in on connectors)\n"
                 f"Hagen-Poiseuille R=8*mu*L/(pi r^4), inject p_in={p_in} at J1, withdraw at VJ1")
    ax.autoscale_view(); ax.margins(0.05); fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print("wrote", out_path)


def main():
    p_in = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    here = os.path.dirname(os.path.abspath(__file__))
    edges = build_edges()
    P, sol = solve(edges, p_in)
    print_table(P, sol, p_in)
    annotate(P, sol, p_in, os.path.join(here, "vessels_pressures.png"))


if __name__ == "__main__":
    main()
