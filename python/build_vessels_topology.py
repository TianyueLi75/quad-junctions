#!/usr/bin/env python3
"""
Arterial/venous vascular-tree topology for `src/ybifurc-vessels-bie.cpp`.

Single source of truth (hand-derived from `arterial_venous_smoothed_nolabels.svg`, verified
against the figure) for the 20-junction geometry:

  * arterial binary tree  (root far left, grows left->right)   junctions 0..9
  * venous binary tree     (root far right, grows right->left)  junctions 10..19 (mirror)
  * 11 leaf connectors A_i <-> V_i joining the two trees across the middle.
    Four form closed racetrack LENSES (both leaves share an arterial sub-junction AND a
    venous sub-junction): pairs (1,2),(3,4),(6,7),(9,10)  ==  the user's A12/A34/A67/A9(10).
    Connectors 0,5,8 are lone (their sibling is an internal subtree, not a co-leaf).

Coordinates are stored in SVG pixels (y-DOWN, as in the file). The C++ driver maps
    X = s*(x-340),  Y = s*(270-y),  Z = 0     (planar z=0; s set at run time).

Run:  bash python/run_py.sh python/build_vessels_topology.py
Outputs:
  * python/vessels_skeleton.png              -- 2D skeleton for side-by-side SVG comparison
  * include/quad_junctions/vessels_tree_data.hpp  -- static tables consumed by the driver
"""
import math
import os

# --------------------------------------------------------------------------------------
# Topology tables (SVG pixel coords, y-down).
# --------------------------------------------------------------------------------------
# Junctions: name, x, y, parent-index (-1 = root), generation (root=0).
# Order MUST be BFS (parents before children) so the driver can read a parent's world seam.
JUNCS = [
    # arterial (root far left, stem points -x)
    ("J1",     80, 270, -1, 0),   # 0  root
    ("J2",    132, 160,  0, 1),   # 1
    ("J3",    132, 380,  0, 1),   # 2
    ("J4",    178, 100,  1, 2),   # 3
    ("J5",    178, 220,  1, 2),   # 4
    ("J6",    178, 320,  2, 2),   # 5
    ("J7",    178, 440,  2, 2),   # 6
    ("Jsub1", 223, 130,  3, 3),   # 7
    ("Jsub2", 223, 350,  5, 3),   # 8
    ("Jsub3", 273, 470,  6, 3),   # 9
    # venous (root far right, stem points +x) -- mirror layout
    ("VJ1",   625, 270, -1, 0),   # 10 root
    ("VJ2",   575, 160, 10, 1),   # 11
    ("VJ3",   575, 380, 10, 1),   # 12
    ("VJ4",   527, 100, 11, 2),   # 13
    ("VJ5",   527, 220, 11, 2),   # 14
    ("VJ6",   527, 320, 12, 2),   # 15
    ("VJ7",   527, 440, 12, 2),   # 16
    ("VJsub1",490, 130, 13, 3),   # 17
    ("VJsub2",490, 350, 15, 3),   # 18
    ("VJsub3",490, 470, 16, 3),   # 19
]
NAME2ID = {n: i for i, (n, *_ ) in enumerate(JUNCS)}

# Leaves: name -> (junction-id, endpoint-x, endpoint-y).  The endpoint is where the SVG
# branch arm transitions to the horizontal run; the driver uses it to pick which branch
# seam (arm1/arm2) points at the connector and as the run waypoint.
LEAVES = {
    "A0":  (NAME2ID["J4"],    205,  70),
    "A1":  (NAME2ID["Jsub1"], 250, 110),
    "A2":  (NAME2ID["Jsub1"], 250, 150),
    "A3":  (NAME2ID["J5"],    205, 190),
    "A4":  (NAME2ID["J5"],    205, 250),
    "A5":  (NAME2ID["J6"],    205, 290),
    "A6":  (NAME2ID["Jsub2"], 250, 330),
    "A7":  (NAME2ID["Jsub2"], 250, 370),
    "A8":  (NAME2ID["J7"],    205, 410),
    "A9":  (NAME2ID["Jsub3"], 300, 450),
    "A10": (NAME2ID["Jsub3"], 300, 490),
    "V0":  (NAME2ID["VJ4"],    490,  70),
    "V1":  (NAME2ID["VJsub1"], 455, 110),
    "V2":  (NAME2ID["VJsub1"], 455, 150),
    "V3":  (NAME2ID["VJ5"],    490, 190),
    "V4":  (NAME2ID["VJ5"],    490, 250),
    "V5":  (NAME2ID["VJ6"],    490, 290),
    "V6":  (NAME2ID["VJsub2"], 455, 330),
    "V7":  (NAME2ID["VJsub2"], 455, 370),
    "V8":  (NAME2ID["VJ7"],    490, 410),
    "V9":  (NAME2ID["VJsub3"], 455, 450),
    "V10": (NAME2ID["VJsub3"], 455, 490),
}

# Connectors: arterial-leaf, venous-leaf, is_lens (0/1).  Indexed 0..10 top-to-bottom.
LENS_IDX = {1, 2, 3, 4, 6, 7, 9, 10}
CONNECTORS = [("A%d" % i, "V%d" % i, 1 if i in LENS_IDX else 0) for i in range(11)]


# --------------------------------------------------------------------------------------
# Geometry helpers (mirror the driver's branch assignment so this plot validates it).
# --------------------------------------------------------------------------------------
def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-12 else (0.0, 0.0)


def stem_dir(jid):
    """Junction stem (arm0) points toward its parent; roots point outward (arterial -x, venous +x)."""
    _, x, y, parent, _ = JUNCS[jid]
    if parent < 0:
        return (-1.0, 0.0) if jid < 10 else (1.0, 0.0)
    px, py = JUNCS[parent][1], JUNCS[parent][2]
    return _unit(px - x, py - y)


def rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def targets_of(jid):
    """List of (kind, key, pos) this junction's two branches feed: child junctions + leaves."""
    out = []
    for cid, (_, x, y, parent, _) in enumerate(JUNCS):
        if parent == jid:
            out.append(("child", cid, (x, y)))
    for lname, (lj, lx, ly) in LEAVES.items():
        if lj == jid:
            out.append(("leaf", lname, (lx, ly)))
    return out


def assign_branches(jid):
    """Map the two 120-deg branch directions to this junction's two targets by best dot match.
    Returns dict target-key -> branch unit-dir."""
    sd = stem_dir(jid)
    b = [rot(sd, +120.0), rot(sd, -120.0)]  # the two branch axes
    tg = targets_of(jid)
    assert len(tg) == 2, f"{JUNCS[jid][0]} has {len(tg)} targets"
    jx, jy = JUNCS[jid][1], JUNCS[jid][2]
    d = [_unit(t[2][0] - jx, t[2][1] - jy) for t in tg]
    s00 = b[0][0] * d[0][0] + b[0][1] * d[0][1] + b[1][0] * d[1][0] + b[1][1] * d[1][1]
    s01 = b[0][0] * d[1][0] + b[0][1] * d[1][1] + b[1][0] * d[0][0] + b[1][1] * d[0][1]
    if s00 >= s01:
        return {tg[0][1]: b[0], tg[1][1]: b[1]}, tg
    return {tg[0][1]: b[1], tg[1][1]: b[0]}, tg


def gen_of(jid):
    return JUNCS[jid][4]


def radius_px(gen, base=6.0, ratio=0.8):
    """Skeleton stroke half-width ~ 0.8^gen (matches the driver's scale = 0.8^gen)."""
    return base * (ratio ** gen)


# --------------------------------------------------------------------------------------
# Skeleton plot.
# --------------------------------------------------------------------------------------
def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    RED, BLUE, GRAY, LENSC = "#B03A2E", "#3E7CAE", "#8A96A3", "#D98A3D"

    # intra-tree arms (single-corner: junction -> branch stub -> child) + leaf stubs.
    for jid, (name, x, y, parent, gen) in enumerate(JUNCS):
        col = RED if jid < 10 else BLUE
        assign, tg = assign_branches(jid)
        rj = radius_px(gen)
        for kind, key, pos in tg:
            bd = assign[key]
            stub = (x + 2.2 * rj * bd[0], y + 2.2 * rj * bd[1])  # short lead along branch dir
            if kind == "child":
                cx, cy = pos
                lw = radius_px(gen_of(key))
                # single corner: stub -> (elbow at child stem entry) -> child center
                ax.plot([x, stub[0]], [y, stub[1]], color=col, lw=lw, solid_capstyle="round")
                ax.plot([stub[0], cx], [stub[1], cy], color=col, lw=lw, solid_capstyle="round")
            else:
                lx, ly = pos  # leaf endpoint; stub -> leaf endpoint (branch arm)
                lw = radius_px(gen)
                ax.plot([x, lx], [y, ly], color=col, lw=lw, solid_capstyle="round")

    # leaf connectors: A endpoint -> (horizontal run) -> V endpoint.
    for ci, (aL, vL, is_lens) in enumerate(CONNECTORS):
        aj, ax_, ay = LEAVES[aL]
        vj, vx, vy = LEAVES[vL]
        c = LENSC if is_lens else GRAY
        lw = radius_px(gen_of(aj)) * 0.9
        ax.plot([ax_, vx], [ay, vy], color=c, lw=lw, solid_capstyle="round",
                zorder=1, alpha=0.9)
        ax.text((ax_ + vx) / 2, (ay + vy) / 2 - 4, str(ci), color=c, fontsize=7,
                ha="center", va="bottom")

    # junction bodies + labels.
    for jid, (name, x, y, parent, gen) in enumerate(JUNCS):
        col = RED if jid < 10 else BLUE
        ax.add_patch(plt.Circle((x, y), radius_px(gen) * 1.3, color=col, zorder=3))
        ax.text(x, y - radius_px(gen) * 1.3 - 3, name, color=col, fontsize=6.5,
                ha="center", va="bottom", zorder=4)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # SVG y is down -> match the figure orientation
    ax.set_title("vessels skeleton (red=arterial L->R, blue=venous R->L, orange=lens connectors)\n"
                 "compare to arterial_venous_smoothed_nolabels.svg")
    ax.autoscale_view()
    ax.margins(0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


# --------------------------------------------------------------------------------------
# Emit the C++ data header.
# --------------------------------------------------------------------------------------
def emit_hpp(path):
    lines = []
    lines.append("// AUTO-GENERATED by python/build_vessels_topology.py -- DO NOT EDIT BY HAND.")
    lines.append("// Arterial/venous vascular-tree topology (SVG pixel coords, y-down).")
    lines.append("#ifndef QUAD_JUNCTIONS_VESSELS_TREE_DATA_HPP")
    lines.append("#define QUAD_JUNCTIONS_VESSELS_TREE_DATA_HPP")
    lines.append("namespace quad_junctions {")
    lines.append("namespace vessels_data {")
    lines.append("")
    lines.append("struct VJunc { int id; double x, y; int parent; int gen; };")
    lines.append("// Connector: arterial junction + arterial leaf endpoint, venous junction + endpoint, is_lens.")
    lines.append("struct VConn { int a_jct; double ax, ay; int v_jct; double vx, vy; int is_lens; };")
    lines.append("")
    lines.append(f"static const int n_junc = {len(JUNCS)};")
    lines.append("static const VJunc juncs[n_junc] = {")
    for jid, (name, x, y, parent, gen) in enumerate(JUNCS):
        lines.append(f"  {{ {jid:2d}, {float(x):6.1f}, {float(y):6.1f}, {parent:3d}, {gen} }},  // {name}")
    lines.append("};")
    lines.append("")
    lines.append(f"static const int n_conn = {len(CONNECTORS)};")
    lines.append("static const VConn conns[n_conn] = {")
    for ci, (aL, vL, is_lens) in enumerate(CONNECTORS):
        aj, ax_, ay = LEAVES[aL]
        vj, vx, vy = LEAVES[vL]
        lines.append(f"  {{ {aj:2d}, {float(ax_):6.1f}, {float(ay):6.1f}, {vj:2d}, "
                     f"{float(vx):6.1f}, {float(vy):6.1f}, {is_lens} }},  // {ci}: {aL}<->{vL}")
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace vessels_data")
    lines.append("}  // namespace quad_junctions")
    lines.append("#endif")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    # sanity: every junction has exactly two branch targets; connectors reference known leaves.
    for jid in range(len(JUNCS)):
        assert len(targets_of(jid)) == 2, f"{JUNCS[jid][0]} !=2 targets"
    assert len(CONNECTORS) == 11
    plot(os.path.join(here, "vessels_skeleton.png"))
    emit_hpp(os.path.join(root, "include", "quad_junctions", "vessels_tree_data.hpp"))
    print("OK: 20 junctions, 11 connectors, lenses at", sorted(LENS_IDX))


if __name__ == "__main__":
    main()
