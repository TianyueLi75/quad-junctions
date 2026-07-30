"""
Spiral flagella on a unit sphere
=================================

Geometry
--------
A unit sphere (radius 1, centered at the origin) with six identical
single-thread "flagella" growing inward from its surface.

Requirements this config satisfies
-----------------------------------
1. Sphere radius 1, centered at origin.
2. Six uniformly spaced flagella, placed at the octahedron vertices
   (+/-x, +/-y, +/-z) -- the most symmetric way to put 6 points on a sphere.
3. Each flagellum is a single thread (no loop): the base end sits ON the
   surface, the free tip terminates INSIDE the sphere. Nothing penetrates
   back out through the surface.
4. The base connection is perpendicular to the surface (the curve leaves
   along the inward normal; tilt grows only as t^2, so it is exactly
   perpendicular at t=0).
5. Each thread spirals (2.2 turns) while leaning via a mild, smooth
   tangential bend, so the six threads curl into their own regions rather
   than all diving at the shared center.
6. Length is shortened (L = 0.9) so each tip stays on the SAME side of the
   sphere center as its base (verified: tip . base_normal > 0).
7. The centerline is inflated into a thin tube of cross-section radius 0.05.
8. One thread (the +z / "blue" one) gets a small extra bias toward the +y
   thread and away from the -y thread; all other threads are identical.

How it is built
---------------
For each anchor n_hat (a unit vector to a surface point):
  * Build a local frame (n_hat, u, v) with u, v perpendicular to n_hat.
  * March a parameter t from 0 (base) to 1 (tip):
      - inward direction leans from -n_hat toward u by tilt = maxTilt * t^2
      - axis point   = R*n_hat + (L*t) * inward_direction
      - smooth bend  += bendAmt * smootherstep(t) * L  along v
                        (smootherstep = 6t^5 - 15t^4 + 10t^3; it has zero 1st
                         AND 2nd derivative at t=0, so the bend eases in with
                         no kink and no curvature jump -> extremely smooth)
      - optional per-thread bias (blue only), projected perpendicular to
        n_hat so the base stays perpendicular, eased by the same smootherstep
      - spiral offset += 0.13*grow * (cos(a)*u + sin(a)*v),
                         a = turns*2*pi*t, grow ramps 0->1 over the first
                         quarter so the spiral starts cleanly at the base
      - clamp any point back to radius R if it would exit the sphere
  * Sweep a circle of radius 0.05 along the centerline with a
    parallel-transport frame (minimal twist) to get the tube.

Tuning
------
  R          sphere radius (leave at 1 unless you rescale everything).
  L          thread length. Larger reaches deeper; above ~1.0 the tip crosses
             the center to the far side (breaks requirement 6). 0.9 keeps a
             small safety margin.
  turns      spiral winding count over the length. Higher = tighter coil.
  maxTilt    how much the axis leans by the tip (radians-ish). Larger splays
             threads more; too large and neighbors overlap.
  bendAmt    magnitude of the smooth tangential curl. Mild (0.32) keeps the
             clean look; raising it curls threads harder into their sectors.
  TUBE       cross-section radius (0.05). Surface-to-surface gap between two
             threads = (centerline separation) - 2*TUBE.
  bias_amt[i], bias_dir[i]
             per-thread nudge. Only index 4 (+z, "blue") is set (0.18 toward
             +y). Set others to steer individual threads; direction is auto-
             projected perpendicular to that thread's base normal.

Run `python flagella.py` to write flagella_config.json.
"""

import json
import math

# ---- parameters (final config) ----
R = 1.0
L = 0.9
TURNS = 2.2
MAX_TILT = 0.35
BEND_AMT = 0.32
SPIRAL_AMP = 0.13
TUBE_RADIUS = 0.05
SEG = 120                      # samples along each centerline

# six uniformly spaced anchors = octahedron vertices
# order/colors: 0 +x, 1 -x, 2 +y (orange), 3 -y (pink), 4 +z (blue), 5 -z
ANCHORS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

# per-thread extra bend bias (world direction, magnitude); only blue (+z) set
BIAS_DIR = [None, None, None, None, (0.0, 1.0, 0.0), None]   # blue -> toward +y
BIAS_AMT = [0.0, 0.0, 0.0, 0.0, 0.18, 0.0]


# ---- vector helpers ----
def norm(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / m for c in v)

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def add(*vs):
    return tuple(sum(c) for c in zip(*vs))

def scl(v, s):
    return tuple(c*s for c in v)

def smootherstep(x):
    x = max(0.0, min(1.0, x))
    return x*x*x*(x*(x*6 - 15) + 10)


def centerline(fi):
    """Return list of (x, y, z) points for flagellum fi."""
    n = norm(ANCHORS[fi])
    ref = (0, 1, 0) if abs(n[1]) < 0.9 else (1, 0, 0)
    u = norm(cross(n, ref))
    v = norm(cross(n, u))

    pts = []
    for i in range(SEG + 1):
        t = i / SEG
        tilt = MAX_TILT * t * t
        inward = norm(add(scl(n, -math.cos(tilt)), scl(u, math.sin(tilt))))
        c = add(scl(n, R), scl(inward, L * t))

        # main smooth tangential bend along v
        c = add(c, scl(v, BEND_AMT * smootherstep(t) * L))

        # optional per-thread bias, projected perpendicular to n
        if BIAS_DIR[fi] is not None:
            bd = BIAS_DIR[fi]
            bd = tuple(bd[k] - n[k] * dot(bd, n) for k in range(3))  # remove normal part
            bd = norm(bd)
            c = add(c, scl(bd, BIAS_AMT[fi] * smootherstep(t) * L))

        # spiral twirl, grows in over the first quarter
        ang = TURNS * 2 * math.pi * t
        grow = min(1.0, t * 4)
        sr = SPIRAL_AMP * grow
        c = add(c, scl(u, math.cos(ang) * sr), scl(v, math.sin(ang) * sr))

        # keep inside the sphere
        m = math.sqrt(dot(c, c))
        if m > R:
            c = scl(c, R / m)

        pts.append(c)
    return pts


def build_config():
    flagella = []
    for fi, anchor in enumerate(ANCHORS):
        cl = centerline(fi)
        n = norm(anchor)
        tip = cl[-1]
        flagella.append({
            "index": fi,
            "anchor": list(anchor),                 # surface attachment direction
            "base_position": list(scl(n, R)),       # point ON the surface
            "tip_position": list(tip),              # free end INSIDE the sphere
            "tip_on_base_side": dot(tip, n) > 0,    # requirement 6 check
            "centerline": [list(p) for p in cl],
            "tube_radius": TUBE_RADIUS,
        })
    return {
        "sphere": {"radius": R, "center": [0, 0, 0]},
        "count": len(ANCHORS),
        "parameters": {
            "L": L, "turns": TURNS, "max_tilt": MAX_TILT,
            "bend_amt": BEND_AMT, "spiral_amp": SPIRAL_AMP,
            "tube_radius": TUBE_RADIUS, "segments": SEG,
        },
        "flagella": flagella,
    }


def min_separation(cfg):
    """Minimum centerline-to-centerline distance between any two threads."""
    cls = [f["centerline"] for f in cfg["flagella"]]
    best = float("inf")
    for a in range(len(cls)):
        for b in range(a + 1, len(cls)):
            for pa in cls[a]:
                for pb in cls[b]:
                    d = math.dist(pa, pb)
                    if d < best:
                        best = d
    return best


if __name__ == "__main__":
    cfg = build_config()
    with open("flagella_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    sep = min_separation(cfg)
    print("wrote flagella_config.json")
    print(f"min centerline separation: {sep:.3f}")
    print(f"tube surface-to-surface gap: {sep - 2*TUBE_RADIUS:.3f}")
    print("all tips on base side:",
          all(f["tip_on_base_side"] for f in cfg["flagella"]))