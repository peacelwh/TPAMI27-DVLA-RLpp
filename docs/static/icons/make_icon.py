"""DVLA-RL++ mark: a bird assembled from image patches.

The bird (the paper's running example) is solid at the head and dissolves into square patches towards the
wings and tail. Pale patches merge into the bird (retained intrinsic evidence), a few rose patches drift away
(rejected nuisance evidence). Deep blue tile with a soft glow.
"""
import math
import random
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ------------------------------------------------------------------ geometry (cubic segments)
BODY = [
    ("M", (96, 33)),
    ("C", (90, 29), (84, 30.5), (81, 35)),
    ("C", (76, 44), (68, 54), (60, 61)),
    ("L", (24, 79)),
    ("C", (35, 77), (43, 74.5), (49, 73.5)),
    ("L", (31, 97)),
    ("C", (43, 89), (53, 80), (62, 73)),
    ("C", (72, 66), (84, 58), (90, 50)),
    ("C", (94, 45.5), (98, 41), (98, 37)),
    ("C", (98, 35.5), (97.4, 34), (96, 33)),
]
HEAD = [
    ("M", (98.5, 37)),
    ("C", (98.5, 42), (94.5, 45.5), (89.5, 45.5)),
    ("C", (84.5, 45.5), (80.5, 41.5), (80.5, 36.5)),
    ("C", (80.5, 31.5), (84.5, 27.5), (89.5, 27.5)),
    ("C", (94.5, 27.5), (98.5, 32), (98.5, 37)),
]
BEAK = [
    ("M", (96.5, 32.8)),
    ("C", (100, 33.6), (104, 35), (108.5, 36.6)),
    ("C", (104, 37.6), (100, 38.6), (97, 40)),
]
NEAR_WING = [
    ("M", (84, 41)),
    ("C", (76, 27), (60, 13), (33, 8)),
    ("C", (49, 17), (59, 30), (64, 48)),
    ("C", (70, 53.5), (79, 51.5), (86, 43)),
]
FAR_WING = [
    ("M", (76, 52)),
    ("C", (64, 45), (49, 39), (29, 37)),
    ("C", (43, 45), (55, 54), (66, 60)),
    ("C", (71, 58.5), (74.5, 55.5), (76, 52)),
]
SHAPES = [FAR_WING, BODY, HEAD, BEAK, NEAR_WING]


def to_d(shape):
    out = []
    for seg in shape:
        cmd, pts = seg[0], seg[1:]
        out.append(cmd + " ".join(f"{x:g} {y:g}" for x, y in pts))
    return " ".join(out) + " Z"


def sample(shape, n=48):
    poly, cur = [], None
    for seg in shape:
        cmd, pts = seg[0], seg[1:]
        if cmd == "M":
            cur = pts[0]; poly.append(cur)
        elif cmd == "L":
            cur = pts[0]; poly.append(cur)
        else:
            p0, (p1, p2, p3) = cur, pts
            for i in range(1, n + 1):
                t = i / n; u = 1 - t
                x = u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0]
                y = u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]
                poly.append((x, y))
            cur = p3
    return poly


POLYS = [sample(s) for s in SHAPES]


def inside(x, y):
    for poly in POLYS:
        c = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                c = not c
        if c:
            return True
    return False


# ------------------------------------------------------------------ dissolve
# progress along the dissolve axis: 0 = solid (upper right), 1 = fully dissolved (lower left)
AX0, AX1 = (80.0, 46.0), (26.0, 66.0)


def progress(x, y):
    vx, vy = AX1[0] - AX0[0], AX1[1] - AX0[1]
    t = ((x - AX0[0]) * vx + (y - AX0[1]) * vy) / (vx * vx + vy * vy)
    return max(0.0, min(1.0, t))


T_SOLID = 0.30   # where the solid bird ends along the dissolve axis


def coverage(cx, cy, cell):
    q = cell * 0.3
    pts = [(cx, cy), (cx - q, cy - q), (cx + q, cy - q), (cx - q, cy + q), (cx + q, cy + q)]
    return sum(inside(x, y) for x, y in pts)


def patches(cell, seed=11, small=False):
    rnd = random.Random(seed)
    kept, rejected = [], []
    vx, vy = AX1[0] - AX0[0], AX1[1] - AX0[1]
    L = math.hypot(vx, vy); ux, uy = vx / L, vy / L
    n = int(130 // cell) + 2
    for gx in range(-1, n):
        for gy in range(-1, n):
            cx, cy = (gx + 0.5) * cell, (gy + 0.5) * cell
            t = progress(cx, cy)
            if t < T_SOLID or coverage(cx, cy, cell) < 2:
                continue
            d = max(0.0, (t - T_SOLID) / (1 - T_SOLID))
            if d > 0.25 and rnd.random() < 0.22 * d:
                continue
            size = cell * (0.90 - 0.42 * d ** 1.2)
            drift = 6.5 * d ** 1.8 * (0.5 + rnd.random())
            op = 1.0 - 0.55 * d ** 1.5
            kept.append((cx + ux * drift, cy + uy * drift, size, op))
    # rejected nuisance patches leave the tail along the drift direction
    rose = [((20, 70), 1.0, 0.9), ((16, 85), 0.8, 0.65), ((28, 95), 0.9, 0.75), ((22, 104), 0.65, 0.5)]
    if small:
        rose = [((12, 70), 1.25, 0.9), ((16, 92), 1.1, 0.75)]
    for (x, y), k, op in rose:
        rejected.append((x, y, cell * k, op))
    return kept, rejected


def rects(items, fill, rx):
    return "\n    ".join(
        f'<rect x="{x - s / 2:.2f}" y="{y - s / 2:.2f}" width="{s:.2f}" height="{s:.2f}" rx="{rx}" '
        f'fill="{fill}" opacity="{op:.2f}"/>' for x, y, s, op in items)


def build(inset=4.0, rx=28.0, small=False):
    cell = 4.6 if not small else 7.0
    vx, vy = AX1[0] - AX0[0], AX1[1] - AX0[1]
    L = math.hypot(vx, vy); ux, uy = vx / L, vy / L; nx, ny = -uy, ux
    T_HALF = T_SOLID - 0.12
    qx, qy = AX0[0] + T_HALF * vx, AX0[1] + T_HALF * vy
    step = []
    for gx in range(-1, int(130 // cell) + 2):
        for gy in range(-1, int(130 // cell) + 2):
            cx, cy = (gx + 0.5) * cell, (gy + 0.5) * cell
            if T_HALF - 0.02 <= progress(cx, cy) < T_SOLID:
                step.append(f'<rect x="{gx * cell - 0.4:.2f}" y="{gy * cell - 0.4:.2f}" width="{cell + 0.8:.2f}" height="{cell + 0.8:.2f}"/>')
    step_cells = "".join(step)
    pts = [(qx + nx * 300, qy + ny * 300), (qx + nx * 300 - ux * 300, qy + ny * 300 - uy * 300),
           (qx - nx * 300 - ux * 300, qy - ny * 300 - uy * 300), (qx - nx * 300, qy - ny * 300)]
    solid_poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    kept, rejected = patches(cell, small=small)
    bird_paths = "\n      ".join(f'<path d="{to_d(s)}"/>' for s in SHAPES)
    w = 128 - 2 * inset
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
  <title>DVLA-RL++</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#4a86c4"/>
      <stop offset="0.5" stop-color="#2a4f82"/>
      <stop offset="1" stop-color="#16233b"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.66" cy="0.32" r="0.55">
      <stop offset="0" stop-color="#a9d2f5" stop-opacity="0.6"/>
      <stop offset="1" stop-color="#9cc8ef" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="wingfill" gradientUnits="userSpaceOnUse" x1="84" y1="40" x2="52" y2="26">
      <stop offset="0" stop-color="#f4f9ff"/>
      <stop offset="1" stop-color="#c3dbf2"/>
    </linearGradient>
    <linearGradient id="birdfill" gradientUnits="userSpaceOnUse" x1="100" y1="26" x2="40" y2="90">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#cfe2f5"/>
    </linearGradient>
    <clipPath id="solid"><polygon points="{solid_poly}"/>{step_cells}</clipPath>
    <clipPath id="tile"><rect x="{inset}" y="{inset}" width="{w}" height="{w}" rx="{rx}"/></clipPath>
  </defs>
  <rect x="{inset}" y="{inset}" width="{w}" height="{w}" rx="{rx}" fill="url(#bg)"/>
  <g clip-path="url(#tile)">
    <rect width="128" height="128" fill="url(#glow)"/>
    <g transform="translate(64 66) scale(0.95) translate(-68 -53)">
    {rects(rejected, "#f2a7b3", 1.2)}
    {rects(kept, "url(#birdfill)", 1.0)}
    <g fill="url(#birdfill)" clip-path="url(#solid)">
      {bird_paths}
      <path d="{to_d(NEAR_WING)}" fill="url(#wingfill)"/>
    </g>
    <circle cx="91.6" cy="34.6" r="{2.1 if not small else 2.8}" fill="#1f3555"/>
    <circle cx="92.3" cy="33.9" r="{0.7 if not small else 0.9}" fill="#ffffff"/>
    </g>
  </g>
  <rect x="{inset + 0.75}" y="{inset + 0.75}" width="{w - 1.5}" height="{w - 1.5}" rx="{rx - 0.75}" fill="none" stroke="#ffffff" stroke-opacity="0.14" stroke-width="1.5"/>
</svg>
'''


open(f"{OUT}/logo.svg", "w").write(build())
open(f"{OUT}/favicon.svg", "w").write(build(inset=2.0, rx=30.0, small=True))
open(f"{OUT}/touch.svg", "w").write(build(inset=0.0, rx=0.01))
print("wrote", OUT)
