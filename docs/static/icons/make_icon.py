"""DVLA-RL++ icon, tied to the method (split-tile template).

left  half  VISION   : a support image as a patch grid. Patches on the bird are kept (bright, blue outline),
                       background reed patches are faded out  -> complementary semantic purification
right half  LANGUAGE : dual-level semantics. Short chips = local attribute phrases, long bar = global description.
                       blue = intrinsic bank, rose = nuisance bank
centre badge         : the RL gate a_l as a gauge mixing the image-guided (blue) and text-guided (rose) branches
"""
import math
import sys

NAVY = "#26364a"
BLUE = "#4f86b8"
BLUE_L = "#8fbbe3"
ROSE = "#c96f7a"
ROSE_L = "#eea3ab"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
T0, B0 = 8.0, 120.0


def edge(y, top, bot):
    return top + (bot - top) * (y - T0) / (B0 - T0)


def poly_left(lt, lb):
    return f"0,0 {edge(0, lt, lb):.2f},0 {edge(128, lt, lb):.2f},128 0,128"


def poly_right(rt, rb):
    return f"{edge(0, rt, rb):.2f},0 128,0 128,128 {edge(128, rt, rb):.2f},128"


# ------------------------------------------------------------------ bird (colour, cartoon outline)
def bird(k=1.0):
    a = f'stroke="{NAVY}" stroke-width="{2.0 * k}" stroke-linejoin="round" stroke-linecap="round"'
    return "\n".join([
        f'<path d="M54 57 L29 69 L33 76 L57 64 Z" fill="#8b5a31" {a}/>',
        f'<path d="M64 62 L62.5 71 M71 61 L71 69" fill="none" {a}/>',
        f'<ellipse cx="68" cy="52" rx="19.5" ry="13" transform="rotate(-20 68 52)" fill="#b8763f" {a}/>',
        '<ellipse cx="75" cy="56" rx="11" ry="6.2" transform="rotate(-22 75 56)" fill="#f4dcb6"/>',
        f'<circle cx="84" cy="38" r="10.5" fill="#b8763f" {a}/>',
        '<path d="M75.5 33.5 Q83 25.5 92 31.5 Q84 30 76.5 36 Z" fill="#734626"/>',
        f'<path d="M55 49 Q68 38 83 46 Q77 60 59 60 Q53 56 55 49 Z" fill="#8b5a31" {a}/>',
        '<path d="M61 50 Q67 47 73 49 M60 54.5 Q67 51.5 76 53.5 M63 58 Q69 56 75 57" fill="none" stroke="#f4dcb6" stroke-width="1.6" stroke-linecap="round"/>',
        f'<path d="M93.5 35.5 L103.5 38.8 L93.2 42 Z" fill="#eaa53a" {a}/>',
        '<circle cx="86.6" cy="36.2" r="2.2" fill="#1b1f24"/><circle cx="87.3" cy="35.4" r="0.75" fill="#fff"/>',
    ])


def vision_panel(small):
    """Support image as a patch grid: the 3x2 block covering the bird is kept, context patches are faded."""
    s = ['<rect x="-10" y="-10" width="150" height="150" fill="url(#img)"/>']
    for x, top in ((14, 16), (24, 26), (56, 12)):
        s.append(f'<path d="M{x} 124 Q{x + 2} 70 {x - 1} {top}" fill="none" stroke="#c4a666" stroke-width="{2.4 if not small else 3.2}" stroke-linecap="round"/>')
    if not small:
        s.append('<ellipse cx="24" cy="22" rx="3.8" ry="8.5" fill="#e3c88f" stroke="#b89a5c" stroke-width="1.2"/>')
        s.append('<ellipse cx="56" cy="18" rx="3.6" ry="8" fill="#e3c88f" stroke="#b89a5c" stroke-width="1.2"/>')
    s.append('<g transform="translate(36.4 48.5) scale(0.74) translate(-66 -52)">')
    s.append(f'<path d="M-40 92 Q60 76 150 56" fill="none" stroke="#8a7a3c" stroke-width="{3.8 if not small else 5.0}" stroke-linecap="round"/>')
    s.append(bird(1.0 if not small else 1.45))
    s.append('</g>')
    cell, x0, y0 = 20.0, 8.0, 8.0
    keep = {(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
    for i in range(-1, 6):
        for j in range(-1, 7):
            if (i, j) in keep:
                continue
            s.append(f'<rect x="{x0 + i * cell}" y="{y0 + j * cell}" width="{cell}" height="{cell}" fill="#ffffff" opacity="0.58"/>')
    gw = 1.1 if not small else 1.7
    for i in range(-1, 7):
        p = x0 + i * cell
        s.append(f'<path d="M{p} -10 V140 M-10 {p} H140" stroke="#ffffff" stroke-width="{gw}"/>')
    kx, ky = x0 + 0 * cell, y0 + 1 * cell
    s.append(f'<rect x="{kx + 1.5}" y="{ky + 1.5}" width="{3 * cell - 3}" height="{2 * cell - 3}" rx="4" fill="none" '
             f'stroke="{BLUE}" stroke-width="{2.6 if not small else 3.6}"/>')
    return "\n    ".join(s)


def language_panel(small, rt, rb):
    s = ['<rect x="-10" y="-10" width="150" height="150" fill="url(#txt)"/>']
    h = 8.0 if not small else 12.0
    if not small:
        rows = [  # y, [(width, colour)]   local intrinsic, local nuisance, global intrinsic
            (20, [(22, BLUE), (14, BLUE)]),
            (33, [(16, BLUE), (22, BLUE)]),
            (49, [(19, ROSE), (13, ROSE)]),
            (65, [(46, BLUE)]),
            (104, [(34, ROSE)]),
        ]
    else:
        rows = [
            (18, [(26, BLUE)]),
            (36, [(20, BLUE), (14, BLUE)]),
            (56, [(26, ROSE)]),
        ]
    for y, chips in rows:
        x = edge(y + h, rt, rb) + (6 if not small else 7)
        if y > 90:
            x = max(x, 84)
        for w, c in chips:
            s.append(f'<rect x="{x:.1f}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{c}"/>')
            x += w + (4 if not small else 5)
    return "\n    ".join(s)


def gauge(cx, cy, r, k=1.0):
    """RL gate: needle mixes the image-guided (blue) and text-guided (rose) arcs."""
    R = r * 0.58
    gx, gy = cx, cy + r * 0.2

    def pt(deg, rad=R):
        t = math.radians(deg)
        return gx + rad * math.cos(t), gy - rad * math.sin(t)

    def arc(a0, a1, col):
        (x0, y0), (x1, y1) = pt(a0), pt(a1)
        return (f'<path d="M{x0:.2f} {y0:.2f} A{R:.2f} {R:.2f} 0 0 1 {x1:.2f} {y1:.2f}" fill="none" '
                f'stroke="{col}" stroke-width="{r * 0.2:.2f}" stroke-linecap="round"/>')

    nx, ny = pt(122, R * 0.92)
    return "\n  ".join([
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{NAVY}" stroke="#ffffff" stroke-width="{3.5 * k}"/>',
        arc(180, 98, BLUE_L),
        arc(82, 0, ROSE_L),
        f'<path d="M{gx:.2f} {gy:.2f} L{nx:.2f} {ny:.2f}" stroke="#ffffff" stroke-width="{r * 0.15:.2f}" stroke-linecap="round"/>',
        f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="{r * 0.16:.2f}" fill="#ffffff"/>',
    ])


def build(shadow=False, inset=8.0, rx=24.0, small=False):
    k = 1.45 if small else 1.0
    lt, lb, rt, rb = (70.0, 52.0, 77.0, 59.0)
    tile = f'x="{inset}" y="{inset}" width="{128 - 2 * inset}" height="{128 - 2 * inset}" rx="{rx}"'
    lp, rp = poly_left(lt, lb), poly_right(rt, rb)
    if small:
        bx, by, br = 97.0, 96.0, 23.0
    else:
        by = 92.0
        bx = (edge(by, lt, lb) + edge(by, rt, rb)) / 2
        br = 18.0
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">',
        '  <title>DVLA-RL++</title>',
        '  <defs>',
        f'    <clipPath id="tile"><rect {tile}/></clipPath>',
        f'    <clipPath id="left"><polygon points="{lp}"/></clipPath>',
        f'    <clipPath id="right"><polygon points="{rp}"/></clipPath>',
        '    <linearGradient id="img" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d6e7f6"/><stop offset="1" stop-color="#efe6d2"/></linearGradient>',
        '    <linearGradient id="txt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fdfaf5"/><stop offset="1" stop-color="#f6efe6"/></linearGradient>',
        '  </defs>',
    ]
    if shadow:
        out.append(f'  <rect x="{inset + 2.5}" y="{inset + 4}" width="{128 - 2 * inset}" height="{128 - 2 * inset}" rx="{rx}" fill="{NAVY}" opacity="0.16"/>')
    out += [
        f'  <rect {tile} fill="#ffffff"/>',
        '  <g clip-path="url(#tile)">',
        f'   <g clip-path="url(#left)">\n    {vision_panel(small)}\n   </g>',
        f'   <g clip-path="url(#right)">\n    {language_panel(small, rt, rb)}\n   </g>',
        f'   <polygon points="{lp}" fill="none" stroke="{NAVY}" stroke-width="{2 * k}"/>',
        f'   <polygon points="{rp}" fill="none" stroke="{NAVY}" stroke-width="{2 * k}"/>',
        '  </g>',
        f'  {gauge(bx, by, br, k)}',
        f'  <rect {tile} fill="none" stroke="{NAVY}" stroke-width="{4 * k}"/>',
        '</svg>',
    ]
    return "\n".join(out) + "\n"


open(f"{OUT}/logo.svg", "w").write(build(shadow=True))
open(f"{OUT}/icon.svg", "w").write(build(inset=4.0, rx=26.0))
open(f"{OUT}/favicon.svg", "w").write(build(inset=3.0, rx=28.0, small=True))
print("wrote logo.svg, icon.svg, favicon.svg to", OUT)
