"""
Turn ascii-motion JSON exports into one small animated SVG.

    # a single looping animation
    python design/ascii_to_svg.py design/coffee_mug.json design/mug.svg

    # an intro played N times, then the loop forever
    python design/ascii_to_svg.py design/coffee_mug.json design/mug.svg \
        --intro design/coffee_mug_load.json --repeat 3 --gap 4

The canvas is cropped to the cells that are ever drawn. Cells present in
every loop frame become one static "body" group; every other distinct frame
becomes a <g> that CSS keyframes switch on and off with steps() timing, so
the SVG animates both inline and as <img src="mug.svg">. Cells are 6x10
units, the tall-monospace proportion ascii-motion shows on screen.

Intro: leading and trailing blank frames in the export are dropped and
replaced by --gap blank frames at the start of each repeat (pass --gap -1 to
keep the export's own blanks). Intro frames identical to the loop's body just
show the body, so the last repeat hands straight over to the loop.

Colours are remapped through COLORS so the dark mug drawn on ascii-motion's
black canvas still reads on Crema's brown background. Edit that table to
recolour; anything not listed keeps its exported colour.
"""
import argparse
import json

COLORS = {
    "#3d251e": "#a86f3b",  # mug body  -> Crema brown
    "#a08679": "#cdb08f",  # steam     -> faded brown
}
CELL_W, CELL_H = 6, 10
BLOCK = "█"


# ------------------------------------------------------------------ frames

def load_frames(path):
    """[(cells, duration_ms)] with cells = {(x, y): (char, colour)}."""
    doc = json.load(open(path, encoding="utf-8"))
    frames = []
    for f in doc["frames"]:
        colors = json.loads(f.get("colors", {}).get("foreground") or "{}")
        cells = {}
        for y, line in enumerate(f.get("content", [])):
            for x, ch in enumerate(line):
                if ch != " ":
                    cells[(x, y)] = (ch, colors.get(f"{x},{y}", "#ffffff"))
        frames.append((cells, f["duration"]))
    return frames


def merge_holds(frames):
    out = []
    for cells, dur in frames:
        if out and out[-1][0] == cells:
            out[-1][1] += dur
        else:
            out.append([cells, dur])
    return [(c, d) for c, d in out]


def strip_blanks(frames):
    while frames and not frames[0][0]:
        frames = frames[1:]
    while frames and not frames[-1][0]:
        frames = frames[:-1]
    return frames


# ------------------------------------------------------------------ drawing

def rect(xs, xe, y, col, x0, y0):
    return (f'<rect x="{(xs - x0) * CELL_W}" y="{(y - y0) * CELL_H}" '
            f'width="{(xe - xs + 1) * CELL_W}" height="{CELL_H}" fill="{col}"/>')


def shapes(cells, x0, y0):
    """Runs of same-coloured block cells become one rect; other chars become text."""
    parts, by_row = [], {}
    for (x, y), (ch, col) in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        by_row.setdefault(y, []).append((x, ch, COLORS.get(col, col)))
    for y, row in by_row.items():
        run = None  # [x_start, x_end, colour]
        for x, ch, col in row:
            if ch == BLOCK and run and run[1] == x - 1 and run[2] == col:
                run[1] = x
                continue
            if run:
                parts.append(rect(*run[:2], y, run[2], x0, y0))
                run = None
            if ch == BLOCK:
                run = [x, x, col]
            else:
                parts.append(f'<text x="{(x - x0) * CELL_W + CELL_W / 2}" '
                             f'y="{(y - y0) * CELL_H + CELL_H * 0.8}" font-size="{CELL_H}" '
                             f'font-family="monospace" text-anchor="middle" fill="{col}">{ch}</text>')
        if run:
            parts.append(rect(*run[:2], y, run[2], x0, y0))
    return "".join(parts)


# ---------------------------------------------------------------- animation

def windows(frames, key):
    """[(start%, end%)] of the frames where key(cells) is true, over one cycle."""
    total = sum(d for _, d in frames)
    out, t = [], 0.0
    for cells, dur in frames:
        if key(cells):
            out.append((100 * t / total, 100 * (t + dur) / total))
        t += dur
    return out


def keyframes(name, on):
    """Opacity 1 inside the `on` windows, 0 elsewhere, held with steps(1,end)."""
    if not on:
        return ""
    keys = [f"0%{{opacity:{1 if on[0][0] == 0 else 0}}}"]
    for s, e in on:
        if s > 0:
            keys.append(f"{s:.4f}%{{opacity:1}}")
        if e < 99.999:
            keys.append(f"{e:.4f}%{{opacity:0}}")
    keys.append(f"100%{{opacity:{1 if on[-1][1] >= 99.999 else 0}}}")
    return f"@keyframes {name}{{{''.join(keys)}}}"


def layers(frames, static, prefix, x0, y0, anim):
    """CSS + <g> per distinct non-static, non-empty frame in `frames`."""
    css, groups, seen = [], [], {}
    for cells, _ in frames:
        if not cells or cells == static:
            continue
        k = frozenset(cells.items())
        if k not in seen:
            seen[k] = f"{prefix}{len(seen) + 1}"
    for k, cls in seen.items():
        on = windows(frames, lambda c, k=k: frozenset(c.items()) == k)
        css.append(keyframes(f"mug-{cls}", on))
        css.append(f".mug .{cls}{{animation:mug-{cls} {anim}}}")
        groups.append(f'<g class="{cls}" opacity="0">{shapes(dict(k), x0, y0)}</g>')
    return css, groups


def build(loop_path, intro_path=None, repeat=3, gap=4):
    loop = merge_holds(load_frames(loop_path))
    body = {k: v for k, v in loop[0][0].items() if all(c.get(k) == v for c, _ in loop)}

    intro = []
    if intro_path:
        raw = load_frames(intro_path)
        step = raw[0][1]
        intro = merge_holds(raw)
        if gap >= 0:
            intro = strip_blanks(intro)
            if gap:
                intro = [({}, gap * step)] + intro
        intro = merge_holds(intro)

    every = [c for c, _ in loop + intro]
    xs = [x for c in every for x, _ in c]
    ys = [y for c in every for _, y in c]
    x0, y0 = min(xs), min(ys)
    w, h = (max(xs) - x0 + 1) * CELL_W, (max(ys) - y0 + 1) * CELL_H

    loop_ms = sum(d for _, d in loop)
    intro_ms = round(sum(d for _, d in intro), 2) * repeat  # same rounding as the CSS below
    css, groups = [], []

    if intro:
        cycle = sum(d for _, d in intro)
        c, g = layers(intro, body, "i", x0, y0, f"{cycle:.2f}ms steps(1,end) 0ms {repeat} forwards")
        css += c; groups += g
        css.append(keyframes("mug-body", windows(intro, lambda cells: cells == body)))
        css.append(f".mug .body{{opacity:0;animation:mug-body {cycle:.2f}ms steps(1,end) 0ms {repeat} forwards}}")

    c, g = layers(loop, body, "f", x0, y0, f"{loop_ms:.2f}ms steps(1,end) {intro_ms:.2f}ms infinite")
    css += c; groups += g
    css.append("@media (prefers-reduced-motion:reduce){.mug *{animation:none!important}.mug .body{opacity:1}}")

    return (f'<svg class="mug" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'style="shape-rendering:crispEdges" role="img" aria-label="steaming coffee mug">'
            f'<style>{"".join(css)}</style>'
            f'<g class="body">{shapes(body, x0, y0)}</g>{"".join(groups)}</svg>')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("loop", help="ascii-motion JSON that loops forever")
    ap.add_argument("out", help="SVG to write")
    ap.add_argument("--intro", help="ascii-motion JSON played first")
    ap.add_argument("--repeat", type=int, default=3, help="times to play the intro (default 3)")
    ap.add_argument("--gap", type=int, default=4,
                    help="blank frames before each intro repeat; -1 keeps the export's blanks (default 4)")
    a = ap.parse_args()
    svg = build(a.loop, a.intro, a.repeat, a.gap)
    open(a.out, "w", encoding="utf-8").write(svg + "\n")
    print(f"wrote {a.out} ({len(svg)} bytes)")
