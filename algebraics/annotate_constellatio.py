# Usage: python3 annotate_constellatio.py [algebraics_h17_7680px.png]
# Writes constellatio_annotated.png and rebuilds constellatio.pdf via pdflatex.

import io
import re
import math
import subprocess
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams.update({
    "mathtext.fontset": "cm",
})

from algebraics.regions.defs import CALLOUT_REGIONS, NUMBER_TAGS, REF_H, REF_W
from common.render import DEGREE_COLORS

SCRIPT_DIR = Path(__file__).resolve().parent
MATH_CAL_SAMPLE = r"$\mathrm{Ag}$"

def _pick_font(candidates: tuple[str, ...]) -> str:
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"no font found; tried {candidates}")

HELV_REG = _pick_font((
    "/usr/share/fonts/gsfonts/NimbusSans-Regular.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
))
HELV_BOLD = _pick_font((
    "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
))
HELV_ITALIC = _pick_font((
    "/usr/share/fonts/gsfonts/NimbusSans-Italic.otf",
    "/usr/share/fonts/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
))
WHITE = (255, 255, 255, 255)
BOXFILL = (8, 8, 14, 178)
LEADER  = (255, 214, 0, 255)

base = overlay = d = None
W = H = sx = sy = 0
MATH_DPI = 120
f_head = f_body = f_legend_h = f_legend = None

LEGEND_TITLE = "Colour key"
LEGEND_ROWS = (
    (1, "degree 1"),
    (2, "degree 2"),
    (3, "degree 3"),
    (4, "degree 4"),
    (5, "degree 5"),
    (6, "degree 6"),
    (7, "degree 7"),
    (8, "degree 8"),
    (0, "degree \u2265 9"),
)
LEGEND_ANCHOR_REF = (3824, 2144)
_MATH_CACHE = {}
_MATH_PT_CACHE = {}

def px(x): return int(round(x * sx))
def py(y): return int(round(y * sy))
def pr(r): return max(2, int(round(r * (sx + sy) / 2)))
def fs(n): return max(8, int(round(n * sx)))

def init_canvas(in_path, ref_w=REF_W, ref_h=REF_H):
    global base, overlay, d, W, H, sx, sy, MATH_DPI, f_head, f_body, f_legend_h, f_legend
    global _MATH_CACHE, _MATH_PT_CACHE
    _MATH_CACHE.clear()
    _MATH_PT_CACHE.clear()

    base = Image.open(in_path).convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    sx, sy = W / ref_w, H / ref_h
    MATH_DPI = max(120, int(round(120 * sx)))
    f_head = ImageFont.truetype(HELV_BOLD, fs(48))
    f_body = ImageFont.truetype(HELV_REG, fs(42))
    f_legend_h = ImageFont.truetype(HELV_BOLD, fs(44))
    f_legend = ImageFont.truetype(HELV_REG, fs(40))

def inter_text_height(font, sample="Ag"):
    bbox = d.textbbox((0, 0), sample, font=font)
    return bbox[3] - bbox[1]

def _render_math_raw(text, fontsize, color=WHITE):
    rgb = tuple(c / 255 for c in color[:3])
    fig = plt.figure(dpi=MATH_DPI)
    fig.patch.set_alpha(0)
    text_obj = fig.text(0, 0, text, fontsize=fontsize, color=rgb, va="bottom", ha="left")
    fig.canvas.draw()
    bbox = text_obj.get_window_extent(fig.canvas.get_renderer()).expanded(1.02, 1.05)
    fig.set_size_inches(bbox.width / MATH_DPI, bbox.height / MATH_DPI)
    text_obj.set_position((-bbox.x0 / MATH_DPI, -bbox.y0 / MATH_DPI))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=MATH_DPI, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

def math_fontsize_for(inter_font):
    key = (inter_font.size, inter_font.getname())
    if key in _MATH_PT_CACHE:
        return _MATH_PT_CACHE[key]
    target = inter_text_height(inter_font)
    lo, hi = 1.0, float(inter_font.size * 3)
    for _ in range(28):
        mid = (lo + hi) / 2
        if _render_math_raw(MATH_CAL_SAMPLE, mid).height < target:
            lo = mid
        else:
            hi = mid
    pt = (lo + hi) / 2
    _MATH_PT_CACHE[key] = pt
    return pt

def render_math(text, inter_font, color=WHITE):
    if text.startswith("$") and text.endswith("$"):
        text = r"$\boldsymbol{" + text[1:-1] + "}$"
    fontsize = math_fontsize_for(inter_font)
    key = (text, fontsize, color[:3], inter_font.size, inter_font.getname())
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    img = _render_math_raw(text, fontsize, color)
    _MATH_CACHE[key] = img
    return img

def italic_font_for(font):
    return ImageFont.truetype(HELV_ITALIC, font.size)

def parse_mixed_line(text):
    parts = []
    pos = 0
    for m in re.finditer(r"\$([^$]*)\$|\*([^*]+)\*", text):
        if m.start() > pos:
            parts.append(("text", text[pos:m.start()]))
        if m.group(1) is not None:
            parts.append(("math", m.group(1)))
        else:
            parts.append(("italic", m.group(2)))
        pos = m.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))
    return parts

def render_text_segment(text, font, color=WHITE):
    if not text:
        return Image.new("RGBA", (0, 0), (0, 0, 0, 0))
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-bbox[0], -bbox[1]), text, font=font, fill=color)
    return img

def render_mixed_line(text, font, color=WHITE):
    parts = parse_mixed_line(text)
    if not parts:
        return Image.new("RGBA", (0, 0), (0, 0, 0, 0))
    if len(parts) == 1 and parts[0][0] == "text":
        return render_text_segment(parts[0][1], font, color)
    segments = []
    for kind, content in parts:
        if kind == "text":
            segments.append(render_text_segment(content, font, color))
        elif kind == "italic":
            segments.append(render_text_segment(content, italic_font_for(font), color))
        else:
            segments.append(render_math(f"${content}$", font, color))
    line_h = max(inter_text_height(font), max(seg.height for seg in segments))
    out = Image.new("RGBA", (sum(seg.width for seg in segments), line_h), (0, 0, 0, 0))
    x = 0
    for seg in segments:
        out.paste(seg, (x, line_h - seg.height), seg)
        x += seg.width
    return out

def line_entry(text, font):
    if "$" in text or "*" in text:
        return ("mixed", text, font)
    return ("text", text, font)

def line_size(entry):
    kind, content, param = entry
    gap = px(16)
    if kind == "text":
        bbox = d.textbbox((0, 0), content, font=param)
        return bbox[2] - bbox[0], bbox[3] - bbox[1] + gap
    img = render_mixed_line(content, param)
    line_h = max(inter_text_height(param), img.height)
    return img.width, line_h + gap

def measure_box(head, lines, pad=None):
    if pad is None:
        pad = px(24)
    entries = ([line_entry(head, f_head)] if head else []) + [
        line_entry(t, f_body) for t in lines
    ]
    sizes = [line_size(e) for e in entries]
    w = max(w for w, _ in sizes) + pad * 2
    h = sum(h for _, h in sizes) + pad * 2
    return w, h, entries, pad

def draw_box(rect, entries_pad):
    entries, pad = entries_pad
    left, top, right, bottom = rect
    d.rounded_rectangle(rect, radius=1, fill=BOXFILL)
    ty = top + pad
    for kind, content, param in entries:
        if kind == "text":
            bbox = d.textbbox((0, 0), content, font=param)
            d.text((left + pad, ty), content, font=param, fill=WHITE)
            ty += bbox[3] - bbox[1] + px(16)
        else:
            img = render_mixed_line(content, param)
            line_h = max(inter_text_height(param), img.height)
            overlay.paste(img, (left + pad, ty), img)
            ty += line_h + px(16)

def rect_from_anchor(anchor_xy, mode, w, h):
    x, y = anchor_xy
    if mode == "la":    return (x, y, x + w, y + h)
    if mode == "ra":    return (x - w, y, x, y + h)
    if mode == "la_b":  return (x, y - h, x + w, y)
    if mode == "ra_b":  return (x - w, y - h, x, y)
    if mode == "ma":    return (x - w // 2, y, x + w // 2, y + h)
    if mode == "ma_b":  return (x - w // 2, y - h, x + w // 2, y)
    return (x, y, x + w, y + h)

def nearest_edge_point(rect, from_xy):
    left, top, right, bottom = rect
    fx, fy = from_xy
    cx = min(max(fx, left), right)
    cy = min(max(fy, top), bottom)
    return cx, cy

def ray_to_rect_edge(rect, origin, direction):
    ox, oy = origin
    dx, dy = direction
    left, top, right, bottom = rect
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return nearest_edge_point(rect, origin)
    hits = []
    if abs(dx) > 1e-9:
        for x_edge in (left, right):
            t = (x_edge - ox) / dx
            if t > 0:
                y = oy + t * dy
                if top - 0.5 <= y <= bottom + 0.5:
                    hits.append((t, x_edge, y))
    if abs(dy) > 1e-9:
        for y_edge in (top, bottom):
            t = (y_edge - oy) / dy
            if t > 0:
                x = ox + t * dx
                if left - 0.5 <= x <= right + 0.5:
                    hits.append((t, x, y_edge))
    if not hits:
        return None
    _, x, y = min(hits, key=lambda h: h[0])
    return (x, y)

def draw_arrow(tip, tail, fill, width):
    tx, ty = tip
    ax, ay = tail
    dx, dy = tx - ax, ty - ay
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    head_len = min(max(width * 3.2, px(22)), length * 0.38)
    head_half = min(max(width * 1.55, px(11)), head_len * 0.62)
    shaft_end = (tx - ux * head_len * 0.55, ty - uy * head_len * 0.55)
    d.line([tail, shaft_end], fill=fill, width=width)
    left = (tx - ux * head_len + nx * head_half, ty - uy * head_len + ny * head_half)
    right = (tx - ux * head_len - nx * head_half, ty - uy * head_len - ny * head_half)
    d.polygon([tip, left, right], fill=fill)

def callout(cx, cy, r, anchor_xy, mode, head, lines, leader_angle_offset_deg=0, leader_length_frac=1.0, leader_width=None, marker="circle"):
    marker_rect = (cx - r, cy - r, cx + r, cy + r)
    width = max(1, px(7))
    if marker == "square":
        d.rectangle(list(marker_rect), outline=WHITE, width=width)
    else:
        d.ellipse(list(marker_rect), outline=WHITE, width=width)
    w, h, entries, pad = measure_box(head, lines)
    rect = rect_from_anchor(anchor_xy, mode, w, h)
    box_center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    if marker == "square":
        start = ray_to_rect_edge(marker_rect, (cx, cy), (box_center[0] - cx, box_center[1] - cy))
        if start is None:
            start = nearest_edge_point(marker_rect, box_center)
    else:
        ang = math.atan2(box_center[1] - cy, box_center[0] - cx)
        start = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    end0 = nearest_edge_point(rect, (cx, cy))
    if leader_angle_offset_deg:
        ang0 = math.atan2(end0[1] - start[1], end0[0] - start[0])
        line_ang = ang0 - math.radians(leader_angle_offset_deg)
        end = ray_to_rect_edge(rect, start, (math.cos(line_ang), math.sin(line_ang)))
        if end is None:
            end = end0
    else:
        end = end0
    if leader_length_frac != 1.0:
        end = (
            start[0] + leader_length_frac * (end[0] - start[0]),
            start[1] + leader_length_frac * (end[1] - start[1]),
        )
    if leader_width is None:
        leader_width = max(2, px(10))
    draw_arrow(start, end, LEADER, leader_width)
    draw_box(rect, (entries, pad))
    return rect

def _legend_rgb(degree_index: int) -> tuple[int, int, int]:
    r, g, b = DEGREE_COLORS[degree_index]
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))

def draw_legend(anchor_ref=LEGEND_ANCHOR_REF):
    pad = px(24)
    swatch = px(32)
    gap = px(18)
    row_gap = px(14)
    title_gap = px(18)

    title_bbox = d.textbbox((0, 0), LEGEND_TITLE, font=f_legend_h)
    title_h = title_bbox[3] - title_bbox[1]
    labels = [label for _, label in LEGEND_ROWS]
    label_bboxes = [d.textbbox((0, 0), label, font=f_legend) for label in labels]
    text_w = max(bb[2] - bb[0] for bb in label_bboxes)
    row_heights = [max(swatch, bb[3] - bb[1]) for bb in label_bboxes]
    panel_w = pad * 2 + swatch + gap + text_w
    panel_h = (
        pad * 2
        + title_h
        + title_gap
        + sum(row_heights)
        + row_gap * (len(LEGEND_ROWS) - 1)
    )

    anchor_x = px(anchor_ref[0])
    anchor_y = min(py(anchor_ref[1]), H - px(16))
    rect = rect_from_anchor((anchor_x, anchor_y), "ra_b", panel_w, panel_h)
    left, top, _, _ = rect
    d.rounded_rectangle(rect, radius=1, fill=BOXFILL)

    ty = top + pad
    d.text((left + pad, ty), LEGEND_TITLE, font=f_legend_h, fill=WHITE)
    ty += title_h + title_gap

    for (degree_index, label), row_h, label_bbox in zip(
        LEGEND_ROWS, row_heights, label_bboxes, strict=True
    ):
        sx = left + pad
        sy = ty + (row_h - swatch) // 2
        rgb = _legend_rgb(degree_index)
        swatch_rect = [sx, sy, sx + swatch, sy + swatch]
        d.rectangle(swatch_rect, fill=rgb + (255,))
        if degree_index == 0:
            d.rectangle(swatch_rect, outline=(160, 160, 160, 255), width=max(1, px(2)))
        label_y = ty + (row_h - (label_bbox[3] - label_bbox[1])) // 2
        d.text((sx + swatch + gap, label_y), label, font=f_legend, fill=WHITE)
        ty += row_h + row_gap

def draw_callouts():
    for region in CALLOUT_REGIONS:
        callout(
            px(region.ref_cx),
            py(region.ref_cy),
            pr(region.ref_r),
            (px(region.anchor_ref_x), py(region.anchor_ref_y)),
            region.anchor_mode,
            region.head,
            list(region.lines),
            leader_angle_offset_deg=region.leader_angle_offset_deg,
            leader_length_frac=region.leader_length_frac,
            leader_width=max(2, int(round(px(10) * (2 / 3 if region.id in ("D", "D_prime", "E") else 1)))),
            marker=region.marker,
        )

def draw_number_tags():
    for tag in NUMBER_TAGS:
        cx, cy, r = px(tag.ref_cx), py(tag.ref_cy), pr(tag.ref_r)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=max(1, px(7)))
        w, h, entries, pad = measure_box(tag.label, [], pad=px(10))
        rect = rect_from_anchor(
            (px(tag.label_ref_x), py(tag.label_ref_y)),
            tag.label_mode,
            w,
            h,
        )
        draw_box(rect, (entries, pad))

def build_pdf(in_path=None):
    in_path = Path(in_path or SCRIPT_DIR / "algebraics_h17_7680px.png")
    annotated_path = SCRIPT_DIR / "constellatio_annotated.png"

    init_canvas(in_path)
    draw_callouts()
    draw_number_tags()
    draw_legend()
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(annotated_path)
    alias_path = SCRIPT_DIR / "constellatio_numerorum_annotated.png"
    out.save(alias_path)
    print("saved", annotated_path.resolve(), (W, H))
    print("saved", alias_path.resolve())

    for stale in ("constellatio.aux", "constellatio.log", "constellatio.pdf"):
        p = SCRIPT_DIR / stale
        if p.exists():
            p.unlink()
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "constellatio.tex"],
        cwd=SCRIPT_DIR,
        check=True,
    )
    pdf_path = SCRIPT_DIR / "constellatio.pdf"
    print("saved", pdf_path.resolve())
    return pdf_path

def main():
    parser = argparse.ArgumentParser(
        description="Annotate algebraics_h17_7680px.png and rebuild constellatio.pdf.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="source PNG (default: algebraics/algebraics_h17_7680px.png)",
    )
    args = parser.parse_args()
    build_pdf(args.input)

if __name__ == "__main__":
    main()
