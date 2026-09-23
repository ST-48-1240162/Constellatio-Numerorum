"""Label high-hit algebraic numbers on a rendered plane.

A simple integer polynomial (low Brooks h) is a computed root. Hits that
land in a disc around that root are attributed to it. Only roots that have
both a high mass and a closed form are drawn, so the PNG from `algebraics.py`
already names 0, ±1, ±1/2, ±i, the sixth roots, φ, and the rest — no later
annotation pass.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from common.render import to_uint8

CATALOG_MAXH = 8
IDENTIFY_TOL = 0.006
NMS_RADIUS = 0.08
HIT_DECIMALS = 3
MIN_MASS_FRAC = 0.0005
MIN_HITS = 200
MAX_LABELS = 24

WHITE = (255, 255, 255, 255)
BOXFILL = (8, 8, 14, 240)


@dataclass(frozen=True)
class RootLabel:
    x: float
    y: float
    mass: float
    text: str


def _as_int(x: float) -> int:
    return int(round(float(x)))


def _gcd_many(values: list[int]) -> int:
    g = 0
    for v in values:
        g = math.gcd(g, abs(v))
    return g or 1


def primitive_coeffs(coeffs: np.ndarray | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    ints = [_as_int(c) for c in coeffs]
    while ints and ints[-1] == 0:
        ints.pop()
    if not ints:
        return ()
    g = _gcd_many(ints)
    if ints[-1] < 0:
        g = -g
    return tuple(c // g for c in ints)


def extract_square_factor(n: int) -> tuple[int, int]:
    """n = k^2 * rest with rest square-free. n >= 0."""
    k = 1
    i = 2
    while i * i <= n:
        while n % (i * i) == 0:
            n //= i * i
            k *= i
        i += 1
    return k, n


def format_rational(num: int, den: int) -> str:
    if den == 0:
        raise ZeroDivisionError("degree-1 constant polynomial")
    if den < 0:
        num, den = -num, -den
    g = math.gcd(num, den)
    num, den = num // g, den // g
    if den == 1:
        return str(num)
    return rf"\frac{{{num}}}{{{den}}}"


def _format_surd(const: int, rad_coeff: int, radicand: int, den: int, imag: bool) -> str:
    g = _gcd_many([const, rad_coeff, den])
    const, rad_coeff, den = const // g, rad_coeff // g, den // g
    if den < 0:
        const, rad_coeff, den = -const, -rad_coeff, -den
    if radicand == 1 and not imag:
        return format_rational(const + rad_coeff, den)
    term = _surd_term(rad_coeff, radicand, imag)
    if const == 0:
        body = term
    elif rad_coeff >= 0:
        body = rf"{const}+{term}"
    else:
        body = rf"{const}{term}"
    if den == 1:
        return body
    return rf"\frac{{{body}}}{{{den}}}"


def _surd_term(coeff: int, radicand: int, imag: bool) -> str:
    if radicand == 1:
        if imag:
            if coeff == 1:
                return r"\mathrm{i}"
            if coeff == -1:
                return r"-\mathrm{i}"
            return rf"{coeff}\mathrm{{i}}"
        return str(coeff)
    if imag:
        if coeff == 1:
            return rf"\mathrm{{i}}\sqrt{{{radicand}}}"
        if coeff == -1:
            return rf"-\mathrm{{i}}\sqrt{{{radicand}}}"
        return rf"{coeff}\mathrm{{i}}\sqrt{{{radicand}}}"
    if coeff == 1:
        return rf"\sqrt{{{radicand}}}"
    if coeff == -1:
        return rf"-\sqrt{{{radicand}}}"
    return rf"{coeff}\sqrt{{{radicand}}}"


def format_quadratic(z: complex, a: int, b: int, c: int) -> str:
    disc = b * b - 4 * a * c
    two_a = 2 * a
    plus = (-b + np.emath.sqrt(disc)) / two_a
    minus = (-b - np.emath.sqrt(disc)) / two_a
    use_plus = abs(plus - z) <= abs(minus - z)
    sign = 1 if use_plus else -1
    imag = disc < 0
    k, rest = extract_square_factor(abs(disc))
    return _format_surd(-b, sign * k, rest, two_a, imag)


def _angle_key(z: complex, n: int, odd: bool) -> tuple[int, int] | None:
    ang = float(np.angle(z))
    if ang < 0:
        ang += 2 * np.pi
    step = np.pi / n
    k = int(round(ang / step))
    if odd and k % 2 == 0:
        return None
    if not odd and k % 2 == 1:
        return None
    if abs(ang - k * step) > 0.04:
        return None
    if odd:
        p, q = k, n
    else:
        p, q = k, 2 * n
        # e^{i π k / n} with even k = e^{2π i (k/2) / n}
        if k % 2 == 0:
            p, q = k // 2, n
    g = math.gcd(p, q) or 1
    return p // g, q // g


def _exp_i_pi(p: int, q: int) -> str:
    if q == 1:
        return "1" if p % 2 == 0 else "-1"
    if (p, q) == (1, 2):
        return "-1"
    if p == 1:
        return rf"e^{{i\pi/{q}}}"
    return rf"e^{{i\pi {p}/{q}}}"


def format_pure_power(z: complex, n: int, rhs: int) -> str | None:
    if rhs == 1:
        key = _angle_key(z, n, odd=False)
        if key is None:
            return None
        return _exp_i_pi(*key)
    if rhs == -1:
        key = _angle_key(z, n, odd=True)
        if key is None:
            ang = float(np.angle(z))
            if ang < 0:
                ang += 2 * np.pi
            k = int(round(ang * n / np.pi))
            if abs(ang - k * np.pi / n) > 0.04 or k % 2 == 0:
                return None
            p, q = k, n
            g = math.gcd(p, q) or 1
            return _exp_i_pi(p // g, q // g)
        return _exp_i_pi(*key)
    return None


def pretty_root(z: complex, coeffs: tuple[int, ...]) -> str | None:
    if not coeffs:
        return None
    deg = len(coeffs) - 1
    if deg < 1:
        return None
    if deg == 1:
        c0, c1 = coeffs[0], coeffs[1]
        return format_rational(-c0, c1)
    if deg == 2:
        return format_quadratic(z, coeffs[2], coeffs[1], coeffs[0])
    if all(c == 0 for c in coeffs[1:-1]):
        c0, cn = coeffs[0], coeffs[-1]
        if cn != 1 and cn != -1:
            return None
        rhs = -c0 // cn if cn * (-c0 // cn) == -c0 else None
        if rhs not in (1, -1):
            return None
        return format_pure_power(z, deg, rhs)
    return None


def _catalog_entries(maxh: int) -> list[tuple[complex, int, int, tuple[int, ...]]]:
    from algebraics.algebraics import enumerate_abs_coeffs, signed_coeffs

    best: dict[tuple[int, int], tuple[int, int, tuple[int, ...], complex]] = {}
    for h, degree, abs_t in enumerate_abs_coeffs(maxh):
        for coeffs in signed_coeffs(abs_t):
            prim = primitive_coeffs(coeffs)
            if len(prim) < 2:
                continue
            try:
                zs = np.roots(np.asarray(prim[::-1], dtype=np.float64))
            except np.linalg.LinAlgError:
                continue
            for z in zs:
                if not np.isfinite(z.real) or not np.isfinite(z.imag):
                    continue
                key = (int(round(z.real * 1000)), int(round(z.imag * 1000)))
                prev = best.get(key)
                if prev is None or (h, degree) < (prev[0], prev[1]):
                    best[key] = (h, degree, prim, complex(float(z.real), float(z.imag)))
    return [(z, h, deg, prim) for (h, deg, prim, z) in best.values()]


@lru_cache(maxsize=4)
def simple_catalog(maxh: int = CATALOG_MAXH) -> tuple[tuple[complex, str], ...]:
    out: list[tuple[complex, str]] = []
    for z, _h, _deg, prim in _catalog_entries(maxh):
        text = pretty_root(z, prim)
        if text is None:
            continue
        out.append((z, text))
    return tuple(out)


def _spatial_hits(points: np.ndarray, weights: np.ndarray, decimals: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.round(points["x"], decimals) + 1j * np.round(points["y"], decimals)
    uniq, inv = np.unique(z, return_inverse=True)
    hits = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(hits, inv, weights)
    return uniq, hits


def _nms(xs: np.ndarray, ys: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    r2 = radius * radius
    for i in order:
        x, y = float(xs[i]), float(ys[i])
        if any((x - float(xs[j])) ** 2 + (y - float(ys[j])) ** 2 <= r2 for j in keep):
            continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def _identify(z: complex, catalog: tuple[tuple[complex, str], ...], tol: float) -> tuple[complex, str] | None:
    best = None
    best_d = tol
    for cz, text in catalog:
        d = abs(cz - z)
        if d <= best_d:
            best_d = d
            best = (cz, text)
    return best


def find_hotspots(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    max_labels: int = MAX_LABELS,
    min_mass_frac: float = MIN_MASS_FRAC,
    min_hits: float = MIN_HITS,
    identify_tol: float = IDENTIFY_TOL,
    nms_radius: float = NMS_RADIUS,
    catalog_maxh: int = CATALOG_MAXH,
) -> tuple[RootLabel, ...]:
    if points.size == 0:
        return ()
    if weights is None:
        weights = np.ones(points.shape[0], dtype=np.float64)
    catalog = simple_catalog(catalog_maxh)
    if not catalog:
        return ()
    zs, hits = _spatial_hits(points, weights, HIT_DECIMALS)
    if zs.size == 0:
        return ()
    floor = max(min_hits, min_mass_frac * float(hits.max()))
    strong = hits >= floor
    zs, hits = zs[strong], hits[strong]
    if zs.size == 0:
        return ()
    keep = _nms(zs.real, zs.imag, hits, nms_radius)
    labels: list[RootLabel] = []
    for i in keep:
        found = _identify(complex(zs[i]), catalog, identify_tol)
        if found is None:
            continue
        cz, text = found
        labels.append(RootLabel(cz.real, cz.imag, float(hits[i]), text))
        if len(labels) >= max_labels:
            break
    return tuple(labels)


def _pick_font(candidates: tuple[str, ...]) -> str:
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"no font found; tried {candidates}")


@lru_cache(maxsize=64)
def _math_png(text: str, fontsize: float, dpi: int) -> "Image.Image":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"mathtext.fontset": "cm"})
    fig = plt.figure(dpi=dpi)
    fig.patch.set_alpha(0)
    wrapped = r"$" + text + "$"
    obj = fig.text(0, 0, wrapped, fontsize=fontsize, color="white", va="bottom", ha="left", usetex=False)
    fig.canvas.draw()
    bbox = obj.get_window_extent(fig.canvas.get_renderer()).expanded(1.12, 1.22)
    fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)
    obj.set_position((-bbox.x0 / dpi, -bbox.y0 / dpi))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _world_to_screen(x: float, y: float, width: int, height: int, ox: float, oy: float, zoom: float) -> tuple[float, float]:
    sx = (x - ox) * zoom + width / 2.0
    sy = height / 2.0 - (y - oy) * zoom
    return sx, sy


def draw_labels_rgb(
    rgb: np.ndarray,
    labels: tuple[RootLabel, ...],
    ox: float,
    oy: float,
    zoom: float,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    height, width, _ = rgb.shape
    base = Image.fromarray(rgb, "RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = width / 1920.0
    pad = max(4, int(round(8 * scale)))
    circle_r = max(6, int(round(11 * scale)))
    circle_w = max(2, int(round(3 * scale)))
    fontsize = max(12.0, 18.0 * scale)
    dpi = max(96, int(round(110 * scale)))
    gap = max(8, int(round(14 * scale)))
    used: list[tuple[int, int, int, int]] = []

    def overlaps(rect):
        l, t, r, b = rect
        for ol, ot, orr, ob in used:
            if l < orr and r > ol and t < ob and b > ot:
                return True
        return False

    visible = []
    for lab in labels:
        sx, sy = _world_to_screen(lab.x, lab.y, width, height, ox, oy, zoom)
        if sx < -40 or sy < -40 or sx > width + 40 or sy > height + 40:
            continue
        visible.append((lab, sx, sy))
    visible.sort(key=lambda item: -item[0].mass)

    for lab, sx, sy in visible:
        cx, cy = int(round(sx)), int(round(sy))
        draw.ellipse(
            [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
            outline=WHITE,
            width=circle_w,
        )
        glyph = _math_png(lab.text, fontsize, dpi)
        w, h = glyph.size
        box_w, box_h = w + pad * 2, h + pad * 2
        on_real = abs(lab.y) < 0.08
        above = (cx - box_w // 2, cy - circle_r - gap - box_h)
        below = (cx - box_w // 2, cy + circle_r + gap)
        if on_real and lab.x < 0:
            ordered = (below, above)
        elif on_real:
            ordered = (above, below)
        else:
            ordered = (above, below)
        candidates = ordered + (
            (cx + circle_r + gap, cy - box_h // 2),
            (cx - circle_r - gap - box_w, cy - box_h // 2),
        )
        chosen = None
        for left, top in candidates:
            rect = (left, top, left + box_w, top + box_h)
            if rect[0] < 4 or rect[1] < 4 or rect[2] > width - 4 or rect[3] > height - 4:
                continue
            if overlaps(rect):
                continue
            chosen = rect
            break
        if chosen is None:
            left, top = candidates[0]
            chosen = (
                min(max(4, left), width - box_w - 4),
                min(max(4, top), height - box_h - 4),
                0,
                0,
            )
            chosen = (chosen[0], chosen[1], chosen[0] + box_w, chosen[1] + box_h)
        left, top, right, bottom = chosen
        draw.rounded_rectangle([left, top, right, bottom], radius=1, fill=BOXFILL)
        overlay.paste(glyph, (left + pad, top + pad), glyph)
        used.append(chosen)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(out)


def apply_root_labels(
    img: np.ndarray,
    points: np.ndarray,
    ox: float,
    oy: float,
    zoom: float,
    weights: np.ndarray | None = None,
    labels: tuple[RootLabel, ...] | None = None,
) -> np.ndarray:
    if labels is None:
        labels = find_hotspots(points, weights)
    rgb = to_uint8(img)
    return draw_labels_rgb(rgb, labels, ox, oy, zoom).astype(np.float32) / 255.0
