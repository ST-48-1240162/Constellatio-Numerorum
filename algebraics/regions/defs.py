"""Callout regions (annotation) and per-degree full-map plot exports."""

from __future__ import annotations

from dataclasses import dataclass

REF_W, REF_H = 3840, 2160
DEFAULT_WIDTH, DEFAULT_HEIGHT = 7680, 4320


def default_zoom(height: int = DEFAULT_HEIGHT) -> float:
    return height / 5.0


def ref_to_screen(ref_x: float, ref_y: float, width: int, height: int) -> tuple[float, float]:
    return ref_x * width / REF_W, ref_y * height / REF_H


def screen_to_world(
    sx: float,
    sy: float,
    width: int,
    height: int,
    ox: float = 0.0,
    oy: float = 0.0,
    zoom: float | None = None,
) -> tuple[float, float]:
    zoom = default_zoom(height) if zoom is None else zoom
    wx = ox + (sx - width / 2.0) / zoom
    wy = oy - (sy - height / 2.0) / zoom
    return wx, wy


def ref_radius_px(ref_r: float, width: int, height: int) -> float:
    return ref_r * (width / REF_W + height / REF_H) / 2.0


def world_radius_from_ref(ref_r: float, width: int, height: int, zoom: float | None = None) -> float:
    zoom = default_zoom(height) if zoom is None else zoom
    return ref_radius_px(ref_r, width, height) / zoom


@dataclass(frozen=True)
class CalloutRegion:
    id: str
    ref_cx: float
    ref_cy: float
    ref_r: float
    head: str
    lines: tuple[str, ...]
    anchor_ref_x: float
    anchor_ref_y: float
    anchor_mode: str
    leader_angle_offset_deg: float = 0.0
    leader_length_frac: float = 1.0
    view_padding: float = 2.8
    point_margin: float = 1.05
    marker: str = "circle"

    def world_circle(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        ox: float = 0.0,
        oy: float = 0.0,
        zoom: float | None = None,
    ) -> tuple[float, float, float]:
        zoom = default_zoom(height) if zoom is None else zoom
        sx, sy = ref_to_screen(self.ref_cx, self.ref_cy, width, height)
        wx, wy = screen_to_world(sx, sy, width, height, ox, oy, zoom)
        rw = world_radius_from_ref(self.ref_r, width, height, zoom)
        return wx, wy, rw

    def view_viewport(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> tuple[float, float, float]:
        cx, cy, r = self.world_circle(width, height)
        span = self.view_padding * 2.0 * r
        zoom = min(width, height) / span
        return cx, cy, zoom

    def circle_radius_px(self, zoom: float, width: int, height: int) -> int:
        _, _, rw = self.world_circle(width, height)
        return max(2, int(round(rw * zoom)))


@dataclass(frozen=True)
class NumberTag:
    ref_cx: float
    ref_cy: float
    ref_r: float
    label: str
    label_ref_x: float
    label_ref_y: float
    label_mode: str = "ma_b"


CALLOUT_REGIONS: tuple[CalloutRegion, ...] = (
    CalloutRegion(
        "A",
        1920, 1080, 145,
        r"A Origin ($z = 0$)",
        (
            r"Every degree contributes roots at $z=0$",
            "(set the constant term to zero) — so all",
            "nine hues pile up here, additively mixing",
            "toward white, with gold where the lower",
            "degrees still dominate in brightness.",
        ),
        1170, 1750, "ma",
    ),
    CalloutRegion(
        "B",
        2352, 1080, 100,
        r"B $z \approx +1$ (rational)",
        (
            "Degree 1 is red in the hue table.",
            "Roots pile up on this integer and",
            "clip the colour to white/gold.",
        ),
        3800, 1180, "ra",
    ),
    CalloutRegion(
        "C",
        1920, 640, 65,
        r"C $z \approx \mathrm{i}$ (quadratic)",
        (
            r"Root of $z^2+1=0$ (degree 2), green in the hue table.",
        ),
        2450, 300, "la",
        leader_angle_offset_deg=30,
    ),
    CalloutRegion(
        "D",
        2137, 706, 80,
        "D Degree-6 rosette",
        (
            r"$z = e^{i\pi/3} = \dfrac{1 + i\sqrt{3}}{2}$",
            "Cyan in the hue table. The six conjugate",
            "roots of one degree-6 family sit close",
            "together as a six-fold flower.",
        ),
        3170, 550, "ra",
        leader_length_frac=3 / 4,
    ),
    CalloutRegion(
        "E",
        1580, 706, 240,
        "E Violet halo",
        (
            "Clustering ring near the unit circle.",
            "Degrees 3-8 (blue, magenta, olive,",
            "grey, white) mix to lavender.",
            "From the separate degree plots, this halo",
            "is mainly degrees 5, 6, 7, 8 and 9.",
        ),
        480, 560, "la",
        leader_length_frac=3 / 4,
        marker="square",
    ),
    CalloutRegion(
        "D_prime",
        2136, 1454, 80,
        r"D$^\prime$ Degree-6 rosette",
        (
            r"$z = e^{-i\pi/3} = \dfrac{1 - i\sqrt{3}}{2}$",
            "Cyan in the hue table. The six conjugate",
            "roots of one degree-6 family sit close",
            "together as a six-fold flower.",
        ),
        3170, 1610, "ra_b",
        leader_length_frac=3 / 4,
    ),
)

CALLOUT_BY_ID = {r.id: r for r in CALLOUT_REGIONS}

NUMBER_TAGS: tuple[NumberTag, ...] = (
    NumberTag(2136, 1080, 42, r"$1/2$", 2136, 990),
    NumberTag(1704, 1080, 42, r"$-1/2$", 1704, 1175, "ma"),
    NumberTag(2784, 1080, 48, r"$2$", 2784, 990),
    NumberTag(1056, 1080, 48, r"$-2$", 1056, 990),
    NumberTag(2568, 1080, 36, r"$3/2$", 2568, 990),
    NumberTag(1272, 1080, 36, r"$-3/2$", 1272, 990),
)


@dataclass(frozen=True)
class DegreePlot:
    id: str
    label: str
    show_degrees: tuple[int, ...] = ()
    show_degree_at_least: int | None = None

    def filter_label(self) -> str:
        if self.show_degrees and len(self.show_degrees) == 1:
            return f"degree {self.show_degrees[0]}"
        if self.show_degree_at_least is not None:
            return f"degree >={self.show_degree_at_least}"
        if self.show_degrees:
            return "degree " + ", ".join(str(d) for d in self.show_degrees)
        return "all degrees"


DEGREE_PLOTS: tuple[DegreePlot, ...] = (
    DegreePlot("deg1", "degree 1 (red)", show_degrees=(1,)),
    DegreePlot("deg2", "degree 2 (green)", show_degrees=(2,)),
    DegreePlot("deg3", "degree 3 (blue)", show_degrees=(3,)),
    DegreePlot("deg4", "degree 4 (olive)", show_degrees=(4,)),
    DegreePlot("deg5", "degree 5 (orange)", show_degrees=(5,)),
    DegreePlot("deg6", "degree 6 (cyan)", show_degrees=(6,)),
    DegreePlot("deg7", "degree 7 (magenta)", show_degrees=(7,)),
    DegreePlot("deg8", "degree 8 (grey)", show_degrees=(8,)),
    DegreePlot("deg9_plus", "degree \u2265 9 (white)", show_degree_at_least=9),
)

DEGREE_PLOT_BY_ID = {p.id: p for p in DEGREE_PLOTS}
