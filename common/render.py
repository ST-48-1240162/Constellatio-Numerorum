"""Additive Lorentzian blobs. Same look as Brooks' 2010 algebraic-numbers sketch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEGREE_COLORS = np.array(
    [
        [1.00, 1.00, 1.00],
        [1.00, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00],
        [0.70, 0.70, 0.00],
        [1.00, 0.60, 0.00],
        [0.00, 1.00, 1.00],
        [1.00, 0.00, 1.00],
        [0.60, 0.60, 0.60],
    ],
    dtype=np.float32,
)


def color_for_degree(degree: np.ndarray) -> np.ndarray:
    out = np.ones((degree.size, 3), dtype=np.float32)
    known = (degree >= 1) & (degree < len(DEGREE_COLORS))
    out[known] = DEGREE_COLORS[degree[known]]
    return out


def lorentz_texture(size: int = 256) -> np.ndarray:
    ys, xs = np.ogrid[:size, :size]
    cx = cy = size / 2
    f = (size / 2) ** 2 / (1.0 + (xs - cx) ** 2 + (ys - cy) ** 2)
    return np.minimum(255.0, f).astype(np.float32) / 255.0


_TEX = None


def texture() -> np.ndarray:
    global _TEX
    if _TEX is None:
        _TEX = lorentz_texture(256)
    return _TEX


def add_points(img: np.ndarray, sx, sy, colors, weights):
    height, width, _ = img.shape
    flat = img.reshape(-1, 3)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = sx - x0
    fy = sy - y0
    for dx, wx in ((0, 1.0 - fx), (1, fx)):
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            xx = x0 + dx
            yy = y0 + dy
            ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            if not np.any(ok):
                continue
            np.add.at(flat, yy[ok] * width + xx[ok], colors[ok] * (wx[ok] * wy[ok] * weights[ok])[:, None])


def splat(img: np.ndarray, sx, sy, colors, half: np.ndarray):
    """Add Lorentzian blobs. half is the quad radius in pixels."""
    tex = texture()
    tsz = tex.shape[0]
    height, width, _ = img.shape
    tiny = half < 1.25
    if np.any(tiny):
        add_points(img, sx[tiny], sy[tiny], colors[tiny], np.clip(half[tiny], 0.12, None))
    large = np.nonzero(~tiny)[0]
    large = large[np.argsort(half[large])[::-1]]
    for i in large:
        hpix = float(half[i])
        cx, cy = float(sx[i]), float(sy[i])
        x1 = max(0, int(np.floor(cx - hpix)))
        x2 = min(width, int(np.ceil(cx + hpix)) + 1)
        y1 = max(0, int(np.floor(cy - hpix)))
        y2 = min(height, int(np.ceil(cy + hpix)) + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        xs = np.arange(x1, x2, dtype=np.float32)
        ys = np.arange(y1, y2, dtype=np.float32)
        u = (xs - (cx - hpix)) * (tsz / (2.0 * hpix))
        v = (ys - (cy - hpix)) * (tsz / (2.0 * hpix))
        ui = np.clip(u.astype(np.int32), 0, tsz - 1)
        vi = np.clip(v.astype(np.int32), 0, tsz - 1)
        img[y1:y2, x1:x2] += tex[np.ix_(vi, ui)][:, :, None] * colors[i]


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(to_uint8(img), "RGB").save(path)
