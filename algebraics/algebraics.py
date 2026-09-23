#!/usr/bin/env python3
"""Algebraic numbers in the complex plane (Brooks 2010)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.render import color_for_degree, save_png, splat, to_uint8  # noqa: E402
from algebraics.root_labels import apply_root_labels, find_hotspots  # noqa: E402


def enumerate_abs_coeffs(maxh: int):
    for h in range(2, maxh + 1):
        t = np.zeros(h, dtype=np.int32)
        for i in range((1 << (h - 1)) - 1, -1, -2):
            t[0] = 0
            k = 0
            for j in range(h - 2, -1, -1):
                if (i >> j) & 1:
                    t[k] += 1
                else:
                    k += 1
                    t[k] = 0
            if k == 0:
                continue
            yield h, k, t[: k + 1].copy()


def signed_coeffs(abs_t: np.ndarray):
    k = len(abs_t) - 1
    nz = int(np.count_nonzero(abs_t))
    for bits in range((1 << (nz - 1)) - 1, -1, -1):
        c = abs_t.astype(np.float64)
        sp = 1
        for l in range(k, -1, -1):
            if abs_t[l] == 0 or l == k:
                continue
            if not (bits & sp):
                c[l] = -c[l]
            sp <<= 1
        yield c


def precalc(maxh: int, progress: bool = True) -> np.ndarray:
    xs, ys, hs, os_ = [], [], [], []
    temps = eqns = roots = 0
    t0 = time.time()
    last_h = 0
    for h, degree, abs_t in enumerate_abs_coeffs(maxh):
        temps += 1
        if progress and h != last_h:
            last_h = h
            print(f"  complexity h={h}/{maxh}  roots so far={roots}", file=sys.stderr)
        for coeffs in signed_coeffs(abs_t):
            eqns += 1
            try:
                zs = np.roots(coeffs[::-1])
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(zs)):
                continue
            for z in zs:
                xs.append(float(z.real))
                ys.append(float(z.imag))
                hs.append(h)
                os_.append(degree)
                roots += 1
    pts = np.zeros(roots, dtype=[("x", "f8"), ("y", "f8"), ("h", "i4"), ("o", "i4")])
    if roots:
        pts["x"] = xs
        pts["y"] = ys
        pts["h"] = hs
        pts["o"] = os_
    if progress:
        print(f"temps={temps} eqns={eqns} roots={roots}  ({time.time() - t0:.1f}s)", file=sys.stderr)
    return pts


def cache_path(maxh: int) -> Path:
    return Path(__file__).resolve().parent / "cache" / f"points_h{maxh}.npz"


def load_or_compute(maxh: int, progress: bool = True) -> np.ndarray:
    path = cache_path(maxh)
    if path.exists():
        return np.load(path)["points"]
    points = precalc(maxh, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points)
    return points


def consolidate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.empty(points.shape[0], dtype=points.dtype)
    keys["x"] = np.round(points["x"], 8)
    keys["y"] = np.round(points["y"], 8)
    keys["h"] = points["h"]
    keys["o"] = points["o"]
    uniq, counts = np.unique(keys, return_counts=True)
    return uniq, counts.astype(np.float32)


def blob_world_radius(h: np.ndarray, k1: float, k2: float) -> np.ndarray:
    return k1 * np.power(k2, h.astype(np.float64) - 3.0)


def render(
    points: np.ndarray,
    width: int,
    height: int,
    ox: float,
    oy: float,
    zoom: float,
    k1: float = 0.125,
    k2: float = 0.5,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.float32)
    if points.size == 0:
        return img
    if weights is None:
        points, weights = consolidate(points)
    colors = color_for_degree(points["o"]) * weights[:, None]
    rad = blob_world_radius(points["h"], k1, k2)
    sx = (points["x"] - ox) * zoom + width / 2.0
    sy = height / 2.0 - (points["y"] - oy) * zoom
    splat(img, sx, sy, colors, 16.0 * rad * zoom)
    return img


def run_viewer(points: np.ndarray, ox: float, oy: float, zoom: float, k1: float, k2: float, size, weights=None, labels=None):
    import pygame

    width, height = size
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Algebraic numbers [Stephen Brooks 2010]")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    dirty = True
    preview = True
    dragging = False
    drag_start = (0, 0)
    origin_start = (ox, oy)
    show_help = True
    last_change = time.time()

    def paint(scale: float):
        rw, rh = max(1, int(width * scale)), max(1, int(height * scale))
        img = render(points, rw, rh, ox, oy, zoom * scale, k1, k2, weights)
        if labels is not None and scale >= 0.99:
            img = apply_root_labels(img, points, ox, oy, zoom * scale, weights, labels=labels)
        surf = pygame.image.frombuffer(to_uint8(img).tobytes(), (rw, rh), "RGB")
        if (rw, rh) != (width, height):
            surf = pygame.transform.smoothscale(surf, (width, height))
        screen.blit(surf, (0, 0))
        if show_help:
            y = 8
            for line in (
                f"roots={len(points)}  zoom={zoom:.1f}  k1={k1:.4g}  k2={k2:.3f}",
                "drag pan   wheel zoom   O reset   Z/X size   C/V decay",
                "S save PNG   H toggle help   Esc quit",
            ):
                screen.blit(font.render(line, True, (220, 220, 220)), (10, y))
                y += 18
        pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = max(event.w, 320), max(event.h, 240)
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                dirty = preview = True
                last_change = time.time()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                drag_start = event.pos
                origin_start = (ox, oy)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - drag_start[0]
                dy = event.pos[1] - drag_start[1]
                ox = origin_start[0] - dx / zoom
                oy = origin_start[1] + dy / zoom
                dirty = preview = True
                last_change = time.time()
            elif event.type == pygame.MOUSEWHEEL:
                factor = 1.15 if event.y > 0 else 1 / 1.15
                mx, my = pygame.mouse.get_pos()
                wx = ox + (mx - width / 2) / zoom
                wy = oy - (my - height / 2) / zoom
                zoom *= factor
                ox = wx - (mx - width / 2) / zoom
                oy = wy + (my - height / 2) / zoom
                dirty = preview = True
                last_change = time.time()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_o:
                    ox, oy = 0.0, 0.0
                    zoom = height / 5.0
                    dirty = preview = True
                    last_change = time.time()
                elif event.key == pygame.K_z:
                    k1 *= 1.3
                    dirty = preview = True
                    last_change = time.time()
                elif event.key == pygame.K_x:
                    k1 /= 1.3
                    dirty = preview = True
                    last_change = time.time()
                elif event.key == pygame.K_c:
                    k2 += 0.05
                    dirty = preview = True
                    last_change = time.time()
                elif event.key == pygame.K_v:
                    k2 -= 0.05
                    dirty = preview = True
                    last_change = time.time()
                elif event.key == pygame.K_h:
                    show_help = not show_help
                    dirty = True
                elif event.key == pygame.K_s:
                    out = Path("algebraics.png")
                    img = render(points, width, height, ox, oy, zoom, k1, k2, weights)
                    if labels is not None:
                        img = apply_root_labels(img, points, ox, oy, zoom, weights, labels=labels)
                    save_png(img, out)
                    print(f"wrote {out.resolve()}", file=sys.stderr)
        if dirty:
            paint(0.4 if preview else 1.0)
            dirty = False
        elif preview and time.time() - last_change > 0.25:
            preview = False
            dirty = True
        clock.tick(60)
    pygame.quit()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--maxh", type=int, default=12)
    p.add_argument("--png", type=Path)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--ox", type=float, help="view centre x in the complex plane")
    p.add_argument("--oy", type=float, help="view centre y in the complex plane")
    p.add_argument("--zoom", type=float, help="pixels per unit (default height/5)")
    p.add_argument("--k1", type=float, default=0.125)
    p.add_argument("--k2", type=float, default=0.5)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--labels", dest="labels", action="store_true", default=True,
                   help="name high-hit roots on the PNG (default)")
    p.add_argument("--no-labels", dest="labels", action="store_false",
                   help="splat only")
    args = p.parse_args(argv)

    if args.recompute:
        path = cache_path(args.maxh)
        if path.exists():
            path.unlink()
    print(f"enumerating polynomials up to h={args.maxh}", file=sys.stderr)
    points = load_or_compute(args.maxh)
    points, weights = consolidate(points)
    print(f"unique blobs={len(points)}", file=sys.stderr)

    width, height = args.width, args.height
    ox = 0.0 if args.ox is None else args.ox
    oy = 0.0 if args.oy is None else args.oy
    zoom = height / 5.0 if args.zoom is None else args.zoom

    labels = find_hotspots(points, weights) if args.labels else None

    if args.png:
        img = render(points, width, height, ox, oy, zoom, args.k1, args.k2, weights)
        if labels is not None:
            img = apply_root_labels(img, points, ox, oy, zoom, weights, labels=labels)
        save_png(img, args.png)
        print(f"wrote {args.png}", file=sys.stderr)
        return 0
    run_viewer(points, ox, oy, zoom, args.k1, args.k2, (width, height), weights, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
