#!/usr/bin/env python3
"""Random stable IIR pole cloud via lattice reflection coefficients.

|k_m| < 1 is Schur, so the step-up polynomial is minimum-phase and every
pole sits inside the unit circle. Color is filter order. Brightness is a
per-pole Q from the 3 dB bandwidth of that resonance (narrower peak,
brighter blob). High-order i.i.d. Verblunsky/reflection draws pile up
near |z|=1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.render import color_for_degree, save_png, splat  # noqa: E402


def reflection_to_direct(k: np.ndarray) -> np.ndarray:
    """Levinson step-up: reflection k[0..M-1] -> A(z) = 1 + a1 z^{-1} + ..."""
    a = np.ones(1, dtype=np.float64)
    for km in k:
        a = np.concatenate([a, [0.0]]) + km * np.concatenate([[0.0], a[::-1]])
    return a


def pole_q(z: np.ndarray) -> np.ndarray:
    """Q ~ 0.5 r / (1-r). Digital 3 dB bandwidth of a pole is ~ (1-r)."""
    r = np.clip(np.abs(z), 0.0, 0.999999)
    return 0.5 * r / (1.0 - r)


def n_for_order(order: int, orders: list[int], n: int) -> int:
    lo, hi = orders[0], orders[-1]
    t = 0.0 if hi == lo else (order - lo) / (hi - lo)
    return max(40, int(n * (0.35 + 0.65 * t)))


def step_up_batch(K: np.ndarray) -> np.ndarray:
    """K is (n, M) reflection rows. Returns (n, M+1) high-to-low A(z) coeffs."""
    n, m = K.shape
    a = np.ones((n, 1), dtype=np.float64)
    for i in range(m):
        z = np.zeros((n, 1), dtype=np.float64)
        a = np.concatenate([a, z], axis=1) + K[:, i : i + 1] * np.concatenate([z, a[:, ::-1]], axis=1)
    return a


def _pack(xs, ys, os_, qs) -> np.ndarray:
    n = sum(a.size for a in xs)
    pts = np.zeros(n, dtype=[("x", "f8"), ("y", "f8"), ("o", "i4"), ("q", "f8")])
    if n:
        pts["x"] = np.concatenate(xs)
        pts["y"] = np.concatenate(ys)
        pts["o"] = np.concatenate(os_)
        pts["q"] = np.concatenate(qs)
    return pts


def sample_poles(orders: list[int], n_each: int, rng: np.random.Generator) -> np.ndarray:
    xs, ys, os_, qs = [], [], [], []
    for order in orders:
        for _ in range(n_each):
            k = rng.uniform(-0.995, 0.995, size=order)
            a = reflection_to_direct(k)
            z = np.roots(a)
            z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
            z = z[np.abs(z) < 1.0]
            if z.size == 0:
                continue
            q = pole_q(z)
            xs.append(z.real)
            ys.append(z.imag)
            os_.append(np.full(z.size, order, dtype=np.int32))
            qs.append(q)
    return _pack(xs, ys, os_, qs)


def sample_poles_batched(orders: list[int], n: int, rng: np.random.Generator, device=None, batch: int = 4096) -> np.ndarray:
    """One reflection matrix per order, then batched companion eigenvalues."""
    from common.gpu_roots import pick_device, roots_batched

    device = device or pick_device()
    xs, ys, os_, qs = [], [], [], []
    for order in orders:
        nf = n_for_order(order, orders, n)
        K = rng.uniform(-0.995, 0.995, size=(nf, order))
        a = step_up_batch(K)
        # algebraics_gpu wants low-to-high, leading last
        ev = roots_batched(a[:, ::-1].copy(), device, batch)
        z = ev.reshape(-1)
        ocol = np.repeat(np.full(nf, order, dtype=np.int32), order)
        ok = np.isfinite(z.real) & np.isfinite(z.imag) & (np.abs(z) < 1.0)
        z, ocol = z[ok], ocol[ok]
        xs.append(z.real)
        ys.append(z.imag)
        os_.append(ocol)
        qs.append(pole_q(z))
        print(f"  order {order:2d}  filters={nf}  poles={z.size}  on {device}", file=sys.stderr)
    return _pack(xs, ys, os_, qs)


def draw_unit_circle(img: np.ndarray, ox: float, oy: float, zoom: float, luma: float = 0.07):
    h, w, _ = img.shape
    ys, xs = np.ogrid[:h, :w]
    wx = (xs - w / 2) / zoom + ox
    wy = (h / 2 - ys) / zoom + oy
    r = np.sqrt(wx * wx + wy * wy)
    ring = np.exp(-((r - 1.0) * zoom) ** 2 / (2 * 0.7**2))
    img += (luma * ring).astype(np.float32)[:, :, None]


def render_poles(
    pts: np.ndarray,
    width: int,
    height: int,
    ox: float,
    oy: float,
    zoom: float,
    k1: float = 0.022,
    k2: float = 0.72,
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.float32)
    draw_unit_circle(img, ox, oy, zoom)
    if pts.size == 0:
        return img

    bright = np.clip(np.log1p(pts["q"]) / 6.5, 0.06, 1.0).astype(np.float32)
    colors = color_for_degree(pts["o"]) * (0.55 * bright)[:, None]
    rad = k1 * np.power(k2, pts["o"].astype(np.float64) - 2.0)
    sx = (pts["x"] - ox) * zoom + width / 2.0
    sy = height / 2.0 - (pts["y"] - oy) * zoom
    splat(img, sx, sy, colors, 16.0 * rad * zoom)
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--orders", default="2-16")
    p.add_argument("--n", type=int, default=350, help="random filters per order")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--png", type=Path, default=Path(__file__).with_name("poles.png"))
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--batch", type=int, default=4096)
    args = p.parse_args(argv)

    if "-" in args.orders:
        a, b = args.orders.split("-", 1)
        orders = list(range(int(a), int(b) + 1))
    else:
        orders = [int(x) for x in args.orders.split(",") if x]

    rng = np.random.default_rng(args.seed)
    print(f"sampling lattice filters for orders {orders[0]}..{orders[-1]}", file=sys.stderr)
    if args.gpu:
        pts = sample_poles_batched(orders, args.n, rng, batch=args.batch)
    else:
        xs = []
        for o in orders:
            nf = n_for_order(o, orders, args.n)
            xs.append(sample_poles([o], nf, rng))
            print(f"  drew {nf} filters of order {o}", file=sys.stderr)
        pts = np.concatenate(xs)
    rim = float(np.mean(np.abs(pts["x"] + 1j * pts["y"]) > 0.9))
    print(f"poles={len(pts)}  fraction |z|>0.9: {rim:.3f}", file=sys.stderr)
    for o in orders:
        sl = pts[pts["o"] == o]
        if sl.size == 0:
            continue
        r = np.abs(sl["x"] + 1j * sl["y"])
        print(f"  order {o:2d}  n={len(sl):5d}  median |z|={np.median(r):.3f}  mean Q={np.mean(sl['q']):.1f}", file=sys.stderr)

    zoom = args.height / 2.45
    img = render_poles(pts, args.width, args.height, 0.0, 0.0, zoom)
    save_png(img, args.png)
    print(f"wrote {args.png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
