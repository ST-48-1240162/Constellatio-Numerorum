#!/usr/bin/env python3
"""Algebraic numbers in the complex plane (discrete scatter, Brooks colours).

Scatter dots coloured by degree. Size from coefficient height, alpha from Brooks
complexity h (or either). Consolidates duplicate roots. Enum: coefficient box or
Brooks bit encoding (--enum box|brooks).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.render import DEGREE_COLORS  # noqa: E402

def degree_color(deg: int) -> tuple[float, float, float]:
    if 1 <= deg < len(DEGREE_COLORS):
        c = DEGREE_COLORS[deg]
        return float(c[0]), float(c[1]), float(c[2])
    return 1.0, 1.0, 1.0


def brooks_h_from_high(high: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(high) + 1.0, axis=-1)


def coeff_matrix_for_degree(deg: int, coeff_range: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All polys of fixed degree. Returns (N, deg+1) low-to-high, height, brooks h."""
    values = np.arange(-coeff_range, coeff_range + 1, dtype=np.int32)
    leading = values[values != 0]
    if deg == 0:
        empty = np.empty(0, dtype=np.float64)
        return np.empty((0, 1), dtype=np.float64), empty, empty
    grids = np.meshgrid(*([values] * deg), indexing="ij")
    rest = np.stack([g.ravel() for g in grids], axis=1)
    n_rest = rest.shape[0]
    lead = np.repeat(leading, n_rest)
    tail = np.tile(rest, (len(leading), 1))
    high = np.column_stack([lead, tail]).astype(np.float64)
    low = high[:, ::-1].copy()
    return low, np.max(np.abs(high), axis=1), brooks_h_from_high(high)


def filter_roots(
    ev: np.ndarray,
    degree: int,
    heights: np.ndarray,
    brooks_hs: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    margin: float,
):
    z = ev.reshape(-1)
    ok = np.isfinite(z.real) & np.isfinite(z.imag)
    ok &= (z.real >= xlim[0] - margin) & (z.real <= xlim[1] + margin)
    ok &= (z.imag >= ylim[0] - margin) & (z.imag <= ylim[1] + margin)
    z = z[ok]
    if z.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int32)
        return z, empty_i, empty_f, empty_f
    deg = np.full(z.size, degree, dtype=np.int32)
    h_row = np.repeat(heights, degree)[ok]
    b_row = np.repeat(brooks_hs, degree)[ok]
    return z, deg, h_row, b_row


def _append_roots(z, deg, height, brooks, roots_all, deg_all, h_all, b_all):
    if z.size:
        roots_all.append(z)
        deg_all.append(deg)
        h_all.append(height)
        b_all.append(brooks)


def collect_roots_gpu_box(
    max_degree: int,
    coeff_range: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    margin: float,
    device=None,
    batch: int = 8192,
):
    from common.gpu_roots import pick_device, roots_batched

    device = device or pick_device()
    roots_all, deg_all, h_all, b_all = [], [], [], []
    for degree in range(1, max_degree + 1):
        coeff, heights, brooks = coeff_matrix_for_degree(degree, coeff_range)
        if len(coeff) == 0:
            continue
        print(f"  box degree={degree}  n={len(coeff)}  on {device}", file=sys.stderr)
        ev = roots_batched(coeff, device, batch)
        z, deg, hh, bh = filter_roots(ev, degree, heights, brooks, xlim, ylim, margin)
        _append_roots(z, deg, hh, bh, roots_all, deg_all, h_all, b_all)
    return _concat_roots(roots_all, deg_all, h_all, b_all)


def collect_roots_gpu_brooks(
    maxh: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    margin: float,
    device=None,
    batch: int = 8192,
):
    from algebraics.algebraics_gpu import collect_by_degree
    from common.gpu_roots import pick_device, roots_batched

    device = device or pick_device()
    roots_all, deg_all, h_all, b_all = [], [], [], []
    by_deg = collect_by_degree(maxh)
    for degree in sorted(by_deg):
        coeff, brooks = by_deg[degree]
        heights = np.max(np.abs(coeff), axis=1)
        print(f"  brooks degree={degree}  n={len(coeff)}  on {device}", file=sys.stderr)
        ev = roots_batched(coeff, device, batch)
        z, deg, hh, bh = filter_roots(ev, degree, heights, brooks.astype(np.float64), xlim, ylim, margin)
        _append_roots(z, deg, hh, bh, roots_all, deg_all, h_all, b_all)
    return _concat_roots(roots_all, deg_all, h_all, b_all)


def _concat_roots(roots_all, deg_all, h_all, b_all):
    if not roots_all:
        empty = np.array([])
        return empty, np.array([], dtype=np.int32), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return (
        np.concatenate(roots_all),
        np.concatenate(deg_all),
        np.concatenate(h_all),
        np.concatenate(b_all),
    )


def enumerate_roots_box(max_degree: int, coeff_range: int):
    values = range(-coeff_range, coeff_range + 1)
    leading = [c for c in values if c != 0]
    for deg in range(1, max_degree + 1):
        for lead in leading:
            for rest in itertools.product(values, repeat=deg):
                high = np.array((lead,) + rest, dtype=np.float64)
                height = float(np.max(np.abs(high)))
                bh = float(brooks_h_from_high(high))
                try:
                    roots = np.roots(high)
                except np.linalg.LinAlgError:
                    continue
                roots = roots[np.isfinite(roots.real) & np.isfinite(roots.imag)]
                for r in roots:
                    yield r, deg, height, bh


def collect_roots_cpu_box(
    max_degree: int,
    coeff_range: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    margin: float = 0.5,
):
    roots, degrees, heights, brooks = [], [], [], []
    for r, deg, height, bh in enumerate_roots_box(max_degree, coeff_range):
        if xlim[0] - margin <= r.real <= xlim[1] + margin and ylim[0] - margin <= r.imag <= ylim[1] + margin:
            roots.append(r)
            degrees.append(deg)
            heights.append(height)
            brooks.append(bh)
    return (
        np.asarray(roots),
        np.asarray(degrees, dtype=np.int32),
        np.asarray(heights, dtype=float),
        np.asarray(brooks, dtype=float),
    )


def consolidate_roots(
    z: np.ndarray,
    deg: np.ndarray,
    height: np.ndarray,
    brooks: np.ndarray,
    ndigits: int = 5,
):
    """One blob per (x, y, degree): min height, min Brooks h, hit count."""
    n = len(z)
    if n == 0:
        empty = np.array([])
        return empty, np.array([], dtype=np.int32), empty, empty, np.array([], dtype=np.int32)
    order = np.lexsort((deg, np.round(z.imag, ndigits), np.round(z.real, ndigits)))
    z = z[order]
    deg = deg[order]
    height = height[order]
    brooks = brooks[order]
    xr = np.round(z.real, ndigits)
    yi = np.round(z.imag, ndigits)
    change = np.ones(n, dtype=bool)
    change[1:] = (xr[1:] != xr[:-1]) | (yi[1:] != yi[:-1]) | (deg[1:] != deg[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    out_z = z[starts]
    out_deg = deg[starts]
    out_height = np.minimum.reduceat(height, starts)
    out_brooks = np.minimum.reduceat(brooks, starts)
    counts = (ends - starts).astype(np.int32)
    return out_z, out_deg, out_height, out_brooks, counts


def cache_path(enum: str, tag: str) -> Path:
    return Path(__file__).resolve().parent / "cache" / f"plane_{enum}_{tag}.npz"


def load_or_collect(
    enum: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    gpu: bool,
    device,
    batch: int,
    max_degree: int,
    coeff_range: int,
    maxh: int,
    cache: Path | None,
    consolidate: bool,
):
    if cache and cache.exists():
        d = np.load(cache)
        print(f"loaded cache {cache}", file=sys.stderr)
        z = d["z"]
        deg = d["deg"]
        height = d["height"]
        brooks = d["brooks"]
        counts = d["counts"]
        return z, deg, height, brooks, counts

    margin = 0.5
    if enum == "brooks":
        if gpu:
            z, deg, height, brooks = collect_roots_gpu_brooks(
                maxh, xlim, ylim, margin, device=device, batch=batch
            )
        else:
            raise SystemExit("brooks enum needs --gpu (use box for CPU)")
    else:
        if gpu:
            z, deg, height, brooks = collect_roots_gpu_box(
                max_degree, coeff_range, xlim, ylim, margin, device=device, batch=batch
            )
        else:
            z, deg, height, brooks = collect_roots_cpu_box(max_degree, coeff_range, xlim, ylim)

    print(f"raw roots={len(z):,}", file=sys.stderr)
    if consolidate:
        z, deg, height, brooks, counts = consolidate_roots(z, deg, height, brooks)
        print(f"unique blobs={len(z):,}", file=sys.stderr)
    else:
        counts = np.ones(len(z), dtype=np.int32)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, z=z, deg=deg, height=height, brooks=brooks, counts=counts)
        print(f"cache {cache}", file=sys.stderr)
    return z, deg, height, brooks, counts


def scatter_metrics(
    height: np.ndarray,
    brooks: np.ndarray,
    counts: np.ndarray,
    weight: str,
    k2: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    height = np.maximum(height, 1.0)
    brooks = np.maximum(brooks, 2.0)
    hit = np.clip(np.sqrt(counts.astype(np.float64)), 1.0, 6.0)
    blob_h = np.clip(40.0 * np.power(k2, brooks - 3.0), 0.5, 40.0)
    blob_coeff = np.clip(60.0 / height, 1.0, 40.0)

    if weight == "height":
        size = blob_coeff * hit
        alpha = np.clip(1.2 / height, 0.05, 0.9) * np.clip(np.log1p(counts) / 3.0, 0.35, 1.0)
    elif weight == "h":
        size = blob_h * hit
        alpha = np.clip(0.85 * np.power(k2, brooks - 3.0), 0.05, 0.9) * np.clip(np.log1p(counts) / 3.0, 0.35, 1.0)
    else:
        size = blob_coeff * hit
        alpha = np.clip(0.85 * np.power(k2, brooks - 3.0), 0.05, 0.9) * np.clip(np.log1p(counts) / 3.0, 0.35, 1.0)
    return np.clip(size, 0.8, 45.0), np.clip(alpha, 0.05, 0.95)


def render_scatter(
    roots: np.ndarray,
    degrees: np.ndarray,
    heights: np.ndarray,
    brooks: np.ndarray,
    counts: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    width: int,
    height: int,
    dpi: int,
    weight: str,
    k2: float,
) -> plt.Figure:
    inches_w = width / dpi
    inches_h = height / dpi
    fig, ax = plt.subplots(figsize=(inches_w, inches_h), facecolor="black", dpi=dpi)
    ax.set_facecolor("black")

    for deg in sorted(set(int(d) for d in degrees), reverse=True):
        mask = degrees == deg
        if not mask.any():
            continue
        size, alpha = scatter_metrics(heights[mask], brooks[mask], counts[mask], weight, k2)
        color = degree_color(int(deg))
        rgba = np.empty((mask.sum(), 4), dtype=np.float32)
        rgba[:, :3] = color
        rgba[:, 3] = alpha
        ax.scatter(
            roots[mask].real,
            roots[mask].imag,
            s=size,
            c=rgba,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--enum", choices=("box", "brooks"), default="box")
    p.add_argument("--weight", choices=("height", "h", "both"), default="both")
    p.add_argument("--max-degree", type=int, default=5, help="box: highest degree")
    p.add_argument("--coeff-range", type=int, default=3, help="box: |coeff| <= L")
    p.add_argument("--maxh", type=int, default=12, help="brooks: complexity cap")
    p.add_argument("--xlim", type=float, nargs=2, default=(-2.2, 2.2))
    p.add_argument("--ylim", type=float, nargs=2, default=(-2.2, 2.2))
    p.add_argument("--width", type=int, default=3840)
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="PNG height in px (default: width * y-range / x-range)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--k2", type=float, default=0.5, help="Brooks blob decay for alpha/size")
    p.add_argument("--png", type=Path, default=Path(__file__).with_name("algebraic_plane_3840px.png"))
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-consolidate", action="store_true")
    p.add_argument("--cache", type=Path, help="npz cache path (default under algebraics/cache/)")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args(argv)

    xlim = tuple(args.xlim)
    ylim = tuple(args.ylim)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    if args.height is not None:
        height = args.height
    elif abs(yspan - xspan) < 1e-9:
        height = int(round(args.width * 9 / 16))
    else:
        height = int(round(args.width * yspan / xspan))
    if args.enum == "box":
        print(
            f"box enum: degree 1..{args.max_degree}, coeffs in [-{args.coeff_range}, {args.coeff_range}]",
            file=sys.stderr,
        )
        tag = f"d{args.max_degree}_L{args.coeff_range}"
    else:
        print(f"brooks enum: maxh={args.maxh}", file=sys.stderr)
        tag = f"h{args.maxh}"

    cache = args.cache
    if cache is None:
        cache = cache_path(args.enum, tag)
    if args.recompute and cache.exists():
        cache.unlink()

    device = None
    if args.gpu:
        import torch

        device = torch.device("cpu") if args.cpu else None

    z, deg, heights, brooks, counts = load_or_collect(
        args.enum,
        xlim,
        ylim,
        args.gpu,
        device,
        args.batch,
        args.max_degree,
        args.coeff_range,
        args.maxh,
        cache,
        not args.no_consolidate,
    )
    for d in sorted(set(int(x) for x in deg)):
        print(f"  degree {d}: {int((deg == d).sum()):,} blobs", file=sys.stderr)

    fig = render_scatter(
        z, deg, heights, brooks, counts, xlim, ylim, args.width, height, args.dpi, args.weight, args.k2
    )
    fig.savefig(args.png, facecolor="black", dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {args.png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
