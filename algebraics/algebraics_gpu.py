"""GPU / batched roots for the algebraic-numbers plot."""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algebraics.algebraics import (  # noqa: E402
    cache_path,
    consolidate,
    enumerate_abs_coeffs,
    render,
)
from algebraics.root_labels import apply_root_labels  # noqa: E402
from common.gpu_roots import pick_device, roots_batched  # noqa: E402
from common.render import save_png  # noqa: E402


def signed_matrix(abs_t: np.ndarray) -> np.ndarray:
    """All sign patterns as rows. Leading coeff stays positive."""
    k = len(abs_t) - 1
    free = np.flatnonzero(abs_t[:k])
    npat = 1 << free.size
    out = np.broadcast_to(abs_t.astype(np.float64), (npat, k + 1)).copy()
    if free.size:
        order = free[::-1]
        bits = np.arange(npat, dtype=np.int64)[:, None]
        pos = (bits & (1 << np.arange(free.size))) != 0
        out[:, order] *= np.where(pos, 1.0, -1.0)
    return out


def collect_by_degree(maxh: int, progress: bool = True) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """degree -> (coeff matrix low-to-high, h per row)."""
    coeffs: dict[int, list[np.ndarray]] = defaultdict(list)
    hs: dict[int, list[int]] = defaultdict(list)
    last_h = 0
    for h, degree, abs_t in enumerate_abs_coeffs(maxh):
        if progress and h != last_h:
            last_h = h
            n = sum(len(v) for v in hs.values())
            print(f"  collect h={h}/{maxh}  polys so far={n}", file=sys.stderr)
        block = signed_matrix(abs_t)
        coeffs[degree].append(block)
        hs[degree].extend([h] * len(block))
    out = {}
    for degree, parts in coeffs.items():
        out[degree] = (np.vstack(parts), np.asarray(hs[degree], dtype=np.int32))
    return out


def precalc_gpu(maxh: int, device: torch.device | None = None, batch: int = 4096, progress: bool = True) -> np.ndarray:
    device = device or pick_device()
    t0 = time.time()
    by_deg = collect_by_degree(maxh, progress=progress)
    xs, ys, hs, os_ = [], [], [], []
    eqns = 0
    for degree in sorted(by_deg):
        coeff_np, h_arr = by_deg[degree]
        eqns += len(coeff_np)
        if progress:
            print(f"  roots degree={degree}  n={len(coeff_np)}  on {device}", file=sys.stderr)
        ev = roots_batched(coeff_np, device, batch)
        finite = np.isfinite(ev.real) & np.isfinite(ev.imag)
        ev = np.where(finite, ev, np.nan)
        for j in range(degree):
            col = ev[:, j]
            ok = np.isfinite(col.real) & np.isfinite(col.imag)
            xs.append(col.real[ok])
            ys.append(col.imag[ok])
            hs.append(h_arr[ok])
            os_.append(np.full(int(ok.sum()), degree, dtype=np.int32))
    points = np.zeros(
        sum(a.size for a in xs),
        dtype=[("x", "f8"), ("y", "f8"), ("h", "i4"), ("o", "i4")],
    )
    if len(points):
        points["x"] = np.concatenate(xs)
        points["y"] = np.concatenate(ys)
        points["h"] = np.concatenate(hs)
        points["o"] = np.concatenate(os_)
    if progress:
        print(f"eqns={eqns} roots={len(points)}  ({time.time() - t0:.1f}s)  {device}", file=sys.stderr)
    return points


def load_or_compute_gpu(maxh: int, device: torch.device | None = None, batch: int = 4096) -> np.ndarray:
    path = cache_path(maxh)
    if path.exists():
        return np.load(path)["points"]
    points = precalc_gpu(maxh, device=device, batch=batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points)
    return points


def render_png(
    points: np.ndarray,
    out,
    width: int = 1920,
    height: int | None = None,
    ox: float | None = None,
    oy: float | None = None,
    zoom: float | None = None,
    k1: float = 0.125,
    k2: float = 0.5,
    labels: bool = False,
):
    points, weights = consolidate(points)
    if height is None:
        height = int(round(width * 9 / 16))
    if ox is None:
        ox = 0.0
    if oy is None:
        oy = 0.0
    if zoom is None:
        zoom = height / 5.0
    img = render(points, width, height, ox, oy, zoom, k1, k2, weights=weights)
    if labels:
        img = apply_root_labels(img, points, ox, oy, zoom, weights)
    save_png(img, Path(out))
    return img


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--maxh", type=int, default=16)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--png", default="algebraics_gpu.png")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int)
    p.add_argument("--ox", type=float)
    p.add_argument("--oy", type=float)
    p.add_argument("--zoom", type=float)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--labels", dest="labels", action="store_true", default=True,
                   help="name high-hit roots on the PNG (default)")
    p.add_argument("--no-labels", dest="labels", action="store_false",
                   help="splat only")
    args = p.parse_args()
    device = torch.device("cpu") if args.cpu else pick_device()
    pts = load_or_compute_gpu(args.maxh, device=device, batch=args.batch)
    render_png(
        pts,
        args.png,
        width=args.width,
        height=args.height,
        ox=args.ox,
        oy=args.oy,
        zoom=args.zoom,
        labels=args.labels,
    )
    print(f"wrote {args.png}")
