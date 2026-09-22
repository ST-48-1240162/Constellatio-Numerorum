#!/usr/bin/env python3
"""Export single-colour full-map PNG/PDF per degree (default 7680×4320)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algebraics.algebraics import consolidate, load_or_compute, render  # noqa: E402
from algebraics.regions.defs import (  # noqa: E402
    DEGREE_PLOT_BY_ID,
    DEGREE_PLOTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DegreePlot,
    default_zoom,
)
from common.render import save_png  # noqa: E402

REGIONS_DIR = Path(__file__).resolve().parent


def filter_by_degrees(
    points: np.ndarray,
    weights: np.ndarray,
    plot: DegreePlot,
) -> tuple[np.ndarray, np.ndarray]:
    if not plot.show_degrees and plot.show_degree_at_least is None:
        return points, weights
    mask = np.zeros(len(points), dtype=bool)
    if plot.show_degrees:
        mask |= np.isin(points["o"], list(plot.show_degrees))
    if plot.show_degree_at_least is not None:
        mask |= points["o"] >= plot.show_degree_at_least
    return points[mask], weights[mask]


def write_summary(
    path: Path,
    plot: DegreePlot,
    width: int,
    height: int,
    maxh: int,
    points: np.ndarray,
    weights: np.ndarray,
) -> None:
    zoom = default_zoom(height)
    lines = [
        f"plot: {plot.id}",
        f"label: {plot.label}",
        f"filter: {plot.filter_label()}",
        f"maxh: {maxh}",
        f"output: {width}x{height}",
        f"viewport: centre (0, 0), zoom {zoom:.3f}",
        f"blobs: {len(points)}",
        f"weight sum: {float(weights.sum()):.1f}",
        "",
        "sample coordinates (x, y, h, degree, weight):",
    ]
    order = np.argsort(-weights)[:12]
    for i in order:
        p = points[i]
        lines.append(
            f"  ({p['x']:.8f}, {p['y']:.8f})  h={int(p['h'])}  deg={int(p['o'])}  w={weights[i]:.1f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, png_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                r"\documentclass[border=0pt]{standalone}",
                r"\usepackage{graphicx}",
                r"\begin{document}",
                rf"\includegraphics[width=16in]{{{png_name}}}",
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_degree_plot(
    plot: DegreePlot,
    all_points: np.ndarray,
    all_weights: np.ndarray,
    width: int,
    height: int,
    maxh: int,
    make_pdf: bool = True,
    out_root: Path | None = None,
) -> Path:
    out_dir = (out_root or REGIONS_DIR) / plot.id
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_points, plot_weights = filter_by_degrees(all_points, all_weights, plot)
    np.savez_compressed(
        out_dir / "points.npz",
        points=plot_points,
        weights=plot_weights,
        maxh=np.array([maxh]),
    )
    write_summary(out_dir / "summary.txt", plot, width, height, maxh, plot_points, plot_weights)

    ox, oy = 0.0, 0.0
    zoom = default_zoom(height)
    img = render(plot_points, width, height, ox, oy, zoom, weights=plot_weights)
    png_path = out_dir / f"{plot.id}.png"
    save_png(img, png_path)

    write_tex(out_dir / f"{plot.id}.tex", png_path.name)
    if make_pdf:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"{plot.id}.tex"],
            cwd=out_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print(
        f"{plot.id}: {len(plot_points)} blobs, {plot.filter_label()}, zoom={zoom:.1f} -> {png_path.name}"
        + (f", {plot.id}.pdf" if make_pdf else "")
    )
    return out_dir


def build_degree_plots(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    maxh: int = 17,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    plot_ids: list[str] | None = None,
    make_pdf: bool = True,
    out_root: Path | None = None,
) -> list[Path]:
    """Build single-colour full-map exports for each degree (or selected subset)."""
    if weights is None:
        points, weights = consolidate(points)
    if plot_ids:
        plots = [DEGREE_PLOT_BY_ID[pid] for pid in plot_ids]
    else:
        plots = list(DEGREE_PLOTS)
    return [
        build_degree_plot(
            plot, points, weights, width, height, maxh, make_pdf=make_pdf, out_root=out_root
        )
        for plot in plots
    ]


# Colab / legacy name
build_regions = build_degree_plots


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maxh", type=int, default=17)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument(
        "--plots",
        nargs="*",
        metavar="ID",
        help="subset of deg1 … deg8 deg9_plus (default: all)",
    )
    p.add_argument("--no-pdf", action="store_true", help="skip pdflatex (PNG/NPZ only)")
    args = p.parse_args(argv)

    if args.plots:
        for pid in args.plots:
            if pid not in DEGREE_PLOT_BY_ID:
                p.error(f"unknown plot {pid!r}; choose from {', '.join(DEGREE_PLOT_BY_ID)}")

    print(f"loading points h={args.maxh}", file=sys.stderr)
    raw = load_or_compute(args.maxh)
    points, weights = consolidate(raw)
    print(f"unique blobs={len(points)}", file=sys.stderr)

    build_degree_plots(
        points,
        weights,
        maxh=args.maxh,
        width=args.width,
        height=args.height,
        plot_ids=args.plots,
        make_pdf=not args.no_pdf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
