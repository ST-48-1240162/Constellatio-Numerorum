#!/usr/bin/env python3
"""Full pipeline for Colab CLI: GPU roots → full PNG → regions → constellatio PDF."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/constellatio-numerorum")
TARBALL = Path("/content/cn-src.tgz")
MAXH, WIDTH, HEIGHT, BATCH = 17, 7680, 4320, 8192
REGION_WIDTH, REGION_HEIGHT = WIDTH, HEIGHT


def ensure_repo() -> Path:
    if TARBALL.exists() and not (ROOT / "algebraics/algebraics.py").exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TARBALL) as tf:
            tf.extractall(ROOT)
        print("extracted", TARBALL, "->", ROOT, flush=True)
    if not (ROOT / "algebraics/algebraics.py").exists():
        raise SystemExit(f"repo missing at {ROOT}; upload {TARBALL} first")
    return ROOT


def main() -> int:
    import os

    repo = ensure_repo()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    subprocess.run(["apt-get", "-qq", "update"], check=False)
    subprocess.run(
        ["apt-get", "-qq", "install", "-y", "texlive-latex-base", "texlive-fonts-recommended"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

    import torch
    from algebraics.algebraics import consolidate
    from algebraics.algebraics_gpu import load_or_compute_gpu, pick_device, render_png
    from algebraics.annotate_constellatio import build_pdf
    from algebraics.regions.build import build_degree_plots

    print("cwd", Path.cwd(), flush=True)
    print("cuda", torch.cuda.is_available(), end=" ", flush=True)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), flush=True)
    else:
        print("(no GPU — will be slow)", flush=True)

    device = pick_device()
    points = load_or_compute_gpu(MAXH, device=device, batch=BATCH)
    print("roots", len(points), flush=True)

    out = Path("algebraics") / f"algebraics_h{MAXH}_{WIDTH}px.png"
    render_png(points, out, width=WIDTH, height=HEIGHT)
    print("wrote", out.resolve(), flush=True)

    blobs, weights = consolidate(points)
    build_degree_plots(blobs, weights, maxh=MAXH, width=REGION_WIDTH, height=REGION_HEIGHT)
    print(f"wrote algebraics/regions/deg*  ({REGION_WIDTH}x{REGION_HEIGHT})", flush=True)

    build_pdf(out)
    print("wrote", (Path("algebraics/constellatio.pdf")).resolve(), flush=True)

    zip_path = Path("/content/constellatio_regions.zip")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", "algebraics/regions")
    print("wrote", zip_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
