"""Batched float64 companion-matrix eigenvalues on CUDA or CPU."""

from __future__ import annotations

import sys

import numpy as np
import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        return torch.device("cuda")
    print("CUDA not available, using CPU float64", file=sys.stderr)
    return torch.device("cpu")


def companion_eigvals(coeffs: torch.Tensor) -> torch.Tensor:
    """coeffs (N, d+1) low-to-high, leading last. Returns (N, d) complex128."""
    n, width = coeffs.shape
    degree = width - 1
    lead = coeffs[:, -1]
    if degree == 1:
        return (-coeffs[:, 0] / lead).unsqueeze(1).to(torch.complex128)
    monic_tail = -coeffs[:, :-1].flip(dims=(1,)) / lead.unsqueeze(1)
    comp = torch.zeros(n, degree, degree, device=coeffs.device, dtype=torch.float64)
    comp[:, 0, :] = monic_tail
    if degree > 1:
        idx = torch.arange(degree - 1, device=coeffs.device)
        comp[:, idx + 1, idx] = 1.0
    return torch.linalg.eigvals(comp)


def roots_batched(coeff_np: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    chunks = []
    for i in range(0, len(coeff_np), batch):
        t = torch.as_tensor(coeff_np[i : i + batch], device=device, dtype=torch.float64)
        ev = companion_eigvals(t)
        chunks.append(ev.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)
