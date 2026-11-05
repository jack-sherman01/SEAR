"""Small dense QP solver for the safety filter (no external dependency).

Solves   min_u (u - u_ref)^T H (u - u_ref)   s.t.  A u >= b
with Hildreth's dual coordinate-ascent method, which is exact for strictly
convex QPs and cheap for the 2-6 constraints of a per-step safety filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPResult:
    u: np.ndarray
    feasible: bool
    max_violation: float
    iterations: int


def solve_min_norm_qp(u_ref: np.ndarray, H: np.ndarray, A: np.ndarray, b: np.ndarray,
                      max_iter: int = 500, tol: float = 1e-9, feas_tol: float = 1e-6) -> QPResult:
    u_ref = np.asarray(u_ref, float)
    A = np.atleast_2d(np.asarray(A, float))
    b = np.asarray(b, float).ravel()
    if A.size == 0:
        return QPResult(u_ref.copy(), True, 0.0, 0)

    row_norm = np.linalg.norm(A, axis=1)
    degenerate = row_norm < 1e-12
    if np.any(degenerate & (b > feas_tol)):  # 0 >= b with b > 0: infeasible regardless of u
        return QPResult(u_ref.copy(), False, float(b[degenerate].max()), 0)
    A, b = A[~degenerate], b[~degenerate]

    # Dual of: min 1/2 u'Hu + f'u  s.t. M u <= g   (M = -A, g = -b, f = -H u_ref)
    Hinv = np.linalg.inv(H)
    M, g = -A, -b
    P = M @ Hinv @ M.T
    d = g - M @ u_ref
    lam = np.zeros(len(b))
    it = 0
    for it in range(1, max_iter + 1):
        prev = lam.copy()
        for i in range(len(lam)):
            w = -(d[i] + P[i] @ lam - P[i, i] * lam[i]) / P[i, i]
            lam[i] = max(0.0, w)
        if np.max(np.abs(lam - prev)) < tol * (1.0 + np.max(np.abs(lam))):
            break
    u = u_ref - Hinv @ M.T @ lam
    viol = float(np.max(np.maximum(0.0, b - A @ u))) if len(b) else 0.0
    scale = 1.0 + float(np.max(np.abs(b))) if len(b) else 1.0
    return QPResult(u, viol <= feas_tol * scale, viol, it)
