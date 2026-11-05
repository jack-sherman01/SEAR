"""Statistical bounds.

"A zero observed failure count will not be described as zero risk; we will
report an appropriate statistical upper confidence bound." (proposal Sec. 6.4)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact upper bound on a Bernoulli probability."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(confidence, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - confidence, k, n - k + 1))


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided exact interval."""
    alpha = 1.0 - confidence
    return (clopper_pearson_lower(k, n, 1 - alpha / 2), clopper_pearson_upper(k, n, 1 - alpha / 2))


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def poisson_rate_upper(k: int, exposure: float, confidence: float = 0.95) -> float:
    """Upper bound on an event rate (events per unit exposure)."""
    if exposure <= 0:
        return math.inf
    return float(stats.chi2.ppf(confidence, 2 * (k + 1)) / 2.0 / exposure)


def mean_ci(x, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-based confidence interval (nan-safe)."""
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (math.nan, math.nan, math.nan)
    m = float(a.mean())
    if a.size < 2:
        return (m, math.nan, math.nan)
    h = float(stats.t.ppf(1 - (1 - confidence) / 2, a.size - 1) * a.std(ddof=1) / math.sqrt(a.size))
    return (m, m - h, m + h)
