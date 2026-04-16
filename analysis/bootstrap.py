"""Bootstrap utility — compute mean with 95% bootstrap CI for any metric.

Used across Day 8 analyses to report robust estimates instead of point values.
"""

from typing import Callable

import numpy as np


def bootstrap_ci(
    data: list | np.ndarray,
    metric_fn: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute (point estimate, CI_low, CI_high) for metric_fn applied to data.

    Args:
        data: array-like of samples
        metric_fn: function to apply to bootstrap resamples (default: mean)
        n_bootstrap: number of bootstrap iterations
        ci: confidence level (default 0.95 → 95% CI)
        seed: RNG seed

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    arr = np.asarray(data)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    point = float(metric_fn(arr))

    boots = []
    for _ in range(n_bootstrap):
        resample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(metric_fn(resample)))

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return point, lo, hi


def format_ci(point: float, lo: float, hi: float, precision: int = 3) -> str:
    """Format as 'X.XXX [lo, hi]'."""
    return f"{point:.{precision}f} [{lo:.{precision}f}, {hi:.{precision}f}]"


def bootstrap_diff(
    group_a: list | np.ndarray,
    group_b: list | np.ndarray,
    metric_fn: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for difference metric_fn(B) - metric_fn(A).

    Useful for deltas (e.g., interceptor ON vs OFF).
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    point = float(metric_fn(b) - metric_fn(a))

    diffs = []
    for _ in range(n_bootstrap):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(metric_fn(rb) - metric_fn(ra)))

    alpha = (1 - ci) / 2
    lo = float(np.percentile(diffs, 100 * alpha))
    hi = float(np.percentile(diffs, 100 * (1 - alpha)))
    return point, lo, hi


if __name__ == "__main__":
    # Quick self-test
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 100)
    p, lo, hi = bootstrap_ci(x)
    print(f"Mean CI: {format_ci(p, lo, hi)}")

    y = rng.normal(5.5, 1.0, 100)
    pd, ld, hd = bootstrap_diff(x, y)
    print(f"Diff CI: {format_ci(pd, ld, hd)}")
