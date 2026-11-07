"""Statistical helpers (pure numpy/scipy) for A_* analyses."""
import math
import numpy as np
from scipy import stats

Z = 1.959963984540054

def wilson(k, n, z=Z):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def newcombe_paired(a, b, c, d, z=Z):
    """Newcombe (1998) method 10 hybrid-score CI for a paired difference p1-p2,
    p1=(a+b)/n (row margin: injected wrong), p2=(a+c)/n (col margin: baseline wrong).
    a=both wrong, b=inj wrong & base right, c=inj right & base wrong, d=both right."""
    n = a+b+c+d
    if n == 0: return (None, None, None)
    p1, p2 = (a+b)/n, (a+c)/n; th = p1 - p2
    l1, u1 = wilson(a+b, n, z); l2, u2 = wilson(a+c, n, z)
    num = a*d - b*c
    if num > 0: num = max(num - n/2, 0)
    den = (a+b)*(c+d)*(a+c)*(b+d)
    phi = num/math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1-l1)**2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)**2))
    du = math.sqrt(max(0.0, (u1-p1)**2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)**2))
    return (th, max(-1.0, th-dl), min(1.0, th+du))

def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial on discordant pairs)."""
    m = b + c
    if m == 0: return 1.0
    return float(min(1.0, stats.binomtest(min(b, c), m, 0.5, alternative="two-sided").pvalue))

def mcnemar_chi2_cc(b, c):
    if b + c == 0: return (0.0, 1.0)
    x = (abs(b-c)-1)**2/(b+c)
    return (x, float(stats.chi2.sf(x, 1)))

def boot_ci(values, B=10000, seed=12345, stat=np.mean):
    v = np.asarray(values, dtype=float)
    if len(v) == 0: return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = stat(v[idx], axis=1) if stat is np.mean else np.array([stat(v[i]) for i in idx])
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))

def cluster_boot_rd(pairs, cluster_key, B=10000, seed=12345):
    """pairs: list of (cluster, inj_wrong, base_wrong). Resample clusters (task_ids) with replacement."""
    rng = np.random.default_rng(seed)
    cl = {}
    for k, x, y in pairs: cl.setdefault(k, []).append(x - y)
    keys = list(cl); sums = np.array([sum(cl[k]) for k in keys], float); cnts = np.array([len(cl[k]) for k in keys], float)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))
