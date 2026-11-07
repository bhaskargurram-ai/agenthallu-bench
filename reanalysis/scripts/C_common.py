"""Shared helpers for C_* interceptor analyses."""
import csv, os, sys, math, random
csv.field_size_limit(sys.maxsize)
REPO = "/Users/challagullayoshitha/Downloads/agentprop_repo/src"
RES = os.path.join(REPO, "data", "results")
OUT = "/Users/challagullayoshitha/Downloads/agentprop_revision/C_interceptor"


def read(fn):
    with open(os.path.join(RES, fn), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tf(v):
    s = str(v).strip().lower()
    return s in ("true", "1", "1.0", "yes")


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value (sum of probs <= observed)."""
    from math import comb
    pk = comb(n, k) * p ** k * (1 - p) ** (n - k)
    tot = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * p ** i * (1 - p) ** (n - i)
        if pi <= pk * (1 + 1e-9):
            tot += pi
    return min(1.0, tot)


def mcnemar_exact(b, c):
    """b = pairs (ctrl bad, treat good), c = (ctrl good, treat bad). Exact two-sided."""
    n = b + c
    if n == 0:
        return 1.0
    return binom_two_sided(min(b, c), n, 0.5)


def hypergeom_sf(x, N, K, n):
    """P(X >= x) for X ~ Hypergeom(N population, K successes, n draws)."""
    from math import comb
    den = comb(N, n)
    s = 0
    for i in range(x, min(K, n) + 1):
        s += comb(K, i) * comb(N - K, n - i)
    return s / den


def boot_ci(vals_fn, n, B=2000, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(vals_fn(idx))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
