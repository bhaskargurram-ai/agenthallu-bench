"""Small stats helpers (no scipy dependency)."""
import math

def z_for(conf):
    return {0.90: 1.6448536269514722, 0.95: 1.959963984540054}[conf]

def wilson(k, n, conf=0.95):
    if n == 0: return (float("nan"), float("nan"))
    z = z_for(conf); p = k / n
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))

def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test (sum of probs <= observed)."""
    if n == 0: return 1.0
    pr = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pr[k]
    return min(1.0, sum(x for x in pr if x <= obs * (1 + 1e-9)))

def mcnemar_exact(b, c):
    return binom_two_sided(min(b, c), b + c)

def newcombe_paired(a, b, c, d, conf=0.95):
    """Newcombe (1998) method 10 CI for p1-p2 with paired data.
    a: both events, b: event in arm1 only, c: event in arm2 only, d: neither. diff = (b-c)/n."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, conf); l2, u2 = wilson(a + c, n, conf)
    D = p1 - p2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = num / math.sqrt(den)
    L = D - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    U = D + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return D, max(-1.0, L), min(1.0, U)

def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def hp(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    obs = hp(a); lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(hp(x) for x in range(lo, hi + 1) if hp(x) <= obs * (1 + 1e-9)))

def cohen_kappa(x, y):
    n = len(x); cats = sorted(set(x) | set(y))
    po = sum(1 for i, j in zip(x, y) if i == j) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den
