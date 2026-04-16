"""Day 8 STEP 7: Achieved power analysis per subgroup.

For each claim in the paper, compute achieved statistical power given:
  - Observed effect size (Cohen's h for proportions, d for means)
  - Observed sample size
  - α = 0.05

Reports which subgroup claims are underpowered (<0.8) so we can either:
  (a) Add a caveat in the paper
  (b) Report only marginal effects
  (c) Justify why we don't split further

Output:
  data/results/power_analysis.json — per-claim power
  Prints a table with flags on underpowered claims
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for comparing two proportions."""
    def phi(p):
        p = max(min(p, 0.9999), 0.0001)
        return 2 * math.asin(math.sqrt(p))
    return abs(phi(p1) - phi(p2))


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_inv_cdf(p: float) -> float:
    """Approx. inverse normal CDF (Beasley-Springer)."""
    if p <= 0 or p >= 1:
        return 0.0
    # Use rational approximation
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def power_two_prop(p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """Power for two-proportion z-test (two-sided)."""
    if n1 == 0 or n2 == 0:
        return 0.0
    h = cohens_h(p1, p2)
    # Effective n for unequal groups: harmonic mean
    n_eff = 2 * n1 * n2 / (n1 + n2)
    z_alpha = normal_inv_cdf(1 - alpha / 2)
    z_beta = h * math.sqrt(n_eff / 2) - z_alpha
    return normal_cdf(z_beta)


def analyze_baseline_vs_p2(baseline_csv: str, p2_csv: str) -> list[dict]:
    b = pd.read_csv(baseline_csv)
    p = pd.read_csv(p2_csv)
    claims = []
    for model in sorted(set(b["model"].unique()) & set(p["model"].unique())):
        bm = b[b["model"] == model]
        pm = p[p["model"] == model]
        if len(bm) == 0 or len(pm) == 0:
            continue
        p1 = 1 - bm["final_correct"].astype(int).mean()  # baseline error rate
        p2 = 1 - pm["final_correct"].astype(int).mean()  # injected error rate
        pwr = power_two_prop(p1, p2, len(bm), len(pm))
        claims.append({
            "claim": f"P2 increases error for {model}",
            "group_a_p": round(p1, 3),
            "group_b_p": round(p2, 3),
            "n_a": len(bm),
            "n_b": len(pm),
            "effect_h": round(cohens_h(p1, p2), 3),
            "power": round(pwr, 3),
            "underpowered": pwr < 0.8,
        })
    return claims


def analyze_domain_splits(p2_csv: str) -> list[dict]:
    p = pd.read_csv(p2_csv)
    claims = []
    domains = p["domain"].dropna().unique()
    for d in domains:
        pd_ = p[p["domain"] == d]
        pnd = p[p["domain"] != d]
        if len(pd_) < 10 or len(pnd) < 10:
            continue
        p1 = 1 - pnd["final_correct"].astype(int).mean()
        p2 = 1 - pd_["final_correct"].astype(int).mean()
        pwr = power_two_prop(p1, p2, len(pnd), len(pd_))
        claims.append({
            "claim": f"Domain {d} differs from others (P2)",
            "group_a_p": round(p1, 3),
            "group_b_p": round(p2, 3),
            "n_a": len(pnd),
            "n_b": len(pd_),
            "effect_h": round(cohens_h(p1, p2), 3),
            "power": round(pwr, 3),
            "underpowered": pwr < 0.8,
        })
    return claims


def analyze_interceptor(intercept_csv: str) -> list[dict]:
    if not os.path.exists(intercept_csv):
        return []
    df = pd.read_csv(intercept_csv)
    claims = []
    for model in df["model"].unique():
        dm = df[df["model"] == model]
        if "condition" not in dm.columns:
            continue
        conds = dm["condition"].unique()
        if len(conds) < 2:
            continue
        # output_only vs early_interceptor hallucination rate
        a = dm[dm["condition"] == "output_only"]
        b = dm[dm["condition"] == "early_interceptor"]
        if len(a) == 0 or len(b) == 0:
            continue
        p1 = a["hallucination"].astype(int).mean()
        p2 = b["hallucination"].astype(int).mean()
        pwr = power_two_prop(p1, p2, len(a), len(b))
        claims.append({
            "claim": f"Interceptor reduces halluc for {model}",
            "group_a_p": round(p1, 3),
            "group_b_p": round(p2, 3),
            "n_a": len(a),
            "n_b": len(b),
            "effect_h": round(cohens_h(p1, p2), 3),
            "power": round(pwr, 3),
            "underpowered": pwr < 0.8,
        })
    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/results/baseline_all.csv")
    parser.add_argument("--p2", default="data/results/p2_all.csv")
    parser.add_argument("--interceptor", default="data/results/interceptor_results.csv")
    parser.add_argument("--output", default="data/results/power_analysis.json")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")

    console.rule("[bold cyan]Day 8 STEP 7 — Power Analysis[/bold cyan]")

    all_claims = []
    baseline_path = os.path.join(base, args.baseline)
    p2_path = os.path.join(base, args.p2)
    if os.path.exists(baseline_path) and os.path.exists(p2_path):
        all_claims.extend(analyze_baseline_vs_p2(baseline_path, p2_path))
        all_claims.extend(analyze_domain_splits(p2_path))
    intercept_path = os.path.join(base, args.interceptor)
    all_claims.extend(analyze_interceptor(intercept_path))

    # Table
    tab = Table(title="Achieved Power by Claim (α=0.05)")
    tab.add_column("Claim")
    tab.add_column("n_A", justify="right")
    tab.add_column("n_B", justify="right")
    tab.add_column("p_A", justify="right")
    tab.add_column("p_B", justify="right")
    tab.add_column("h", justify="right")
    tab.add_column("Power", justify="right")
    tab.add_column("Flag")
    under = 0
    for c in all_claims:
        flag = "[red]UNDER[/red]" if c["underpowered"] else "[green]OK[/green]"
        if c["underpowered"]:
            under += 1
        tab.add_row(c["claim"], str(c["n_a"]), str(c["n_b"]),
                    f"{c['group_a_p']:.3f}", f"{c['group_b_p']:.3f}",
                    f"{c['effect_h']:.2f}", f"{c['power']:.2f}", flag)
    console.print(tab)
    console.print(f"\n[bold]{under}/{len(all_claims)} claims underpowered (<0.8)[/bold]")

    out_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "claims": all_claims,
            "summary": {
                "total_claims": len(all_claims),
                "underpowered": under,
                "well_powered": len(all_claims) - under,
            }
        }, f, indent=2)
    console.print(f"[green]Saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
