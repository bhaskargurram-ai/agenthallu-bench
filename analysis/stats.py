"""Statistical tests for AgentHallu-Bench P2 results.

Tests:
  kruskal_error_type: Kruskal-Wallis H across 4 error types
  mannwhitney_tiers: Mann-Whitney U frontier vs efficient vs open
  domain_comparison: Kruskal-Wallis across 3 domains
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from rich.console import Console
from rich.table import Table

console = Console()


def kruskal_error_type(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis H test: are EPS distributions different across 4 error types?"""
    groups = []
    labels = []
    for et in ["type_mismatch", "out_of_range", "missing_required", "semantic_wrong"]:
        subset = df[df["error_type"] == et]["eps"].dropna()
        if len(subset) > 0:
            groups.append(subset.values)
            labels.append(et)

    if len(groups) < 2:
        return {"test": "kruskal_error_type", "statistic": 0, "p_value": 1.0,
                "significant": False, "groups": labels, "n_per_group": [len(g) for g in groups]}

    stat, p = scipy_stats.kruskal(*groups)

    # Effect size: eta-squared = (H - k + 1) / (N - k)
    N = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (stat - k + 1) / (N - k) if N > k else 0

    return {
        "test": "kruskal_error_type",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
        "effect_size_eta_sq": round(float(eta_sq), 4),
        "groups": labels,
        "n_per_group": [len(g) for g in groups],
        "group_means": {l: round(float(g.mean()), 3) for l, g in zip(labels, groups)},
    }


def mannwhitney_tiers(df: pd.DataFrame) -> list[dict]:
    """Mann-Whitney U tests between model tiers."""
    from config import MODELS

    results = []

    tier_map = {}
    for mk, info in MODELS.items():
        tier_map[mk] = info.get("tier", "unknown")

    df_with_tier = df.copy()
    df_with_tier["tier"] = df_with_tier["model"].map(tier_map)

    tier_pairs = [("frontier", "efficient"), ("frontier", "open"), ("efficient", "open")]

    for t1, t2 in tier_pairs:
        g1 = df_with_tier[df_with_tier["tier"] == t1]["eps"].dropna()
        g2 = df_with_tier[df_with_tier["tier"] == t2]["eps"].dropna()

        if len(g1) < 2 or len(g2) < 2:
            results.append({
                "test": f"mannwhitney_{t1}_vs_{t2}",
                "statistic": 0, "p_value": 1.0, "significant": False,
                "n1": len(g1), "n2": len(g2),
            })
            continue

        stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")

        # Effect size: rank-biserial correlation
        n1, n2 = len(g1), len(g2)
        r = 1 - (2 * stat) / (n1 * n2) if n1 * n2 > 0 else 0

        results.append({
            "test": f"mannwhitney_{t1}_vs_{t2}",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "effect_size_r": round(float(r), 4),
            "n1": n1, "n2": n2,
            "mean1": round(float(g1.mean()), 3),
            "mean2": round(float(g2.mean()), 3),
        })

    return results


def domain_comparison(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis across domains."""
    groups = []
    labels = []
    for domain in ["weather", "calendar", "medical"]:
        subset = df[df["domain"] == domain]["eps"].dropna()
        if len(subset) > 0:
            groups.append(subset.values)
            labels.append(domain)

    if len(groups) < 2:
        return {"test": "domain_comparison", "statistic": 0, "p_value": 1.0, "significant": False}

    stat, p = scipy_stats.kruskal(*groups)
    N = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (stat - k + 1) / (N - k) if N > k else 0

    return {
        "test": "domain_comparison",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
        "effect_size_eta_sq": round(float(eta_sq), 4),
        "groups": labels,
        "n_per_group": [len(g) for g in groups],
        "group_means": {l: round(float(g.mean()), 3) for l, g in zip(labels, groups)},
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical tests for P2 results")
    parser.add_argument("--input", type=str, default="data/results/p2_results.csv")
    parser.add_argument("--tests", type=str, default="kruskal_error_type,mannwhitney_tiers,domain_comparison")
    parser.add_argument("--output", type=str, default="data/results/stats_p2.json")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    df = pd.read_csv(os.path.join(base, args.input))
    console.print(f"Loaded {len(df)} rows for statistical analysis")

    tests_to_run = [t.strip() for t in args.tests.split(",")]
    all_results = []

    if "kruskal_error_type" in tests_to_run:
        r = kruskal_error_type(df)
        all_results.append(r)

    if "mannwhitney_tiers" in tests_to_run:
        tier_results = mannwhitney_tiers(df)
        all_results.extend(tier_results)

    if "domain_comparison" in tests_to_run:
        r = domain_comparison(df)
        all_results.append(r)

    # Print results table
    table = Table(title="Statistical Tests — P2 Results")
    table.add_column("Test")
    table.add_column("Statistic", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Significant?", justify="center")
    table.add_column("Effect Size", justify="right")

    for r in all_results:
        sig = "[green]YES[/green]" if r.get("significant") else "[red]NO[/red]"
        es = r.get("effect_size_eta_sq") or r.get("effect_size_r") or "—"
        if isinstance(es, float):
            es = f"{es:.4f}"
        table.add_row(
            r["test"],
            f"{r['statistic']:.4f}",
            f"{r['p_value']:.6f}",
            sig,
            str(es),
        )

    console.print(table)

    # Save
    output_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"[green]Stats saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
