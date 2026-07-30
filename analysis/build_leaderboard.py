"""Build the AgentProp-Bench leaderboard CSV from the released result files.

Reads the canonical 13-model rejudged results plus the fabrication and
tool-call-rate summaries, and writes a single tidy leaderboard.csv that both the
HF Space (leaderboard/app.py) and the `agentprop leaderboard` command render.

Usage:
  ./venv/bin/python analysis/build_leaderboard.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "data", "results")
OUT = os.path.join(ROOT, "leaderboard", "leaderboard.csv")

# Display name + provenance for each model key.
META = {
    "gpt4o":            ("GPT-4o",              "proprietary"),
    "gpt4o_mini":       ("GPT-4o-mini",         "proprietary"),
    "gpt41_mini":       ("GPT-4.1-mini",        "proprietary"),
    "gpt41_nano":       ("GPT-4.1-nano",        "proprietary"),
    "gpt35_turbo":      ("GPT-3.5-turbo",       "proprietary"),
    "o3_mini":          ("o3-mini",             "proprietary"),
    "gemini_20_flash":  ("Gemini-2.0-Flash",    "proprietary"),
    "gemini_25_flash":  ("Gemini-2.5-Flash",    "proprietary"),
    "deepseek_v3":      ("DeepSeek-V3",         "proprietary"),
    "qwen25_3b":        ("Qwen2.5-3B",          "open-weight"),
    "qwen25_7b":        ("Qwen2.5-7B",          "open-weight"),
    "qwen25_14b":       ("Qwen2.5-14B",         "open-weight"),
    "hermes3_llama8b":  ("Hermes-3-Llama-3.1-8B", "open-weight"),
}


def main():
    df = pd.read_csv(os.path.join(RESULTS, "p2_all_rejudged.csv"))
    fab = json.load(open(os.path.join(RESULTS, "fabricated_tooluse.json")))
    tcr = json.load(open(os.path.join(RESULTS, "tool_call_rates.json")))

    agg = df.groupby("model").agg(
        n=("task_id", "count"),
        eps=("eps", "mean"),
        weps=("weps", "mean"),
        reached_output=("reached_output", "mean"),
        acc_ensemble=("ensemble_correct", "mean"),
        acc_substring=("heuristic_correct", "mean"),
    ).round(3)

    rows = []
    for key, (name, kind) in META.items():
        if key not in agg.index:
            continue
        r = agg.loc[key]
        rows.append({
            "model": name,
            "kind": kind,
            "n": int(r["n"]),
            "propagation_eps": round(float(r["eps"]), 3),
            "reached_output_rate": round(float(r["reached_output"]), 3),
            "fabrication_pct": fab.get(key, ""),
            "tool_call_pct": (tcr.get(key) or {}).get("tool_call_rate_pct", ""),
            "acc_validated_judge": round(float(r["acc_ensemble"]), 3),
            "acc_substring_judge": round(float(r["acc_substring"]), 3),
        })

    out = pd.DataFrame(rows).sort_values("propagation_eps", ascending=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
