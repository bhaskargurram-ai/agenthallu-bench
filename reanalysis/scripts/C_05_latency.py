"""C_05: Interceptor CPU overhead on REAL traces vs measured agent LLM latency.

Times the exact checks of experiments/run_interceptor.py (L1 per tool step, L2 per LLM message, L3 on the
final answer) on every GPT-4o-mini trace re-run in C_02, and compares with the wall-clock LLM latency
recorded in the same traces. Also re-runs the authors' synthetic benchmark logic for reference.
"""
import json, os, re, sys, time, statistics, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import OUT

KW = {"error", "invalid", "unknown", "failed", "not found", "missing", "incorrect", "wrong"}


def l1(step):
    pe = step.get("validation_errors") or []
    if pe: return True
    tr = step.get("tool_result")
    return isinstance(tr, dict) and "error" in tr


def l2(text):
    lower = text.lower()
    return sum(1 for k in KW if k in lower) >= 2


def l3(final, obs):
    if not final or not obs: return False
    a = set(re.findall(r"-?\d+\.?\d*", final)); o = set()
    for x in obs: o.update(re.findall(r"-?\d+\.?\d*", str(x)))
    return bool({n for n in a - o if len(n) > 1 or int(float(n)) > 9})


def monitor(tr):
    obs = []; n1 = 0
    for s in tr["steps"]:
        l2(s["llm_text"])
        if "tool_name" in s:
            obs.append(s["obs"])
            if l1(s): n1 += 1
    return n1 >= 2 or l3(tr["final_answer"], obs)


def main():
    traces = []
    for p in sorted(glob.glob(f"{OUT}/traces/*.jsonl")):
        traces += [json.loads(l) for l in open(p)]
    per = []
    for tr in traces:
        for _ in range(50): monitor(tr)  # warmup
        t0 = time.perf_counter()
        for _ in range(500): monitor(tr)
        per.append((time.perf_counter() - t0) / 500 * 1e6)
    llm = [sum(t["llm_ms"]) * 1000 for t in traces]  # microseconds
    ratio = [per[i] / llm[i] for i in range(len(traces)) if llm[i] > 0]
    res = {"n_traces": len(traces),
           "monitor_us_per_trace": {"median": statistics.median(per), "mean": statistics.mean(per),
                                    "p95": sorted(per)[int(0.95 * len(per))], "max": max(per)},
           "agent_llm_ms_per_trace": {"median": statistics.median(llm) / 1000, "mean": statistics.mean(llm) / 1000,
                                      "p5": sorted(llm)[int(0.05 * len(llm))] / 1000},
           "agent_llm_calls_per_trace_mean": statistics.mean(len(t["llm_ms"]) for t in traces),
           "overhead_fraction_of_llm_latency": {"median": statistics.median(ratio), "max": max(ratio)},
           "agent_cost_usd_per_trace_mean": statistics.mean(t["cost"] for t in traces),
           "extra_llm_calls": 0, "extra_tokens": 0,
           "note": "single process, Python " + sys.version.split()[0] + "; LLM latency = wall-clock of OpenAI chat calls in C_02"}
    json.dump(res, open(f"{OUT}/latency_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
