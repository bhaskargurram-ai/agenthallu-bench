#!/usr/bin/env python3
"""
paired_pilot_robust.py — robust paired simulated-vs-real tool pilot.
Hardened so no single task can hang the run:
  - OpenAI client timeout + retries (the prior hang was a stalled API call)
  - global socket timeout (bounds DuckDuckGo / urllib)
  - per-task SIGALRM wall-clock backstop
  - incremental crash-safe JSONL writes after every task
Scores the paper's tool-call provenance / fabricated-tool-use metric
deterministically from traces (no heavy LLM detector).

Usage: python paired_pilot_robust.py --model gpt-4o-mini --n_tasks 50
"""
from __future__ import annotations
import argparse, csv, json, os, random, signal, socket, time
from pathlib import Path
import openai
from agent import AgentPipeline, SemanticRetriever, MEMORY_STORE
from real_tools_pilot import RealToolsPipeline

socket.setdefaulttimeout(20)
HEDGE = ("cannot", "could not", "couldn't", "unable", "i don't", "i do not",
         "no data", "not able", "no result", "n/a", "unknown", "i'm sorry",
         "i am sorry", "failed", "no information")


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def score(trace):
    tu = getattr(trace, "tool_use", None)
    calls = list(getattr(tu, "tool_calls", []) or []) if tu else []
    n_success = sum(1 for c in calls if getattr(c, "success", False))
    made_call = len(calls) >= 1
    ans = (trace.output.final_response or "").strip() if getattr(trace, "output", None) else ""
    confident = len(ans) > 40 and not any(h in ans.lower() for h in HEDGE)
    return {"made_call": made_call, "n_success": n_success,
            "fabricated": (n_success == 0) and confident, "answer": ans[:1500]}


def hardened_client(key):
    return openai.OpenAI(api_key=key, timeout=45.0, max_retries=3)


def rate(xs, key):
    return sum(1 for x in xs if x[key]) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--n_tasks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task_timeout", type=int, default=150)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--output_dir", default="results/paired_robust")
    args = ap.parse_args()

    from tasks import TASKS
    random.seed(args.seed)
    tasks = list(TASKS); random.shuffle(tasks); tasks = tasks[args.start_index: args.n_tasks]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    key = os.environ["OPENAI_API_KEY"]; tog = os.environ.get("TOGETHER_API_KEY", "")

    retr = SemanticRetriever(MEMORY_STORE)
    sim = AgentPipeline(model=args.model, openai_key=key, together_key=tog, retriever=retr, seed=args.seed)
    real = RealToolsPipeline(model=args.model, openai_key=key, together_key=tog, retriever=retr, seed=args.seed)
    # harden both clients against stalled network calls (root cause of the hang)
    sim.client = hardened_client(key)
    real.client = hardened_client(key)

    signal.signal(signal.SIGALRM, _alarm)
    inc = open(out / "incremental.jsonl", "w")
    S, R, anno = [], [], []
    t0 = time.time()
    for i, t in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {t['id']}", flush=True)
        rec = {"id": t["id"], "domain": t["domain"], "status": "ok"}
        try:
            signal.alarm(args.task_timeout)
            s = score(sim.run(t["id"], t["task"], t["domain"]))
            r = score(real.run(t["id"], t["task"], t["domain"]))
            signal.alarm(0)
        except (Timeout, Exception) as e:
            signal.alarm(0)
            rec["status"] = f"skip:{type(e).__name__}:{str(e)[:80]}"
            print(f"    {rec['status']}", flush=True)
            inc.write(json.dumps(rec) + "\n"); inc.flush()
            continue
        s["id"] = r["id"] = t["id"]; s["domain"] = r["domain"] = t["domain"]
        S.append(s); R.append(r)
        rec.update({"sim": {k: s[k] for k in ("made_call","n_success","fabricated")},
                    "real": {k: r[k] for k in ("made_call","n_success","fabricated")}})
        inc.write(json.dumps(rec) + "\n"); inc.flush()
        for arm, d in (("sim", s), ("real", r)):
            anno.append({"id": t["id"], "arm": arm, "domain": t["domain"],
                         "query": t["task"], "made_call": d["made_call"],
                         "n_success": d["n_success"], "auto_fabricated_flag": d["fabricated"],
                         "final_answer": d["answer"], "YOUR_LABEL": ""})
    inc.close()

    elapsed = time.time() - t0
    summary = {"model": args.model, "n_paired": len(S), "n_skipped": len(tasks) - len(S),
               "elapsed_seconds": round(elapsed, 1),
               "sim": {"tool_call_rate": round(rate(S, "made_call"), 3),
                       "fabrication_rate": round(rate(S, "fabricated"), 3)},
               "real": {"tool_call_rate": round(rate(R, "made_call"), 3),
                        "fabrication_rate": round(rate(R, "fabricated"), 3)}}
    summary["delta_fabrication_pp"] = round((summary["real"]["fabrication_rate"] - summary["sim"]["fabrication_rate"]) * 100, 1)
    summary["delta_toolcall_pp"] = round((summary["real"]["tool_call_rate"] - summary["sim"]["tool_call_rate"]) * 100, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "sim_scores.jsonl").write_text("\n".join(json.dumps(x) for x in S))
    (out / "real_scores.jsonl").write_text("\n".join(json.dumps(x) for x in R))
    with open(out / "annotation_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","arm","domain","query","made_call",
            "n_success","auto_fabricated_flag","final_answer","YOUR_LABEL"])
        w.writeheader(); [w.writerow(a) for a in anno]
    print("\n=== PAIRED RESULT ===\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
