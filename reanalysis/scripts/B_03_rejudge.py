"""Critique 3: re-judge the 100 calibration traces with alternative prompts and judges.

Inputs to every judge are identical to the original ensemble run
(experiments/rejudge_p2_ensemble.py): query[:400], reference[:300], answer[:300],
taken from data/results/p2_all_rejudged.csv (11 rows have an empty query there,
exactly as the original judges and the human annotators saw them).

Output: B_judge/B03_rejudge_raw.jsonl (appended incrementally; re-runs skip done keys).
Usage: python3 -I B_03_rejudge.py [--judges a,b] [--prompts P0,P1] [--budget 4.0]
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from B_common import *  # noqa
from B_openai_key import get_key

RAW = os.path.join(OUT, "B03_rejudge_raw.jsonl")

PROMPTS = {
    # P0: verbatim original ensemble prompt
    "P0_original": """You are an impartial judge evaluating whether an agent's final answer is correct.

QUESTION: {query}
REFERENCE ANSWER: {reference}
AGENT ANSWER: {candidate}

Respond with a single token: YES if the agent answer substantively matches the reference,
NO otherwise. Accept minor wording differences, units expressed differently, or extra
explanation as long as the core fact is correct.

YOUR VERDICT (YES or NO):""",
    # P1: explicit rubric = the written human-annotator criterion
    "P1_rubric": """You are grading a tool-using AI agent. Decide whether the agent's FINAL ANSWER is correct.

User request: {query}
Reference (expected outcome): {reference}
Agent final answer: {candidate}

Rules:
- YES = the answer substantively matches the reference / accomplishes the request. Accept minor wording
  differences, different phrasing or units, or extra explanation, as long as the core fact is correct.
- NO = the core fact is wrong, missing, or unsupported: e.g. wrong city, wrong date, wrong patient or ID,
  wrong value, a refusal or error message, a promise to act later, or a made-up result.
Judge only correctness of the core fact, not style or length.

Answer with exactly one word: YES or NO.""",
    # P2: request-centric framing (reference shown last, as a hint only)
    "P2_request_centric": """Did the assistant correctly complete the user's request?

[User request]
{query}

[Assistant's final response]
{candidate}

[Description of the expected outcome]
{reference}

Say YES only if the response delivers what was requested for the right entities (same city, date,
patient, event, drugs, etc.) and does not report failure, refuse, or ask to try again. Otherwise say NO.
Output YES or NO and nothing else.""",
    # P3: brief reasoning then verdict
    "P3_reason_then_verdict": """You are an impartial judge evaluating whether an agent's final answer is correct.

QUESTION: {query}
REFERENCE ANSWER: {reference}
AGENT ANSWER: {candidate}

First, in at most two short sentences, state whether the agent's answer addresses the right entities
(e.g. city, date, patient, event, drug) and whether it actually provides the requested result.
Then on a new line write exactly "VERDICT: YES" if the agent answer substantively matches the reference
(minor wording differences, units, or extra explanation are fine as long as the core fact is correct),
or "VERDICT: NO" otherwise.""",
}

PRICE = {  # USD per 1M tokens (input, output) — public list prices
    "gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0), "gpt-5-mini": (0.25, 2.0), "gpt-4.1-nano": (0.10, 0.40),
}

lock = threading.Lock()
spent = [0.0]


def parse(text, prompt_id):
    t = (text or "").strip().upper()
    if prompt_id.startswith("P3"):
        m = re.findall(r"VERDICT:\s*(YES|NO)", t)
        if m:
            return 1 if m[-1] == "YES" else 0
        return None
    m = re.match(r"\W*(YES|NO)\b", t)
    if m:
        return 1 if m.group(1) == "YES" else 0
    if "YES" in t and "NO" not in t:
        return 1
    if "NO" in t and "YES" not in t:
        return 0
    return None


def call(client, model, prompt, prompt_id):
    kw = dict(model=model, messages=[{"role": "user", "content": prompt}])
    if model.startswith("gpt-5"):
        kw["reasoning_effort"] = "minimal"
        kw["max_completion_tokens"] = 400 if prompt_id.startswith("P3") else 200
    else:
        kw["temperature"] = 0
        kw["max_tokens"] = 160 if prompt_id.startswith("P3") else 5
    last = None
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kw)
            u = r.usage
            pi, po = PRICE[model]
            cost = (u.prompt_tokens * pi + u.completion_tokens * po) / 1e6
            return r.choices[0].message.content, cost, u.prompt_tokens, u.completion_tokens
        except Exception as e:  # retry with exponential backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after retries: {type(last).__name__}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="gpt-4o-mini,gpt-4.1-mini,gpt-4.1,gpt-5-mini")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=get_key(), timeout=60, max_retries=0)
    p2 = load_p2()
    ids = sample_ids()

    done = set()
    if os.path.exists(RAW):
        with open(RAW) as f:
            for line in f:
                d = json.loads(line)
                if d.get("verdict") is not None or d.get("error") is None:
                    done.add((d["judge"], d["prompt"], d["session_id"]))
                spent[0] += d.get("cost", 0.0)
    print(f"already done: {len(done)} calls, prior spend ${spent[0]:.4f}")

    jobs = []
    for j in args.judges.split(","):
        for pid in args.prompts.split(","):
            for s in ids:
                if (j, pid, s) not in done:
                    jobs.append((j, pid, s))
    print(f"jobs: {len(jobs)}")

    def work(job):
        j, pid, s = job
        r = p2[s]
        prompt = PROMPTS[pid].format(query=str(r["query"])[:400], reference=str(r["ground_truth"])[:300],
                                     candidate=str(r["final_answer"])[:300])
        with lock:
            if spent[0] > args.budget:
                return None
        try:
            text, cost, ti, to = call(client, j, prompt, pid)
            rec = {"judge": j, "prompt": pid, "session_id": s, "raw": text, "verdict": parse(text, pid),
                   "cost": cost, "tok_in": ti, "tok_out": to}
        except Exception as e:
            rec = {"judge": j, "prompt": pid, "session_id": s, "raw": None, "verdict": None,
                   "cost": 0.0, "error": str(e)[:300]}
        with lock:
            spent[0] += rec["cost"]
            with open(RAW, "a") as f:  # synchronous incremental write
                f.write(json.dumps(rec) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, jb) for jb in jobs]):
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} spent=${spent[0]:.4f}", flush=True)
    print(f"total spend ${spent[0]:.4f}")


if __name__ == "__main__":
    main()
