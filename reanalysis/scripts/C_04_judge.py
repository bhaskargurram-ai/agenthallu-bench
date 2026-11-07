"""C_04: LLM-judge (GPT-4o-mini, the single judge with highest human agreement in the paper, kappa=0.567 vs A1)
for (a) the original interceptor CSV answers (control + non-abstained interceptor answers; answers truncated
to 200 chars in the CSV) and (b) the full-trace re-runs. Prompt copied from experiments/run_ensemble_judge.py.
Cache: C_interceptor/judge_cache.jsonl keyed by sha1(query|gt|answer[:300]).
"""
import csv, hashlib, json, os, sys, glob, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import read, tf, REPO, OUT
from C_02_rerun_traces import load_key

JUDGE_PROMPT = """You are an impartial judge evaluating whether an agent's final answer is correct.

QUESTION: {query}
REFERENCE ANSWER: {reference}
AGENT ANSWER: {candidate}

Respond with a single token: YES if the agent answer substantively matches the reference,
NO otherwise. Accept minor wording differences, units expressed differently, or extra
explanation as long as the core fact is correct.

YOUR VERDICT (YES or NO):"""
CACHE = f"{OUT}/judge_cache.jsonl"


def key(q, gt, a):
    return hashlib.sha1(f"{q[:400]}|{str(gt)[:300]}|{str(a)[:300]}".encode()).hexdigest()


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            d = json.loads(l); c[d["k"]] = d["v"]
    return c


def main():
    import openai
    client = openai.OpenAI(api_key=load_key())
    tasks = {t["task_id"]: t for t in json.load(open(f"{REPO}/data/tasks/full_2000_tasks.json"))}
    tasks.update({t["task_id"]: t for t in json.load(open(f"{REPO}/data/tasks/ecommerce_300_tasks.json"))})
    items = []
    for fn in ["interceptor_control_600.csv", "interceptor_heldout_600.csv"]:
        for r in read(fn):
            if tf(r["abstained"]):
                continue
            t = tasks[r["task_id"]]
            items.append((t["query"], t["correct_final_answer"], r["final_answer"]))
    for p in glob.glob(f"{OUT}/traces/*.jsonl"):
        for l in open(p):
            d = json.loads(l)
            items.append((d["query"], d["gt"], d["final_answer"]))
    cache = load_cache()
    todo = {}
    for q, gt, a in items:
        k = key(q, gt, a)
        if k not in cache:
            todo[k] = (q, gt, a)
    print("items", len(items), "to judge", len(todo))
    lock = threading.Lock(); spend = [0.0]
    f = open(CACHE, "a")

    def judge(k, q, gt, a):
        prompt = JUDGE_PROMPT.format(query=q[:400], reference=str(gt)[:300], candidate=str(a)[:300])
        for att in range(4):
            try:
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                                                   max_tokens=5, temperature=0.0)
                break
            except Exception:
                if att == 3: return
                time.sleep(2 ** att)
        v = (r.choices[0].message.content or "").strip().upper().startswith("YES")
        c = r.usage.prompt_tokens * 0.15e-6 + r.usage.completion_tokens * 0.6e-6
        with lock:
            spend[0] += c
            f.write(json.dumps({"k": k, "v": v}) + "\n"); f.flush()
    with ThreadPoolExecutor(8) as ex:
        for k, (q, gt, a) in todo.items():
            ex.submit(judge, k, q, gt, a)
    print(f"judge spend ${spend[0]:.4f}")


if __name__ == "__main__":
    main()
