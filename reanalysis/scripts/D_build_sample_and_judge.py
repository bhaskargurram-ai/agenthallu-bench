"""Build stratified blind labeling sample and run the provisional LLM judge.
Usage: OPENAI_API_KEY=... python3 -I D_build_sample_and_judge.py <p2_all_rejudged.csv> <outdir>
Judges ALL no-tool-call traces (734) + every sampled row; results cached in <outdir>/llm_judge_cache.jsonl."""
import csv, json, os, random, sys, math, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D_fab_rule import classify
csv.field_size_limit(10**9)
src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(src)))
for x in rows:
    x["made_tool_call"] = 0 if not x["tool_selected"].strip() else 1
    c = classify(bool(x["made_tool_call"]), x["final_answer"])
    x["rule_flag"] = int(c["fabricated"]); x["rule_reason"] = c["reason"]
    x["stratum"] = "A_flagged" if x["rule_flag"] else ("B_notool_unflagged" if x["made_tool_call"] == 0 else "C_toolcall")

# ---- stratified sample (seeded) ----
TARGET = {"A_flagged": 65, "B_notool_unflagged": 70, "C_toolcall": 15}
rng = random.Random(20261030)
sample = []
for st, n in TARGET.items():
    pool = collections.defaultdict(list)
    for x in rows:
        if x["stratum"] == st: pool[x["model"]].append(x)
    N = sum(len(v) for v in pool.values())
    # proportional allocation with a floor of min(2, available) per model
    alloc = {m: min(len(v), max(min(2, len(v)), round(n * len(v) / N))) for m, v in pool.items()}
    while sum(alloc.values()) > n:
        m = max(alloc, key=lambda k: alloc[k] - n * len(pool[k]) / N); alloc[m] -= 1
    while sum(alloc.values()) < n:
        m = max((k for k in alloc if alloc[k] < len(pool[k])), key=lambda k: n * len(pool[k]) / N - alloc[k]); alloc[m] += 1
    for m, k in alloc.items():
        sample += rng.sample(pool[m], k)
rng.shuffle(sample)
for i, x in enumerate(sample):
    x["item_id"] = f"F{i+1:03d}"

# ---- LLM judge ----
import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=4)
PROMPT = open(os.path.join(outdir, "llm_judge_prompt.txt")).read()
JUDGE = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
cache_p = os.path.join(outdir, "llm_judge_cache.jsonl")
cache = {}
if os.path.exists(cache_p):
    for l in open(cache_p):
        d = json.loads(l); cache[d["session_id"]] = d

def judge(x):
    if x["session_id"] in cache: return cache[x["session_id"]]
    p = PROMPT.format(query=x["query"], made_tool_call=x["made_tool_call"], final_answer=x["final_answer"][:3000])
    r = client.chat.completions.create(model=JUDGE, temperature=0, max_tokens=120,
                                       response_format={"type": "json_object"},
                                       messages=[{"role": "user", "content": p}])
    try:
        j = json.loads(r.choices[0].message.content)
    except Exception:
        j = {"label": "PARSE_ERROR"}
    return {"session_id": x["session_id"], "judge_model": JUDGE, **j}

todo = {x["session_id"]: x for x in rows if x["made_tool_call"] == 0}
for x in sample: todo[x["session_id"]] = x
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(judge, todo.values()))
with open(cache_p, "w") as f:
    for d in res: f.write(json.dumps(d) + "\n")
J = {d["session_id"]: d for d in res}

# ---- write sheets ----
with open(os.path.join(outdir, "fabrication_labeling_sheet_BLIND.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "model", "query", "made_tool_call", "final_answer", "LABEL", "NOTE"])
    for x in sample: w.writerow([x["item_id"], x["model"], x["query"], x["made_tool_call"], x["final_answer"], "", ""])
strata_N = collections.Counter((x["stratum"], x["model"]) for x in rows)
strata_n = collections.Counter((x["stratum"], x["model"]) for x in sample)
with open(os.path.join(outdir, "fabrication_labeling_KEY.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "session_id", "model", "stratum", "cell_N", "cell_n", "weight", "rule_flag", "rule_reason",
                "llm_label", "llm_category", "llm_confidence"])
    for x in sample:
        k = (x["stratum"], x["model"]); d = J[x["session_id"]]
        w.writerow([x["item_id"], x["session_id"], x["model"], x["stratum"], strata_N[k], strata_n[k],
                    round(strata_N[k] / strata_n[k], 4), x["rule_flag"], x["rule_reason"],
                    d.get("label"), d.get("category"), d.get("confidence")])
# full-population per-trace table (rule + LLM) for rate estimation
with open(os.path.join(outdir, "..", "per_trace_fabrication.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["session_id", "model", "domain", "made_tool_call", "rule_flag", "rule_reason", "llm_label", "llm_category"])
    for x in rows:
        d = J.get(x["session_id"], {"label": "NOT" if x["made_tool_call"] else "", "category": "TOOL_CALLED" if x["made_tool_call"] else ""})
        w.writerow([x["session_id"], x["model"], x["domain"], x["made_tool_call"], x["rule_flag"], x["rule_reason"], d.get("label"), d.get("category")])
print("sample", len(sample), collections.Counter(x["stratum"] for x in sample))
print("models in sample", collections.Counter(x["model"] for x in sample))
print("judge labels", collections.Counter(d.get("label") for d in res))
