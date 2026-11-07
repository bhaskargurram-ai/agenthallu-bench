"""Second independent LLM judge (default gpt-4o, temp 0) over the same traces judged by the first judge.
Usage: OPENAI_API_KEY=... python3 -I D_second_judge.py <p2_all_rejudged.csv> <labeling_dir> [model]"""
import csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import openai
csv.field_size_limit(10**9)
src, d = sys.argv[1], sys.argv[2]
JUDGE = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o"
P = {x["session_id"]: x for x in csv.DictReader(open(src))}
ids = [json.loads(l)["session_id"] for l in open(os.path.join(d, "llm_judge_cache.jsonl"))]
PROMPT = open(os.path.join(d, "llm_judge_prompt.txt")).read()
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=4)
out_p = os.path.join(d, f"llm_judge2_{JUDGE}_cache.jsonl")
cache = {}
if os.path.exists(out_p):
    for l in open(out_p):
        j = json.loads(l); cache[j["session_id"]] = j
def judge(sid):
    if sid in cache: return cache[sid]
    x = P[sid]; mtc = 0 if not x["tool_selected"].strip() else 1
    r = client.chat.completions.create(model=JUDGE, temperature=0, max_tokens=120, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(query=x["query"], made_tool_call=mtc, final_answer=x["final_answer"][:3000])}])
    try: j = json.loads(r.choices[0].message.content)
    except Exception: j = {"label": "PARSE_ERROR"}
    return {"session_id": sid, "judge_model": JUDGE, **j}
with ThreadPoolExecutor(8) as ex: res = list(ex.map(judge, ids))
with open(out_p, "w") as f:
    for j in res: f.write(json.dumps(j) + "\n")
import collections; print(collections.Counter(j.get("label") for j in res))
