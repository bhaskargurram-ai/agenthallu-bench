"""C_02: Re-run GPT-4o-mini on the interceptor task set WITHOUT any interceptor, logging FULL traces
(every thought, action, params, validation errors, tool result, final answer, per-call latency), so the
Interceptor's three layers can be replayed offline at any threshold.

Agent loop, prompt, temperature (0.0), max_tokens (2048), 5-iteration cap and correctness heuristic are
copied from agentprop_repo/src/experiments/run_interceptor.py / run_interceptor_control.py.

Runs (--run):
  clean      : the 600 gpt4o_mini task_ids of interceptor_heldout_600.csv, no injection
  p2sem      : same 600 tasks, P2 semantic_wrong injection on the first tool call (as run_p2_experiment.py)
  retail     : ecommerce_300_tasks.json split=test (270, unseen domain/tools for the interceptor), no injection
  retail_p2sem: same retail tasks with semantic_wrong injection
Output: C_interceptor/traces/<run>_<tag>.jsonl
"""
import argparse, csv, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import C_tools as T

csv.field_size_limit(sys.maxsize)
REPO = "/Users/challagullayoshitha/Downloads/agentprop_repo/src"
OUT = "/Users/challagullayoshitha/Downloads/agentprop_revision/C_interceptor/traces"
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6
BUDGET = 3.0


def load_key():
    with open("/Users/challagullayoshitha/Downloads/caliche-landman-ai/.env") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key")


def heuristic_correct(gt, final_answer):
    return bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                        any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))


class Spend:
    def __init__(self): self.v = 0.0; self.l = threading.Lock()
    def add(self, c):
        with self.l:
            self.v += c
            if self.v > BUDGET: raise RuntimeError("budget exceeded")


def run_task(client, task, inject, spend, show_schema=False):
    domain = task["domain"]
    tools = T.list_tools(domain)
    if show_schema:  # fixed harness: expose parameter JSON schemas to the agent
        tool_desc = "\n".join(f"- {t['name']}: {t['description']} Parameters (JSON schema): {json.dumps(t['parameters'])}" for t in tools)
    else:  # original AgentProp harness: names + descriptions only
        tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    system_msg = ("You are an agent that answers questions using tools.\n"
                  "Follow this format exactly:\n"
                  "Thought: [reasoning]\nAction: [tool_name]\nAction Input: {\"param\": \"value\"}\n\n"
                  "When done, respond with:\nFinal Answer: [your answer]\n\n"
                  f"Available tools:\n{tool_desc}")
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": task["query"]}]
    steps, final_answer, cost, injection_done, injected = [], "", 0.0, False, None
    llm_ms = []
    for it in range(5):
        t0 = time.time()
        try:
            r = None
            for attempt in range(4):
                try:
                    r = client.chat.completions.create(model="gpt-4o-mini", messages=messages,
                                                       max_tokens=2048, temperature=0.0)
                    break
                except Exception as e:  # retry w/ backoff
                    if attempt == 3: raise
                    time.sleep(2 ** attempt)
        except Exception as e:
            final_answer = f"ERROR: {e}"; break
        llm_ms.append((time.time() - t0) * 1000)
        c = r.usage.prompt_tokens * PRICE_IN + r.usage.completion_tokens * PRICE_OUT
        cost += c; spend.add(c)
        text = r.choices[0].message.content or ""
        st = {"iter": it, "llm_text": text, "in_tok": r.usage.prompt_tokens, "out_tok": r.usage.completion_tokens,
              "latency_ms": llm_ms[-1]}
        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            st["final"] = True; steps.append(st); break
        am = re.search(r"Action:\s*(\w+)", text)
        im = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if am:
            tool_name = am.group(1).strip()
            try:
                params = json.loads(im.group(1)) if im else {}
            except Exception:
                params = {}
            exec_params = params
            if inject and not injection_done:
                if tool_name in T.TOOL_SCHEMAS:
                    exec_params = T.semantic_wrong(params)
                    injected = {"tool": tool_name, "orig": params, "injected": exec_params}
                injection_done = True
            ex = T.execute(tool_name, exec_params)
            obs = json.dumps(ex["result"], default=str)
            st.update({"tool_name": tool_name, "params": params, "exec_params": exec_params,
                       "validation_errors": ex["validation_errors"], "tool_result": ex["result"], "obs": obs})
            steps.append(st)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            st["no_action"] = True; steps.append(st)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})
    if not final_answer:
        final_answer = "Max steps reached."
    gt = task.get("correct_final_answer", "")
    return {"task_id": task["task_id"], "domain": domain, "source": task.get("source", ""),
            "difficulty": task.get("difficulty", ""), "query": task["query"], "gt": gt,
            "required_tools": task.get("required_tools", []),
            "correct_tool_sequence": task.get("correct_tool_sequence", []),
            "steps": steps, "final_answer": final_answer, "cost": cost, "llm_ms": llm_ms,
            "inject": inject, "injection": injected, "show_schema": show_schema,
            "heuristic_correct": heuristic_correct(gt, final_answer)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["clean", "p2sem", "retail", "retail_p2sem"])
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show_schema", action="store_true")
    a = ap.parse_args()
    import openai
    client = openai.OpenAI(api_key=load_key())
    if a.run in ("clean", "p2sem"):
        tasks = {t["task_id"]: t for t in json.load(open(f"{REPO}/data/tasks/full_2000_tasks.json"))}
        ids = sorted({r["task_id"] for r in csv.DictReader(open(f"{REPO}/data/results/interceptor_heldout_600.csv", newline=""))
                      if r["model"] == "gpt4o_mini"})
        work = [tasks[i] for i in ids]
    else:
        work = [t for t in json.load(open(f"{REPO}/data/tasks/ecommerce_300_tasks.json")) if t.get("split") == "test"]
    if a.limit: work = work[:a.limit]
    inject = a.run in ("p2sem", "retail_p2sem")
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{a.run}_{a.tag}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["task_id"] for l in open(path)}
    work = [t for t in work if t["task_id"] not in done]
    spend = Spend(); lock = threading.Lock(); n = 0
    with open(path, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_task, client, t, inject, spend, a.show_schema) for t in work]
        for fu in as_completed(futs):
            try:
                rec = fu.result()
            except Exception as e:
                print("FAIL", e); continue
            with lock:
                f.write(json.dumps(rec, default=str) + "\n"); f.flush(); n += 1
    print(f"{a.run}_{a.tag}: wrote {n} (+{len(done)} existing), spend ${spend.v:.4f}")


if __name__ == "__main__":
    main()
