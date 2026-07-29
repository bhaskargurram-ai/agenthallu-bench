"""Native function-calling tool-call-rate probe (addresses the harness-specificity
caveat for the fabricated-tool-use finding).

For a model served by vLLM with --enable-auto-tool-choice, sends the same P2
semantic-wrong task queries but via the OpenAI *native* function-calling API
(tools=..., tool_choice="auto") instead of the prompted-ReAct text format, and
measures the fraction of responses that emit a real tool_call. Comparing this to
the prompted-ReAct tool-call rate shows whether a low rate is harness-specific.

Usage:
  VLLM_BASE_URL=http://localhost:8000/v1 \
  python experiments/native_toolcall_probe.py --model qwen25_7b --limit 200 \
    --tasks full_500_tasks.json --output data/results/native_toolcall_<m>.json
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.tool_executor import ToolExecutor


def to_openai_tools(tools):
    out = []
    for t in tools:
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", "")[:300],
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # served-model-name on the vLLM endpoint
    ap.add_argument("--tasks", default="full_500_tasks.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    import openai
    base = os.path.join(os.path.dirname(__file__), "..")
    client = openai.OpenAI(api_key="EMPTY", base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    te = ToolExecutor()

    tasks = json.load(open(os.path.join(base, "data", "tasks", args.tasks)))
    tasks = [t for t in tasks if t.get("split", "test") == "test" and t.get("domain") != "knowledge"][:args.limit]

    SYS = "You are an agent that answers questions using the provided tools. Call a tool when one is needed."

    def probe(task):
        domain = task.get("domain")
        tools = te.get_tools_for_domain(domain) if domain else te.list_tools()
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": task["query"]}],
                tools=to_openai_tools(tools), tool_choice="auto",
                max_tokens=512, temperature=0.0,
            )
            tc = r.choices[0].message.tool_calls
            return 1 if tc else 0
        except Exception as e:
            return ("err", str(e)[:120])

    called = 0; errs = 0; n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(probe, tasks):
            if isinstance(res, tuple):
                errs += 1
                if errs <= 2:
                    print("  err:", res[1])
            else:
                called += res; n += 1
    rate = round(100 * called / n, 1) if n else None
    out = {"model": args.model, "n": n, "native_tool_call_rate_pct": rate,
           "tool_calls": called, "errors": errs}
    print(json.dumps(out))
    if args.output:
        json.dump(out, open(os.path.join(base, args.output), "w"), indent=2)


if __name__ == "__main__":
    main()
