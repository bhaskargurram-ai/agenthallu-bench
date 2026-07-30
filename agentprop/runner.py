"""Core evaluation loop for the AgentProp-Bench harness.

Runs the released prompted-ReAct protocol (identical to the paper's baseline) for
any model, so a one-command run produces a leaderboard-comparable row. Reuses the
repo's ModelRouter (OpenAI / Google / OpenRouter / local vLLM) and ToolExecutor.

A model spec that isn't already a key in config.MODELS is registered on the fly
so users can score arbitrary models without editing config:
    agentprop eval --model gpt-4o-mini            # known key
    agentprop eval --model gpt-4o-2024-11-20 --api openai
    agentprop eval --model my-model --api local   # VLLM_BASE_URL endpoint
"""
import json
import re
import time
import uuid

_INFER = [
    ("openai", ("gpt-", "o1", "o3", "o4", "chatgpt")),
    ("google", ("gemini",)),
]


def resolve_model(model, api=None):
    """Ensure `model` is a usable key in config.MODELS; register it if needed.

    Returns the model key to hand to ModelRouter.
    """
    from config import MODELS
    if model in MODELS:
        return model
    if not api:
        api = "openrouter"  # safe default: OpenRouter proxies most model ids
        for backend, prefixes in _INFER:
            if model.lower().startswith(prefixes):
                api = backend
                break
    MODELS[model] = {
        "api": api,
        "model_id": model,
        "max_tokens": 2048,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
    }
    return model


def run_task(task, router, tools_exec, max_steps=5):
    """Run one task with the prompted-ReAct loop. Returns a result dict."""
    domain = task.get("domain")
    query = task["query"]
    tools = (tools_exec.get_tools_for_domain(domain)
             if domain and domain != "knowledge" else tools_exec.list_tools())
    tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    known_tools = {t["name"] for t in tools_exec.list_tools()}

    system_msg = (
        "You are an agent that answers questions using tools.\n"
        "Follow this format exactly:\n"
        'Thought: [reasoning]\nAction: [tool_name]\nAction Input: {"param": "value"}\n\n'
        "When done, respond with:\nFinal Answer: [your answer]\n\n"
        f"Available tools:\n{tool_desc}"
    )
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": query}]

    cost = tin = tout = steps = 0
    final_answer, tool_selected, emitted_real_call = "", "", False

    for _ in range(max_steps):
        try:
            r = router.complete(messages)
        except Exception as e:
            final_answer = f"ERROR: {e}"
            break
        cost += r["cost"]; tin += r["input_tokens"]; tout += r["output_tokens"]; steps += 1
        text = r["text"]

        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            break

        action = re.search(r"Action:\s*(\w+)", text)
        params_m = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if action:
            name = action.group(1).strip()
            tool_selected = tool_selected or name
            if name in known_tools:  # a real, registered tool call (vs. a hallucinated tool)
                emitted_real_call = True
            try:
                params = json.loads(params_m.group(1)) if params_m else {}
            except (json.JSONDecodeError, AttributeError):
                params = {}
            exec_result = tools_exec.execute(name, params)
            obs = json.dumps(exec_result.get("result"), default=str)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached."

    gt = task.get("correct_final_answer", "") or ""
    gt_tool = ""
    seq = task.get("correct_tool_sequence") or []
    if seq:
        gt_tool = seq[0].get("tool", "")
    tool_correct = (tool_selected == gt_tool) if gt_tool else True
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                          any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))

    return {
        "task_id": task.get("task_id"), "domain": domain,
        "tool_selected": tool_selected, "tool_correct": tool_correct,
        "emitted_real_call": emitted_real_call, "final_correct": final_correct,
        "final_answer": final_answer[:300], "cost": cost,
        "input_tokens": tin, "output_tokens": tout, "steps": steps,
    }


def evaluate(model, tasks, api=None, max_steps=5, workers=3, progress=True):
    """Run all tasks for one model and return the list of per-task result dicts."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agent.model_router import ModelRouter
    from agent.tool_executor import ToolExecutor

    key = resolve_model(model, api)
    tools_exec = ToolExecutor()
    results = []

    def _one(t):
        return run_task(t, ModelRouter(key), tools_exec, max_steps=max_steps)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if progress and done % 10 == 0:
                print(f"  ...{done}/{len(tasks)}", flush=True)
            try:
                results.append(fut.result())
            except Exception as e:  # keep going; one bad task shouldn't kill the run
                results.append({"task_id": futs[fut].get("task_id"), "final_answer": f"ERROR: {e}",
                                "final_correct": False, "tool_correct": False,
                                "emitted_real_call": False, "cost": 0, "steps": 0,
                                "input_tokens": 0, "output_tokens": 0,
                                "domain": futs[fut].get("domain"), "tool_selected": ""})
    return results
