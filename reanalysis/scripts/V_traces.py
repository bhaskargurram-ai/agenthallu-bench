"""V_traces.py -- read-only loaders that turn every available AgentProp trace into one record format.

Record: {session_id, task_id, model, source, condition, final_answer, calls:[call_record], mode, n_exec_hint}
  mode = 'full'        : per-step tool calls + observations available (SQLite trace DBs, C_interceptor reruns)
  mode = 'answer_only' : only the final answer is available (open-weight CSVs, sweep / self-consistency DBs)

Executed parameters.  The P2 runner logs the agent's EMITTED params in steps.tool_params_raw and the corrupted
params in the injections table (first tool call only).  We therefore set the executed params of the first action
to injections.injected_value.  For DBs whose runners do not log injections (multiseed logs post-injection params
directly), the executed params are reconciled against the observation by deterministic replay (see reconcile()).
Data folders are only READ (sqlite in read-only URI mode); no code from them is imported or executed.
"""
import csv
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_core as V  # noqa: E402
import C_tools as SIM  # noqa: E402

csv.field_size_limit(sys.maxsize)
ROOT = "/Users/challagullayoshitha/Desktop/LifeNavigator/agenthallu_bench/data"
TRACES = f"{ROOT}/traces"
RESULTS = f"{ROOT}/results"
TASKS = f"{ROOT}/tasks"
REV = "/Users/challagullayoshitha/Downloads/agentprop_revision"

FULL_DBS = {  # db -> source label
    "baseline_day6": "baseline", "p2_day6": "p2", "interceptor_day8": "interceptor",
    "interceptor_control": "interceptor_control", "p4_day7": "p4", "ablation_day8": "ablation",
    "multiseed_day8": "multiseed",
}
ANSWER_ONLY_DBS = {"sweep_day8": "sweep", "self_consistency_day8": "self_consistency"}
EXCLUDED_DBS = {
    "p3_day7": "multi-turn: final answer responds to run-time generated follow-up turns that are not in the task spec",
    "baseline_smoke_traces": "development smoke test (3 sessions)", "baseline_traces": "development test (6 sessions)",
    "baseline": "simulated baseline ('baseline_simulated', not an LLM)", "day2_test": "development test (gpt-4o, 10)",
    "day3_injector_test": "injector unit test with mock agents",
}
OPEN_CSVS = {  # csv -> (source, condition column or fixed)
    "baseline_qwen25_3b.csv": "baseline", "baseline_qwen25_7b.csv": "baseline", "baseline_qwen25_14b.csv": "baseline",
    "baseline_hermes3_llama8b.csv": "baseline", "baseline_mistral_7b.csv": "baseline",
    "baseline_phi35_mini.csv": "baseline",
    "p2_qwen25_3b.csv": "p2", "p2_qwen25_7b.csv": "p2", "p2_qwen25_14b.csv": "p2", "p2_hermes3_llama8b.csv": "p2",
    "p2_mistral_7b.csv": "p2", "p2_phi35_mini.csv": "p2",
    "interceptor_qwen25_7b.csv": "interceptor_open", "interceptor_qwen25_14b.csv": "interceptor_open",
    "interceptor_hermes3_llama8b.csv": "interceptor_open",
    "interceptor_qwen25_7b_schemaonly.csv": "interceptor_open_schemaonly",
    "interceptor_qwen25_14b_schemaonly.csv": "interceptor_open_schemaonly",
    "interceptor_hermes3_llama8b_schemaonly.csv": "interceptor_open_schemaonly",
}
RERUN_JSONL = ["clean_r1", "clean_r2", "clean_schema", "p2sem_r1", "p2sem_schema", "retail_r1", "retail_schema",
               "retail_p2sem_r1", "retail_p2sem_schema"]


def load_tasks():
    t = {}
    for f in ["full_2000_tasks.json", "ecommerce_300_tasks.json", "full_500_tasks.json", "weather_tasks.json",
              "calendar_tasks.json", "medical_tasks.json"]:
        with open(f"{TASKS}/{f}") as fh:
            for x in json.load(fh):
                t.setdefault(x["task_id"], x)
    return t


def _ro(db):
    return sqlite3.connect(f"file:{TRACES}/{db}.db?mode=ro", uri=True)


def _j(s):
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def reconcile(call, stats):
    """For stateless tools: check the logged observation equals deterministic replay of the executed params.
    If not, test whether it equals replay of semantic_wrong(params) (unlogged P2 injection)."""
    if call["tool"] not in V.STATELESS or not call["success"] or not isinstance(call["params"], dict):
        return call
    try:
        rep = V.sim(call["tool"], call["params"])
    except Exception:
        stats["replay_exception"] = stats.get("replay_exception", 0) + 1
        return call
    obs = call["result"]
    if call["tool"] == "get_forecast":  # forecast dates = datetime.now() at run time: compare everything else
        strip = lambda o: {**o, "forecast": [{k: v for k, v in e.items() if k != "date"} for e in o.get("forecast", [])]}
        rep, obs = strip(rep), strip(obs)
    key = f"replay_{call['tool']}"
    if rep == obs:
        stats[key + "_match"] = stats.get(key + "_match", 0) + 1
        return call
    try:
        alt = SIM.semantic_wrong(call["params"])
        rep2 = V.sim(call["tool"], alt)
        if call["tool"] == "get_forecast":
            rep2 = strip(rep2)
    except Exception:
        rep2 = None
    if rep2 == obs:
        stats["reconciled_unlogged_injection"] = stats.get("reconciled_unlogged_injection", 0) + 1
        call = dict(call, params=alt)
        return call
    stats[key + "_mismatch"] = stats.get(key + "_mismatch", 0) + 1
    stats.setdefault("mismatch_examples", []).append({"tool": call["tool"], "params": call["params"],
                                                       "logged": call["result"], "replay": rep})
    return call


def load_db(db, source, stats):
    con = _ro(db)
    sess = con.execute("select session_id, task_id, model, domain, injection_type, final_answer from sessions").fetchall()
    inj = {}
    for sid, orig, injv in con.execute("select session_id, original_value, injected_value from injections"):
        inj.setdefault(sid, (orig, injv))
    steps = {}
    for sid, stype, tname, praw, tres, perr, content in con.execute(
            "select session_id, step_type, tool_name, tool_params_raw, tool_result, param_errors, content from steps "
            "order by session_id, step_id"):
        steps.setdefault(sid, []).append((stype, tname, praw, tres, perr, content))
    out = []
    for sid, tid, model, dom, itype, fa in sess:
        calls = []
        pending = None
        first = True
        for stype, tname, praw, tres, perr, content in steps.get(sid, []):
            if stype == "action":
                pending = (tname, _j(praw) if praw else {})
            elif stype == "observation" and pending is not None:
                tool, pe = pending
                pexec = pe
                if first and sid in inj and source == "p2":
                    o, injv = inj[sid]
                    ov, iv = _j(o), _j(injv)
                    if ov != pe:
                        stats["injection_original_mismatch"] = stats.get("injection_original_mismatch", 0) + 1
                    pexec = iv if isinstance(iv, dict) else pe
                res = _j(tres) if tres else _j(content)
                errs = _j(perr) or []
                c = V.call_record(tool, pe, pexec, res, errs)
                c = reconcile(c, stats)
                calls.append(c)
                pending = None
                first = False
        cond = itype or "baseline"
        out.append({"session_id": sid, "task_id": tid, "model": model, "source": source if dom != "ecommerce" or
                    source not in ("baseline", "p2") else source + "_retail", "condition": cond,
                    "final_answer": fa or "", "calls": calls, "mode": "full", "n_exec_hint": None, "db": db})
    return out


def load_answer_only_db(db, source):
    con = _ro(db)
    out = []
    for sid, tid, model, dom, itype, fa in con.execute(
            "select session_id, task_id, model, domain, injection_type, final_answer from sessions"):
        out.append({"session_id": sid, "task_id": tid, "model": model, "source": source, "condition": itype or "",
                    "final_answer": fa or "", "calls": [], "mode": "answer_only", "n_exec_hint": None, "db": db})
    return out


def load_open_csvs():
    out, seen = [], set()
    for f, source in OPEN_CSVS.items():
        p = f"{RESULTS}/{f}"
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r["session_id"] in seen:
                    continue
                seen.add(r["session_id"])
                steps = r.get("steps")
                n_exec = None
                if source == "baseline" and steps not in (None, ""):
                    # run_baseline.py: 2 steps (thought+final) + 3 per executed tool call
                    s = int(steps)
                    n_exec = (s - 2) // 3 if s >= 2 else 0
                if "final_answer" not in r:
                    continue
                cond = r.get("error_type") or r.get("condition") or source
                out.append({"session_id": r["session_id"], "task_id": r["task_id"], "model": r["model"],
                            "source": "open_" + source, "condition": cond, "final_answer": r["final_answer"] or "",
                            "calls": [], "mode": "answer_only", "n_exec_hint": n_exec, "db": f})
    return out


def load_reruns(stats):
    out = []
    for tag in RERUN_JSONL:
        p = f"{REV}/C_interceptor/traces/{tag}.jsonl"
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                calls = []
                for st in r["steps"]:
                    if "tool_name" in st:
                        c = V.call_record(st["tool_name"], st.get("params"), st.get("exec_params", st.get("params")),
                                          st.get("tool_result"), st.get("validation_errors"))
                        calls.append(reconcile(c, stats))
                out.append({"session_id": f"rerunC_{tag}_{r['task_id']}_{i}", "task_id": r["task_id"],
                            "model": "gpt4o_mini", "source": "rerun_C", "condition": tag,
                            "final_answer": r.get("final_answer", ""), "calls": calls, "mode": "full",
                            "n_exec_hint": None, "db": tag})
    return out


def load_all():
    stats = {}
    recs = []
    for db, src in FULL_DBS.items():
        recs += load_db(db, src, stats)
    for db, src in ANSWER_ONLY_DBS.items():
        recs += load_answer_only_db(db, src)
    recs += load_open_csvs()
    recs += load_reruns(stats)
    return recs, stats
