"""V_00_dev: development scoring pass (prints distributions + random examples for inspection)."""
import sys, os, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_traces as L, V_core as V
recs, stats = L.load_all()
tasks = L.load_tasks()
specs = {}
out = []
for r in recs:
    t = tasks[r["task_id"]]
    if r["task_id"] not in specs: specs[r["task_id"]] = V.build_spec(t)
    s = V.score_trace(t, r["calls"], r["final_answer"], spec=specs[r["task_id"]], answer_only=r["mode"]=="answer_only", n_exec_hint=r["n_exec_hint"])
    out.append((r, s))
c = collections.Counter((r["source"], s["outcome"]) for r, s in out)
for k in sorted(c): print(k, c[k])
print(collections.Counter(s.template for s in specs.values()))
import pickle; pickle.dump([(r["session_id"], s) for r, s in out], open(sys.argv[1], "wb"))
