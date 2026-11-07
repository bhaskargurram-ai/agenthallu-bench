"""C_09: Can the layers detect P2 propagation at all? Label-free test using injection ground truth.
For tasks where a schema-valid semantic_wrong injection was applied (p2sem_<tag>), compare each layer's firing
rate with its firing rate on the SAME task_ids in the un-injected run (clean_<tag>). A detector of propagation
must fire more often on injected traces. Also: does the injected wrong value / its numbers appear in observations
(i.e. is L3 structurally blind)?"""
import json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import OUT, mcnemar_exact
from C_03_replay import features, POLICIES

res = {}
for tag, clean, p2 in [("schema", "clean_schema", "p2sem_schema"), ("original", "clean_r1", "p2sem_r1"),
                       ("retail_schema", "retail_schema", "retail_p2sem_schema"), ("retail_original", "retail_r1", "retail_p2sem_r1")]:
    C = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{OUT}/traces/{clean}.jsonl")}
    P = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{OUT}/traces/{p2}.jsonl")}
    ids = []
    grounded_vals = 0
    for t, d in P.items():
        inj = d.get("injection")
        ts = [s for s in d["steps"] if "tool_name" in s]
        if inj and ts and not ts[0]["validation_errors"] and t in C:
            ids.append(t)
            # injected value echoed in the observation (so the observation is self-consistent with the wrong arg)
            changed = [v for k, v in inj["injected"].items() if inj["orig"].get(k) != v]
            if changed and all(str(v).lower() in ts[0]["obs"].lower() for v in changed):
                grounded_vals += 1
    out = {"n_schema_valid_injected": len(ids), "injected_value_echoed_in_observation": grounded_vals}
    for p in ["L1>=1", "L1>=2", "L2paper>=1", "L2code>=2", "L3", "L3 (enumerator-fixed)", "ORIG (L1>=2 | L3; as run)",
              "PAPER-COVERAGE (L1>=1 | L2paper>=1 | L3)", "Arg-grounding (diag. L4)"]:
        fc = [POLICIES[p](features(C[t])) for t in ids]
        fp = [POLICIES[p](features(P[t])) for t in ids]
        b = sum(1 for i in range(len(ids)) if fp[i] and not fc[i]); c = sum(1 for i in range(len(ids)) if fc[i] and not fp[i])
        out[p] = {"fire_rate_injected": sum(fp) / len(ids) if ids else None, "fire_rate_clean_same_tasks": sum(fc) / len(ids) if ids else None,
                  "mcnemar_p": mcnemar_exact(b, c), "b_c": [b, c]}
    # answer mentions the injected (wrong) value -> propagated into the final answer
    prop = 0
    for t in ids:
        inj = P[t]["injection"]
        changed = [v for k, v in inj["injected"].items() if inj["orig"].get(k) != v]
        if changed and any(str(v).lower() in P[t]["final_answer"].lower() for v in changed):
            prop += 1
    out["final_answer_mentions_injected_value"] = prop
    res[tag] = out
json.dump(res, open(f"{OUT}/propagation_detection.json", "w"), indent=1)
print(json.dumps(res, indent=1))
