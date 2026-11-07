"""V_00_inspect: print random scored examples for (source, outcome) to audit the rules (development)."""
import sys, os, json, random, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_traces as L, V_core as V
pk, src, oc, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
S = dict(pickle.load(open(pk, "rb")))
recs, _ = L.load_all(); tasks = L.load_tasks()
cand = [r for r in recs if r["source"] == src and S[r["session_id"]]["outcome"] == oc]
random.Random(seed).shuffle(cand)
for r in cand[:n]:
    s = S[r["session_id"]]
    print("="*100); print(r["session_id"], "|", tasks[r["task_id"]]["query"])
    for c in r["calls"]:
        print("  CALL", c["tool"], json.dumps(c["params"])[:150], "->", json.dumps(c["result"])[:220], "| exec" if c["executed"] else "| SCHEMA_ERR")
    print("  ANSWER:", r["final_answer"][:600].replace("\n"," / "))
    print("  ", {k: s.get(k) for k in ["T","P","S","A","T_missing","per_slot","A_recall","A_precision","A_key_fail","ungrounded","claim_detail","conditions","asserts"]})
