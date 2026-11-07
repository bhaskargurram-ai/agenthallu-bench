"""C2_00: structural exploration of the open-weight interceptor CSVs (original project). Read-only."""
import csv, sys, json, random, hashlib, collections, os
csv.field_size_limit(sys.maxsize)
ORIG = os.path.expanduser("~/Desktop/LifeNavigator/agenthallu_bench")
RES = f"{ORIG}/data/results"
def rd(fn): return list(csv.DictReader(open(f"{RES}/{fn}", newline="", encoding="utf-8")))
def h(fn): return hashlib.sha256(open(f"{RES}/{fn}", "rb").read()).hexdigest()[:16]
T = json.load(open(f"{ORIG}/data/tasks/full_500_tasks.json"))
test = [t["task_id"] for t in T if t.get("split", "test") == "test"]
random.Random(42).shuffle(test); exp300 = set(test[:300])
files = ["interceptor_qwen25_7b.csv", "interceptor_qwen25_14b.csv", "interceptor_hermes3_llama8b.csv",
         "interceptor_qwen25_7b_schemaonly.csv", "interceptor_qwen25_14b_schemaonly.csv",
         "interceptor_hermes3_llama8b_schemaonly.csv", "interceptor_open_models.csv"]
allrows = {}
for fn in files:
    R = rd(fn); allrows[fn] = R
    c = collections.Counter((r["model"], r["condition"]) for r in R)
    print("==", fn, h(fn), len(R), dict(c))
    for (m, cond) in c:
        ids = [r["task_id"] for r in R if r["model"] == m and r["condition"] == cond]
        print("   ", m, cond, "uniq", len(set(ids)), "== seed42 first300:", set(ids) == exp300,
              collections.Counter(i.rsplit("_", 1)[0] for i in ids))
    print("   abstain_reason prefixes:", collections.Counter(r["abstain_reason"][:24] for r in R if r["condition"] != "no_interceptor"))
    print("   final_answer ERROR rows:", sum(r["final_answer"].startswith("ERROR") for r in R),
          " Max steps:", sum(r["final_answer"].startswith("Max steps") for r in R))
# open_models == concat of three main files?
key = lambda r: tuple(r[k] for k in r)
cat = sorted(map(key, allrows[files[0]] + allrows[files[1]] + allrows[files[2]]))
print("open_models == concat(main3):", cat == sorted(map(key, allrows["interceptor_open_models.csv"])))
# controls in main vs schemaonly: same run?
for m, a, b in [("qwen25_7b", files[0], files[3]), ("qwen25_14b", files[1], files[4]), ("hermes3_llama8b", files[2], files[5])]:
    A = {r["task_id"]: r for r in allrows[a] if r["condition"] == "no_interceptor"}
    B = {r["task_id"]: r for r in allrows[b] if r["condition"] == "no_interceptor"}
    same_ans = sum(A[t]["final_answer"] == B[t]["final_answer"] for t in A if t in B)
    same_lab = sum(A[t]["final_correct"] == B[t]["final_correct"] for t in A if t in B)
    print(m, "ctrl main vs schemaonly: shared", len(set(A) & set(B)), "same answer", same_ans, "same label", same_lab,
          "session ids equal", sum(A[t]["session_id"] == B[t]["session_id"] for t in A if t in B))
