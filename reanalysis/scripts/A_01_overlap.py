import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0, _REPO + "/reanalysis/scripts")
from A_common import *
from collections import Counter, defaultdict
p2 = read("p2_all_rejudged.csv"); bl = read("baseline_all_rejudged.csv")
for fn in P2_FILES[1:]:
    o = read(fn); m = o[0]["model"]
    a = {r["session_id"]: r for r in p2 if r["model"] == m}
    same = sum(1 for r in o if r["session_id"] in a)
    diffcols = Counter()
    for r in o:
        if r["session_id"] in a:
            for c in r:
                if c != "_file" and r[c] != a[r["session_id"]].get(c): diffcols[c] += 1
    print(fn, "in all:", same, "/", len(o), "differing cells:", dict(diffcols))
for fn in BL_FILES[1:]:
    o = read(fn); m = o[0]["model"]
    a = {r["session_id"]: r for r in bl if r["model"] == m}
    same = sum(1 for r in o if r["session_id"] in a)
    diffcols = Counter()
    for r in o:
        if r["session_id"] in a:
            for c in r:
                if c != "_file" and r[c] != a[r["session_id"]].get(c): diffcols[c] += 1
    print(fn, "in all:", same, "/", len(o), "differing cells:", dict(diffcols))
blk = defaultdict(list)
for r in bl: blk[(r["task_id"], r["model"])].append(r)
for m in sorted(set(r["model"] for r in p2)):
    pr = [r for r in p2 if r["model"] == m]
    matched = sum(1 for r in pr if (r["task_id"], m) in blk)
    print(m, "p2 n", len(pr), "matched baseline same task+model:", matched)
# task id sets
p2t = Counter(r["task_id"] for r in p2)
print("distinct p2 tasks", len(p2t))
blt = set(r["task_id"] for r in bl)
print("p2 tasks with any baseline", sum(1 for t in p2t if t in blt))
# columns values
for c in ["tool_correct","final_correct","ensemble_correct","heuristic_correct","gpt4o_verdict","gemini_25_flash_verdict","gpt4o_mini_verdict","reached_output","eps","cascade_count"]:
    print(c, Counter(r[c] for r in p2).most_common(8))
for c in ["tool_correct","final_correct","ensemble_correct","heuristic_correct","gpt4o_verdict","gemini_25_flash_verdict","gpt4o_mini_verdict","steps"]:
    print("BL", c, Counter(r[c] for r in bl).most_common(8))
