import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0, _REPO + "/reanalysis/scripts")
from A_common import *
from collections import Counter, defaultdict
p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
bl = read("baseline_all_rejudged.csv")
blk = {(r["task_id"], r["model"]): r for r in bl}
same_gt = same_q = same_tool = n = 0; tooldiff = Counter()
for r in p2:
    b = blk.get((r["task_id"], r["model"]))
    if not b: continue
    n += 1; same_gt += r["ground_truth"] == b["ground_truth"]; same_q += r["query"] == b["query"]
    same_tool += r["tool_selected"] == b["tool_selected"]
    if r["tool_selected"] != b["tool_selected"]: tooldiff[(r["model"], r["tool_selected"] == "", b["tool_selected"] == "")] += 1
print("paired", n, "same gt", same_gt, "same query", same_q, "same first tool", same_tool)
print("first-tool mismatches (model, p2 empty, bl empty):", tooldiff.most_common(30))
print("BL final_correct==heuristic_correct:", sum(r["final_correct"] == r["heuristic_correct"] for r in bl), len(bl))
print("BL knowledge gt empty:", Counter((r["ground_truth"] == "") for r in bl if r["domain"] == "knowledge"))
print("BL knowledge ensemble:", Counter(r["ensemble_correct"] for r in bl if r["domain"] == "knowledge"))
print("BL knowledge heuristic:", Counter(r["heuristic_correct"] for r in bl if r["domain"] == "knowledge"))
for m in sorted(set(r["model"] for r in bl)):
    rows = [r for r in bl if r["model"] == m]
    nk = [r for r in rows if r["domain"] != "knowledge"]
    print(m, len(rows), "ens all %.1f" % (100*sum(tf(r["ensemble_correct"]) for r in rows)/len(rows)),
          "heur all %.1f" % (100*sum(tf(r["heuristic_correct"]) for r in rows)/len(rows)),
          "| non-knowledge n", len(nk), "ens %.1f" % (100*sum(tf(r["ensemble_correct"]) for r in nk)/len(nk)),
          "heur %.1f" % (100*sum(tf(r["heuristic_correct"]) for r in nk)/len(nk)),
          "noTool %.1f" % (100*sum(r["tool_selected"] == "" for r in nk)/len(nk)))
