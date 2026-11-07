"""Critique 5: build a blind, stratified sheet for an additional human-labeling round.

Strata (sizes fixed a priori):
  OW_S2   open-weight models, EPS>=2 (S2=1)        4 models x 8  = 32
  OW_LO   open-weight models, EPS<2                4 models x 8  = 32
  PR_S2   proprietary models, EPS>=2, not in the original 100      = 26
  ANCHOR  re-label of original calibration traces (drift check)    = 10
Within each model-stratum cell, sampling is domain-balanced where possible.
No judge verdicts are written to the annotator sheet; they live in the private key file.
"""
import random
import sys
from collections import defaultdict
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from B_common import *  # noqa

OPEN = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b"]
rng = random.Random(SEED)
p2 = load_p2()
TASKS = load_tasks()
calib = sample_ids()
calib_set = set(calib)
ann = load_annotators()


def eps(r):
    return int(r["eps"] or 0)


def pick(pool, k):
    """Domain-balanced sampling without replacement."""
    by = defaultdict(list)
    for r in pool:
        by[r["domain"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out = []
    doms = sorted(by)
    while len(out) < k and any(by.values()):
        for d in doms:
            if by[d] and len(out) < k:
                out.append(by[d].pop())
    return out


rows = [r for r in p2.values() if r["session_id"] not in calib_set]
chosen = []  # (row, stratum, cell_population, cell_n)
for m in OPEN:
    for stratum, cond in [("OW_S2", lambda r: eps(r) >= 2), ("OW_LO", lambda r: eps(r) < 2)]:
        pool = [r for r in rows if r["model"] == m and cond(r)]
        got = pick(pool, 8)
        chosen += [(r, stratum, len(pool), len(got)) for r in got]

prop_models = sorted({r["model"] for r in p2.values()} - set(OPEN))
pr_pool = {m: [r for r in rows if r["model"] == m and eps(r) >= 2] for m in prop_models}
# 26 rows over 9 proprietary models: 3 each, minus 1 from the largest pools -> proportional-ish
alloc = {m: min(3, len(pr_pool[m])) for m in prop_models}
while sum(alloc.values()) > 26:
    m = max(alloc, key=lambda x: (alloc[x], len(pr_pool[x])))
    alloc[m] -= 1
for m in prop_models:
    got = pick(pr_pool[m], alloc[m])
    chosen += [(r, "PR_S2", len(pr_pool[m]), len(got)) for r in got]

# Anchors: 5 unanimous + 5 items where the 3 annotators disagreed (or as many as exist)
unan = [s for s in calib if len({ann[a][s] for a in ann}) == 1]
dis = [s for s in calib if len({ann[a][s] for a in ann}) > 1]
rng.shuffle(unan); rng.shuffle(dis)
anchors = dis[:5] + unan[:10 - min(5, len(dis))]
chosen += [(p2[s], "ANCHOR", 100, 10) for s in anchors]

assert len({r["session_id"] for r, *_ in chosen}) == len(chosen)
rng.shuffle(chosen)

sheet, key = [], []
for i, (r, stratum, pop, n) in enumerate(chosen, 1):
    item = f"L2_{i:03d}"
    query = r["query"] or TASKS.get(r["task_id"], {}).get("query", "")
    sheet.append({"item_id": item, "domain": r["domain"], "query": query,
                  "reference_answer": r["ground_truth"], "agent_final_answer": r["final_answer"],
                  "YOUR_LABEL": "", "UNSURE": "", "NOTES": ""})
    key.append({"item_id": item, "session_id": r["session_id"], "model": r["model"],
                "family": "open_weight" if r["model"] in OPEN else "proprietary",
                "stratum": stratum, "eps": r["eps"], "S2": int(eps(r) >= 2), "domain": r["domain"],
                "cell_population": pop, "cell_sample": n,
                "sampling_weight": round(pop / n, 4) if stratum != "ANCHOR" else "",
                "query_was_empty_in_csv": int(not r["query"]),
                "answer_len": len(r["final_answer"] or ""),
                "answer_truncated_for_original_judges": int(len(r["final_answer"] or "") > 300),
                "gpt4o_verdict": r["gpt4o_verdict"], "gemini_25_flash_verdict": r["gemini_25_flash_verdict"],
                "gpt4o_mini_verdict": r["gpt4o_mini_verdict"], "ensemble_correct": r["ensemble_correct"],
                "heuristic_correct": r["heuristic_correct"],
                "orig_labels": json.dumps({a: ann[a][r["session_id"]] for a in ann}) if stratum == "ANCHOR" else ""})

L2 = os.path.join(OUT, "labeling_round2")
os.makedirs(L2, exist_ok=True)
for name, data in [("labeling_sheet_round2_BLANK.csv", sheet), ("PRIVATE_key_round2.csv", key)]:
    with open(os.path.join(L2, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

from collections import Counter
print(len(sheet), Counter(k["stratum"] for k in key), Counter(k["model"] for k in key))
print(Counter((k["family"], k["S2"]) for k in key), Counter(k["domain"] for k in key))
