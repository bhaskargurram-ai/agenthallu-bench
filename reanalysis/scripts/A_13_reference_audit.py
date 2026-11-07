"""Reviewer 1, point 2: why is baseline accuracy so low? Audit references, tool adherence, answer form, judges.

Reference classes (rule-based, auditable):
  empty            : ground_truth cell is empty (all 'knowledge' tasks) -> unscorable; heuristic is False by code
                     (`bool(gt and ...)`), ensemble judged 0/949 correct
  query_echo       : 'Completed: <the query verbatim>'
  descriptive      : a templated description of the task ('Weather analysis for Dubai.', 'Event deleted.') whose
                     content words all occur in the query or in a closed template vocabulary -> carries no gold value
  concrete_value   : contains at least one content token that is neither in the query nor in the template vocabulary
                     (candidate gold value; printed for manual inspection)
Answer forms: no_tool (no Action executed), placeholder ('[...]' bracketed template echo, 'Max steps reached.',
'ERROR:'), substantive (anything else).
"""
import sys, json, re
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from collections import Counter, defaultdict
import numpy as np
from A_common import *
from A_stats import wilson
from A_10_paired import TOOLS, MODEL_ORDER

TEMPLATE = set("""completed created event scheduled retrieved record for patient conditions medications drug interaction
check checked schedule and listed all events on calendar the weather analysis forecast hourly current vs historical
temperature in includes humidity condition data conditions appointment deleted complete is it going to of a an with
what s what's give me show my""".split())

def toks(s): return [t for t in re.findall(r"[a-z0-9_:'\-\.@]+", (s or "").lower().replace("'s", "")) if t.strip(".'")]

def ref_class(gt, q):
    if not gt.strip(): return "empty", []
    if gt.startswith("Completed:") and gt[len("Completed:"):].strip().lower().rstrip(".?") == q.strip().lower().rstrip(".?"):
        return "query_echo", []
    qt = set(t.strip(".'") for t in toks(q))
    novel = [t.strip(".'") for t in toks(gt) if t.strip(".'") not in qt and t.strip(".'") not in TEMPLATE and len(t.strip(".'")) > 1]
    return ("concrete_value" if novel else "descriptive"), novel

def answer_form(r):
    fa = (r["final_answer"] or "").strip()
    if fa.startswith("ERROR:") or fa == "Max steps reached.": return "error_or_maxsteps"
    if re.fullmatch(r"\[.*\]", fa, flags=re.S): return "placeholder_bracket"
    if r["tool_selected"] == "": return "no_tool_substantive"
    return "tool_substantive"

def main():
    bl = read("baseline_all_rejudged.csv"); p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
    res = {}
    # ---- reference classes over distinct tasks
    tasks = {}
    for r in bl + p2: tasks.setdefault(r["task_id"], (r["domain"], r["ground_truth"], r["query"] or tasks.get(r["task_id"], ("", "", ""))[2]))
    for r in bl: tasks[r["task_id"]] = (r["domain"], r["ground_truth"], r["query"])  # baseline has queries for all
    cls = {}; novel_examples = {}
    for t, (d, gt, q) in tasks.items():
        c, nv = ref_class(gt, q); cls[t] = c
        if nv: novel_examples[t] = {"ref": gt, "query": q, "novel_tokens": nv}
    p2_tasks = set(r["task_id"] for r in p2)
    res["reference_classes_all_450_baseline_tasks"] = dict(Counter(cls[t] for t in set(r["task_id"] for r in bl)))
    res["reference_classes_200_P2_tasks"] = dict(Counter(cls[t] for t in p2_tasks))
    res["reference_classes_by_domain"] = {d: dict(Counter(cls[t] for t, v in tasks.items() if v[0] == d)) for d in sorted(set(v[0] for v in tasks.values()))}
    res["concrete_value_candidates"] = novel_examples
    print("reference classes (450 baseline tasks):", res["reference_classes_all_450_baseline_tasks"])
    print("reference classes (200 P2 tasks):", res["reference_classes_200_P2_tasks"])
    print("by domain:", res["reference_classes_by_domain"])
    for t, v in list(novel_examples.items())[:60]: print("  CANDIDATE", t, "|", v["ref"], "| novel:", v["novel_tokens"])
    # ---- baseline decomposition
    J = {"ensemble": "ensemble_correct", "heuristic": "heuristic_correct", "gpt4o_mini": "gpt4o_mini_verdict"}
    def acc(rows, col): return (sum(tf(r[col]) for r in rows) / len(rows)) if rows else None
    dec = {}
    for m in [x for x in MODEL_ORDER if any(r["model"] == x for r in bl)] + ["_all"]:
        rows = [r for r in bl if (m == "_all" or r["model"] == m)]
        nk = [r for r in rows if r["domain"] != "knowledge"]
        forms = Counter(answer_form(r) for r in nk)
        o = {"n": len(rows), "n_knowledge_empty_ref": sum(r["domain"] == "knowledge" for r in rows)}
        for j, col in J.items():
            o[f"acc_all_{j}"] = acc(rows, col); o[f"acc_nonknowledge_{j}"] = acc(nk, col)
            for f in ["tool_substantive", "no_tool_substantive", "placeholder_bracket", "error_or_maxsteps"]:
                sub = [r for r in nk if answer_form(r) == f]
                o[f"acc_{f}_{j}"] = acc(sub, col)
            tc = [r for r in nk if answer_form(r) == "tool_substantive" and tf(r["tool_correct"])]
            o[f"acc_toolcorrect_substantive_{j}"] = acc(tc, col)
        o["answer_forms_nonknowledge"] = dict(forms)
        o["tool_correct_rate_nonknowledge"] = acc(nk, "tool_correct")
        o["first_tool_valid_rate_nonknowledge"] = sum(r["tool_selected"] in TOOLS for r in nk) / len(nk)
        o["answers_longer_than_300_chars_judge_truncation"] = sum(len(r["final_answer"]) > 300 for r in nk)
        dec[m] = o
    res["baseline_decomposition"] = dec
    a = dec["_all"]
    print("\nALL baseline (n=%d): ens acc all=%.3f nonK=%.3f heur all=%.3f nonK=%.3f" % (a["n"], a["acc_all_ensemble"], a["acc_nonknowledge_ensemble"], a["acc_all_heuristic"], a["acc_nonknowledge_heuristic"]))
    print(" forms:", a["answer_forms_nonknowledge"], "tool_correct", round(a["tool_correct_rate_nonknowledge"], 3))
    for f in ["tool_substantive", "no_tool_substantive", "placeholder_bracket", "error_or_maxsteps"]:
        print(f"  {f:22s} ens={a['acc_'+f+'_ensemble']} heur={a['acc_'+f+'_heuristic']} g4m={a['acc_'+f+'_gpt4o_mini']}")
    print("  tool-correct & substantive: ens", a["acc_toolcorrect_substantive_ensemble"], "heur", a["acc_toolcorrect_substantive_heuristic"])
    for m, o in dec.items():
        if m == "_all": continue
        print(f" {m:16s} ensAll={o['acc_all_ensemble']:.3f} heurAll={o['acc_all_heuristic']:.3f} ensNK={o['acc_nonknowledge_ensemble']:.3f} "
              f"forms={o['answer_forms_nonknowledge']} ens|toolsubst={o['acc_tool_substantive_ensemble']}")
    # ---- error-mass decomposition (ensemble, all 4550 baseline traces)
    wrong = [r for r in bl if not tf(r["ensemble_correct"])]
    em = Counter()
    for r in wrong:
        if r["domain"] == "knowledge": em["empty_reference_knowledge"] += 1
        else: em[answer_form(r)] += 1
    res["baseline_ensemble_error_mass"] = {"n_wrong": len(wrong), "n_total": len(bl), "by_cause": dict(em)}
    print("\nensemble-wrong mass:", res["baseline_ensemble_error_mass"])
    # ---- judge strictness evidence: human labels by answer form
    p2map = {r["session_id"]: r for r in read("p2_all_rejudged.csv")}
    hl = read("human_labeling_sheet_Bhaskar_labeled.csv"); yl = {r["session_id"]: r["YOUR_LABEL"].upper() for r in read("human_labeling_sheet_Yoshitha_labeled.csv")}
    hs = defaultdict(Counter)
    for h in hl:
        r = p2map[h["session_id"]]; f = answer_form(r)
        e = tf(r["ensemble_correct"]); hb = h["YOUR_LABEL"].upper() == "YES"
        hs[f]["n"] += 1; hs[f]["human_yes"] += hb; hs[f]["ens_yes"] += e; hs[f]["heur_yes"] += tf(r["heuristic_correct"])
        hs[f]["ens_no_human_yes"] += (hb and not e); hs[f]["ens_yes_human_no"] += (e and not hb)
        hs[f]["yoshitha_yes"] += yl.get(h["session_id"], "") == "YES"
    res["human_labels_by_answer_form"] = {k: dict(v) for k, v in hs.items()}
    print("\nhuman labels by answer form:", res["human_labels_by_answer_form"])
    # examples of ensemble-wrong / human-correct substantive answers
    ex = []
    for h in hl:
        r = p2map[h["session_id"]]
        if h["YOUR_LABEL"].upper() == "YES" and not tf(r["ensemble_correct"]):
            ex.append({"task": r["task_id"], "model": r["model"], "ref": r["ground_truth"], "answer": r["final_answer"][:220]})
    res["examples_ens_wrong_human_correct"] = ex
    for e in ex[:8]: print("  EX", e)
    # placeholder answers judged correct by heuristic (heuristic leniency)
    ph = [r for r in bl if r["domain"] != "knowledge" and answer_form(r) == "placeholder_bracket"]
    res["placeholder_examples"] = [{"model": r["model"], "ref": r["ground_truth"], "answer": r["final_answer"][:120],
                                    "heuristic": r["heuristic_correct"], "ensemble": r["ensemble_correct"]} for r in ph[:10]]
    json.dump(res, open(os.path.join(OUT, "reference_audit.json"), "w"), indent=2, default=float)

if __name__ == "__main__":
    main()
