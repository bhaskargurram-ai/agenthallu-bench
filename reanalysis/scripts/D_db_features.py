"""Structural per-session features from the original P2 trace DB (read-only).
Usage: python3 -I D_db_features.py <p2_day6.db> <p2_all_rejudged.csv> <out.csv>"""
import csv, sqlite3, sys, re
csv.field_size_limit(10**9)
db, csvp, out = sys.argv[1:4]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
rows = list(csv.DictReader(open(csvp)))
ERR = re.compile(r"(error|invalid|not found|missing|failed|exception|out of range|must be)", re.I)
res = []
KNOWN = set(r[0] for r in c.execute("select distinct tool_name from steps where step_type='action' and tool_name is not null"))
for x in rows:
    steps = c.execute("select step_type, content, tool_name, tool_result, param_errors from steps "
                      "where session_id=? order by step_id", (x["session_id"],)).fetchall()
    if not steps:
        continue
    acts = [s for s in steps if s[0] == "action"]
    obs = [s for s in steps if s[0] == "observation"]
    fin_idx = max((i for i, s in enumerate(steps) if s[0] == "final"), default=None)
    prev = steps[fin_idx - 1] if fin_idx and steps[fin_idx - 1][0] == "thought" else None
    prev_txt = prev[1] if prev else ""
    obs_err = [s for s in obs if ERR.search((s[3] or "") + " " + (s[1] or ""))]
    res.append({
        "session_id": x["session_id"], "model": x["model"],
        "csv_tool_selected": x["tool_selected"].strip(),
        "db_n_actions": len(acts), "db_n_observations": len(obs),
        "db_has_final": int(fin_idx is not None),
        "final_turn_has_action_text": int(bool(re.search(r"\bAction\s*:", prev_txt))),
        "final_turn_has_observation_text": int(bool(re.search(r"\bObservation\s*:", prev_txt))),
        "all_observations_error": int(bool(obs) and len(obs_err) == len(obs)),
        "n_steps": len(steps),
        "inline_action_known_tool": int(bool(re.search(r"\bAction\s*:\s*`?(\w+)", prev_txt)) and
                                        re.search(r"\bAction\s*:\s*`?(\w+)", prev_txt).group(1) in KNOWN),
        "inline_n_actions": len(re.findall(r"\bAction\s*:", prev_txt)),
    })
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
print("wrote", len(res))
