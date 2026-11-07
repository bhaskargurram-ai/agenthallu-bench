"""Surface/slot metric functions shared by B_02 (round-1 analysis), B_04 and the round-2 scorer.
Rules fixed a priori (see B_02 docstring)."""
import re
import string
import unicodedata

# ---------------------------------------------------------------- gazetteers
# Cities = every city appearing in any task file + every substitute the
# semantic_wrong injector can produce (injector/parameter_injector.py).
TASK_CITIES = ['Bangkok', 'Beijing', 'Berlin', 'Cairo', 'Dubai', 'Istanbul', 'Lagos', 'London',
               'Mexico City', 'Moscow', 'Mumbai', 'New York', 'Paris', 'Rome', 'Seoul', 'Singapore',
               'Sydney', 'São Paulo', 'Tokyo', 'Toronto']
INJ_CITIES = ['Lyon', 'Manchester', 'Osaka', 'Los Angeles', 'Rio de Janeiro', 'Munich', 'Melbourne',
              'Vancouver', 'Chicago', 'Houston', 'Delhi', 'Abuja', 'Reykjavik']
CITIES = TASK_CITIES + INJ_CITIES
DRUGS = ['albuterol', 'amlodipine', 'aspirin', 'atorvastatin', 'cetirizine', 'fluticasone', 'ibuprofen',
         'levothyroxine', 'lisinopril', 'metformin', 'metoprolol', 'omeprazole', 'sertraline',
         'sumatriptan', 'warfarin', 'acetaminophen']
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
# Non-answer / refusal / error / self-flagged-inconsistency patterns (fixed a priori).
NON_ANSWER = re.compile(
    r"(max steps reached|\[waiting|cannot|can't|can not|unable to|not able to|could not|couldn't|"
    r"failed|an error|error occurred|not available|no data|please provide|please let me know|"
    r"i will (?:try|now|need)|let me (?:try|check|retry)|instead of|mix[- ]?up|appears to be (?:an )?"
    r"(?:incorrect|inaccurate)|seems (?:to be )?(?:incorrect|inaccurate)|discrepanc)", re.I)


def clean(s):
    s = unicodedata.normalize("NFKC", s or "")
    for ch in "‐‑‒–—−":
        s = s.replace(ch, "-")
    return s.replace("’", "'").replace("‘", "'")


UNIT_RE = re.compile(r"(°\s*[cf]\b|°|\bdegrees?\b|\bcelsius\b|\bfahrenheit\b|\bkm/h\b|\bkmh\b|\bmph\b|%|\bpercent\b)", re.I)


def normalize(s):
    s = clean(s).lower()
    s = UNIT_RE.sub(" ", s)
    s = s.translate(str.maketrans({c: " " for c in string.punctuation}))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return 0.0
    common = {}
    for t in set(p):
        common[t] = min(p.count(t), r.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def numbers(s):
    return [float(x) for x in NUM_RE.findall(clean(s).replace("-", " - ").replace(" - ", " ")) ]


def numeric_match(ans, ref, rel=0.01, ab=0.5):
    rn = numbers(ref)
    if not rn:
        return 0, False  # no numeric gold value -> cannot match
    an = numbers(ans)
    ok = all(any(abs(a - r) <= max(ab, rel * abs(r)) for a in an) for r in rn)
    return int(ok), True


def date_forms(iso):
    y, m, d = iso.split("-")
    mi = int(m) - 1
    mon = MONTHS[mi]
    dd = str(int(d))
    return [iso, f"{mon} {dd}", f"{mon[:3]} {dd}", f"{dd} {mon}", f"{dd} {mon[:3]}", f"{m}/{d}/{y}",
            f"{int(m)}/{dd}"]


def contains(text_l, ent):
    return re.search(r"(?<![a-z0-9])" + re.escape(ent.lower()) + r"(?![a-z0-9])", text_l) is not None


def extract_entities(query):
    q = clean(query)
    ql = q.lower()
    ents = {"date": re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q),
            "city": [c for c in CITIES if contains(ql, c)],
            "patient": re.findall(r"\bP\d{3}\b", q),
            "event": re.findall(r"\bEVT_\d+\b", q),
            "drug": [d for d in DRUGS if contains(ql, d)],
            "title": re.findall(r"'([^']{3,60})'", q)}
    return {k: sorted(set(v)) for k, v in ents.items() if v}


def present(ans_l, typ, ent):
    if typ == "date":
        return any(f in ans_l for f in date_forms(ent))
    return contains(ans_l, ent)


def conflicts(ans, ents):
    """Same-type entities in the answer that the query did not ask for."""
    al = clean(ans).lower()
    out = []
    qc = set(ents.get("city", []))
    for c in CITIES:
        if c not in qc and contains(al, c) and not any(c.lower() in x.lower() for x in qc):
            out.append(("city", c))
    qp = set(ents.get("patient", []))
    if qp:
        out += [("patient", p) for p in set(re.findall(r"\bP\d{3}\b", clean(ans))) if p not in qp]
    qe = set(ents.get("event", []))
    if qe:
        out += [("event", e) for e in set(re.findall(r"\bEVT_\d+\b", clean(ans))) if e not in qe]
    qd = set(ents.get("drug", []))
    if qd:
        out += [("drug", d) for d in DRUGS if d not in qd and contains(al, d)]
    return out


def heuristic_code(ans, gt):
    """Verbatim logic of experiments/run_p2_experiment.py L171-172."""
    return int(bool(gt and (gt.lower()[:20] in ans.lower()[:200] or
                            any(w in ans.lower() for w in gt.lower().split()[:3] if len(w) > 3))))


def heuristic_paper_text(ans, gt):
    """Heuristic as *described* in the paper: first-20 chars OR any word >3 chars."""
    return int(bool(gt and (gt.lower()[:20] in ans.lower()[:200] or
                            any(w in ans.lower() for w in gt.lower().split() if len(w) > 3))))


STOP = set("the a an and or of in on for to with my me i is are be what whats what's show get give "
           "check if it its at from all that this then also any have has do does".split())


def reference_kind(gt, query):
    """Classify the reference string."""
    g = clean(gt)
    if g.lower().startswith("completed:"):
        return "echo_of_query"
    gt_toks = [t for t in normalize(g).split() if t not in STOP]
    q_toks = set(normalize(query).split())
    novel = [t for t in gt_toks if t not in q_toks]
    # A reference carries a gold *value* only if it contains a number or entity not in the query.
    novel_numbers = [t for t in novel if re.fullmatch(r"\d+(\.\d+)?", t)]
    novel_caps = [w for w in re.findall(r"\b[A-Z][a-z]+\b", g) if w.lower() not in q_toks
                  and w.lower() not in STOP and w not in ("Created", "Checked", "Scheduled", "Retrieved",
                                                          "Calendar", "Events", "Event", "Conditions",
                                                          "Patient", "Drug", "Weather", "Current", "Forecast",
                                                          "Hourly", "The", "Completed")]
    if novel_numbers or novel_caps:
        return "possible_gold_value"
    return "task_description"




def slot_verdicts(query, answer):
    ents = extract_entities(query)
    al = clean(answer).lower()
    miss = [(t, e) for t, es in ents.items() for e in es if not present(al, t, e)]
    conf = conflicts(answer, ents)
    non_ans = bool(NON_ANSWER.search(clean(answer)))
    return {"slot_present": int(not miss), "slot_noconflict": int(not miss and not conf),
            "slot_strict": int(not miss and not conf and not non_ans), "non_answer": int(non_ans),
            "missing": miss, "conflicts": conf, "entities": ents}
