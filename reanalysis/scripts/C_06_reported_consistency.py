"""C_06: Arithmetic consistency audit of the REPORTED Table 8 rows (raw open-weight interceptor data absent).

For each row: find n in [50, 1000] for which every reported percentage is reproduced by an integer count
(rounded to 1 dp) and Sv+Bk == #abstentions (Sv/Bk defined as abstained & ctrl-hallucinated / abstained &
ctrl-correct, the definition that reproduces 133/68 and 16/1 from the GPT-4o-mini/Gemini CSVs).
Then compare Ctrl-Int (count) with Sv under both orderings, and compute the random-abstention benchmark
at matched abstention: E[Int] = Ctrl * (1 - Abst).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import OUT

ROWS = {  # model: ctrl%, int%, abst%, (Sv,Bk) submitted, (Sv,Bk) arXiv v2
    "GPT-4o-mini": (55.8, 32.8, 33.5, (133, 68), (133, 68)),
    "Gemini-2.0-Flash": (44.5, 43.2, 2.8, (16, 1), (16, 1)),
    "Qwen2.5-7B": (55.3, 31.3, 55.3, (91, 75), (75, 91)),
    "Qwen2.5-14B": (45.0, 37.0, 27.0, (24, 57), (24, 57)),
    "Hermes-3-Llama-8B": (43.0, 39.0, 13.3, (24, 16), (16, 24)),
}


def ok(pct, n):
    c = round(pct * n / 100)
    return abs(round(100 * c / n, 1) - pct) < 1e-9, c


res = {}
for m, (cp, ip, ap, sub, v2) in ROWS.items():
    ns = []
    for n in range(50, 1001):
        a, C = ok(cp, n); b, I = ok(ip, n); c, A = ok(ap, n)
        if a and b and c and A == sum(sub):
            ns.append({"n": n, "ctrl_h": C, "int_h": I, "abst": A})
    r = {"consistent_n": [x["n"] for x in ns]}
    if ns:
        x = ns[0]
        n = x["n"]
        r.update(x)
        r["ctrl_minus_int"] = x["ctrl_h"] - x["int_h"]
        for lab, (sv, bk) in [("submitted", sub), ("arxiv_v2", v2)]:
            r[lab] = {"Sv": sv, "Bk": bk, "precision": sv / (sv + bk),
                      "implied_net_answered_flips(Int - (Ctrl - Sv))": x["int_h"] - (x["ctrl_h"] - sv),
                      "precision_minus_base_rate": sv / (sv + bk) - x["ctrl_h"] / n}
        r["random_abstention_expected_int_pct"] = cp * (1 - ap / 100)
        r["int_minus_random_pp"] = ip - cp * (1 - ap / 100)
        # the paper's CIs match an UNPAIRED two-arm bootstrap (GPT-4o-mini half-width 5.6pp); normal approx:
        import math
        p1, p2 = cp / 100, ip / 100
        hw = 196 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        r["unpaired_95CI_delta_pp"] = [ip - cp - hw, ip - cp + hw]
    res[m] = r
json.dump(res, open(f"{OUT}/reported_table_consistency.json", "w"), indent=1)
print(json.dumps(res, indent=1))
