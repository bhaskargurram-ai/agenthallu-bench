"""Critique 4: self-preference (own-family leniency) of LLM judges.

(A) Human-anchored, n=100: for each judge, bias = judge YES-rate - human(majority3) YES-rate,
    FP rate (judge YES | human NO), FN rate (judge NO | human YES), on own vs other traces;
    difference (own - other) with stratified bootstrap 95% CI.
    'own' at two levels: exact same model, and same vendor family.
(B) Human-free, n=2150 (all P2 rows): relative leniency diff-in-diff between an OpenAI judge and
    Gemini-2.5-Flash: [P(J_openai YES) - P(J_gemini YES)] on OpenAI-agent traces minus the same on
    Gemini-agent traces. >0 means each judge favors its own vendor relative to the other.
"""
import sys
from collections import defaultdict
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from B_common import *  # noqa

ann = load_annotators()
ids = sample_ids()
p2 = load_p2()
maj = {s: int(sum(ann[a][s] for a in ann) >= 2) for s in ids}

VENDOR = {"gpt4o": "openai", "gpt4o_mini": "openai", "gpt41_mini": "openai", "gpt41_nano": "openai",
          "gpt35_turbo": "openai", "o3_mini": "openai", "gemini_20_flash": "google",
          "gemini_25_flash": "google", "deepseek_v3": "deepseek", "qwen25_3b": "alibaba",
          "qwen25_7b": "alibaba", "qwen25_14b": "alibaba", "hermes3_llama8b": "meta_nous"}

# judge name -> (verdict dict over the 100, vendor, exact agent model it corresponds to)
J = {"gpt4o(stored,P0)": ({s: to01(p2[s]["gpt4o_verdict"]) for s in ids}, "openai", "gpt4o"),
     "gpt4o_mini(stored,P0)": ({s: to01(p2[s]["gpt4o_mini_verdict"]) for s in ids}, "openai", "gpt4o_mini"),
     "gemini25flash(stored,P0)": ({s: to01(p2[s]["gemini_25_flash_verdict"]) for s in ids}, "google", "gemini_25_flash"),
     "ensemble(stored)": ({s: to01(p2[s]["ensemble_correct"]) for s in ids}, "openai", None)}
raw = defaultdict(dict)
with open(os.path.join(OUT, "B03_rejudge_raw.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        raw[(d["judge"], d["prompt"])][d["session_id"]] = d["verdict"] or 0
for (j, pid), m in raw.items():
    exact = {"gpt-4o-mini": "gpt4o_mini", "gpt-4.1-mini": "gpt41_mini", "gpt-4o": "gpt4o"}.get(j)
    J[f"{j}({pid})"] = (m, "openai", exact)


def stats(sub, m):
    n = len(sub)
    hy = [s for s in sub if maj[s]]
    hn = [s for s in sub if not maj[s]]
    return {"n": n,
            "bias": (sum(m[s] for s in sub) - sum(maj[s] for s in sub)) / n if n else float("nan"),
            "fp_rate": sum(m[s] for s in hn) / len(hn) if hn else float("nan"),
            "fn_rate": sum(1 - m[s] for s in hy) / len(hy) if hy else float("nan"),
            "n_humanNO": len(hn), "n_humanYES": len(hy)}


def boot_diff(own, other, m, field, seed=SEED):
    rng = random.Random(seed)
    vals = []
    for _ in range(N_BOOT):
        a = [own[rng.randrange(len(own))] for _ in own]
        b = [other[rng.randrange(len(other))] for _ in other]
        v = stats(a, m)[field] - stats(b, m)[field]
        if v == v:
            vals.append(v)
    vals.sort()
    return (vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1])


resA = {}
for name, (m, vendor, exact) in J.items():
    e = {}
    levels = [("same_vendor", lambda s: VENDOR[p2[s]["model"]] == vendor)]
    if exact:
        levels.append(("exact_model", lambda s, ex=exact: p2[s]["model"] == ex))
    for lvl, f in levels:
        own = [s for s in ids if f(s)]
        other = [s for s in ids if not f(s)]
        so, sx = stats(own, m), stats(other, m)
        e[lvl] = {"own": so, "other": sx}
        for field in ["bias", "fp_rate", "fn_rate"]:
            e[lvl][f"diff_{field}_own_minus_other"] = {"diff": so[field] - sx[field],
                                                       "ci95": boot_diff(own, other, m, field)}
    resA[name] = e

# (B) human-free diff-in-diff over all 2150 P2 rows (stored verdicts)
allr = list(p2.values())


def did(rows, jo, jg):
    oa = [r for r in rows if VENDOR[r["model"]] == "openai"]
    ga = [r for r in rows if VENDOR[r["model"]] == "google"]
    if not oa or not ga:
        return float("nan")
    d_o = sum(to01(r[jo]) - to01(r[jg]) for r in oa) / len(oa)
    d_g = sum(to01(r[jo]) - to01(r[jg]) for r in ga) / len(ga)
    return d_o - d_g


resB = {}
for jo in ["gpt4o_verdict", "gpt4o_mini_verdict"]:
    point = did(allr, jo, "gemini_25_flash_verdict")
    rng = random.Random(SEED)
    oa = [r for r in allr if VENDOR[r["model"]] == "openai"]
    ga = [r for r in allr if VENDOR[r["model"]] == "google"]
    vals = []
    for _ in range(2000):
        rs = [oa[rng.randrange(len(oa))] for _ in oa] + [ga[rng.randrange(len(ga))] for _ in ga]
        vals.append(did(rs, jo, "gemini_25_flash_verdict"))
    vals.sort()
    # per-vendor raw gaps
    gap = {}
    for v in ["openai", "google", "deepseek", "alibaba", "meta_nous"]:
        rows = [r for r in allr if VENDOR[r["model"]] == v]
        gap[v] = {"n": len(rows), "openai_judge_yes": sum(to01(r[jo]) for r in rows) / len(rows),
                  "gemini_judge_yes": sum(to01(r["gemini_25_flash_verdict"]) for r in rows) / len(rows)}
    resB[f"{jo}_vs_gemini_25_flash"] = {"did_openaiAgents_minus_googleAgents": point,
                                        "ci95": (vals[50], vals[1949]), "per_vendor": gap}

# (C) human-free exact-self DiD over all 2150 rows: judge J vs reference judge R,
#     [P(J YES)-P(R YES)] on traces produced by J's own model minus the same on all other traces.
def did_exact(rows, own_model, jcol, rcol):
    own = [r for r in rows if r["model"] == own_model]
    oth = [r for r in rows if r["model"] != own_model]
    f = lambda rr: sum(to01(r[jcol]) - to01(r[rcol]) for r in rr) / len(rr)
    return f(own) - f(oth), len(own)


resC = {}
for own_model, jcol, rcol in [("gpt4o", "gpt4o_verdict", "gemini_25_flash_verdict"),
                              ("gpt4o_mini", "gpt4o_mini_verdict", "gemini_25_flash_verdict"),
                              ("gemini_25_flash", "gemini_25_flash_verdict", "gpt4o_verdict"),
                              ("gemini_25_flash", "gemini_25_flash_verdict", "gpt4o_mini_verdict")]:
    point, n_own = did_exact(allr, own_model, jcol, rcol)
    rng = random.Random(SEED)
    own = [r for r in allr if r["model"] == own_model]
    oth = [r for r in allr if r["model"] != own_model]
    vals = sorted(did_exact([own[rng.randrange(len(own))] for _ in own] + [oth[rng.randrange(len(oth))] for _ in oth],
                            own_model, jcol, rcol)[0] for _ in range(2000))
    resC[f"{jcol}_minus_{rcol}_on_{own_model}"] = {"n_own": n_own, "did": point, "ci95": (vals[50], vals[1949])}

res = {"A_human_anchored_n100": resA, "B_human_free_n2150": resB, "C_human_free_exact_self": resC,
       "n_by_vendor_in_100": {v: sum(VENDOR[p2[s]["model"]] == v for s in ids) for v in set(VENDOR.values())}}
save_json(res, "B07_self_preference.json")

print(res["n_by_vendor_in_100"])
for name, e in resA.items():
    for lvl, x in e.items():
        d = x["diff_bias_own_minus_other"]
        fp = x["diff_fp_rate_own_minus_other"]
        print(f"{name:40s} {lvl:12s} n_own={x['own']['n']:3d} bias_own={x['own']['bias']:+.2f} "
              f"bias_other={x['other']['bias']:+.2f} diff={d['diff']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] "
              f"FPdiff={fp['diff']:+.2f} [{fp['ci95'][0]:+.2f},{fp['ci95'][1]:+.2f}]")
print(json.dumps(resB, indent=1))
print(json.dumps(resC, indent=1))
