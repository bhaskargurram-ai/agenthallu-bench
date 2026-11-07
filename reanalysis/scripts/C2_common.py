"""Shared helpers for C2_* (open-weight interceptor) analyses. Reads the ORIGINAL project as data only."""
import csv, os, sys, json, random
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import tf, mcnemar_exact, hypergeom_sf, wilson  # noqa: F401
ORIG = os.path.expanduser("~/Desktop/LifeNavigator/agenthallu_bench")
ORIG_RES = f"{ORIG}/data/results"
REPO = "/Users/challagullayoshitha/Downloads/agentprop_repo/src"
OUT = "/Users/challagullayoshitha/Downloads/agentprop_revision/C2_openweight"
OPEN = ["qwen25_7b", "qwen25_14b", "hermes3_llama8b"]
PRETTY = {"gpt4o_mini": "GPT-4o-mini", "gemini_20_flash": "Gemini-2.0-Flash", "qwen25_7b": "Qwen2.5-7B",
          "qwen25_14b": "Qwen2.5-14B", "hermes3_llama8b": "Hermes-3-Llama-8B"}


def read_orig(fn):
    with open(os.path.join(ORIG_RES, fn), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def src(tid):
    return tid.rsplit("_", 1)[0]


def layer(reason):
    if not reason:
        return None
    if reason.startswith("Early interceptor L3"):
        return "L3"
    if reason.startswith("Early interceptor: L1"):
        return "L1"
    return "other"


def paired_boot(a, b, B=10000, seed=42):
    """95% percentile CI of mean(b)-mean(a) resampling task indices jointly."""
    rng = random.Random(seed)
    n = len(a)
    out = []
    for _ in range(B):
        s = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += b[i] - a[i]
        out.append(s / n)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]


def unpaired_normal_ci(p1, p2, n):
    import math
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    d = p2 - p1
    return d - 1.96 * se, d + 1.96 * se


def random_abstention(ch, k, B=20000, seed=7):
    """ch: control-hallucination indicators. Abstain on k uniformly random tasks.
    Returns exact expected hallucination rate after abstention and the simulated distribution of 'saved'."""
    n = len(ch)
    H = sum(ch)
    exp_h = H * (1 - k / n) / n
    rng = random.Random(seed)
    idx = list(range(n))
    sims = []
    for _ in range(B):
        S = rng.sample(idx, k)
        sims.append(sum(ch[i] for i in S))
    return exp_h, sims
