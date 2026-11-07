import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""Shared loaders + agreement statistics for the B_judge (RQ1) revision analyses.

All inputs are read as data (csv module; multi-line quoted fields safe).
Run scripts with `python3 -I`; they add this directory to sys.path explicitly.
"""
import csv
import json
import os
import random

REPO = _REPO + ""
RES = os.path.join(REPO, "data/results")
OUT = _REPO + "/reanalysis/B_judge"
os.makedirs(OUT, exist_ok=True)

ANNOT_FILES = {
    "bhaskar": os.path.join(RES, "human_labeling_sheet_Bhaskar_labeled.csv"),
    "yoshitha": os.path.join(RES, "human_labeling_sheet_Yoshitha_labeled.csv"),
    "third": _REPO + "/data/results/human_labels_third_annotator.csv",
}
YOSHITHA_COMPLETED = os.path.join(REPO, "human_labeling_sheet_Yoshitha_completed.csv")

N_BOOT = 10000
SEED = 20261030


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm_label(v):
    v = (v or "").strip().upper()
    if v in ("YES", "Y", "1", "TRUE", "CORRECT"):
        return 1
    if v in ("NO", "N", "0", "FALSE", "INCORRECT"):
        return 0
    return None


def to01(v):
    return 1 if str(v).strip().lower() in ("true", "1", "yes", "y") else 0


def load_annotators():
    out = {}
    for name, path in ANNOT_FILES.items():
        out[name] = {r["session_id"]: norm_label(r["YOUR_LABEL"]) for r in read_csv(path)}
    return out


def load_p2():
    return {r["session_id"]: r for r in read_csv(os.path.join(RES, "p2_all_rejudged.csv"))}


def load_tasks():
    tasks = {}
    for fn in ["calendar_tasks.json", "medical_tasks.json", "weather_tasks.json",
               "full_2000_tasks.json", "full_500_tasks.json", "ecommerce_300_tasks.json"]:
        p = os.path.join(REPO, "data/tasks", fn)
        if os.path.exists(p):
            with open(p) as f:
                for t in json.load(f):
                    tasks.setdefault(t["task_id"], t)
    return tasks


def sample_ids():
    """The 100 calibration session_ids, in the order of annotator-1's sheet."""
    return [r["session_id"] for r in read_csv(ANNOT_FILES["bhaskar"])]


# ---------------------------------------------------------------- statistics
def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def raw_agree(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def fleiss_kappa(ratings):
    """ratings: list of per-item lists of 0/1 labels (same #raters)."""
    N = len(ratings)
    n = len(ratings[0])
    p1 = sum(sum(r) for r in ratings) / (N * n)
    p0 = 1 - p1
    Pi = []
    for r in ratings:
        y = sum(r)
        no = n - y
        Pi.append((y * (y - 1) + no * (no - 1)) / (n * (n - 1)))
    Pbar = sum(Pi) / N
    Pe = p1 ** 2 + p0 ** 2
    if Pe >= 1:
        return float("nan")
    return (Pbar - Pe) / (1 - Pe)


def gwet_ac1(a, b):
    n = len(a)
    po = raw_agree(a, b)
    pi = (sum(a) / n + sum(b) / n) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def bootstrap(stat_fn, n_items, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over items. stat_fn(idx_list) -> float."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n_items) for _ in range(n_items)]
        v = stat_fn(idx)
        if v == v:  # drop NaN
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return (lo, hi)


def kappa_ci(a, b, seed=SEED):
    k = cohen_kappa(a, b)
    ci = bootstrap(lambda idx: cohen_kappa([a[i] for i in idx], [b[i] for i in idx]),
                   len(a), seed=seed)
    return k, ci


def r3(x):
    return None if x is None or x != x else round(x, 3)


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)
