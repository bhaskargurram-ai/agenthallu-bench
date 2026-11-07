"""C2_04: inventory ORIGINAL project vs GitHub repo src. Hash compare; CSV row/column diffs. Writes data_inventory.csv."""
import os, hashlib, csv, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import ORIG, REPO, OUT
csv.field_size_limit(sys.maxsize)
SKIP = {"venv", ".git", "__pycache__", ".pytest_cache", "node_modules", ".claude", "agentprop_bench.egg-info"}


def walk(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        for f in fn:
            if f == ".DS_Store":
                continue
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = p
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def csvinfo(p):
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f)
            hdr = next(r, [])
            n = sum(1 for _ in r)
        return hdr, n
    except Exception as e:
        return [f"ERR {e}"], -1


O, R = walk(ORIG), walk(REPO)
rows = []
for rel in sorted(set(O) | set(R)):
    top = rel.split(os.sep)[0]
    in_o, in_r = rel in O, rel in R
    st = "both"
    note = ""
    if in_o and not in_r:
        st = "original_only"
    elif in_r and not in_o:
        st = "repo_only"
    else:
        so, sr = os.path.getsize(O[rel]), os.path.getsize(R[rel])
        if so == sr and sha(O[rel]) == sha(R[rel]):
            st = "identical"
        else:
            st = "differs"
            if rel.endswith(".csv"):
                ho, no = csvinfo(O[rel]); hr, nr = csvinfo(R[rel])
                add = [c for c in ho if c not in hr]; rem = [c for c in hr if c not in ho]
                note = f"rows orig={no} repo={nr}; cols only_orig={add} only_repo={rem}"
            else:
                note = f"bytes orig={so} repo={sr}"
    size = os.path.getsize(O[rel]) if in_o else os.path.getsize(R[rel])
    rows.append({"relpath": rel, "top_dir": top, "status": st, "bytes": size,
                 "mtime_orig": int(os.path.getmtime(O[rel])) if in_o else "",
                 "is_data": rel.endswith((".csv", ".json", ".jsonl", ".db", ".parquet")) and not rel.endswith("package.json"),
                 "note": note})
with open(f"{OUT}/data_inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
c = collections.Counter((r["status"], r["top_dir"]) for r in rows)
summ = collections.Counter(r["status"] for r in rows)
print(summ)
for (s, t), v in sorted(c.items()):
    print(f"{s:14s} {t:30s} {v}")
print("\n--- DATA files original_only (data/, analysis/, hf_dataset/, experiments/ json/csv/db) ---")
for r in rows:
    if r["status"] == "original_only" and r["is_data"] and r["top_dir"] in ("data", "analysis", "hf_dataset", "experiments", "leaderboard"):
        print(r["relpath"], r["bytes"])
print("\n--- DATA files differing ---")
for r in rows:
    if r["status"] == "differs" and r["is_data"]:
        print(r["relpath"], r["note"])
print("\n--- DATA files repo_only ---")
for r in rows:
    if r["status"] == "repo_only" and r["is_data"]:
        print(r["relpath"], r["bytes"])
