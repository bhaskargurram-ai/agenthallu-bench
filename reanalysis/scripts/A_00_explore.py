import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0, _REPO + "/reanalysis/scripts")
from A_common import *
from collections import Counter, defaultdict
for fn in P2_FILES + BL_FILES:
    rows = read(fn)
    print("==", fn, len(rows), "cols:", len(rows[0]) if rows else 0)
    print("  models:", Counter(r["model"] for r in rows))
    if "error_type" in rows[0]:
        print("  error_types:", Counter(r["error_type"] for r in rows))
    print("  domains:", Counter(r["domain"] for r in rows))
    sids = Counter(r["session_id"] for r in rows); print("  dup session ids:", sum(v-1 for v in sids.values()))
    key = Counter((r["task_id"], r["model"], r.get("error_type","")) for r in rows)
    print("  rows per (task,model,et): ", Counter(key.values()))
