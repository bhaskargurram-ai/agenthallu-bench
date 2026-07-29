"""Generate 300 retail/e-commerce tasks (2 seeds × 150)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.domains.ecommerce_api import generate_all_tasks

out = []
for si, seed in enumerate([42, 1042]):
    rng = random.Random(seed)
    batch = generate_all_tasks(rng)
    for t in batch:
        t = dict(t)
        t["task_id"] = f"ecommerce_s{si+1}_{len(out):04d}"
        t["split"] = "val" if len(out) < 30 else "test"
        out.append(t)

from collections import Counter
print("Total:", len(out))
print("Difficulty:", Counter(t["difficulty"] for t in out))
print("Splits:", Counter(t["split"] for t in out))

path = os.path.join(os.path.dirname(__file__), "..", "data", "tasks", "ecommerce_300_tasks.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("Saved:", path)
