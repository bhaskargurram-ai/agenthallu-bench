"""A1: Scale up the task suite from 500 to ~2000 tasks.

Strategy: Run the existing generators with multiple seeds and merge,
deduplicating by (domain, query) to ensure variety. Also re-label splits
so ~10% val, ~90% test, and the first 500 are preserved for reproducibility.

Output: data/tasks/full_2000_tasks.json
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

from benchmark.domains.weather_api import generate_all_tasks as gen_weather
from benchmark.domains.calendar_api import generate_all_tasks as gen_calendar
from benchmark.domains.medical_api import generate_all_tasks as gen_medical

console = Console()
logger = logging.getLogger(__name__)

MULTI_TURN_FRACTION = 0.30
MULTI_TURN_NUM_TURNS = 6


def fingerprint(task: dict) -> str:
    key = f"{task.get('domain')}|{task.get('query','').strip().lower()}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=6,
                        help="Number of distinct seeds to draw from. ~450 tasks per seed.")
    parser.add_argument("--output", type=str, default="data/tasks/full_2000_tasks.json")
    parser.add_argument("--preserve", type=str, default="data/tasks/full_500_tasks.json",
                        help="Preserve task_ids and order of this file at the top.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    base = os.path.join(os.path.dirname(__file__), "..")
    out_path = os.path.join(base, args.output)
    preserve_path = os.path.join(base, args.preserve)

    all_tasks = []
    seen = set()

    # Preserve the original 500 first (stable IDs)
    if os.path.exists(preserve_path):
        with open(preserve_path) as f:
            preserved = json.load(f)
        for t in preserved:
            fp = fingerprint(t)
            if fp in seen:
                continue
            seen.add(fp)
            all_tasks.append(t)
        console.print(f"Preserved [bold]{len(all_tasks)}[/bold] original tasks")

    # Generate fresh batches with distinct seeds
    for si, seed in enumerate([42 + 1000 * (k + 1) for k in range(args.seeds)]):
        rng = random.Random(seed)
        batch = []
        batch += gen_weather(rng)
        batch += gen_calendar(rng)
        batch += gen_medical(rng)
        rng2 = random.Random(seed + 7)
        rng2.shuffle(batch)

        before = len(all_tasks)
        for t in batch:
            fp = fingerprint(t)
            if fp in seen:
                continue
            # Re-assign a global task_id that does not clash
            t = dict(t)
            t["task_id"] = f"{t['domain']}_s{si+1}_{len(all_tasks):04d}"
            seen.add(fp)
            all_tasks.append(t)
            if len(all_tasks) >= args.target:
                break
        console.print(f"seed={seed}: +{len(all_tasks) - before} unique (total {len(all_tasks)})")
        if len(all_tasks) >= args.target:
            break

    all_tasks = all_tasks[: args.target]

    # Multi-turn flags (re-run on additions only, leaving preserved flags intact)
    rng_mt = random.Random(7)
    for t in all_tasks:
        if t.get("multi_turn") is None:
            t["multi_turn"] = rng_mt.random() < MULTI_TURN_FRACTION
            t["num_turns"] = MULTI_TURN_NUM_TURNS if t["multi_turn"] else 1

    # Splits: preserve existing, mark new as test (90%) or val (10%)
    rng_sp = random.Random(11)
    for t in all_tasks:
        if not t.get("split"):
            t["split"] = "val" if rng_sp.random() < 0.10 else "test"

    # Summary
    from collections import Counter
    dom = Counter(t.get("domain") for t in all_tasks)
    diff = Counter(t.get("difficulty") for t in all_tasks)
    sp = Counter(t.get("split") for t in all_tasks)
    console.print(f"[bold]Total: {len(all_tasks)}[/bold]")
    console.print(f"Domains: {dict(dom)}")
    console.print(f"Difficulty: {dict(diff)}")
    console.print(f"Splits: {dict(sp)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_tasks, f, indent=2)
    console.print(f"[green]Saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
