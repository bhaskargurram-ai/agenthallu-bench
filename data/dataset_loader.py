"""Dataset loader — builds the 500-task budget-optimized benchmark.

Sources:
  300 synthetic (from Days 1-2 task files)
  100 ToolBench (real API tasks, filtered to our domains)
  100 HaluEval (hallucination QA, wrapped with tool context)

Shuffle seed=42, split into 50 val / 450 test.
Save to data/tasks/full_500_tasks.json
"""

import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

from config import RANDOM_SEED

logger = logging.getLogger(__name__)
console = Console()

TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


def load_synthetic(limit: int = 300) -> list[dict]:
    """Load synthetic tasks from Days 1-2 JSON files (up to 300)."""
    tasks = []
    for domain in ["weather", "calendar", "medical"]:
        path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
        if os.path.exists(path):
            with open(path) as f:
                domain_tasks = json.load(f)
            # Take up to 100 per domain
            tasks.extend(domain_tasks[:100])
            logger.info("Loaded %d synthetic %s tasks", min(len(domain_tasks), 100), domain)
        else:
            logger.warning("Missing synthetic tasks: %s", path)

    if len(tasks) < limit:
        logger.warning("Only %d synthetic tasks found (wanted %d)", len(tasks), limit)

    return tasks[:limit]


def load_toolbench(limit: int = 100) -> list[dict]:
    """Load and adapt ToolBench tasks.

    Filters to tasks matching weather/calendar/medical keywords.
    Wraps them in our task format with appropriate tool mappings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[yellow]datasets library not installed. Run: pip install datasets[/yellow]")
        console.print("[yellow]Generating synthetic ToolBench-style tasks instead...[/yellow]")
        return _generate_toolbench_fallback(limit)

    try:
        console.print("[cyan]Loading ToolBench from HuggingFace...[/cyan]")
        ds = load_dataset("ToolBench/ToolBench", split="test[:500]")
    except Exception as e:
        logger.warning("ToolBench download failed: %s. Using fallback.", e)
        return _generate_toolbench_fallback(limit)

    # Domain keyword filters
    weather_kw = {"weather", "temperature", "forecast", "rain", "wind", "climate", "humidity"}
    calendar_kw = {"calendar", "event", "meeting", "schedule", "appointment", "reminder"}
    medical_kw = {"medical", "patient", "drug", "doctor", "health", "diagnosis", "prescription"}

    tasks = []
    rng = random.Random(RANDOM_SEED)

    for i, row in enumerate(ds):
        query = str(row.get("query", row.get("instruction", "")))
        query_lower = query.lower()

        # Classify domain
        domain = None
        if any(kw in query_lower for kw in weather_kw):
            domain = "weather"
        elif any(kw in query_lower for kw in calendar_kw):
            domain = "calendar"
        elif any(kw in query_lower for kw in medical_kw):
            domain = "medical"

        if domain is None:
            continue

        # Map to our tools
        tool_map = {
            "weather": ("get_weather", {"city": "Unknown", "date": "2024-01-15"}),
            "calendar": ("create_event", {"title": "Meeting", "date": "2024-01-15", "time": "10:00", "duration_minutes": 60}),
            "medical": ("get_patient_record", {"patient_id": "P001"}),
        }
        tool_name, default_params = tool_map[domain]

        tasks.append({
            "task_id": f"toolbench_{i:04d}",
            "domain": domain,
            "difficulty": rng.choice(["easy", "medium", "hard"]),
            "query": query[:500],
            "source": "toolbench",
            "required_tools": [tool_name],
            "correct_tool_sequence": [
                {"tool": tool_name, "params": default_params, "expected_result_keys": []}
            ],
            "correct_final_answer": f"ToolBench task: {query[:100]}",
            "multi_turn": False,
            "num_turns": 1,
        })

        if len(tasks) >= limit:
            break

    if len(tasks) < limit:
        logger.warning("Only %d ToolBench tasks matched domains (wanted %d). Adding fallbacks.", len(tasks), limit)
        tasks.extend(_generate_toolbench_fallback(limit - len(tasks)))

    console.print(f"  ToolBench: {len(tasks)} tasks loaded")
    return tasks[:limit]


def _generate_toolbench_fallback(limit: int) -> list[dict]:
    """Generate synthetic ToolBench-style tasks when download fails."""
    rng = random.Random(RANDOM_SEED + 1000)
    tasks = []

    templates = {
        "weather": [
            "What's the current weather in {city}?",
            "Is it going to rain in {city} tomorrow?",
            "Give me a {days}-day forecast for {city}.",
            "What was the temperature in {city} last week?",
        ],
        "calendar": [
            "Schedule a {title} for next Monday at {time}.",
            "What meetings do I have on {date}?",
            "Create a recurring weekly {title}.",
            "Cancel my {title} on {date}.",
        ],
        "medical": [
            "Look up patient {pid}'s medication list.",
            "Check if {drug_a} interacts with {drug_b}.",
            "Schedule a {type} appointment for patient {pid}.",
            "What are the side effects of {drug_a}?",
        ],
    }
    cities = ["London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Mumbai"]
    titles = ["Team Meeting", "Sprint Review", "Design Review", "1:1", "Workshop"]
    drugs = ["metformin", "lisinopril", "atorvastatin", "aspirin", "ibuprofen"]

    for i in range(limit):
        domain = rng.choice(["weather", "calendar", "medical"])
        template = rng.choice(templates[domain])

        query = template.format(
            city=rng.choice(cities),
            days=rng.randint(3, 7),
            title=rng.choice(titles),
            time=f"{rng.randint(8,17):02d}:00",
            date=f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            pid=f"P{rng.randint(1,5):03d}",
            drug_a=rng.choice(drugs),
            drug_b=rng.choice(drugs),
            type=rng.choice(["checkup", "follow_up", "specialist"]),
        )

        tool_map = {
            "weather": ("get_weather", {"city": "London", "date": "2024-01-15"}),
            "calendar": ("create_event", {"title": "Meeting", "date": "2024-01-15", "time": "10:00", "duration_minutes": 60}),
            "medical": ("get_patient_record", {"patient_id": "P001"}),
        }
        tool_name, params = tool_map[domain]

        tasks.append({
            "task_id": f"toolbench_syn_{i:04d}",
            "domain": domain,
            "difficulty": rng.choice(["easy", "medium", "hard"]),
            "query": query,
            "source": "toolbench_synthetic",
            "required_tools": [tool_name],
            "correct_tool_sequence": [
                {"tool": tool_name, "params": params, "expected_result_keys": []}
            ],
            "correct_final_answer": f"Completed: {query[:100]}",
            "multi_turn": False,
            "num_turns": 1,
        })

    return tasks


def load_halueval(limit: int = 100) -> list[dict]:
    """Load HaluEval QA tasks and wrap with search_knowledge tool.

    Filters to hallucination=yes cases, wraps each with tool context.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[yellow]datasets library not installed. Using fallback.[/yellow]")
        return _generate_halueval_fallback(limit)

    try:
        console.print("[cyan]Loading HaluEval from HuggingFace...[/cyan]")
        ds = load_dataset("pminervini/HaluEval", "qa", split="data[:300]")
    except Exception as e:
        logger.warning("HaluEval download failed: %s. Using fallback.", e)
        return _generate_halueval_fallback(limit)

    tasks = []
    rng = random.Random(RANDOM_SEED + 2000)

    for i, row in enumerate(ds):
        # HaluEval has hallucinated_answer field — if present, it's a hallucination case
        has_hallucinated = bool(row.get("hallucinated_answer", ""))
        if not has_hallucinated:
            continue

        question = str(row.get("question", ""))
        knowledge = str(row.get("knowledge", ""))
        answer = str(row.get("answer", ""))
        hallucinated_answer = str(row.get("hallucinated_answer", ""))

        if not question:
            continue

        # Wrap with search_knowledge tool
        tasks.append({
            "task_id": f"halueval_{i:04d}",
            "domain": "knowledge",
            "difficulty": rng.choice(["medium", "hard"]),
            "query": f"Use the search_knowledge tool to answer: {question}",
            "source": "halueval",
            "required_tools": ["search_knowledge"],
            "correct_tool_sequence": [
                {
                    "tool": "search_knowledge",
                    "params": {"query": question},
                    "expected_result_keys": ["answer", "source"],
                    "context": knowledge[:500],
                }
            ],
            "correct_final_answer": answer[:500],
            "hallucinated_answer": hallucinated_answer[:500],
            "multi_turn": False,
            "num_turns": 1,
        })

        if len(tasks) >= limit:
            break

    if len(tasks) < limit:
        logger.warning("Only %d HaluEval tasks matched (wanted %d). Adding fallbacks.", len(tasks), limit)
        tasks.extend(_generate_halueval_fallback(limit - len(tasks)))

    console.print(f"  HaluEval: {len(tasks)} tasks loaded")
    return tasks[:limit]


def _generate_halueval_fallback(limit: int) -> list[dict]:
    """Generate synthetic HaluEval-style tasks when download fails."""
    rng = random.Random(RANDOM_SEED + 3000)
    tasks = []

    qa_pairs = [
        ("What is the capital of France?", "Paris is the capital of France.", "Lyon is the capital of France."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.", "Charles Dickens wrote Romeo and Juliet."),
        ("What is the boiling point of water?", "Water boils at 100°C at standard pressure.", "Water boils at 90°C at standard pressure."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet.", "Saturn is the largest planet."),
        ("Who discovered penicillin?", "Alexander Fleming discovered penicillin.", "Louis Pasteur discovered penicillin."),
        ("What year did World War II end?", "World War II ended in 1945.", "World War II ended in 1944."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 km/s.", "The speed of light is approximately 150,000 km/s."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O.", "The chemical formula for water is H2O2."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa.", "Michelangelo painted the Mona Lisa."),
        ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain.", "K2 is the tallest mountain."),
    ]

    for i in range(limit):
        question, correct, hallucinated = qa_pairs[i % len(qa_pairs)]
        # Add variation
        if i >= len(qa_pairs):
            question = f"Question variant {i}: {question}"

        tasks.append({
            "task_id": f"halueval_syn_{i:04d}",
            "domain": "knowledge",
            "difficulty": rng.choice(["medium", "hard"]),
            "query": f"Use the search_knowledge tool to answer: {question}",
            "source": "halueval_synthetic",
            "required_tools": ["search_knowledge"],
            "correct_tool_sequence": [
                {
                    "tool": "search_knowledge",
                    "params": {"query": question},
                    "expected_result_keys": ["answer"],
                }
            ],
            "correct_final_answer": correct,
            "hallucinated_answer": hallucinated,
            "multi_turn": False,
            "num_turns": 1,
        })

    return tasks


def load_budget_optimized(seed: int = RANDOM_SEED) -> list[dict]:
    """Load exactly 500 tasks optimized for budget experiments.

    300 synthetic + 100 ToolBench + 100 HaluEval
    Shuffle with seed, split 50 val / 450 test.
    Save to data/tasks/full_500_tasks.json
    """
    console.print("\n[bold cyan]Loading Budget-Optimized 500-Task Dataset[/bold cyan]\n")

    # Load each source
    synthetic = load_synthetic(300)
    # Tag source
    for t in synthetic:
        t.setdefault("source", "synthetic")

    toolbench = load_toolbench(100)
    halueval = load_halueval(100)

    # Combine
    all_tasks = synthetic + toolbench + halueval
    console.print(f"\nTotal tasks before shuffle: {len(all_tasks)}")

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(all_tasks)

    # Split: 50 val, rest test
    val_tasks = all_tasks[:50]
    test_tasks = all_tasks[50:]

    # Tag split
    for t in val_tasks:
        t["split"] = "val"
    for t in test_tasks:
        t["split"] = "test"

    all_tasks = val_tasks + test_tasks

    # Save
    output_path = os.path.join(TASKS_DIR, "full_500_tasks.json")
    with open(output_path, "w") as f:
        json.dump(all_tasks, f, indent=2)
    console.print(f"[green]Saved {len(all_tasks)} tasks to {output_path}[/green]")

    # Print breakdown table
    table = Table(title="Dataset Breakdown")
    table.add_column("Source")
    table.add_column("Count", justify="right")
    table.add_column("Domains")
    table.add_column("Split (val/test)", justify="right")

    from collections import Counter

    source_counts = Counter(t.get("source", "synthetic") for t in all_tasks)
    for source, count in source_counts.most_common():
        domains = set(t["domain"] for t in all_tasks if t.get("source", "synthetic") == source)
        val_count = sum(1 for t in val_tasks if t.get("source", "synthetic") == source)
        test_count = sum(1 for t in test_tasks if t.get("source", "synthetic") == source)
        table.add_row(
            source,
            str(count),
            ", ".join(sorted(domains)),
            f"{val_count} / {test_count}",
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(all_tasks)}[/bold]",
        "—",
        f"[bold]{len(val_tasks)} / {len(test_tasks)}[/bold]",
        style="bold green",
    )
    console.print(table)

    return all_tasks


if __name__ == "__main__":
    load_budget_optimized()
