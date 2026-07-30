"""Task loading for the AgentProp-Bench harness.

Loads the released tasks from the repo (or a user-supplied JSON path) and lets the
CLI slice by domain / split / count so a quick `agentprop eval` runs on a small
comparable subset by default.
"""
import json
import os


def _repo_root():
    """Repo root = directory containing config.py (works for `pip install -e .`)."""
    try:
        import config
        return os.path.dirname(os.path.abspath(config.__file__))
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_tasks_path():
    root = _repo_root()
    for name in ("full_500_tasks.json", "full_2000_tasks.json"):
        p = os.path.join(root, "data", "tasks", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No bundled task file found under data/tasks/. Pass --tasks.")


def load_tasks(path=None, split=None, domain=None, n=None):
    """Return a list of task dicts, optionally filtered/truncated.

    path:   JSON file of tasks (defaults to the bundled 500-task set).
    split:  keep only tasks with this split value ("test"/"val"); None = all.
    domain: keep only this domain; None = all.
    n:      cap to the first n tasks after filtering.
    """
    path = path or default_tasks_path()
    with open(path) as f:
        tasks = json.load(f)
    if split:
        tasks = [t for t in tasks if t.get("split", "test") == split]
    if domain:
        tasks = [t for t in tasks if t.get("domain") == domain]
    if n:
        tasks = tasks[: int(n)]
    return tasks
