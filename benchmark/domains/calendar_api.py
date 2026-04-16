"""Calendar domain task templates for benchmark generation.

Produces 150 tasks: 50 easy, 50 medium, 50 hard.
"""

import random

TITLES = [
    "Team Standup", "Sprint Review", "1:1 with Manager", "Design Review",
    "Client Call", "Lunch Meeting", "Board Meeting", "Code Review Session",
    "Product Demo", "Budget Review", "Training Session", "Interview",
    "Strategy Meeting", "All Hands", "Retrospective",
]

DATES = [
    "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-08", "2024-07-10",
    "2024-07-15", "2024-07-22", "2024-08-01", "2024-08-05", "2024-08-12",
    "2024-08-20", "2024-09-01",
]

TIMES = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]

ATTENDEES = [
    "alice@company.com", "bob@company.com", "carol@company.com",
    "dave@company.com", "eve@company.com", "frank@company.com",
]


def generate_easy_tasks(rng: random.Random, start_id: int = 0) -> list[dict]:
    """50 easy tasks: single tool call."""
    tasks = []
    templates = [
        {
            "query": "Create a meeting called '{title}' on {date} at {time} for {dur} minutes.",
            "tool": "create_event",
            "params_fn": lambda t, d, tm, dur, r: {"title": t, "date": d, "time": tm, "duration_minutes": dur},
            "result_keys": ["event_id", "status"],
            "answer_fn": lambda t, d: f"Created event '{t}' on {d}.",
        },
        {
            "query": "What meetings do I have on {date}?",
            "tool": "get_events",
            "params_fn": lambda t, d, tm, dur, r: {"date": d},
            "result_keys": ["events"],
            "answer_fn": lambda t, d: f"Events on {d}.",
        },
        {
            "query": "Schedule a {dur}-minute '{title}' for {date} at {time}.",
            "tool": "create_event",
            "params_fn": lambda t, d, tm, dur, r: {"title": t, "date": d, "time": tm, "duration_minutes": dur},
            "result_keys": ["event_id", "status"],
            "answer_fn": lambda t, d: f"Scheduled '{t}' on {d}.",
        },
        {
            "query": "Show me my calendar for {date}.",
            "tool": "get_events",
            "params_fn": lambda t, d, tm, dur, r: {"date": d},
            "result_keys": ["events"],
            "answer_fn": lambda t, d: f"Calendar for {d}.",
        },
        {
            "query": "Delete event {event_id}.",
            "tool": "delete_event",
            "params_fn": lambda t, d, tm, dur, r: {"event_id": f"EVT_{r.randint(1000,9999)}"},
            "result_keys": ["success"],
            "answer_fn": lambda t, d: "Event deleted.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        title = rng.choice(TITLES)
        date = rng.choice(DATES)
        time_ = rng.choice(TIMES)
        dur = rng.choice([30, 45, 60, 90])
        event_id = f"EVT_{rng.randint(1000, 9999)}"
        params = t["params_fn"](title, date, time_, dur, rng)

        tasks.append({
            "task_id": f"calendar_{start_id + i + 1:03d}",
            "domain": "calendar",
            "difficulty": "easy",
            "query": t["query"].format(title=title, date=date, time=time_, dur=dur, event_id=event_id),
            "required_tools": [t["tool"]],
            "correct_tool_sequence": [{
                "tool": t["tool"],
                "params": params,
                "expected_result_keys": t["result_keys"],
            }],
            "correct_final_answer": t["answer_fn"](title, date),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": [],
        })
    return tasks


def generate_medium_tasks(rng: random.Random, start_id: int = 50) -> list[dict]:
    """50 medium tasks: 2 tool calls with dependency."""
    tasks = []
    templates = [
        {
            "query": "Create a '{title}' meeting on {date} at {time} for {dur} minutes with {att1} and {att2}, then show me all events for that day.",
            "tools": ["create_event", "get_events"],
            "params_fn": lambda t, d, tm, dur, atts, r: [
                {"title": t, "date": d, "time": tm, "duration_minutes": dur, "attendees": atts},
                {"date": d},
            ],
            "result_keys": [["event_id", "status"], ["events"]],
            "answer_fn": lambda t, d: f"Created '{t}' and listed all events on {d}.",
        },
        {
            "query": "Check my schedule for {date} and if the {time} slot is free, create a '{title}' for {dur} minutes.",
            "tools": ["get_events", "create_event"],
            "params_fn": lambda t, d, tm, dur, atts, r: [
                {"date": d},
                {"title": t, "date": d, "time": tm, "duration_minutes": dur},
            ],
            "result_keys": [["events"], ["event_id"]],
            "answer_fn": lambda t, d: f"Checked schedule and scheduled '{t}' on {d}.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        title = rng.choice(TITLES)
        date = rng.choice(DATES)
        time_ = rng.choice(TIMES)
        dur = rng.choice([30, 45, 60])
        atts = rng.sample(ATTENDEES, 2)
        params_list = t["params_fn"](title, date, time_, dur, atts, rng)

        tasks.append({
            "task_id": f"calendar_{start_id + i + 1:03d}",
            "domain": "calendar",
            "difficulty": "medium",
            "query": t["query"].format(
                title=title, date=date, time=time_, dur=dur,
                att1=atts[0], att2=atts[1],
            ),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {
                    "tool": tool,
                    "params": params,
                    "expected_result_keys": keys,
                }
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](title, date),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": [],
        })
    return tasks


def generate_hard_tasks(rng: random.Random, start_id: int = 100) -> list[dict]:
    """50 hard tasks: 3+ tool calls, multi-step reasoning."""
    tasks = []
    templates = [
        {
            "query": "I need to reorganize my {date} schedule. First check what's there, then delete event {eid}, and create a new '{title}' at {time} for {dur} minutes with {att1}, {att2}, and {att3}.",
            "tools": ["get_events", "delete_event", "create_event"],
            "params_fn": lambda t, d, tm, dur, atts, eid, r: [
                {"date": d},
                {"event_id": eid},
                {"title": t, "date": d, "time": tm, "duration_minutes": dur, "attendees": atts},
            ],
            "result_keys": [["events"], ["success"], ["event_id"]],
            "answer_fn": lambda t, d: f"Reorganized {d} schedule with new '{t}'.",
            "rag_docs": ["calendar_000", "calendar_001"],
        },
        {
            "query": "Create three meetings on {date}: '{title1}' at 09:00 for 30 min, '{title2}' at 11:00 for 60 min, and '{title3}' at 14:00 for 45 min.",
            "tools": ["create_event", "create_event", "create_event"],
            "params_fn": lambda t, d, tm, dur, atts, eid, r: [
                {"title": t, "date": d, "time": "09:00", "duration_minutes": 30},
                {"title": TITLES[(TITLES.index(t) + 1) % len(TITLES)], "date": d, "time": "11:00", "duration_minutes": 60},
                {"title": TITLES[(TITLES.index(t) + 2) % len(TITLES)], "date": d, "time": "14:00", "duration_minutes": 45},
            ],
            "result_keys": [["event_id"], ["event_id"], ["event_id"]],
            "answer_fn": lambda t, d: f"Created three meetings on {d}.",
            "rag_docs": ["calendar_002"],
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        title = rng.choice(TITLES)
        title1 = title
        title2 = TITLES[(TITLES.index(title) + 1) % len(TITLES)]
        title3 = TITLES[(TITLES.index(title) + 2) % len(TITLES)]
        date = rng.choice(DATES)
        time_ = rng.choice(TIMES)
        dur = rng.choice([30, 45, 60, 90])
        atts = rng.sample(ATTENDEES, 3)
        eid = f"EVT_{rng.randint(1000, 9999)}"
        params_list = t["params_fn"](title, date, time_, dur, atts, eid, rng)

        tasks.append({
            "task_id": f"calendar_{start_id + i + 1:03d}",
            "domain": "calendar",
            "difficulty": "hard",
            "query": t["query"].format(
                title=title, title1=title1, title2=title2, title3=title3,
                date=date, time=time_, dur=dur, eid=eid,
                att1=atts[0], att2=atts[1], att3=atts[2],
            ),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {
                    "tool": tool,
                    "params": params,
                    "expected_result_keys": keys,
                }
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](title, date),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": t.get("rag_docs", []),
        })
    return tasks


def generate_all_tasks(rng: random.Random) -> list[dict]:
    """Generate all 150 calendar domain tasks."""
    tasks = []
    tasks.extend(generate_easy_tasks(rng, start_id=0))
    tasks.extend(generate_medium_tasks(rng, start_id=50))
    tasks.extend(generate_hard_tasks(rng, start_id=100))
    return tasks
