"""Weather domain task templates for benchmark generation.

Produces 150 tasks: 50 easy, 50 medium, 50 hard.
Easy = single tool call. Medium = 2 tool calls. Hard = 3+ tools + RAG.
"""

import random
from typing import Any

CITIES = [
    "New York", "London", "Tokyo", "Paris", "Sydney", "Mumbai", "Berlin",
    "Toronto", "São Paulo", "Cairo", "Seoul", "Moscow", "Dubai", "Singapore",
    "Bangkok", "Rome", "Istanbul", "Beijing", "Mexico City", "Lagos",
]

DATES = [
    "2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-22",
    "2024-06-15", "2024-07-08", "2024-08-19", "2024-09-30", "2024-10-12",
    "2024-11-25", "2024-12-01",
]


def generate_easy_tasks(rng: random.Random, start_id: int = 0) -> list[dict]:
    """50 easy tasks: single tool call with simple params."""
    tasks = []
    templates = [
        {
            "query": "What is the weather in {city} on {date}?",
            "tool": "get_weather",
            "params_fn": lambda c, d, r: {"city": c, "date": d},
            "result_keys": ["temperature", "humidity", "condition"],
            "answer_fn": lambda c, d: f"The weather in {c} on {d} includes temperature, humidity, and condition data.",
        },
        {
            "query": "What's the temperature in {city} on {date} in fahrenheit?",
            "tool": "get_weather",
            "params_fn": lambda c, d, r: {"city": c, "date": d, "unit": "fahrenheit"},
            "result_keys": ["temperature", "humidity", "condition"],
            "answer_fn": lambda c, d: f"The temperature in {c} on {d} in fahrenheit.",
        },
        {
            "query": "Give me a {days}-day forecast for {city}.",
            "tool": "get_forecast",
            "params_fn": lambda c, d, r: {"city": c, "days": r.choice([3, 5, 7])},
            "result_keys": ["forecast"],
            "answer_fn": lambda c, d: f"Forecast for {c}.",
        },
        {
            "query": "What events are on {city}'s weather station for {date}? I mean weather conditions.",
            "tool": "get_weather",
            "params_fn": lambda c, d, r: {"city": c, "date": d},
            "result_keys": ["temperature", "humidity", "condition"],
            "answer_fn": lambda c, d: f"Weather conditions in {c} on {d}.",
        },
        {
            "query": "Show me the hourly forecast for {city} for the next day.",
            "tool": "get_forecast",
            "params_fn": lambda c, d, r: {"city": c, "days": 1, "include_hourly": True},
            "result_keys": ["forecast"],
            "answer_fn": lambda c, d: f"Hourly forecast for {c}.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        city = rng.choice(CITIES)
        date = rng.choice(DATES)
        days = rng.choice([3, 5, 7])
        params = t["params_fn"](city, date, rng)

        tasks.append({
            "task_id": f"weather_{start_id + i + 1:03d}",
            "domain": "weather",
            "difficulty": "easy",
            "query": t["query"].format(city=city, date=date, days=days),
            "required_tools": [t["tool"]],
            "correct_tool_sequence": [{
                "tool": t["tool"],
                "params": params,
                "expected_result_keys": t["result_keys"],
            }],
            "correct_final_answer": t["answer_fn"](city, date),
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
            "query": "What's the weather in {city} on {date}, and also give me a {days}-day forecast?",
            "tools": ["get_weather", "get_forecast"],
            "params_fn": lambda c, d, r: [
                {"city": c, "date": d},
                {"city": c, "days": r.choice([3, 5, 7])},
            ],
            "result_keys": [["temperature", "humidity", "condition"], ["forecast"]],
            "answer_fn": lambda c, d: f"Current weather and forecast for {c}.",
        },
        {
            "query": "Compare today's weather in {city} ({date}) with the historical average from {start} to {end}.",
            "tools": ["get_weather", "get_historical"],
            "params_fn": lambda c, d, r: [
                {"city": c, "date": d},
                {"city": c, "start_date": "2023-01-01", "end_date": "2023-01-07"},
            ],
            "result_keys": [["temperature"], ["data"]],
            "answer_fn": lambda c, d: f"Current vs historical weather for {c}.",
        },
        {
            "query": "Get the weather for {city} on {date} and check if it will change in the next {days} days.",
            "tools": ["get_weather", "get_forecast"],
            "params_fn": lambda c, d, r: [
                {"city": c, "date": d},
                {"city": c, "days": r.choice([3, 5])},
            ],
            "result_keys": [["temperature", "condition"], ["forecast"]],
            "answer_fn": lambda c, d: f"Weather analysis for {c}.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        city = rng.choice(CITIES)
        date = rng.choice(DATES)
        days = rng.choice([3, 5, 7])
        params_list = t["params_fn"](city, date, rng)

        tasks.append({
            "task_id": f"weather_{start_id + i + 1:03d}",
            "domain": "weather",
            "difficulty": "medium",
            "query": t["query"].format(
                city=city, date=date, days=days,
                start="2023-01-01", end="2023-01-07",
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
            "correct_final_answer": t["answer_fn"](city, date),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": [],
        })
    return tasks


def generate_hard_tasks(rng: random.Random, start_id: int = 100) -> list[dict]:
    """50 hard tasks: 3+ tool calls, RAG context needed, multi-step reasoning."""
    tasks = []
    templates = [
        {
            "query": "I'm planning a trip to {city}. Get today's weather ({date}), a {days}-day forecast, and historical data for last January to understand climate patterns.",
            "tools": ["get_weather", "get_forecast", "get_historical"],
            "params_fn": lambda c, d, r: [
                {"city": c, "date": d},
                {"city": c, "days": r.choice([7, 10, 14])},
                {"city": c, "start_date": "2024-01-01", "end_date": "2024-01-31"},
            ],
            "result_keys": [["temperature", "condition"], ["forecast"], ["data"]],
            "answer_fn": lambda c, d: f"Comprehensive weather analysis for trip to {c}.",
            "rag_docs": ["weather_000", "weather_001"],
        },
        {
            "query": "Compare the current weather in {city} and {city2} on {date}, then get forecasts for both cities for the next {days} days.",
            "tools": ["get_weather", "get_weather", "get_forecast", "get_forecast"],
            "params_fn": lambda c, d, r: [
                {"city": c, "date": d},
                {"city": CITIES[(CITIES.index(c) + 1) % len(CITIES)], "date": d},
                {"city": c, "days": r.choice([5, 7])},
                {"city": CITIES[(CITIES.index(c) + 1) % len(CITIES)], "days": r.choice([5, 7])},
            ],
            "result_keys": [["temperature"], ["temperature"], ["forecast"], ["forecast"]],
            "answer_fn": lambda c, d: f"Comparative weather analysis for {c} and another city.",
            "rag_docs": ["weather_002", "weather_003"],
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        city = rng.choice(CITIES)
        city2 = CITIES[(CITIES.index(city) + 1) % len(CITIES)]
        date = rng.choice(DATES)
        days = rng.choice([5, 7, 10, 14])
        params_list = t["params_fn"](city, date, rng)

        tasks.append({
            "task_id": f"weather_{start_id + i + 1:03d}",
            "domain": "weather",
            "difficulty": "hard",
            "query": t["query"].format(city=city, city2=city2, date=date, days=days),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {
                    "tool": tool,
                    "params": params,
                    "expected_result_keys": keys,
                }
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](city, date),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": t.get("rag_docs", []),
        })
    return tasks


def generate_all_tasks(rng: random.Random) -> list[dict]:
    """Generate all 150 weather domain tasks."""
    tasks = []
    tasks.extend(generate_easy_tasks(rng, start_id=0))
    tasks.extend(generate_medium_tasks(rng, start_id=50))
    tasks.extend(generate_hard_tasks(rng, start_id=100))
    return tasks
