"""E-commerce / Retail domain task templates.

Produces 150 tasks: 50 easy, 50 medium, 50 hard.
Easy = single tool call. Medium = 2 tools (search → order, etc).
Hard = 3+ tools (search, inspect, order with multiple items).
"""

import random

QUERIES = [
    "wireless headphones", "python book", "yoga mat", "running shoes",
    "coffee mug", "bluetooth speaker", "smart watch", "desk lamp",
    "tennis racket", "water bottle", "winter jacket", "laptop stand",
    "ceramic pan", "resistance bands", "knife set",
]

CATEGORIES = ["electronics", "books", "clothing", "home", "sports"]

ADDRESSES = [
    "123 Main St, San Francisco, CA 94110",
    "456 Oak Ave, Seattle, WA 98101",
    "789 Pine Rd, Austin, TX 78701",
    "321 Maple Ln, Boston, MA 02108",
    "654 Birch Dr, Denver, CO 80202",
    "987 Cedar Blvd, Chicago, IL 60601",
]


def generate_easy_tasks(rng: random.Random, start_id: int = 0) -> list[dict]:
    tasks = []
    templates = [
        {
            "query": "Search the catalog for '{q}'.",
            "tool": "search_products",
            "params_fn": lambda q, c, a, r: {"query": q},
            "result_keys": ["results", "total_matches"],
            "answer_fn": lambda q, c: f"Search results for '{q}'.",
        },
        {
            "query": "Find {cat} items matching '{q}' under $200.",
            "tool": "search_products",
            "params_fn": lambda q, c, a, r: {"query": q, "category": c, "max_price": 200},
            "result_keys": ["results"],
            "answer_fn": lambda q, c: f"{c.title()} matches for '{q}' under $200.",
        },
        {
            "query": "Get details for product {pid}.",
            "tool": "get_product",
            "params_fn": lambda q, c, a, r: {"product_id": f"PRD_{r.randint(1000, 9999)}"},
            "result_keys": ["name", "price_usd", "rating"],
            "answer_fn": lambda q, c: "Product details retrieved.",
        },
        {
            "query": "Show me info on product {pid}.",
            "tool": "get_product",
            "params_fn": lambda q, c, a, r: {"product_id": f"PRD_{r.randint(1000, 9999)}"},
            "result_keys": ["name", "price_usd"],
            "answer_fn": lambda q, c: "Product details.",
        },
        {
            "query": "Order 1 unit of product {pid} to {addr}.",
            "tool": "place_order",
            "params_fn": lambda q, c, a, r: {
                "product_id": f"PRD_{r.randint(1000, 9999)}",
                "quantity": 1,
                "shipping_address": a,
            },
            "result_keys": ["order_id", "status"],
            "answer_fn": lambda q, c: "Order placed.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        q = rng.choice(QUERIES)
        c = rng.choice(CATEGORIES)
        a = rng.choice(ADDRESSES)
        pid = f"PRD_{rng.randint(1000, 9999)}"
        params = t["params_fn"](q, c, a, rng)

        tasks.append({
            "task_id": f"ecommerce_{start_id + i + 1:03d}",
            "domain": "ecommerce",
            "difficulty": "easy",
            "query": t["query"].format(q=q, cat=c, pid=pid, addr=a),
            "required_tools": [t["tool"]],
            "correct_tool_sequence": [{
                "tool": t["tool"],
                "params": params,
                "expected_result_keys": t["result_keys"],
            }],
            "correct_final_answer": t["answer_fn"](q, c),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": [],
        })
    return tasks


def generate_medium_tasks(rng: random.Random, start_id: int = 50) -> list[dict]:
    tasks = []
    templates = [
        {
            "query": "Search for '{q}' in {cat}, then get details on the first result.",
            "tools": ["search_products", "get_product"],
            "params_fn": lambda q, c, a, pid, r: [
                {"query": q, "category": c},
                {"product_id": pid},
            ],
            "result_keys": [["results"], ["name", "price_usd"]],
            "answer_fn": lambda q, c: f"Searched '{q}' in {c} and retrieved first result.",
        },
        {
            "query": "Get details on product {pid}, then if it is in stock order 1 to {addr}.",
            "tools": ["get_product", "place_order"],
            "params_fn": lambda q, c, a, pid, r: [
                {"product_id": pid},
                {"product_id": pid, "quantity": 1, "shipping_address": a},
            ],
            "result_keys": [["in_stock", "price_usd"], ["order_id"]],
            "answer_fn": lambda q, c: "Checked stock and placed order.",
        },
        {
            "query": "Search for '{q}' in {cat}, and order 2 of product {pid} to {addr}.",
            "tools": ["search_products", "place_order"],
            "params_fn": lambda q, c, a, pid, r: [
                {"query": q, "category": c},
                {"product_id": pid, "quantity": 2, "shipping_address": a, "payment_method": "paypal"},
            ],
            "result_keys": [["results"], ["order_id"]],
            "answer_fn": lambda q, c: f"Searched and ordered product.",
        },
    ]
    for i in range(50):
        t = templates[i % len(templates)]
        q = rng.choice(QUERIES)
        c = rng.choice(CATEGORIES)
        a = rng.choice(ADDRESSES)
        pid = f"PRD_{rng.randint(1000, 9999)}"
        params_list = t["params_fn"](q, c, a, pid, rng)

        tasks.append({
            "task_id": f"ecommerce_{start_id + i + 1:03d}",
            "domain": "ecommerce",
            "difficulty": "medium",
            "query": t["query"].format(q=q, cat=c, pid=pid, addr=a),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {"tool": tool, "params": params, "expected_result_keys": keys}
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](q, c),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": [],
        })
    return tasks


def generate_hard_tasks(rng: random.Random, start_id: int = 100) -> list[dict]:
    tasks = []
    templates = [
        {
            "query": "I want to buy a {q}. Search for '{q}' in {cat}, check details on product {pid}, "
                     "and if it's in stock and costs under $300, order 1 to {addr} with paypal.",
            "tools": ["search_products", "get_product", "place_order"],
            "params_fn": lambda q, c, a, pid, r: [
                {"query": q, "category": c, "max_price": 300},
                {"product_id": pid},
                {"product_id": pid, "quantity": 1, "shipping_address": a, "payment_method": "paypal"},
            ],
            "result_keys": [["results"], ["in_stock", "price_usd"], ["order_id"]],
            "answer_fn": lambda q, c: f"Bought {q} after search and stock verification.",
            "rag_docs": [],
        },
        {
            "query": "Compare product {pid1} and product {pid2}, then order 3 of the cheaper one to {addr}.",
            "tools": ["get_product", "get_product", "place_order"],
            "params_fn": lambda q, c, a, pid, r: [
                {"product_id": pid},
                {"product_id": f"PRD_{(int(pid.split('_')[1]) + 1) % 10000:04d}"},
                {"product_id": pid, "quantity": 3, "shipping_address": a},
            ],
            "result_keys": [["price_usd"], ["price_usd"], ["order_id"]],
            "answer_fn": lambda q, c: "Compared products and ordered cheaper one.",
            "rag_docs": [],
        },
        {
            "query": "Find '{q}' items under $150 in {cat}, inspect product {pid}, and order 2 units to {addr} with gift_card.",
            "tools": ["search_products", "get_product", "place_order"],
            "params_fn": lambda q, c, a, pid, r: [
                {"query": q, "category": c, "max_price": 150},
                {"product_id": pid},
                {"product_id": pid, "quantity": 2, "shipping_address": a, "payment_method": "gift_card"},
            ],
            "result_keys": [["results"], ["name"], ["order_id"]],
            "answer_fn": lambda q, c: f"Found budget {q}, inspected, and ordered.",
            "rag_docs": [],
        },
    ]
    for i in range(50):
        t = templates[i % len(templates)]
        q = rng.choice(QUERIES)
        c = rng.choice(CATEGORIES)
        a = rng.choice(ADDRESSES)
        pid = f"PRD_{rng.randint(1000, 9999)}"
        pid1 = pid
        pid2 = f"PRD_{(rng.randint(1000, 9999))}"
        params_list = t["params_fn"](q, c, a, pid, rng)

        tasks.append({
            "task_id": f"ecommerce_{start_id + i + 1:03d}",
            "domain": "ecommerce",
            "difficulty": "hard",
            "query": t["query"].format(q=q, cat=c, pid=pid, pid1=pid1, pid2=pid2, addr=a),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {"tool": tool, "params": params, "expected_result_keys": keys}
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](q, c),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": t.get("rag_docs", []),
        })
    return tasks


def generate_all_tasks(rng: random.Random) -> list[dict]:
    tasks = []
    tasks.extend(generate_easy_tasks(rng, start_id=0))
    tasks.extend(generate_medium_tasks(rng, start_id=50))
    tasks.extend(generate_hard_tasks(rng, start_id=100))
    return tasks
