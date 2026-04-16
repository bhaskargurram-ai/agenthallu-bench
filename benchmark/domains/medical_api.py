"""Medical domain task templates for benchmark generation.

Produces 150 tasks: 50 easy, 50 medium, 50 hard.
"""

import random

PATIENT_IDS = ["P001", "P002", "P003", "P004", "P005"]
DOCTOR_IDS = ["D001", "D002", "D003", "D004", "D005"]

DRUGS = [
    "metformin", "lisinopril", "atorvastatin", "albuterol", "sertraline",
    "omeprazole", "levothyroxine", "ibuprofen", "amlodipine", "metoprolol",
    "warfarin", "aspirin", "fluticasone", "sumatriptan", "cetirizine",
]

APPT_TYPES = ["checkup", "follow_up", "specialist", "emergency"]

DATES = [
    "2024-08-01", "2024-08-05", "2024-08-10", "2024-08-15", "2024-08-20",
    "2024-09-01", "2024-09-10", "2024-09-15", "2024-10-01", "2024-10-15",
    "2024-11-01", "2024-12-01",
]


def generate_easy_tasks(rng: random.Random, start_id: int = 0) -> list[dict]:
    """50 easy tasks: single tool call."""
    tasks = []
    templates = [
        {
            "query": "Get the medical record for patient {pid}.",
            "tool": "get_patient_record",
            "params_fn": lambda pid, r: {"patient_id": pid},
            "result_keys": ["name", "dob", "conditions", "medications"],
            "answer_fn": lambda pid: f"Retrieved record for patient {pid}.",
        },
        {
            "query": "Check if {drug_a} and {drug_b} interact with each other.",
            "tool": "check_drug_interaction",
            "params_fn": lambda pid, r: {"drug_a": r.choice(DRUGS[:7]), "drug_b": r.choice(DRUGS[7:])},
            "result_keys": ["interaction", "severity"],
            "answer_fn": lambda pid: "Drug interaction check completed.",
        },
        {
            "query": "Schedule a {atype} appointment for patient {pid} with doctor {did} on {date}.",
            "tool": "schedule_appointment",
            "params_fn": lambda pid, r: {
                "patient_id": pid,
                "doctor_id": r.choice(DOCTOR_IDS),
                "date": r.choice(DATES),
                "appointment_type": r.choice(APPT_TYPES),
            },
            "result_keys": ["appointment_id", "status"],
            "answer_fn": lambda pid: f"Appointment scheduled for {pid}.",
        },
        {
            "query": "What conditions does patient {pid} have?",
            "tool": "get_patient_record",
            "params_fn": lambda pid, r: {"patient_id": pid, "fields": ["conditions"]},
            "result_keys": ["conditions"],
            "answer_fn": lambda pid: f"Conditions for patient {pid} retrieved.",
        },
        {
            "query": "What medications is patient {pid} currently taking?",
            "tool": "get_patient_record",
            "params_fn": lambda pid, r: {"patient_id": pid, "fields": ["medications"]},
            "result_keys": ["medications"],
            "answer_fn": lambda pid: f"Medications for patient {pid} retrieved.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        pid = rng.choice(PATIENT_IDS)
        did = rng.choice(DOCTOR_IDS)
        date = rng.choice(DATES)
        atype = rng.choice(APPT_TYPES)
        params = t["params_fn"](pid, rng)
        drug_a = params.get("drug_a", "metformin")
        drug_b = params.get("drug_b", "ibuprofen")

        tasks.append({
            "task_id": f"medical_{start_id + i + 1:03d}",
            "domain": "medical",
            "difficulty": "easy",
            "query": t["query"].format(pid=pid, did=did, date=date, atype=atype, drug_a=drug_a, drug_b=drug_b),
            "required_tools": [t["tool"]],
            "correct_tool_sequence": [{
                "tool": t["tool"],
                "params": params,
                "expected_result_keys": t["result_keys"],
            }],
            "correct_final_answer": t["answer_fn"](pid),
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
            "query": "Get patient {pid}'s record and check if their medications {drug_a} and {drug_b} have any interactions.",
            "tools": ["get_patient_record", "check_drug_interaction"],
            "params_fn": lambda pid, da, db, r: [
                {"patient_id": pid},
                {"drug_a": da, "drug_b": db},
            ],
            "result_keys": [["name", "medications"], ["interaction", "severity"]],
            "answer_fn": lambda pid: f"Patient {pid} record and drug interaction check complete.",
        },
        {
            "query": "Look up patient {pid}'s conditions and schedule a {atype} with doctor {did} on {date}.",
            "tools": ["get_patient_record", "schedule_appointment"],
            "params_fn": lambda pid, da, db, r: [
                {"patient_id": pid, "fields": ["conditions"]},
                {"patient_id": pid, "doctor_id": r.choice(DOCTOR_IDS), "date": r.choice(DATES), "appointment_type": r.choice(APPT_TYPES)},
            ],
            "result_keys": [["conditions"], ["appointment_id"]],
            "answer_fn": lambda pid: f"Retrieved conditions and scheduled appointment for {pid}.",
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        pid = rng.choice(PATIENT_IDS)
        did = rng.choice(DOCTOR_IDS)
        date = rng.choice(DATES)
        atype = rng.choice(APPT_TYPES)
        drug_a = rng.choice(DRUGS[:7])
        drug_b = rng.choice(DRUGS[7:])
        params_list = t["params_fn"](pid, drug_a, drug_b, rng)

        tasks.append({
            "task_id": f"medical_{start_id + i + 1:03d}",
            "domain": "medical",
            "difficulty": "medium",
            "query": t["query"].format(pid=pid, did=did, date=date, atype=atype, drug_a=drug_a, drug_b=drug_b),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {
                    "tool": tool,
                    "params": params,
                    "expected_result_keys": keys,
                }
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](pid),
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
            "query": "Review patient {pid}'s full record, check interactions between all their medications, and schedule a {atype} follow-up with {did} on {date}.",
            "tools": ["get_patient_record", "check_drug_interaction", "schedule_appointment"],
            "params_fn": lambda pid, did, date, atype, r: [
                {"patient_id": pid},
                {"drug_a": "metformin", "drug_b": "lisinopril"},
                {"patient_id": pid, "doctor_id": did, "date": date, "appointment_type": atype},
            ],
            "result_keys": [["name", "conditions", "medications"], ["interaction", "severity"], ["appointment_id"]],
            "answer_fn": lambda pid: f"Full review and follow-up scheduled for {pid}.",
            "rag_docs": ["medical_000", "medical_001"],
        },
        {
            "query": "Patient {pid} is being prescribed {drug_new}. Check their current medications, verify no interactions with {drug_new}, and schedule a specialist appointment with {did} on {date}.",
            "tools": ["get_patient_record", "check_drug_interaction", "schedule_appointment"],
            "params_fn": lambda pid, did, date, atype, r: [
                {"patient_id": pid, "fields": ["medications"]},
                {"drug_a": r.choice(DRUGS[:5]), "drug_b": r.choice(DRUGS[5:10])},
                {"patient_id": pid, "doctor_id": did, "date": date, "appointment_type": "specialist"},
            ],
            "result_keys": [["medications"], ["interaction"], ["appointment_id"]],
            "answer_fn": lambda pid: f"Medication review and specialist appointment for {pid}.",
            "rag_docs": ["medical_002", "medical_003"],
        },
    ]

    for i in range(50):
        t = templates[i % len(templates)]
        pid = rng.choice(PATIENT_IDS)
        did = rng.choice(DOCTOR_IDS)
        date = rng.choice(DATES)
        atype = rng.choice(APPT_TYPES)
        drug_new = rng.choice(DRUGS[5:])
        params_list = t["params_fn"](pid, did, date, atype, rng)

        tasks.append({
            "task_id": f"medical_{start_id + i + 1:03d}",
            "domain": "medical",
            "difficulty": "hard",
            "query": t["query"].format(pid=pid, did=did, date=date, atype=atype, drug_new=drug_new),
            "required_tools": t["tools"],
            "correct_tool_sequence": [
                {
                    "tool": tool,
                    "params": params,
                    "expected_result_keys": keys,
                }
                for tool, params, keys in zip(t["tools"], params_list, t["result_keys"])
            ],
            "correct_final_answer": t["answer_fn"](pid),
            "multi_turn": False,
            "num_turns": 1,
            "rag_docs_needed": t.get("rag_docs", []),
        })
    return tasks


def generate_all_tasks(rng: random.Random) -> list[dict]:
    """Generate all 150 medical domain tasks."""
    tasks = []
    tasks.extend(generate_easy_tasks(rng, start_id=0))
    tasks.extend(generate_medium_tasks(rng, start_id=50))
    tasks.extend(generate_hard_tasks(rng, start_id=100))
    return tasks
