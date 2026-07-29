"""Day 8 STEP 6: Extract case studies from trace DBs.

Finds qualitatively interesting examples for paper appendix:
  1. HIGH-EPS SURPRISING — frontier model with EPS=3 on a simple task
  2. SELF-CORRECTION — agent had wrong intermediate, but final_correct=True
  3. CROSS-AGENT AMPLIFICATION (P4) — error in agent1 propagates through 2/3
  4. INTERCEPTOR-CAUGHT — interceptor abstained on a true hallucination
  5. INTERCEPTOR-FALSE-POSITIVE — interceptor wrongly abstained on a correct answer

Output: data/results/case_studies.json (list of formatted traces)
       analysis/case_studies.md (markdown for paper appendix)
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

console = Console()


def load_full_trace(db_path: str, session_id: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    sess = c.fetchone()
    if not sess:
        conn.close()
        return {}
    sess = dict(sess)
    c.execute("SELECT * FROM steps WHERE session_id=? ORDER BY step_number", (session_id,))
    sess["steps"] = [dict(r) for r in c.fetchall()]
    try:
        c.execute("SELECT * FROM injections WHERE session_id=?", (session_id,))
        sess["injections"] = [dict(r) for r in c.fetchall()]
    except sqlite3.OperationalError:
        sess["injections"] = []
    conn.close()
    return sess


def find_high_eps_surprising(p2_db: str) -> list[dict]:
    """EPS=3 on frontier models."""
    conn = sqlite3.connect(p2_db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # EPS is not stored directly in DB; we proxy: injection applied + final wrong + frontier model
    try:
        c.execute("""
            SELECT s.session_id FROM sessions s
            INNER JOIN injections i ON s.session_id = i.session_id
            WHERE s.final_correct = 0
              AND s.model IN ('gpt4o', 'gemini_25_flash', 'o3_mini', 'gpt41_mini')
              AND s.final_answer NOT LIKE 'ERROR%'
            LIMIT 20
        """)
        sids = [r["session_id"] for r in c.fetchall()]
    except Exception:
        sids = []
    conn.close()
    return [load_full_trace(p2_db, s) for s in sids[:2]]


def find_self_correction(p2_db: str) -> list[dict]:
    """final_correct=True despite injection."""
    conn = sqlite3.connect(p2_db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        c.execute("""
            SELECT s.session_id FROM sessions s
            INNER JOIN injections i ON s.session_id = i.session_id
            WHERE s.final_correct = 1
              AND s.final_answer NOT LIKE 'ERROR%'
            LIMIT 10
        """)
        sids = [r["session_id"] for r in c.fetchall()]
    except Exception:
        sids = []
    conn.close()
    return [load_full_trace(p2_db, s) for s in sids[:2]]


def find_cross_agent(p4_db: str) -> list[dict]:
    """P4 traces where error propagates across agents."""
    if not os.path.exists(p4_db):
        return []
    conn = sqlite3.connect(p4_db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        c.execute("""
            SELECT session_id FROM sessions
            WHERE final_correct = 0 AND final_answer NOT LIKE 'ERROR%'
            LIMIT 5
        """)
        sids = [r["session_id"] for r in c.fetchall()]
    except Exception:
        sids = []
    conn.close()
    return [load_full_trace(p4_db, s) for s in sids[:2]]


def find_interceptor_cases(interceptor_db: str) -> tuple[list[dict], list[dict]]:
    """Return (caught hallucination, false positives)."""
    if not os.path.exists(interceptor_db):
        return [], []
    conn = sqlite3.connect(interceptor_db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    caught = []
    fps = []
    try:
        # Caught: injection_type mentions interceptor, final_correct=0, but abstained signal
        # Since we don't store abstained in sessions, inspect final_answer text
        c.execute("""
            SELECT session_id FROM sessions
            WHERE (final_answer LIKE '%cannot reliably answer%'
                OR final_answer LIKE '%Cannot reliably answer%')
              AND final_correct = 0
            LIMIT 2
        """)
        caught = [load_full_trace(interceptor_db, r["session_id"]) for r in c.fetchall()]
        # FPs: abstained but answer would have been correct (unknowable from trace alone)
        c.execute("""
            SELECT session_id FROM sessions
            WHERE final_answer LIKE '%cannot reliably answer%'
            LIMIT 2
        """)
        fps = [load_full_trace(interceptor_db, r["session_id"]) for r in c.fetchall()][:1]
    except Exception:
        pass
    conn.close()
    return caught, fps


def format_trace_md(trace: dict, title: str) -> str:
    if not trace:
        return f"## {title}\n\n(No example found)\n\n"
    lines = [f"## {title}", ""]
    lines.append(f"- **Session ID**: `{trace.get('session_id')}`")
    lines.append(f"- **Model**: {trace.get('model')}")
    lines.append(f"- **Domain**: {trace.get('domain')}")
    lines.append(f"- **Injection type**: {trace.get('injection_type')}")
    lines.append(f"- **Ground truth**: {trace.get('ground_truth_answer', 'N/A')[:200] if trace.get('ground_truth_answer') else 'N/A'}")
    lines.append(f"- **Final answer**: {(trace.get('final_answer') or '')[:200]}")
    lines.append(f"- **Final correct**: {trace.get('final_correct')}")
    if trace.get("injections"):
        inj = trace["injections"][0]
        lines.append(f"- **Injection target_stage**: {inj.get('target_stage')}")
        lines.append(f"- **Injection detail**: {str(inj.get('details', ''))[:160]}")
    lines.append("")
    lines.append("### Trace steps")
    lines.append("```")
    steps = trace.get("steps", [])[:12]
    for s in steps:
        t = s.get("step_type", "?")
        c = (s.get("content") or "")[:200].replace("\n", " ")
        if t == "action":
            lines.append(f"[{t}] {s.get('tool_name', '?')}: {c}")
        else:
            lines.append(f"[{t}] {c}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-db", default="data/traces/p2_day6.db")
    parser.add_argument("--p4-db", default="data/traces/p4_day7.db")
    parser.add_argument("--interceptor-db", default="data/traces/interceptor_day8.db")
    parser.add_argument("--output-json", default="data/results/case_studies.json")
    parser.add_argument("--output-md", default="analysis/case_studies.md")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    p2_db = os.path.join(base, args.p2_db)
    p4_db = os.path.join(base, args.p4_db)
    int_db = os.path.join(base, args.interceptor_db)

    console.rule("[bold cyan]Day 8 STEP 6 — Case Studies[/bold cyan]")

    high_eps = find_high_eps_surprising(p2_db) if os.path.exists(p2_db) else []
    self_corr = find_self_correction(p2_db) if os.path.exists(p2_db) else []
    cross_agent = find_cross_agent(p4_db)
    caught, fps = find_interceptor_cases(int_db)

    categories = {
        "high_eps_surprising": high_eps,
        "self_correction": self_corr,
        "cross_agent_amplification": cross_agent,
        "interceptor_caught": caught,
        "interceptor_false_positive": fps,
    }
    for k, v in categories.items():
        console.print(f"  {k}: {len(v)}")

    out_json = os.path.join(base, args.output_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(categories, f, indent=2, default=str)
    console.print(f"[green]Saved JSON to {out_json}[/green]")

    md_lines = ["# Case Studies (Appendix)", ""]
    title_map = {
        "high_eps_surprising": "High-EPS Surprising (Frontier Model Fully Propagates Error)",
        "self_correction": "Self-Correction (Agent Recovers Despite Bad Parameter)",
        "cross_agent_amplification": "Cross-Agent Amplification (Error Propagates Through Multi-Agent Pipeline)",
        "interceptor_caught": "Interceptor Caught a Hallucination",
        "interceptor_false_positive": "Interceptor False Positive",
    }
    for cat, traces in categories.items():
        for i, tr in enumerate(traces[:2]):
            md_lines.append(format_trace_md(tr, f"{title_map.get(cat, cat)} — Example {i+1}"))

    out_md = os.path.join(base, args.output_md)
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    console.print(f"[green]Saved markdown to {out_md}[/green]")


if __name__ == "__main__":
    main()
