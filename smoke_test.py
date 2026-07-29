"""Day 1 End-to-End Smoke Test.

Runs 3 test queries (one per domain) through the full agent pipeline.
Prints the full agent trace to console using rich.
"""

import json
import logging
import sys
import os

# Ensure agenthallu_bench is on path
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config import LOG_LEVEL
from agent.tool_executor import ToolExecutor
from agent.rag_retriever import RAGRetriever
from agent.memory_manager import MemoryManager
from agent.react_agent import ReActAgent

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

console = Console()

TEST_QUERIES = [
    {
        "domain": "weather",
        "query": "What is the weather in London on 2024-06-15? Give me the temperature and condition.",
    },
    {
        "domain": "calendar",
        "query": "Create a team meeting for 2024-07-01 at 10:00 AM for 60 minutes, then show me all events on that date.",
    },
    {
        "domain": "medical",
        "query": "Get the medical record for patient P001, then check if their medications metformin and lisinopril have any interactions.",
    },
]


def print_agent_trace(response_dict: dict) -> None:
    """Print a rich-formatted agent trace."""
    console.print()
    console.rule(f"[bold cyan]Query: {response_dict['query']}")
    console.print(f"[dim]Model: {response_dict['model']} | Steps: {response_dict['num_steps']}[/dim]")
    console.print()

    for step in response_dict["steps"]:
        step_type = step["step_type"]
        content = step["content"]

        if step_type == "retrieval":
            console.print(Panel(
                content[:300] + ("..." if len(content) > 300 else ""),
                title="[yellow]RAG Retrieval[/yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            ))
        elif step_type == "thought":
            console.print(Panel(
                content,
                title=f"[blue]Thought (step {step['step_id']})[/blue]",
                border_style="blue",
                box=box.ROUNDED,
            ))
        elif step_type == "action":
            tool_info = f"Tool: {step.get('tool_name', '?')}\nParams: {json.dumps(step.get('tool_params', {}), indent=2)}"
            console.print(Panel(
                tool_info,
                title=f"[green]Action (step {step['step_id']})[/green]",
                border_style="green",
                box=box.ROUNDED,
            ))
        elif step_type == "observation":
            result_str = json.dumps(step.get("tool_result", {}), indent=2, default=str)
            console.print(Panel(
                result_str[:500],
                title=f"[magenta]Observation (step {step['step_id']})[/magenta]",
                border_style="magenta",
                box=box.ROUNDED,
            ))
        elif step_type == "final":
            console.print(Panel(
                content,
                title="[bold red]Final Answer[/bold red]",
                border_style="red",
                box=box.DOUBLE,
            ))

    console.print()


def run_smoke_test():
    """Run 3 domain queries and print traces."""
    console.print(Panel(
        "[bold]AgentHallu-Bench Day 1 Smoke Test[/bold]\n"
        "Testing: ToolExecutor + RAGRetriever + MemoryManager + ReActAgent\n"
        "3 queries (weather, calendar, medical)",
        title="Smoke Test",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    # Initialize components
    console.print("[dim]Initializing components...[/dim]")
    tool_executor = ToolExecutor()
    rag_retriever = RAGRetriever()
    rag_retriever.initialize()
    console.print("[green]Components initialized.[/green]")

    # Summary table
    summary_table = Table(title="Smoke Test Results", box=box.ROUNDED)
    summary_table.add_column("Domain", style="cyan")
    summary_table.add_column("Query", style="white", max_width=50)
    summary_table.add_column("Steps", style="yellow")
    summary_table.add_column("Tools Called", style="green")
    summary_table.add_column("Final Answer", style="red", max_width=60)

    for test in TEST_QUERIES:
        domain = test["domain"]
        query = test["query"]

        console.print(f"\n[bold]Running {domain} query...[/bold]")

        memory_manager = MemoryManager()
        agent = ReActAgent(
            model_key="gpt4o",
            tool_executor=tool_executor,
            rag_retriever=rag_retriever,
            memory_manager=memory_manager,
        )

        try:
            response = agent.run(query, domain=domain)
            response_dict = response.to_dict()
            print_agent_trace(response_dict)

            # Collect tools called
            tools_called = [s.tool_name for s in response.steps if s.step_type == "action" and s.tool_name]

            summary_table.add_row(
                domain,
                query[:50],
                str(response_dict["num_steps"]),
                ", ".join(tools_called) if tools_called else "none",
                response.final_answer[:60],
            )
        except Exception as e:
            console.print(f"[red]Error running {domain} query: {e}[/red]")
            import traceback
            traceback.print_exc()
            summary_table.add_row(domain, query[:50], "ERROR", str(e)[:30], "FAILED")

    console.print()
    console.print(summary_table)
    console.print()
    console.print("[bold green]Smoke test complete.[/bold green]")


if __name__ == "__main__":
    run_smoke_test()
