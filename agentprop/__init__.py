"""AgentProp-Bench — one-command evaluation harness for tool-using LLM agents.

    pip install agentprop-bench
    agentprop eval --model gpt-4o-mini

Paper: https://arxiv.org/abs/2604.16706
Code:  https://github.com/bhaskargurram-ai/agenthallu-bench
"""

__version__ = "1.0.0"

from agentprop.data import load_tasks  # noqa: E402
from agentprop.report import summarize  # noqa: E402
from agentprop.runner import evaluate  # noqa: E402

__all__ = ["load_tasks", "evaluate", "summarize", "__version__"]
