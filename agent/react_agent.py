"""ReAct agent loop using OpenAI client directly (no LangChain).

Parses Thought/Action/Action Input/Observation with regex.
max_steps=10, stops on "Final Answer:".
Returns the full step list as part of response.
"""

import json
import logging
import re
import time
from typing import Any, Optional

from openai import OpenAI

from config import MODELS, OPENAI_API_KEY, RANDOM_SEED
from agent.tool_executor import ToolExecutor
from agent.rag_retriever import RAGRetriever
from agent.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

MAX_STEPS = 10

SYSTEM_PROMPT = """You are an agent that answers questions using tools and retrieved context.
Follow this format exactly:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON parameters for the tool]

You will then receive an Observation with the tool result.
Repeat Thought/Action/Action Input/Observation as needed.

When you have enough information to answer, respond with:
Thought: I now have enough information to answer.
Final Answer: [your answer to the original question]

Available tools:
{tool_descriptions}

{rag_context}

{memory_context}
"""


def _format_tool_descriptions(tools: list[dict]) -> str:
    """Format tool list for the system prompt."""
    lines = []
    for t in tools:
        params = t["parameters"]
        props = params.get("properties", {})
        required = params.get("required", [])
        param_strs = []
        for pname, pschema in props.items():
            req = " (required)" if pname in required else ""
            param_strs.append(f"    - {pname}: {pschema.get('type', 'any')}{req} — {pschema.get('description', '')}")
        lines.append(f"- {t['name']}: {t['description']}")
        lines.extend(param_strs)
    return "\n".join(lines)


def _parse_action(text: str) -> Optional[tuple[str, dict]]:
    """Parse Action and Action Input from LLM output.

    Returns (tool_name, params_dict) or None.
    """
    # Match Action: <tool_name>
    action_match = re.search(r"Action:\s*(\w+)", text)
    if not action_match:
        return None

    tool_name = action_match.group(1).strip()

    # Match Action Input: <json>
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not input_match:
        # Try to find JSON on the line after "Action Input:"
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if not input_match:
            return tool_name, {}
        raw = input_match.group(1).strip()
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            params = {}
        return tool_name, params

    try:
        params = json.loads(input_match.group(1))
    except json.JSONDecodeError:
        params = {}

    return tool_name, params


def _parse_final_answer(text: str) -> Optional[str]:
    """Extract Final Answer from LLM output."""
    match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class AgentStep:
    """One step in the agent trace."""

    def __init__(
        self,
        step_id: int,
        step_type: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_params: Optional[dict] = None,
        tool_result: Any = None,
        timestamp: Optional[float] = None,
        model: Optional[str] = None,
        token_count: int = 0,
    ):
        self.step_id = step_id
        self.step_type = step_type
        self.content = content
        self.tool_name = tool_name
        self.tool_params = tool_params
        self.tool_result = tool_result
        self.timestamp = timestamp or time.time()
        self.model = model
        self.token_count = token_count

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "tool_result": self.tool_result,
            "timestamp": self.timestamp,
            "model": self.model,
            "token_count": self.token_count,
        }


class AgentResponse:
    """Complete agent response with trace."""

    def __init__(self, query: str, final_answer: str, steps: list[AgentStep], model: str):
        self.query = query
        self.final_answer = final_answer
        self.steps = steps
        self.model = model

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "model": self.model,
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


class ReActAgent:
    """ReAct agent with tool use, RAG, and memory."""

    def __init__(
        self,
        model_key: str = "gpt4o",
        tool_executor: Optional[ToolExecutor] = None,
        rag_retriever: Optional[RAGRetriever] = None,
        memory_manager: Optional[MemoryManager] = None,
        tracer: Optional[Any] = None,
    ):
        self.model_key = model_key
        self.model_name = MODELS.get(model_key, model_key)
        self.tool_executor = tool_executor or ToolExecutor()
        self.rag_retriever = rag_retriever
        self.memory_manager = memory_manager or MemoryManager()
        self.tracer = tracer  # Optional TraceLogger instance
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("ReActAgent initialized with model=%s tracer=%s", self.model_name, tracer is not None)

    def run(
        self, query: str, domain: Optional[str] = None, session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Run the agent on a query. Returns AgentResponse with full trace."""
        steps: list[AgentStep] = []
        step_counter = 0
        turn_id = 0

        # Get tool descriptions
        if domain:
            tools = self.tool_executor.get_tools_for_domain(domain)
        else:
            tools = self.tool_executor.list_tools()
        tool_desc = _format_tool_descriptions(tools)

        # Get RAG context
        rag_context = ""
        if self.rag_retriever and domain:
            try:
                rag_results = self.rag_retriever.retrieve(query, domain)
                if rag_results:
                    rag_context = "Retrieved context:\n" + "\n---\n".join(
                        r["text"] for r in rag_results[:3]
                    )
                    step_counter += 1
                    steps.append(AgentStep(
                        step_id=step_counter,
                        step_type="retrieval",
                        content=rag_context[:500],
                        model=self.model_name,
                    ))
                    # Trace retrieval
                    if self.tracer and session_id:
                        scores = [r.get("score", 0.0) for r in rag_results[:3]]
                        self.tracer.log_retrieval(session_id, turn_id, query, rag_results[:3], scores)
            except Exception as e:
                logger.error("RAG retrieval failed: %s", e)

        # Get memory context
        memory_context = self.memory_manager.get_context()

        # Build system prompt
        system_msg = SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc,
            rag_context=rag_context if rag_context else "No retrieved context available.",
            memory_context=memory_context if memory_context else "No prior memory.",
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]
        self.memory_manager.add_message("user", query)

        # Agent loop
        for iteration in range(MAX_STEPS):
            logger.info("Agent step %d/%d", iteration + 1, MAX_STEPS)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                    seed=RANDOM_SEED,
                )
                llm_output = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else 0
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                llm_output = f"Thought: LLM call failed with error: {e}\nFinal Answer: I encountered an error and cannot complete this request."
                token_count = 0

            # Check for Final Answer
            final_answer = _parse_final_answer(llm_output)
            if final_answer:
                step_counter += 1
                # Log the thought if present
                thought_match = re.search(r"Thought:\s*(.+?)(?=Final Answer:)", llm_output, re.DOTALL)
                if thought_match:
                    steps.append(AgentStep(
                        step_id=step_counter,
                        step_type="thought",
                        content=thought_match.group(1).strip(),
                        model=self.model_name,
                        token_count=token_count,
                    ))
                    step_counter += 1

                steps.append(AgentStep(
                    step_id=step_counter,
                    step_type="final",
                    content=final_answer,
                    model=self.model_name,
                    token_count=token_count,
                ))
                # Trace final answer
                if self.tracer and session_id:
                    self.tracer.log_step(
                        session_id, turn_id, step_counter, "final",
                        final_answer, token_count=token_count,
                    )

                self.memory_manager.add_message("assistant", final_answer)
                logger.info("Agent reached Final Answer at step %d", iteration + 1)
                return AgentResponse(query, final_answer, steps, self.model_name)

            # Parse Thought
            thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", llm_output, re.DOTALL)
            if thought_match:
                step_counter += 1
                thought_content = thought_match.group(1).strip()
                steps.append(AgentStep(
                    step_id=step_counter,
                    step_type="thought",
                    content=thought_content,
                    model=self.model_name,
                    token_count=token_count,
                ))
                # Trace thought
                if self.tracer and session_id:
                    self.tracer.log_step(
                        session_id, turn_id, step_counter, "thought",
                        thought_content, token_count=token_count,
                    )

            # Parse Action
            action_result = _parse_action(llm_output)
            if action_result:
                tool_name, tool_params = action_result

                step_counter += 1
                steps.append(AgentStep(
                    step_id=step_counter,
                    step_type="action",
                    content=f"Action: {tool_name}\nAction Input: {json.dumps(tool_params)}",
                    tool_name=tool_name,
                    tool_params=tool_params,
                    model=self.model_name,
                ))
                # Trace action
                if self.tracer and session_id:
                    self.tracer.log_step(
                        session_id, turn_id, step_counter, "action",
                        f"Action: {tool_name}",
                        tool_name=tool_name,
                        tool_params_raw=tool_params,
                    )

                # Execute tool
                exec_result = self.tool_executor.execute(tool_name, tool_params)
                observation = json.dumps(exec_result["result"], default=str)

                step_counter += 1
                steps.append(AgentStep(
                    step_id=step_counter,
                    step_type="observation",
                    content=observation,
                    tool_name=tool_name,
                    tool_result=exec_result["result"],
                ))
                # Trace observation
                if self.tracer and session_id:
                    self.tracer.log_step(
                        session_id, turn_id, step_counter, "observation",
                        observation[:1000],
                        tool_name=tool_name,
                        tool_params_validated=exec_result.get("validated_params"),
                        tool_result=exec_result["result"],
                        param_errors=exec_result.get("validation_errors"),
                    )

                # Update memory
                turn_id += 1
                self.memory_manager.add_observation(turn_id, observation)
                # Trace memory write
                if self.tracer and session_id:
                    snapshot = self.memory_manager.get_working_memory_snapshot()
                    self.tracer.log_memory_write(session_id, turn_id, "observation", snapshot, step_counter)

                # Append to messages for next iteration
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # No action parsed — LLM didn't follow format
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "user", "content": "Please follow the format: Thought/Action/Action Input or provide a Final Answer."})

        # Max steps reached
        step_counter += 1
        fallback = "I was unable to complete the task within the maximum number of steps."
        steps.append(AgentStep(
            step_id=step_counter,
            step_type="final",
            content=fallback,
            model=self.model_name,
        ))
        logger.warning("Agent hit max_steps=%d without Final Answer", MAX_STEPS)
        return AgentResponse(query, fallback, steps, self.model_name)
