"""
agent.py — AgentHalBench v3
----------------------------
Four-stage LLM agent pipeline with multi-provider support:
  - OpenAI (GPT family)
  - Together AI (Llama, Mistral, Mixtral, DeepSeek, Qwen via API)
  - HuggingFace transformers (local) — for fully reproducible cross-family replication

P0 #5: Cross-family replication via HuggingFace transformers (local).
P0 #1: Seed propagation through every stochastic step.
P0 #7: Plan elaboration captured in PlanningTrace for confound analysis.

Author: Bhaskar Gurram, Zasti Inc.
"""

from __future__ import annotations

import json
import os
import random
import time
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Trace dataclasses ──────────────────────────────────────────────

@dataclass
class PlanningTrace:
    task: str
    plan: list
    reasoning: str
    plan_raw: str
    # P0 #7: capture plan elaboration metrics for confound analysis
    n_steps: int = 0
    avg_step_length: float = 0.0
    has_dependencies: bool = False  # mentions "after", "based on", etc.
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    return_value: str
    success: bool
    error_msg: Optional[str] = None


@dataclass
class ToolUseTrace:
    planned_action: str
    tool_calls: list
    invocation_decision: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MemoryTrace:
    query: str
    retrieved_context: list
    context_used: str
    source: str
    # P2 #16: retrieval ablation — record retrieval config
    retriever_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OutputTrace:
    final_response: str
    sampled_responses: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentTrace:
    task_id: str
    task: str
    domain: str
    model: str
    seed: int = 42                    # P0 #1: seed traceability
    planning: Optional[PlanningTrace] = None
    tool_use: Optional[ToolUseTrace] = None
    memory: Optional[MemoryTrace] = None
    output: Optional[OutputTrace] = None
    task_success: Optional[bool] = None
    total_latency_s: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, default=str)


# ── Tool registry — P1 #15: expanded to 12 tools with semantic overlap ──

TOOLS = {
    "search_web":      "General web search | params: query(str)",
    "search_arxiv":    "Search arXiv preprints | params: query(str)",
    "search_news":     "Search recent news articles | params: query(str)",
    "run_code":        "Execute Python code in sandbox | params: code(str)",
    "calculate":       "Numerical math calculation | params: expression(str)",
    "solve_symbolic":  "Symbolic math (algebra, calculus) | params: expression(str)",
    "retrieve_memory": "Query internal memory store | params: query(str), top_k(int)",
    "summarize":       "Summarize long text | params: text(str), max_length(int)",
    "translate":       "Translate text | params: text(str), target_lang(str)",
    "read_file":       "Read a local file | params: filepath(str)",
    "write_file":      "Write a local file | params: filepath(str), content(str)",
    "send_email":      "Compose email draft | params: to(str), subject(str), body(str)",
}
TOOL_LIST = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())


# ── Memory store import ───────────────────────────────────────────
# (Keep your existing 320–500 document MEMORY_STORE here.
#  Imported as in original agent.py — not duplicated for brevity.)
from agent_memory_store import MEMORY_STORE  # noqa: F401  (or paste inline)


# ── Semantic Retrieval (P2 #16: configurable embedding model) ─────

class SemanticRetriever:
    """
    Configurable retrieval backend.
    Supports any sentence-transformers model for ablation studies.
    """

    SUPPORTED_MODELS = {
        "minilm":       "all-MiniLM-L6-v2",       # 22M params, fast
        "mpnet":        "all-mpnet-base-v2",      # 110M params, strong
        "bge-large":    "BAAI/bge-large-en-v1.5", # 335M params, top-tier
    }

    def __init__(self, documents: list, model_key: str = "minilm",
                 device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        import torch

        model_name = self.SUPPORTED_MODELS.get(model_key, model_key)
        self.model_name = model_name
        self.model_key = model_key
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.documents = documents
        self.doc_embeddings = self.model.encode(
            documents, batch_size=64,
            show_progress_bar=False, normalize_embeddings=True,
        )
        logger.info(
            f"SemanticRetriever[{model_key} → {model_name}]: "
            f"encoded {len(documents)} docs on {device} ✓"
        )

    def retrieve(self, query: str, top_k: int = 3) -> list:
        query_emb = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        )
        scores = np.dot(self.doc_embeddings, query_emb[0])
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


# ── LLM client — multi-provider with HuggingFace local backend ────

OPENAI_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
    "gpt-4-turbo", "gpt-3.5-turbo",
}

# Together AI hosted (cloud API)
TOGETHER_MODELS = {
    "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
}

# HuggingFace local (downloaded weights, runs on local GPU/CPU)
# Prefix with "hf:" to force local HF execution.
# Example: "hf:meta-llama/Llama-3.1-8B-Instruct"
HF_PREFIX = "hf:"


class HuggingFaceClient:
    """
    Local HuggingFace transformers backend.

    P0 #5: Cross-family replication without API dependency.
    Supports 4-bit / 8-bit quantization via bitsandbytes for
    fitting larger models on consumer GPUs.

    Recommended models for AgentHalBench cross-family replication:
      - meta-llama/Llama-3.1-8B-Instruct        (16GB fp16)
      - meta-llama/Llama-3.1-70B-Instruct       (~40GB 4-bit)
      - Qwen/Qwen2.5-7B-Instruct                (15GB fp16)
      - Qwen/Qwen2.5-72B-Instruct               (~45GB 4-bit)
      - mistralai/Mistral-7B-Instruct-v0.3      (15GB fp16)
      - google/gemma-2-27b-it                   (~18GB 4-bit)
    """

    def __init__(
        self,
        model_id: str,
        quantization: Optional[str] = None,   # "4bit", "8bit", or None
        max_new_tokens: int = 1024,
        hf_token: Optional[str] = None,
        seed: int = 42,
    ):
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        )

        # Strip "hf:" prefix
        if model_id.startswith(HF_PREFIX):
            model_id = model_id[len(HF_PREFIX):]
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        # P0 #1: deterministic init
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        token = hf_token or os.environ.get("HF_TOKEN")

        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading HF model: {model_id} (quant={quantization})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
            token=token,
            trust_remote_code=False,
        )
        self.model.eval()
        logger.info(f"HF model {model_id} loaded ✓")

    def complete(self, messages: list, temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 seed: Optional[int] = None) -> str:
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self.model.device)

        do_sample = temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return text.strip()

    def sample_n(self, messages: list, n: int = 3,
                 temperature: float = 0.9,
                 base_seed: int = 42) -> list:
        """Generate N independent samples. P0 #1: deterministic per-sample seed."""
        samples = []
        for i in range(n):
            try:
                # Different seed per sample, but deterministic given base_seed
                samples.append(
                    self.complete(messages, temperature=temperature,
                                  seed=base_seed + i + 1)
                )
            except Exception as e:
                logger.warning(f"HF sample {i} failed: {e}")
        return samples


class LLMClient:
    """
    Unified client across OpenAI, Together AI, and HuggingFace local.

    Routing rules:
      - "gpt-*" → OpenAI
      - "hf:*"  → HuggingFace local (transformers)
      - everything else → Together AI cloud API
    """

    def __init__(
        self,
        model: str,
        openai_key: str = "",
        together_key: str = "",
        hf_token: str = "",
        hf_quantization: Optional[str] = None,
        seed: int = 42,
    ):
        self.model = model
        self.seed = seed
        self._hf_client: Optional[HuggingFaceClient] = None

        if model.startswith(HF_PREFIX):
            self.provider = "hf_local"
            self._hf_client = HuggingFaceClient(
                model_id=model, quantization=hf_quantization,
                hf_token=hf_token, seed=seed,
            )
            self.client = self._hf_client
        elif model in OPENAI_MODELS or model.startswith("gpt-"):
            self.provider = "openai"
            self.client = openai.OpenAI(api_key=openai_key)
        else:
            from together import Together
            self.provider = "together"
            self.client = Together(api_key=together_key)

    def complete(self, messages: list, temperature: float = 0.7,
                 max_tokens: int = 1024,
                 seed: Optional[int] = None) -> str:
        if self.provider == "hf_local":
            return self._hf_client.complete(
                messages, temperature=temperature,
                max_tokens=max_tokens, seed=seed,
            )

        # OpenAI / Together both implement chat.completions.create
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # OpenAI accepts seed parameter; Together does too on most models
        if seed is not None:
            kwargs["seed"] = seed
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except TypeError:
            # Fallback if provider doesn't accept seed kw
            kwargs.pop("seed", None)
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

    def sample_n(self, messages: list, n: int = 3,
                 temperature: float = 0.9,
                 base_seed: Optional[int] = None) -> list:
        """N independent samples for SelfCheckGPT consistency check."""
        if self.provider == "hf_local":
            return self._hf_client.sample_n(
                messages, n=n, temperature=temperature,
                base_seed=base_seed or self.seed,
            )

        samples = []
        bs = base_seed or self.seed
        for i in range(n):
            try:
                samples.append(
                    self.complete(messages, temperature=temperature,
                                  seed=bs + i + 1)
                )
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
        return samples


# ── Plan-elaboration scoring helpers (P0 #7) ──────────────────────

_DEP_MARKERS = (
    "after ", "based on", "using the", "from the previous", "then ",
    "subsequently", "depending on", "if the", "once ", "given the result",
)


def _score_plan_elaboration(plan: list[str]) -> tuple[int, float, bool]:
    """Capture plan-specificity metrics for confound analysis (Finding F2)."""
    n = len(plan)
    avg_len = (sum(len(s) for s in plan) / n) if n else 0.0
    joined = " ".join(plan).lower()
    has_deps = any(marker in joined for marker in _DEP_MARKERS)
    return n, avg_len, has_deps


# ── Agent Pipeline ────────────────────────────────────────────────

class AgentPipeline:
    """
    Four-stage agentic pipeline with seed-aware execution.
    """

    def __init__(self, model: str, openai_key: str = "",
                 together_key: str = "", hf_token: str = "",
                 hf_quantization: Optional[str] = None,
                 n_selfcheck_samples: int = 3,
                 retriever: Optional[SemanticRetriever] = None,
                 retriever_model: str = "minilm",
                 top_k: int = 3,
                 seed: int = 42):
        self.llm = LLMClient(
            model=model, openai_key=openai_key, together_key=together_key,
            hf_token=hf_token, hf_quantization=hf_quantization, seed=seed,
        )
        self.model = model
        self.seed = seed
        self.n_samples = n_selfcheck_samples
        self.top_k = top_k
        self.retriever = retriever or SemanticRetriever(
            MEMORY_STORE, model_key=retriever_model,
        )

    def _run_tool(self, tool_name: str, args: dict) -> tuple:
        if tool_name == "calculate":
            try:
                result = eval(args.get("expression", "0"),
                              {"__builtins__": {}}, {})
                return str(result), True
            except Exception as e:
                return f"Error: {e}", False
        elif tool_name == "retrieve_memory":
            results = self.retriever.retrieve(
                args.get("query", ""), args.get("top_k", self.top_k))
            return "\n".join(results), True
        elif tool_name in TOOLS:
            return f"[Simulated {tool_name}: {str(args)[:60]}]", True
        return f"Unknown tool: {tool_name}", False

    def stage_planning(self, task: str) -> PlanningTrace:
        prompt = f"""Decompose this task into numbered actionable steps.
Task: {task}
Available tools:
{TOOL_LIST}

Format:
REASONING: <your thinking>
PLAN:
1. <step>
2. <step>"""
        raw = self.llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.7, seed=self.seed,
        )
        reasoning, plan = "", []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("REASONING:"):
                reasoning = line[10:].strip()
            elif line and line[0].isdigit() and "." in line:
                plan.append(line.split(".", 1)[-1].strip())
        if not plan:
            plan = [task]

        n_steps, avg_len, has_deps = _score_plan_elaboration(plan)
        return PlanningTrace(
            task=task, plan=plan, reasoning=reasoning, plan_raw=raw,
            n_steps=n_steps, avg_step_length=round(avg_len, 1),
            has_dependencies=has_deps,
        )

    def stage_tool_use(self, action: str,
                       context: str = "") -> ToolUseTrace:
        prompt = f"""Select the best tool for this action and provide arguments as JSON.
Action: {action}
Context: {context}
Tools:
{TOOL_LIST}

JSON only:
{{"decision":"use_tool" or "no_tool_needed","tool_name":"<name or null>","arguments":{{<args>}}}}"""
        raw = self.llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.2, seed=self.seed,
        )
        tool_calls, decision = [], "no_tool_needed"
        try:
            s, e = raw.find("{"), raw.rfind("}") + 1
            parsed = json.loads(raw[s:e])
            decision = parsed.get("decision", "no_tool_needed")
            tname = parsed.get("tool_name")
            targs = parsed.get("arguments") or {}
            if decision == "use_tool" and tname:
                result, ok = self._run_tool(tname, targs)
                tool_calls.append(ToolCall(
                    tool_name=tname, arguments=targs,
                    return_value=result, success=ok))
        except Exception as ex:
            logger.warning(f"Tool parse error: {ex}")
            decision = "parse_error"
        return ToolUseTrace(
            planned_action=action, tool_calls=tool_calls,
            invocation_decision=decision,
        )

    def stage_memory(self, query: str) -> MemoryTrace:
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        ctx_str = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(retrieved))
        prompt = f"""Given this query and retrieved context, extract the most relevant information.
Query: {query}
Retrieved context:
{ctx_str}

Respond:
RELEVANT: <extracted information from the retrieved context above>
SOURCE: retrieved"""
        raw = self.llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.2, seed=self.seed,
        )
        relevant, source = raw[:300], "retrieved"
        for line in raw.splitlines():
            if line.startswith("RELEVANT:"):
                relevant = line[9:].strip()
            elif line.startswith("SOURCE:"):
                source = line[7:].strip().strip("'\"")
        return MemoryTrace(
            query=query, retrieved_context=retrieved,
            context_used=relevant, source=source,
            retriever_model=self.retriever.model_key,
            top_k=self.top_k,
        )

    def stage_output(self, task: str, plan: list,
                     tool_result: str, memory: str) -> OutputTrace:
        plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        prompt = f"""Task: {task}
Plan executed:
{plan_str}
Tool result: {tool_result or 'None'}
Retrieved memory context: {memory}

Generate a complete, accurate response based on the above:"""
        messages = [{"role": "user", "content": prompt}]
        primary = self.llm.complete(messages, temperature=0.5,
                                    seed=self.seed)
        samples = self.llm.sample_n(messages, n=self.n_samples,
                                    temperature=0.9, base_seed=self.seed)
        return OutputTrace(
            final_response=primary, sampled_responses=samples,
        )

    def run(self, task_id: str, task: str, domain: str) -> AgentTrace:
        t0 = time.time()
        trace = AgentTrace(
            task_id=task_id, task=task, domain=domain,
            model=self.model, seed=self.seed,
        )
        try:
            trace.planning = self.stage_planning(task)
            first_action = (trace.planning.plan[0]
                            if trace.planning.plan else task)
            ctx = f"Task: {task}\nReasoning: {trace.planning.reasoning}"
            trace.tool_use = self.stage_tool_use(first_action, ctx)
            tool_result = ""
            if trace.tool_use.tool_calls:
                tc = trace.tool_use.tool_calls[0]
                tool_result = f"{tc.tool_name} → {tc.return_value[:200]}"
            memory_query = f"{task} {first_action}"
            trace.memory = self.stage_memory(memory_query)
            trace.output = self.stage_output(
                task, trace.planning.plan,
                tool_result, trace.memory.context_used,
            )
            trace.task_success = True
        except Exception as e:
            logger.error(f"Pipeline error {task_id}: {e}")
            trace.task_success = False
        trace.total_latency_s = round(time.time() - t0, 2)
        return trace
