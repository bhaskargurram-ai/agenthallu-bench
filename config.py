"""Central config for all models, paths, and experiment settings."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    # TIER 1: Frontier (use sparingly — 50 tasks each for comparison only)
    "gpt4o": {
        "api": "openai",
        "model_id": "gpt-4o",
        "max_tokens": 2048,
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 10.00,
        "tier": "frontier",
        "task_limit": 50,
    },
    "gemini_25_flash": {
        "api": "google",
        "model_id": "gemini-2.5-flash",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.30,
        "cost_per_1m_output": 2.50,
        "tier": "frontier",
        "task_limit": 50,
    },
    "o3_mini": {
        "api": "openai",
        "model_id": "o3-mini",
        "max_tokens": 2048,
        "cost_per_1m_input": 1.10,
        "cost_per_1m_output": 4.40,
        "tier": "frontier",
        "task_limit": 50,
    },
    # TIER 2: Efficient (main workhorses)
    "gpt4o_mini": {
        "api": "openai",
        "model_id": "gpt-4o-mini",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "tier": "efficient",
        "task_limit": 500,
    },
    "gemini_20_flash": {
        "api": "google",
        "model_id": "gemini-2.0-flash",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.40,
        "tier": "efficient",
        "task_limit": 500,
    },
    "gpt35_turbo": {
        "api": "openai",
        "model_id": "gpt-3.5-turbo",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.50,
        "cost_per_1m_output": 1.50,
        "tier": "efficient",
        "task_limit": 500,
    },
    "gpt41_mini": {
        "api": "openai",
        "model_id": "gpt-4.1-mini",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.40,
        "cost_per_1m_output": 1.60,
        "tier": "efficient",
        "task_limit": 500,
    },
    "gpt41_nano": {
        "api": "openai",
        "model_id": "gpt-4.1-nano",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.40,
        "tier": "efficient",
        "task_limit": 500,
    },
    # TIER 3: Open source (via OpenRouter)
    "deepseek_v3": {
        "api": "openrouter",
        "model_id": "deepseek/deepseek-chat",
        "max_tokens": 2048,
        "cost_per_1m_input": 0.27,
        "cost_per_1m_output": 1.10,
        "tier": "open",
        "task_limit": 300,
    },
    # TIER 4: Open-weight, served locally via vLLM (OpenAI-compatible endpoint).
    # model_id == served-model-name given to vLLM; hf_repo is the download source.
    # cost=0 (local GPU). Same prompted-ReAct harness as all other tiers.
    "qwen25_3b": {
        "api": "local", "model_id": "qwen25_3b", "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "",
    },
    "qwen25_7b": {
        "api": "local", "model_id": "qwen25_7b", "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "",
    },
    "qwen25_14b": {
        "api": "local", "model_id": "qwen25_14b", "hf_repo": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "--quantization awq_marlin",
    },
    "phi35_mini": {
        "api": "local", "model_id": "phi35_mini", "hf_repo": "microsoft/Phi-3.5-mini-instruct",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "--max-model-len 16384",
    },
    "mistral_7b": {
        "api": "local", "model_id": "mistral_7b", "hf_repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "",
    },
    # Non-Qwen open-weight candidates (different labs) for cross-family generalization.
    "hermes3_llama8b": {
        "api": "local", "model_id": "hermes3_llama8b", "hf_repo": "NousResearch/Hermes-3-Llama-3.1-8B",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "--max-model-len 8192",
    },
    "internlm25_7b": {
        "api": "local", "model_id": "internlm25_7b", "hf_repo": "internlm/internlm2_5-7b-chat",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "--max-model-len 8192 --trust-remote-code",
    },
    # Gated on HuggingFace — requires HF_TOKEN with accepted license. Served only if token present.
    "llama31_8b": {
        "api": "local", "model_id": "llama31_8b", "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "max_tokens": 2048, "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0,
        "tier": "open_local", "task_limit": 500, "vllm_extra": "", "gated": True,
    },
}

# Model sets for different experiment phases (10 models total)
FULL_MODELS = ["gpt4o_mini", "gemini_20_flash", "deepseek_v3", "gpt35_turbo", "gpt41_mini", "gpt41_nano"]
FRONTIER_MODELS = ["gpt4o", "gemini_25_flash", "o3_mini"]
DEV_MODELS = ["gemini_20_flash"]
# Open-weight models served locally via vLLM. Ungated set runs without an HF token;
# llama31_8b is added only when HF_TOKEN (with accepted Meta license) is available.
LOCAL_MODELS = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "mistral_7b"]
LOCAL_MODELS_GATED = ["llama31_8b"]

# Budget
BUDGET_LIMIT_USD = float(os.getenv("BUDGET_LIMIT_USD", "50.0"))

# ── RAG settings ─────────────────────────────────────────────────────────────
RAG_CHUNK_SIZE = 512
RAG_TOP_K = 5
CHROMA_PERSIST_DIR = "./data/chroma"

# ── Benchmark settings ───────────────────────────────────────────────────────
TASKS_PER_DOMAIN = 150          # 150 x 3 domains = 450 tasks
MULTI_TURN_SESSION_LENGTH = 8   # turns per P3 session
MULTI_AGENT_CHAIN_LENGTH = 3    # agents in P4 chain

# ── Error injection rates ────────────────────────────────────────────────────
INJECTION_RATE = 1.0            # inject error in every task for experiments

# ── P2 parameter error types ─────────────────────────────────────────────────
PARAM_ERROR_TYPES = [
    "type_mismatch",            # string where int expected
    "out_of_range",             # value outside valid bounds
    "missing_required",         # omit required parameter
    "semantic_wrong",           # correct type, wrong meaning
]

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
TASKS_DIR = "./data/tasks"
TRACES_DIR = "./data/traces"
RESULTS_DIR = "./data/results"

# ── Trace DB ─────────────────────────────────────────────────────────────────
TRACE_DB_PATH = "./data/traces/traces.db"
