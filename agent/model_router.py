"""Unified model router for OpenAI, Google Gemini, and OpenRouter APIs.

Provides a single complete() method that routes to the correct backend
based on the model config's "api" field.
"""

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes LLM calls to the correct API backend."""

    def __init__(self, model_key: str):
        from config import MODELS
        if model_key not in MODELS:
            raise ValueError(f"Unknown model key: '{model_key}'. Available: {list(MODELS.keys())}")
        self.model_key = model_key
        self.model = MODELS[model_key]
        self.api = self.model["api"]
        logger.info("ModelRouter initialized: %s → %s (%s)", model_key, self.model["model_id"], self.api)

    def complete(self, messages: list[dict], **kwargs) -> dict:
        """Send messages to the model and return structured result.

        Returns:
            {
                "text": str,
                "input_tokens": int,
                "output_tokens": int,
                "cost": float,
                "latency_ms": float,
            }
        """
        if self.api == "openai":
            return self._call_openai(messages, **kwargs)
        elif self.api == "google":
            return self._call_google(messages, **kwargs)
        elif self.api == "openrouter":
            return self._call_openrouter(messages, **kwargs)
        else:
            raise ValueError(f"Unknown API backend: '{self.api}'")

    def _compute_cost(self, usage) -> float:
        """Compute cost in USD from token usage."""
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = (
            input_tokens * self.model["cost_per_1m_input"] / 1_000_000
            + output_tokens * self.model["cost_per_1m_output"] / 1_000_000
        )
        return cost

    def _call_openai(self, messages: list[dict], **kwargs) -> dict:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_id = self.model["model_id"]
        max_tok = kwargs.get("max_tokens", self.model.get("max_tokens", 2048))

        # o3/o1 reasoning models use max_completion_tokens, not max_tokens
        is_reasoning = model_id.startswith(("o1", "o3", "o4"))
        create_kwargs = {
            "model": model_id,
            "messages": messages,
        }
        if is_reasoning:
            create_kwargs["max_completion_tokens"] = max_tok
        else:
            create_kwargs["max_tokens"] = max_tok
            create_kwargs["temperature"] = kwargs.get("temperature", 0.0)

        start = time.time()
        response = client.chat.completions.create(**create_kwargs)
        latency_ms = (time.time() - start) * 1000
        return {
            "text": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cost": self._compute_cost(response.usage),
            "latency_ms": latency_ms,
        }

    def _call_google(self, messages: list[dict], **kwargs) -> dict:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(self.model["model_id"])

        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        system_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_text = content
            elif role == "user":
                text = f"{system_text}\n\n{content}" if system_text and not gemini_contents else content
                gemini_contents.append({"role": "user", "parts": [text]})
                if system_text and not len(gemini_contents) > 1:
                    system_text = ""  # Only prepend system to first user message
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [content]})

        start = time.time()
        response = model.generate_content(
            gemini_contents,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.model.get("max_tokens", 2048)),
                temperature=kwargs.get("temperature", 0.0),
            ),
        )
        latency_ms = (time.time() - start) * 1000

        # Extract token counts from usage metadata
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cost = (
            input_tokens * self.model["cost_per_1m_input"] / 1_000_000
            + output_tokens * self.model["cost_per_1m_output"] / 1_000_000
        )

        return {
            "text": response.text or "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
        }

    def _call_openrouter(self, messages: list[dict], **kwargs) -> dict:
        """OpenRouter uses identical API to OpenAI — just different base_url."""
        import openai
        client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        start = time.time()
        response = client.chat.completions.create(
            model=self.model["model_id"],
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.model.get("max_tokens", 2048)),
            temperature=kwargs.get("temperature", 0.0),
            extra_headers={
                "HTTP-Referer": "https://github.com/agenthallu-bench",
                "X-Title": "AgentHallu-Bench Research",
            },
        )
        latency_ms = (time.time() - start) * 1000
        return {
            "text": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cost": self._compute_cost(response.usage),
            "latency_ms": latency_ms,
        }
