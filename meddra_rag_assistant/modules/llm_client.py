from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

try:  # Optional .env support
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency optional
    load_dotenv = None


def _load_default_env() -> None:
    """Load environment variables from common .env locations."""
    env_paths = [
        Path(".env"),
        Path(__file__).resolve().parents[1] / ".env",
    ]

    if load_dotenv is not None:
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        return

    # Fallback loader if python-dotenv is not installed.
    for env_path in env_paths:
        if not env_path.exists():
            continue
        with env_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_default_env()

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai optional for local-only setups
    OpenAI = None  # type: ignore


@dataclass
class LLMConfig:
    backend: str
    openai: Dict[str, Any]
    ollama: Dict[str, Any]
    openrouter: Dict[str, Any]


class LLMClient:
    """Abstraction layer for OpenAI, Ollama, and OpenRouter LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.backend = config.backend.lower()
        self._openai_client: Optional[OpenAI] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LLMClient":
        backend = config.get("backend", "openai")
        openai_cfg = config.get("openai", {})
        ollama_cfg = config.get("ollama", {})
        openrouter_cfg = config.get("openrouter", {})
        return cls(
            LLMConfig(
                backend=backend,
                openai=openai_cfg,
                ollama=ollama_cfg,
                openrouter=openrouter_cfg,
            )
        )

    @property
    def openai_client(self) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Install it or switch backend to 'ollama'.")
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        if self.backend == "openai":
            return self._generate_openai(prompt, system_prompt=system_prompt, **kwargs)
        if self.backend == "ollama":
            return self._generate_ollama(prompt, system_prompt=system_prompt, **kwargs)
        if self.backend == "openrouter":
            return self._generate_openrouter(prompt, system_prompt=system_prompt, **kwargs)
        raise ValueError(f"Unsupported LLM backend: {self.backend}")

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _generate_openai(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        model = self.config.openai.get("model", "gpt-4o-mini")
        temperature = kwargs.get("temperature", self.config.openai.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens", self.config.openai.get("max_tokens"))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _generate_ollama(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        model = self.config.ollama.get("model", "mistral")
        options = dict(self.config.ollama.get("options", {}))
        options.update(kwargs.get("options", {}))

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": options,
        }
        url = self.config.ollama.get("url", "http://localhost:11434/api/generate")
        timeout = self.config.ollama.get("timeout", 120)

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        if "response" in data:
            return str(data["response"]).strip()
        if "message" in data:
            return str(data["message"]).strip()
        return str(data).strip()

    def _generate_openrouter(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        cfg = self.config.openrouter
        model = cfg.get("model", "meta-llama/llama-3.1-8b-instruct")
        temperature = kwargs.get("temperature", cfg.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens", cfg.get("max_tokens"))

        api_key = cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Provide it via environment variable or config.llm.openrouter.api_key."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        extra = cfg.get("extra_body", {})
        if extra:
            payload.update(extra)

        url = cfg.get("url", "https://openrouter.ai/api/v1/chat/completions")
        timeout = cfg.get("timeout", 120)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        header_overrides = cfg.get("headers", {})
        if header_overrides:
            headers.update(header_overrides)

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter response missing choices: {data}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return str(content).strip()
