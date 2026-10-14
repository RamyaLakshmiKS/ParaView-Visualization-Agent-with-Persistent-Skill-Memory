#!/usr/bin/env python3
"""
groq_client.py — a thin Groq chat wrapper with EXACT token accounting.

Why exact tokens: the Groq API returns `usage` with real prompt/completion
token counts for every call. We use those directly rather than any proxy. This
is NOT a FLOPs/efficiency experiment — we only use token counts to (a) enforce a
deliberately small LIVE-CONTEXT budget so the context-as-a-file method is forced
to actually manage information, and (b) report how big the live context was each
turn. No FLOPs, no cost modeling.

The key is read from the repo-root .env (GROQ_API_KEY=...). The key value is
never printed.
"""
from __future__ import annotations

import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple

import requests

MODEL = "qwen/qwen3.8-27b"
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def _load_key() -> str:
    """Parse GROQ_API_KEY from the repo-root .env without echoing it."""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    if not ENV_PATH.exists():
        raise RuntimeError(f"no .env at {ENV_PATH} and GROQ_API_KEY not set")
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "GROQ_API_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError("GROQ_API_KEY not found in .env")


class GroqChat:
    """One Groq chat endpoint, reused across turns."""

    def __init__(self, model: str = MODEL, temperature: float = 0.3,
                 max_tokens: int = 3000, timeout: int = 120):
        self._key = _load_key()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # running totals, for reporting only (not an efficiency claim)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    def complete(self, messages: List[Dict], temperature: Optional[float] = None,
                 max_retries: int = 6) -> Tuple[str, Dict]:
        """Return (assistant_text, usage_dict). usage has exact token counts."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json"}
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(BASE_URL, headers=headers, json=payload,
                                  timeout=self.timeout)
                if r.status_code == 429:
                    # honor Retry-After if present, else exponential backoff
                    ra = r.headers.get("retry-after")
                    try:
                        wait = float(ra) if ra else min(60, 2 ** attempt * 3)
                    except ValueError:
                        wait = min(60, 2 ** attempt * 3)
                    time.sleep(wait + 1)
                    last_err = f"429 rate limit (attempt {attempt+1}, waited {wait:.0f}s)"
                    continue
                r.raise_for_status()
                j = r.json()
                text = j["choices"][0]["message"]["content"] or ""
                usage = j.get("usage", {}) or {}
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                self.n_calls += 1
                return text, usage
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError(f"Groq call failed after {max_retries} tries: {last_err}")


def count_prompt_tokens(messages: List[Dict], model: str = MODEL) -> int:
    """Exact prompt-token count via a cheap zero-generation probe call.

    Groq returns `prompt_tokens` in usage; by asking for max_tokens=1 we get the
    server's exact tokenization of the messages. Used to measure live-context
    size each turn without guessing. Falls back to a chars/4 estimate on error.
    """
    try:
        key = _load_key()
        r = requests.post(
            BASE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": 1, "temperature": 0},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("usage", {}).get("prompt_tokens", 0)
    except Exception:
        chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, round(chars / 4))


if __name__ == "__main__":
    # smoke test (prints no secrets)
    c = GroqChat()
    txt, usage = c.complete([{"role": "user", "content": "Reply with exactly: OK"}])
    print("reply:", repr(txt[:80]))
    print("usage:", usage)
