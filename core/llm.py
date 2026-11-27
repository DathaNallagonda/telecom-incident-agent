"""LLM client: Gemini main, Groq fallback, disk cache and call logging.

Decision #10 in the plan doc: gemini-3.8-flash is the main model and
openai/gpt-oss-120b on Groq is the fallback. Timeouts are short and retries
are few, because Flash models sometimes return 503/504 under high demand and
we would rather fall back than stall.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd, load
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Extra Gemini hop: 3.8 Flash often hits free-tier quota (429) or 503s.
GEMINI_BACKUP_MODEL = os.getenv("GEMINI_BACKUP_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))


def _providers() -> list[str]:
    """Model chain, e.g. LLM_PROVIDERS=groq pins the evaluation to one model."""
    return [p.strip() for p in os.getenv("LLM_PROVIDERS", "gemini,groq").split(",") if p.strip()]


def _max_retries() -> int:
    return int(os.getenv("LLM_MAX_RETRIES", "1"))


LOG_PATH = Path(os.getenv("LLM_LOG_PATH", ROOT / "logs" / "llm_calls.jsonl"))
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ROOT / ".cache" / "llm"))


class LLMUnavailableError(RuntimeError):
    """No API key is configured for any provider."""


def _secret(name: str) -> str | None:
    """Read a key from the environment, then from Streamlit secrets if running there."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


# --- disk cache --------------------------------------------------------------
class DiskCache(BaseCache):
    """Cache keyed by a hash of the prompt and the model settings.

    Saves free-tier quota during evaluation; temperature is 0, so a cached
    answer is what the model would most likely have said anyway.
    """

    def __init__(self, directory: Path = CACHE_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, prompt: str, llm_string: str) -> Path:
        key = hashlib.sha256(f"{llm_string}\n{prompt}".encode()).hexdigest()
        return self.dir / f"{key}.json"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        path = self._path(prompt, llm_string)
        if not path.exists():
            return None
        try:
            return [load(g) for g in json.loads(path.read_text(encoding="utf-8"))]
        except Exception:
            return None  # a corrupt entry is just a miss

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        path = self._path(prompt, llm_string)
        path.write_text(json.dumps([dumpd(g) for g in return_val]), encoding="utf-8")

    def clear(self, **kwargs: Any) -> None:
        for p in self.dir.glob("*.json"):
            p.unlink()


def enable_cache(enabled: bool = True) -> None:
    set_llm_cache(DiskCache() if enabled else None)


# --- call logging ------------------------------------------------------------
class CallLogger(BaseCallbackHandler):
    """Append one JSON line per LLM call: model, tokens, latency, error."""

    def __init__(self, path: Path = LOG_PATH) -> None:
        self.path = Path(path)
        self._start: dict[UUID, tuple[float, str]] = {}

    def _model(self, serialized: dict, kwargs: dict) -> str:
        params = kwargs.get("invocation_params") or {}
        return (
            (kwargs.get("metadata") or {}).get("ls_model_name")
            or params.get("model")
            or params.get("model_name")
            or (serialized.get("kwargs") or {}).get("model")
            or "unknown"
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):  # noqa: ANN001
        self._start[run_id] = (time.perf_counter(), self._model(serialized or {}, kwargs))

    def _write(self, record: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # logging must never break a run (e.g. read-only filesystem)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start, model = self._start.pop(run_id, (time.perf_counter(), "unknown"))
        usage: dict = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            pass
        self._write(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": model,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "latency_ms": round((time.perf_counter() - start) * 1000),
                "ok": True,
            }
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        start, model = self._start.pop(run_id, (time.perf_counter(), "unknown"))
        self._write(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": model,
                "latency_ms": round((time.perf_counter() - start) * 1000),
                "ok": False,
                "error": f"{type(error).__name__}: {str(error)[:200]}",
            }
        )


LOGGER = CallLogger()


# --- models ------------------------------------------------------------------
def _gemini(model: str = GEMINI_MODEL) -> BaseChatModel | None:
    key = _secret("GEMINI_API_KEY")
    if not key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0,
        timeout=TIMEOUT_S,
        max_retries=_max_retries(),
        callbacks=[LOGGER],
    )


def _groq() -> BaseChatModel | None:
    key = _secret("GROQ_API_KEY")
    if not key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=key,
        temperature=0,
        timeout=TIMEOUT_S,
        max_retries=_max_retries(),
        callbacks=[LOGGER],
    )


def chat_models() -> list[BaseChatModel]:
    """Available models in priority order (default: Gemini 3.8, Gemini 3.6, Groq)."""
    candidates: list[BaseChatModel | None] = []
    for provider in _providers():
        if provider == "gemini":
            candidates += [_gemini(GEMINI_MODEL), _gemini(GEMINI_BACKUP_MODEL)]
        elif provider == "groq":
            candidates.append(_groq())
    models = [m for m in candidates if m is not None]
    if not models:
        raise LLMUnavailableError(
            "No LLM API key found. Set GEMINI_API_KEY and/or GROQ_API_KEY in .env "
            "(or in the Streamlit app's Secrets)."
        )
    return models


def with_fallback(build: Any) -> Runnable:
    """Apply `build(model)` to every available model and chain them as fallbacks.

    Example: with_fallback(lambda m: m.with_structured_output(Schema))
    """
    runnables: Sequence[Runnable] = [build(m) for m in chat_models()]
    if len(runnables) == 1:
        return runnables[0]
    return runnables[0].with_fallbacks(list(runnables[1:]))


def get_llm() -> Runnable:
    """Plain chat model with fallback."""
    return with_fallback(lambda m: m)


def structured(schema: type, include_raw: bool = False) -> Runnable:
    """Chat model forced to return `schema`, with fallback.

    Groq's gpt-oss models often answer in prose when a tool call is forced,
    so Groq uses strict JSON-schema mode instead of function calling.
    """

    def build(m: BaseChatModel) -> Runnable:
        if m.__class__.__name__ == "ChatGroq":
            return m.with_structured_output(schema, method="json_schema", include_raw=include_raw)
        return m.with_structured_output(schema, include_raw=include_raw)

    return with_fallback(build)


enable_cache(os.getenv("LLM_CACHE", "1") != "0")
