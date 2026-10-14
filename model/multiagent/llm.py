"""LLM provider factory, scripted mock model, and failure classification.

Providers are configured via llm.ini ([LLM] section); secrets are read
from environment variables named by ``api_key_env`` and never stored in the
tracked config file.

    provider = openai_compatible | azure | mock

``mock`` is reserved for tests/offline replay: a ScriptedChatModel whose
responses come from a caller-supplied responder function.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


class StructuredResponseError(RuntimeError):
    """Raised when a structured-output call exhausts its retries.

    ``category`` is one of the RQ2 failure categories (PROVIDER_FAILURE or
    PARSE_FAILURE) and is propagated verbatim into the run's prediction record.
    """

    def __init__(self, category: str, detail: str):
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


def classify_llm_error(exc: BaseException) -> str:
    """Map an exception from a structured/chat call to a failure category."""
    try:
        from openai import APIError, APITimeoutError, APIConnectionError
    except Exception:  # pragma: no cover - openai is always installed here
        APIError = APITimeoutError = APIConnectionError = ()  # type: ignore
    try:
        from pydantic import ValidationError
    except Exception:  # pragma: no cover
        ValidationError = ()  # type: ignore

    if ValidationError and isinstance(exc, ValidationError):
        return "PARSE_FAILURE"
    if APIError and isinstance(exc, (APIError, APITimeoutError, APIConnectionError)):
        return "PROVIDER_FAILURE"
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "connection", "ratelimit", "api")):
        return "PROVIDER_FAILURE"
    if any(k in name for k in ("validation", "json", "parse", "schema")):
        return "PARSE_FAILURE"
    return "PROVIDER_FAILURE"


def _resolve_api_key(sec, key_env: str) -> str:
    """API key resolution: environment variable (preferred) or the local
    ``api_key`` entry in llm.ini (untracked/local use only)."""
    return os.environ.get(key_env) or sec.get("api_key", "").strip()


def load_llm(config) -> BaseChatModel:
    """Build a chat model from the [LLM] section of the merged config."""
    sec = config["LLM"]
    provider = sec.get("provider", "azure").strip().lower()
    temperature = float(sec.get("temperature", 0))
    timeout = float(sec.get("request_timeout", 60))
    max_retries = int(sec.get("request_retries", 5))

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        key_env = sec.get("api_key_env", "OPENAI_API_KEY")
        api_key = _resolve_api_key(sec, key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found: set the {key_env} environment variable, "
                "or fill api_key in config/llm.ini."
            )
        kwargs: Dict[str, Any] = {}
        base_url = sec.get("base_url", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(
            model=sec["deployment_name"],
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        key_env = sec.get("api_key_env", "OPENAI_API_KEY")
        api_key = _resolve_api_key(sec, key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found: set the {key_env} environment variable, "
                "or fill api_key in config/llm.ini."
            )
        endpoint = os.environ.get("OPENAI_ENDPOINT", sec.get("endpoint", ""))
        return AzureChatOpenAI(
            deployment_name=sec["deployment_name"],
            openai_api_version=sec["openai_api_version"],
            api_key=api_key,
            azure_endpoint=endpoint,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    if provider == "mock":
        raise RuntimeError(
            "provider=mock must be injected programmatically "
            "(ScriptedChatModel); it cannot be built from config."
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Scripted mock model (tests / offline replay)
# ---------------------------------------------------------------------------

# A responder receives (role, call_index, messages, mode) where mode is
# "structured" (must return a dict coercible to the requested schema) or
# "generate" (must return an AIMessage, possibly with tool_calls).
Responder = Callable[[str, int, List[BaseMessage], str], Any]
RoleDetector = Callable[[List[BaseMessage]], str]


def default_role_detector(messages: List[BaseMessage]) -> str:
    """Detect the speaking agent from marker lines in the system prompt.

    Payment prompts embed an explicit ``Role: <NAME>`` line in every member
    prompt; supervisor prompts contain the member roster, which identifies
    the team.
    """
    text = "\n".join(str(m.content) for m in messages[:3])
    if "You are a supervisor tasked with managing a conversation" in text:
        if "Payment Action Generator" in text:
            return "Payment Main Supervisor"
        return "Payment Evaluation Supervisor"
    for line in text.splitlines():
        if line.startswith("Role: "):
            return line.split("Role: ", 1)[1].strip()
    return "UNKNOWN"


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model driven by a caller-supplied responder script."""

    responder: Responder
    role_detector: RoleDetector = default_role_detector
    call_counts: Dict[str, int] = Field(default_factory=dict)
    bound_tool_names: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # -- internal helpers ---------------------------------------------------
    def _next_index(self, role: str, mode: str) -> int:
        key = f"{role}|{mode}"
        idx = self.call_counts.get(key, 0)
        self.call_counts[key] = idx + 1
        return idx

    @staticmethod
    def _to_messages(inp: Any) -> List[BaseMessage]:
        if hasattr(inp, "to_messages"):
            return list(inp.to_messages())
        return list(inp)

    # -- BaseChatModel interface ---------------------------------------------
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        role = self.role_detector(messages)
        idx = self._next_index(role, "generate")
        out = self.responder(role, idx, list(messages), "generate")
        if isinstance(out, AIMessage):
            message = out
        else:
            message = AIMessage(content=str(out))
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:  # pragma: no cover - cosmetic
        return "scripted-mock"

    # -- capability shims ------------------------------------------------------
    def bind_tools(self, tools: Sequence, **kwargs):
        self.bound_tool_names = [getattr(t, "name", str(t)) for t in tools]
        return self

    def with_structured_output(self, schema: Type[BaseModel], **kwargs):
        def _call(inp: Any):
            messages = self._to_messages(inp)
            role = self.role_detector(messages)
            idx = self._next_index(role, "structured")
            out = self.responder(role, idx, messages, "structured")
            if isinstance(out, schema):
                return out
            if isinstance(out, dict):
                return schema(**out)
            raise TypeError(
                f"scripted response for role={role} idx={idx} is not "
                f"{schema.__name__}-compatible: {type(out)}"
            )

        return RunnableLambda(_call)
