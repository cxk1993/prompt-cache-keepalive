"""Generic types for the provider-agnostic prompt-cache keepalive library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str | list[Any] = ""
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ChatConfig:
    """Configuration for a chat completion request."""

    max_tokens: int = 1
    temperature: float = 0.0
    timeout: float = 30.0
    model: str = ""
    # Provider-specific extra fields
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeResult:
    """Result of a keepalive probe."""

    ok: bool
    cache_read_tokens: int = 0
    error: str | None = None


class ProbeFn(Protocol):
    """Protocol for a provider-agnostic keepalive probe function.

    Implement this to send a lightweight probe request to your LLM provider.
    The function receives the messages and config that should be sent, and
    returns a ProbeResult indicating whether the cache was hit.

    Example OpenAI-compatible implementation:

        async def openai_probe(
            messages: list[Message],
            config: ChatConfig,
        ) -> ProbeResult:
            client = OpenAI(api_key=..., base_url=...)
            resp = await client.chat.completions.create(
                model=config.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=1,
            )
            cached = resp.usage and (resp.usage.prompt_tokens_details or {}).get(
                "cached_tokens", 0
            ) or 0
            return ProbeResult(ok=True, cache_read_tokens=cached)
    """

    async def __call__(
        self,
        messages: list[Message],
        config: ChatConfig,
    ) -> ProbeResult:
        ...