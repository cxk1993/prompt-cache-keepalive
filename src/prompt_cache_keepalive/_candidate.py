"""Storage-free handoff for session prompt-cache keepalive candidates.

The candidate captures only a provider-visible prefix that has already appeared
at the start of a successful real request. The service may later replay that
prefix as an auxiliary provider call; the candidate itself never mutates
session history and contains no persistence behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._types import ChatConfig, Message


@dataclass(frozen=True, slots=True)
class PromptCacheKeepaliveCandidate:
    """Immutable-enough snapshot of one proven provider request prefix.

    Captured from a successful real turn. Contains everything needed to
    replay a lightweight probe that keeps the provider's prompt cache warm.
    """

    session_key: str
    provider_id: str
    model: str
    messages: tuple[Message, ...]
    tools: tuple[Any, ...]
    config: ChatConfig


__all__ = ["PromptCacheKeepaliveCandidate"]