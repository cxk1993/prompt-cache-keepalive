"""prompt-cache-keepalive — Keep LLM prompt caches warm across idle gaps.

Provider-agnostic keepalive loop for LLM prompt caches. Extracted from
OpenSquilla PR #1036, which was built on the original design by @cxk1993
(PR #1015).

The core insight: most LLM providers have a ~5-minute (300s) prompt cache TTL.
If you send a lightweight probe within 80% of that TTL, the next real turn
reads from cache instead of re-creating it. This library automates that.
"""

from __future__ import annotations

from ._candidate import PromptCacheKeepaliveCandidate
from ._service import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    DEFAULT_TTL_SECONDS,
    MAX_IDLE_TIMEOUT_SECONDS,
    MAX_TTL_SECONDS,
    MIN_IDLE_TIMEOUT_SECONDS,
    MIN_TTL_SECONDS,
    PROBE_RATIO,
    PromptCacheKeepaliveService,
)
from ._types import ChatConfig, Message, ProbeFn, ProbeResult

__all__ = [
    "ChatConfig",
    "DEFAULT_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_TTL_SECONDS",
    "MAX_IDLE_TIMEOUT_SECONDS",
    "MAX_TTL_SECONDS",
    "MIN_IDLE_TIMEOUT_SECONDS",
    "MIN_TTL_SECONDS",
    "PROBE_RATIO",
    "Message",
    "ProbeFn",
    "ProbeResult",
    "PromptCacheKeepaliveCandidate",
    "PromptCacheKeepaliveService",
]