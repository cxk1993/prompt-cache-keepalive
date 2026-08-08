"""Opt-in, session-scoped provider prompt-cache keepalive service.

Extracted from OpenSquilla PR #1036 (built on the original design by @cxk1993
in PR #1015) and decoupled from the OpenSquilla runtime. Provider-agnostic:
the caller injects a ``ProbeFn`` that knows how to send a lightweight probe
request to their LLM provider.

Design (kept from the original):
- Leases and candidate snapshots are process memory only. A restart returns
  every session to the safe default (off).
- A keepalive cycle is bounded by an idle window (default 60 min). The probe
  fires at 80% of the configured TTL, so with a 5-min TTL you get a probe
  about every 4 minutes and at most ~14 probes before automatic pause.
- A real user turn enqueued by the caller cancels an in-flight probe.
- A cache miss, missing cache telemetry, or provider error stops the lease.
- Supports a single global auxiliary slot (concurrency guard) so probes never
  compete with real turns.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ._candidate import PromptCacheKeepaliveCandidate
from ._types import ChatConfig, Message, ProbeFn, ProbeResult

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300
MIN_TTL_SECONDS = 300
MAX_TTL_SECONDS = 86_400
DEFAULT_IDLE_TIMEOUT_SECONDS = 3_600
MIN_IDLE_TIMEOUT_SECONDS = 300
MAX_IDLE_TIMEOUT_SECONDS = 86_400
PROBE_RATIO = 0.8
_MAX_SNAPSHOT_CHARS = 2_000_000
_PROBE_MESSAGE = "prompt-cache-keepalive probe; reply with a single token."


@dataclass(slots=True)
class _Lease:
    ttl_seconds: int
    idle_timeout_seconds: int
    generation: int = 0
    enabled: bool = True
    state: str = "waiting"
    reason: str | None = "waiting_for_stable_prefix"
    candidate: PromptCacheKeepaliveCandidate | None = None
    timer_task: asyncio.Task[None] | None = None
    probe_task: asyncio.Task[None] | None = None
    next_probe_at_ms: int | None = None
    last_probe_at_ms: int | None = None
    last_cache_hit_tokens: int = 0
    idle_deadline_monotonic: float | None = None
    idle_expires_at_ms: int | None = None


class PromptCacheKeepaliveService:
    """Keep a proven prompt prefix warm without creating an Agent turn.

    Args:
        probe_fn: Async callable that sends a lightweight probe to the LLM
            provider and returns how many cached tokens were read.
        global_slot: If provided, an ``asyncio.Semaphore`` that serializes
            auxiliary probes. A single shared instance means only one probe
            runs at a time across all sessions, matching the original design.
    """

    def __init__(
        self,
        *,
        probe_fn: ProbeFn,
        global_slot: asyncio.Semaphore | None = None,
    ) -> None:
        self._probe_fn = probe_fn
        self._global_slot = global_slot or asyncio.Semaphore(1)
        self._leases: dict[str, _Lease] = {}
        self._closed = False

    # ------------------------------------------------------------------ #
    # Public control surface
    # ------------------------------------------------------------------ #
    async def set_enabled(
        self,
        session_key: str,
        *,
        enabled: bool,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Enable or disable keepalive for a session. Returns the new status."""
        key = self._canonicalize(session_key)
        previous = self._leases.get(key)
        if previous is not None:
            previous.generation += 1
            self._cancel_timer(previous)
            self._cancel_probe(previous)
        if not enabled:
            self._leases.pop(key, None)
            return self.status(key)

        self._leases[key] = _Lease(
            ttl_seconds=ttl_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        return self.status(key)

    def is_enabled(self, session_key: str) -> bool:
        lease = self._leases.get(self._canonicalize(session_key))
        return bool(lease is not None and lease.enabled)

    def renew(
        self,
        session_key: str,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        """(Re)arm the idle window for an existing lease. Call this on every
        successful real turn so the idle timeout resets. Lightweight, sync."""
        key = self._canonicalize(session_key)
        lease = self._leases.get(key)
        if lease is None or not lease.enabled:
            return
        lease.ttl_seconds = ttl_seconds
        lease.idle_timeout_seconds = idle_timeout_seconds
        lease.idle_deadline_monotonic = time.monotonic() + idle_timeout_seconds
        lease.idle_expires_at_ms = int((time.time() + idle_timeout_seconds) * 1000)

    def record_candidate(self, candidate: PromptCacheKeepaliveCandidate) -> None:
        """Capture a successful real-turn prefix only for an armed lease."""
        if self._closed:
            return
        key = self._canonicalize(candidate.session_key)
        lease = self._leases.get(key)
        if lease is None or not lease.enabled:
            return
        safe_candidate, reason = self._safe_copy(candidate)
        lease.generation += 1
        self._cancel_timer(lease)
        self._cancel_probe(lease)
        if safe_candidate is None:
            lease.candidate = None
            lease.state = "waiting"
            lease.reason = reason
            lease.next_probe_at_ms = None
            lease.idle_deadline_monotonic = None
            lease.idle_expires_at_ms = None
            return
        lease.candidate = safe_candidate
        lease.state = "scheduled"
        lease.reason = None
        lease.idle_deadline_monotonic = time.monotonic() + lease.idle_timeout_seconds
        lease.idle_expires_at_ms = int(
            (time.time() + lease.idle_timeout_seconds) * 1000
        )
        self._schedule(key, lease, delay=lease.ttl_seconds * PROBE_RATIO)

    def refresh_required(
        self, session_key: str, reason: str = "session_history_changed"
    ) -> None:
        """Keep the opt-in setting but discard a stale provider prefix."""
        key = self._canonicalize(session_key)
        lease = self._leases.get(key)
        if lease is None or not lease.enabled:
            return
        lease.generation += 1
        self._cancel_timer(lease)
        self._cancel_probe(lease)
        lease.candidate = None
        lease.state = "waiting"
        lease.reason = reason
        lease.idle_deadline_monotonic = None
        lease.idle_expires_at_ms = None

    def status(self, session_key: str) -> dict[str, Any]:
        key = self._canonicalize(session_key)
        lease = self._leases.get(key)
        if lease is None:
            return {
                "enabled": False,
                "ttlSeconds": DEFAULT_TTL_SECONDS,
                "intervalSeconds": int(DEFAULT_TTL_SECONDS * PROBE_RATIO),
                "idleTimeoutSeconds": DEFAULT_IDLE_TIMEOUT_SECONDS,
                "idleExpiresAt": None,
                "state": "off",
                "reason": None,
                "hasSnapshot": False,
                "nextProbeAt": None,
                "lastProbeAt": None,
                "lastCacheHitTokens": 0,
                "provider": None,
                "model": None,
            }
        candidate = lease.candidate
        return {
            "enabled": lease.enabled,
            "ttlSeconds": lease.ttl_seconds,
            "intervalSeconds": int(lease.ttl_seconds * PROBE_RATIO),
            "idleTimeoutSeconds": lease.idle_timeout_seconds,
            "idleExpiresAt": lease.idle_expires_at_ms,
            "state": lease.state,
            "reason": lease.reason,
            "hasSnapshot": candidate is not None,
            "nextProbeAt": lease.next_probe_at_ms,
            "lastProbeAt": lease.last_probe_at_ms,
            "lastCacheHitTokens": lease.last_cache_hit_tokens,
            "provider": candidate.provider_id if candidate is not None else None,
            "model": candidate.model if candidate is not None else None,
        }

    async def invalidate(self, session_key: str) -> None:
        """Drop an armed lease when the underlying session identity changes."""
        await self.set_enabled(session_key, enabled=False)

    async def close(self) -> None:
        """Cancel all timers and probes and release leases."""
        if self._closed:
            return
        self._closed = True
        tasks: list[asyncio.Task[None]] = []
        for lease in self._leases.values():
            lease.generation += 1
            if lease.timer_task is not None and not lease.timer_task.done():
                lease.timer_task.cancel()
                tasks.append(lease.timer_task)
            if lease.probe_task is not None and not lease.probe_task.done():
                lease.probe_task.cancel()
                tasks.append(lease.probe_task)
        self._leases.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _canonicalize(session_key: str) -> str:
        key = session_key.strip()
        if not key:
            raise ValueError("session_key must be a non-empty string")
        return key

    def _lease_valid(self, key: str, lease: _Lease, generation: int) -> bool:
        """Check if the lease is still valid for the given generation."""
        current = self._leases.get(key)
        return (
            current is not None
            and current is lease
            and current.generation == generation
            and current.enabled
            and current.candidate is not None
        )

    @staticmethod
    def _cancel_timer(lease: _Lease) -> None:
        task = lease.timer_task
        lease.timer_task = None
        lease.next_probe_at_ms = None
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            task.cancel()

    @staticmethod
    def _cancel_probe(lease: _Lease) -> None:
        task = lease.probe_task
        lease.probe_task = None
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            task.cancel()

    def _safe_copy(
        self, candidate: PromptCacheKeepaliveCandidate
    ) -> tuple[PromptCacheKeepaliveCandidate | None, str | None]:
        """Reject candidates that cannot be safely replayed as a probe."""
        if candidate.config.extra.get("thinking"):
            return None, "thinking_requests_are_not_supported"
        if candidate.config.extra.get("output_json_schema") is not None:
            return None, "structured_output_requests_are_not_supported"
        if any(
            getattr(block, "type", "") in {"image", "document", "thinking"}
            for message in candidate.messages
            if isinstance(message.content, list)
            for block in message.content
        ):
            return None, "multimodal_or_thinking_history_is_not_supported"
        try:
            messages = tuple(
                self._copy_message(message) for message in candidate.messages
            )
            tools = tuple(candidate.tools)
            config = ChatConfig(
                max_tokens=candidate.config.max_tokens,
                temperature=candidate.config.temperature,
                timeout=candidate.config.timeout,
                model=candidate.config.model,
                extra=dict(candidate.config.extra),
            )
            snapshot_chars = sum(self._message_chars(m) for m in messages)
            snapshot_chars += sum(len(str(t)) for t in tools)
        except Exception:
            return None, "snapshot_copy_failed"
        if snapshot_chars > _MAX_SNAPSHOT_CHARS:
            return None, "stable_prefix_too_large"
        return (
            PromptCacheKeepaliveCandidate(
                session_key=candidate.session_key,
                provider_id=candidate.provider_id,
                model=candidate.model,
                messages=messages,
                tools=tools,
                config=config,
            ),
            None,
        )

    @staticmethod
    def _copy_message(message: Message) -> Message:
        content = message.content
        if isinstance(content, list):
            content = list(content)
        return Message(
            role=message.role,
            content=content,
            name=message.name,
            tool_calls=list(message.tool_calls) if message.tool_calls else None,
            tool_call_id=message.tool_call_id,
        )

    @staticmethod
    def _message_chars(message: Message) -> int:
        if isinstance(message.content, str):
            return len(message.content)
        return len(str(message.content))

    def _schedule(self, key: str, lease: _Lease, *, delay: float) -> None:
        if self._closed or not lease.enabled or lease.candidate is None:
            return
        generation = lease.generation
        delay = max(0.0, float(delay))
        wake_for_idle_expiry = False
        if lease.idle_deadline_monotonic is not None:
            remaining = max(0.0, lease.idle_deadline_monotonic - time.monotonic())
            if delay >= remaining:
                delay = remaining
                wake_for_idle_expiry = True
        lease.next_probe_at_ms = (
            None if wake_for_idle_expiry else int((time.time() + delay) * 1000)
        )
        lease.timer_task = asyncio.create_task(
            self._run_after(key, generation, delay, wake_for_idle_expiry),
            name=f"prompt-cache-keepalive-{key}",
        )

    async def _run_after(
        self,
        key: str,
        generation: int,
        delay: float,
        wake_for_idle_expiry: bool = False,
    ) -> None:
        try:
            await asyncio.sleep(delay)
            lease = self._leases.get(key)
            if (
                lease is None
                or not lease.enabled
                or lease.generation != generation
                or lease.candidate is None
            ):
                return
            if wake_for_idle_expiry or (
                lease.idle_deadline_monotonic is not None
                and time.monotonic() >= lease.idle_deadline_monotonic
            ):
                self._pause_for_idle_timeout(key, generation)
                return
            lease.state = "probing"
            lease.next_probe_at_ms = None
            lease.state = "scheduled"
            lease.reason = "waiting_for_slot"
            await self._run_probe(key, lease, generation)
        except asyncio.CancelledError:
            lease = self._leases.get(key)
            if lease is not None and lease.generation == generation and lease.enabled:
                lease.state = "waiting"
                lease.reason = "preempted_by_user_turn"
                lease.next_probe_at_ms = None
            raise
        except Exception as exc:  # noqa: BLE001 - auxiliary work is fail-closed
            self._stop(key, generation, "probe_failed")
            log.info("prompt_cache_keepalive.probe_failed", extra={"error": str(exc)})

    async def _run_probe(self, key: str, lease: _Lease, generation: int) -> None:
        candidate = lease.candidate
        if candidate is None:
            return
        async with self._global_slot:
            # Re-check after acquiring the slot — lease may have been invalidated
            if not self._lease_valid(key, lease, generation):
                return
            result = await self._probe_once(key, lease, generation)
            current = self._leases.get(key)
            if current is None or current.generation != generation or not current.enabled:
                return
            current.last_probe_at_ms = int(time.time() * 1000)
            current.last_cache_hit_tokens = result.cache_read_tokens
            if not result.ok:
                self._stop(key, generation, result.error or "provider_error")
                return
            if result.cache_read_tokens <= 0:
                self._stop(key, generation, "cache_miss_or_usage_unreported")
                return
            current.state = "scheduled"
            current.reason = None
            self._schedule(
                key,
                current,
                delay=current.ttl_seconds * PROBE_RATIO,
            )

    async def _probe_once(
        self, key: str, lease: _Lease, generation: int
    ) -> ProbeResult:
        candidate = lease.candidate
        if candidate is None:
            return ProbeResult(ok=False, error="no_candidate")
        messages = [
            Message(
                role=m.role,
                content=m.content,
                name=m.name,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
            )
            for m in candidate.messages
        ]
        messages.append(Message(role="user", content=_PROBE_MESSAGE))
        try:
            return await self._probe_fn(messages, candidate.config)
        except Exception as exc:  # noqa: BLE001
            log.info("prompt_cache_keepalive.probe_error", extra={"error": str(exc)})
            return ProbeResult(ok=False, error="provider_error")

    def _stop(self, key: str, generation: int, reason: str) -> None:
        lease = self._leases.get(key)
        if lease is None or lease.generation != generation:
            return
        lease.enabled = False
        lease.state = "stopped"
        lease.reason = reason
        lease.next_probe_at_ms = None
        lease.timer_task = None
        lease.probe_task = None
        lease.candidate = None
        lease.idle_deadline_monotonic = None
        lease.idle_expires_at_ms = None

    def _pause_for_idle_timeout(self, key: str, generation: int) -> None:
        lease = self._leases.get(key)
        if lease is None or lease.generation != generation or not lease.enabled:
            return
        lease.state = "paused"
        lease.reason = "idle_timeout"
        lease.next_probe_at_ms = None
        lease.timer_task = None
        lease.probe_task = None
        lease.candidate = None
        lease.idle_deadline_monotonic = None


__all__ = [
    "DEFAULT_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_TTL_SECONDS",
    "MAX_IDLE_TIMEOUT_SECONDS",
    "MAX_TTL_SECONDS",
    "MIN_IDLE_TIMEOUT_SECONDS",
    "MIN_TTL_SECONDS",
    "PROBE_RATIO",
    "PromptCacheKeepaliveService",
]