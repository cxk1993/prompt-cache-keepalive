"""Tests for prompt_cache_keepalive service."""

from __future__ import annotations

import asyncio

import pytest

from prompt_cache_keepalive import (
    ChatConfig,
    Message,
    ProbeResult,
    PromptCacheKeepaliveCandidate,
    PromptCacheKeepaliveService,
)


def make_candidate(session_key: str = "sess-1") -> PromptCacheKeepaliveCandidate:
    return PromptCacheKeepaliveCandidate(
        session_key=session_key,
        provider_id="openai",
        model="gpt-test",
        messages=(Message(role="system", content="You are a helpful assistant."),),
        tools=(),
        config=ChatConfig(model="gpt-test"),
    )


@pytest.fixture
def events():
    return []


async def _ok_fn(messages, config, cached_tokens=1):
    return ProbeResult(ok=True, cache_read_tokens=cached_tokens)


@pytest.mark.asyncio
async def test_disabled_by_default(events):
    svc = PromptCacheKeepaliveService(probe_fn=_ok_fn)
    try:
        st = svc.status("sess-1")
        assert st["enabled"] is False
        assert st["state"] == "off"
        assert svc.is_enabled("sess-1") is False
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_enable_then_off(events):
    svc = PromptCacheKeepaliveService(probe_fn=_ok_fn)
    try:
        await svc.set_enabled("sess-1", enabled=True)
        assert svc.is_enabled("sess-1") is True
        await svc.set_enabled("sess-1", enabled=False)
        assert svc.is_enabled("sess-1") is False
        assert svc.status("sess-1")["state"] == "off"
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_record_candidate_schedules_probe(events):
    """Recording a candidate should arm the lease and eventually probe."""
    fired = asyncio.Event()

    async def probe(messages, config):
        fired.set()
        return ProbeResult(ok=True, cache_read_tokens=100)

    svc = PromptCacheKeepaliveService(probe_fn=probe)
    try:
        await svc.set_enabled("sess-1", enabled=True, ttl_seconds=5)
        svc.record_candidate(make_candidate())
        st = svc.status("sess-1")
        assert st["enabled"] is True
        assert st["state"] == "scheduled"
        assert st["hasSnapshot"] is True
        # Probe interval = 5 * 0.8 = 4s; wait for it
        await asyncio.wait_for(fired.wait(), timeout=6)
        st = svc.status("sess-1")
        assert st["lastCacheHitTokens"] == 100
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_cache_miss_stops_lease(events):
    async def probe(m, c):
        return ProbeResult(ok=True, cache_read_tokens=0)

    svc = PromptCacheKeepaliveService(probe_fn=probe)
    try:
        await svc.set_enabled("sess-1", enabled=True, ttl_seconds=1)
        svc.record_candidate(make_candidate())
        # probe interval = 1 * 0.8 = 0.8s; wait for it to run and stop
        await asyncio.sleep(1.2)
        st = svc.status("sess-1")
        assert st["state"] == "stopped"
        assert st["reason"] == "cache_miss_or_usage_unreported"
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_provider_error_stops_lease(events):
    async def probe(messages, config):
        raise RuntimeError("boom")

    svc = PromptCacheKeepaliveService(probe_fn=probe)
    try:
        await svc.set_enabled("sess-1", enabled=True, ttl_seconds=1)
        svc.record_candidate(make_candidate())
        await asyncio.sleep(1.2)
        st = svc.status("sess-1")
        assert st["state"] == "stopped"
        assert st["reason"] == "provider_error"
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_refresh_required_discards_snapshot(events):
    svc = PromptCacheKeepaliveService(probe_fn=_ok_fn)
    try:
        await svc.set_enabled("sess-1", enabled=True)
        svc.record_candidate(make_candidate())
        svc.refresh_required("sess-1", reason="session_history_changed")
        st = svc.status("sess-1")
        assert st["enabled"] is True  # opt-in persists
        assert st["hasSnapshot"] is False
        assert st["state"] == "waiting"
        assert st["reason"] == "session_history_changed"
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_invalidate_disables(events):
    svc = PromptCacheKeepaliveService(probe_fn=_ok_fn)
    try:
        await svc.set_enabled("sess-1", enabled=True)
        await svc.invalidate("sess-1")
        assert svc.is_enabled("sess-1") is False
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_slot_serializes_probes(events):
    """Probes from different sessions should not run concurrently."""
    active = 0
    peak = 0

    async def probe(messages, config):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return ProbeResult(ok=True, cache_read_tokens=100)

    svc = PromptCacheKeepaliveService(probe_fn=probe)
    try:
        await svc.set_enabled("sess-1", enabled=True, ttl_seconds=1)
        await svc.set_enabled("sess-2", enabled=True, ttl_seconds=1)
        svc.record_candidate(make_candidate("sess-1"))
        svc.record_candidate(make_candidate("sess-2"))
        await asyncio.sleep(1.2)
        assert peak == 1
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_renew_extends_idle_window(events):
    """renew() should keep the lease alive by resetting the idle deadline."""
    svc = PromptCacheKeepaliveService(probe_fn=_ok_fn)
    try:
        await svc.set_enabled(
            "sess-1", enabled=True, ttl_seconds=5, idle_timeout_seconds=10
        )
        svc.record_candidate(make_candidate())
        # Renew a few times to keep it active
        for _ in range(5):
            svc.renew("sess-1")
            await asyncio.sleep(0.1)
        st = svc.status("sess-1")
        assert st["enabled"] is True
        assert st["idleExpiresAt"] is not None
    finally:
        await svc.close()


def _ok(cached_tokens: int) -> ProbeResult:
    return ProbeResult(ok=True, cache_read_tokens=cached_tokens)