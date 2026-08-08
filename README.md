# prompt-cache-keepalive

> 你的 LLM prompt cache 有个 5 分钟的 TTL，给它一个心跳。

**Provider 无关的缓存保活库**，让 LLM 的 prompt cache 在空闲期间保持温暖，避免下一个 turn 冷启动重建前缀。

## 核心洞察

大多数 LLM provider 的 prompt cache 有约 **5 分钟（300s）的 TTL**。如果在空闲时每隔 **TTL×0.8** 秒发一个轻量级 probe，下一个真实 turn 就能**读缓存**而不是**重建缓存**。

## 出处

本库从 [OpenSquilla PR #1036](https://github.com/opensquilla/opensquilla/pull/1036) 提取核心逻辑并解耦成独立包。PR #1036 基于 [@cxk1993](https://github.com/cxk1993) 的原始设计 [PR #1015](https://github.com/opensquilla/opensquilla/pull/1015) 构建，保留 Co-authored-by 归属。

## 快速开始

```python
import asyncio
from prompt_cache_keepalive import (
    PromptCacheKeepaliveService,
    PromptCacheKeepaliveCandidate,
    ChatConfig,
    Message,
    ProbeResult,
)

# 1. 实现 probe function（把保活请求发到你的 LLM provider）
async def my_probe(messages, config):
    # 这里调用你的 LLM，返回缓存命中 token 数
    # 示例：OpenAI 兼容 API
    # from openai import AsyncOpenAI
    # client = AsyncOpenAI(api_key=..., base_url=...)
    # resp = await client.chat.completions.create(...)
    return ProbeResult(ok=True, cache_read_tokens=100)

# 2. 创建保活服务
svc = PromptCacheKeepaliveService(probe_fn=my_probe)

async def main():
    # 3. 为一个 session 开启保活
    await svc.set_enabled("session-1", enabled=True, ttl_seconds=300)

    # 4. 每次真实 turn 成功后，上报已验证的 provider 前缀
    svc.record_candidate(PromptCacheKeepaliveCandidate(
        session_key="session-1",
        provider_id="openai",
        model="gpt-4",
        messages=(Message(role="system", content="You are a helpful assistant."),),
        tools=(),
        config=ChatConfig(model="gpt-4"),
    ))

    # 5. 每次真实 turn 调用 renew() 重置空闲超时
    svc.renew("session-1")

    # 6. 查询状态
    print(svc.status("session-1"))

    # 7. 关闭
    await svc.close()

asyncio.run(main())
```

## 核心概念

### 生命周期

```
用户发消息 → 调用 renew() 重置空闲计时
            ↓ 空闲超时（默认 60 分钟）
            暂停保活，等待下次用户消息
            ↓ 用户回来发新消息
            调用 record_candidate() 重新装填前缀
```

### 状态机

| 状态 | 含义 |
|------|------|
| `off` | 未开启 |
| `waiting` | 已开启，等待稳定前缀 |
| `scheduled` | 有前缀，等待下次 probe |
| `probing` | 正在发送 probe 请求 |
| `paused` | 空闲超时暂停 |
| `stopped` | 因 cache miss/provider error 永久停止 |

### 自动停止条件

- **Cache miss**：probe 返回的缓存命中 token 数 ≤ 0
- **Provider error**：probe 请求异常
- **Idle timeout**：超过空闲窗口（默认 60 分钟）没有 renew
- **Preemption**：用户新 turn 抢占 probe

## API

### `PromptCacheKeepaliveService(probe_fn, global_slot=None)`

| 方法 | 说明 |
|------|------|
| `set_enabled(key, enabled, ttl_seconds, idle_timeout_seconds)` | 开启/关闭 session 保活 |
| `is_enabled(key)` | 检查是否已开启 |
| `renew(key)` | 重置空闲超时（每次真实 turn 后调用） |
| `record_candidate(candidate)` | 上报已验证的 provider 前缀 |
| `refresh_required(key, reason)` | 保持开启但丢弃过期前缀 |
| `status(key)` | 获取当前状态 |
| `invalidate(key)` | 完全关闭一个 session 的保活 |
| `close()` | 关闭所有保活任务 |

### `ProbeFn` 协议

```python
async def probe_fn(messages: list[Message], config: ChatConfig) -> ProbeResult:
    ...
```

## 设计原则

- **进程内存**：重启后所有 session 回到默认关闭状态
- **全局 slot**：所有 session 共享一个 probe 并发槽，不会抢占真实 turn
- **零依赖**：核心库纯 Python 标准库，不需要任何第三方包
- **Provider 无关**：你只需要实现一个 `ProbeFn` 异步函数

## 许可证

MIT