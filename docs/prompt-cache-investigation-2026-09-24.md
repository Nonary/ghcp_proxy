**Prompt-cache investigation — 2026-09-24**

Follow-up to [2026-09-22](prompt-cache-investigation-2026-09-22.md), using the
Mac export `ghcp_proxy_requests_2026-09-24_172139` (usage log 14:38–22:21 UTC,
1,489 requests; request trace 18:24–22:21 UTC, 502 requests; 1,000 body dumps)
and live probes of the Copilot runtime (SDK 1.0.14 / runtime 1.0.85, Luna and
Sol, prompts of at most about 17k tokens).

**Turn boundaries are fixed**

Every turn boundary with body dumps kept its cache unless something changed
the prefix: 17 of 23, including one after 16 idle minutes. The six misses:

| Boundary | Cause |
|---|---|
| 18:03:45 `c42116bb`, 21:40:27 `ee750029` | Reasoning effort changed (high→xhigh, high→max) |
| 18:10:23 `cdd7ab08`, 20:17:04 `84643479` | 31 min and 2 h idle: Copilot's cache had expired (below) |
| 22:03:43 `9ee2e75f` | Compaction: proxy bugs 1 and 2 below |
| 22:05:04 `079696fa` | First request after that compaction (new prefix) |

**Proxy bug 1 — every SDK compaction missed the cache for the whole context**

`_handle_copilot_sdk_responses` set `tool_choice: "none"` on compaction turns.
`build_tool_registration` treats that as "no tools", so the session options
changed: the live session was evicted, resumed from disk with no tools, and
the runtime sent the whole history without tool definitions (over HTTP, which
the runtime uses for tool-less sessions). Tools render before the conversation,
so nothing after the short system message could match.

`9ee2e75f` (Sol, manual compaction): 120,554 input tokens, 0 cached.

Reproduced end to end (real uvicorn/Starlette app serving `handle_responses`,
real runtime, Codex-like layout: short instructions, ~4.5k-token conversation):

| Compaction turn | Session | Model call | Cached |
|---|---|---|---:|
| `tool_choice: "none"` in the body (before) | `resume_disk`, tools removed | HTTP | 0 / 5,762 |
| tools kept (after) | `reuse_live` | WebSocket, `tool_choice: "none"` | 4,554 / 5,834 |

A continuation probe gave the same picture: dropping `tools` → 0 / 4,601;
`tool_choice: "none"` with the tools kept → 4,506 / 4,636 (control 4,578).

Fix: compaction keeps the caller's tools; the request handler sets
`tool_choice: "none"` on the compaction turn's model calls (rejections are
handled like the other request changes).

**Proxy bug 2 — every interrupt evicted the live session**

Starlette (uvicorn reports ASGI 2.3) cancels a streaming response through an
anyio cancel scope, which keeps cancelling every await inside the generator.
`_stream_turn`'s `await asyncio.shield(_abort_and_settle(...))` therefore
returned `CancelledError` at once, `abort_settled` stayed false and the finally
block evicted the session — the 2026-09-22 "park after a settled abort" never
took effect outside unit tests, which cancel once. The `is_disconnected()`
path aborted and evicted too.

End-to-end check (same harness, interrupt after the first text delta, then a
new message): before, the live session was gone and the next turn was
`resume_disk` / `not_connected`; after, `reuse_live`. The runtime itself goes
idle within milliseconds of `session.abort()` (measured before and during
generation).

Fix: `_settle_interrupted_turn` runs the abort, the wait for `session.idle`
and the release in a task outside the cancel scope; `_reuse_live_session`
waits for it (bounded). `GeneratorExit` is handled the same way.

The runtime closes its WebSocket on abort with the response in flight, so the
upstream chain is not kept. The live session resends its exact in-memory
history on a new connection, and a byte-identical full resend hits the cache
(4,578 / 4,601 in a probe). A `previous_response_id` from another connection is
rejected ("previous_response_id does not belong to this connection").

**Losses the proxy does not cause — controlled tests**

The same one-turn Sol tool loop (xhigh, nine model calls, a unique prefix per
run) through three paths, interleaved:

| Path | Runs Copilot served from its 512-token block cache |
|---|---|
| The runtime's own transport — no request handler, no proxy code | 2 of 3 |
| The proxy's request handler, as in production | 2 of 3 |
| The proxy's handler without the injected `prompt_cache_key` | 3 of 3 |

In a block-cache run every continuation reads the same 512-aligned prefix
(11,776 for seven calls, then 12,288) instead of the previous call's input;
the runs look the same on every path, and exact runs (every call reads the
previous input minus 3) also occur on every path. Block-cache runs happened
with the proxy's `reasoning_summary_delivery=sequential_cutoff` (the proxy
rows send it) and without it (one of three separate runs without it).

- *22:00:19–22:01:16, Sol.* Six consecutive runtime continuations on an
  intact live WebSocket chain (1–3 new items each, input growing normally, no
  proxy rewrite, nothing else running through the proxy) read 100,256,
  100,256, 0, 0, 0, 100,256, although each reported writing its prompt; the
  chain then hit fully again by itself. 100,256 is the input of `041a212e`, a
  43-second response. About 360k uncached tokens. This is the serving
  variability above at 100k tokens; it could not be reproduced on demand.
  Model calls now record `X-Copilot-Service-Request-Id` and Copilot's
  WebSocket session id, so such calls can be reported.
- *Idle gaps.* Copilot's cache outlives 20 idle minutes but not 31: a fresh
  full resend hit after 12 minutes (6,528 / 6,550) and after 20 minutes
  (7,046 / 7,068) and read nothing after 31 minutes (0 / 7,081); the runtime's
  usage events report `_cache_ttl_seconds: 1800`. Requesting
  `prompt_cache_retention: "24h"` (which Copilot echoes on every response)
  explicitly changed nothing: accepted, and 0 / 6,550 after 31 minutes
  against 0 / 7,082 without it. Copilot also closes idle
  model WebSockets itself (`1000 idle timeout`) within 12 minutes, so no
  socket survives a long pause. The 20:17:04 request (2 h idle) resent the
  full history with 165 reasoning items restored and the pre-gap token
  count, and still read nothing.
- *22:08, Sol.* A WebSocket call failed and the runtime retried over HTTP;
  two more requests failed. The trace did not record why; model calls now
  record upstream errors. HTTP and WebSocket share Copilot's cache (Luna: an
  HTTP full request after WebSocket calls read 4,598 / 4,601).

**Other observations**

- `capi.enable_web_socket_responses=False` (2026-09-23) only applies to
  sessions without tools; with a tool registered the runtime still opens a
  WebSocket (probed with and without the option). Codex sessions always have
  tools, so all of them still use WebSocket.
- Each WebSocket `response.create` from the runtime carries `initiator`,
  `agent_task_id` and `headers` (`X-Interaction-Id`, `X-Interaction-Type`,
  `X-Agent-Task-Id`, `X-Client-Session-Id`, `Copilot-Harness-Id`). Response ids
  are opaque 492-character values.
- The 117 gpt-6-luna swarm sessions each diverge inside one large user message,
  so only the ~15k-token system and tool prefix is shared between them.
- The runtime's own background compaction goes out over HTTP with the full
  history; the reasoning ledger restored 78 reasoning items into one
  (`811485e4`). Its usage was not recorded before this change.

**Diagnostics added**

`copilot_sdk_session` now records `error` (the session error behind a failed
turn), `interrupted`, and `background_model_calls` (calls made between
requests). Each model call records `interaction`, `connection`,
`pooled_socket`, `tool_choice`, upstream `error` events, Copilot's
`service_request_id` and `copilot_websocket_session`, and for HTTP calls
`status`, error bodies and usage. `tools/analyze-prompt-cache-trace.py`
compares items without `item_hash` (which also covers caller metadata the SDK
does not forward), so turn boundaries are detected again; it adds
`sdk_call_errors`, `sdk_background_calls`, `sdk_interrupted`, `sdk_error`, and
a `miss_cause` for every request that read less than the conversation's
previous input (`--summary` totals the lost tokens per cause). For this
trace:

| miss_cause | Requests | Lost tokens |
|---|---:|---:|
| cold_cache (22:00) | 3 | 331,664 |
| compaction (bug 1, and the new prefix after it) | 2 | 238,884 |
| settings_changed (effort high→max) | 1 | 149,722 |
| retried_after_failed_call (22:08) | 1 | 32,367 |
| stale_cache (22:00) | 4 | 28,153 |

**Verify**

Deploy, capture a new export, then:

```powershell
.\.venv\Scripts\python.exe tools\analyze-prompt-cache-trace.py <export>\raw\request-trace-<date>.jsonl --summary
```

Compaction rows should show `sdk_operation=reuse_live` and a hit; the request
after an interrupt should be `reuse_live`, not `resume_disk`/`not_connected`;
`session_resumed` should only follow configuration changes or gaps over 30
minutes.
