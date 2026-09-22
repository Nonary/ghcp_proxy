**Prompt-cache investigation — 2026-09-22**

Follow-up to [2026-09-21](prompt-cache-investigation-2026-09-21.md), using the
same trace (`~/Downloads/request-trace.jsonl.zip`, SHA-256 `1a0750dc…`), a native
Codex usage log as a no-proxy control, and live probes of the Copilot runtime's
model wire (logged-in Copilot user, small prompts).

**The pattern**

Cache is lost on the first request of a new user turn when the previous turn
made more than one model call. Tool round-trips inside a turn keep their cache,
and so does a turn boundary after a one-call answer.

| Trace, SDK route | Turn boundary | Inside a turn |
|---|---:|---:|
| Sol | 4/4 lost | 0/41 |
| Luna | 2/4 lost | 6/356 (all compactions) |

At every lost boundary the upstream prompt grew less than what was appended, or
shrank (`input_shortfall` = input − previous input − previous output):
`0ca434` −3,335, `5197bb` −630, `342b9d` −152, Luna `f9ea3d` −12,389. The two
Luna boundaries that kept their cache followed one-call turns, whose reasoning
sat at the tail of the prompt.

Native Codex → OpenAI traffic (usage log with turn IDs, no proxy) lost cache at
0/20 Sol turn boundaries, but showed random single-request misses on Sol (12/501)
and Astra, and none on Luna (0/81).

**What the wire shows**

- Copilot's runtime calls Sol — and tool-using sessions generally — over a
  WebSocket (`wss://…githubcopilot.com/responses`). Each request chains
  `previous_response_id` and carries only new items; reasoning is only visible
  in response output. It sends no `prompt_cache_key`.
- A disk resume reconnects and resends the full history **without** encrypted
  reasoning. A WebSocket drop inside a live session resends it **with** reasoning.
- Copilot caches along the chain. A full-history request reuses only earlier
  *full* requests — in a chained conversation, the first user message
  (5,810 of 5,988 tokens after a resume). This held with the reasoning restored,
  with a byte-equivalent live resend, and on the same upstream connection.
- A new connection can land on a cold backend: a live reconnect with intact
  reasoning got `cached_tokens: 0`, after which hits came in 512-token blocks —
  the Sol shape in the trace.
- Sending the resumed full history as a continuation of the kept connection's
  chain (`previous_response_id` + only the new items) restored the hit:
  5,926 of 5,976 tokens instead of 5,810.
- After `session.abort()` the runtime emits `session.idle` and no late events;
  the aborted session serves the next turn normally.

This explains the trace: every destroy/resume replaced the chain with a full
resend (hitting ≈ the conversation's first request: 5,120, 6,656, 1,548), and
Sol's resends often landed cold (`0ca434` = 0), so the next turn missed again
(`71f914`) despite an intact prefix (shortfall −462 ≈ its own 516 reasoning
tokens at the tail).

On the Excel route the proxy forwarded every encrypted reasoning item unchanged,
yet Terra's turn boundary `1963cc` had a −250 shortfall and fell back to 39,796
(the previous turn's start): Basispoints drops prior-turn reasoning for Terra.
Luna on the same route kept it (shortfall +18, full hit).

**Changes**

- The session stays live after an interrupted turn once the abort settles, like
  completed turns since 2026-09-21.
- `_UpstreamRequestHandler` now also handles WebSockets. When the runtime drops
  an idle upstream WebSocket it is kept for the session's next connection
  (`GHCP_SDK_WEBSOCKET_POOL_SECONDS`, default 600), and a full resend that
  matches that connection's chain item for item is sent as a continuation. A
  rejected continuation falls back to the full request before the runtime sees
  the error. This covers resumes caused by a new message during a pending tool
  call, configuration changes, failed turns and eviction.
- `sdk_reasoning_ledger` records each session's encrypted reasoning (response
  output over WebSocket, request input over HTTP), persists it per session under
  `copilot-sdk/reasoning/`, and restores it into full-history requests that lost
  it, including after a proxy restart. For HTTP models this restores the cached
  prefix; over WebSocket it keeps the model's prior reasoning in context.
- Each session gets its own `prompt_cache_key` (`GHCP_SDK_PROMPT_CACHE_KEY=0`
  disables it). Copilot accepted it on both transports.
- HTTP rejections of any change are retried unmodified and the change is not
  applied again for that model (or session, for restored reasoning); WebSocket
  invalid-request errors do the same.
- Traces record `copilot_sdk_session.model_calls`; `tools/analyze-prompt-cache-trace.py`
  adds `turn_boundary`, `input_shortfall`, the reuse-miss reason and model-call
  summaries.

**Remaining limits**

A proxy restart (including auto-update) closes every upstream WebSocket, so the
next turn of each open conversation is a full resend on a new connection.
Upstream routing to a cold backend and the Excel route's server-side reasoning
handling are outside the proxy. Summary delivery
(`reasoning_summary_delivery`) is still only added to HTTP model calls, so it
does not reach Sol's WebSocket calls.

**Verify**

Capture a trace with the change deployed, then:

```powershell
.\.venv\Scripts\python.exe tools\analyze-prompt-cache-trace.py "$HOME\Downloads\request-trace.jsonl.zip"
```

Look at turn-boundary rows: `sdk_operation`, `sdk_reuse_miss`, `sdk_resumed_chain`
and `sdk_call_cached` show whether each boundary stayed live, resumed its
chain, or fell back to a full resend.
