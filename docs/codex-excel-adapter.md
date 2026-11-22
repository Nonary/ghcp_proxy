# Codex / Excel Responses adaptation

Excel-suffixed model IDs use the dedicated Basispoints Excel Responses route,
not the Copilot SDK. `proxy.py` selects `_handle_excel_responses`, and
`excel_upstream.py` translates Codex Responses requests into the Excel wire
shape. The Excel adapter owns that wire conversion; it does not reuse the SDK
session lifecycle or runtime tool registration.

The Basispoints request schema does not accept `parallel_tool_calls`, even when
false. Client tool-call preferences must not be forwarded as that field; doing
so causes HTTP 422 (`Invalid request body`). The relay prompt specifies one
client-tool envelope per transport call instead.

## Codex v2 compatibility

- Top-level `tools`, Responses Lite `additional_tools`, and loaded
  `tool_search_output.tools` declarations join the client-tool catalog.
  Current top-level declarations win over replayed catalogs, and catalog
  items are omitted from conversation history.
- Namespaced tool identities are restored as a leaf `name` plus a separate
  `namespace` on returned function and custom-tool calls.
- Flattened MCP names such as `mcp__codex_app__list_threads` resolve to a unique
  registered catalog identity (`mcp__codex_app.list_threads`) for schema lookups
  and tool calls. Exact names take precedence; ambiguous aliases are rejected.
  Deferred tools still require completed schema disclosure, and action
  arguments are validated against the original tool schema.
- The lean connector relay unwraps up to two redundant reserved transport
  envelopes from model retries, retaining the native call identity for replay.
  Real registered connector actions are not treated as reserved wrappers.
- Deferred tool-definition lookups return at most four definitions at a time.
  Larger lookups page through still-undisclosed definitions when repeated,
  rather than leaking the native relay to Codex as an unsupported tool call.
- For `collaboration.spawn_agent`, `collaboration.send_message`, and
  `collaboration.followup_task`, the Excel prompt schema omits the Codex-only
  `encrypted: true` annotation on the `message` parameter. Synthesized Codex
  function-call items carry `encrypted_function_args: []` in JSON and SSE,
  selecting Codex's plaintext collaboration path.
- Plaintext `agent_message` content is translated to an Excel input message.
  A new trailing opaque encrypted assignment is rejected with HTTP 400 rather
  than forwarding an incomplete readable header. Historical encrypted
  assignments are skipped because their payload cannot be recovered.
- Basispoints task identities use the current Codex thread, not the shared root
  session's prompt-cache key. Parents, children, and nested children keep
  separate task state while retaining the shared prompt cache.
- Codex lifecycle turn IDs keep tool continuations, queued agent messages, and
  compaction within the same Excel turn. Iterations count only that turn's tool
  outputs. A follow-up assignment starts a new turn without replacing the
  thread's task; clients without lifecycle metadata use user or agent messages
  as turn boundaries.
  When Codex omits per-item turn tags, its lifecycle start time and recognized
  UUIDv7 replay item timestamps delimit the current turn instead, so queued
  assignments and child replies do not reset its iteration count.

The proxy still treats client tools as declarations: it converts the Excel
model's `run_officejs` transport call into a Codex client-tool call, but it does
not execute the client tool or modify the workbook.

## Context compaction

The Codex model catalog and the default Basispoints `context_management`
threshold use the same per-model compaction limit. Excel models with a
272,000-token context use the shared Copilot default of 240,000 tokens, rather
than an earlier Excel-only limit. Luna's existing 200,000-token context retains
its 180,000-token limit. Newly discovered aliases inherit the default Excel
model's limits unless discovery supplies more specific capabilities.

Explicit client `context_management` lists, including an empty list, are
preserved. These defaults apply to both the lean external-client profile and
the legacy Office profile. Copilot's route is unchanged.

Successful Excel JSON and SSE responses declare `x-reasoning-included: true`.
Basispoints already counts replayed encrypted reasoning in `usage.input_tokens`,
but does not supply this header. Without it, Codex adds a separate estimate for
old reasoning on top of the reported usage. For example, 129k reported tokens
plus 111k estimated reasoning can trigger the 240k limit while the context
meter still shows roughly 50%. The marker prevents that double count without
changing token totals, billing, encrypted reasoning replay, or compaction limits.
The proxy adds this marker only to 2xx HTTP responses.

Restart the proxy to load the change. Existing chats receive the corrected
usage signal on their next successful model response; a new chat or Codex app
restart is not required for this fix.

## Focused verification

```sh
./.venv/bin/python -m unittest test_excel_upstream -v
```

The tests cover catalog loading and precedence, namespace restoration,
collaboration plaintext markers across JSON/SSE tool calls, plaintext agent
message translation, explicit rejection of opaque new assignments, isolated
parent/child task identities, and follow-up/queued-message turn continuity.
