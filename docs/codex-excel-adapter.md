# Codex / Excel Responses adaptation

Excel-suffixed model IDs use the dedicated Basispoints Excel Responses route,
not the Copilot SDK. `proxy.py` selects `_handle_excel_responses`, and
`excel_upstream.py` translates Codex Responses requests into the Excel wire
shape. The Excel adapter owns that wire conversion; it does not reuse the SDK
session lifecycle or runtime tool registration.

## Codex v2 compatibility

- Top-level `tools`, Responses Lite `additional_tools`, and loaded
  `tool_search_output.tools` declarations join the client-tool catalog.
  Current top-level declarations win over replayed catalogs, and catalog
  items are omitted from conversation history.
- Namespaced tool identities are restored as a leaf `name` plus a separate
  `namespace` on returned function and custom-tool calls.
- For `collaboration.spawn_agent`, `collaboration.send_message`, and
  `collaboration.followup_task`, the Excel prompt schema omits the Codex-only
  `encrypted: true` annotation on the `message` parameter. Synthesized Codex
  function-call items carry `encrypted_function_args: []` in JSON and SSE,
  selecting Codex's plaintext collaboration path.
- Plaintext `agent_message` content is translated to an Excel input message.
  A new trailing opaque encrypted assignment is rejected with HTTP 400 rather
  than forwarding an incomplete readable header. Historical encrypted
  assignments are skipped because their payload cannot be recovered.

The proxy still treats client tools as declarations: it converts the Excel
model's `run_officejs` transport call into a Codex client-tool call, but it does
not execute the client tool or modify the workbook.

## Focused verification

```sh
./.venv/bin/python -m unittest test_excel_upstream -v
```

The tests cover catalog loading and precedence, namespace restoration,
collaboration plaintext markers across JSON/SSE tool calls, plaintext agent
message translation, and explicit rejection of opaque new assignments.
