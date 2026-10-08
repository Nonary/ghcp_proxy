# Codex / Copilot SDK tool adaptation

`codex_sdk_adapter.py` is the bidirectional tool-format boundary for the SDK
Responses route. `copilot_sdk_upstream.py` retains responsibility for session
ownership, cache continuity, event collection, reasoning, usage, and SSE delivery.
Both JSON responses and SSE output items use the same adapter conversions.

## Request and response mapping

| Codex representation | Copilot SDK representation | Return to Codex |
| --- | --- | --- |
| Standard top-level `tools` | Declaration-only `Tool` objects | Original tool identities |
| Responses Lite `additional_tools` input items | Same catalog as standard requests; latest replayed declaration wins | Catalog items are not assistant messages |
| `tool_search_output.tools` | Loaded declarations join the catalog | SDK owns its own tool search; internal search is not exposed as an external client call |
| Namespaced function | Collision-safe flat runtime alias | `function_call` with original `name` and `namespace` |
| `functions` namespace | Default, unqualified namespace | No explicit namespace is needed |
| Custom/free-form tool | Private runtime alias and JSON schema containing a string `input` field | `custom_tool_call` with raw `input`, not its JSON wrapper |
| String tool output | `textResultForLlm` | Resumes the pending SDK request |
| Text/image output array | Ordered SDK `contents` blocks | Resumes the pending SDK request without losing inline image data |

Declarations retain JSON parameter schemas, explicit/inherited loading
preferences, namespace descriptions, and custom-tool format instructions. When
a request contains both a current top-level catalog and replayed Lite catalogs,
the top-level declaration takes precedence for the same callable identity.
Updating a Lite declaration keeps its position in the catalog so aliases do not
change merely because a schema was updated.

One deliberate schema exception is the `message` parameter on the
`collaboration` namespace's `spawn_agent`, `send_message`, and `followup_task`
functions. Codex v2 declares it with `encrypted: true`. The SDK honors that
annotation, but the SDK prompt renderer cannot consume opaque agent-message
content. The adapter removes that annotation from the SDK-only schema and
returns `encrypted_function_args: []` on those function calls, including both
the added and completed SSE items. An explicitly empty list selects Codex's
`DirectPlaintextMessage` compatibility path; omitting the field does not.
Other namespaces' encryption annotations are left untouched.

This follows the current upstream contracts in
[`ToolCall::direct_source`](https://github.com/openai/codex/blob/529cd6b860/codex-rs/core/src/tools/router.rs)
and
[`agent_message_from_tool`](https://github.com/openai/codex/blob/529cd6b860/codex-rs/core/src/tools/handlers/multi_agents_v2.rs).

Tools remain declaration-only: Codex executes them, not the proxy or SDK host.

## Continuation and recovery

The existing `ghcpsdk_` call ID encoding is retained. New namespaced calls also
encode their namespace, alongside the SDK session, pending request, original
tool name, and function/custom type. Previously issued IDs still decode.

Trailing tool results can be accompanied by new caller instructions or catalog
items. A supplied result type, name, or namespace that conflicts with the pending
call is rejected with HTTP 400 before opening an SDK client. Namespace and name
remain optional on the output item because identity is carried by the call ID.
Plaintext `agent_message` items are caller input too, so queued messages and
follow-up assignments can accompany pending tool results without hiding the
continuation or losing their steering text.

The live-session path sends structured results through
`tools.handle_pending_tool_call`. If pending work or the entire session has been
lost, the existing transcript/prompt fallback preserves qualified tool names and
supplies inline result images as native SDK blob attachments.

The SDK route explicitly preserves tool-output images during input sanitization.
Other routes retain their existing image-sanitization behavior.

## Limits

- The SDK accepts JSON-schema tools, not native Responses custom-tool grammars.
  Grammar/text format declarations are retained as model-facing instructions;
  this is not constrained grammar decoding or local grammar validation.
- Tool-result images must use inline base64 data URLs. Remote URLs and file-ID
  images fail explicitly rather than being silently replaced with placeholders.
  The adapter does not fetch arbitrary URLs.
- Hosted tool types without a declaration-only SDK equivalent remain outside
  this adapter's support. It does not turn SDK-internal search into a Codex tool
  call or attempt to reproduce hosted-tool execution.
- This adapter does not change the separate SDK reasoning-persistence or
  compaction implementation.
- Previously emitted encrypted collaboration assignments cannot be decrypted
  by this adapter. A new trailing opaque `agent_message` fails with HTTP 400
  instead of sending only its readable `Payload:` header to a model. Resend
  the assignment after loading the plaintext fix. Historical opaque items do
  not prevent a new plaintext follow-up from resuming an existing session.

## Focused offline verification

```sh
./.venv/bin/python -m unittest test_codex_sdk_adapter -v
```

These checks exercise catalog updates, alias collisions, custom input formats,
namespace replay, structured results, invalid continuations, recovery, and both
JSON and SSE request/result round trips. They use synthetic session events and
the installed SDK's real RPC serialization types; they do not call a live model.

An opt-in probe checks the upstream half with one real Copilot model turn:

```sh
./.venv/bin/python tools/verify-sdk-collaboration-plaintext.py --live --model gpt-6-luna
```

It sends only a synthetic canary assignment, captures the external tool request,
and checks the exact plaintext plus the returned compatibility marker. It does
not execute the follow-up or spawn Codex agents. SDK state is isolated in a
temporary directory, and its probe session is disconnected/deleted before exit.

Excel-suffixed models do not use this Copilot SDK adapter. They have a separate
Responses wire adapter with the corresponding v2 catalog, collaboration, and
agent-message mappings documented in [the Excel adapter notes](codex-excel-adapter.md).
