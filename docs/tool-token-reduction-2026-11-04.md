# Excel relay tool-token reduction — November 4, 2026

## Measured change

On a captured desktop request with 59 registered client tools, local
`o200k_base` estimates for the bridge instructions and catalog fall from
**13,356 to 2,149 tokens**: **11,207 fewer tokens, or 83.9%**.
Four tools remain eager in that capture and 55 are disclosed on demand.
These are local prompt-text estimates, not post-deployment billed usage.
No authenticated model calls were made for that initial verification. The
follow-up below includes controlled authenticated gateway measurements.

The previous catalog had approximately 5.7k description tokens and 6k schema
tokens, plus names and relay instructions. It was not a 4k catalog.

## What changed

- Large Excel relay catalogs initially include full definitions only for
  `exec_command`, `write_stdin`, `apply_patch`, `update_plan` when registered,
  and the metadata output carrier `mcp__node_repl.js`. Explicit caller
  `defer_loading=false` preferences also keep a tool eager.
- Every other registered tool remains in a compact name index. The model can
  request 1–4 exact tool definitions with the proxy-local
  `__ghcp_proxy_tool_schema` relay envelope.
- The proxy maps that request to the client's existing Node tool, using only
  `nodeRepl.write(<JSON-quoted literal string>)`. This prints metadata; it
  performs no requested action, filesystem access, network request, or
  evaluation of policy text. It does not introduce a nonexistent client tool
  or require an extra server-side model loop.
- Full descriptions, permission instructions, parameter constraints, and
  custom grammars appear in the metadata output before execution. An attempted
  undisclosed action returns metadata instead of executing. The model must
  read it and call the action separately.
- Completed disclosures are recognized from matching call/output history and
  a definition digest. Changed definitions require disclosure again.
  Original schemas still validate actual emitted client calls.
- The relay preserves the original native call identity during replay, so
  generated metadata-printing arguments are not duplicated into the gateway's
  conversation history. The stable catalog prefix does not grow or change
  when a tool is disclosed.
- Ordinary schema notation now also compacts nested `allOf`, `anyOf`, `oneOf`,
  `$defs`, and `definitions` without expanding references or dropping their
  meanings. Unknown/conditional schema dialects still fall back to JSON.

This is client-side metadata disclosure, **not** the gateway gaining native
OpenAI tool search. The first use of an optional tool requires an additional
client tool round trip; already disclosed tools do not require that step
again while their completed disclosure remains in history. If many optional
tools are used, their definitions naturally accumulate in conversation context.

## Compatibility and fallback

Lazy disclosure is enabled by default only for catalogs larger than 20 tools
with the declared Node metadata-output contract. Missing/incompatible carriers,
small catalogs, forced tool choices, and a real caller-tool collision with the
reserved lookup name retain the eager catalog. A failed, denied, or truncated
metadata result also restores the full catalog instead of causing a retry loop.
That failure fallback can cause a one-time cache-prefix change.

Set `GHCP_EXCEL_LAZY_TOOL_CATALOG=0` to retain the eager behavior. Restart the
proxy to apply code or environment changes. No model route, caller instructions,
permissions, gateway tool-version IDs, or live runtime settings were changed
in the initial lazy-disclosure change.
Existing unrelated worktree changes were preserved; the running proxy was not
restarted during this change.

## Rechecking the apparent 22k gateway overhead

Four separate cold, message-only captured requests have these measurements:

| Local visible-text estimate | Reported input tokens | Difference |
| ---: | ---: | ---: |
| 2,578 | 24,930 | 22,352 |
| 9,305 | 31,655 | 22,350 |
| 18,632 | 40,985 | 22,353 |
| 18,633 | 40,986 | 22,353 |

These are `gpt-6-luna-excel` captures with zero reported cached input, not
inferences from a single large coding conversation. Local usage normalization
passes the reported input counter through; it does not add a 22k constant.
The nearly fixed residual is consistent with gateway-injected context and
protocol framing, rather than the client catalog being counted twice. It
does **not** precisely expose/tokenize the hidden server prompt, establish
that all of it is necessary, or prove that no server-side optimization exists.
No supported way to disable that injection was verified in the initial change.
The follow-up below identifies a smaller gateway profile instead.

The budget tool also now counts plaintext tool arguments and outputs, not just
message content. Omitting tool history would incorrectly attribute it to
server overhead on long conversations. Encrypted reasoning remains explicitly
outside the local plaintext estimate.

## Offline verification

```sh
./.venv/bin/python -m unittest -q test_excel_tool_disclosure \
  test_tool_token_budget test_excel_upstream.ExcelUpstreamTests
./.venv/bin/python tools/verify-tool-token-budget.py \
  --archive '/absolute/path/to/request-body.json' --excel-only
./.venv/bin/python -m py_compile excel_upstream.py tool_catalog.py \
  tools/verify-tool-token-budget.py test_excel_tool_disclosure.py \
  test_tool_token_budget.py
git diff --check
```

Coverage includes schema policies and grammars, original validation, native
replay, stream conversion/usage preservation, malformed input, injection-safe
literal rendering, forced choices, eager opt-out, failure fallback, and cache
prefix stability. A local Node execution independently verified that a
JavaScript-looking policy string is printed literally and never evaluated.

## Follow-up: stop forcing the Office profile

The proxy was adding hardcoded Office identity headers even for an external
coding client. The gateway used them to select its larger Excel system prompt
and native tool bundle. This is a separate source of overhead from the client
tool catalog. The gateway's `response.created` event exposes the resolved
instructions and tools, which confirms that the extra context is real.

Controlled requests using the same account, endpoint, model, and tiny user
message measured:

| Gateway profile | Instructions, local estimate | Native tools | Reported input |
| --- | ---: | ---: | ---: |
| Forced Excel identity | 13,210 | 26 | 22,382 |
| Default, without Office identity | 9,566 | 21 | 17,627 |

The instruction counts are local `o200k_base` estimates, not usage totals.
Serialized tool-schema estimates were 8,617 and 6,330 respectively; these JSON
estimates do not represent the gateway's exact native-tool tokenization and
must not be added to reported usage. The authoritative profile-only difference
is **4,755 input tokens**. Setting just the agent-profile header to `default`
while retaining the other Office headers did not select the smaller profile.

### Implemented behavior

- External Responses requests omit the Office/browser identity, copied user
  agent, and Stainless impersonation headers, but retain bearer authentication,
  account IDs, authentication mode, and ordinary request headers. Session
  discovery/capture requests keep their existing behavior.
- External bodies no longer pin the captured Excel tool-bundle version; the
  gateway chooses its matching default bundle. Caller model selection, source
  tools, original policies, schema validation, and task/turn identity remain
  unchanged.
- The default bundle has no `run_officejs`. It does declare
  `run_connector_action`, now used solely as an intercepted native carrier
  with the exact reserved action reference `__ghcp_proxy_client_tool__`.
  Its `params` field holds the client envelope as an object, not JSON text.
  No remote connector is executed.
- A deterministic provider-computed local action catalog registers that
  virtual relay in the stable prologue. Its source is explicitly local and
  `remote_connectors_fetched` is false. This is not a remote connector lookup
  or an additional model call. Without the catalog, the model tried native
  connector discovery instead of the relay.
- Function/custom tools, lazy schema disclosure, streaming, native call IDs,
  result replay, and client-tool fallback after proxy restart remain supported.
  Known plaintext Office relay history migrates to the new carrier without
  changing call IDs. Old Office envelopes still decode.

Set **`GHCP_EXCEL_LEAN_RELAY=0`** to restore both the Office identity and the
legacy `run_officejs` relay. Restart the proxy after code/environment changes.
No live daemon restart or runtime-setting changes were made during verification.

### Full matched-request verification

Replayed the archived greeting request
`e80a3b437a8344edafc202c69911cd20`, preserving its caller input and complete
registered tools. Preparing it in legacy mode reproduced the archived input
exactly. The real gateway reported:

| Same caller input and tools | Input tokens | Cached input |
| --- | ---: | ---: |
| Archived Office relay | 29,496 | 0 |
| New complete lean adapter | 24,551 | 0 |

That is **4,945 fewer actual input tokens (16.8%)**, including the revised
relay prologue. This is a physical context reduction, not cache accounting
or subtracting a constant from the displayed counter. The matched GHCP capture
reported 19,190, so approximately **5.4k tokens of route difference remain**;
this change does not eliminate the gateway's default spreadsheet prompt.
Production rejected a custom prompt-base override and relay-only top-level
tools, so neither is used.

A separate live two-stage contract test used the actual adapter's preparation,
native-call decoding, and result replay. The model returned the exact requested
`exec_command` call through the new carrier, then accepted its stubbed result
and returned the expected final text without another tool call. No shell or
remote connector action was executed by that test. Stage two naturally reused
18,608 cached input tokens. No automatic paid cache-priming calls were added.

Offline validation: **156 tests run, one skipped; all others passed**, using:

```sh
./.venv/bin/python -m unittest -q test_excel_lean_profile \
  test_excel_tool_disclosure test_tool_token_budget test_excel_upstream \
  test_copilot_sdk_upstream.CopilotSdkTranslationTests
```

Syntax checks and `git diff --check` also pass. The original Office-focused
assertions explicitly run in legacy mode, alongside new default-profile tests
for authentication preservation, opt-out, local registration, schema validation,
reserved-action rejection, namespace collisions, lazy disclosure, replay,
streaming, stable prefixes, compaction triggers, and source immutability.
