# Excel relay tool-token reduction — November 4, 2026

## Measured change

On a captured desktop request with 59 registered client tools, local
`o200k_base` estimates for the bridge instructions and catalog fall from
**13,356 to 2,149 tokens**: **11,207 fewer tokens, or 83.9%**.
Four tools remain eager in that capture and 55 are disclosed on demand.
These are local prompt-text estimates, not post-deployment billed usage.
No authenticated model calls were made for this verification.

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
permissions, gateway tool-version IDs, or live runtime settings were changed.
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
No supported way to disable that injection was verified in this change.

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
