# Initial-request tool token overhead

## Evidence

Recent completed first requests on the Excel-backed route reported roughly
43,500–43,800 input tokens. Captured requests contained about 71 KB of caller
tool definitions, including the desktop app's plugin namespaces. The proxy
expanded every callable tool into a text relay catalog because the Excel
gateway exposes its own server tools rather than accepting that caller catalog
as native function definitions.

For the captured `fa3bcb82…` request (`gpt-6-sol-excel`):

- Reported upstream input: **43,538 tokens**, including **22,132 cached** tokens.
- Local `o200k_base` estimate of visible message text before this change:
  **21,193 tokens**.
- Local estimate with the compact relay catalog: **18,628 tokens**, a reduction
  of **2,565**. This is a local text estimate, not measured post-fix billing.
- Approximately **22k tokens** were not explained by the visible message text.
  Small requests on this same gateway also reported around **22k input tokens**.
  This is consistent with gateway-injected Excel instructions and tool
  definitions, not a duplicated user message. These observations do not expose
  or precisely tokenize the hidden server prompt.

Cached tokens are included in the input/context total. A 44k input total does
not mean that all 44k tokens were fresh input.

## Fixes

### Default Copilot SDK route

The adapter previously set `defer="never"` on every caller tool. Plugin tools
are now eligible for the runtime's native tool search, with a total-tool
threshold of 20. Core top-level coding tools remain eager. Explicit leaf or
namespace `defer_loading=false` keeps a tool eager; `true` makes it eligible
for deferral. The `functions` namespace is treated as the core namespace.

This works in the existing hostless `mode="empty"`: no additional filesystem,
shell, or other unrelated built-ins are enabled. `tool_choice="none"` and
requests without caller tools keep both client tools and tool search disabled.
The same registration path handles the app's Responses Lite `additional_tools`
declarations.

A synthetic test against the actual SDK runtime, with every outbound request
intercepted, verified **43 deferred / 11 eager** tools from the captured catalog.
It also compared the generated deferred wire definitions with the runtime's
eager definitions and confirmed that descriptions and schemas were unchanged.
Local estimates of eagerly loaded schema JSON fell from **15,264** to **1,972**
tokens, excluding any hosted search/index overhead.
This verifies request construction, not real hosted search execution or billed
token savings. Deferred definitions still appear in the request JSON; native
tool search, not smaller HTTP request bytes, is what avoids eagerly loading
them into model context.

### Excel text relay route

- Tool descriptions are preserved verbatim outside escaped JSON strings.
- Ordinary schema scaffolding uses compact notation with an explicit legend.
  Constraints and descriptions remain present; unions, references, and unusual
  schema shapes fall back to full JSON.
- Qualified tool names no longer repeat separate namespace and leaf aliases.
- The reminder references the preceding catalog instead of repeating all names.
- Responses Lite declarations are cataloged once, rather than being passed
  through as conversation input. Original schemas still validate every emitted
  client tool call.
- The catalog and reminder remain in the stable prefix. Encrypted reasoning,
  native relay call identities, compaction triggers, and cache routing remain
  unchanged.

This route **does not** gain native lazy loading of caller tools. Its catalog
still contains every available client tool. The gateway's own large base
context is not removed or overridden. Therefore this change does not promise
that Excel-backed first requests will fall below 22k, or even below 40k for a
large desktop catalog. Removing that floor would require a separately verified
gateway capability, not deleting client instructions or inventing a tools
version ID.

## Verification

Use the repository virtualenv:

```sh
./.venv/bin/python -m unittest -q test_tool_token_budget \
  test_copilot_sdk_upstream.CopilotSdkTranslationTests \
  test_excel_upstream.ExcelUpstreamTests
./.venv/bin/python tools/verify-tool-token-budget.py
./.venv/bin/python tools/verify-tool-token-budget.py --archive /absolute/path/to/request-body.json
```

The runtime verification uses synthetic responses and credentials, does not
make paid model calls, and prints counts only. `tiktoken` is optional tooling for
local estimates, not a new runtime dependency. Restart the proxy to load code
changes; the first request after a schema/prompt layout change can have a
one-time cache-prefix miss.

## Official protocol reference

[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
states that callable definitions count as input/context tokens, recommends
keeping fewer than about 20 functions initially available, and describes native
tool search for loading deferred tools only when needed. The private Excel
gateway's injected context and supported controls are not established by this
public API documentation; the conclusions above are grounded in local request
and usage evidence.
