# codex-proxy

Local reverse proxy for Codex and Claude Code using GitHub Copilot upstream.

## What It Does

- Serves a local OpenAI-compatible endpoint at `http://localhost:8000/v1`
- Proxies Codex Responses API traffic to GitHub Copilot
- Also works with Claude Code
- Handles GitHub Copilot auth automatically on startup
- Exposes a local dashboard at `http://localhost:8000/`
- Shows tracked GitHub premium request usage for traffic that passes through the proxy
- Tracks session, token, and estimated cost data from the proxy's own request log

## Run It

For a quick macOS setup, assuming `python3`, `node`, and `npx` are already installed:

```bash
./install_macos.sh
```

That creates a local `.venv`, installs the Python dependencies, and leaves Codex/Claude activation to the dashboard so the backup-aware switch behavior is preserved.

From `~/sources/codex-proxy`:

```bash
python3 proxy.py
```

On first run, if no valid Copilot token is cached, the proxy will automatically start the GitHub device-flow login and prompt you in the terminal. After authorization, it caches the token and continues serving.

When running correctly, the proxy listens on:

```bash
http://localhost:8000/v1
```

The dashboard lives at:

```bash
http://localhost:8000/
```

## Configure Codex

Update `~/.codex/config.toml` so Codex points at the proxy:

```toml
# Primary Model Configuration
model_provider = "custom"
model = "gpt-5.4"
model_reasoning_effort = "high"
approvals_reviewer = "user"

[model_providers.custom]
name = "OpenAI"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

Important:

- `name` must be `"OpenAI"`
- `wire_api` must be `"responses"`
- `base_url` must point at the local proxy

## Configure Claude Code

Claude Code can use the same proxy, but the setup is different from Codex:

- create `~/.claude/settings.json` if it does not already exist

Recommended `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "128000",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"
  },
  "effortLevel": "medium"
}
```

Important:

- `ANTHROPIC_BASE_URL` should point at the proxy root, not `/v1`
- `ANTHROPIC_AUTH_TOKEN` can be any placeholder string because the proxy handles upstream auth
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` keeps Claude Code's local `/context` and auto-compact threshold aligned with the `128k` upstream input limit currently enforced on the Copilot-backed Claude path
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` aligns Claude Code's default `max_tokens` budget with the `64k` upstream output limit
- if you already enabled the proxy before these settings existed, re-enable Claude from the dashboard once so the missing caps are written into `~/.claude/settings.json`

## Billing Behavior

This repo is designed to bill like normal GitHub Copilot usage as closely as possible.

The basic idea is simple:

- normal requests are treated as user traffic
- if you want a request treated as agent traffic, prefix the prompt with `_`
- the proxy strips that leading `_` before forwarding the prompt upstream
- Claude Haiku requests are always treated as agent traffic
- the first unprefixed request is user traffic
- tool calls themselves are free

Examples:

```text
refactor the parser and run tests
```

This is sent as user traffic.

```text
_continue the tool-driven fix
```

This is sent as agent traffic, and the upstream model receives `continue the tool-driven fix`.

## Practical Use Of `_`

Use `_` only when you explicitly want agent initiator semantics for that request.

Example:

```text
_finish the refactor and run the tests
```

Use this sparingly.

- omit `_` when you want that specific request treated as user traffic
- use `_` when you want the request treated as agent traffic
- Claude Haiku requests are always treated as agent traffic
- while any proxied request is still active, the next user-looking request is forced to agent traffic
- after any proxied activity, the safeguard stays active for 15 more seconds and refreshes again on the next request start or finish
- remember that the leading `_` is removed before forwarding upstream
- be pragmatic

## Proxy Activation

The dashboard activation controls are meant to behave like a light switch.

- enabling a client that is already enabled is a no-op
- disabling a client that is already disabled is a no-op
- an existing config is backed up only when the proxy first replaces it
- disabling restores the latest backup when one exists
- if no backup exists, disabling removes the proxy-managed config

## Dashboard Data Sources

The dashboard combines two local data sources:

- proxy request logs in `~/.config/ghcp_proxy/usage-log.jsonl` for tracked GitHub premium request usage
- built-in token tracking and model pricing for Claude and GPT requests that pass through the proxy

The local cost estimates use the common model rates configured in `proxy.py`, including cached-input pricing where the provider publishes it.

If GitHub does not expose an official remaining premium-request count in the Copilot token payload, the dashboard shows a tracked remaining value based on successful proxied requests for the current month.

You can force official Copilot billing lookups with a token in one of two ways:

```bash
export GHCP_GITHUB_BILLING_TOKEN="gho_xxx"
```

You can also configure it from the dashboard at `http://localhost:8000/ui` under **GitHub Billing Token**.

If you are billed through an organization or enterprise account, set:

```bash
export GHCP_GITHUB_BILLING_SCOPE=org
export GHCP_GITHUB_BILLING_TARGET=my-org-slug
```

The dashboard cache is also persisted in SQLite for fast startup/refreshes:

```bash
export GHCP_CACHE_DB_PATH="~/.config/ghcp_proxy/ghcp-dashboard-cache.sqlite3"
```

