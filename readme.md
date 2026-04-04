# codex-proxy

Local reverse proxy for Codex and Claude Code using GitHub Copilot upstream.

## What It Does

- Serves a local OpenAI-compatible endpoint at `http://localhost:8000/v1`
- Proxies Codex Responses API traffic to GitHub Copilot
- Also works with Claude Code
- Handles GitHub Copilot auth automatically on startup
- Exposes a local dashboard at `http://localhost:8000/`
- Shows tracked GitHub premium request usage for traffic that passes through the proxy
- Pulls recent session, token, and savings data from both Claude Code and Codex via `ccusage`

## Run It

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

## Add Codex Instructions

It is still useful to configure `~/.codex/AGENTS.md`.

That file helps the terminal handle long-running work and follow-up turns more cleanly, even though the proxy now defaults requests to agent traffic unless you explicitly opt into a user request with `+`.

From `~/sources/codex-proxy`:

```bash
cp AGENTS.md ~/.codex/AGENTS.md
```

## Configure Claude Code

Claude Code can use the same proxy, but the setup is different from Codex:

- copy this repo's `AGENTS.md` to `~/.claude/CLAUDE.md`
- create `~/.claude/settings.json` if it does not already exist

It is still useful to configure `~/.claude/CLAUDE.md` for the same reason.

From `~/sources/codex-proxy`:

```bash
cp AGENTS.md ~/.claude/CLAUDE.md
```

Recommended `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"
  },
  "effortLevel": "medium"
}
```

Important:

- `ANTHROPIC_BASE_URL` should point at the proxy root, not `/v1`
- `ANTHROPIC_AUTH_TOKEN` can be any placeholder string because the proxy handles upstream auth
- Claude should use the same instruction file content, but the file must be named `CLAUDE.md`

## Billing Behavior

This repo is designed to bill like normal GitHub Copilot usage as closely as possible.

The basic idea is simple:

- every request is sent as agent traffic by default
- if you want a request treated as a user request, prefix the prompt with `+`
- the proxy strips that leading `+` before forwarding the prompt upstream
- tool calls themselves are free

Examples:

```text
refactor the parser and run tests
```

This is sent as agent traffic.

```text
+explain the parser architecture first
```

This is sent as a user request, and the upstream model receives `explain the parser architecture first`.

## Practical Use Of `+`

Use `+` only when you explicitly want user initiator semantics for that request.

Example:

```text
+finish the refactor and run the tests
```

Use this sparingly.

- use `+` when you want that specific request treated as user traffic
- omit `+` when you want the request treated as agent traffic
- remember that the leading `+` is removed before forwarding upstream
- be pragmatic

## Dashboard Data Sources

The dashboard combines two local data sources:

- proxy request logs in `~/.config/ghcp_proxy/usage-log.jsonl` for tracked GitHub premium request usage
- `ccusage` reports for Claude Code and Codex session/token/cost history

By default the proxy will try:

- `ccusage` for Claude Code
- `npx --yes @ccusage/codex@latest` for Codex

You can override either command if you already have a preferred local install:

```bash
export GHCP_CLAUDE_CCUSAGE_COMMAND="ccusage"
export GHCP_CODEX_CCUSAGE_COMMAND="npx --yes @ccusage/codex@latest"
```

If GitHub does not expose an official remaining premium-request count in the Copilot token payload, the dashboard shows a tracked remaining value based on successful proxied requests for the current month.

