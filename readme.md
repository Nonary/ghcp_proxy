# codex-proxy

Local reverse proxy for Codex and Claude Code using GitHub Copilot upstream.

## What It Does

- Serves a local OpenAI-compatible endpoint at `http://localhost:8000/v1`
- Proxies Codex Responses API traffic to GitHub Copilot
- Also works with Claude Code
- Supports GitHub Copilot auth from the local dashboard on first run
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

Once the dashboard is open, Settings -> Background Proxy can install a user
login starter so GHCP Proxy starts in the background automatically. The same
panel can install shell helpers:

- PowerShell on Windows: `Start-GHProxy` and `Stop-GHProxy`
- zsh on macOS: `start-ghproxy` and `stop-ghproxy`

## Auto Update

When GHCP Proxy is started from a git checkout, it checks the configured
upstream branch and fast-forwards itself when a safe update is available. The
proxy restarts itself after a successful update.

After startup, the proxy keeps checking for safe fast-forward updates every 15
minutes. If an update is applied while upstream requests are active, the proxy
lets those requests finish before restarting. The dashboard shows when a restart
is pending or scheduled.

The updater is conservative:

- uses the checkout's existing upstream branch, such as `origin/main`
- runs `git fetch` and only applies `git merge --ff-only`
- skips updates when the checkout is ahead of or diverged from upstream
- defaults to **user mode**, which stashes pending GHCP Proxy folder edits,
  fast-forwards, then reapplies those edits
- blocks the upgrade if pending changes cannot be safely reapplied, and the
  dashboard offers an explicit "Apply upgrade anyway" override that discards
  pending local changes before upgrading
- supports **developer mode**, which never stashes or discards local code
  changes during upgrades and instead blocks until you commit or remove them
- records the most recent result in the user state directory

Environment knobs:

```bash
export GHCP_AUTO_UPDATE=0                     # disable
export GHCP_AUTO_UPDATE_MODE=developer        # default: user
export GHCP_AUTO_UPDATE_INTERVAL_SECONDS=900  # default: 15 minutes
export GHCP_AUTO_UPDATE_GIT_TIMEOUT_SECONDS=60 # default git command timeout
```

If upstream non-streaming requests are timing out, you can raise the proxy timeout before starting it:

```bash
export GHCP_UPSTREAM_TIMEOUT_SECONDS=300
python3 proxy.py
```

`GHCP_UPSTREAM_TIMEOUT_SECONDS` applies to upstream non-streaming requests, including `/v1/responses/compact`. The default is `300` seconds.

On first run, if no valid Copilot token is cached, the proxy still starts the local server immediately. Open the dashboard, start GitHub sign-in there, then click through to GitHub and enter the device code shown on the page. After authorization, the proxy caches the token and starts serving authenticated upstream traffic without needing a terminal prompt.

When running correctly, the proxy binds only to loopback and listens on:

```bash
http://localhost:8000/v1
```

The dashboard lives at:

```bash
http://localhost:8000/
```

## Configure Codex

Enable Codex from the dashboard when the proxy is running. Activation writes proxy-managed files under `~/.codex`:

- `managed_config.toml`
- `ghcp-proxy-models.json`

If `~/.codex/config.toml` already exists, activation also injects the proxy's
provider/catalog wiring there so Codex loads the generated model catalog while
preserving the user's selected `model`, reasoning effort, approvals, and
project trust entries.

The managed Codex config written by the dashboard is:

```toml
model_provider = "custom"
approvals_reviewer = "user"
model_catalog_json = "~/.codex/ghcp-proxy-models.json"
model_context_window = 272000
model_auto_compact_token_limit = 240000

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
- if you want to explicitly try a request as user traffic, prefix the prompt with `+`
- the proxy strips a leading `_` or `+` before forwarding the prompt upstream
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

```text
+summarize only the latest compiler error
```

This is sent as user traffic when only the safeguard would have flipped it to agent, and the upstream model receives `summarize only the latest compiler error`.

## Practical Use Of `_`

Use `_` only when you explicitly want agent initiator semantics for that request.

Example:

```text
_finish the refactor and run the tests
```

Use this sparingly.

- omit `_` when you want that specific request treated as user traffic
- use `+` when you want to disable only the safeguard for that specific request
- use `_` when you want the request treated as agent traffic
- `+` does not override logic that already forces agent traffic, such as Claude Haiku or other explicit agent-only paths
- Claude Haiku requests are always treated as agent traffic
- while any proxied request is still active, the next user-looking request is forced to agent traffic
- after any proxied activity, the safeguard stays active for 15 more seconds and refreshes again on the next request start or finish
- remember that a leading `_` or `+` is removed before forwarding upstream
- be pragmatic

## Proxy Activation

The dashboard activation controls are meant to behave like a light switch.

- enabling a client that is already enabled is a no-op
- disabling a client that is already disabled is a no-op
- Codex activation writes `~/.codex/managed_config.toml` and `~/.codex/ghcp-proxy-models.json`
- if `~/.codex/config.toml` exists, Codex activation injects the proxy/provider/catalog keys there too
- an existing managed config is backed up only when the proxy first replaces it
- disabling restores the latest managed-config backup when one exists
- if no backup exists, disabling removes the proxy-managed files

## Dashboard Data Sources

The dashboard combines two local data sources:

- proxy request logs in the OS user state directory for tracked GitHub Copilot traffic
- built-in token tracking and model pricing for Claude and GPT requests that pass through the proxy

The local cost estimates use the common model rates configured in `proxy.py`, including cached-input pricing where the provider publishes it.

### Premium Request Quota

Premium-request quota is read directly from the `x-quota-snapshot-*` response headers that GitHub Copilot sends back on every chat completion. No billing token, GitHub REST call, or plan picker is required — the proxy just reflects whatever upstream reports for the most recent finished request.

The dashboard surfaces the `premium_interactions` snapshot:

- included quota for the current period (e.g. 300 for Pro, 1000 for Enterprise, etc.; "Unlimited" when upstream reports `ent=-1`)
- absolute used / remaining and percent used / remaining
- reset date and `resets in N days` countdown
- overage and whether overage is permitted

Until you make at least one request through the proxy in the current session, the panel shows "Awaiting first request to capture quota".

The dashboard cache is also persisted in SQLite for fast startup/refreshes:

```bash
export GHCP_CACHE_DB_PATH="~/Library/Caches/ghcp_proxy/ghcp-dashboard-cache.sqlite3"
```
