# GHCP Proxy

Local reverse proxy for Codex and Claude Code using GitHub Copilot upstream.

## Table Of Contents

- [What It Does](#what-it-does)
- [Setup Overview](#setup-overview)
- [Quick Start For Technical Users](#quick-start-for-technical-users)
- [Guided Setup For Non-Technical Users](#guided-setup-for-non-technical-users)
- [Setup Troubleshooting](#setup-troubleshooting)
- [Auto Update](#auto-update)
- [Client Activation](#client-activation)
- [Premium Requests](#premium-requests)
- [Request Prefixes](#request-prefixes)
- [Safeguard](#safeguard)
- [Dashboard Data Sources](#dashboard-data-sources)

## What It Does

- Serves a local OpenAI-compatible endpoint at `http://localhost:8000/v1`
- Proxies Codex Responses API traffic to GitHub Copilot
- Also works with Claude Code
- Activates Codex and Claude Code from the local dashboard
- Supports GitHub Copilot auth from the local dashboard on first run
- Exposes a local dashboard at `http://localhost:8000/`
- Shows tracked GitHub premium request usage for traffic that passes through the proxy
- Tracks session, token, and estimated cost data from the proxy's own request log

## Setup Overview

GHCP Proxy is a Python app with a web dashboard. Setup has two parts:

1. Start the local proxy with Python.
2. Use the dashboard at `http://localhost:8000/` to sign in to GitHub and enable
   Codex or Claude Code.

You need the following before starting:

- a GitHub account with GitHub Copilot access
- Python 3.10 or newer
- a local copy of this GHCP Proxy folder
- Codex and/or Claude Code installed, if you want the dashboard to configure them

Node.js is not required for normal proxy startup. Once `proxy.py` is running,
client setup is handled from the web dashboard; you usually do not need to edit
`~/.codex` or `~/.claude` by hand.

The proxy listens only on your local machine:

```text
http://localhost:8000/v1
```

The dashboard is:

```text
http://localhost:8000/
```

## Quick Start For Technical Users

Use this path if you are comfortable with Python virtual environments and a
terminal. Run these commands from the GHCP Proxy checkout.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python proxy.py
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\proxy.py
```

If PowerShell blocks activation, run the venv Python directly:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\proxy.py
```

Then open `http://localhost:8000/`, complete GitHub sign-in, go to
Settings -> Client Proxy, enable Codex and/or Claude Code, and restart any
already-open client sessions.

## Guided Setup For Non-Technical Users

Use this path if you want a little more context around each step. It does the
same thing as the quick start, just with the checkpoints spelled out.

### 1. Install Python

Install Python 3.10 or newer for your platform. If you are not sure whether you
already have Python, open a terminal and check.

If you need to install it, download Python from
[python.org/downloads](https://www.python.org/downloads/). On Windows, keep the
Python launcher option enabled during install.

Check your Python version.

macOS/Linux:

```bash
python3 --version
```

Windows PowerShell:

```powershell
py -3 --version
```

If that prints a Python version, you can continue. If it says the command was
not found, install Python first, then close and reopen your terminal.

### 2. Open The Project Folder

Open a terminal in the GHCP Proxy folder. That is the folder containing
`proxy.py`, `requirements.txt`, and `readme.md`.

Confirm you are in the right folder:

```bash
ls
```

Windows PowerShell:

```powershell
dir
```

You should see `proxy.py` and `requirements.txt`.

### 3. Create A Python Environment

The recommended setup is a `.venv` folder inside the GHCP Proxy folder. This
keeps the proxy's Python packages separate from your system Python.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks `Activate.ps1`, skip activation and use the direct
PowerShell commands in the install and launch steps below.

Alternative: put the virtual environment in your user profile instead of this
checkout.

macOS/Linux:

```bash
python3 -m venv ~/.venvs/ghcp_proxy
source ~/.venvs/ghcp_proxy/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv "$env:USERPROFILE\.venvs\ghcp_proxy"
& "$env:USERPROFILE\.venvs\ghcp_proxy\Scripts\Activate.ps1"
```

### 4. Install Dependencies

With the environment active, install the required Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are not using activation on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Start The Proxy

With the environment active:

```bash
python proxy.py
```

Windows PowerShell:

```powershell
python .\proxy.py
```

Without activation on Windows:

```powershell
.\.venv\Scripts\python.exe .\proxy.py
```

When startup works, the terminal prints something like:

```text
Starting GHCP proxy on http://localhost:8000 (loopback only)
Dashboard     : GET  /ui
```

Keep this terminal open while using GHCP Proxy. Closing it stops the proxy
unless you install the background startup helper later.

### 6. Finish Setup In The Dashboard

Open this URL in your browser:

```text
http://localhost:8000/
```

On first run, the dashboard will guide you through GitHub sign-in. After GitHub
authorizes the device code, return to the dashboard.

Then:

1. Open Settings -> Client Proxy.
2. Enable Codex, Claude Code, or both.
3. Restart any already-open Codex or Claude Code sessions.

You are set up when:

- the dashboard loads without an auth warning
- Settings -> Client Proxy shows the client you want as enabled
- new Codex or Claude Code sessions send requests through GHCP Proxy
- the dashboard Requests view starts showing recent traffic

### 7. Optional Background Startup

Settings -> Background Proxy can install a login starter so GHCP Proxy starts in
the background automatically. The same panel can install shell helpers:

- PowerShell on Windows: `Start-GHProxy` and `Stop-GHProxy`
- zsh on macOS: `start-ghproxy` and `stop-ghproxy`

## Setup Troubleshooting

If setup gets bumpy, start here.

### Python Command Not Found

If `python3 --version` or `py -3 --version` fails, Python is not available in
that terminal. Install Python 3.10 or newer, then close and reopen the terminal.

On Windows, keep the Python launcher enabled during install. If `python` opens
the Microsoft Store instead of Python, use `py -3` commands.

### `venv` Creation Fails

Make sure you are using Python 3.10 or newer:

```bash
python3 --version
```

Some Linux distributions package virtual environment support separately. If
`python3 -m venv .venv` says `venv` is missing, install your distribution's
Python venv package, then run the command again.

### PowerShell Will Not Activate `.venv`

If this command is blocked:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\proxy.py
```

### `No module named fastapi` Or `No module named uvicorn`

The dependencies were installed into a different Python than the one running
`proxy.py`. Use the same virtual environment for install and launch:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python proxy.py
```

Windows PowerShell without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\proxy.py
```

### Dashboard Does Not Open

Check the terminal running `proxy.py`.

- If it is still running, open `http://localhost:8000/`.
- If it exited with an error, fix the error and start `proxy.py` again.
- If the browser says it cannot connect, the proxy is not running or another
  program is blocking port `8000`.

### Port 8000 Is Already In Use

GHCP Proxy uses `127.0.0.1:8000`. If startup says the address is already in use,
another copy is probably already running.

Close any old GHCP Proxy terminal windows, or use the dashboard/background
helper to stop the background copy if one was installed. Then start `proxy.py`
again.

### GitHub Sign-In Does Not Complete

Keep the dashboard open and follow the device-code flow exactly. Make sure you
sign in with the GitHub account that has Copilot access. After GitHub approves
the device code, return to the dashboard and wait for it to refresh.

### Codex Or Claude Code Still Uses Its Old Provider

After enabling a client from Settings -> Client Proxy, fully restart any open
Codex or Claude Code sessions. These tools usually read their config only when a
new session starts.

If it still does not work, return to Settings -> Client Proxy, disable the
client, enable it again, then start a fresh client session.

### Upstream Requests Time Out

Raise the upstream timeout before starting the proxy:

```bash
export GHCP_UPSTREAM_TIMEOUT_SECONDS=300
python proxy.py
```

Windows PowerShell:

```powershell
$env:GHCP_UPSTREAM_TIMEOUT_SECONDS = "300"
python .\proxy.py
```

`GHCP_UPSTREAM_TIMEOUT_SECONDS` applies to upstream non-streaming requests,
including `/v1/responses/compact`. The default is `300` seconds.

### Background Startup Gets Confusing

The first setup is easiest with a visible terminal running `proxy.py`. Install
background startup only after the dashboard is working. If a background copy is
already installed, use Settings -> Background Proxy or the installed helper
commands to stop it before starting a manual copy.

## Auto Update

When GHCP Proxy is started from a git checkout, it checks the configured
upstream branch and updates itself when a safe upgrade is available. The proxy
restarts itself after a successful update.

After startup, the proxy keeps checking for safe updates every 15 minutes. If an
update is applied while upstream requests are active, the proxy
lets those requests finish before restarting. The dashboard shows when a restart
is pending or scheduled.

The updater is conservative:

- uses the checkout's existing upstream branch, such as `origin/main`
- runs `git fetch` and applies fast-forward updates when possible
- rebases committed local changes onto upstream when the checkout has diverged
- skips updates when the checkout is only ahead of upstream
- defaults to **user mode**, which stashes pending GHCP Proxy folder edits,
  updates, then reapplies those edits
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

## Client Activation

Use the dashboard's Settings -> Client Proxy controls to enable or disable
Codex and Claude Code. Treat those controls as the source of truth: they write
the required local files, preserve backups, and can restore the previous state
when a client is disabled.

Activation currently manages:

- Codex provider wiring and generated model catalog
- Claude Code proxy environment settings
- context and output caps needed for the Copilot-backed Claude path
- backups of proxy-managed config files before first replacement

You normally only need to start GHCP Proxy, open the dashboard, sign in to
GitHub if prompted, then enable the clients you want.

### Codex Details

Codex activation writes proxy-managed files under `~/.codex`:

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

The important parts are that Codex sees an OpenAI-compatible Responses provider
at `http://localhost:8000/v1` and loads the proxy-generated model catalog.

### Claude Code Details

Claude Code activation writes the proxy environment into
`~/.claude/settings.json`. The managed values look like this:

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

The important parts are:

- `ANTHROPIC_BASE_URL` should point at the proxy root, not `/v1`
- `ANTHROPIC_AUTH_TOKEN` can be any placeholder string because the proxy handles upstream auth
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` keeps Claude Code's local `/context` and auto-compact threshold aligned with the `128k` upstream input limit currently enforced on the Copilot-backed Claude path
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` aligns Claude Code's default `max_tokens` budget with the `64k` upstream output limit
- if you enabled Claude before these caps existed, disable and re-enable Claude
  from the dashboard once so the missing values are written

### Activation Behavior

The dashboard activation controls are meant to behave like a light switch.

- enabling a client that is already enabled is a no-op
- disabling a client that is already disabled is a no-op
- Codex activation writes `~/.codex/managed_config.toml` and `~/.codex/ghcp-proxy-models.json`
- if `~/.codex/config.toml` exists, Codex activation injects the proxy/provider/catalog keys there too
- an existing managed config is backed up only when the proxy first replaces it
- disabling restores the latest managed-config backup when one exists
- if no backup exists, disabling removes the proxy-managed files

## Premium Requests

GHCP Proxy tries to match GitHub Copilot's billing semantics as closely as the
local proxy can. The dashboard shows two related things:

- the quota snapshot GitHub sends back in `x-quota-snapshot-*` response headers
- the proxy's local premium-request estimate for requests it has seen

For the local estimate, only requests resolved as user traffic count against
premium usage. Requests resolved as agent traffic are recorded for history,
cost, and debugging, but count as `0` premium requests locally.

When a user request is counted, GHCP Proxy applies the model's premium-request
multiplier. Examples:

- most current Sonnet/GPT/Gemini Pro models: `1`
- Haiku, GPT mini, Gemini Flash, and Grok code-fast style models: usually `0.33`
- Opus-class and GPT-5.5-class models: higher multipliers such as `3` or `7.5`
- some older or mini models: `0`

The exact multiplier table lives in `constants.py`.

Tool calls do not add premium requests on their own. What matters is the model
request that produced or consumed the tool call.

## Request Prefixes

Most prompts should be sent without a prefix. GHCP Proxy inspects the request
shape and chooses user or agent initiator semantics automatically.

Use a prefix only when you want to override that default for a specific turn:

- `_` asks the proxy to treat the request as agent traffic
- `+` asks the proxy to treat the request as user traffic even if only the safeguard would have flipped it to agent traffic
- the prefix is stripped before the prompt is forwarded upstream
- `_foo` and `_ foo` both forward as `foo`; `+foo` and `+ foo` both forward as `foo`

Examples:

```text
refactor the parser and run tests
```

This is sent as user traffic.

```text
_continue the tool-driven fix
```

This is sent as agent traffic. The upstream model receives
`continue the tool-driven fix`.

```text
+summarize only the latest compiler error
```

This is sent as user traffic when only the safeguard would have flipped it to
agent. The upstream model receives `summarize only the latest compiler error`.

`+` is intentionally narrow. It does not override hard agent rules such as
Claude Haiku, explicit subagent traffic, compaction, or approval/security
monitor paths.

## Safeguard

The safeguard protects the boundary between user turns and tool-driven follow-up
traffic. It exists because coding clients often replay transcripts, continue
tool chains, or emit helper requests that can look user-like if you only inspect
one message in isolation.

The current rules are:

- the first real unprefixed user prompt is user traffic
- subagent traffic is agent traffic
- Claude Haiku traffic is agent traffic
- compaction and other forced internal paths are agent traffic
- after a proxied request finishes, user-looking traffic inside the cooldown window is treated as agent traffic
- `+` bypasses only that cooldown safeguard for the current request
- `_` explicitly chooses agent traffic for the current request

The default cooldown is `15` seconds. You can change it from the dashboard's
Safeguards page, or by editing the persisted safeguard config in the user config
directory. The file is `safeguard.json` under `GHCP_CONFIG_DIR` when that
environment variable is set; otherwise it uses the platform app config
directory, such as `~/Library/Application Support/ghcp_proxy` on macOS. The
allowed range is `0` to `600` seconds.

Practical guidance:

- leave prompts unprefixed for ordinary work
- use `_` when you are deliberately continuing an agent/tool chain
- use `+` when you are starting a fresh user turn immediately after prior proxy activity and you want that turn counted as user traffic
- do not use `+` to fight hard agent classifications; it is only a safeguard bypass

The dashboard's Requests and Safeguards views show the resolved initiator,
candidate initiator, cooldown, and safeguard reason for recent requests.

## Dashboard Data Sources

The dashboard combines two local data sources:

- proxy request logs in the OS user state directory for tracked GitHub Copilot traffic
- built-in token tracking and model pricing for Claude and GPT requests that pass through the proxy

The local cost estimates use the common model rates configured in `proxy.py`, including cached-input pricing where the provider publishes it.

### Premium Request Quota

Premium-request quota is read directly from the `x-quota-snapshot-*` response headers that GitHub Copilot sends back on every chat completion. No billing token, GitHub REST call, or plan picker is required; the proxy just reflects whatever upstream reports for the most recent finished request.

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
