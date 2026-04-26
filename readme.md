# GHCP Proxy

Local reverse proxy for Codex and Claude Code using GitHub Copilot upstream.

## Setup

You need Python 3.10 or newer, a GitHub account with Copilot access, and Codex
or Claude Code installed if you want GHCP Proxy to configure those clients.

From this folder, start the proxy:

```bash
python3 proxy.py
```

On Windows PowerShell:

```powershell
py -3 .\proxy.py
```

Then open the dashboard:

```text
http://localhost:8000/
```

In the dashboard:

1. Sign in to GitHub if prompted.
2. Open **Integrations**.
3. Enable Codex, Claude Code, or both.
4. Install the shell commands.
5. Optionally enable startup so GHCP Proxy starts at login.
6. Restart any already-open Codex or Claude Code sessions.

That is the normal setup. You do not need Node.js, `npx`, or hand-edited
`~/.codex` / `~/.claude` config files.

## First-Time Python Setup

If `python3 proxy.py` fails because packages are missing, create a local Python
environment once and install the dependencies:

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
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\proxy.py
```

After the dashboard is working, use **Integrations** to install the convenient
start/stop commands:

- macOS zsh: `start-ghproxy` and `stop-ghproxy`
- Windows PowerShell: `Start-GHProxy` and `Stop-GHProxy`

## Daily Use

If startup is enabled, GHCP Proxy starts automatically when you sign in.

If you prefer manual control, use the installed command:

macOS:

```bash
start-ghproxy
```

Windows PowerShell:

```powershell
Start-GHProxy
```

The proxy listens only on your local machine:

```text
http://localhost:8000/v1
```

The dashboard is:

```text
http://localhost:8000/
```

## Troubleshooting

Start here if setup does not work.

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

### `No module named fastapi` Or `No module named uvicorn`

The dependencies were installed into a different Python than the one running
`proxy.py`. Use the local virtual environment for install and launch:

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

After enabling a client from **Integrations**, fully restart any open Codex or
Claude Code sessions. These tools usually read their config only when a new
session starts.

If it still does not work, return to **Integrations**, disable the client,
enable it again, then start a fresh client session.

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
startup only after the dashboard is working. If a background copy is already
installed, use **Integrations** or the installed helper commands to stop it
before starting a manual copy.

## What It Does

- Serves a local OpenAI-compatible endpoint at `http://localhost:8000/v1`
- Proxies Codex Responses API traffic to GitHub Copilot
- Also works with Claude Code
- Installs Codex and Claude Code integrations from the local dashboard
- Supports GitHub Copilot auth from the local dashboard on first run
- Shows tracked GitHub premium request usage for traffic that passes through the proxy
- Tracks session, token, and estimated cost data from the proxy's own request log

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

## Integrations

Use the dashboard's **Integrations** page as the source of truth for local
setup. It can:

- connect Codex to GHCP Proxy
- connect Claude Code to GHCP Proxy
- install start/stop shell commands
- enable or disable startup at login
- restore the previous client configuration when an integration is disabled

The dashboard writes the required local config files and keeps backups before
replacing anything. Most users should not edit `~/.codex` or `~/.claude` by
hand.

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
