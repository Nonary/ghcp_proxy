# GHCP Proxy

Local reverse proxy for Codex and Claude Code using GitHub Copilot upstream.

## Read This First

GHCP Proxy is an unofficial local proxy. It uses GitHub Copilot upstream in a
way GitHub may not support, and you should assume that misuse can violate
GitHub's terms or acceptable-use rules. GitHub's API terms say abusive or
excessive requests can lead to temporary or permanent suspension of API access,
and GitHub's acceptable-use rules include service usage limits.

Use your own account, respect GitHub's limits, and do not use this project to
evade billing or quotas. Rate limits make runaway usage less likely than it used
to be, but they are not permission to abuse the service. If you mark every
ordinary request as free/agent traffic or try to bypass premium-request
accounting, you are taking on account risk, including possible suspension or a
permanent ban.

GitHub remains the source of truth for billing and enforcement:

- [GitHub Copilot premium requests](https://docs.github.com/copilot/managing-copilot/monitoring-usage-and-entitlements/about-premium-requests)
- [GitHub Copilot usage limits](https://docs.github.com/en/copilot/concepts/rate-limits)
- [GitHub API terms](https://docs.github.com/github/site-policy/github-terms-of-service#h-api-terms)
- [GitHub Acceptable Use Policies](https://docs.github.com/site-policy/acceptable-use-policies/github-acceptable-use-policies)

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

## Premium Requests And Billing

GitHub Copilot bills and limits usage through premium requests, model
multipliers, monthly allowances, budgets, and rate limits. GHCP Proxy tries to
mirror GitHub's billing semantics where it can, but the dashboard is only a
local estimate. Your GitHub account and billing pages are the source of truth.

Practical rules:

- normal prompts you send should count as user traffic
- autonomous tool calls should not add premium requests by themselves
- model requests that produce or consume tool calls can still count
- premium models can consume more than one premium request through model
  multipliers
- requests resolved as agent traffic are tracked locally but estimated as `0`
  premium requests

For responsible use, leave prompts unprefixed most of the time and let the proxy
classify traffic. Use `_` only when you are deliberately continuing a tool-driven
agent workflow. Use `+` only when you are starting a fresh user request
immediately after prior proxy activity and want it counted as user traffic.

The dashboard shows recent request classification, local premium-request
estimates, and the latest upstream quota snapshot seen in GitHub's
`x-quota-snapshot-*` response headers. Treat that display as guidance, not a
guarantee.

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
