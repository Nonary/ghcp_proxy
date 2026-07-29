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
ordinary request as free/agent traffic or try to bypass billing accounting, you
are taking on account risk, including possible suspension or a permanent ban.

GitHub remains the source of truth for billing and enforcement:

- [GitHub Copilot models and token pricing](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing)
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

## Token Pricing And Billing

GitHub Copilot usage-based billing prices input, cached-input, cache-write, and
output tokens according to the selected model, then converts the cost to GitHub
AI Credits (1 credit = $0.01 USD). GHCP Proxy applies the published per-token
rates to locally observed usage, but the dashboard remains an estimate. Your
GitHub account and billing pages are the source of truth.

For responsible use, leave prompts unprefixed most of the time and let the proxy
classify traffic. Use `_` only when you are deliberately continuing a tool-driven
agent workflow. Use `+` only when you are starting a fresh user request
immediately after prior proxy activity and want it counted as user traffic.

The dashboard shows recent requests, token counts, and estimated cost by token
type and provider. Treat that display as guidance, not a guarantee.

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

## GPT Excel Upstream

`gpt-excel` is a Responses-only model that sends requests to the official
ChatGPT Excel add-in backend instead of GitHub Copilot. It is experimental and
unofficial. The backend may change without notice, and using an add-in session
outside the add-in may be unsupported by OpenAI. Use only your own account and
session.

The Excel credential is deliberately separate from GitHub authentication. On
Windows it is encrypted with the current user's DPAPI key and reloaded after
proxy restarts. The bearer token and account ID are never stored as plaintext
or returned to the dashboard.

Prerequisites:

- Excel desktop with the official ChatGPT add-in open and signed in.
- The add-in WebView2 launched with remote debugging on port `9222`.
- Node.js 22 or newer for the one-shot session primer.

The easiest setup is through the dashboard:

1. Open **Integrations**.
2. Under **GPT Excel**, click **Capture Excel session**.
3. Send one message in the ChatGPT Excel add-in.
4. Wait for the dashboard to report **Ready · encrypted with Windows DPAPI**.

The command-line primer remains available:

```powershell
.\prime-excel-session.ps1
```

When prompted, send one message in the ChatGPT Excel add-in. The primer copies
only the bearer/account/client header bundle to the loopback proxy endpoint; it
does not print the token. GHCP Proxy encrypts the allowlisted bundle with DPAPI.
Check non-secret status with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/config/excel-session
```

After priming, select `gpt-excel` in Codex. Requests for all other models still
use GitHub Copilot. The Excel service currently accepts `gpt-5.5` on the wire
but may report the actual routed model (observed as `gpt-5.6-sol`) in Responses
events.

Basispoints rejects client-supplied tool schemas and injects its own
Excel-specific tools. GHCP Proxy therefore removes client tool declarations and
describes the caller's function and custom tools through a constrained relay
protocol. Tool requests are converted back into standard Responses
`function_call` or `custom_tool_call` events, and Codex executes them locally
under its normal approval policy. Tool outputs are accepted by Basispoints on
the following turn, so shell inspection and file-editing loops work without
sending unsupported tool schemas to the Excel service. The bridge intentionally
requests one client tool at a time; parallel client tool calls are not exposed
for `gpt-excel`.

Replayed history keeps its native Responses shape: Basispoints accepts
`function_call` / `function_call_output` items for tools it never declared
(verified by live replays), and the model only maintains its tool state
machine when it sees the standard item shapes. `update_plan` outputs carry an
appended steering directive — without it the plan-happy upstream harness
re-plans forever instead of working. Reasoning items with `encrypted_content`
are replayed unchanged (GPT-5 models repeat their previous action without
that continuity); bare reasoning items (which the stateless upstream would
reject) are dropped.

The rewrite is deterministic and the prompt is laid out for the upstream
prompt cache: the instructions and the tool catalog lead, conversation history
follows, and only a short protocol reminder trails. The catalog originally sat
last, because A/B replays showed the model ignores the marker protocol when
nothing restates it near generation — but a trailing ~3.5k-token catalog caps
the cache prefix at its first byte and re-bills it on every single turn. Wire
captures showed a floor of roughly 3.8k fresh input tokens per request from
that alone; replaying those same captures through the current layout cuts
fresh input by about 60%. The compact trailing reminder keeps the recency the
model needs at a fraction of the cost. Set
`GHCP_EXCEL_CATALOG_AT_PROMPT_END=1` to restore the old suffix layout if the
reminder ever stops holding the model to the protocol.

Byte-for-byte determinism is the other half of the cache story: the same
client request must produce the same outbound bytes, or a retry looks like new
work. Two things used to break that. The proxy stamps Copilot's per-request
affinity headers (`x-request-id`, `x-github-request-id`, `x-agent-task-id`) on
outbound requests and strips them for Responses upstreams — but that check
matched the path `/responses` exactly, so the Excel gateway's
`/basispoints/api/responses` fell through to the Copilot branch and got a fresh
identifier on every request. The check now matches by suffix. Separately, the
`turn_id` metadata was a random UUID per request; it and `task_id` are now
derived from the conversation identity and turn index, so a retried turn keeps
its identity.

Codex's `prompt_cache_key` is forwarded for cache routing (disable with
`GHCP_EXCEL_FORWARD_PROMPT_CACHE_KEY=0`), and the `task_id` metadata is derived
from it so one conversation keeps one task identity across turns.

Cost tracking prices `gpt-excel` at GPT-5.6 token rates (the observed routed
model), including the 272k long-context tier. The dashboard shows that spend
in OpenAI Credits ($0.04 = 1 Credit) instead of Copilot AIC; aggregate totals
that mix providers remain in AIC.

Clear both the active and encrypted session without stopping the proxy:

```powershell
Invoke-RestMethod -Method Delete http://127.0.0.1:8000/api/config/excel-session
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

### GitHub Auth Cannot Reach GitHub

If sign-in fails with a name-lookup, DNS, timeout, VPN, or firewall message,
first prove the terminal can reach GitHub:

```bash
curl -I https://github.com
```

If that fails, reconnect Wi-Fi or VPN and try again. If it works, start the
proxy again with `start-ghproxy` or `python3 proxy.py`.

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
- Shows input, cached-input, cache-write, output-token, and estimated AI-credit cost data
- Tracks session, token, and estimated cost data from the proxy's own request log
- Saves recent request prompt transcripts for dashboard drill-downs; set
  `GHCP_REQUEST_PROMPT_ARCHIVE_DIR` to override that local archive location

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
