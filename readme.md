# GHCP Proxy

GHCP Proxy provides an OpenAI-compatible local endpoint for using GitHub Copilot with Codex and the ChatGPT app. Claude Code is also supported through a compatibility integration.

For organizations that provide the official ChatGPT Excel add-in but do not enable ChatGPT for Work or direct Codex access, GHCP Proxy can alternatively route Codex through the backend available to the authenticated Excel add-in session.

The dashboard handles authentication, integrations, usage, cost estimates, and optional startup management.

## Supported Backends

| Client                | Backend                        | Use                                                              |
| --------------------- | ------------------------------ | ---------------------------------------------------------------- |
| Codex and ChatGPT app | GitHub Copilot                 | Default                                                          |
| Codex                 | ChatGPT Excel backend          | Alternative when organizational access is provided through Excel |
| Claude Code           | GHCP Proxy compatibility route | Optional                                                         |

GHCP Proxy listens on loopback only.

```text
API:       http://127.0.0.1:8000/v1
Dashboard: http://127.0.0.1:8000/
```

## Quick Start

### Prerequisites

* Python 3.11 or newer
* GitHub Copilot access when using the Copilot backend
* Codex, the ChatGPT app, or Claude Code for the client you want to configure
* Excel desktop with the official ChatGPT add-in signed in when using the Excel backend

### macOS and Linux

From the repository directory:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python proxy.py
```

### Windows PowerShell

From the repository directory:

```powershell
py -3 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe ./proxy.py
```

Then open `http://127.0.0.1:8000/`.

On the first run:

1. Sign in to GitHub if prompted.
2. Open **Integrations**.
3. Enable the clients you want to use.
4. Optionally install the start and stop commands.
5. Optionally enable startup at login.
6. Restart any clients that were already running.

Node.js, `npx`, and manually edited client configuration files are not required for normal setup.

## Daily Use

Start the proxy with the repository virtual environment:

```bash
./.venv/bin/python proxy.py
```

If you installed the helper commands:

```bash
start-ghproxy
```

On Windows PowerShell:

```powershell
Start-GHProxy
```

After changing an integration, restart the affected client so it reloads its provider configuration.

## Backends

### GitHub Copilot

GitHub Copilot is the default backend. Use it when your GitHub account has Copilot access.

### ChatGPT Excel

Some organizations enable the official ChatGPT add-in for Excel without enabling ChatGPT for Work or direct Codex access.

Because the Excel add-in already provides an authenticated OpenAI backend and can support code-execution workflows, similar coding tasks can be performed from Excel. However, reproducing an agent such as Codex inside Excel requires additional prompting and orchestration.

GHCP Proxy removes that indirection by connecting Codex directly to the backend available through the authenticated Excel add-in session. This lets Codex handle the coding workflow while the model requests use the organization's existing Excel access.

In current testing, this has used substantially fewer tokens than recreating a comparable Codex workflow through a large orchestration prompt inside Excel. This is an observed result, not a guaranteed token-reduction ratio.

Use one of the Excel model aliases to select this route:

```text
gpt-6-astra-excel
gpt-5.6-luna-excel
gpt-5.6-terra-excel
gpt-5.6-sol-excel
```

The aliases support `low`, `medium`, `high`, and `xhigh` reasoning effort. `x-high` is accepted as an alias for `xhigh`.

Models without an `-excel` alias continue to use GitHub Copilot.

No Excel-specific prompt prefix or additional prompting syntax is required.

## Excel Setup

The Excel backend uses the authenticated session created by the official ChatGPT add-in.

It does not require network capture, DevTools, a debugging port, custom certificates, or operating-system proxy changes.

### Windows

Excel desktop must have the ChatGPT add-in signed in at least once. GHCP Proxy discovers the session from Office WebView2 local storage.

Check the detected session status with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/config/excel-session
```

### macOS

Excel desktop must have the ChatGPT add-in open and signed in. GHCP Proxy discovers the session from the local WebKit storage used by Excel.

If the session is missing or expired, reopen or refresh the signed-in Excel task pane and retry the request.

### Clear the Excel Session

Clear the cached session without stopping the proxy:

```powershell
Invoke-RestMethod -Method Delete http://127.0.0.1:8000/api/config/excel-session
```

Then reopen or refresh the ChatGPT add-in before retrying.

## Usage and Billing

The dashboard tracks usage and provides local cost estimates by backend and token type.

GitHub Copilot and Excel-backed usage are tracked separately. Provider-side usage and billing records remain authoritative.

The Excel route uses access already available through the authenticated Excel add-in session. GHCP Proxy does not create or modify organizational entitlements.

For current GitHub Copilot pricing and limits, see:

* [Models and pricing](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing)
* [Usage limits](https://docs.github.com/en/copilot/concepts/rate-limits)

## Integrations

The dashboard's **Integrations** page manages local client configuration. It can:

* connect Codex to GHCP Proxy
* connect the ChatGPT app to GHCP Proxy
* connect Claude Code to GHCP Proxy
* install start and stop commands
* enable or disable startup at login
* restore previous client configuration when an integration is disabled

Existing client configuration is backed up before replacement. Most users should manage integrations through the dashboard rather than editing configuration files manually.

## Configuration

Set environment variables before starting the proxy.

| Variable                        | Purpose                                     | Default or notes               |
| ------------------------------- | ------------------------------------------- | ------------------------------ |
| `GHCP_UPSTREAM_TIMEOUT_SECONDS` | Timeout for upstream non-streaming requests | `300` seconds                  |
| `GHCP_UPSTREAM_PROXY`           | Proxy for HTTP and HTTPS upstream traffic   | Can be overridden per protocol |
| `GHCP_HTTP_PROXY`               | HTTP upstream proxy                         | Optional                       |
| `GHCP_HTTPS_PROXY`              | HTTPS upstream proxy                        | Optional                       |
| `GHCP_NO_PROXY`                 | Hosts excluded from proxying                | Optional                       |
| `GHCP_UPSTREAM_TLS_VERIFY`      | Upstream TLS certificate verification       | Configure as required          |
| `GHCP_UPSTREAM_HTTP2`           | HTTP/2 for upstream requests                | Configure as required          |

Standard `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` variables are also honored.

### Enterprise Proxy Example

```bash
export GHCP_UPSTREAM_PROXY=http://proxy.example.com:8080
export GHCP_UPSTREAM_TLS_VERIFY=1
export GHCP_UPSTREAM_HTTP2=0

./.venv/bin/python proxy.py
```

## Troubleshooting

### Missing Python packages

Install dependencies and launch the proxy using the same virtual environment:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python proxy.py
```

Windows PowerShell:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe ./proxy.py
```

### Dashboard does not open

Check the terminal running `proxy.py`.

If it exited, resolve the reported error and restart it. If another process is using port `8000`, stop the old GHCP Proxy instance or conflicting process first.

### GitHub sign-in does not complete

Keep the dashboard open while completing the GitHub device-code flow with the account that has Copilot access.

After approval, return to the dashboard and wait for the status to refresh.

### Client still uses its old provider

Completely restart the client after changing its integration.

If necessary, disable and re-enable the integration from the dashboard and start a new client session.

### Upstream requests time out

Increase the timeout before starting the proxy:

```bash
export GHCP_UPSTREAM_TIMEOUT_SECONDS=600
./.venv/bin/python proxy.py
```

Windows PowerShell:

```powershell
$env:GHCP_UPSTREAM_TIMEOUT_SECONDS = "600"
./.venv/Scripts/python.exe ./proxy.py
```

### Excel session is missing or expired

Open the official ChatGPT add-in in Excel and confirm that it is signed in.

Refresh or reopen the task pane, then check:

```text
http://127.0.0.1:8000/api/config/excel-session
```

If necessary, clear the cached session and retry.

### Excel requests use the wrong backend

Make sure the selected model uses an `-excel` alias:

```text
gpt-6-astra-excel
gpt-5.6-luna-excel
gpt-5.6-terra-excel
gpt-5.6-sol-excel
```

Models without an Excel alias use GitHub Copilot.
