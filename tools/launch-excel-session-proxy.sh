#!/usr/bin/env bash
set -euo pipefail

port="8899"
timeout_seconds="90"
hold_seconds="300"
restart_excel="1"
system_proxy="1"

usage() {
  cat <<'USAGE'
Usage: launch-excel-session-proxy.sh [--port PORT] [--timeout SECONDS] [--hold SECONDS] [--no-restart] [--env-only]

Launch Microsoft Excel on macOS through the GHCP Proxy Excel capture proxy.
Start "Start proxy capture" in the dashboard first, trust the displayed local
CA certificate, then run this helper and send one ChatGPT Excel add-in message.

By default this temporarily enables the macOS Web Proxy and Secure Web Proxy for
active network services, opens Excel, waits for capture, and restores the prior
proxy settings before exiting. Use --env-only to skip system proxy changes and
launch Excel only with process proxy environment variables.

Options:
  --port PORT          Capture proxy port. Defaults to 8899.
  --timeout SECONDS    How long to wait for Excel to quit before reopening.
                       Defaults to 90.
  --hold SECONDS       How long to keep temporary system proxy settings enabled.
                       Defaults to 300.
  --no-restart         Do not ask a running Excel instance to quit first.
  --env-only           Only set proxy environment variables on the Excel launch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      port="${2:-}"
      shift 2
      ;;
    --timeout)
      timeout_seconds="${2:-}"
      shift 2
      ;;
    --hold)
      hold_seconds="${2:-}"
      shift 2
      ;;
    --no-restart)
      restart_excel="0"
      shift
      ;;
    --env-only)
      system_proxy="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper only works on macOS." >&2
  exit 1
fi
if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "Port must be between 1 and 65535." >&2
  exit 1
fi
if ! [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || (( timeout_seconds < 1 )); then
  echo "Timeout must be a positive number of seconds." >&2
  exit 1
fi
if ! [[ "$hold_seconds" =~ ^[0-9]+$ ]] || (( hold_seconds < 1 )); then
  echo "Hold must be a positive number of seconds." >&2
  exit 1
fi

excel_running() {
  osascript -e 'application "Microsoft Excel" is running' 2>/dev/null
}

enabled_network_services() {
  networksetup -listallnetworkservices \
    | sed '1{/^An asterisk/d;}' \
    | grep -v '^\*' \
    | while IFS= read -r service; do
        [[ -n "$service" ]] && printf '%s\n' "$service"
      done
}

proxy_state_line() {
  local kind="$1"
  local service="$2"
  if [[ "$kind" == "secure" ]]; then
    networksetup -getsecurewebproxy "$service" 2>/dev/null \
      | awk -F': ' '
          /^Enabled:/ { enabled=$2 }
          /^Server:/ { server=$2 }
          /^Port:/ { port=$2 }
          END { printf "%s\t%s\t%s", enabled, server, port }
        '
  else
    networksetup -getwebproxy "$service" 2>/dev/null \
      | awk -F': ' '
          /^Enabled:/ { enabled=$2 }
          /^Server:/ { server=$2 }
          /^Port:/ { port=$2 }
          END { printf "%s\t%s\t%s", enabled, server, port }
        '
  fi
}

set_proxy_state() {
  local kind="$1"
  local service="$2"
  local enabled="$3"
  local server="$4"
  local saved_port="$5"
  local command_prefix="webproxy"
  [[ "$kind" == "secure" ]] && command_prefix="securewebproxy"

  if [[ "$enabled" == "Yes" && -n "$server" && -n "$saved_port" && "$saved_port" != "0" ]]; then
    networksetup "-set${command_prefix}" "$service" "$server" "$saved_port" >/dev/null
    networksetup "-set${command_prefix}state" "$service" on >/dev/null
  else
    networksetup "-set${command_prefix}state" "$service" off >/dev/null
  fi
}

restore_file=""
restore_proxy_settings() {
  [[ -n "$restore_file" && -f "$restore_file" ]] || return 0
  echo "Restoring previous macOS proxy settings."
  while IFS=$'\t' read -r kind service enabled server saved_port; do
    [[ -n "$kind" && -n "$service" ]] || continue
    set_proxy_state "$kind" "$service" "$enabled" "$server" "$saved_port" || true
  done < "$restore_file"
  rm -f "$restore_file"
}

if [[ "$restart_excel" == "1" && "$(excel_running)" == "true" ]]; then
  echo "Asking Microsoft Excel to quit. Save or cancel from Excel if prompted."
  osascript -e 'tell application "Microsoft Excel" to quit' >/dev/null
  deadline=$((SECONDS + timeout_seconds))
  while [[ "$(excel_running)" == "true" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Microsoft Excel is still running; launch cancelled so the proxy env vars are not ignored." >&2
      exit 1
    fi
    sleep 1
  done
elif [[ "$restart_excel" == "0" && "$(excel_running)" == "true" ]]; then
  echo "Warning: Excel is already running, so macOS may reuse it without the proxy env vars." >&2
fi

proxy_url="http://127.0.0.1:${port}"
if [[ "$system_proxy" == "1" ]]; then
  if ! command -v networksetup >/dev/null 2>&1; then
    echo "networksetup was not found; rerun with --env-only or set the proxy manually." >&2
    exit 1
  fi
  restore_file="$(mktemp "${TMPDIR:-/tmp}/ghcp-excel-proxy-state.XXXXXX")"
  trap restore_proxy_settings EXIT INT TERM
  while IFS= read -r service; do
    [[ -n "$service" ]] || continue
    printf 'web\t%s\t%s\n' "$service" "$(proxy_state_line web "$service")" >> "$restore_file"
    printf 'secure\t%s\t%s\n' "$service" "$(proxy_state_line secure "$service")" >> "$restore_file"
    networksetup -setwebproxy "$service" 127.0.0.1 "$port" >/dev/null
    networksetup -setsecurewebproxy "$service" 127.0.0.1 "$port" >/dev/null
    networksetup -setwebproxystate "$service" on >/dev/null
    networksetup -setsecurewebproxystate "$service" on >/dev/null
  done < <(enabled_network_services)

  if [[ ! -s "$restore_file" ]]; then
    echo "No active macOS network services were found to proxy." >&2
    exit 1
  fi
  echo "Temporary macOS web proxy enabled on active network services."
fi

open -b com.microsoft.Excel \
  --env "HTTPS_PROXY=${proxy_url}" \
  --env "HTTP_PROXY=${proxy_url}" \
  --env "ALL_PROXY=${proxy_url}" \
  --env "https_proxy=${proxy_url}" \
  --env "http_proxy=${proxy_url}" \
  --env "all_proxy=${proxy_url}" \
  --env "NO_PROXY=localhost,127.0.0.1,::1" \
  --env "no_proxy=localhost,127.0.0.1,::1"

if [[ "$system_proxy" == "1" ]]; then
  echo "Opened Microsoft Excel with system proxy and proxy environment variables set to ${proxy_url}."
  echo "Send one ChatGPT Excel add-in message now."
  echo "Press Enter after capture to restore proxy settings, or wait ${hold_seconds}s for automatic restore."
  read -r -t "$hold_seconds" _ || true
else
  echo "Opened Microsoft Excel with proxy environment variables set to ${proxy_url}."
  echo "If the add-in ignores process env vars, rerun without --env-only so the helper can temporarily set the macOS proxy."
fi
