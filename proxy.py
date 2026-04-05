"""
Lightweight GitHub Copilot reverse proxy — Responses API path.
Designed for Codex / codex-mini / gpt-5.1-codex and any model that
requires the Responses API instead of Chat Completions.

Usage:
  python ghcp_proxy.py
  → If no token exists, prompts you to authorize via GitHub device flow
  → Then starts serving on http://localhost:8000

Configure Codex:
  export OPENAI_BASE_URL=http://localhost:8000/v1
  export OPENAI_API_KEY=anything
"""

import asyncio
import base64
import compression.zstd as pyzstd
import gzip
import json
import glob
import os
import shutil
import sqlite3
import sys
import time
import zlib
from collections import deque
from datetime import datetime, timezone
from threading import Lock, Thread
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from initiator_policy import InitiatorPolicy

try:
    import brotli
except ImportError:
    brotli = None

app = FastAPI()

# ─── Constants ────────────────────────────────────────────────────────────────
GITHUB_CLIENT_ID        = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL  = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_KEY_URL      = "https://api.github.com/copilot_internal/v2/token"
GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com"

OPENCODE_VERSION = "1.3.13"
OPENCODE_HEADER_VERSION = "OpenCode/1.0"
OPENCODE_INTEGRATION_ID = "vscode-chat"
UPSTREAM_REQUESTS_PER_WINDOW = 5
UPSTREAM_REQUEST_WINDOW_SECONDS = 1.0

TOKEN_DIR         = os.path.expanduser("~/.config/ghcp_proxy")
ACCESS_TOKEN_FILE = os.path.join(TOKEN_DIR, "access-token")
BILLING_TOKEN_FILE = os.path.join(TOKEN_DIR, "billing-token")
API_KEY_FILE      = os.path.join(TOKEN_DIR, "api-key.json")
USAGE_LOG_FILE    = os.path.join(TOKEN_DIR, "usage-log.jsonl")
REQUEST_ERROR_LOG_FILE = os.path.join(TOKEN_DIR, "request-errors.log")
PROXY_BASE_URL    = "http://localhost:8000"
CODEX_PROXY_BASE_URL = f"{PROXY_BASE_URL}/v1"
DASHBOARD_BASE_URL = "http://localhost:8000"
DASHBOARD_FILE    = os.path.join(os.path.dirname(__file__), "dashboard.html")
SQLITE_CACHE_FILE = os.path.join(
    os.path.expanduser(os.environ.get("GHCP_CACHE_DB_PATH", os.path.join(TOKEN_DIR, ".ghcp_proxy-cache-v2.sqlite3")))
)
CODEX_CONFIG_DIR    = os.path.expanduser("~/.codex")
CODEX_CONFIG_FILE   = os.path.join(CODEX_CONFIG_DIR, "config.toml")
CLAUDE_CONFIG_DIR   = os.path.expanduser("~/.claude")
CLAUDE_SETTINGS_FILE = os.path.join(CLAUDE_CONFIG_DIR, "settings.json")
CODEX_PROXY_CONFIG = """\
model_provider = "custom"
model = "gpt-5.4"
model_reasoning_effort = "high"
approvals_reviewer = "user"

[model_providers.custom]
name = "OpenAI"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
"""
CLAUDE_PROXY_SETTINGS = {
    "env": {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
        "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1",
    },
    "effortLevel": "medium",
}
MAX_STORED_USAGE_EVENTS = 400
FORWARDED_REQUEST_HEADERS = (
    "session_id",
    "x-client-request-id",
    "x-openai-subagent",
)
FORWARDED_SERVER_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "x-github-request-id",
)
FAKE_COMPACTION_PREFIX = "ghcp_proxy_summary_v1:"
FAKE_COMPACTION_SUMMARY_LABEL = "[Compacted conversation summary]"
COMPACTION_SUMMARY_PROMPT = """Your task is to create a comprehensive, detailed summary of the entire conversation that captures all essential information needed to seamlessly continue the work without any loss of context. This summary will be used to compact the conversation while preserving critical technical details, decisions, and progress.
## Recent Context Analysis
Pay special attention to the most recent agent commands and tool executions that led to this summarization being triggered. Include:
- **Last Agent Commands**: What specific actions/tools were just executed
- **Tool Results**: Key outcomes from recent tool calls (truncate if very long, but preserve essential information)
- **Immediate State**: What was the system doing right before summarization
- **Triggering Context**: What caused the token budget to be exceeded
## Analysis Process
Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts systematically:
1. **Chronological Review**: Go through the conversation chronologically, identifying key phases and transitions
2. **Intent Mapping**: Extract all explicit and implicit user requests, goals, and expectations
3. **Technical Inventory**: Catalog all technical concepts, tools, frameworks, and architectural decisions
4. **Code Archaeology**: Document all files, functions, and code patterns that were discussed or modified
5. **Progress Assessment**: Evaluate what has been completed vs. what remains pending
6. **Context Validation**: Ensure all critical information for continuation is captured
7. **Recent Commands Analysis**: Document the specific agent commands and tool results from the most recent operations
## Summary Structure
Your summary must include these sections in order, following the exact format below:
<analysis>
[Chronological Review: Walk through conversation phases: initial request -> exploration -> implementation -> debugging -> current state]
[Intent Mapping: List each explicit user request with message context]
[Technical Inventory: Catalog all technologies, patterns, and decisions mentioned]
[Code Archaeology: Document every file, function, and code change discussed]
[Progress Assessment: What's done vs. pending with specific status]
[Context Validation: Verify all continuation context is captured]
[Recent Commands Analysis: Last agent commands executed, tool results (truncated if long), immediate pre-summarization state]

</analysis>

<summary>
1. Conversation Overview:
- Primary Objectives: [All explicit user requests and overarching goals with exact quotes]
- Session Context: [High-level narrative of conversation flow and key phases]
- User Intent Evolution: [How user's needs or direction changed throughout conversation]
2. Technical Foundation:
- [Core Technology 1]: [Version/details and purpose]
- [Framework/Library 2]: [Configuration and usage context]
- [Architectural Pattern 3]: [Implementation approach and reasoning]
- [Environment Detail 4]: [Setup specifics and constraints]
3. Codebase Status:
- [File Name 1]:
- Purpose: [Why this file is important to the project]
- Current State: [Summary of recent changes or modifications]
- Key Code Segments: [Important functions/classes with brief explanations]
- Dependencies: [How this relates to other components]
- [File Name 2]:
- Purpose: [Role in the project]
- Current State: [Modification status]
- Key Code Segments: [Critical code blocks]
- [Additional files as needed]
4. Problem Resolution:
- Issues Encountered: [Technical problems, bugs, or challenges faced]
- Solutions Implemented: [How problems were resolved and reasoning]
- Debugging Context: [Ongoing troubleshooting efforts or known issues]
- Lessons Learned: [Important insights or patterns discovered]
5. Progress Tracking:
- Completed Tasks: [What has been successfully implemented with status indicators]
- Partially Complete Work: [Tasks in progress with current completion status]
- Validated Outcomes: [Features or code confirmed working through testing]
6. Active Work State:
- Current Focus: [Precisely what was being worked on in most recent messages]
- Recent Context: [Detailed description of last few conversation exchanges]
- Working Code: [Code snippets being modified or discussed recently]
- Immediate Context: [Specific problem or feature being addressed before summary]
7. Recent Operations:
- Last Agent Commands: [Specific tools/actions executed just before summarization with exact command names]
- Tool Results Summary: [Key outcomes from recent tool executions - truncate long results but keep essential info]
- Pre-Summary State: [What the agent was actively doing when token budget was exceeded]
- Operation Context: [Why these specific commands were executed and their relationship to user goals]
8. Continuation Plan:
- [Pending Task 1]: [Details and specific next steps with verbatim quotes]
- [Pending Task 2]: [Requirements and continuation context]
- [Priority Information]: [Which tasks are most urgent or logically sequential]
- [Next Action]: [Immediate next step with direct quotes from recent messages]

</summary>

## Quality Guidelines
- **Precision**: Include exact filenames, function names, variable names, and technical terms
- **Completeness**: Capture all context needed to continue without re-reading the full conversation
- **Clarity**: Write for someone who needs to pick up exactly where the conversation left off
- **Verbatim Accuracy**: Use direct quotes for task specifications and recent work context
- **Technical Depth**: Include enough detail for complex technical decisions and code patterns
- **Logical Flow**: Present information in a way that builds understanding progressively
This summary should serve as a comprehensive handoff document that enables seamless continuation of all active work streams while preserving the full technical and contextual richness of the original conversation."""

_upstream_rate_limit_lock = asyncio.Lock()
_upstream_request_timestamps = deque()
_usage_log_lock = Lock()
_recent_usage_events = deque(maxlen=MAX_STORED_USAGE_EVENTS)
_session_request_id_lock = Lock()
_latest_server_request_ids_by_chain: dict[tuple[str, str], str] = {}
_active_server_request_ids_by_request: dict[str, dict[str, str | None]] = {}
_initiator_policy = InitiatorPolicy()
_ccusage_cache_lock = Lock()
_sqlite_cache_lock = Lock()
_sqlite_cache_enabled = True
_sqlite_cache_error = None
_premium_cache_lock = Lock()
_dashboard_stream_subscribers = set()
_dashboard_stream_lock = Lock()
_dashboard_stream_version = 0
_ccusage_cache = {
    "loaded_at": 0.0,
    "payload": None,
    "refreshing": False,
    "last_error": None,
    "last_started_at": None,
}
_premium_cache = {
    "loaded_at": 0.0,
    "payload": None,
    "refreshing": False,
    "last_error": None,
    "last_started_at": None,
}
CCUSAGE_CACHE_TTL_SECONDS = 300.0
PREMIUM_CACHE_TTL_SECONDS = 300.0

MODEL_PRICING = {
    "claude-haiku-3": {
        "provider": "Anthropic",
        "input_per_million": 0.25,
        "cached_input_per_million": 0.025,
        "output_per_million": 1.25,
    },
    "claude-haiku-3-5": {
        "provider": "Anthropic",
        "input_per_million": 0.80,
        "cached_input_per_million": 0.08,
        "output_per_million": 4.00,
    },
    "claude-haiku-4-5": {
        "provider": "Anthropic",
        "input_per_million": 1.00,
        "cached_input_per_million": 0.10,
        "output_per_million": 5.00,
    },
    "claude-sonnet-3-5": {
        "provider": "Anthropic",
        "input_per_million": 3.00,
        "cached_input_per_million": 0.30,
        "output_per_million": 15.00,
    },
    "claude-sonnet-3-7": {
        "provider": "Anthropic",
        "input_per_million": 3.00,
        "cached_input_per_million": 0.30,
        "output_per_million": 15.00,
    },
    "claude-sonnet-4.6": {
        "provider": "Anthropic",
        "input_per_million": 3.00,
        "cached_input_per_million": 0.30,
        "output_per_million": 15.00,
    },
    "claude-opus-3": {
        "provider": "Anthropic",
        "input_per_million": 15.00,
        "cached_input_per_million": 1.50,
        "output_per_million": 75.00,
    },
    "claude-opus-4.1": {
        "provider": "Anthropic",
        "input_per_million": 15.00,
        "cached_input_per_million": 1.50,
        "output_per_million": 75.00,
    },
    "claude-opus-4.6": {
        "provider": "Anthropic",
        "input_per_million": 5.00,
        "cached_input_per_million": 0.50,
        "output_per_million": 25.00,
    },
    "gpt-4.1": {
        "provider": "OpenAI",
        "input_per_million": 2.00,
        "cached_input_per_million": 0.50,
        "output_per_million": 8.00,
    },
    "gpt-4.1-mini": {
        "provider": "OpenAI",
        "input_per_million": 0.40,
        "cached_input_per_million": 0.10,
        "output_per_million": 1.60,
    },
    "gpt-4.1-nano": {
        "provider": "OpenAI",
        "input_per_million": 0.10,
        "cached_input_per_million": 0.025,
        "output_per_million": 0.40,
    },
    "gpt-4o": {
        "provider": "OpenAI",
        "input_per_million": 2.50,
        "cached_input_per_million": 1.25,
        "output_per_million": 10.00,
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "input_per_million": 0.15,
        "cached_input_per_million": 0.075,
        "output_per_million": 0.60,
    },
    "gpt-5": {
        "provider": "OpenAI",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.125,
        "output_per_million": 10.00,
    },
    "gpt-5-mini": {
        "provider": "OpenAI",
        "input_per_million": 0.25,
        "cached_input_per_million": 0.025,
        "output_per_million": 2.00,
    },
    "gpt-5-nano": {
        "provider": "OpenAI",
        "input_per_million": 0.05,
        "cached_input_per_million": 0.005,
        "output_per_million": 0.40,
    },
    "gpt-5.1": {
        "provider": "OpenAI",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.125,
        "output_per_million": 10.00,
    },
    "gpt-5.1-codex": {
        "provider": "OpenAI",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.125,
        "output_per_million": 10.00,
    },
    "gpt-5.1-codex-max": {
        "provider": "OpenAI",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.125,
        "output_per_million": 10.00,
    },
    "gpt-5.1-codex-mini": {
        "provider": "OpenAI",
        "input_per_million": 0.25,
        "cached_input_per_million": 0.025,
        "output_per_million": 2.00,
    },
    "gpt-5.2": {
        "provider": "OpenAI",
        "input_per_million": 1.75,
        "cached_input_per_million": 0.175,
        "output_per_million": 14.00,
    },
    "gpt-5.2-codex": {
        "provider": "OpenAI",
        "input_per_million": 1.75,
        "cached_input_per_million": 0.175,
        "output_per_million": 14.00,
    },
    "gpt-5.3-codex": {
        "provider": "OpenAI",
        "input_per_million": 1.75,
        "cached_input_per_million": 0.175,
        "output_per_million": 14.00,
    },
    "gpt-5.4": {
        "provider": "OpenAI",
        "input_per_million": 2.50,
        "cached_input_per_million": 0.25,
        "output_per_million": 15.00,
    },
    "gpt-5.4-mini": {
        "provider": "OpenAI",
        "input_per_million": 0.75,
        "cached_input_per_million": 0.075,
        "output_per_million": 4.50,
    },
    "gpt-5.4-nano": {
        "provider": "OpenAI",
        "input_per_million": 0.20,
        "cached_input_per_million": 0.02,
        "output_per_million": 1.25,
    },
}

MODEL_PRICING_ALIASES = {
    "anthropic/claude-haiku-3": "claude-haiku-3",
    "anthropic/claude-haiku-3.5": "claude-haiku-3-5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
    "anthropic/claude-opus-3": "claude-opus-3",
    "anthropic/claude-opus-4": "claude-opus-4.1",
    "anthropic/claude-opus-4.0": "claude-opus-4.1",
    "anthropic/claude-opus-4.1": "claude-opus-4.1",
    "anthropic/claude-opus-4.5": "claude-opus-4.6",
    "anthropic/claude-sonnet-3.5": "claude-sonnet-3-5",
    "anthropic/claude-sonnet-3.7": "claude-sonnet-3-7",
    "anthropic/claude-sonnet-4": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.6",
    "claude-haiku-4.5": "claude-haiku-4-5",
    "claude-haiku-3.5": "claude-haiku-3-5",
    "claude-haiku-3": "claude-haiku-3",
    "claude-opus-4": "claude-opus-4.1",
    "claude-opus-4.0": "claude-opus-4.1",
    "claude-opus-4.1": "claude-opus-4.1",
    "claude-opus-4.5": "claude-opus-4.6",
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-opus-3": "claude-opus-3",
    "claude-sonnet-3.5": "claude-sonnet-3-5",
    "claude-sonnet-3.7": "claude-sonnet-3-7",
    "claude-sonnet-4": "claude-sonnet-4.6",
    "claude-sonnet-4.5": "claude-sonnet-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "gpt-5.4 mini": "gpt-5.4-mini",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.4 nano": "gpt-5.4-nano",
    "gpt-5.4-nano": "gpt-5.4-nano",
    "gpt-5 mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5 nano": "gpt-5-nano",
    "gpt-5-nano": "gpt-5-nano",
}

PREMIUM_REQUEST_MULTIPLIERS = {
    "claude-haiku-4-5": 0.33,
    "claude-opus-4.5": 3.0,
    "claude-opus-4.6": 3.0,
    "claude-sonnet-4": 1.0,
    "claude-sonnet-4.5": 1.0,
    "claude-sonnet-4.6": 1.0,
    "gpt-4.1": 0.0,
    "gpt-4o": 0.0,
    "gpt-5.1": 1.0,
    "gpt-5.1-codex": 1.0,
    "gpt-5.1-codex-max": 1.0,
    "gpt-5.1-codex-mini": 0.33,
    "gpt-5.2": 1.0,
    "gpt-5.2-codex": 1.0,
    "gpt-5.3-codex": 1.0,
    "gpt-5.4": 1.0,
    "gpt-5.4-mini": 0.33,
    "gpt-5-mini": 0.0,
    "gemini-2.5-pro": 1.0,
    "gemini-3-flash": 0.33,
    "gemini-3-pro": 1.0,
    "gemini-3.1-pro": 1.0,
    "raptor-mini": 0.0,
}

SKU_PREMIUM_ALLOWANCES = {
    "copilot_free": 50,
    "free": 50,
    "pro": 300,
    "business": 300,
    "enterprise": 1000,
    "plus": 1500,
}


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _coerce_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _extract_payload_usage(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None

    usage = payload.get("usage")
    return _normalize_usage_payload(usage)


def _normalize_usage_payload(usage: dict | None) -> dict | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")

    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")

    cached_tokens = usage.get("cache_read_input_tokens")
    cached_tokens_from_raw_details = False
    if cached_tokens is None:
        cached_tokens = usage.get("cached_input_tokens")
    if cached_tokens is None:
        for details_key in ("input_tokens_details", "prompt_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict):
                cached_tokens = details.get("cached_tokens")
                if cached_tokens is not None:
                    cached_tokens_from_raw_details = True
                    break

    cache_creation_tokens = usage.get("cache_creation_input_tokens")
    reasoning_tokens = usage.get("reasoning_output_tokens")
    if reasoning_tokens is None:
        for details_key in ("output_tokens_details", "completion_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
                if reasoning_tokens is not None:
                    break
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = usage.get("totalTokens")

    normalized_input_tokens = _coerce_int(input_tokens, default=0)
    normalized_output_tokens = _coerce_int(output_tokens, default=0)
    normalized_cached_tokens = _coerce_int(cached_tokens, default=0)
    normalized_cache_creation_tokens = _coerce_int(cache_creation_tokens, default=0)
    normalized_reasoning_tokens = _coerce_int(reasoning_tokens, default=0)
    if cached_tokens_from_raw_details and normalized_cached_tokens > 0:
        normalized_input_tokens = max(0, normalized_input_tokens - normalized_cached_tokens)

    normalized_total_tokens = _coerce_int(total_tokens, default=None)
    if cached_tokens_from_raw_details:
        normalized_total_tokens = normalized_input_tokens + normalized_output_tokens
    elif normalized_total_tokens is None:
        normalized_total_tokens = normalized_input_tokens + normalized_output_tokens

    return {
        "input_tokens": normalized_input_tokens,
        "output_tokens": normalized_output_tokens,
        "total_tokens": normalized_total_tokens,
        "cached_input_tokens": normalized_cached_tokens,
        "cache_creation_input_tokens": normalized_cache_creation_tokens,
        "reasoning_output_tokens": normalized_reasoning_tokens,
    }


def _server_request_chain_key(
    session_id: str | None,
    client_request_id: str | None,
    subagent: str | None,
) -> tuple[str, str]:
    scope = None
    if isinstance(session_id, str) and session_id:
        scope = f"session:{session_id}"
    elif isinstance(client_request_id, str) and client_request_id:
        scope = f"client:{client_request_id}"
    else:
        scope = "global"
    normalized_subagent = subagent if isinstance(subagent, str) and subagent else "__root__"
    return (scope, normalized_subagent)


def _remember_server_request_id(event: dict | None):
    if not isinstance(event, dict):
        return
    chain_key = _server_request_chain_key(
        event.get("session_id"),
        event.get("client_request_id"),
        event.get("subagent"),
    )
    server_request_id = event.get("server_request_id")
    if not isinstance(server_request_id, str) or not server_request_id:
        return
    with _session_request_id_lock:
        _latest_server_request_ids_by_chain[chain_key] = server_request_id


def _remember_active_server_request_id(event: dict | None):
    if not isinstance(event, dict):
        return
    request_id = event.get("request_id")
    server_request_id = event.get("server_request_id")
    if not isinstance(request_id, str) or not request_id:
        return
    if not isinstance(server_request_id, str) or not server_request_id:
        return
    with _session_request_id_lock:
        _active_server_request_ids_by_request[request_id] = {
            "session_id": event.get("session_id"),
            "client_request_id": event.get("client_request_id"),
            "subagent": event.get("subagent"),
            "initiator": event.get("initiator"),
            "server_request_id": server_request_id,
        }


def _forget_active_server_request_id(request_id: str | None):
    if not isinstance(request_id, str) or not request_id:
        return
    with _session_request_id_lock:
        _active_server_request_ids_by_request.pop(request_id, None)


def _get_active_server_request_id_for_request(
    session_id: str | None,
    client_request_id: str | None,
    subagent: str | None,
    *,
    initiator: str | None = None,
) -> str | None:
    target_scope = _server_request_chain_key(session_id, client_request_id, subagent)
    with _session_request_id_lock:
        for context in reversed(list(_active_server_request_ids_by_request.values())):
            context_scope = _server_request_chain_key(
                context.get("session_id"),
                context.get("client_request_id"),
                context.get("subagent"),
            )
            if context_scope != target_scope:
                continue
            if initiator is not None and context.get("initiator") != initiator:
                continue
            server_request_id = context.get("server_request_id")
            if isinstance(server_request_id, str) and server_request_id:
                return server_request_id
    return None


def _get_latest_server_request_id_for_request(
    session_id: str | None,
    client_request_id: str | None,
    subagent: str | None,
) -> str | None:
    chain_key = _server_request_chain_key(session_id, client_request_id, subagent)
    with _session_request_id_lock:
        return _latest_server_request_ids_by_chain.get(chain_key)


def _resolve_server_request_id(
    request: Request,
    initiator: str | None,
) -> tuple[str, str | None]:
    explicit_server_request_id = (
        request.headers.get("x-request-id")
        or request.headers.get("request-id")
        or request.headers.get("x-github-request-id")
    )
    if explicit_server_request_id:
        return explicit_server_request_id, explicit_server_request_id

    session_id = request.headers.get("session_id")
    client_request_id = request.headers.get("x-client-request-id")
    subagent = request.headers.get("x-openai-subagent")

    if initiator == "agent":
        active_server_request_id = _get_active_server_request_id_for_request(
            session_id,
            client_request_id,
            subagent,
            initiator="user",
        )
        if isinstance(active_server_request_id, str) and active_server_request_id:
            return active_server_request_id, active_server_request_id

        prior_server_request_id = _get_latest_server_request_id_for_request(
            session_id,
            client_request_id,
            subagent,
        )
        if isinstance(prior_server_request_id, str) and prior_server_request_id:
            return prior_server_request_id, prior_server_request_id

    generated_server_request_id = str(uuid4())
    return generated_server_request_id, explicit_server_request_id


def _normalize_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower().replace("_", "-")
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    if normalized.startswith("openai/"):
        normalized = normalized.split("/", 1)[1]
    normalized = MODEL_PRICING_ALIASES.get(normalized, normalized)
    return normalized


def _usage_event_model_name(event: dict | None) -> str | None:
    if not isinstance(event, dict):
        return None

    for key in ("response_model", "resolved_model", "requested_model"):
        model_name = event.get(key)
        normalized = _normalize_model_name(model_name)
        if normalized:
            return normalized
    return None


def _usage_event_source(event: dict | None) -> str:
    if not isinstance(event, dict):
        return "codex"

    model_name = _usage_event_model_name(event)
    if isinstance(model_name, str):
        if model_name.startswith("claude-"):
            return "claude"
        if model_name.startswith("gpt-"):
            return "codex"

    path = str(event.get("path") or "")
    if path.endswith("/messages"):
        return "claude"
    if path.endswith("/responses") or path.endswith("/responses/compact") or path.endswith("/chat/completions"):
        return "codex"
    return "codex"


def _usage_event_group_key(event: dict | None) -> tuple[str, str]:
    if not isinstance(event, dict):
        return ("codex", "unknown")

    source = _usage_event_source(event)
    group_id = None
    for key in ("session_id", "client_request_id", "server_request_id", "request_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            group_id = value
            break

    if not isinstance(group_id, str) or not group_id:
        group_id = "unknown"
    return (source, group_id)


def _pricing_entry_for_model(model_name: str | None) -> dict | None:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return None
    return MODEL_PRICING.get(normalized)


def _anthropic_cache_creation_rate_per_million(entry: dict) -> float:
    input_rate = _coerce_float(entry.get("input_per_million"))
    cache_write_rate = _coerce_float(entry.get("cache_write_5m_per_million"))
    if cache_write_rate > 0:
        return cache_write_rate
    # The proxy does not record cache TTL separately, so default to the 5m write rate.
    return round(input_rate * 1.25, 6)


def _usage_event_cost(model_name: str | None, usage: dict | None) -> float:
    if not isinstance(usage, dict):
        return 0.0

    entry = _pricing_entry_for_model(model_name)
    if not isinstance(entry, dict):
        return 0.0

    input_tokens = _coerce_int(usage.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens"))
    cached_input_tokens = _coerce_int(usage.get("cached_input_tokens"))
    if cached_input_tokens == 0 and usage.get("cache_read_input_tokens") is not None:
        cached_input_tokens = _coerce_int(usage.get("cache_read_input_tokens"))
    cache_creation_input_tokens = _coerce_int(usage.get("cache_creation_input_tokens"))
    reasoning_output_tokens = _coerce_int(usage.get("reasoning_output_tokens"))

    input_rate = _coerce_float(entry.get("input_per_million"))
    output_rate = _coerce_float(entry.get("output_per_million"))
    cached_rate = entry.get("cached_input_per_million")
    if cached_rate is None and str(entry.get("provider") or "").lower() == "anthropic":
        cached_rate = round(input_rate * 0.1, 6)
    cached_rate = _coerce_float(cached_rate, default=input_rate)

    cache_creation_rate = input_rate
    if str(entry.get("provider") or "").lower() == "anthropic":
        cache_creation_rate = _anthropic_cache_creation_rate_per_million(entry)

    billable_output_tokens = output_tokens + reasoning_output_tokens
    return (
        (input_tokens * input_rate)
        + (cached_input_tokens * cached_rate)
        + (cache_creation_input_tokens * cache_creation_rate)
        + (billable_output_tokens * output_rate)
    ) / 1_000_000.0


def _premium_request_multiplier(model_name: str | None) -> float:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return 1.0
    return PREMIUM_REQUEST_MULTIPLIERS.get(normalized, 1.0)


def _load_api_key_payload() -> dict:
    try:
        with open(API_KEY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_sqlite_cache_unavailable(error: str):
    global _sqlite_cache_enabled, _sqlite_cache_error
    if _sqlite_cache_enabled:
        print(f"[sqlite] cache disabled: {error}", flush=True)
        _sqlite_cache_enabled = False
        _sqlite_cache_error = error


def _sqlite_connect() -> sqlite3.Connection:
    if not _sqlite_cache_enabled:
        raise RuntimeError("sqlite cache disabled")
    os.makedirs(TOKEN_DIR, exist_ok=True)
    connection = sqlite3.connect(SQLITE_CACHE_FILE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _init_sqlite_cache():
    if not _sqlite_cache_enabled:
        return False
    try:
        with _sqlite_cache_lock:
            with _sqlite_connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
        return True
    except Exception as exc:
        _set_sqlite_cache_unavailable(str(exc))
        return False


def _sqlite_cache_get(cache_key: str) -> dict | None:
    if not _init_sqlite_cache():
        return None
    try:
        with _sqlite_cache_lock:
            with _sqlite_connect() as connection:
                row = connection.execute(
                    "SELECT payload_json, updated_at FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
    except Exception as exc:
        _set_sqlite_cache_unavailable(str(exc))
        return None
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("loaded_at", row["updated_at"])
    return payload


def _sqlite_cache_get_latest(cache_key_prefix: str) -> dict | None:
    if not _init_sqlite_cache():
        return None
    try:
        with _sqlite_cache_lock:
            with _sqlite_connect() as connection:
                row = connection.execute(
                    "SELECT payload_json, updated_at FROM cache_entries WHERE cache_key LIKE ? ORDER BY updated_at DESC LIMIT 1",
                    (f"{cache_key_prefix}%",),
                ).fetchone()
    except Exception as exc:
        _set_sqlite_cache_unavailable(str(exc))
        return None
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("loaded_at", row["updated_at"])
    return payload


def _sqlite_cache_put(cache_key: str, payload: dict):
    if not isinstance(payload, dict):
        return
    if not _init_sqlite_cache():
        return
    updated_at = _utc_now_iso()
    serialized = json.dumps(payload, separators=(",", ":"), default=_json_default)
    try:
        with _sqlite_cache_lock:
            with _sqlite_connect() as connection:
                connection.execute(
                    """
                    INSERT INTO cache_entries (cache_key, payload_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (cache_key, serialized, updated_at),
                )
                connection.commit()
    except Exception as exc:
        _set_sqlite_cache_unavailable(f"failed to write cache key '{cache_key}': {exc}")


def _infer_premium_allowance(api_key_payload: dict) -> int | None:
    override = os.environ.get("GHCP_PREMIUM_REQUESTS_INCLUDED")
    if override:
        return _coerce_int(override, default=None)

    sku = str(api_key_payload.get("sku") or "").strip().lower()
    if not sku:
        return None

    if "plus" in sku:
        return SKU_PREMIUM_ALLOWANCES["plus"]
    if "enterprise" in sku:
        return SKU_PREMIUM_ALLOWANCES["enterprise"]
    if "business" in sku:
        return SKU_PREMIUM_ALLOWANCES["business"]
    if "pro" in sku:
        return SKU_PREMIUM_ALLOWANCES["pro"]
    if "free" in sku:
        return SKU_PREMIUM_ALLOWANCES["free"]
    return None


def _extract_quota_summary(api_key_payload: dict) -> dict:
    included = _infer_premium_allowance(api_key_payload)
    sku = api_key_payload.get("sku")
    reset_date = api_key_payload.get("limited_user_reset_date")
    quotas = api_key_payload.get("limited_user_quotas")
    remaining = None
    if isinstance(quotas, dict):
        for key in ("remaining", "premium_requests_remaining", "available", "left"):
            if key in quotas:
                remaining = _coerce_float(quotas.get(key), default=None)
                break

    return {
        "sku": sku,
        "included": included,
        "official_remaining": remaining,
        "reset_date": reset_date,
    }


def _github_rest_headers(access_token: str, scheme: str = "Bearer") -> dict:
    authorization = (
        f"Bearer {access_token}"
        if scheme.lower() == "bearer"
        else f"token {access_token}"
    )
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": authorization,
        "X-GitHub-Api-Version": "2026-03-10",
        "User-Agent": "ghcp-proxy-dashboard",
    }


def _github_rest_get_json(access_token: str, url: str, params: dict | None = None) -> tuple[dict, str]:
    if not access_token:
        raise RuntimeError("No GitHub token provided for GitHub REST call")
    last_error = None
    for scheme in ("Bearer", "token"):
        headers = _github_rest_headers(access_token, scheme=scheme)
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=headers, params=params)
                status = response.status_code
                if status == 401:
                    last_error = f"{scheme} auth failed: 401 unauthorized"
                    continue
                if status == 403:
                    body = (response.text or "").strip()
                    last_error = (
                        f"{scheme} auth got 403 forbidden: {body[:240] if body else 'forbidden'}"
                    )
                    if scheme == "Bearer":
                        continue
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"GitHub REST request to {url} timed out: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"GitHub REST response from {url} was not JSON: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if scheme == "Bearer" and exc.response.status_code in {401, 403}:
                continue
            raise RuntimeError(f"GitHub REST request to {url} failed: {exc.response.status_code} {exc.response.text}") from exc
        except Exception as exc:
            raise RuntimeError(f"GitHub REST request to {url} failed ({scheme}): {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GitHub REST response from {url} had non-dict payload")
        return payload, scheme
    raise RuntimeError(last_error or f"GitHub REST request to {url} failed with unsupported authentication scheme")


def _load_cached_github_identity() -> dict | None:
    return _sqlite_cache_get("github_identity")


def _fetch_github_identity(access_token: str) -> dict:
    cached = _load_cached_github_identity()
    if isinstance(cached, dict) and isinstance(cached.get("login"), str) and cached.get("login"):
        return cached

    payload, _ = _github_rest_get_json(access_token, "https://api.github.com/user")
    if not isinstance(payload, dict) or not isinstance(payload.get("login"), str):
        raise RuntimeError("GitHub user API did not return a valid login")

    identity = {"login": payload["login"]}
    _sqlite_cache_put("github_identity", identity)
    return identity


def _load_billing_org_candidates(access_token: str) -> list[str]:
    try:
        payload, _ = _github_rest_get_json(access_token, "https://api.github.com/user/orgs")
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    candidates = []
    for org in payload:
        if not isinstance(org, dict):
            continue
        login = org.get("login")
        if isinstance(login, str) and login:
            candidates.append(login)
    return candidates


def _billing_target_from_env_or_identity(identity: dict) -> tuple[str, str]:
    scope = str(os.environ.get("GHCP_GITHUB_BILLING_SCOPE") or "user").strip().lower()
    target = str(os.environ.get("GHCP_GITHUB_BILLING_TARGET") or "").strip()
    if scope == "user":
        return "user", target or identity["login"]
    if scope in {"organization", "org"}:
        if not target:
            raise RuntimeError("GHCP_GITHUB_BILLING_TARGET is required when GHCP_GITHUB_BILLING_SCOPE=org")
        return "org", target
    if scope == "enterprise":
        if not target:
            raise RuntimeError("GHCP_GITHUB_BILLING_TARGET is required when GHCP_GITHUB_BILLING_SCOPE=enterprise")
        return "enterprise", target
    raise RuntimeError(f"Unsupported GHCP_GITHUB_BILLING_SCOPE: {scope}")


def _official_premium_cache_key(scope: str, target: str, year: int, month: int) -> str:
    return f"premium_usage:{scope}:{target}:{year:04d}:{month:02d}"


def _premium_usage_endpoint(scope: str, target: str) -> str:
    if scope == "user":
        return f"https://api.github.com/users/{target}/settings/billing/premium_request/usage"
    if scope == "org":
        return f"https://api.github.com/organizations/{target}/settings/billing/premium_request/usage"
    if scope == "enterprise":
        return f"https://api.github.com/enterprises/{target}/settings/billing/premium_request/usage"
    raise RuntimeError(f"Unsupported billing scope: {scope}")


def _extract_explicit_remaining_from_billing_payload(payload: dict) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidate_paths = (
        ("remainingQuota",),
        ("remaining",),
        ("quota", "remaining"),
        ("includedUsage", "remaining"),
        ("entitlement", "remaining"),
    )
    for path in candidate_paths:
        current = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return _coerce_float(current, default=None)
    return None


def _extract_included_from_billing_payload(payload: dict) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("included", "entitlement", "allowed", "quota"):
        if key in payload:
            value = _coerce_int(payload.get(key), default=None)
            if value is not None:
                return value
    usage_summary = payload.get("summary")
    if isinstance(usage_summary, dict):
        for key in ("included", "allowed", "quota"):
            if key in usage_summary:
                value = _coerce_int(usage_summary.get(key), default=None)
                if value is not None:
                    return value
    return None


def _infer_remaining_from_billing_payload(payload: dict, included: int | None) -> tuple[float | None, float]:
    usage_items = payload.get("usageItems") if isinstance(payload, dict) else None
    if not isinstance(usage_items, list):
        return None, 0.0

    included_used = 0.0
    total_used = 0.0
    for item in usage_items:
        if not isinstance(item, dict):
            continue
        total_used += _coerce_float(item.get("grossQuantity"))
        discount_quantity = item.get("discountQuantity")
        net_quantity = item.get("netQuantity")
        if discount_quantity is not None:
            included_used += _coerce_float(discount_quantity)
        elif net_quantity is not None:
            included_used += max(_coerce_float(item.get("grossQuantity")) - _coerce_float(net_quantity), 0.0)
    if included is None:
        return None, total_used
    return max(included - included_used, 0.0), total_used


def _empty_official_premium_payload() -> dict:
    return {
        "available": False,
        "loaded_at": None,
        "scope": None,
        "target": None,
        "remaining": None,
        "used": 0.0,
        "included": None,
        "reset_date": None,
        "source": "github-rest-billing-api",
        "raw": None,
        "error": None,
        "inference": None,
    }


def _collect_official_premium_payload(now: datetime | None = None, *, skip_cache: bool = False) -> dict:
    current = now or _utc_now()
    access_token = _load_billing_token() or _load_access_token()
    if not access_token:
        raise RuntimeError(
            "No GitHub OAuth token is available for billing API requests. "
            "Set GHCP_GITHUB_BILLING_TOKEN, set a billing token via /api/config/billing-token (UI), or run proxy auth."
        )

    identity = _fetch_github_identity(access_token)
    scope, target = _billing_target_from_env_or_identity(identity)
    quota_summary = _extract_quota_summary(_load_api_key_payload())
    included = quota_summary.get("included")
    explicit_scope = os.environ.get("GHCP_GITHUB_BILLING_SCOPE")
    candidates: list[tuple[str, str]] = []
    seen_candidates = set()
    candidates.append((scope, target))
    seen_candidates.add(f"{scope}:{target}")
    if not explicit_scope:
        for org_login in _load_billing_org_candidates(access_token):
            candidate_key = f"org:{org_login}"
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            candidates.append(("org", org_login))

    payload = None
    last_error = None
    for attempt_scope, attempt_target in candidates:
        cache_key = _official_premium_cache_key(attempt_scope, attempt_target, current.year, current.month)
        if not skip_cache:
            cached = _sqlite_cache_get(cache_key)
            if isinstance(cached, dict):
                return cached

        endpoint = _premium_usage_endpoint(attempt_scope, attempt_target)
        params = {"year": current.year, "month": current.month}
        fetch_attempts = (
            (params, "with month filter"),
            ({}, "without month filter"),
        )
        for attempt_params, attempt_label in fetch_attempts:
            try:
                payload, _ = _github_rest_get_json(access_token, endpoint, params=attempt_params)
                scope, target = attempt_scope, attempt_target
                break
            except Exception as exc:
                last_error = str(exc)
                if "without month filter" in attempt_label:
                    print(f"Billing API fallback attempt failed for {attempt_scope}:{attempt_target} ({attempt_label}): {exc}", flush=True)
                    continue
                print(f"Billing API month-filtered call failed ({attempt_label}): {exc}", flush=True)
        if payload is not None:
            break

    if payload is None:
        raise RuntimeError(f"GitHub billing API failed after probing identities: {last_error}")

    endpoint_included = _extract_included_from_billing_payload(payload if isinstance(payload, dict) else {})
    effective_included = _coerce_int(endpoint_included, default=None)
    if effective_included is None:
        effective_included = _coerce_int(included, default=None)

    explicit_remaining = _extract_explicit_remaining_from_billing_payload(payload if isinstance(payload, dict) else {})
    inferred_remaining, total_used = _infer_remaining_from_billing_payload(
        payload if isinstance(payload, dict) else {}, effective_included
    )
    reset_date = quota_summary.get("reset_date")
    if isinstance(payload, dict):
        reset_date = payload.get("resetDate") or payload.get("reset_date") or reset_date
    result = {
        "available": True,
        "loaded_at": _utc_now_iso(),
        "scope": scope,
        "target": target,
        "remaining": explicit_remaining if explicit_remaining is not None else inferred_remaining,
        "used": total_used,
        "included": effective_included,
        "reset_date": reset_date,
        "source": "github-rest-billing-api",
        "raw": payload if isinstance(payload, dict) else None,
        "error": None,
        "inference": "explicit" if explicit_remaining is not None else "usageItems",
    }
    _sqlite_cache_put(cache_key, result)
    _sqlite_cache_put("premium_usage:latest", result)
    return result


def _current_billing_month_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or _utc_now()
    start = datetime(current.year, current.month, 1, tzinfo=timezone.utc)
    if current.month == 12:
        end = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _load_usage_history():
    if not os.path.exists(USAGE_LOG_FILE):
        return

    try:
        with open(USAGE_LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    normalized_usage = _normalize_usage_payload(payload.get("usage"))
                    if isinstance(normalized_usage, dict):
                        payload["usage"] = normalized_usage
                        if payload.get("cost_usd") is None:
                            payload["cost_usd"] = _usage_event_cost(_usage_event_model_name(payload), normalized_usage)
                    _recent_usage_events.append(payload)
                    _remember_server_request_id(payload)
    except OSError:
        pass


def _record_usage_event(event: dict):
    if not isinstance(event, dict):
        return

    os.makedirs(TOKEN_DIR, exist_ok=True)
    serialized = json.dumps(event, separators=(",", ":"), default=_json_default)
    with _usage_log_lock:
        with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(serialized)
            f.write("\n")
        _recent_usage_events.append(event)
    _remember_server_request_id(event)
    _notify_dashboard_stream_listeners()


def _record_request_error(event: dict):
    if not isinstance(event, dict):
        return

    os.makedirs(TOKEN_DIR, exist_ok=True)
    serialized = json.dumps(event, separators=(",", ":"), default=_json_default)
    with open(REQUEST_ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(serialized)
        f.write("\n")


def _register_dashboard_stream_listener() -> asyncio.Queue[int]:
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
    with _dashboard_stream_lock:
        _dashboard_stream_subscribers.add(queue)
    return queue


def _unregister_dashboard_stream_listener(queue: asyncio.Queue[int]):
    with _dashboard_stream_lock:
        _dashboard_stream_subscribers.discard(queue)


def _notify_dashboard_stream_listeners():
    global _dashboard_stream_version

    _dashboard_stream_version += 1
    with _dashboard_stream_lock:
        if not _dashboard_stream_subscribers:
            return
        subscribers = list(_dashboard_stream_subscribers)

    for queue in subscribers:
        try:
            queue.put_nowait(_dashboard_stream_version)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(_dashboard_stream_version)
            except asyncio.QueueEmpty:
                pass
            except RuntimeError:
                _unregister_dashboard_stream_listener(queue)


class _SSEUsageCapture:
    def __init__(self, stream_type: str):
        self.stream_type = stream_type
        self.buffer = ""
        self.usage = None

    def _has_text(self, value) -> bool:
        if not isinstance(value, str):
            return False
        return bool(value.strip())

    def _consume_chat_payload(self, payload: dict) -> bool:
        if isinstance(payload.get("usage"), dict):
            self.usage = _normalize_usage_payload(payload["usage"])

        choices = payload.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        delta = first_choice.get("delta") if isinstance(first_choice, dict) else {}
        return self._has_text(_extract_text_from_chat_delta(delta))

    def _consume_responses_payload(self, payload: dict) -> bool:
        event_type = str(payload.get("type", "")).strip().lower()
        response = payload.get("response")
        if isinstance(response, dict):
            if isinstance(response.get("usage"), dict):
                self.usage = _normalize_usage_payload(response["usage"])
        elif isinstance(payload.get("usage"), dict):
            self.usage = _normalize_usage_payload(payload["usage"])

        has_output = False
        if event_type == "response.output_text.delta":
            has_output = self._has_text(payload.get("delta"))
        if event_type == "response.output_text.done":
            has_output = self._has_text(payload.get("text"))
        if event_type == "response.output_item.added":
            item = payload.get("item")
            if isinstance(item, dict):
                has_output = self._has_text(_extract_item_text(item))
        if event_type == "response.content_part.added":
            part = payload.get("part")
            if isinstance(part, dict):
                has_output = self._has_text(
                    part.get("text") or part.get("input_text") or part.get("output_text")
                )
        return has_output

    def feed(self, chunk) -> bool:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = str(chunk)

        self.buffer += text
        normalized = self.buffer.replace("\r\n", "\n")
        saw_output = False

        while "\n\n" in normalized:
            raw_block, normalized = normalized.split("\n\n", 1)
            _event_name, data = _parse_sse_block(raw_block)
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if self.stream_type == "chat":
                saw_output = self._consume_chat_payload(payload) or saw_output
            else:
                saw_output = self._consume_responses_payload(payload) or saw_output

        self.buffer = normalized
        return saw_output


def _start_usage_event(
    request: Request,
    requested_model: str | None,
    resolved_model: str | None,
    initiator: str | None,
    request_id: str | None = None,
    request_body: dict | None = None,
    upstream_path: str | None = None,
    outbound_headers: dict | None = None,
) -> dict:
    server_request_id, prior_server_request_id = _resolve_server_request_id(request, initiator)
    if isinstance(outbound_headers, dict):
        outbound_headers["x-request-id"] = server_request_id
        outbound_headers["x-github-request-id"] = server_request_id
    event = {
        "request_id": request_id or uuid4().hex,
        "started_at": _utc_now_iso(),
        "path": request.url.path,
        "method": request.method,
        "upstream_path": upstream_path,
        "requested_model": requested_model,
        "resolved_model": resolved_model or requested_model,
        "initiator": initiator,
        "session_id": request.headers.get("session_id"),
        "client_request_id": request.headers.get("x-client-request-id"),
        "subagent": request.headers.get("x-openai-subagent"),
        "server_request_id": server_request_id,
        "prior_server_request_id": prior_server_request_id,
        "_started_monotonic": time.perf_counter(),
    }
    _remember_server_request_id(event)
    _remember_active_server_request_id(event)
    return event


def _mark_usage_event_first_output(event: dict | None):
    if not isinstance(event, dict):
        return
    if event.get("_first_output_monotonic") is None:
        event["_first_output_monotonic"] = time.perf_counter()


def _finish_usage_event(
    event: dict | None,
    status_code: int,
    *,
    upstream: httpx.Response | None = None,
    response_payload: dict | None = None,
    response_text: str | None = None,
    usage: dict | None = None,
):
    if not isinstance(event, dict):
        return

    finished_at = _utc_now()
    _forget_active_server_request_id(event.get("request_id"))
    _initiator_policy.note_request_finished(event.get("request_id"), finished_at=finished_at)
    finished_event = {
        **{key: value for key, value in event.items() if not str(key).startswith("_")},
        "finished_at": finished_at.isoformat(),
        "status_code": status_code,
        "success": status_code < 400,
    }

    started_monotonic = event.get("_started_monotonic")
    if isinstance(started_monotonic, (int, float)):
        finished_event["duration_ms"] = max(0, int(round((time.perf_counter() - started_monotonic) * 1000)))

    first_output_monotonic = event.get("_first_output_monotonic")
    if isinstance(started_monotonic, (int, float)) and isinstance(first_output_monotonic, (int, float)):
        finished_event["time_to_first_token_ms"] = max(0, int(round((first_output_monotonic - started_monotonic) * 1000)))

    if upstream is not None:
        for header_name in ("x-request-id", "request-id", "x-github-request-id"):
            header_value = upstream.headers.get(header_name)
            if header_value:
                finished_event["upstream_request_id"] = header_value
                break
        content_type = upstream.headers.get("content-type")
        if content_type:
            finished_event["response_content_type"] = content_type

    if isinstance(response_payload, dict):
        payload_response_id = response_payload.get("id")
        if isinstance(payload_response_id, str):
            finished_event["response_id"] = payload_response_id
        payload_model = response_payload.get("model")
        if isinstance(payload_model, str):
            finished_event["response_model"] = payload_model

    derived_usage = usage
    if isinstance(derived_usage, dict):
        derived_usage = _normalize_usage_payload(derived_usage)
    if derived_usage is None and isinstance(response_payload, dict):
        derived_usage = _extract_payload_usage(response_payload)
    if isinstance(derived_usage, dict):
        finished_event["usage"] = derived_usage

    model_name = (
        finished_event.get("response_model")
        or finished_event.get("resolved_model")
        or finished_event.get("requested_model")
    )
    finished_event["cost_usd"] = _usage_event_cost(model_name, derived_usage)
    finished_event["premium_requests"] = _premium_request_multiplier(model_name) if status_code < 400 else 0.0
    _remember_server_request_id(finished_event)
    _record_usage_event(finished_event)


def _snapshot_usage_events() -> list[dict]:
    with _usage_log_lock:
        return list(_recent_usage_events)


def _empty_ccusage_payload() -> dict:
    return {
        "available": True,
        "generated_by": "local-request-log",
        "loaded_at": None,
        "sources": {},
        "errors": [],
    }


def _new_usage_aggregate_bucket() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "cost_usd": 0.0,
        "_model_order": [],
        "_models": {},
        "_last_activity_dt": None,
        "project_path": None,
        "request_count": 0,
    }


def _ingest_usage_event(bucket: dict, event: dict):
    if not isinstance(bucket, dict) or not isinstance(event, dict):
        return

    usage = _normalize_usage_payload(event.get("usage")) or {}
    event_cost = event.get("cost_usd")
    if not isinstance(event_cost, (int, float)):
        event_cost = _usage_event_cost(_usage_event_model_name(event), usage)

    event_time = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
    if event_time is not None:
        current_last = bucket.get("_last_activity_dt")
        if not isinstance(current_last, datetime) or event_time > current_last:
            bucket["_last_activity_dt"] = event_time

    project_path = event.get("project_path")
    if bucket.get("project_path") is None and isinstance(project_path, str) and project_path:
        bucket["project_path"] = project_path

    input_tokens = _coerce_int(usage.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens"))
    cached_input_tokens = _coerce_int(usage.get("cached_input_tokens"))
    cache_creation_tokens = _coerce_int(usage.get("cache_creation_input_tokens"))
    reasoning_output_tokens = _coerce_int(usage.get("reasoning_output_tokens"))
    model_name = _usage_event_model_name(event) or "unknown"

    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens
    bucket["cached_input_tokens"] += cached_input_tokens
    bucket["cache_creation_tokens"] += cache_creation_tokens
    bucket["reasoning_output_tokens"] += reasoning_output_tokens
    bucket["cost_usd"] += _coerce_float(event_cost)
    bucket["request_count"] += 1

    model_bucket = bucket["_models"].setdefault(model_name, {"inputTokens": 0})
    model_bucket["inputTokens"] += input_tokens
    if model_name not in bucket["_model_order"]:
        bucket["_model_order"].append(model_name)


def _finalize_usage_bucket(bucket: dict, source: str, *, session_id: str | None = None, month: str | None = None) -> dict:
    if not isinstance(bucket, dict):
        return {}

    last_activity_dt = bucket.get("_last_activity_dt")
    last_activity = last_activity_dt.isoformat() if isinstance(last_activity_dt, datetime) else None
    models = bucket.get("_models") if isinstance(bucket.get("_models"), dict) else {}
    model_order = bucket.get("_model_order") if isinstance(bucket.get("_model_order"), list) else []
    result = {
        "sessionId": session_id,
        "lastActivity": last_activity,
        "projectPath": bucket.get("project_path"),
        "inputTokens": bucket.get("input_tokens", 0),
        "outputTokens": bucket.get("output_tokens", 0),
        "totalTokens": bucket.get("total_tokens", 0),
        "requestCount": bucket.get("request_count", 0),
    }

    if month is not None:
        result["month"] = month

    if source == "claude":
        result["cacheReadTokens"] = bucket.get("cached_input_tokens", 0)
        result["cacheCreationTokens"] = bucket.get("cache_creation_tokens", 0)
        result["totalCost"] = bucket.get("cost_usd", 0.0)
        result["modelsUsed"] = list(model_order)
    else:
        result["cachedInputTokens"] = bucket.get("cached_input_tokens", 0)
        result["reasoningOutputTokens"] = bucket.get("reasoning_output_tokens", 0)
        result["costUSD"] = bucket.get("cost_usd", 0.0)
        result["models"] = {name: value for name, value in models.items()}

    return result


def _collect_ccusage_payload() -> dict:
    usage_events = _snapshot_usage_events()
    source_month_buckets: dict[str, dict[str, dict]] = {}
    source_session_buckets: dict[str, dict[str, dict]] = {}

    for event in usage_events:
        event_time = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
        if event_time is None:
            continue

        source = _usage_event_source(event)
        month_key = _month_key(event_time)
        group_source, group_id = _usage_event_group_key(event)
        session_key = group_id
        if not isinstance(session_key, str) or not session_key:
            session_key = event.get("server_request_id") or event.get("request_id") or "unknown"
        if group_source != source:
            source = group_source

        source_month_bucket = source_month_buckets.setdefault(source, {})
        month_bucket = source_month_bucket.setdefault(month_key, _new_usage_aggregate_bucket())
        _ingest_usage_event(month_bucket, event)

        source_session_bucket = source_session_buckets.setdefault(source, {})
        session_bucket = source_session_bucket.setdefault(session_key, _new_usage_aggregate_bucket())
        _ingest_usage_event(session_bucket, event)

    result = {
        "available": True,
        "generated_by": "local-request-log",
        "loaded_at": _utc_now_iso(),
        "sources": {},
        "errors": [],
    }

    for source in sorted(set(source_month_buckets) | set(source_session_buckets)):
        month_buckets = source_month_buckets.get(source, {})
        session_buckets = source_session_buckets.get(source, {})
        monthly_rows = []
        for month_key, bucket in month_buckets.items():
            monthly_rows.append(_finalize_usage_bucket(bucket, source, month=month_key))
        monthly_rows.sort(key=lambda row: row.get("month") or "", reverse=True)

        session_rows = []
        for session_id, bucket in session_buckets.items():
            session_rows.append(_finalize_usage_bucket(bucket, source, session_id=session_id))
        session_rows.sort(key=lambda row: row.get("lastActivity") or "", reverse=True)

        source_payload = {
            "available": bool(monthly_rows or session_rows),
            "monthly": {"monthly": monthly_rows},
            "sessions": {"sessions": session_rows},
            "error": None,
        }
        result["sources"][source] = source_payload

    result["available"] = any(item.get("available") for item in result["sources"].values())
    return result


def _refresh_ccusage_cache_sync():
    with _ccusage_cache_lock:
        _ccusage_cache["refreshing"] = True
        _ccusage_cache["last_started_at"] = _utc_now_iso()

    try:
        result = _collect_ccusage_payload()
        _sqlite_cache_put("ccusage:v1", result)
        with _ccusage_cache_lock:
            _ccusage_cache["loaded_at"] = time.monotonic()
            _ccusage_cache["payload"] = result
            _ccusage_cache["last_error"] = None
    except Exception as exc:
        with _ccusage_cache_lock:
            _ccusage_cache["last_error"] = str(exc)
    finally:
        with _ccusage_cache_lock:
            _ccusage_cache["refreshing"] = False


def _trigger_ccusage_refresh(force: bool = False):
    with _ccusage_cache_lock:
        payload = _ccusage_cache.get("payload")
        loaded_at = _ccusage_cache.get("loaded_at", 0.0)
        refreshing = _ccusage_cache.get("refreshing", False)
        is_stale = payload is None or (time.monotonic() - loaded_at) >= CCUSAGE_CACHE_TTL_SECONDS
        should_refresh = force or is_stale
        if refreshing or not should_refresh:
            return
        _ccusage_cache["refreshing"] = True
        _ccusage_cache["last_started_at"] = _utc_now_iso()

    def _runner():
        try:
            result = _collect_ccusage_payload()
            _sqlite_cache_put("ccusage:v1", result)
            with _ccusage_cache_lock:
                _ccusage_cache["loaded_at"] = time.monotonic()
                _ccusage_cache["payload"] = result
                _ccusage_cache["last_error"] = None
        except Exception as exc:
            with _ccusage_cache_lock:
                _ccusage_cache["last_error"] = str(exc)
        finally:
            with _ccusage_cache_lock:
                _ccusage_cache["refreshing"] = False

    Thread(target=_runner, daemon=True).start()


def _get_ccusage_payload(force_refresh: bool = False) -> dict:
    _trigger_ccusage_refresh(force=force_refresh)

    with _ccusage_cache_lock:
        payload = _ccusage_cache.get("payload") or _empty_ccusage_payload()
        age_seconds = None
        loaded_at = _ccusage_cache.get("loaded_at", 0.0)
        if loaded_at:
            age_seconds = max(0.0, time.monotonic() - loaded_at)
        annotated = {
            **payload,
            "refreshing": bool(_ccusage_cache.get("refreshing")),
            "age_seconds": age_seconds,
            "last_error": _ccusage_cache.get("last_error"),
            "last_started_at": _ccusage_cache.get("last_started_at"),
        }
    return annotated


def _refresh_official_premium_cache_sync():
    with _premium_cache_lock:
        _premium_cache["refreshing"] = True
        _premium_cache["last_started_at"] = _utc_now_iso()

    try:
        result = _collect_official_premium_payload(skip_cache=True)
        _sqlite_cache_put("premium_usage:latest", result)
        with _premium_cache_lock:
            _premium_cache["loaded_at"] = time.monotonic()
            _premium_cache["payload"] = result
            _premium_cache["last_error"] = None
    except Exception as exc:
        with _premium_cache_lock:
            _premium_cache["last_error"] = str(exc)
    finally:
        with _premium_cache_lock:
            _premium_cache["refreshing"] = False


def _trigger_official_premium_refresh(force: bool = False):
    with _premium_cache_lock:
        payload = _premium_cache.get("payload")
        loaded_at = _premium_cache.get("loaded_at", 0.0)
        refreshing = _premium_cache.get("refreshing", False)
        is_stale = payload is None or (time.monotonic() - loaded_at) >= PREMIUM_CACHE_TTL_SECONDS
        should_refresh = force or is_stale
        if refreshing or not should_refresh:
            return
        _premium_cache["refreshing"] = True
        _premium_cache["last_started_at"] = _utc_now_iso()

    def _runner():
        try:
            result = _collect_official_premium_payload(skip_cache=force)
            _sqlite_cache_put("premium_usage:latest", result)
            with _premium_cache_lock:
                _premium_cache["loaded_at"] = time.monotonic()
                _premium_cache["payload"] = result
                _premium_cache["last_error"] = None
        except Exception as exc:
            with _premium_cache_lock:
                _premium_cache["last_error"] = str(exc)
        finally:
            with _premium_cache_lock:
                _premium_cache["refreshing"] = False

    Thread(target=_runner, daemon=True).start()


def _monotonic_loaded_at_from_payload(payload: dict | None) -> float:
    if not isinstance(payload, dict):
        return time.monotonic()

    loaded_at_value = payload.get("loaded_at")
    if isinstance(loaded_at_value, (int, float)):
        return _coerce_float(loaded_at_value, default=time.monotonic())

    if isinstance(loaded_at_value, str):
        parsed_at = _parse_iso_datetime(loaded_at_value)
        if parsed_at is not None:
            age_seconds = (_utc_now() - parsed_at).total_seconds()
            if age_seconds >= 0:
                return max(0.0, time.monotonic() - age_seconds)

    return time.monotonic()


def _seed_cached_payloads_from_sqlite():
    ccusage_payload = _sqlite_cache_get("ccusage:v1")
    if isinstance(ccusage_payload, dict):
        with _ccusage_cache_lock:
            _ccusage_cache["payload"] = ccusage_payload
            _ccusage_cache["loaded_at"] = _monotonic_loaded_at_from_payload(ccusage_payload)
            _ccusage_cache["last_error"] = ccusage_payload.get("error") or None
            _ccusage_cache["refreshing"] = False
            _ccusage_cache["last_started_at"] = None
            if ccusage_payload.get("generated_by") != "local-request-log":
                _ccusage_cache["loaded_at"] = 0.0

    premium_payload = _sqlite_cache_get_latest("premium_usage:")
    if isinstance(premium_payload, dict):
        with _premium_cache_lock:
            _premium_cache["payload"] = premium_payload
            _premium_cache["loaded_at"] = _monotonic_loaded_at_from_payload(premium_payload)
            _premium_cache["last_error"] = premium_payload.get("error")
            _premium_cache["refreshing"] = False
            _premium_cache["last_started_at"] = None


def _get_official_premium_payload(force_refresh: bool = False) -> dict:
    if force_refresh:
        _refresh_official_premium_cache_sync()
    else:
        _trigger_official_premium_refresh(force=False)
    with _premium_cache_lock:
        payload = _premium_cache.get("payload") or _empty_official_premium_payload()
        age_seconds = None
        loaded_at = _premium_cache.get("loaded_at", 0.0)
        if loaded_at:
            age_seconds = max(0.0, time.monotonic() - loaded_at)
        annotated = {
            **payload,
            "refreshing": bool(_premium_cache.get("refreshing")),
            "age_seconds": age_seconds,
            "last_error": _premium_cache.get("last_error"),
            "last_started_at": _premium_cache.get("last_started_at"),
        }
    return annotated


def _month_key(value: datetime) -> str:
    return value.strftime("%Y-%m")


def _month_key_for_source_row(source: str, row: dict) -> str | None:
    raw_value = row.get("month")
    if not isinstance(raw_value, str):
        return None
    for fmt in ("%Y-%m", "%b %Y"):
        try:
            return datetime.strptime(raw_value, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return raw_value if source == "claude" else None


def _normalize_session(source: str, session: dict) -> dict:
    if source == "claude":
        models = session.get("modelsUsed") or list((session.get("models") or {}).keys())
        cost_usd = _coerce_float(session.get("totalCost"))
        if cost_usd == 0.0 and session.get("costUSD") is not None:
            cost_usd = _coerce_float(session.get("costUSD"))
        cached_tokens = _coerce_int(session.get("cacheReadTokens"))
        if cached_tokens == 0 and session.get("cachedInputTokens") is not None:
            cached_tokens = _coerce_int(session.get("cachedInputTokens"))
        cache_creation_tokens = _coerce_int(session.get("cacheCreationTokens"))
        if cache_creation_tokens == 0 and session.get("cacheCreationInputTokens") is not None:
            cache_creation_tokens = _coerce_int(session.get("cacheCreationInputTokens"))
        reasoning_tokens = 0
    else:
        models = list((session.get("models") or {}).keys())
        cost_usd = _coerce_float(session.get("costUSD"))
        cached_tokens = _coerce_int(session.get("cachedInputTokens"))
        cache_creation_tokens = 0
        reasoning_tokens = _coerce_int(session.get("reasoningOutputTokens"))

    return {
        "source": source,
        "session_id": session.get("sessionId"),
        "last_activity": session.get("lastActivity"),
        "project_path": session.get("projectPath"),
        "input_tokens": _coerce_int(session.get("inputTokens")),
        "output_tokens": _coerce_int(session.get("outputTokens")),
        "total_tokens": _coerce_int(session.get("totalTokens")),
        "cached_input_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "cost_usd": cost_usd,
        "models": models,
    }


def _normalize_month_row(source: str, row: dict) -> dict:
    if source == "claude":
        models = row.get("modelsUsed") or []
        cost_usd = _coerce_float(row.get("totalCost"))
        cached_tokens = _coerce_int(row.get("cacheReadTokens"))
        cache_creation_tokens = _coerce_int(row.get("cacheCreationTokens"))
        reasoning_tokens = 0
    else:
        models = list((row.get("models") or {}).keys())
        cost_usd = _coerce_float(row.get("costUSD"))
        cached_tokens = _coerce_int(row.get("cachedInputTokens"))
        cache_creation_tokens = 0
        reasoning_tokens = _coerce_int(row.get("reasoningOutputTokens"))

    return {
        "source": source,
        "month_key": _month_key_for_source_row(source, row),
        "month_label": row.get("month"),
        "input_tokens": _coerce_int(row.get("inputTokens")),
        "output_tokens": _coerce_int(row.get("outputTokens")),
        "total_tokens": _coerce_int(row.get("totalTokens")),
        "cached_input_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "cost_usd": cost_usd,
        "models": models,
    }


def _combine_month_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        month_key = row.get("month_key")
        if not month_key:
            continue
        current = grouped.setdefault(
            month_key,
            {
                "month_key": month_key,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_tokens": 0,
                "reasoning_output_tokens": 0,
                "cost_usd": 0.0,
                "sources": {},
            },
        )
        current["input_tokens"] += row.get("input_tokens", 0)
        current["output_tokens"] += row.get("output_tokens", 0)
        current["total_tokens"] += row.get("total_tokens", 0)
        current["cached_input_tokens"] += row.get("cached_input_tokens", 0)
        current["cache_creation_tokens"] += row.get("cache_creation_tokens", 0)
        current["reasoning_output_tokens"] += row.get("reasoning_output_tokens", 0)
        current["cost_usd"] += row.get("cost_usd", 0.0)
        current["sources"][row["source"]] = row

    return [grouped[key] | {"cost_usd": round(grouped[key]["cost_usd"], 4)} for key in sorted(grouped.keys(), reverse=True)]


def _build_dashboard_payload(force_refresh: bool = False) -> dict:
    now = _utc_now()
    month_start, month_end = _current_billing_month_bounds(now)
    current_month_key = _month_key(now)
    usage_events = _snapshot_usage_events()
    current_month_events = []
    for event in usage_events:
        recorded_at = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
        if recorded_at is None:
            continue
        if month_start <= recorded_at < month_end:
            current_month_events.append(event)

    premium_used = round(sum(_coerce_float(event.get("premium_requests")) for event in current_month_events), 2)
    quota_summary = _extract_quota_summary(_load_api_key_payload())
    included = quota_summary.get("included")
    has_tracked_premium_data = bool(current_month_events)
    tracked_remaining = None
    if included is not None and has_tracked_premium_data:
        tracked_remaining = round(max(included - premium_used, 0.0), 2)
    official_premium = _get_official_premium_payload(force_refresh=force_refresh)
    official_included = official_premium.get("included")
    official_used = official_premium.get("used")
    official_remaining = official_premium.get("remaining")
    official_reset_date = official_premium.get("reset_date")
    official_available = bool(official_premium.get("available"))

    ccusage_payload = _get_ccusage_payload(force_refresh=force_refresh)
    normalized_months = []
    normalized_sessions = []
    per_source_month = {}
    for source, source_payload in (ccusage_payload.get("sources") or {}).items():
        if not isinstance(source_payload, dict) or not source_payload.get("available"):
            continue

        monthly_payload = source_payload.get("monthly")
        monthly_rows = monthly_payload.get("monthly") if isinstance(monthly_payload, dict) else []
        if isinstance(monthly_rows, list):
            for row in monthly_rows:
                if isinstance(row, dict):
                    normalized = _normalize_month_row(source, row)
                    normalized_months.append(normalized)
                    if normalized.get("month_key") == current_month_key:
                        per_source_month[source] = normalized

        sessions_payload = source_payload.get("sessions")
        session_rows = sessions_payload.get("sessions") if isinstance(sessions_payload, dict) else []
        if isinstance(session_rows, list):
            for session in session_rows:
                if isinstance(session, dict):
                    normalized_sessions.append(_normalize_session(source, session))

    normalized_sessions.sort(key=lambda item: item.get("last_activity") or "", reverse=True)
    month_history = _combine_month_rows(normalized_months)
    current_month_usage = {
        "month_key": current_month_key,
        "input_tokens": sum(item.get("input_tokens", 0) for item in per_source_month.values()),
        "output_tokens": sum(item.get("output_tokens", 0) for item in per_source_month.values()),
        "total_tokens": sum(item.get("total_tokens", 0) for item in per_source_month.values()),
        "cached_input_tokens": sum(item.get("cached_input_tokens", 0) for item in per_source_month.values()),
        "cache_creation_tokens": sum(item.get("cache_creation_tokens", 0) for item in per_source_month.values()),
        "reasoning_output_tokens": sum(item.get("reasoning_output_tokens", 0) for item in per_source_month.values()),
        "cost_usd": round(sum(item.get("cost_usd", 0.0) for item in per_source_month.values()), 4),
        "sources": per_source_month,
    }

    recent_requests = sorted(
        current_month_events,
        key=lambda item: item.get("finished_at") or item.get("started_at") or "",
        reverse=True,
    )[:100]

    return {
        "generated_at": now.isoformat(),
        "premium": {
            "sku": quota_summary.get("sku") or official_premium.get("raw", {}).get("sku"),
            "included": official_included if official_included is not None else included,
            "used": official_used if official_available and official_used is not None else premium_used,
            "tracked_remaining": tracked_remaining,
            "has_tracked_data": has_tracked_premium_data,
            "official_remaining": official_remaining if official_available else quota_summary.get("official_remaining"),
            "reset_date": official_reset_date if official_reset_date is not None else quota_summary.get("reset_date"),
            "source": official_premium.get("source") if official_available else "proxy-request-log",
            "official": official_premium,
        },
        "current_month": {
            "label": current_month_key,
            "start_at": month_start.isoformat(),
            "end_at": month_end.isoformat(),
            "proxy_requests": len(current_month_events),
            "sessions": len(normalized_sessions),
            "ccusage": current_month_usage,
        },
        "recent_sessions": normalized_sessions[:20],
        "recent_requests": recent_requests,
        "month_history": month_history[:12],
        "ccusage": ccusage_payload,
    }


_load_usage_history()
_initiator_policy.seed_from_usage_events(_snapshot_usage_events())
_seed_cached_payloads_from_sqlite()
_trigger_ccusage_refresh()


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def _gh_headers(access_token: str = None) -> dict:
    h = {
        "accept": "application/json",
        "editor-version": "vscode/1.85.1",
        "editor-plugin-version": "copilot/1.155.0",
        "user-agent": "GithubCopilot/1.155.0",
        "accept-encoding": "gzip,deflate,br",
        "content-type": "application/json",
    }
    if access_token:
        h["authorization"] = f"token {access_token}"
    return h


async def parse_json_request(request: Request) -> dict:
    raw_body = await request.body()
    try:
        if not raw_body:
            return {}
        content_encoding = str(request.headers.get("content-encoding", "")).strip().lower()

        if content_encoding == "gzip":
            raw_body = gzip.decompress(raw_body)
        elif content_encoding == "deflate":
            raw_body = zlib.decompress(raw_body)
        elif content_encoding == "zstd":
            raw_body = pyzstd.decompress(raw_body)
        elif content_encoding == "br":
            if brotli is None:
                raise HTTPException(status_code=400, detail="Invalid JSON body: unsupported brotli request encoding")
            raw_body = brotli.decompress(raw_body)
        elif raw_body.startswith(b"\x1f\x8b"):
            raw_body = gzip.decompress(raw_body)
        elif raw_body.startswith(b"\x28\xb5\x2f\xfd"):
            raw_body = pyzstd.decompress(raw_body)

        return json.loads(raw_body)
    except HTTPException:
        raise
    except Exception:
        path = getattr(getattr(request, "url", None), "path", "?")
        content_type = str(request.headers.get("content-type", "")).strip()
        content_encoding = str(request.headers.get("content-encoding", "")).strip().lower()
        preview_hex = raw_body[:24].hex()
        preview_text = raw_body[:160].decode("utf-8", errors="replace")
        _record_request_error(
            {
                "at": _utc_now_iso(),
                "path": path,
                "content_type": content_type,
                "content_encoding": content_encoding,
                "body_len": len(raw_body),
                "preview_hex": preview_hex,
                "preview_text": preview_text,
            }
        )
        print(
            f"WARN: Invalid JSON body path={path} content_type={content_type!r} "
            f"content_encoding={content_encoding!r} body_len={len(raw_body)} preview_hex={preview_hex}",
            flush=True,
        )
        raise HTTPException(status_code=400, detail="Invalid JSON body")


def _load_access_token() -> str | None:
    try:
        tok = open(ACCESS_TOKEN_FILE).read().strip()
        return tok or None
    except OSError:
        return None


def _load_billing_token() -> str | None:
    env_token = os.environ.get("GHCP_GITHUB_BILLING_TOKEN", "").strip()
    if env_token:
        return env_token

    try:
        tok = open(BILLING_TOKEN_FILE, encoding="utf-8").read().strip()
        return tok or None
    except OSError:
        return None


def _save_billing_token(token: str):
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(BILLING_TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(token.strip())


def _clear_billing_token():
    try:
        os.remove(BILLING_TOKEN_FILE)
    except OSError:
        pass


def _billing_token_status() -> dict[str, bool | str]:
    env_token = os.environ.get("GHCP_GITHUB_BILLING_TOKEN", "").strip()
    if env_token:
        return {"configured": True, "source": "environment", "readonly": True}

    try:
        tok = open(BILLING_TOKEN_FILE, encoding="utf-8").read().strip()
    except OSError:
        tok = ""
    return {"configured": bool(tok), "source": "file" if tok else "none", "readonly": False}


def _backup_config_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None

    os.makedirs(os.path.dirname(path), exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.ghcp-proxy.bak.{timestamp}"
    attempt = 1
    while os.path.exists(backup_path):
        attempt += 1
        backup_path = f"{path}.ghcp-proxy.bak.{timestamp}.{attempt}"

    shutil.copy2(path, backup_path)
    return backup_path


def _latest_backup_path(path: str) -> str | None:
    backups = [entry for entry in glob.glob(f"{path}.ghcp-proxy.bak.*") if os.path.isfile(entry)]
    if not backups:
        return None
    return max(backups, key=lambda entry: os.path.getmtime(entry))


def _parse_toml_values(content: str) -> dict:
    current_section: str | None = None
    data: dict[str, object] = {}
    sections: dict[str, dict[str, str]] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            sections.setdefault(current_section, {})
            continue

        if "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        value = value.split("#", 1)[0].strip()
        if not value:
            continue

        if (value[0] in {"'", "\""} and value[-1] == value[0]):
            value = value[1:-1]

        if current_section is None:
            data[key] = value
        else:
            section = sections.setdefault(current_section, {})
            section[key] = value
            data[current_section] = section

    return data


def _codex_proxy_status() -> dict[str, bool | str | None]:
    status = {
        "client": "codex",
        "configured": False,
        "exists": False,
        "path": CODEX_CONFIG_FILE,
        "backup_path": None,
        "error": "",
        "status_message": "config file not found",
    }

    if not os.path.exists(CODEX_CONFIG_FILE):
        return status

    status["exists"] = True
    status["status_message"] = "exists but not configured for proxy"

    try:
        with open(CODEX_CONFIG_FILE, encoding="utf-8") as f:
            parsed = _parse_toml_values(f.read())
    except OSError as exc:
        status["error"] = f"failed to read {CODEX_CONFIG_FILE}: {exc}"
        return status
    except Exception as exc:
        status["error"] = f"failed to parse {CODEX_CONFIG_FILE}: {exc}"
        return status

    model_providers = parsed.get("model_providers.custom")
    provider_cfg = model_providers if isinstance(model_providers, dict) else {}
    active = (
        parsed.get("model_provider") == "custom"
        and isinstance(provider_cfg, dict)
        and provider_cfg.get("name") == "OpenAI"
        and provider_cfg.get("base_url") == CODEX_PROXY_BASE_URL
        and provider_cfg.get("wire_api") == "responses"
    )
    status["configured"] = bool(active)
    if active:
        status["status_message"] = "proxy configured"
    return status


def _empty_proxy_status(client: str, path: str) -> dict[str, bool | str | None]:
    return {
        "client": client,
        "configured": False,
        "exists": False,
        "path": path,
        "backup_path": None,
        "error": "",
        "status_message": "unknown",
    }


def _claude_proxy_status() -> dict[str, bool | str | None]:
    status = {
        "client": "claude",
        "configured": False,
        "exists": False,
        "path": CLAUDE_SETTINGS_FILE,
        "backup_path": None,
        "error": "",
        "status_message": "settings file not found",
    }

    if not os.path.exists(CLAUDE_SETTINGS_FILE):
        return status

    status["exists"] = True
    status["status_message"] = "exists but not configured for proxy"

    try:
        with open(CLAUDE_SETTINGS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        status["error"] = f"failed to read {CLAUDE_SETTINGS_FILE}: {exc}"
        return status
    except json.JSONDecodeError as exc:
        status["error"] = f"failed to parse {CLAUDE_SETTINGS_FILE}: {exc}"
        return status
    except Exception as exc:
        status["error"] = f"failed to parse {CLAUDE_SETTINGS_FILE}: {exc}"
        return status

    env = payload.get("env") if isinstance(payload, dict) else None
    active = (
        isinstance(env, dict)
        and env.get("ANTHROPIC_BASE_URL") == DASHBOARD_BASE_URL
        and env.get("CLAUDE_CODE_DISABLE_1M_CONTEXT") == "1"
        and isinstance(env.get("ANTHROPIC_AUTH_TOKEN"), str)
        and env.get("ANTHROPIC_AUTH_TOKEN") != ""
    )
    status["configured"] = bool(active)
    if active:
        status["status_message"] = "proxy configured"
    return status


def _write_codex_proxy_config() -> dict[str, bool | str | None]:
    status = _codex_proxy_status()
    if status.get("error"):
        return status
    if status.get("configured"):
        status["backup_path"] = _latest_backup_path(CODEX_CONFIG_FILE)
        status["status_message"] = "proxy already enabled"
        return status

    backup_path = _backup_config_file(CODEX_CONFIG_FILE)
    os.makedirs(CODEX_CONFIG_DIR, exist_ok=True)
    with open(CODEX_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(CODEX_PROXY_CONFIG)
        f.write("\n")
    status = _codex_proxy_status()
    status["backup_path"] = backup_path
    status["status_message"] = "installed proxy config"
    return status


def _write_claude_proxy_settings() -> dict[str, bool | str | None]:
    status = _claude_proxy_status()
    if status.get("error"):
        return status
    if status.get("configured"):
        status["backup_path"] = _latest_backup_path(CLAUDE_SETTINGS_FILE)
        status["status_message"] = "proxy already enabled"
        return status

    backup_path = _backup_config_file(CLAUDE_SETTINGS_FILE)
    os.makedirs(CLAUDE_CONFIG_DIR, exist_ok=True)
    with open(CLAUDE_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(CLAUDE_PROXY_SETTINGS, f, indent=2)
        f.write("\n")
    status = _claude_proxy_status()
    status["backup_path"] = backup_path
    status["status_message"] = "installed proxy settings"
    return status


def _disable_client_proxy_config(config_path: str, status_fn) -> dict[str, bool | str | None]:
    status = status_fn()
    if not isinstance(status, dict):
        status = _empty_proxy_status("unknown", config_path)
    if status.get("error"):
        return status
    if not isinstance(config_path, str) or not config_path:
        status["error"] = "invalid config path"
        return status
    if not status.get("configured"):
        status["backup_path"] = _latest_backup_path(config_path)
        status["restored_from_backup"] = False
        status["status_message"] = "proxy already disabled"
        return status

    backup_path = _latest_backup_path(config_path)
    restored_from_backup = False
    operation_message = ""

    try:
        if backup_path:
            shutil.move(backup_path, config_path)
            restored_from_backup = True
            operation_message = f"restored config from backup ({backup_path})"
        else:
            if os.path.exists(config_path):
                os.remove(config_path)
                operation_message = "removed proxy config"
            else:
                operation_message = "config file already absent"
    except Exception as exc:
        status["error"] = f"failed to disable proxy config: {exc}"
        return status

    status = status_fn()
    status["backup_path"] = backup_path
    status["restored_from_backup"] = restored_from_backup
    if operation_message:
        status["status_message"] = operation_message
    if status.get("error"):
        return status
    return status


def _disable_codex_proxy_config() -> dict[str, bool | str | None]:
    return _disable_client_proxy_config(CODEX_CONFIG_FILE, _codex_proxy_status)


def _disable_claude_proxy_settings() -> dict[str, bool | str | None]:
    return _disable_client_proxy_config(CLAUDE_SETTINGS_FILE, _claude_proxy_status)


def _proxy_client_status_payload() -> dict[str, object]:
    codex_status = _codex_proxy_status()
    claude_status = _claude_proxy_status()
    codex_status["backup_path"] = _latest_backup_path(CODEX_CONFIG_FILE)
    claude_status["backup_path"] = _latest_backup_path(CLAUDE_SETTINGS_FILE)
    codex_status["restored_from_backup"] = False
    claude_status["restored_from_backup"] = False
    return {"clients": {"codex": codex_status, "claude": claude_status}}


def _normalize_proxy_targets(payload: dict) -> list[str]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    raw_targets = payload.get("targets")
    if raw_targets is None:
        raw_targets = payload.get("target")
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    elif isinstance(raw_targets, (list, tuple, set)):
        raw_targets = list(raw_targets)
    else:
        raise HTTPException(status_code=400, detail='Request body must include "targets" or "target".')

    selected = set()
    for raw in raw_targets:
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail="Each target must be a string.")
        target = raw.strip().lower()
        if target in {"both", "all"}:
            selected.update({"codex", "claude"})
        elif target in {"codex", "claude"}:
            selected.add(target)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported target: {raw}")

    if not selected:
        raise HTTPException(status_code=400, detail="No valid targets provided.")

    return sorted(selected)


def _save_access_token(token: str):
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(ACCESS_TOKEN_FILE, "w") as f:
        f.write(token)


def _load_api_key() -> str | None:
    try:
        data = json.load(open(API_KEY_FILE))
        if data["expires_at"] > datetime.now().timestamp():
            return data["token"]
    except Exception:
        pass
    return None


def _get_api_base() -> str:
    """Use the endpoint embedded in api-key.json if present, else default."""
    try:
        data = json.load(open(API_KEY_FILE))
        return data.get("endpoints", {}).get("api") or GITHUB_COPILOT_API_BASE
    except Exception:
        return GITHUB_COPILOT_API_BASE


def _device_flow() -> str:
    """
    Interactive GitHub OAuth device flow.
    Prints the verification URL and code to the terminal, then polls until
    the user authorizes (or times out after ~60 seconds).
    Returns the GitHub OAuth access token.
    """
    with httpx.Client() as c:
        r = c.post(
            GITHUB_DEVICE_CODE_URL,
            headers=_gh_headers(),
            json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user read:org"},
        )
        r.raise_for_status()
        info = r.json()

    print("", flush=True)
    print("─" * 60, flush=True)
    print("  GitHub Copilot — Authorization Required", flush=True)
    print("─" * 60, flush=True)
    print(f"  1. Open:  {info['verification_uri']}", flush=True)
    print(f"  2. Enter: {info['user_code']}", flush=True)
    print("─" * 60, flush=True)
    print("  Waiting for authorization...", flush=True)

    interval = info.get("interval", 5)
    max_attempts = max(12, info.get("expires_in", 60) // interval)

    with httpx.Client() as c:
        for attempt in range(max_attempts):
            time.sleep(interval)
            r = c.post(
                GITHUB_ACCESS_TOKEN_URL,
                headers=_gh_headers(),
                json={
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": info["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            d = r.json()

            if "access_token" in d:
                print("  Authorized successfully.", flush=True)
                print("-" * 60, flush=True)
                print("", flush=True)
                _save_access_token(d["access_token"])
                return d["access_token"]

            error = d.get("error", "")
            if error == "authorization_pending":
                dots = "." * ((attempt % 3) + 1)
                print(f"  Waiting{dots}", end="\r", flush=True)
                continue
            elif error == "slow_down":
                interval += 5
                continue
            elif error in ("expired_token", "access_denied"):
                print(f"\n  Authorization failed: {error}", flush=True)
                break
            else:
                print(f"\n  Unexpected response: {d}", flush=True)
                break

    raise RuntimeError("Device flow failed — could not obtain access token.")


def _refresh_api_key(access_token: str) -> str:
    """Exchange OAuth access token for a short-lived GHCP API key (~30 min TTL)."""
    with httpx.Client() as c:
        r = c.get(GITHUB_API_KEY_URL, headers=_gh_headers(access_token))
        r.raise_for_status()
        data = r.json()
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(API_KEY_FILE, "w") as f:
        json.dump(data, f)
    return data["token"]


def get_api_key() -> str:
    """Returns a valid GHCP API key, refreshing transparently when expired."""
    key = _load_api_key()
    if key:
        return key
    access_token = _load_access_token() or _device_flow()
    return _refresh_api_key(access_token)


def ensure_authenticated():
    """
    Called at startup — before the server accepts any requests.
    Runs the full auth flow interactively in the terminal if needed.
    """
    print("Checking GitHub Copilot authentication...", flush=True)
    try:
        key = get_api_key()
        print("Authenticated. GHCP API key valid.", flush=True)
        return key
    except Exception as e:
        print(f"\nAuthentication failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


_trigger_official_premium_refresh()


# ─── Header builder ───────────────────────────────────────────────────────────

def build_copilot_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "User-Agent": f"opencode/{OPENCODE_VERSION}",
        "Openai-Intent": "conversation-edits",
        "Editor-Version": OPENCODE_HEADER_VERSION,
        "Editor-Plugin-Version": OPENCODE_HEADER_VERSION,
        "Copilot-Integration-Id": OPENCODE_INTEGRATION_ID,
    }


# ─── Responses API helpers ────────────────────────────────────────────────────


def _apply_forwarded_request_headers(headers: dict, request: Request):
    for header_name in FORWARDED_REQUEST_HEADERS:
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value

    forwarded_server_request_id = None
    for header_name in FORWARDED_SERVER_REQUEST_ID_HEADERS:
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value
            if forwarded_server_request_id is None:
                forwarded_server_request_id = header_value

    if forwarded_server_request_id is not None:
        headers.setdefault("x-request-id", forwarded_server_request_id)
        headers.setdefault("x-github-request-id", forwarded_server_request_id)


def _initiator_log_label(initiator: str | None) -> str:
    return "Agent" if initiator == "agent" else "User"


def log_proxy_request(
    request: Request,
    requested_model: str | None,
    resolved_model: str | None,
    initiator: str | None,
):
    parts = [
        "INFO:",
        f"Proxy request ({_initiator_log_label(initiator)}):",
        f"{request.method} {request.url.path}",
    ]
    if requested_model is not None:
        parts.append(f"requested_model={requested_model}")
    if resolved_model is not None and resolved_model != requested_model:
        parts.append(f"resolved_model={resolved_model}")
    print(" ".join(parts), flush=True)


def _prune_upstream_request_timestamps(now: float):
    cutoff = now - UPSTREAM_REQUEST_WINDOW_SECONDS
    while _upstream_request_timestamps and _upstream_request_timestamps[0] <= cutoff:
        _upstream_request_timestamps.popleft()


async def throttle_upstream_request(now: float | None = None):
    while True:
        async with _upstream_rate_limit_lock:
            current_time = time.monotonic() if now is None else now
            _prune_upstream_request_timestamps(current_time)

            if len(_upstream_request_timestamps) < UPSTREAM_REQUESTS_PER_WINDOW:
                _upstream_request_timestamps.append(current_time)
                return

            oldest = _upstream_request_timestamps[0]
            delay = max(0.0, (oldest + UPSTREAM_REQUEST_WINDOW_SECONDS) - current_time)

        await asyncio.sleep(delay)


async def throttled_client_post(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    await throttle_upstream_request()
    return await client.post(url, **kwargs)


async def throttled_client_send(client: httpx.AsyncClient, request: httpx.Request, **kwargs) -> httpx.Response:
    await throttle_upstream_request()
    return await client.send(request, **kwargs)

def _extract_item_text(item) -> str:
    if not isinstance(item, dict):
        return ""

    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("text"), str):
                parts.append(entry["text"])
            elif isinstance(entry.get("input_text"), str):
                parts.append(entry["input_text"])
        return "".join(parts)

    if isinstance(item.get("text"), str):
        return item["text"]
    if isinstance(item.get("input_text"), str):
        return item["input_text"]
    return ""
def has_vision_input(value, depth=0, max_depth=10) -> bool:
    """Recursively find type='input_image' anywhere in the input tree."""
    if depth > max_depth or value is None:
        return False
    if isinstance(value, list):
        return any(has_vision_input(i, depth + 1, max_depth) for i in value)
    if not isinstance(value, dict):
        return False
    if str(value.get("type", "")).lower() == "input_image":
        return True
    content = value.get("content")
    if isinstance(content, list):
        return any(has_vision_input(i, depth + 1, max_depth) for i in content)
    return False


def model_requires_anthropic_beta(model_name) -> bool:
    if not isinstance(model_name, str):
        return False
    normalized = model_name.strip().lower()
    return "claude" in normalized or normalized.startswith("anthropic")


def normalize_upstream_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return model_name

    normalized = model_name.strip().lower()
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def resolve_copilot_model_name(model_name: str | None) -> str | None:
    normalized = normalize_upstream_model_name(model_name)
    if not isinstance(normalized, str):
        return model_name

    if normalized in ("claude-opus-4.6", "claude-sonnet-4.6", "claude-haiku-4.5"):
        return normalized

    if "opus" in normalized:
        return "claude-opus-4.6"
    if "sonnet" in normalized:
        return "claude-sonnet-4.6"
    if "haiku" in normalized:
        return "claude-haiku-4.5"
    return normalized


def _extract_text_content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _normalize_anthropic_cache_control(value):
    if not isinstance(value, dict):
        return None

    cache_type = value.get("type")
    if isinstance(cache_type, str) and cache_type:
        return {"type": cache_type}

    if "ephemeral" in value:
        return {"type": "ephemeral"}

    return dict(value)


def _attach_copilot_cache_control(target: dict, source: dict) -> dict:
    cache_control = _normalize_anthropic_cache_control(source.get("cache_control"))
    if cache_control is not None:
        target["copilot_cache_control"] = cache_control
    return target


def _anthropic_image_block_to_chat(item: dict) -> dict:
    source = item.get("source")
    if not isinstance(source, dict):
        raise ValueError("Anthropic image block is missing a valid source object")

    source_type = str(source.get("type", "")).lower()
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise ValueError("Anthropic base64 image source must include media_type and data strings")
        return _attach_copilot_cache_control(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
            item,
        )
    if source_type == "url":
        image_url = source.get("url")
        if not isinstance(image_url, str):
            raise ValueError("Anthropic URL image source must include a url string")
        return _attach_copilot_cache_control({"type": "image_url", "image_url": {"url": image_url}}, item)

    raise ValueError(f"Unsupported Anthropic image source type: {source_type}")


def _normalize_anthropic_content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    raise ValueError("Anthropic message content must be a string or content block list")


def _anthropic_text_or_image_block_to_chat(item: dict) -> dict | None:
    item_type = str(item.get("type", "")).lower()
    if item_type == "text":
        text = item.get("text")
        if isinstance(text, str):
            return _attach_copilot_cache_control({"type": "text", "text": text}, item)
        return None
    if item_type == "image":
        return _anthropic_image_block_to_chat(item)
    return None


def _anthropic_system_to_chat_content(system):
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        return ""

    converted = []
    has_copilot_cache_control = False
    for item in system:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() != "text":
            raise ValueError("Anthropic system content currently supports text blocks only")
        text = item.get("text")
        if not isinstance(text, str):
            continue
        part = _attach_copilot_cache_control({"type": "text", "text": text}, item)
        if "copilot_cache_control" in part:
            has_copilot_cache_control = True
        converted.append(part)

    if not converted:
        return ""
    if not has_copilot_cache_control:
        return "".join(part.get("text", "") for part in converted)
    return converted


def _anthropic_blocks_to_chat_content(blocks: list[dict]):
    converted = []
    for item in blocks:
        content_item = _anthropic_text_or_image_block_to_chat(item)
        if content_item is not None:
            converted.append(content_item)
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type in {"tool_use", "tool_result"}:
            raise ValueError(f"Anthropic block type {item_type} cannot be converted into chat message content directly")
        raise ValueError(f"Unsupported Anthropic content block type: {item_type}")

    if not converted:
        return ""
    if (
        len(converted) == 1
        and converted[0].get("type") == "text"
        and set(converted[0].keys()).issubset({"type", "text"})
    ):
        return converted[0].get("text", "")
    return converted


def _anthropic_tool_result_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            raise ValueError("Anthropic tool_result content currently supports text blocks only")
        return "".join(parts)
    raise ValueError("Anthropic tool_result content must be a string or list of text blocks")


def _anthropic_tool_use_to_chat_tool_call(item: dict) -> dict:
    tool_name = item.get("name")
    tool_id = item.get("id")
    tool_input = item.get("input")
    if not isinstance(tool_name, str) or not isinstance(tool_id, str):
        raise ValueError("Anthropic tool_use blocks must include string id and name")
    if tool_input is None:
        tool_input = {}
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(tool_input, separators=(",", ":")),
        },
    }


def anthropic_message_to_chat_messages(message: dict) -> list[dict]:
    role = str(message.get("role", "")).lower()
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported Anthropic role: {role}")

    blocks = _normalize_anthropic_content_blocks(message.get("content"))

    if role == "assistant":
        content_blocks = []
        tool_calls = []
        for item in blocks:
            item_type = str(item.get("type", "")).lower()
            if item_type == "tool_use":
                tool_calls.append(_anthropic_tool_use_to_chat_tool_call(item))
                continue
            content_item = _anthropic_text_or_image_block_to_chat(item)
            if content_item is None:
                raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
            content_blocks.append(item)

        assistant_message = {"role": "assistant"}
        assistant_message["content"] = _anthropic_blocks_to_chat_content(content_blocks)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return [assistant_message]

    chat_messages = []
    buffered_user_blocks = []
    for item in blocks:
        item_type = str(item.get("type", "")).lower()
        if item_type == "tool_result":
            if buffered_user_blocks:
                chat_messages.append(
                    {
                        "role": "user",
                        "content": _anthropic_blocks_to_chat_content(buffered_user_blocks),
                    }
                )
                buffered_user_blocks = []
            tool_use_id = item.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                raise ValueError("Anthropic tool_result blocks must include tool_use_id")
            tool_text = _anthropic_tool_result_content_to_text(item.get("content", ""))
            if item.get("is_error") is True:
                tool_text = f"[tool_error]\n{tool_text}"
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": tool_text,
            }
            cache_control = _normalize_anthropic_cache_control(item.get("cache_control"))
            if cache_control is not None:
                tool_message["copilot_cache_control"] = cache_control
            chat_messages.append(tool_message)
            continue
        content_item = _anthropic_text_or_image_block_to_chat(item)
        if content_item is None:
            raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
        buffered_user_blocks.append(item)

    if buffered_user_blocks:
        chat_messages.append(
            {
                "role": "user",
                "content": _anthropic_blocks_to_chat_content(buffered_user_blocks),
            }
        )

    return chat_messages


def anthropic_tools_to_chat(tools) -> list[dict]:
    if not isinstance(tools, list):
        raise ValueError("Anthropic tools must be a list")

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            raise ValueError("Anthropic tools must include a string name")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", "") if isinstance(tool.get("description"), str) else "",
                    "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def anthropic_tool_choice_to_chat(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            name = tool_choice.get("name")
            if not isinstance(name, str):
                raise ValueError("Anthropic tool_choice type=tool must include name")
            return {"type": "function", "function": {"name": name}}
        if choice_type == "none":
            return "none"
    if isinstance(tool_choice, str):
        normalized = tool_choice.lower()
        if normalized in {"auto", "none"}:
            return normalized
        if normalized == "any":
            return "required"
    raise ValueError("Unsupported Anthropic tool_choice value")


async def anthropic_request_to_chat(body: dict, api_base: str, api_key: str) -> dict:
    source_messages = body.get("messages")
    if not isinstance(source_messages, list):
        raise ValueError("Anthropic request must include a messages array")

    chat_messages = []
    system_content = _anthropic_system_to_chat_content(body.get("system"))
    if system_content:
        chat_messages.append({"role": "system", "content": system_content})

    for message in source_messages:
        if not isinstance(message, dict):
            continue
        chat_messages.extend(anthropic_message_to_chat_messages(message))

    payload = {
        "model": resolve_copilot_model_name(body.get("model")),
        "messages": chat_messages,
        "stream": bool(body.get("stream", False)),
    }

    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}

    for source_key, target_key in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ):
        value = body.get(source_key)
        if value is not None:
            payload[target_key] = value

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        budget_tokens = thinking.get("budget_tokens")
        if isinstance(budget_tokens, int):
            payload["thinking_budget"] = budget_tokens

    if body.get("tools") is not None:
        payload["tools"] = anthropic_tools_to_chat(body.get("tools"))

    mapped_tool_choice = anthropic_tool_choice_to_chat(body.get("tool_choice"))
    if mapped_tool_choice is not None:
        payload["tool_choice"] = mapped_tool_choice

    return payload


def _chat_usage_to_anthropic(usage) -> dict:
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cache_read_input_tokens = prompt_details.get("cached_tokens", 0) or 0

    return {
        "input_tokens": max(0, prompt_tokens - cache_read_input_tokens),
        "output_tokens": completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read_input_tokens,
    }


def _chat_stop_reason_to_anthropic(value) -> str | None:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
        "tool_calls": "tool_use",
    }
    if not isinstance(value, str):
        return None
    return mapping.get(value, value)


def _extract_chat_message_text(message: dict) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _parse_tool_call_arguments(arguments) -> dict:
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _chat_message_to_anthropic_content(message: dict) -> list[dict]:
    content = []
    text = _extract_chat_message_text(message)
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{uuid4().hex}",
                    "name": function.get("name", ""),
                    "input": _parse_tool_call_arguments(function.get("arguments")),
                }
            )

    if not content:
        content.append({"type": "text", "text": ""})
    return content


def chat_completion_to_anthropic(payload: dict, fallback_model=None) -> dict:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    usage = _chat_usage_to_anthropic(payload.get("usage") if isinstance(payload, dict) else {})

    return {
        "id": payload.get("id") if isinstance(payload, dict) else f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": (payload.get("model") if isinstance(payload, dict) else None) or fallback_model,
        "content": _chat_message_to_anthropic_content(message if isinstance(message, dict) else {}),
        "stop_reason": _chat_stop_reason_to_anthropic(first_choice.get("finish_reason") if isinstance(first_choice, dict) else None),
        "stop_sequence": None,
        "usage": usage,
    }


def _anthropic_error_type_for_status(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code in {503, 529}:
        return "overloaded_error"
    return "api_error"


def anthropic_error_payload_from_openai(payload, status_code: int, fallback_message: str | None = None) -> dict:
    if isinstance(payload, dict):
        if payload.get("type") == "error" and isinstance(payload.get("error"), dict):
            error = payload["error"]
            error_type = error.get("type")
            message = error.get("message")
            if isinstance(error_type, str) and isinstance(message, str) and message:
                return payload

        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        message = error.get("message") if isinstance(error.get("message"), str) else None
        error_type = error.get("type") if isinstance(error.get("type"), str) else None
        detail = payload.get("detail") if isinstance(payload.get("detail"), str) else None

        return {
            "type": "error",
            "error": {
                "type": error_type or _anthropic_error_type_for_status(status_code),
                "message": message or detail or fallback_message or "Request failed",
            },
        }

    return {
        "type": "error",
        "error": {
            "type": _anthropic_error_type_for_status(status_code),
            "message": fallback_message or "Request failed",
        },
    }


def anthropic_error_response(status_code: int, message: str, error_type: str | None = None, headers: dict | None = None) -> JSONResponse:
    payload = {
        "type": "error",
        "error": {
            "type": error_type or _anthropic_error_type_for_status(status_code),
            "message": message,
        },
    }
    return JSONResponse(content=payload, status_code=status_code, headers=headers)


def anthropic_error_response_from_upstream(upstream: httpx.Response) -> JSONResponse:
    headers = {}
    retry_after = upstream.headers.get("retry-after")
    if retry_after:
        headers["retry-after"] = retry_after

    fallback_message = ""
    try:
        fallback_message = upstream.text.strip()
    except Exception:
        fallback_message = ""
    if not fallback_message:
        fallback_message = f"Upstream request failed with status {upstream.status_code}"

    try:
        payload = upstream.json()
    except json.JSONDecodeError:
        payload = None

    translated = anthropic_error_payload_from_openai(payload, upstream.status_code, fallback_message)
    return anthropic_error_response(
        upstream.status_code,
        translated["error"]["message"],
        translated["error"]["type"],
        headers=headers or None,
    )


def _openai_error_type_for_status(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    return "server_error"


def openai_error_response(status_code: int, message: str, error_type: str | None = None, code=None, param=None, headers: dict | None = None) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": error_type or _openai_error_type_for_status(status_code),
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(content=payload, status_code=status_code, headers=headers)


def _upstream_request_error_status_and_message(exc: httpx.RequestError) -> tuple[int, str]:
    if isinstance(exc, httpx.TimeoutException):
        prefix = "Upstream request timed out"
        status_code = 504
    elif isinstance(exc, httpx.ConnectError):
        prefix = "Upstream connection failed"
        status_code = 502
    else:
        prefix = "Upstream request failed"
        status_code = 502

    detail = str(exc).strip()
    return status_code, f"{prefix}: {detail}" if detail else prefix


def _http_exception_detail_to_message(detail) -> str:
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message:
            return message
    return "Request failed"


def _sse_encode(event_name: str, payload: dict) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def _parse_sse_block(raw_block: str) -> tuple[str | None, str | None]:
    event_name = None
    data_lines = []
    for line in raw_block.replace("\r\n", "\n").split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return event_name, None
    return event_name, "\n".join(data_lines)


async def _iter_sse_messages(byte_iter):
    buffer = ""
    async for chunk in byte_iter:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8")
        else:
            buffer += str(chunk)

        normalized = buffer.replace("\r\n", "\n")
        while "\n\n" in normalized:
            raw_block, normalized = normalized.split("\n\n", 1)
            event_name, data = _parse_sse_block(raw_block)
            if data is not None:
                yield event_name, data
        buffer = normalized

    trailing = buffer.strip()
    if trailing:
        event_name, data = _parse_sse_block(trailing)
        if data is not None:
            yield event_name, data


def _extract_text_from_chat_delta(delta) -> str:
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
    return ""


def _extract_tool_call_deltas(delta) -> list[dict]:
    if not isinstance(delta, dict):
        return []
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


async def proxy_anthropic_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    fallback_model: str,
    timeout: int = 300,
    usage_event: dict | None = None,
) -> Response:
    """
    Translate upstream chat-completions SSE into Anthropic Messages SSE.
    """
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        await client.aclose()
        return anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = anthropic_error_payload_from_openai(_extract_upstream_json_payload(upstream), upstream.status_code, fallback_message)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return anthropic_error_response_from_upstream(upstream)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "content-type": "text/event-stream; charset=utf-8",
    }

    async def stream_translated():
        message_id = f"msg_{uuid4().hex}"
        model_name = fallback_model
        input_tokens = 0
        output_tokens = 0
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0
        stop_reason = None
        message_started = False
        last_message_start_usage = None
        next_block_index = 0
        text_block = None
        tool_blocks = {}
        active_block = None
        stream_closed = False
        response_text_parts: list[str] = []

        async def emit_block_stop(block):
            if not isinstance(block, dict) or block.get("closed") is True:
                return
            block["closed"] = True
            yield _sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": block["anthropic_index"],
                },
            )

        async def ensure_message_started():
            nonlocal message_started, last_message_start_usage
            if message_started:
                return
            usage_payload = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }
            yield _sse_encode(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_name,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage_payload,
                    },
                },
            )
            last_message_start_usage = usage_payload
            message_started = True

        async def refresh_message_started_usage():
            nonlocal last_message_start_usage
            if not message_started:
                return

            usage_payload = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }
            if usage_payload == last_message_start_usage:
                return

            # Claude Code copies usage from message_start into the assistant
            # message yielded at content_block_stop. OpenAI chat streams usually
            # surface usage only on the final chunk, so we refresh message_start
            # with the late-arriving totals before the block closes.
            yield _sse_encode(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_name,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage_payload,
                    },
                },
            )
            last_message_start_usage = usage_payload

        async def ensure_text_block():
            nonlocal next_block_index, text_block, active_block

            if text_block is not None and text_block.get("closed") is not True:
                active_block = ("text", text_block["anthropic_index"])
                return

            if active_block is not None:
                block_type, block_key = active_block
                active = text_block if block_type == "text" else tool_blocks.get(block_key)
                if active is not None:
                    async for event in emit_block_stop(active):
                        yield event
                active_block = None

            text_block = {
                "anthropic_index": next_block_index,
                "closed": False,
            }
            next_block_index += 1
            active_block = ("text", text_block["anthropic_index"])
            yield _sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": text_block["anthropic_index"],
                    "content_block": {"type": "text", "text": ""},
                },
            )

        async def ensure_tool_block(tool_delta: dict):
            nonlocal next_block_index, active_block

            openai_index = tool_delta.get("index", 0)
            state = tool_blocks.get(openai_index)
            if state is None:
                state = {
                    "anthropic_index": next_block_index,
                    "id": tool_delta.get("id") if isinstance(tool_delta.get("id"), str) else f"toolu_{uuid4().hex}",
                    "name": "",
                    "arguments": "",
                    "closed": False,
                    "started": False,
                }
                tool_blocks[openai_index] = state
                next_block_index += 1

            function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
            if isinstance(tool_delta.get("id"), str):
                state["id"] = tool_delta["id"]
            if isinstance(function.get("name"), str) and function.get("name"):
                if not state["name"]:
                    state["name"] = function["name"]
                elif function["name"] not in state["name"]:
                    state["name"] += function["name"]
            if isinstance(function.get("arguments"), str) and function.get("arguments"):
                state["arguments"] += function["arguments"]

            if state["started"] and state["closed"] is not True:
                active_block = ("tool", openai_index)
                return

            if active_block is not None:
                block_type, block_key = active_block
                active = text_block if block_type == "text" else tool_blocks.get(block_key)
                if active is not None:
                    async for event in emit_block_stop(active):
                        yield event
                active_block = None

            state["started"] = True
            state["closed"] = False
            active_block = ("tool", openai_index)
            yield _sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state["anthropic_index"],
                    "content_block": {
                        "type": "tool_use",
                        "id": state["id"],
                        "name": state["name"],
                        "input": {},
                    },
                },
            )

        try:
            async for _event_name, data in _iter_sse_messages(upstream.aiter_bytes()):
                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if isinstance(payload.get("id"), str):
                    message_id = payload["id"]
                if isinstance(payload.get("model"), str):
                    model_name = payload["model"]

                usage = payload.get("usage")
                if isinstance(usage, dict):
                    anthropic_usage = _chat_usage_to_anthropic(usage)
                    input_tokens = anthropic_usage.get("input_tokens", input_tokens) or input_tokens
                    output_tokens = anthropic_usage.get("output_tokens", output_tokens) or output_tokens
                    cache_creation_input_tokens = (
                        anthropic_usage.get("cache_creation_input_tokens", cache_creation_input_tokens)
                        or cache_creation_input_tokens
                    )
                    cache_read_input_tokens = (
                        anthropic_usage.get("cache_read_input_tokens", cache_read_input_tokens) or cache_read_input_tokens
                    )

                async for event in ensure_message_started():
                    yield event
                async for event in refresh_message_started_usage():
                    yield event

                choices = payload.get("choices")
                first_choice = choices[0] if isinstance(choices, list) and choices else {}
                delta = first_choice.get("delta") if isinstance(first_choice, dict) else {}
                text_delta = _extract_text_from_chat_delta(delta)
                if text_delta:
                    response_text_parts.append(text_delta)
                    _mark_usage_event_first_output(usage_event)
                    async for event in ensure_text_block():
                        yield event
                    yield _sse_encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_block["anthropic_index"],
                            "delta": {"type": "text_delta", "text": text_delta},
                        },
                    )

                for tool_delta in _extract_tool_call_deltas(delta):
                    async for event in ensure_tool_block(tool_delta):
                        yield event
                    tool_state = tool_blocks.get(tool_delta.get("index", 0))
                    function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
                    arguments_chunk = function.get("arguments")
                    if isinstance(arguments_chunk, str) and arguments_chunk:
                        yield _sse_encode(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_state["anthropic_index"],
                                "delta": {"type": "input_json_delta", "partial_json": arguments_chunk},
                            },
                        )

                finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
                mapped_stop_reason = _chat_stop_reason_to_anthropic(finish_reason)
                if mapped_stop_reason is not None:
                    stop_reason = mapped_stop_reason

            async for event in ensure_message_started():
                yield event
            async for event in refresh_message_started_usage():
                yield event

            if active_block is not None:
                block_type, block_key = active_block
                active = text_block if block_type == "text" else tool_blocks.get(block_key)
                if active is not None:
                    async for event in emit_block_stop(active):
                        yield event
                active_block = None

            if text_block is None and not tool_blocks:
                empty_text_block = {
                    "anthropic_index": next_block_index,
                    "closed": False,
                }
                yield _sse_encode(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": empty_text_block["anthropic_index"],
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                async for event in emit_block_stop(empty_text_block):
                    yield event

            yield _sse_encode(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason or "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": cache_creation_input_tokens,
                        "cache_read_input_tokens": cache_read_input_tokens,
                    },
                },
            )
            yield _sse_encode("message_stop", {"type": "message_stop"})
            stream_closed = True
        finally:
            response_payload = {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "stop_reason": stop_reason or "end_turn",
                "content": [{"type": "text", "text": "".join(response_text_parts)}] if response_text_parts else [],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation_input_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                },
            }
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text="".join(response_text_parts) if response_text_parts else None,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

        if not stream_closed:
            yield _sse_encode("message_stop", {"type": "message_stop"})

    return StreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


def build_chat_headers_for_request(
    request: Request,
    messages,
    model_name: str,
    api_key: str,
    request_id: str | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request)

    initiator = _initiator_policy.resolve_chat_messages(messages, model_name, request_id=request_id)
    headers["X-Initiator"] = initiator

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and ("image_url" in item or item.get("type") == "image_url"):
                        headers["Copilot-Vision-Request"] = "true"
                        break
                if headers.get("Copilot-Vision-Request") == "true":
                    break

    if model_requires_anthropic_beta(model_name):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers
def _anthropic_messages_has_vision(messages) -> bool:
    if not isinstance(messages, list):
        return False
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).lower()
            if part_type == "image":
                return True
            if part_type == "tool_result" and isinstance(part.get("content"), list):
                for nested in part["content"]:
                    if isinstance(nested, dict) and str(nested.get("type", "")).lower() == "image":
                        return True
    return False


def build_anthropic_headers_for_request(
    request: Request,
    body: dict,
    api_key: str,
    request_id: str | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request)

    messages = body.get("messages")
    initiator = _initiator_policy.resolve_anthropic_messages(messages, body.get("model"), request_id=request_id)
    headers["X-Initiator"] = initiator

    if _anthropic_messages_has_vision(messages):
        headers["Copilot-Vision-Request"] = "true"

    if model_requires_anthropic_beta(body.get("model")):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers


def _strip_anthropic_cache_control(value):
    if isinstance(value, list):
        return [_strip_anthropic_cache_control(item) for item in value]

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key == "cache_control":
                continue
            sanitized[key] = _strip_anthropic_cache_control(item)
        return sanitized

    return value


def prepare_anthropic_outbound_body(body: dict, resolved_model: str | None) -> dict:
    allowed_keys = {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "metadata",
        "stop_sequences",
        "stream",
        "tools",
        "tool_choice",
        "service_tier",
        "thinking",
        "container",
        "mcp_servers",
        "betas",
        "output_config",
    }

    outbound = {}
    dropped = []
    for key, value in body.items():
        if key in allowed_keys:
            outbound[key] = value
        else:
            dropped.append(key)

    outbound["model"] = resolved_model

    if dropped:
        print(f"Anthropic proxy dropped unsupported fields: {', '.join(sorted(dropped))}", flush=True)

    return _strip_anthropic_cache_control(outbound)


def encode_fake_compaction(summary_text: str) -> str:
    encoded = base64.urlsafe_b64encode(summary_text.encode("utf-8")).decode("ascii")
    return f"{FAKE_COMPACTION_PREFIX}{encoded}"


def decode_fake_compaction(encrypted_content: str) -> str | None:
    if not isinstance(encrypted_content, str) or not encrypted_content.startswith(FAKE_COMPACTION_PREFIX):
        return None

    encoded = encrypted_content[len(FAKE_COMPACTION_PREFIX) :]
    try:
        decoded = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
    except Exception:
        return None
    return decoded or None


def _summary_message_item(summary_text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": f"{FAKE_COMPACTION_SUMMARY_LABEL}\n{summary_text}",
            }
        ],
    }


def input_contains_compaction(input_items) -> bool:
    if not isinstance(input_items, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "compaction" for item in input_items)


def sanitize_input(input_items):
    """
    Preserve encrypted_content in reasoning items for multi-turn correctness.
    Expand locally synthesized compaction items into a readable summary message.
    Convert other compaction items into reasoning items for GHCP compatibility.
    Strip status=None which GHCP rejects.
    Pass everything else through unchanged.
    """
    if not isinstance(input_items, list):
        return input_items  # plain string — pass through untouched

    result = []
    for item in input_items:
        if not isinstance(item, dict):
            result.append(item)
            continue

        item_type = item.get("type")
        if item_type == "compaction":
            encrypted_content = item.get("encrypted_content")
            summary_text = decode_fake_compaction(encrypted_content)
            if summary_text is not None:
                result.append(_summary_message_item(summary_text))
                continue
            if isinstance(encrypted_content, str) and encrypted_content:
                result.append(
                    {
                        "type": "reasoning",
                        "encrypted_content": encrypted_content,
                    }
                )
                continue
            result.append(item)
            continue

        if item_type != "reasoning":
            result.append(item)
            continue

        filtered = {}
        for k, v in item.items():
            if k == "encrypted_content":
                if v is not None:
                    filtered[k] = v   # always preserve if present
                continue
            if k == "status" and v is None:
                continue              # strip status=None
            if v is not None:
                filtered[k] = v
        result.append(filtered)
    return result


def build_responses_headers_for_request(
    request: Request,
    body: dict,
    api_key: str,
    force_initiator: str | None = None,
    request_id: str | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request)

    had_input = "input" in body
    effective_input, initiator = _initiator_policy.resolve_responses_input(
        body.get("input"),
        body.get("model"),
        force_initiator=force_initiator,
        request_id=request_id,
    )
    if had_input:
        body["input"] = effective_input
    headers["X-Initiator"] = initiator

    if has_vision_input(effective_input):
        headers["Copilot-Vision-Request"] = "true"

    if model_requires_anthropic_beta(body.get("model")):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers


def build_fake_compaction_request(body: dict) -> dict:
    request_input = body.get("input")
    if isinstance(request_input, list):
        request_input = sanitize_input(request_input)

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        request_input = [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Original conversation instructions to preserve:\n{instructions}",
                    }
                ],
            },
            *(request_input if isinstance(request_input, list) else []),
        ]

    return {
        "model": body.get("model"),
        "instructions": COMPACTION_SUMMARY_PROMPT,
        "input": request_input if request_input is not None else [],
        "stream": False,
        "store": False,
        "tools": [],
        "parallel_tool_calls": False,
        "include": [],
        "reasoning": body.get("reasoning"),
        "text": body.get("text"),
    }


def extract_response_output_text(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = payload.get("output")
    if not isinstance(output, list):
        return None

    parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        if str(item.get("role", "")).lower() != "assistant":
            continue
        text = _extract_item_text(item).strip()
        if text:
            parts.append(text)

    if not parts:
        return None
    return "\n\n".join(parts)


def build_fake_compaction_response(body: dict, summary_text: str, usage=None) -> dict:
    return {
        "id": f"resp_{uuid4().hex}",
        "object": "response.compaction",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model"),
        "output": [
            {
                "type": "compaction",
                "encrypted_content": encode_fake_compaction(summary_text),
            }
        ],
        "usage": usage
        if isinstance(usage, dict)
        else {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _render_dashboard_html() -> str:
    with open(DASHBOARD_FILE, encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def dashboard_root():
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/ui", response_class=HTMLResponse)
async def dashboard():
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


@app.get("/api/dashboard")
async def dashboard_api(request: Request):
    refresh = request.query_params.get("refresh", "").lower() in {"1", "true", "yes"}
    payload = await asyncio.to_thread(_build_dashboard_payload, refresh)
    return JSONResponse(content=payload)


@app.get("/api/dashboard/stream")
async def dashboard_stream(request: Request):
    heartbeat_seconds = 20
    queue = _register_dashboard_stream_listener()
    last_version = _dashboard_stream_version

    async def stream():
        nonlocal last_version
        try:
            initial_payload = await asyncio.to_thread(_build_dashboard_payload, False)
            yield _sse_encode("dashboard", initial_payload)
            while True:
                if await request.is_disconnected():
                    break

                try:
                    version = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except asyncio.TimeoutError:
                    yield _sse_encode("heartbeat", {"at": _utc_now_iso()})
                    continue

                if version == last_version:
                    continue
                last_version = version
                payload = await asyncio.to_thread(_build_dashboard_payload, False)
                yield _sse_encode("dashboard", payload)
        finally:
            _unregister_dashboard_stream_listener(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/config/billing-token")
async def billing_token_status_api():
    return JSONResponse(content=_billing_token_status())


@app.post("/api/config/billing-token")
async def billing_token_config_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    if bool(payload.get("clear")):
        _clear_billing_token()
        return JSONResponse(content=_billing_token_status())

    token = payload.get("token")
    if token is None:
        raise HTTPException(status_code=400, detail="Missing token field")
    if not isinstance(token, str):
        raise HTTPException(status_code=400, detail="Token must be a string")
    token = token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token must not be empty")

    current_status = _billing_token_status()
    if current_status.get("readonly"):
        raise HTTPException(
            status_code=409,
            detail="Billing token is configured via GHCP_GITHUB_BILLING_TOKEN and cannot be changed via UI.",
        )

    _save_billing_token(token)
    return JSONResponse(content=_billing_token_status())


@app.get("/api/config/client-proxy")
async def client_proxy_status_api():
    return JSONResponse(content=_proxy_client_status_payload())


@app.post("/api/config/client-proxy")
async def client_proxy_install_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    targets = _normalize_proxy_targets(payload)
    action = payload.get("action", "enable")
    if not isinstance(action, str):
        raise HTTPException(status_code=400, detail='Action must be "enable" or "disable".')

    action = action.strip().lower()
    if action == "install":
        action = "enable"
    if action not in {"enable", "disable"}:
        raise HTTPException(status_code=400, detail='Unsupported action. Use "enable" or "disable".')

    clients = {}

    for target in targets:
        try:
            if action == "disable":
                if target == "codex":
                    clients[target] = _disable_codex_proxy_config()
                else:
                    clients[target] = _disable_claude_proxy_settings()
            elif target == "codex":
                clients[target] = _write_codex_proxy_config()
            else:
                clients[target] = _write_claude_proxy_settings()
        except Exception as exc:
            clients[target] = _empty_proxy_status(
                target,
                CODEX_CONFIG_FILE if target == "codex" else CLAUDE_SETTINGS_FILE,
            )
            clients[target]["error"] = str(exc)
            clients[target]["status_message"] = "failed to write config"

    return JSONResponse(
        content={
            "clients": clients,
            "message": (
                "Proxy enabled for: "
                if action == "enable"
                else "Proxy disabled for: "
            )
            + (
                ", ".join(
                    target
                    for target, payload in sorted(clients.items())
                    if not payload.get("error")
                )
                or "none"
            ),
        }
    )


def _extract_upstream_json_payload(upstream: httpx.Response) -> dict | None:
    content_type = upstream.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return None
    try:
        payload = upstream.json()
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_upstream_text(upstream: httpx.Response) -> str | None:
    try:
        text = upstream.text
    except Exception:
        return None
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return text[:4096]


def proxy_non_streaming_response(upstream: httpx.Response) -> Response:
    """
    Preserve the upstream status code and body shape.

    Most endpoints return JSON, but compaction can return non-JSON payloads
    such as SSE-style frames. When JSON parsing fails, fall back to relaying
    the raw body with the upstream content type instead of crashing.
    """
    headers = {}
    for name in ("content-type", "cache-control", "retry-after"):
        value = upstream.headers.get(name)
        if value:
            headers[name] = value

    content_type = upstream.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            return JSONResponse(
                content=upstream.json(),
                status_code=upstream.status_code,
                headers=headers,
            )
        except json.JSONDecodeError:
            pass

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
    )


async def proxy_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    timeout: int = 300,
    usage_event: dict | None = None,
    stream_type: str = "responses",
) -> Response:
    """
    Relay an upstream SSE response while preserving upstream error statuses.

    If the upstream request fails before the stream starts, return the upstream
    error body as a normal HTTP response instead of masking it as 200 SSE.
    """
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        await client.aclose()
        return openai_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    content_type = upstream.headers.get("content-type")
    if content_type:
        response_headers["content-type"] = content_type

    async def stream_upstream():
        capture = _SSEUsageCapture(stream_type)
        try:
            async for chunk in upstream.aiter_bytes():
                if capture.feed(chunk):
                    _mark_usage_event_first_output(usage_event)
                yield chunk
        finally:
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                usage=capture.usage if isinstance(capture.usage, dict) else None,
            )
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_upstream(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


# ─── Route: /v1/responses  (Codex / Responses API) ───────────────────────────

@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))
    request_id = uuid4().hex

    # Sanitize input (multi-turn encrypted_content passthrough)
    raw_input = body.get("input")
    has_compaction_input = input_contains_compaction(raw_input)
    if raw_input is not None:
        body["input"] = sanitize_input(raw_input)

    try:
        api_key = get_api_key()
    except Exception as e:
        return openai_error_response(401, f"GHCP auth failed: {e}")

    headers = build_responses_headers_for_request(
        request,
        body,
        api_key,
        force_initiator="agent" if has_compaction_input else None,
        request_id=request_id,
    )
    upstream_url = f"{_get_api_base().rstrip('/')}/responses"
    is_streaming = body.get("stream", False)
    log_proxy_request(request, body.get("model"), body.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        body.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=body,
        upstream_path="/responses",
        outbound_headers=headers,
    )

    if is_streaming:
        return await proxy_streaming_response(
            upstream_url,
            headers,
            body,
            timeout=300,
            usage_event=usage_event,
            stream_type="responses",
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = await throttled_client_post(client, upstream_url, headers=headers, json=body)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        except httpx.RequestError as exc:
            status_code, message = _upstream_request_error_status_and_message(exc)
            _finish_usage_event(usage_event, status_code, response_text=message)
            return openai_error_response(status_code, message)
        except Exception:
            _finish_usage_event(usage_event, 599)
            raise


@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))
    request_id = uuid4().hex

    try:
        api_key = get_api_key()
    except Exception as e:
        return openai_error_response(401, f"GHCP auth failed: {e}")

    summary_request = build_fake_compaction_request(body)
    headers = build_responses_headers_for_request(
        request,
        summary_request,
        api_key,
        force_initiator="agent",
        request_id=request_id,
    )
    upstream_url = f"{_get_api_base().rstrip('/')}/responses"
    log_proxy_request(request, body.get("model"), summary_request.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        summary_request.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=summary_request,
        upstream_path="/responses",
        outbound_headers=headers,
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(client, upstream_url, headers=headers, json=summary_request)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        return openai_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        raise

    if upstream.status_code >= 400:
        _finish_usage_event(
            usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=_extract_upstream_json_payload(upstream),
            response_text=_extract_upstream_text(upstream),
        )
        return proxy_non_streaming_response(upstream)

    try:
        upstream_payload = upstream.json()
    except json.JSONDecodeError as e:
        _finish_usage_event(
            usage_event,
            502,
            upstream=upstream,
            response_text=f"Invalid JSON from upstream summarization response: {e}",
        )
        return openai_error_response(502, f"Invalid JSON from upstream summarization response: {e}")

    summary_text = extract_response_output_text(upstream_payload)
    if not summary_text:
        _finish_usage_event(
            usage_event,
            502,
            upstream=upstream,
            response_payload=upstream_payload,
            response_text="Upstream summarization response did not include assistant text output",
        )
        return openai_error_response(502, "Upstream summarization response did not include assistant text output")

    compacted_response = build_fake_compaction_response(body, summary_text, upstream_payload.get("usage"))
    _finish_usage_event(
        usage_event,
        upstream.status_code,
        upstream=upstream,
        response_payload=compacted_response,
    )
    return JSONResponse(content=compacted_response, status_code=200)


# ─── Route: /v1/chat/completions  (non-Codex models) ─────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    For models that still use the Chat API.
    Codex does NOT use this endpoint — it uses /v1/responses above.
    """
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))
    request_id = uuid4().hex

    messages = body.get("messages", [])

    try:
        api_key = get_api_key()
    except Exception as e:
        return openai_error_response(401, f"GHCP auth failed: {e}")

    headers = build_chat_headers_for_request(request, messages, body.get("model"), api_key, request_id=request_id)

    upstream_url = f"{_get_api_base().rstrip('/')}/chat/completions"
    is_streaming = body.get("stream", False)
    log_proxy_request(request, body.get("model"), body.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        body.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=body,
        upstream_path="/chat/completions",
        outbound_headers=headers,
    )

    if is_streaming:
        return await proxy_streaming_response(
            upstream_url,
            headers,
            body,
            timeout=300,
            usage_event=usage_event,
            stream_type="chat",
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = await throttled_client_post(client, upstream_url, headers=headers, json=body)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        except httpx.RequestError as exc:
            status_code, message = _upstream_request_error_status_and_message(exc)
            _finish_usage_event(usage_event, status_code, response_text=message)
            return openai_error_response(status_code, message)
        except Exception:
            _finish_usage_event(usage_event, 599)
            raise


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Anthropic-compatible route.
    Translate Anthropic Messages payloads onto GHCP's OpenAI-compatible chat
    endpoint so Claude Code can reuse Copilot's cache semantics.
    """
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return anthropic_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))
    request_id = uuid4().hex

    try:
        api_key = get_api_key()
    except Exception as e:
        return anthropic_error_response(401, f"GHCP auth failed: {e}")

    api_base = _get_api_base()
    try:
        outbound_body = await anthropic_request_to_chat(body, api_base, api_key)
    except ValueError as e:
        return anthropic_error_response(400, str(e))

    headers = build_anthropic_headers_for_request(request, body, api_key, request_id=request_id)
    upstream_url = f"{api_base.rstrip('/')}/chat/completions"
    log_proxy_request(request, body.get("model"), outbound_body.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        outbound_body.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=outbound_body,
        upstream_path="/chat/completions",
        outbound_headers=headers,
    )

    if outbound_body.get("stream"):
        return await proxy_anthropic_streaming_response(
            upstream_url,
            headers,
            outbound_body,
            outbound_body.get("model"),
            timeout=300,
            usage_event=usage_event,
        )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(client, upstream_url, headers=headers, json=outbound_body)
        if upstream.status_code >= 400:
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = anthropic_error_payload_from_openai(_extract_upstream_json_payload(upstream), upstream.status_code, fallback_message)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return anthropic_error_response(
                upstream.status_code,
                error_payload["error"]["message"],
                error_payload["error"]["type"],
                {"retry-after": upstream.headers.get("retry-after")} if upstream.headers.get("retry-after") else None,
            )
        translated = chat_completion_to_anthropic(upstream.json(), fallback_model=outbound_body.get("model"))
        _finish_usage_event(
            usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=translated,
        )
        return JSONResponse(content=translated, status_code=upstream.status_code)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        return anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        raise


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Run auth interactively in the terminal BEFORE the server starts.
    # If no token exists this will print the device flow prompt and block until
    # the user authorizes (or the script exits on failure).
    ensure_authenticated()

    # Step 2: Start the server in the foreground on this terminal.
    print("Starting GHCP proxy on http://localhost:8000", flush=True)
    print("  Responses API : POST /v1/responses", flush=True)
    print("  Compaction    : POST /v1/responses/compact", flush=True)
    print("  Chat API      : POST /v1/chat/completions", flush=True)
    print("", flush=True)
    print("  Set in your shell:", flush=True)
    print("    export OPENAI_BASE_URL=http://localhost:8000/v1", flush=True)
    print("    export OPENAI_API_KEY=anything", flush=True)
    print("", flush=True)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
