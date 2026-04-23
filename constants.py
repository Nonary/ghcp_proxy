"""
Static configuration data, constants, and pricing tables for ghcp_proxy.

Extracted from proxy.py to keep the main module focused on runtime logic.
This file contains only module-level constants, dicts, and string data —
no functions or classes.
"""

import os

# ─── Constants ────────────────────────────────────────────────────────────────
GITHUB_CLIENT_ID        = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL  = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_KEY_URL      = "https://api.github.com/copilot_internal/v2/token"
GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com"

OPENCODE_VERSION = "1.3.13"
OPENCODE_INTEGRATION_ID = "vscode-chat"
GITHUB_API_VERSION = "2026-01-09"
UPSTREAM_REQUESTS_PER_WINDOW = 5
UPSTREAM_REQUEST_WINDOW_SECONDS = 1.0
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 300

TOKEN_DIR         = os.path.expanduser("~/.config/ghcp_proxy")
ACCESS_TOKEN_FILE = os.path.join(TOKEN_DIR, "access-token")
API_KEY_FILE      = os.path.join(TOKEN_DIR, "api-key.json")
MODEL_ROUTING_CONFIG_FILE = os.path.join(TOKEN_DIR, "model-routing.json")
CLIENT_PROXY_SETTINGS_FILE = os.path.join(TOKEN_DIR, "client-proxy.json")
LEGACY_PREMIUM_PLAN_CONFIG_FILE = os.path.join(TOKEN_DIR, "premium-plan.json")
LEGACY_BILLING_TOKEN_FILE = os.path.join(TOKEN_DIR, "billing-token")
SAFEGUARD_CONFIG_FILE = os.path.join(TOKEN_DIR, "safeguard.json")
USAGE_LOG_FILE    = os.path.join(TOKEN_DIR, "usage-log.jsonl")
REQUEST_ERROR_LOG_FILE = os.path.join(TOKEN_DIR, "request-errors.log")
REQUEST_TRACE_LOG_FILE = os.path.join(TOKEN_DIR, "request-trace.jsonl")
PROXY_BASE_URL    = "http://localhost:8000"
CODEX_PROXY_BASE_URL = f"{PROXY_BASE_URL}/v1"
DASHBOARD_BASE_URL = "http://localhost:8000"
DASHBOARD_FILE    = os.path.join(os.path.dirname(__file__), "dashboard.html")
SQLITE_CACHE_FILE = os.path.join(
    os.path.expanduser(os.environ.get("GHCP_CACHE_DB_PATH", os.path.join(TOKEN_DIR, ".ghcp_proxy-cache-v2.sqlite3")))
)
CODEX_CONFIG_DIR    = os.path.expanduser("~/.codex")
CODEX_PRIMARY_CONFIG_FILE = os.path.join(CODEX_CONFIG_DIR, "config.toml")
CODEX_MANAGED_CONFIG_FILE = os.path.join(CODEX_CONFIG_DIR, "managed_config.toml")
CODEX_PROXY_MODEL_CATALOG_FILE = os.path.join(CODEX_CONFIG_DIR, "ghcp-proxy-models.json")
# Codex presents the usable window at 95% of this raw prompt limit; 272k
# therefore reports as the expected ~258k before auto compaction.
CODEX_PROXY_MODEL_CONTEXT_WINDOW = 272000
CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT = 120000
CLAUDE_CONFIG_DIR   = os.path.expanduser("~/.claude")
CLAUDE_SETTINGS_FILE = os.path.join(CLAUDE_CONFIG_DIR, "settings.json")
CLAUDE_MAX_CONTEXT_TOKENS = "128000"
CLAUDE_MAX_OUTPUT_TOKENS = "64000"
# Default GPT model to fall back to when Codex requests a compact against a
# chat-backed target (Claude/Gemini/Grok). Those targets do not handle Codex
# compact payloads as cleanly as the native Responses path, so we route the
# compact call to a GPT model instead. Per-mapping override lives in
# model-routing.json.
DEFAULT_COMPACT_FALLBACK_MODEL = "gpt-5.4"
# Default reasoning_effort to forward upstream for Claude-family models when
# the inbound Anthropic body does not carry an explicit effort selection.
# Claude Code commonly emits ``thinking: {"type":"adaptive"}`` regardless of
# the user's picker choice, so translation falls back to this configured level.
CLAUDE_DEFAULT_REASONING_EFFORT = "medium"
CODEX_PROXY_CONFIG = """\
model_provider = "custom"
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
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": CLAUDE_MAX_CONTEXT_TOKENS,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": CLAUDE_MAX_OUTPUT_TOKENS,
    },
    "effortLevel": "medium",
}
DETAILED_REQUEST_HISTORY_LIMIT = 5000
# Rolling retention for the request-trace log. Appends past this count are
# compacted down to the most recent N rows via a temp-file rewrite (same
# pattern as the usage log). See proxy._enforce_trace_retention.
REQUEST_TRACE_HISTORY_LIMIT = 1000
# Trim only when we've drifted this far past the limit so the rewrite cost
# is amortized across many appends instead of firing on every line.
REQUEST_TRACE_RETENTION_SLACK = 64
# Per-field cap for body payloads captured in the trace log. Keeps a
# 1000-row file to a bounded size even when upstream system prompts are
# multi-megabyte. See proxy._trim_trace_field.
REQUEST_TRACE_BODY_MAX_BYTES = 8192
# Maximum characters of each prompt slot (system/user) captured alongside
# usage events and the request trace so the dashboard can surface a
# human-readable preview without retaining full upstream payloads.
REQUEST_PROMPT_PREVIEW_MAX_CHARS = 4000
# Maximum characters of the assistant's reasoning ("thinking") text retained on
# the finished usage event so dashboards can surface the first portion of the
# model's chain of thought without bloating per-request rows.
RESPONSE_REASONING_PREVIEW_MAX_CHARS = 1000
FORWARDED_REQUEST_HEADERS = (
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
COMPACTION_SUMMARY_PROMPT = """Please create a detailed summary of the conversation so far. The history is being compacted so moving forward, all conversation history will be removed and you'll only have this summary to work from. Be sure to make note of the user's explicit requests, your actions, and any key technical details.

The summary should include the following parts:
1. <overview> - high-level summary of goals and approach
2. <history> - chronological analysis of the conversation
3. <work_done> - changes made, current state, and any issues encountered
4. <technical_details> - key concepts, decisions, and quirks discovered
5. <important_files> - files central to the work and why they matter
6. <next_steps> - pending tasks and planned actions
7. <checkpoint_title> - 2-6 word description of the main work done

---

## Section Guidelines

### Overview

Provide a concise summary (2-3 sentences) capturing the user's goals, intent, and expectations. Describe your overall approach and strategy for addressing their needs, and note any constraints or requirements that were established.
This section should give a reader immediate clarity on what this conversation is about and how you're tackling it.

### History

Capture the narrative arc of the conversation\u2014what was asked for, what was done, and how the work evolved. Structure this around the user's requests: each request becomes an entry with the actions you took nested underneath, in chronological order.
Note any major pivots or changes in direction, and include outcomes where relevant\u2014especially for debugging or when something didn't go as expected. Focus on meaningful actions, not granular details of every exchange.

### Work Done

Document the concrete work completed during this conversation. This section should enable someone to pick up exactly where you left off. Include:

- Files created, modified, or deleted
- Tasks completed and their outcomes
- What you were most recently working on
- Current state: what works, what doesn't, what's untested

### Technical Details

Capture the technical knowledge that would be painful to rediscover. Think of this as a knowledge base for your future self\u2014anything that took effort to learn belongs here. This includes:

- Key concepts and architectural decisions (with rationale)
- Issues encountered and how they were resolved
- Quirks, gotchas, or non-obvious behaviors
- Dependencies, versions, or environment details that matter
- Workarounds or constraints you discovered

Also make note of any questions that remain unanswered or assumptions that you aren't fully confident about.

### Important Files

List the files most central to the task, prioritizing those you've actively worked on over files you merely viewed. This isn't an exhaustive inventory\u2014it's a curated list of what matters most for continuing the work. For each file, include:

- The file name
- Why it's important to the project
- Summary of changes made (if any)
- Key line numbers or sections to reference

### Next Steps

If there's pending work, describe what you were actively working on when compaction occurred. List remaining tasks, outline your planned approach, and flag any blockers or open questions.
If you've finished all requested work, you can simply note that no next steps are needed.

### Checkpoint Title

Provide a concise 2-6 word title capturing the essence of what was accomplished in this work segment. This title will be used to identify this checkpoint when reviewing session history. Examples:
- "Implementing user authentication"
- "Fixing database connection bugs"
- "Refactoring payment module"
- "Adding unit tests for API"

---

## Example

Here is an example of the structure you should follow:
<example>
<overview>
[2-3 sentences describing the user's goals and your approach]
</overview>
<history>
1. The user asked to [request]
   - [action taken]
   - [action taken]
   - [outcome/result]

2. The user asked to [request]
   - [action taken]
   - [action taken]
   - [outcome/result]
</history>
<work_done>
Files updated:
- [file]: [what changed]

Work completed:
- [x] [Task]
- [x] [Task]
- [ ] [Task in progress or incomplete]
</work_done>
<technical_details>
- [Key technical concept or decision]
- [Issue encountered and how it was resolved]
- [Non-obvious behavior or quirk discovered]
- [Unresolved question or uncertain area]
</technical_details>
<important_files>
- [file1]
   - [Why it matters]
   - [Changes made, if any]
   - [Key line numbers]
- [file2]
   - [Why it matters]
   - [Changes made, if any]
   - [Key line numbers]
</important_files>
<next_steps>
Remaining work:
- [Task]
- [Task]

Immediate next steps:
- [Action to take]
- [Action to take]
</next_steps>

<checkpoint_title>Concise 2-6 word description of this checkpoint</checkpoint_title>
</example>

---

Please write the summary now, following the structure and guidelines above. Be concise where possible, but don't sacrifice important context for brevity."""

# ─── Model pricing & SKU tables ──────────────────────────────────────────────
MODEL_PRICING = {
    "claude-haiku-4-5": {
        "provider": "Anthropic",
        "input_per_million": 1.00,
        "cached_input_per_million": 0.10,
        "output_per_million": 5.00,
    },
    "claude-sonnet-4.6": {
        "provider": "Anthropic",
        "input_per_million": 3.00,
        "cached_input_per_million": 0.30,
        "output_per_million": 15.00,
    },
    "claude-opus-4.6": {
        "provider": "Anthropic",
        "input_per_million": 5.00,
        "cached_input_per_million": 0.50,
        "output_per_million": 25.00,
    },
    "claude-opus-4.7": {
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
    "gpt-5-mini": {
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
    "gpt-5.5": {
        "provider": "OpenAI",
        "input_per_million": 5.00,
        "cached_input_per_million": 0.50,
        "output_per_million": 30.00,
    },
    "gpt-5.4-mini": {
        "provider": "OpenAI",
        "input_per_million": 0.75,
        "cached_input_per_million": 0.075,
        "output_per_million": 4.50,
    },
    "gemini-3-flash-preview": {
        "provider": "Google",
        "input_per_million": 0.10,
        "cached_input_per_million": 0.025,
        "output_per_million": 0.40,
    },
    "gemini-3.1-pro-preview": {
        "provider": "Google",
        "input_per_million": 1.75,
        "cached_input_per_million": 0.175,
        "output_per_million": 14.00,
    },
    "grok-code-fast-1": {
        "provider": "xAI",
        "input_per_million": 0.20,
        "cached_input_per_million": 0.05,
        "output_per_million": 1.00,
    },
}

MODEL_PRICING_ALIASES = {
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
    "anthropic/claude-opus-4.5": "claude-opus-4.6",
    "anthropic/claude-sonnet-4": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.6",
    "claude-haiku-4.5": "claude-haiku-4-5",
    "claude-opus-4.5": "claude-opus-4.6",
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4": "claude-sonnet-4.6",
    "claude-sonnet-4.5": "claude-sonnet-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "gpt-5.4 mini": "gpt-5.4-mini",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5 mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
}

PREMIUM_REQUEST_MULTIPLIERS = {
    "claude-haiku-4-5": 0.33,
    "claude-opus-4.5": 3.0,
    "claude-opus-4.6": 3.0,
    "claude-opus-4.7": 7.5,
    "claude-sonnet-4": 1.0,
    "claude-sonnet-4.5": 1.0,
    "claude-sonnet-4.6": 1.0,
    "gpt-4.1": 0.0,
    "gpt-4o": 0.0,
    "gpt-5.2": 1.0,
    "gpt-5.2-codex": 1.0,
    "gpt-5.3-codex": 1.0,
    "gpt-5.4": 1.0,
    "gpt-5.5": 1.0,
    "gpt-5.4-mini": 0.33,
    "gpt-5-mini": 0.0,
    "gemini-2.5-pro": 1.0,
    "gemini-3-flash": 0.33,
    "gemini-3-flash-preview": 0.33,
    "gemini-3-pro": 1.0,
    "gemini-3.1-pro": 1.0,
    "gemini-3.1-pro-preview": 1.0,
    "grok-code-fast-1": 0.33,
    "raptor-mini": 0.0,
}

# ─── Safeguard defaults ──────────────────────────────────────────────────────
SAFEGUARD_DEFAULT_COOLDOWN_SECONDS = 15.0
SAFEGUARD_MIN_COOLDOWN_SECONDS = 0.0
SAFEGUARD_MAX_COOLDOWN_SECONDS = 600.0
