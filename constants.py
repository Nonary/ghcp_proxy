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
OPENCODE_HEADER_VERSION = "OpenCode/1.0"
OPENCODE_INTEGRATION_ID = "vscode-chat"
UPSTREAM_REQUESTS_PER_WINDOW = 5
UPSTREAM_REQUEST_WINDOW_SECONDS = 1.0
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 300

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
CLAUDE_MAX_CONTEXT_TOKENS = "128000"
CLAUDE_MAX_OUTPUT_TOKENS = "64000"
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
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": CLAUDE_MAX_CONTEXT_TOKENS,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": CLAUDE_MAX_OUTPUT_TOKENS,
    },
    "effortLevel": "medium",
}
DETAILED_REQUEST_HISTORY_LIMIT = 1000
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

# ─── Premium request tracking ────────────────────────────────────────────────
PREMIUM_CACHE_TTL_SECONDS = 60.0

# ─── Model pricing & SKU tables ──────────────────────────────────────────────
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
