"""OpenAI Excel add-in upstream support.

The official Excel add-in uses a ChatGPT session rather than the GitHub
Copilot token used by the rest of this proxy.  Keep that credential isolated
in memory and expose only non-secret status information.
"""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from app_paths import user_state_dir


MODEL_ID = "gpt-excel"
UPSTREAM_MODEL = "gpt-5.5"
EXTERNAL_CLIENT_INSTRUCTIONS = (
    "This request is relayed by an external OpenAI Responses API client, not by "
    "the live Excel workbook. Do not call server-injected Excel, Office, connector, "
    "or workbook tools. Return the answer as assistant text."
)
TOOL_CALL_MARKER_OPEN = "<codex_tool_call>"
TOOL_CALL_MARKER_CLOSE = "</codex_tool_call>"
TOOL_OUTPUT_MARKER_OPEN = "<codex_tool_output"
TOOL_OUTPUT_MARKER_CLOSE = "</codex_tool_output>"
_TOOL_CALL_PATTERN = re.compile(
    re.escape(TOOL_CALL_MARKER_OPEN)
    + r"\s*(\{.*?\})\s*"
    + re.escape(TOOL_CALL_MARKER_CLOSE),
    re.DOTALL,
)
RESPONSES_URL = os.environ.get(
    "GHCP_EXCEL_RESPONSES_URL",
    "https://bps.openai.com/basispoints/api/responses",
).strip()
# prompt_cache_key is the documented OpenAI cache-routing control; set to 0
# only if the Basispoints gateway ever starts rejecting the parameter.
FORWARD_PROMPT_CACHE_KEY = os.environ.get(
    "GHCP_EXCEL_FORWARD_PROMPT_CACHE_KEY", "1"
).strip().lower() not in {"0", "false", "no", "off"}
# Escape hatch back to the pre-cache-fix layout (full catalog as the prompt
# suffix) in case the compact trailing reminder ever stops holding the model to
# the marker protocol.  See _client_tool_protocol_reminder for why it moved.
CATALOG_AT_PROMPT_END = os.environ.get(
    "GHCP_EXCEL_CATALOG_AT_PROMPT_END", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SESSION_FILE = os.path.join(user_state_dir(), "excel-session.dpapi")

_ALLOWED_CAPTURED_HEADERS = frozenset(
    {
        "authorization",
        "chatgpt-account-id",
        "user-agent",
        "x-basispoints-auth-mode",
        "x-openai-account-id",
        "x-openai-account-user-id",
        "x-openai-internal-basispoints-browser-name",
        "x-openai-internal-basispoints-browser-ua-brands",
        "x-openai-internal-basispoints-browser-ua-mobile",
        "x-openai-internal-basispoints-browser-ua-platform",
        "x-openai-internal-basispoints-client-agent-profile",
        "x-openai-internal-basispoints-client-editor",
        "x-openai-internal-basispoints-client-host",
        "x-openai-internal-basispoints-client-platform",
        "x-openai-internal-basispoints-client-platform-class",
        "x-openai-internal-basispoints-client-product",
        "x-openai-internal-basispoints-client-runtime",
        "x-openai-internal-basispoints-office-host",
        "x-openai-internal-basispoints-office-platform",
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
    }
)

_DEFAULT_CLIENT_HEADERS = {
    "x-basispoints-auth-mode": "chatgpt",
    "x-openai-internal-basispoints-client-agent-profile": "excel",
    "x-openai-internal-basispoints-client-editor": "excel",
    "x-openai-internal-basispoints-client-host": "office",
    "x-openai-internal-basispoints-client-platform": "excel",
    "x-openai-internal-basispoints-client-platform-class": "PC",
    "x-openai-internal-basispoints-client-product": "basispoints-excel-plugin",
    "x-openai-internal-basispoints-client-runtime": "desktop",
    "x-openai-internal-basispoints-office-host": "Excel",
    "x-openai-internal-basispoints-office-platform": "PC",
    "x-stainless-arch": "unknown",
    "x-stainless-lang": "js",
    "x-stainless-os": "Unknown",
    "x-stainless-package-version": "6.31.0",
    "x-stainless-retry-count": "0",
    "x-stainless-runtime": "browser:chrome",
}

LOCAL_MODEL_CAPABILITIES = {
    MODEL_ID: {
        "auto_compact_token_limit": 180_000,
        "context_window": 200_000,
        "display_name": "GPT Excel",
        "input_modalities": ["text"],
        "max_context_window": 200_000,
        "messages_endpoint_supported": False,
        "model_picker_enabled": True,
        "parallel_tool_calls": False,
        "provider": "OpenAI Excel",
        "reasoning_efforts": ["medium", "xhigh"],
        "supported_endpoints": ["/responses"],
        "vision": False,
    }
}


def is_excel_model(model: object) -> bool:
    return isinstance(model, str) and model.strip().lower() == MODEL_ID


def local_model_payload() -> dict[str, object]:
    return {
        "id": MODEL_ID,
        "object": "model",
        "created": 0,
        "owned_by": "openai-excel",
    }


def merge_local_model_capabilities(capabilities: dict[str, dict] | None) -> dict[str, dict]:
    merged = dict(capabilities or {})
    merged.update({key: dict(value) for key, value in LOCAL_MODEL_CAPABILITIES.items()})
    return merged


def merge_local_models_payload(payload: dict | None) -> dict:
    result = dict(payload or {})
    raw_data = result.get("data")
    data = [dict(item) for item in raw_data if isinstance(item, dict)] if isinstance(raw_data, list) else []
    if not any(item.get("id") == MODEL_ID for item in data):
        data.append(local_model_payload())
    result["object"] = result.get("object") or "list"
    result["data"] = data
    return result


def client_tool_types(source: dict) -> dict[str, str]:
    if str(source.get("tool_choice") or "").strip().lower() == "none":
        return {}
    result: dict[str, str] = {}
    tools = source.get("tools")
    if not isinstance(tools, list):
        return result
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        name = tool.get("name")
        if tool_type not in {"function", "custom"} or not isinstance(name, str):
            continue
        normalized_name = name.strip()
        if normalized_name:
            result[normalized_name] = tool_type
    return result


def _client_tool_specs(source: dict) -> dict[str, dict]:
    allowed_tools = client_tool_types(source)
    result: dict[str, dict] = {}
    tools = source.get("tools")
    if not isinstance(tools, list):
        return result
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name in allowed_tools:
            result[name] = tool
    return result


def _value_matches_schema(value: object, schema: object) -> bool:
    if not isinstance(schema, dict) or not schema:
        return True
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        return any(
            _value_matches_schema(value, {**schema, "type": candidate})
            for candidate in expected_type
        )
    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required")
        if isinstance(required, list) and any(
            isinstance(key, str) and key not in value for key in required
        ):
            return False
        properties = schema.get("properties")
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False and any(
                key not in properties for key in value
            ):
                return False
            for key, nested_value in value.items():
                nested_schema = properties.get(key)
                if nested_schema is not None and not _value_matches_schema(
                    nested_value,
                    nested_schema,
                ):
                    return False
    elif expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if item_schema is not None and any(
            not _value_matches_schema(item, item_schema) for item in value
        ):
            return False
    elif expected_type == "string" and not isinstance(value, str):
        return False
    elif expected_type == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        return False
    elif expected_type == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        return False
    elif expected_type == "boolean" and not isinstance(value, bool):
        return False
    elif expected_type == "null" and value is not None:
        return False
    enum = schema.get("enum")
    return not isinstance(enum, list) or value in enum


_PLAN_STATUS_BY_ALIAS = {
    "pending": "pending",
    "not_started": "pending",
    "todo": "pending",
    "planned": "pending",
    "queued": "pending",
    "blocked": "pending",
    "in_progress": "in_progress",
    "active": "in_progress",
    "started": "in_progress",
    "doing": "in_progress",
    "current": "in_progress",
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "finished": "completed",
}


def _normalize_plan_status(status: object) -> str | None:
    if not isinstance(status, str):
        return None
    key = status.strip().lower().replace("-", "_").replace(" ", "_")
    return _PLAN_STATUS_BY_ALIAS.get(key, status)


def _normalize_native_function_arguments(name: str, arguments: dict) -> dict:
    if name != "update_plan":
        return arguments
    plan = arguments.get("plan")
    if not isinstance(plan, list):
        return arguments
    normalized_plan = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, str):
            step = item.get("description")
        if not isinstance(step, str):
            step = item.get("title")
        status = _normalize_plan_status(item.get("status"))
        if isinstance(step, str) and isinstance(status, str):
            normalized_plan.append({"step": step, "status": status})
    normalized: dict[str, object] = {"plan": normalized_plan}
    explanation = arguments.get("explanation")
    if not isinstance(explanation, str):
        explanation = arguments.get("summary")
    if isinstance(explanation, str) and explanation:
        normalized["explanation"] = explanation
    return normalized


def extract_native_client_tool_call(
    response: dict | None,
    source: dict,
) -> dict[str, str] | None:
    if not isinstance(response, dict):
        return None
    specs = _client_tool_specs(source)
    output = response.get("output")
    if not isinstance(output, list):
        return None
    native_calls = [
        item
        for item in output
        if isinstance(item, dict)
        and item.get("type") in {"function_call", "custom_tool_call"}
    ]
    if len(native_calls) != 1:
        return None
    native = native_calls[0]
    name = native.get("name")
    if not isinstance(name, str) or name not in specs:
        return None
    spec = specs[name]
    expected_type = str(spec.get("type") or "").strip().lower()
    if native.get("type") == "function_call" and expected_type == "function":
        raw_arguments = native.get("arguments")
        if not isinstance(raw_arguments, str):
            return None
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        arguments = _normalize_native_function_arguments(name, arguments)
        if not _value_matches_schema(arguments, spec.get("parameters")):
            return None
        call_id = f"call_{uuid4().hex}"
        return {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(
                arguments,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        }
    if native.get("type") == "custom_tool_call" and expected_type == "custom":
        custom_input = native.get("input")
        if not isinstance(custom_input, str):
            return None
        call_id = f"call_{uuid4().hex}"
        return {
            "type": "custom_tool_call",
            "id": f"ctc_{call_id}",
            "call_id": call_id,
            "name": name,
            "input": custom_input,
        }
    return None


def _client_tool_protocol_instructions(source: dict) -> str:
    allowed_tools = client_tool_types(source)
    if not allowed_tools:
        return EXTERNAL_CLIENT_INSTRUCTIONS

    tool_catalog: list[dict[str, object]] = []
    for tool in source.get("tools", []):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or name not in allowed_tools:
            continue
        tool_type = allowed_tools[name]
        entry: dict[str, object] = {
            "type": tool_type,
            "name": name,
        }
        description = tool.get("description")
        if isinstance(description, str) and description:
            entry["description"] = description
        if tool_type == "function":
            parameters = tool.get("parameters")
            entry["parameters"] = parameters if isinstance(parameters, dict) else {}
        else:
            custom_format = tool.get("format")
            if isinstance(custom_format, dict):
                entry["format"] = custom_format
        tool_catalog.append(entry)

    catalog_json = json.dumps(
        tool_catalog,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        "This request is relayed by an external Codex Responses API client, not "
        "by the live Excel workbook. Every native server-injected tool in this "
        "environment is unavailable to this relayed request, including list_skills, "
        "Excel, Office, connector, workbook, and web-search tools. Never emit a "
        "native tool call. The only valid tool mechanism is the Codex client marker "
        "protocol below. The Codex client has the local tools in the JSON catalog. "
        "Never claim shell, filesystem, or workspace access is unavailable when the "
        "catalog contains a suitable tool. For local repository inspection, prefer "
        "shell_command when it is present. When a client tool is needed, return "
        "exactly one tool request and no other text. For a function tool, use "
        f'{TOOL_CALL_MARKER_OPEN}{{"name":"TOOL_NAME","arguments":{{...}}}}'
        f"{TOOL_CALL_MARKER_CLOSE}. For a custom tool, use "
        f'{TOOL_CALL_MARKER_OPEN}{{"name":"TOOL_NAME","input":"RAW_INPUT"}}'
        f"{TOOL_CALL_MARKER_CLOSE}. Arguments must follow the catalog schema. "
        "The proxy converts the marker into a real Responses tool call, and the "
        "next request will contain its tool output. In the conversation history, "
        "your earlier tool requests and their results appear as standard "
        "function_call and function_call_output items. Use each output and "
        "continue, requesting another tool the same way when needed or "
        "returning the final assistant answer normally. Never repeat a tool "
        "request whose output is already present in the transcript. Plan or "
        "status tools such as update_plan only record progress for the user: "
        "never re-issue an unchanged plan, and after updating the plan take a "
        "substantive action before updating it again. Available client tools:\n"
        + catalog_json
        + "\nRemember: do not use native server tools. A client tool request must be "
        + TOOL_CALL_MARKER_OPEN
        + " JSON "
        + TOOL_CALL_MARKER_CLOSE
        + " and nothing else."
    )


def _client_tool_protocol_reminder(source: dict) -> str:
    """Compact recency cue that stands in for the full catalog at the tail.

    The catalog itself is ~3.5k tokens.  While it sat at the end of the prompt
    it re-billed as fresh input on *every* turn: the upstream prompt cache can
    only extend to the point where the previous request diverged, and appending
    new history in front of a trailing catalog puts that divergence right at
    the catalog's first byte.  Wire captures showed a hard floor of ~3.8k fresh
    input tokens per request for exactly that reason.  So the catalog moved
    into the cached prefix and this reminder - a couple of hundred bytes -
    carries the recency that the original A/B replays showed the model needs.
    """
    allowed_tools = client_tool_types(source)
    if not allowed_tools:
        return ""
    reminder = (
        "Reminder: every native server-injected tool (list_skills, Excel, "
        "Office, connector, workbook, web search) is unavailable to this "
        "relayed request. Never emit a native tool call. To use a client tool, "
        f"reply with exactly one {TOOL_CALL_MARKER_OPEN} JSON "
        f"{TOOL_CALL_MARKER_CLOSE} marker and no other text, following the "
        "catalog schema in the developer message above. Client tools: "
        + ", ".join(sorted(allowed_tools))
        + "."
    )
    if "shell_command" in allowed_tools:
        reminder += " For repository inspection use shell_command."
    if "update_plan" in allowed_tools:
        reminder += (
            " Never re-issue an unchanged plan; after update_plan take a "
            "substantive action before planning again."
        )
    return reminder


# Appended to every update_plan tool output. The upstream harness makes the
# model plan-happy; live A/B replays against Basispoints showed a directive
# inside the freshest tool output is what reliably moves it from re-planning
# to substantive marker tool calls.
PLAN_TOOL_OUTPUT_STEERING = (
    "Do not call update_plan again now. Take the next step immediately using "
    "a client tool from the catalog: reply with exactly one "
    f"{TOOL_CALL_MARKER_OPEN} marker and no other text."
)
PLAN_TOOL_OUTPUT_SHELL_HINT = " For repository inspection use shell_command."


def extract_client_tool_call(
    text: str,
    allowed_tools: dict[str, str],
) -> dict[str, str] | None:
    if not isinstance(text, str) or not allowed_tools:
        return None
    match = _TOOL_CALL_PATTERN.search(text)
    if match is None:
        return None
    try:
        marker = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(marker, dict):
        return None
    name = marker.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    tool_type = allowed_tools.get(name)
    if tool_type == "function":
        arguments = marker.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        call_id = f"call_{uuid4().hex}"
        return {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(
                arguments,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        }
    if tool_type == "custom":
        custom_input = marker.get("input")
        if not isinstance(custom_input, str):
            return None
        call_id = f"call_{uuid4().hex}"
        return {
            "type": "custom_tool_call",
            "id": f"ctc_{call_id}",
            "call_id": call_id,
            "name": name,
            "input": custom_input,
        }
    return None


DUPLICATE_PLAN_UPDATE_TEXT = (
    "Plan update skipped: the previous plan update is still current. "
    "Continuing with the existing plan."
)


def _canonical_plan_arguments(arguments: object) -> str:
    """Canonical form of update_plan arguments, ignoring the explanation.

    Amnesiac re-plans repeat the same steps with trivially reworded
    explanations; only the plan itself decides whether an update progresses.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if isinstance(arguments, dict) and isinstance(arguments.get("plan"), list):
        arguments = arguments["plan"]
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return str(arguments)


def is_plan_update_churn(tool_call: dict | None, source: dict) -> bool:
    """True when a plan update is stuck churning instead of progressing.

    Codex legitimately issues two plan updates back to back (create the plan,
    then mark the first step in progress), so a single consecutive update is
    allowed unless it is an exact repeat of the previous one. A third
    consecutive plan update with no substantive tool call in between is
    churn. A user message resets the window: after human input the model may
    always update the plan.
    """
    if not isinstance(tool_call, dict) or tool_call.get("name") != "update_plan":
        return False
    raw_input = source.get("input")
    if not isinstance(raw_input, list):
        return False

    consecutive = 0
    latest_plan_call: dict | None = None
    for item in reversed(raw_input):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "custom_tool_call"}:
            if item.get("name") != "update_plan":
                break
            consecutive += 1
            if latest_plan_call is None:
                latest_plan_call = item
            continue
        if item_type == "message" or (item_type is None and "role" in item):
            if str(item.get("role") or "").strip().lower() == "user":
                break
            continue
        # Tool outputs, reasoning, and other bookkeeping items neither break
        # nor extend the consecutive-plan-update chain.
    if consecutive == 0 or latest_plan_call is None:
        return False
    if consecutive >= 2:
        return True
    return _canonical_plan_arguments(
        latest_plan_call.get("arguments")
    ) == _canonical_plan_arguments(tool_call.get("arguments"))


def strip_tool_call_markers(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return _TOOL_CALL_PATTERN.sub("", text)


def response_payload_with_text(response: dict | None, text: str) -> dict[str, object]:
    result = dict(response or {})
    result.setdefault("id", f"resp_{uuid4().hex}")
    result.setdefault("object", "response")
    result.setdefault("created_at", int(time.time()))
    result["status"] = "completed"
    result["model"] = MODEL_ID
    result["output"] = [
        {
            "type": "message",
            "id": f"msg_{uuid4().hex}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
    ]
    result["error"] = None
    result["incomplete_details"] = None
    return result


def response_payload_with_tool_call(
    response: dict | None,
    tool_call: dict[str, str],
) -> dict[str, object]:
    result = dict(response or {})
    result.setdefault("id", f"resp_{uuid4().hex}")
    result.setdefault("object", "response")
    result.setdefault("created_at", int(time.time()))
    result["status"] = "completed"
    result["model"] = MODEL_ID
    result["output"] = [{**tool_call, "status": "completed"}]
    result["error"] = None
    result["incomplete_details"] = None
    return result


def _decode_jwt_exp(authorization: str) -> float | None:
    token = authorization.split(None, 1)[1]
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padding = "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
        expiration = payload.get("exp")
        return float(expiration) if isinstance(expiration, (int, float)) else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _windows_dpapi_transform(data: bytes, *, decrypt: bool) -> bytes:
    if sys.platform != "win32":
        raise RuntimeError("secure Excel session persistence requires Windows DPAPI")

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_buffer = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(
        len(data),
        ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte)),
    )
    output_blob = _DataBlob()
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if decrypt:
        crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        succeeded = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            flags,
            ctypes.byref(output_blob),
        )
    else:
        description = "ghcp_proxy GPT Excel session"
        crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            wintypes.LPCWSTR,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        succeeded = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            description,
            None,
            None,
            None,
            flags,
            ctypes.byref(output_blob),
        )
    if not succeeded:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))


def _protect_windows_data(data: bytes) -> bytes:
    return _windows_dpapi_transform(data, decrypt=False)


def _unprotect_windows_data(data: bytes) -> bytes:
    return _windows_dpapi_transform(data, decrypt=True)


class ExcelSessionStore:
    def __init__(self, persistence_file: str | None = None):
        self._lock = threading.Lock()
        self._headers: dict[str, str] = {}
        self._configured_at: float | None = None
        self._expires_at: float | None = None
        self._persistence_file = persistence_file
        self._persistence_error = ""

    def configure(self, raw_headers: object, *, persist: bool = True) -> dict[str, object]:
        if not isinstance(raw_headers, dict):
            raise ValueError("headers must be a JSON object")

        headers: dict[str, str] = {}
        for raw_name, raw_value in raw_headers.items():
            if not isinstance(raw_name, str) or not isinstance(raw_value, str):
                continue
            name = raw_name.strip().lower()
            value = raw_value.strip()
            if name in _ALLOWED_CAPTURED_HEADERS and value and len(value) <= 32_768:
                headers[name] = value

        authorization = headers.get("authorization", "")
        if not authorization.lower().startswith("bearer ") or len(authorization.split(None, 1)) != 2:
            raise ValueError("a Bearer authorization header is required")

        chatgpt_account = headers.get("chatgpt-account-id")
        openai_account = headers.get("x-openai-account-id")
        if not chatgpt_account and not openai_account:
            raise ValueError("a ChatGPT account ID header is required")
        if chatgpt_account and openai_account and chatgpt_account != openai_account:
            raise ValueError("captured account ID headers do not match")
        account_id = chatgpt_account or openai_account
        headers["chatgpt-account-id"] = account_id
        headers["x-openai-account-id"] = account_id

        for name, value in _DEFAULT_CLIENT_HEADERS.items():
            headers.setdefault(name, value)

        expires_at = _decode_jwt_exp(authorization)
        now = time.time()
        if expires_at is not None and expires_at <= now:
            raise ValueError("the captured ChatGPT bearer token is already expired")

        with self._lock:
            self._headers = headers
            self._configured_at = now
            self._expires_at = expires_at
            self._persistence_error = ""
        if persist and self._persistence_file:
            try:
                self._save()
            except (OSError, RuntimeError) as exc:
                with self._lock:
                    self._persistence_error = str(exc)
        return self.status()

    def clear(self) -> dict[str, object]:
        with self._lock:
            self._headers = {}
            self._configured_at = None
            self._expires_at = None
            self._persistence_error = ""
        if self._persistence_file:
            try:
                os.remove(self._persistence_file)
            except FileNotFoundError:
                pass
            except OSError as exc:
                with self._lock:
                    self._persistence_error = str(exc)
        return self.status()

    def load(self) -> dict[str, object]:
        if not self._persistence_file or not os.path.isfile(self._persistence_file):
            return self.status()
        try:
            with open(self._persistence_file, "rb") as handle:
                protected = handle.read()
            payload = json.loads(_unprotect_windows_data(protected))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                raise ValueError("unsupported encrypted Excel session format")
            self.configure(payload.get("headers"), persist=False)
        except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            with self._lock:
                self._headers = {}
                self._configured_at = None
                self._expires_at = None
                self._persistence_error = str(exc)
        return self.status()

    def _save(self) -> None:
        if not self._persistence_file:
            return
        with self._lock:
            payload = {
                "version": 1,
                "headers": dict(self._headers),
            }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        protected = _protect_windows_data(raw)
        directory = os.path.dirname(self._persistence_file)
        os.makedirs(directory, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".excel-session-",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(protected)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._persistence_file)
        except Exception:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
            raise

    def status(self) -> dict[str, object]:
        with self._lock:
            configured = bool(self._headers)
            configured_at = self._configured_at
            expires_at = self._expires_at
            persistence_error = self._persistence_error
        expired = expires_at is not None and expires_at <= time.time()
        return {
            "configured": configured,
            "expired": expired,
            "configured_at": configured_at,
            "expires_at": expires_at,
            "model": MODEL_ID,
            "upstream_model": UPSTREAM_MODEL,
            "upstream_url": RESPONSES_URL,
            "storage": (
                "memory-and-windows-dpapi"
                if self._persistence_file and os.path.isfile(self._persistence_file)
                else "memory-only"
            ),
            "persistence_supported": sys.platform == "win32",
            "persisted": bool(
                self._persistence_file
                and os.path.isfile(self._persistence_file)
            ),
            "persistence": "windows-dpapi" if sys.platform == "win32" else "unavailable",
            "persistence_error": persistence_error,
        }

    def request_headers(self, *, stream: bool) -> dict[str, str]:
        with self._lock:
            headers = dict(self._headers)
            expires_at = self._expires_at
        if not headers:
            raise RuntimeError(
                "GPT Excel is not primed. Run the Excel session primer and send one prompt in the Excel add-in."
            )
        if expires_at is not None and expires_at <= time.time():
            raise RuntimeError(
                "The GPT Excel session has expired. Run the Excel session primer again."
            )
        headers.update(
            {
                "accept": "text/event-stream" if stream else "application/json",
                "content-type": "application/json",
                "origin": "https://bps.openai.com",
            }
        )
        return headers


def _message_item(role: str, text: str) -> dict:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


def _item_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _steered_tool_output(
    item: dict,
    call_names: dict[str, str],
    allowed_tools: dict[str, str],
) -> dict:
    call_id = item.get("call_id")
    name = call_names.get(call_id) if isinstance(call_id, str) else None
    output_text = _item_text(item.get("output"))
    if name == "update_plan":
        steering = PLAN_TOOL_OUTPUT_STEERING
        if "shell_command" in allowed_tools:
            steering += PLAN_TOOL_OUTPUT_SHELL_HINT
        prefix = f"{output_text.strip()}\n" if output_text.strip() else ""
        return {**item, "output": prefix + steering}
    if not output_text.strip() and isinstance(item.get("output"), (str, type(None))):
        # A blank body reads as a failed call and provokes retries; make
        # success explicit.
        return {**item, "output": "(tool call succeeded with no output)"}
    return item


def translate_input_items(
    raw_input: object,
    allowed_tools: dict[str, str] | None = None,
) -> list:
    """Map Codex Responses input items onto the Excel wire vocabulary.

    Tool-call history is replayed natively: live replays against Basispoints
    confirmed it accepts ``function_call`` / ``function_call_output`` items
    for tools that were never declared, and the model only maintains its tool
    state machine (instead of re-issuing the previous call) when it sees the
    standard item shapes.  ``update_plan`` outputs additionally carry a
    steering directive - without it the plan-happy upstream harness updates
    the plan forever instead of doing work.  Reasoning items that carry
    ``encrypted_content`` (which the upstream issues by default) are replayed
    unchanged for turn-to-turn continuity; bare reasoning items are dropped
    because with ``store: false`` the upstream rejects them.  The rendering is
    deterministic so replayed turns serialize identically on every request and
    keep the upstream prompt-cache prefix stable.
    """
    if isinstance(raw_input, str):
        return [_message_item("user", raw_input)]
    if not isinstance(raw_input, list):
        return []

    call_names: dict[str, str] = {}
    result: list = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type in {"function_call", "custom_tool_call"}:
            call_id = item.get("call_id")
            name = item.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                call_names[call_id] = name
            result.append(item)
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            result.append(_steered_tool_output(item, call_names, allowed_tools or {}))
            continue
        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                result.append(item)
            continue
        if item_type == "item_reference":
            continue
        result.append(item)
    return result


def _conversation_fingerprint(input_items: list) -> str:
    """Stable conversation identity for clients that send no cache key.

    The first input item is the root of the conversation and does not change
    as turns are appended, so hashing it keeps one identity per conversation
    without inventing a random one per request.
    """
    for item in input_items:
        if isinstance(item, dict):
            rendered = json.dumps(item, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return "anonymous"


def _cache_key(source: dict) -> str | None:
    for key in ("prompt_cache_key", "promptCacheKey", "session_id", "sessionId"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prepare_responses_body(source: dict) -> dict:
    """Translate a standard Responses request to the Excel add-in wire shape."""
    output: dict[str, object] = {
        "model": UPSTREAM_MODEL,
        "stream": bool(source.get("stream", False)),
        "store": False,
    }

    raw_input = source.get("input")
    input_items = translate_input_items(raw_input, client_tool_types(source))
    # Captured before the prologue is prepended: the injected instructions and
    # catalog are identical across conversations, so only the caller's own
    # first history item identifies this conversation.
    history_root = _conversation_fingerprint(input_items)

    # Prompt layout is chosen for the upstream prompt cache: everything that is
    # stable across a conversation leads, so each turn only re-bills the newly
    # appended history plus the short trailing reminder. Putting the ~3.5k-token
    # catalog last instead (the old layout, still reachable through
    # GHCP_EXCEL_CATALOG_AT_PROMPT_END) forced the cache prefix to end at the
    # catalog's first byte and re-billed it on every request.
    prologue: list = []
    instructions = source.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        prologue.append(_message_item("developer", instructions))
    catalog = _message_item("developer", _client_tool_protocol_instructions(source))
    if CATALOG_AT_PROMPT_END:
        input_items = prologue + input_items + [catalog]
    else:
        prologue.append(catalog)
        reminder = _client_tool_protocol_reminder(source)
        input_items = prologue + input_items
        if reminder:
            input_items.append(_message_item("developer", reminder))
    output["input"] = input_items

    cache_key = _cache_key(source)
    if cache_key and FORWARD_PROMPT_CACHE_KEY:
        output["prompt_cache_key"] = cache_key

    reasoning = source.get("reasoning")
    requested_effort = (
        reasoning.get("effort")
        if isinstance(reasoning, dict)
        else source.get("reasoning_effort")
    )
    normalized_effort = (
        requested_effort.strip().lower()
        if isinstance(requested_effort, str)
        else ""
    )
    # Direct Basispoints probes confirm only these two wire values. Unknown or
    # stale catalog values fall back to medium rather than producing a 422.
    output["reasoning_effort"] = (
        normalized_effort
        if normalized_effort in {"medium", "xhigh"}
        else "medium"
    )

    context_management = source.get("context_management")
    output["context_management"] = (
        context_management
        if isinstance(context_management, list)
        else [{"type": "compaction", "compact_threshold": 200_000}]
    )

    metadata: dict[str, str] = {}
    raw_metadata = source.get("metadata")
    if isinstance(raw_metadata, dict):
        for key, value in raw_metadata.items():
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                metadata[key[:64]] = str(value)[:512]
    tool_output_items = (
        sum(
            1
            for item in raw_input
            if isinstance(item, dict)
            and item.get("type") in {"function_call_output", "custom_tool_call_output"}
        )
        if isinstance(raw_input, list)
        else 0
    )
    iteration = str(tool_output_items + 1)
    metadata.setdefault("agent_iteration", iteration)
    # Identifiers are derived, never random: the same client request must
    # serialize to the same bytes every time. A conversation keeps one task
    # identity across turns, and a retried turn keeps its turn identity, so a
    # retry is recognisable as the same turn rather than as new work.
    conversation = cache_key or history_root
    metadata.setdefault(
        "task_id",
        str(uuid5(NAMESPACE_URL, f"ghcp-proxy/gpt-excel/{conversation}")),
    )
    metadata.setdefault(
        "turn_id",
        str(uuid5(NAMESPACE_URL, f"ghcp-proxy/gpt-excel/{conversation}/turn/{iteration}")),
    )
    output["metadata"] = metadata
    return output


excel_session_store = ExcelSessionStore(SESSION_FILE)
