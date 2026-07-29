"""OpenAI Excel add-in upstream support.

The official Excel add-in uses a ChatGPT session rather than the GitHub
Copilot token used by the rest of this proxy.  Keep that credential isolated
in memory and expose only non-secret status information.
"""

from __future__ import annotations

import base64
import copy
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
from collections import OrderedDict
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
# Kept only to replay calls produced by proxy versions that used the old text
# marker protocol. New calls travel through Basispoints' declared
# ``run_officejs`` function and are intercepted before any Office code runs.
CLIENT_TOOL_RELAY_PREFIX = "codex_client__"
CLIENT_MARKER_CALL_ID_PREFIX = "call_ghcp_excel_marker_"
NATIVE_FALLBACK_CALL_ID_PREFIX = "call_ghcp_excel_native_"
CLIENT_TOOL_TRANSPORT_NAME = "run_officejs"
TOOLS_VERSION_METADATA_KEY = "bps_tools_version_id"
_TOOLS_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
_NATIVE_CALL_CACHE_LIMIT = 512
_native_call_cache_lock = threading.Lock()
_native_call_cache: OrderedDict[str, dict] = OrderedDict()
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
# the client-tool transport protocol. See _client_tool_protocol_reminder.
CATALOG_AT_PROMPT_END = os.environ.get(
    "GHCP_EXCEL_CATALOG_AT_PROMPT_END", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SESSION_FILE = (
    os.path.join(user_state_dir(), "excel-session.dpapi")
    if sys.platform == "win32"
    else None
)

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


def relay_tool_name(name: str) -> str:
    """Return the legacy non-colliding marker name used before run_officejs."""
    return CLIENT_TOOL_RELAY_PREFIX + name


def _original_client_tool_name(
    name: object,
    allowed_tools: dict[str, str],
) -> str | None:
    if not isinstance(name, str):
        return None
    if name.startswith(CLIENT_TOOL_RELAY_PREFIX):
        candidate = name[len(CLIENT_TOOL_RELAY_PREFIX) :]
        return candidate if candidate in allowed_tools else None
    # Accept the old, unprefixed marker format for in-flight responses. Native
    # Basispoints calls also arrive unprefixed; schema validation below decides
    # whether one can safely stand in for a same-named client tool.
    return name if name in allowed_tools else None


def _remember_native_call(item: dict) -> None:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return
    with _native_call_cache_lock:
        _native_call_cache[call_id] = copy.deepcopy(item)
        _native_call_cache.move_to_end(call_id)
        while len(_native_call_cache) > _NATIVE_CALL_CACHE_LIMIT:
            _native_call_cache.popitem(last=False)


def _remembered_native_call(call_id: object) -> dict | None:
    if not isinstance(call_id, str) or not call_id:
        return None
    with _native_call_cache_lock:
        item = _native_call_cache.get(call_id)
        if item is None:
            return None
        _native_call_cache.move_to_end(call_id)
        return copy.deepcopy(item)


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


def _transport_envelope(native: dict) -> dict | None:
    if (
        native.get("type") != "function_call"
        or native.get("name") != CLIENT_TOOL_TRANSPORT_NAME
    ):
        return None
    raw_arguments = native.get("arguments")
    if not isinstance(raw_arguments, str):
        return None
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    code = arguments.get("code")
    if not isinstance(code, str):
        return None
    try:
        envelope = json.loads(code)
    except json.JSONDecodeError:
        # Models occasionally wrap the requested JSON in a code fence or a
        # one-line assignment despite the exact-format instruction. Decode the
        # first complete JSON object without ever evaluating the surrounding
        # text as JavaScript.
        decoder = json.JSONDecoder()
        envelope = None
        for index, character in enumerate(code):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(code[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                envelope = candidate
                break
    return envelope if isinstance(envelope, dict) else None


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


def _restore_native_function_arguments(name: str, arguments: object) -> object:
    """Restore the Basispoints schema after a native call visits Codex."""
    if name != "update_plan":
        return arguments
    parsed = arguments
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return arguments
    if not isinstance(parsed, dict) or not isinstance(parsed.get("plan"), list):
        return arguments

    native_plan: list[dict[str, str]] = []
    for index, item in enumerate(parsed["plan"]):
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        status = item.get("status")
        if not isinstance(step, str) or not isinstance(status, str):
            continue
        native_plan.append(
            {
                "id": f"step{index + 1}",
                "description": step,
                "status": status,
                "result": "",
            }
        )
    explanation = parsed.get("explanation")
    native = {
        "summary": (
            explanation
            if isinstance(explanation, str) and explanation
            else "Update task plan"
        ),
        "plan": native_plan,
    }
    return json.dumps(native, separators=(",", ":"), ensure_ascii=False)


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
    allowed_tools = client_tool_types(source)
    envelope = _transport_envelope(native)
    name = (
        _original_client_tool_name(envelope.get("name"), allowed_tools)
        if envelope is not None
        else _original_client_tool_name(native.get("name"), allowed_tools)
    )
    if name is None or name not in specs:
        return None
    spec = specs[name]
    expected_type = str(spec.get("type") or "").strip().lower()
    if expected_type == "function":
        if envelope is not None:
            arguments = envelope.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
        else:
            if native.get("type") != "function_call":
                return None
            raw_arguments = native.get("arguments")
            if not isinstance(raw_arguments, str):
                return None
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        if envelope is None:
            arguments = _normalize_native_function_arguments(name, arguments)
        if not _value_matches_schema(arguments, spec.get("parameters")):
            return None
        native_call_id = native.get("call_id")
        call_id = (
            native_call_id
            if isinstance(native_call_id, str) and native_call_id
            else f"{NATIVE_FALLBACK_CALL_ID_PREFIX}{uuid4().hex}"
        )
        native_item_id = native.get("id")
        _remember_native_call(native)
        return {
            "type": "function_call",
            "id": (
                native_item_id
                if isinstance(native_item_id, str) and native_item_id
                else f"fc_{call_id}"
            ),
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(
                arguments,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        }
    if expected_type == "custom":
        custom_input = (
            envelope.get("input")
            if envelope is not None
            else native.get("input")
        )
        if envelope is None and native.get("type") != "custom_tool_call":
            return None
        if not isinstance(custom_input, str):
            return None
        native_call_id = native.get("call_id")
        call_id = (
            native_call_id
            if isinstance(native_call_id, str) and native_call_id
            else f"{NATIVE_FALLBACK_CALL_ID_PREFIX}{uuid4().hex}"
        )
        native_item_id = native.get("id")
        _remember_native_call(native)
        return {
            "type": "custom_tool_call",
            "id": (
                native_item_id
                if (
                    envelope is None
                    and isinstance(native_item_id, str)
                    and native_item_id
                )
                else f"ctc_{call_id}"
            ),
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
        "by the live Excel workbook. The native run_officejs function is a "
        "transport endpoint owned by this proxy for this request. The proxy "
        "intercepts it before execution, so it never runs Office code or changes "
        "the workbook. Every client tool in the JSON catalog is available through "
        "that transport. Other native server-injected Excel, Office, connector, "
        "workbook, list_skills, and web-search tools are unavailable. "
        "Never claim shell, filesystem, or workspace access is unavailable when the "
        "catalog contains a suitable tool. For repository inspection, invoke "
        "shell_command through run_officejs when shell_command is present. "
        "To invoke a function client tool, call native run_officejs with: summary "
        "and extended_summary describing the action; code containing exactly one "
        'compact JSON object {"name":"TOOL_NAME","arguments":{...}}; '
        "destructive=false; and references=[]. To invoke a custom client tool, "
        'put {"name":"TOOL_NAME","input":"RAW_INPUT"} in code instead. Do not put '
        "JavaScript or OfficeJS in code. TOOL_NAME and its payload must follow the "
        "catalog exactly. The proxy converts this native function call into the "
        "real client tool call, then replays the original run_officejs identity "
        "with the client tool result on the next request. Interpret that result as "
        "the named client tool's output. Native update_plan may be used normally "
        "when update_plan is in the catalog, but after it succeeds take the next "
        "substantive action through run_officejs. Do not stop at commentary saying "
        "you will take an action: make the tool call in the same response. Never "
        "repeat a tool request whose output is already present. Available client "
        "tools:\n"
        + catalog_json
        + "\nRemember: client tool use must be a native run_officejs function call "
        "whose code field is the JSON transport envelope. It is not Office code."
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
        "Reminder: run_officejs is the proxy-owned client-tool transport and "
        "never executes Office code for this request. To use a client tool, make "
        "the native run_officejs call now with the JSON transport envelope in its "
        "code field, following the catalog above. Do not merely say you will act "
        "or that access is unavailable. Client tools: "
        + ", ".join(sorted(allowed_tools))
        + ". Other native tools are unavailable."
    )
    if "shell_command" in allowed_tools:
        reminder += " For repository inspection transport shell_command."
    if "update_plan" in allowed_tools:
        reminder += (
            " Native update_plan is allowed for progress; after its result, "
            "take the next substantive action through run_officejs."
        )
    return reminder


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
    marker_name = marker.get("name")
    if not isinstance(marker_name, str):
        return None
    name = _original_client_tool_name(marker_name.strip(), allowed_tools)
    if name is None:
        return None
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
        call_id = f"{CLIENT_MARKER_CALL_ID_PREFIX}{uuid4().hex}"
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
        call_id = f"{CLIENT_MARKER_CALL_ID_PREFIX}{uuid4().hex}"
        return {
            "type": "custom_tool_call",
            "id": f"ctc_{call_id}",
            "call_id": call_id,
            "name": name,
            "input": custom_input,
        }
    return None


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
        self._tools_version_id: str | None = None
        self._configured_at: float | None = None
        self._expires_at: float | None = None
        self._persistence_file = persistence_file
        self._persistence_error = ""

    def configure(
        self,
        raw_headers: object,
        *,
        tools_version_id: object = None,
        persist: bool = True,
    ) -> dict[str, object]:
        if not isinstance(raw_headers, dict):
            raise ValueError("headers must be a JSON object")
        if tools_version_id is not None and (
            not isinstance(tools_version_id, str)
            or not _TOOLS_VERSION_PATTERN.fullmatch(tools_version_id.strip())
        ):
            raise ValueError("tools_version_id is not a valid Basispoints version ID")
        normalized_tools_version_id = (
            tools_version_id.strip() if isinstance(tools_version_id, str) else None
        )

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
            self._tools_version_id = normalized_tools_version_id
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
            self._tools_version_id = None
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
            self.configure(
                payload.get("headers"),
                tools_version_id=payload.get("tools_version_id"),
                persist=False,
            )
        except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            with self._lock:
                self._headers = {}
                self._tools_version_id = None
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
                "tools_version_id": self._tools_version_id,
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
            tools_version_id = self._tools_version_id
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
            "tools_version_id": tools_version_id,
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

    def tools_version_id(self) -> str | None:
        with self._lock:
            return self._tools_version_id

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


def _normalized_tool_output(
    item: dict,
    call_origins: dict[str, str],
) -> dict:
    call_id = item.get("call_id")
    origin = call_origins.get(call_id) if isinstance(call_id, str) else None
    if origin == "update_plan":
        # The Basispoints update_plan executor returns this object. Codex's
        # client-side status tool instead returns the display string
        # "Plan updated"; replaying that string leaves the server-native tool
        # state unresolved and makes the model plan again.
        return {**item, "output": '{"status":"ok"}'}
    normalized = (
        {**item, "type": "function_call_output"}
        if (
            origin == CLIENT_TOOL_TRANSPORT_NAME
            and item.get("type") == "custom_tool_call_output"
        )
        else item
    )
    output_text = _item_text(normalized.get("output"))
    if not output_text.strip() and isinstance(
        normalized.get("output"), (str, type(None))
    ):
        # A blank body reads as a failed call and provokes retries; make
        # success explicit.
        return {**normalized, "output": "(tool call succeeded with no output)"}
    return normalized


def _fallback_transport_call(item: dict) -> dict:
    """Rebuild a transport call if the proxy restarted between call and result."""
    name = str(item.get("name") or "")
    if item.get("type") == "custom_tool_call":
        envelope: dict[str, object] = {
            "name": name,
            "input": item.get("input") if isinstance(item.get("input"), str) else "",
        }
    else:
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        envelope = {"name": name, "arguments": arguments}
    call_id = str(item.get("call_id") or f"{NATIVE_FALLBACK_CALL_ID_PREFIX}{uuid4().hex}")
    native_arguments = {
        "summary": f"Run client tool {name}",
        "extended_summary": f"Relay {name} through the external Codex client",
        "code": json.dumps(envelope, separators=(",", ":"), ensure_ascii=False),
        "destructive": False,
        "references": [],
    }
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": CLIENT_TOOL_TRANSPORT_NAME,
        "arguments": json.dumps(
            native_arguments,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "status": "completed",
    }


def _strip_client_only_item_metadata(item: dict) -> dict:
    """Remove caller transport metadata that destabilizes prompt caching.

    Codex stamps every input item with an
    ``internal_chat_message_metadata_passthrough.turn_id``. The value is not
    part of the Responses item vocabulary Basispoints needs, and identical
    developer/environment messages receive a different value in every new
    conversation. Forwarding it therefore makes byte-identical prompt content
    diverge immediately after the cached tool catalog.

    Native calls remembered from the Basispoints response bypass this helper
    and are replayed exactly, preserving the server item identity required by
    encrypted reasoning.
    """
    if "internal_chat_message_metadata_passthrough" not in item:
        return item
    sanitized = dict(item)
    sanitized.pop("internal_chat_message_metadata_passthrough", None)
    return sanitized


def translate_input_items(
    raw_input: object,
    allowed_tools: dict[str, str] | None = None,
) -> list:
    """Map Codex Responses input items onto the Excel wire vocabulary.

    Tool-call history keeps the native Responses items that Basispoints needs
    for its state machine. New client calls use Basispoints' declared
    ``run_officejs`` function as an intercepted transport. The original native
    item is retained in a bounded cache because Codex drops its server item ID
    when submitting the tool result; replaying only the name and call_id makes
    encrypted reasoning treat the result as unrelated. Native update_plan
    results are translated from Codex's display text to the ``{"status":"ok"}``
    object returned by Excel's real executor. Calls created by older proxy
    versions retain their namespaced marker representation.
    Reasoning items that carry ``encrypted_content`` (which the upstream issues
    by default) are replayed unchanged for turn-to-turn continuity; bare
    reasoning items are dropped because with ``store: false`` the upstream
    rejects them. The rendering is deterministic so replayed turns serialize
    identically on every request and keep the upstream prompt-cache prefix
    stable.
    """
    if isinstance(raw_input, str):
        return [_message_item("user", raw_input)]
    if not isinstance(raw_input, list):
        return []

    call_origins: dict[str, str] = {}
    result: list = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item = _strip_client_only_item_metadata(item)
        item_type = str(item.get("type") or "").strip().lower()
        if item_type in {"function_call", "custom_tool_call"}:
            name = item.get("name")
            call_id = item.get("call_id")
            marker_relay = (
                isinstance(call_id, str)
                and call_id.startswith(CLIENT_MARKER_CALL_ID_PREFIX)
            )
            remembered = _remembered_native_call(call_id)
            if remembered is not None:
                native_name = remembered.get("name")
                if isinstance(call_id, str) and isinstance(native_name, str):
                    call_origins[call_id] = native_name
                result.append(remembered)
            elif isinstance(name, str) and name and marker_relay:
                upstream_name = relay_tool_name(name)
                if isinstance(call_id, str):
                    call_origins[call_id] = upstream_name
                result.append({**item, "name": upstream_name})
            elif isinstance(name, str) and name:
                if name == "update_plan":
                    if isinstance(call_id, str):
                        call_origins[call_id] = name
                    result.append(
                        {
                            **item,
                            "arguments": _restore_native_function_arguments(
                                name,
                                item.get("arguments"),
                            ),
                        }
                    )
                else:
                    fallback = _fallback_transport_call(item)
                    if isinstance(call_id, str):
                        call_origins[call_id] = CLIENT_TOOL_TRANSPORT_NAME
                    result.append(fallback)
            else:
                result.append(item)
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            result.append(_normalized_tool_output(item, call_origins))
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


def _agent_turn_state(raw_input: object) -> tuple[str, str]:
    """Return a stable user-turn fingerprint and Excel agent iteration.

    The Excel add-in holds ``turn_id`` constant while it executes any number of
    tools for one user message. Only ``agent_iteration`` advances. Treating
    every tool output as a new turn makes Basispoints discard the prior plan
    state and start planning again.
    """
    if isinstance(raw_input, str):
        rendered = json.dumps(raw_input, ensure_ascii=False)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest(), "1"
    if not isinstance(raw_input, list):
        return "anonymous", "1"

    last_user_index = -1
    for index, item in enumerate(raw_input):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "user":
            last_user_index = index

    turn_prefix = (
        raw_input[: last_user_index + 1]
        if last_user_index >= 0
        else raw_input[:1]
    )
    rendered = json.dumps(
        turn_prefix,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    iteration_outputs = sum(
        1
        for item in raw_input[last_user_index + 1 :]
        if isinstance(item, dict)
        and item.get("type")
        in {"function_call_output", "custom_tool_call_output"}
    )
    return fingerprint, str(iteration_outputs + 1)


def prepare_responses_body(
    source: dict,
    *,
    tools_version_id: str | None = None,
) -> dict:
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
    # first history item identifies this conversation. Use the raw items here
    # so client turn metadata can still disambiguate otherwise identical
    # no-cache-key conversations without leaking into the cached prompt.
    history_root = _conversation_fingerprint(
        raw_input if isinstance(raw_input, list) else input_items
    )

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
    turn_fingerprint, iteration = _agent_turn_state(raw_input)
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
        str(
            uuid5(
                NAMESPACE_URL,
                f"ghcp-proxy/gpt-excel/{conversation}/turn/{turn_fingerprint}",
            )
        ),
    )
    if tools_version_id and _TOOLS_VERSION_PATTERN.fullmatch(tools_version_id):
        metadata[TOOLS_VERSION_METADATA_KEY] = tools_version_id
    output["metadata"] = metadata
    return output


excel_session_store = ExcelSessionStore(SESSION_FILE)
