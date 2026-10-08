"""Bidirectional Codex Responses / Responses Lite tool adapter.

Keep wire-format knowledge here, separate from the Copilot session lifecycle.
The SDK accepts flat JSON-schema tools; Codex also accepts namespaces, free-form
tools, incremental catalogs, and structured tool results. Runtime aliases must
never become client-visible tool identities.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


CALL_ID_PREFIX = "ghcpsdk_"
CATALOG_ITEM_TYPES = {"additional_tools", "tool_search_output"}
_VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_PLAINTEXT_COLLABORATION_TOOLS = {"spawn_agent", "send_message", "followup_task"}


def _is_collaboration_message_tool(namespace: str | None, name: str) -> bool:
    return namespace == "collaboration" and name in _PLAINTEXT_COLLABORATION_TOOLS


class AdapterValidationError(ValueError):
    """A caller-supplied tool item cannot be safely adapted to the SDK."""


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content, ensure_ascii=False)
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part.get("type") in {"input_image", "image_url"}:
                parts.append("[image supplied by client]")
            else:
                # Preserve unknown structured output instead of silently losing it.
                parts.append(json.dumps(part, ensure_ascii=False))
    return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class ToolMetadata:
    original_name: str
    tool_type: str
    namespace: str | None = None


@dataclass
class ToolCall:
    request_id: str
    name: str
    tool_type: str
    arguments: Any
    item_id: str = field(default_factory=lambda: f"fc_{uuid4().hex}")
    namespace: str | None = None


@dataclass(frozen=True)
class PendingToolResult:
    session_id: str
    request_id: str
    output: str
    tool_name: str = ""
    namespace: str | None = None
    contents: list[dict] | None = None

    def sdk_payload(self) -> dict:
        payload = {"textResultForLlm": self.output, "resultType": "success"}
        if self.contents is not None:
            payload["contents"] = self.contents
        return payload

    def image_attachments(self) -> list[dict]:
        return [
            {"type": "blob", "data": part["data"], "mimeType": part["mimeType"],
             "displayName": f"tool-result-{index}." + part["mimeType"].partition("/")[2]}
            for index, part in enumerate(self.contents or [], 1)
            if part["type"] == "image"
        ]


def encode_call_id(
    session_id: str, request_id: str, *, tool_name: str, tool_type: str,
    namespace: str | None = None,
) -> str:
    value = {"s": session_id, "r": request_id, "n": tool_name, "t": tool_type}
    if namespace:
        value["ns"] = namespace
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return CALL_ID_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_call_id(call_id: Any) -> dict[str, str] | None:
    if not isinstance(call_id, str) or not call_id.startswith(CALL_ID_PREFIX):
        return None
    encoded = call_id[len(CALL_ID_PREFIX):]
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if not all(isinstance(value.get(key), str) and value[key] for key in ("s", "r", "n", "t")):
        return None
    if value["t"] not in {"function", "custom"}:
        return None
    if "ns" in value and (not isinstance(value["ns"], str) or not value["ns"]):
        return None
    return value


def _safe_name(name: str, used: set[str]) -> str:
    candidate = name if _VALID_TOOL_NAME.fullmatch(name) else re.sub(r"[^A-Za-z0-9_-]", "_", name)
    base = candidate = candidate or "tool"
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _flatten_tools(specs: Any, namespace: str | None = None, *, defer_loading=None, context=""):
    for spec in specs if isinstance(specs, list) else []:
        if not isinstance(spec, dict):
            continue
        if spec.get("type") == "namespace":
            name = spec.get("name")
            inherited = spec.get("defer_loading", defer_loading)
            description = spec.get("description")
            child_context = context
            if isinstance(description, str) and description:
                child_context += f"\nNamespace {name}: {description}"
            yield from _flatten_tools(
                spec.get("tools"), name if isinstance(name, str) and name else namespace,
                defer_loading=inherited if isinstance(inherited, bool) else defer_loading,
                context=child_context,
            )
        else:
            spec = copy.deepcopy(spec)
            if defer_loading is not None and "defer_loading" not in spec:
                spec["defer_loading"] = defer_loading
            if context:
                description = spec.get("description")
                description = description if isinstance(description, str) else ""
                spec["description"] = (context.strip() + "\n" + description).rstrip()
            yield (None if namespace == "functions" else namespace), spec


def _result_contents(output: Any) -> list[dict] | None:
    if not isinstance(output, list):
        return None
    contents = []
    for part in output:
        if isinstance(part, dict) and part.get("type") in {"input_image", "image_url"}:
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url.startswith("data:"):
                raise AdapterValidationError("Copilot SDK tool-result images require an inline base64 data URL")
            header, separator, data = url.partition(",")
            mime_type = header[5:].split(";", 1)[0]
            if not separator or ";base64" not in header or not mime_type.startswith("image/"):
                raise AdapterValidationError("Invalid inline tool-result image")
            try:
                decoded = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise AdapterValidationError("Invalid base64 tool-result image") from exc
            if not decoded:
                raise AdapterValidationError("Empty tool-result image")
            contents.append({"type": "image", "data": data, "mimeType": mime_type})
        else:
            text = text_from_content([part])
            if text:
                contents.append({"type": "text", "text": text})
    return contents


class CodexSdkAdapter:
    """One request's callable catalog and the shared round-trip wire rules."""

    def __init__(self) -> None:
        self.names: dict[str, ToolMetadata] = {}
        self.tool_options: list[dict] = []

    @classmethod
    def from_request(cls, body: dict) -> CodexSdkAdapter:
        adapter = cls()
        if body.get("tool_choice") == "none":
            return adapter
        current_tools = list(_flatten_tools(body.get("tools")))
        declarations = list(current_tools)
        items = body.get("input")
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("type") in CATALOG_ITEM_TYPES:
                declarations.extend(_flatten_tools(item.get("tools")))
        # Lite replays earlier catalogs. The latest declaration is authoritative,
        # but keep the first insertion's position so aliases remain stable.
        catalog = {}
        for namespace, spec in declarations:
            name = spec.get("name")
            if spec.get("type") in {"function", "custom"} and isinstance(name, str) and name:
                catalog[namespace, name] = spec
        # A standard top-level catalog describes this request, not a historical
        # Lite update. Prefer it when a client supplies both representations.
        for namespace, spec in current_tools:
            identity = namespace, spec.get("name")
            if spec.get("type") in {"function", "custom"} and isinstance(identity[1], str) and identity in catalog:
                catalog[identity] = spec
        used: set[str] = set()
        for (namespace, name), spec in catalog.items():
            tool_type = spec["type"]
            safe_name = _safe_name(f"{namespace or ''}{name}", used)
            description = spec.get("description")
            description = description if isinstance(description, str) else ""
            parameters = spec.get("parameters")
            if tool_type == "custom":
                # Avoid SDK built-in custom-tool collisions (notably apply_patch).
                safe_name = _safe_name(f"ghcp_custom_{safe_name}", used)
                description = (description.rstrip() +
                    "\nReturn this custom tool's complete raw input in the JSON `input` field.").strip()
                if isinstance(spec.get("format"), dict):
                    description += "\nThe raw input must obey this format: " + json.dumps(
                        spec["format"], ensure_ascii=False, separators=(",", ":"),
                    )
                parameters = {"type": "object", "properties": {"input": {"type": "string"}},
                              "required": ["input"], "additionalProperties": False}
            elif not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            if tool_type == "function" and _is_collaboration_message_tool(namespace, name):
                # Codex v2 annotates message strings with encrypted:true. CAPI
                # honors it, but the SDK prompt renderer cannot forward opaque
                # agent_message blocks. Use Codex's explicit plaintext path:
                # suppress server encryption here and mark the returned call.
                properties = parameters.get("properties")
                message_schema = properties.get("message") if isinstance(properties, dict) else None
                if isinstance(message_schema, dict):
                    message_schema.pop("encrypted", None)
            adapter.names[safe_name] = ToolMetadata(name, tool_type, namespace)
            adapter.tool_options.append({
                "name": safe_name, "description": description, "parameters": parameters,
                "overrides_built_in_tool": tool_type == "function", "skip_permission": True,
                "defer": "auto" if spec.get("defer_loading") is True or (
                    spec.get("defer_loading") is not False and (namespace is not None or name.startswith("mcp__"))
                ) else "never",
            })
        return adapter

    def tool_call(self, data: Any) -> ToolCall:
        safe_name = str(getattr(data, "tool_name", "tool"))
        metadata = self.names.get(safe_name, ToolMetadata(safe_name, "function"))
        return ToolCall(
            request_id=str(getattr(data, "request_id")), name=metadata.original_name,
            tool_type=metadata.tool_type, arguments=getattr(data, "arguments", {}),
            item_id=f"{'ctc' if metadata.tool_type == 'custom' else 'fc'}_{uuid4().hex}",
            namespace=metadata.namespace,
        )

    @staticmethod
    def arguments_text(call: ToolCall) -> str:
        arguments = call.arguments
        if call.tool_type == "custom":
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError:
                    return arguments
                if not isinstance(decoded, dict):
                    return arguments
                arguments = decoded
            if isinstance(arguments, dict):
                if isinstance(arguments.get("input"), str):
                    return arguments["input"]
                if call.name == "apply_patch" and isinstance(arguments.get("patch"), str):
                    return arguments["patch"]
            return text_from_content(arguments)
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
                return arguments
            except json.JSONDecodeError:
                return json.dumps({"input": arguments}, ensure_ascii=False)
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def tool_item(session_id: str, call: ToolCall, *, completed: bool = True) -> dict:
        item = {
            "type": "custom_tool_call" if call.tool_type == "custom" else "function_call",
            "id": call.item_id,
            "call_id": encode_call_id(session_id, call.request_id, tool_name=call.name,
                                      tool_type=call.tool_type, namespace=call.namespace),
            "name": call.name, "status": "completed" if completed else "in_progress",
        }
        if call.namespace:
            item["namespace"] = call.namespace
        if call.tool_type == "function" and _is_collaboration_message_tool(call.namespace, call.name):
            # [] is meaningful: Codex ToolCall::direct_source selects
            # DirectPlaintextMessage only for an explicitly empty list.
            # Omitting this field labels even ordinary JSON text as encrypted.
            item["encrypted_function_args"] = []
        item["input" if call.tool_type == "custom" else "arguments"] = CodexSdkAdapter.arguments_text(call)
        return item

    @staticmethod
    def render_tool_call(item: dict) -> str:
        name = item.get("name", "")
        namespace = item.get("namespace")
        if namespace and namespace != "functions":
            name = f"{namespace}.{name}"
        payload = item.get("input") if item.get("type") == "custom_tool_call" else item.get("arguments")
        return f"Assistant tool call {name}: {text_from_content(payload)}"

    @staticmethod
    def is_caller_message(item: Any) -> bool:
        return isinstance(item, str) or (isinstance(item, dict) and (
            item.get("type") == "agent_message" or (
                item.get("type") in {None, "message"}
                and item.get("role") in {"user", "developer", "system"}
            )
        ))

    @staticmethod
    def validate_new_agent_message(value: Any) -> None:
        if not isinstance(value, list):
            return
        for item in reversed(value):
            if isinstance(item, dict) and item.get("type") in CATALOG_ITEM_TYPES:
                continue
            if isinstance(item, dict) and item.get("type") == "agent_message":
                content = item.get("content")
                if isinstance(content, list) and any(
                    isinstance(part, dict) and part.get("type") == "encrypted_content"
                    for part in content
                ):
                    raise AdapterValidationError(
                        "Copilot SDK cannot read an encrypted agent-message payload. "
                        "Resend the assignment using the SDK plaintext collaboration path; "
                        "existing ciphertext cannot be recovered by this adapter."
                    )
            break

    @staticmethod
    def continuation(value: Any) -> tuple[str, list[PendingToolResult]] | None:
        if not isinstance(value, list):
            return None
        trailing = []
        for item in reversed(value):
            if CodexSdkAdapter.is_caller_message(item) or (
                isinstance(item, dict) and item.get("type") in CATALOG_ITEM_TYPES
            ):
                continue
            if not isinstance(item, dict) or item.get("type") not in {"function_call_output", "custom_tool_call_output"}:
                break
            decoded = decode_call_id(item.get("call_id"))
            if decoded is None:
                return None
            expected_type = "custom_tool_call_output" if decoded["t"] == "custom" else "function_call_output"
            if item["type"] != expected_type:
                raise AdapterValidationError("Tool output type does not match the pending Copilot SDK call")
            namespace = decoded.get("ns")
            output_namespace = item.get("namespace")
            if namespace is not None and output_namespace is not None and output_namespace != namespace:
                raise AdapterValidationError("Tool output namespace does not match the pending Copilot SDK call")
            if item.get("name") is not None and item["name"] != decoded["n"]:
                raise AdapterValidationError("Tool output name does not match the pending Copilot SDK call")
            trailing.append(PendingToolResult(
                decoded["s"], decoded["r"], text_from_content(item.get("output")), decoded["n"],
                namespace=namespace, contents=_result_contents(item.get("output")),
            ))
        if not trailing:
            return None
        trailing.reverse()
        session_id = trailing[0].session_id
        if any(result.session_id != session_id for result in trailing):
            return None
        return session_id, trailing
