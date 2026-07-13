"""Compatibility helpers for Codex multi-agent request shapes."""

from __future__ import annotations

import json
from collections.abc import Mapping


_SPAWN_AGENT_COMPAT_MARKER = "ghcp_proxy multi-agent compatibility"
_SPAWN_AGENT_COMPAT_NOTE = (
    "\n\n[ghcp_proxy multi-agent compatibility]\n"
    "`gpt-5.4-mini` is a valid model override. When `agent_type`, `model`, "
    "`reasoning_effort`, or `service_tier` is overridden, `fork_context` must "
    "be false or omitted. A full-history fork inherits those settings from "
    "the parent. Use only the agent id returned by a successful spawn when "
    "calling `wait_agent`."
)


def _non_empty_string(value) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _json_mapping(value) -> Mapping | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _metadata_mappings(body: dict | None) -> list[Mapping]:
    if not isinstance(body, dict):
        return []
    client_metadata = _json_mapping(body.get("client_metadata"))
    if client_metadata is None:
        return []

    mappings: list[Mapping] = [client_metadata]
    for key in (
        "x-codex-turn-metadata",
        "x_codex_turn_metadata",
        "codex_turn_metadata",
        "turn_metadata",
    ):
        parsed = _json_mapping(client_metadata.get(key))
        if parsed is not None:
            mappings.append(parsed)
    return mappings


def _source_is_subagent(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"subagent", "sub-agent", "agent"}
    if not isinstance(value, Mapping):
        return False
    for key in ("subagent", "sub-agent"):
        if key in value:
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                if _source_is_subagent(candidate):
                    return True
            elif isinstance(candidate, str):
                if candidate.strip().lower() in {
                    "subagent",
                    "sub-agent",
                    "agent",
                    "true",
                    "1",
                    "yes",
                }:
                    return True
            elif candidate is True:
                return True
    for key in ("type", "kind", "name"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip().lower() in {
            "subagent",
            "sub-agent",
        }:
            return True
    return False


def _nested_agent_mapping(mapping: Mapping) -> Mapping | None:
    for key in ("subagent", "sub-agent", "thread_spawn", "agent"):
        candidate = mapping.get(key)
        if isinstance(candidate, Mapping):
            return _nested_agent_mapping(candidate) or candidate
    for key in ("thread_source", "source"):
        source = mapping.get(key)
        if isinstance(source, Mapping) and _source_is_subagent(source):
            return _nested_agent_mapping(source) or source
    return None


def codex_subagent_identity(body: dict | None) -> str | None:
    """Return a stable identity for current Codex child-thread metadata.

    Older Codex builds sent ``x-openai-subagent``. Current builds describe the
    thread in ``client_metadata.x-codex-turn-metadata`` instead. The proxy
    removes client metadata before forwarding, so callers must derive the
    worker identity from the inbound body first.
    """
    mappings = _metadata_mappings(body)
    if not mappings:
        return None

    is_subagent = False
    nested: list[Mapping] = []
    for mapping in mappings:
        if _source_is_subagent(mapping.get("thread_source")) or _source_is_subagent(
            mapping.get("source")
        ):
            is_subagent = True
        agent_mapping = _nested_agent_mapping(mapping)
        if agent_mapping is not None:
            is_subagent = True
            nested.append(agent_mapping)

    if not is_subagent:
        return None

    identity_sources = [*reversed(nested), *reversed(mappings)]
    for key in (
        "agent_path",
        "agent_id",
        "thread_id",
        "session_id",
        "agent_nickname",
        "agent_role",
    ):
        for mapping in identity_sources:
            value = _non_empty_string(mapping.get(key))
            if value:
                return f"codex:{value}"
    return "codex:subagent"


def _patched_spawn_agent_tool(tool: dict) -> tuple[dict, bool]:
    if tool.get("name") != "spawn_agent":
        return tool, False
    description = tool.get("description")
    if not isinstance(description, str) or _SPAWN_AGENT_COMPAT_MARKER in description:
        return tool, False

    parameters = tool.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    # This is the Codex multi_agent_v1 contract that produced the captured
    # failure. Newer collaboration tools use a different fork shape and do not
    # accept model overrides, so leave those schemas untouched.
    if not isinstance(properties, Mapping) or not {
        "fork_context",
        "model",
    }.issubset(properties):
        return tool, False

    patched = dict(tool)
    patched["description"] = description.rstrip() + _SPAWN_AGENT_COMPAT_NOTE

    patched_parameters = dict(parameters)
    patched_properties = dict(properties)

    fork_context = properties.get("fork_context")
    if isinstance(fork_context, Mapping):
        patched_fork_context = dict(fork_context)
        existing = _non_empty_string(fork_context.get("description")) or ""
        patched_fork_context["description"] = (
            existing.rstrip()
            + " Full-history forks inherit agent type, model, reasoning effort, and "
            "service tier; set this false or omit it when overriding any of them."
        ).strip()
        patched_properties["fork_context"] = patched_fork_context

    model = properties.get("model")
    if isinstance(model, Mapping):
        patched_model = dict(model)
        existing = _non_empty_string(model.get("description")) or ""
        patched_model["description"] = (
            existing.rstrip() + " `gpt-5.4-mini` is a valid override."
        ).strip()
        patched_properties["model"] = patched_model

    patched_parameters["properties"] = patched_properties
    patched["parameters"] = patched_parameters
    return patched, True


def _patched_tool_node(node):
    if not isinstance(node, dict):
        return node, False

    patched, changed = _patched_spawn_agent_tool(node)
    nested_tools = patched.get("tools")
    if not isinstance(nested_tools, list):
        return patched, changed

    patched_nested = []
    nested_changed = False
    for nested in nested_tools:
        patched_child, child_changed = _patched_tool_node(nested)
        patched_nested.append(patched_child)
        nested_changed = nested_changed or child_changed
    if not nested_changed:
        return patched, changed
    if not changed:
        patched = dict(patched)
    patched["tools"] = patched_nested
    return patched, True


def normalize_codex_agent_tools(
    body: dict,
    *,
    diagnostics: list[dict] | None = None,
) -> dict:
    """Clarify current multi-agent constraints without mutating the input."""
    if not isinstance(body, dict) or not isinstance(body.get("tools"), list):
        return body

    patched_tools = []
    changed = False
    for tool in body["tools"]:
        patched_tool, tool_changed = _patched_tool_node(tool)
        patched_tools.append(patched_tool)
        changed = changed or tool_changed
    if not changed:
        return body

    patched_body = dict(body)
    patched_body["tools"] = patched_tools
    if isinstance(diagnostics, list):
        diagnostics.append(
            {
                "kind": "codex_multi_agent",
                "action": "clarify_spawn_agent_contract",
            }
        )
    return patched_body


__all__ = ["codex_subagent_identity", "normalize_codex_agent_tools"]
