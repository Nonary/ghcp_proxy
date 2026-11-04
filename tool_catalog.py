"""Compact schema notation for text-only tool relays.

This is prompt presentation, not a replacement for JSON Schema validation.
Call validation always uses the original caller schema. Unusual schemas stay
in JSON; only ordinary object/array/type scaffolding is rendered as notation.
"""

from __future__ import annotations

import json


SCHEMA_NOTATION = (
    "Parameter notation: {\"key\":type} requires key; {\"key\"?:type} makes it "
    "optional. [...] is an array. ... allows extra object keys; ...:schema "
    "constrains them. Without ... extra keys are forbidden. JSON annotations "
    "after a type retain its constraints/descriptions. In annotations, allOf, anyOf, "
    "oneOf, $defs, and definitions contain schemas in the same notation; their "
    "JSON Schema meanings and $ref paths are unchanged. Other schemas use full JSON. "
    "Only the notation is compacted: use the exact parameter names and obey every constraint."
)


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _annotations(values: dict) -> str:
    fields = []
    for key, value in values.items():
        if key in {"allOf", "anyOf", "oneOf"} and isinstance(value, list):
            rendered = "[" + ",".join(compact_schema(item) for item in value) + "]"
        elif key in {"$defs", "definitions"} and isinstance(value, dict):
            rendered = "{" + ",".join(
                f"{_json(name)}:{compact_schema(item)}"
                for name, item in value.items()
            ) + "}"
        else:
            rendered = _json(value)
        fields.append(f"{_json(key)}:{rendered}")
    return "{" + ",".join(fields) + "}"


def compact_schema(schema: object) -> str:
    """Render ordinary schema scaffolding without dropping its constraints."""
    if not isinstance(schema, dict):
        return _json(schema)
    if any(key in schema for key in {
        "$dynamicRef",
        "not", "if", "then", "else", "patternProperties", "dependentSchemas",
        "prefixItems", "unevaluatedProperties", "unevaluatedItems",
    }):
        return _json(schema)
    remaining = dict(schema)
    kind = remaining.pop("type", None)
    if kind == "object":
        properties = remaining.get("properties", {})
        required = remaining.get("required", [])
        additional = remaining.get("additionalProperties", True)
        if (
            not isinstance(properties, dict)
            or not isinstance(required, list)
            or any(not isinstance(key, str) or key not in properties for key in required)
            or not isinstance(additional, (bool, dict))
        ):
            return _json(schema)
        remaining.pop("properties", None)
        remaining.pop("required", None)
        remaining.pop("additionalProperties", None)
        fields = [
            f'{_json(key)}{"" if key in required else "?"}:{compact_schema(value)}'
            for key, value in properties.items()
        ]
        if additional is not False:
            fields.append("..." + (":" + compact_schema(additional) if isinstance(additional, dict) else ""))
        rendered = "{" + ",".join(fields) + "}"
    elif kind == "array" and isinstance(remaining.get("items"), (dict, bool)):
        rendered = "[" + compact_schema(remaining.pop("items")) + "]"
    elif isinstance(kind, str) and kind in {"string", "number", "integer", "boolean", "null"}:
        rendered = kind
    elif "type" not in schema and any(key in remaining for key in {
        "$ref", "$defs", "definitions", "allOf", "anyOf", "oneOf",
    }):
        return _annotations(remaining)
    else:
        # Preserve type unions, legacy tuple arrays, and unknown schema
        # dialects verbatim rather than guessing a lossy equivalent.
        return _json(schema)
    if remaining:
        rendered += " " + _annotations(remaining)
    return rendered
