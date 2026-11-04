"""Inspect actual hostless SDK tool-search wiring without model calls or secrets.

Run with the repo virtualenv. --archive can compare a locally captured Excel
request with the compact relay catalog; only counts, never prompts, are printed.
Optional tiktoken supplies local estimates, not authoritative upstream billing.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from copilot import CopilotClient
from copilot.copilot_request_handler import CopilotRequestHandler

import copilot_sdk_upstream as sdk
import excel_upstream


class SyntheticUpstream(CopilotRequestHandler):
    def __init__(self):
        self.bodies: list[dict] = []

    async def send_request(self, request, ctx):
        # No discovery, telemetry, credentials, or real model traffic escapes.
        if request.url.host != "127.0.0.1" or request.url.path != "/responses":
            raise AssertionError(f"Unexpected SDK request: {request.url.host}{request.url.path}")
        body = json.loads(await request.aread())
        self.bodies.append(body)
        response = {
            "id": "resp_synthetic", "object": "response", "created_at": 1780000000,
            "model": body["model"], "status": "completed", "output": [{
                "type": "message", "id": "msg_synthetic", "role": "assistant",
                "status": "completed", "content": [{
                    "type": "output_text", "text": "Done.", "annotations": [],
                }],
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        events = [{"type": "response.created", "response": {
            **response, "status": "in_progress", "output": [],
        }}]
        for suffix in ("added", "done"):
            events.append({"type": f"response.output_item.{suffix}", "output_index": 0, "item": response["output"][0]})
        events.append({"type": "response.completed", "response": response})
        wire = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire, request=request)


async def verify(source: dict):
    registration = sdk.build_tool_registration(source)
    options = sdk._session_options({**source, "input": "Hello", "stream": True}, registration)
    options["model"] = "gpt-5.6-sol"
    options["infinite_sessions"] = {"enabled": False}
    options["provider"] = {
        "type": "openai", "wire_api": "responses", "base_url": "http://127.0.0.1:1",
        "api_key": "synthetic-not-a-credential",
    }
    capture = SyntheticUpstream()
    with tempfile.TemporaryDirectory(prefix="ghcp-tool-budget-") as root:
        client = CopilotClient(
            github_token="synthetic-not-a-credential", use_logged_in_user=False,
            mode="empty", base_directory=root, request_handler=capture,
        )
        try:
            await client.start()
            session = await client.create_session(**options)
            await session.send_and_wait(prompt="Hello", timeout=30)
            body = capture.bodies[-1]
            tools = body.get("tools", [])
            by_name = {tool.get("name"): tool for tool in tools if tool.get("type") == "function"}
            expected_deferred = sum(tool.defer == "auto" for tool in registration.tools)
            assert len(by_name) == len(registration.tools), "Tool declarations were lost"
            for tool in registration.tools:
                assert by_name[tool.name].get("defer_loading", False) == (tool.defer == "auto"), tool.name
                assert by_name[tool.name]["description"] == tool.description, "Tool policy changed"
            assert any(tool.get("type") == "tool_search" for tool in tools), "Native tool search missing"
            await session.disconnect()

            # The runtime normalizes JSON Schema (e.g. adding an object type).
            # Compare with its own eager wire output, not raw caller schemas.
            eager_options = {**options, "tools": [replace(tool, defer="never") for tool in registration.tools], "tool_search": {"enabled": False}}
            eager_session = await client.create_session(**eager_options)
            await eager_session.send_and_wait(prompt="Hello", timeout=30)
            eager_tools = {tool.get("name"): tool for tool in capture.bodies[-1].get("tools", [])}
            for name, tool in by_name.items():
                assert {key: value for key, value in tool.items() if key != "defer_loading"} == eager_tools[name], "Deferral changed a tool schema or policy"
            await eager_session.disconnect()
            result = {
                "runtime": "hostless", "model_calls": len(capture.bodies),
                "declared_tools": len(registration.tools), "deferred_tools": expected_deferred,
                "eager_tools": len(registration.tools) - expected_deferred, "native_tool_search": True,
                "schemas_and_policies_unchanged": True,
            }
            try:
                import tiktoken
            except ImportError:
                pass
            else:
                encoder = tiktoken.get_encoding("o200k_base")
                count = lambda values: len(encoder.encode(json.dumps(values, separators=(",", ":"), ensure_ascii=False)))
                result.update({
                    "estimated_eager_schema_tokens_before": count(list(eager_tools.values())),
                    "estimated_eager_schema_tokens_after": count([tool for tool in tools if not tool.get("defer_loading")]),
                })
            print(json.dumps(result))
        finally:
            await client.stop()


def plaintext_input(body):
    """Count message/tool plaintext, never ciphertext as if it were model text."""
    def text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(text(item) for item in value)
        if isinstance(value, dict):
            if "text" in value or "content" in value:
                return text(value.get("text", value.get("content")))
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return "" if value is None else str(value)

    parts = []
    for item in body.get("input", []):
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "message")
        if kind == "function_call":
            parts.append(text(item.get("arguments")))
        elif kind == "custom_tool_call":
            parts.append(text(item.get("input")))
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            parts.append(text(item.get("output")))
        elif kind == "reasoning":
            parts.append(text(item.get("summary", [])))
        elif kind == "message":
            parts.append(text(item.get("content", "")))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Local request-body archive (summaries only)")
    parser.add_argument("--excel-only", action="store_true", help="Measure an archive without starting the synthetic SDK runtime")
    args = parser.parse_args()
    if args.excel_only and not args.archive:
        parser.error("--excel-only requires --archive")
    if args.archive:
        archive = json.loads(args.archive.read_text())
        source = archive["request_body"]
        before = archive.get("upstream_body", {})
        after = excel_upstream.prepare_responses_body(source)
        before_text, after_text = plaintext_input(before), plaintext_input(after)
        eager_catalog = "\n".join(
            excel_upstream._render_client_tool_definition(info)
            for info in excel_upstream._client_tool_specs(source).values()
        )
        compact_catalog = excel_upstream._client_tool_protocol_instructions(source)
        archived_catalog = next((excel_upstream._item_text(item.get("content"))
            for item in before.get("input", []) if isinstance(item, dict)
            and "Available client tools:" in excel_upstream._item_text(item.get("content"))), eager_catalog)
        specs = excel_upstream._client_tool_specs(source)
        result = {
            "excel_visible_chars_before": len(before_text), "excel_visible_chars_after": len(after_text),
            "registered_tools": len(specs), "deferred_tools": len(excel_upstream._deferred_tool_specs(source, specs)),
            "measurement": "Local plaintext estimate; excludes server context, ciphertext, and protocol framing.",
            "model_calls": 0,
        }
        try:
            import tiktoken
        except ImportError:
            pass
        else:
            encoder = tiktoken.get_encoding("o200k_base")
            result.update({
                "estimated_visible_tokens_before": len(encoder.encode(before_text, disallowed_special=())),
                "estimated_visible_tokens_after": len(encoder.encode(after_text, disallowed_special=())),
                "estimated_catalog_tokens_before": len(encoder.encode(archived_catalog, disallowed_special=())),
                "estimated_catalog_tokens_after": len(encoder.encode(compact_catalog, disallowed_special=())),
            })
        print(json.dumps(result))
        if args.excel_only:
            return
    else:
        source = {"tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
            {"type": "namespace", "name": "mcp__demo", "tools": [
                {"type": "function", "name": f"read_{i}", "description": f"Read value {i} safely.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}}
                for i in range(40)
            ]},
        ]}
    asyncio.run(verify(source))


if __name__ == "__main__":
    main()
