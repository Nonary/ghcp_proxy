"""Exercise the real Copilot runtime against synthetic, intercepted Responses.

Run with the repo virtualenv. No credentials or model calls are used; the SDK
may download its matching runtime on first use. Checks model-wire history, not
synthetic cached_tokens. --output keeps the synthetic wire bodies for inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from copilot import CopilotClient
from copilot.copilot_request_handler import CopilotRequestHandler

import copilot_sdk_upstream as sdk
import excel_upstream
import format_translation


class SyntheticUpstream(CopilotRequestHandler):
    def __init__(self, reasoning: bool):
        self.bodies: list[dict] = []
        self.reasoning = reasoning

    async def send_request(self, request, ctx):
        # Never forward a request, including unexpected discovery/telemetry.
        if request.url.host != "127.0.0.1" or request.url.path != "/responses":
            raise AssertionError(f"Unexpected SDK request: {request.url}")
        body = json.loads(await request.aread())
        self.bodies.append(body)
        turn = len(self.bodies)
        output = []
        if self.reasoning:
            output.append({
                "type": "reasoning", "id": f"rs_{turn}",
                "summary": [{"type": "summary_text", "text": "Synthetic summary."}],
                "encrypted_content": f"opaque-synthetic-reasoning-{turn}",
            })
        if turn == 1:
            output.append({
                "type": "function_call", "id": "fc_first", "call_id": "call_first",
                "name": "inspect", "arguments": "{}", "status": "completed",
            })
        else:
            output.append({
                "type": "message", "id": f"msg_{turn}", "role": "assistant",
                "status": "completed", "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []},
                ],
            })
        response = {
            "id": f"resp_{turn}", "object": "response", "created_at": 1780000000,
            "model": body["model"], "status": "completed", "output": output,
            "usage": {"input_tokens": 2000, "output_tokens": 100, "total_tokens": 2100,
                      "input_tokens_details": {"cached_tokens": 0}},
        }
        events = [{"type": "response.created", "response": {
            **response, "status": "in_progress", "output": [],
        }}]
        for index, item in enumerate(output):
            for suffix in ("added", "done"):
                events.append({"type": f"response.output_item.{suffix}",
                               "output_index": index, "item": item})
        events.append({"type": "response.completed", "response": response})
        wire = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=wire, request=request)


class LocalClient:
    """Use production SDK options, overriding only the provider destination."""

    def __init__(self, client):
        self.client = client

    async def list_models(self):
        return []

    @staticmethod
    def options(options):
        return {**options, "provider": {
            "type": "openai", "wire_api": "responses",
            "base_url": "http://127.0.0.1:1", "api_key": "synthetic-not-a-credential",
        }}

    async def create_session(self, **options):
        return await self.client.create_session(**self.options(options))

    async def resume_session(self, session_id, **options):
        return await self.client.resume_session(session_id, **self.options(options))


def prefix_length(previous, current):
    count = 0
    for left, right in zip(previous, current):
        # The runtime regenerates assistant message IDs even on live replay.
        # Keep call IDs, reasoning IDs, content and every ordering boundary.
        left, right = copy.deepcopy(left), copy.deepcopy(right)
        for item in (left, right):
            if item.get("type") == "message" and item.get("role") == "assistant":
                item.pop("id", None)
        if left != right:
            break
        count += 1
    return count


async def replay(model, reasoning, destroy_after_answer, root):
    capture = SyntheticUpstream(reasoning)
    client = CopilotClient(mode="empty", use_logged_in_user=False,
                           base_directory=str(root / "runtime"), request_handler=capture)
    body = {
        "model": model, "stream": True, "store": False, "session_id": "synthetic-thread",
        "instructions": "Stable instructions.", "reasoning": {"effort": "high"},
        "tools": [{"type": "function", "name": "inspect", "description": "Inspect fixture",
                   "parameters": {"type": "object", "properties": {}}}],
        "input": [{"type": "message", "role": "developer", "content": "Stable developer."},
                  {"type": "message", "role": "user", "content": "First question."}],
    }
    diagnostics = []
    with patch.object(sdk, "_SDK_STATE_DIR", str(root / "proxy")), \
         patch.object(sdk, "_live_sessions", {}), \
         patch.object(sdk, "_pending_alias_watermark", {}), \
         patch.object(sdk, "_get_client", return_value=LocalClient(client)):
        try:
            await client.start()
            for turn in range(3):
                registration = sdk.build_tool_registration(body)
                diagnostic = {}
                session, dispatch = await sdk._open_session(body, registration, diagnostics=diagnostic)
                diagnostics.append(diagnostic)
                outcome = await sdk._wait_for_outcome(session, dispatch, registration)
                payload = sdk._response_payload(body, session.session_id, outcome, f"client_{turn}")
                sdk._commit_alias_watermark(session.session_id, success=True)
                if destroy_after_answer and not outcome.calls:
                    await sdk._evict_live_session(session.session_id)
                else:
                    await sdk._release_session(session, outcome, completed=True)
                body["input"].extend(payload["output"])
                if outcome.calls:
                    call = next(item for item in payload["output"] if item["type"] == "function_call")
                    body["input"].append({"type": "function_call_output",
                                          "call_id": call["call_id"], "output": "Inspection result."})
                else:
                    body["input"].append({"type": "message", "role": "user", "content": "Next question."})
        finally:
            await sdk._evict_all_live_sessions()
            await client.stop()
    assert len(capture.bodies) == 3
    before, after = capture.bodies[1:]
    for field in ("model", "instructions", "tools", "reasoning", "text", "prompt_cache_key"):
        assert before.get(field) == after.get(field), field
    shared = prefix_length(before["input"], after["input"])
    reasoning_count = sum(item.get("type") == "reasoning" for item in after["input"])
    if reasoning and destroy_after_answer:
        assert shared < len(before["input"]) and reasoning_count == 0
    else:
        assert shared == len(before["input"]), (shared, before, after)
        assert reasoning_count == (2 if reasoning else 0)
    (root / "wire.json").write_text(json.dumps(capture.bodies, indent=2), encoding="utf-8")
    return {"model": model, "old_lifecycle": destroy_after_answer,
            "previous_input_items": len(before["input"]), "reused_prefix_items": shared,
            "reasoning_items_retained": reasoning_count,
            "session_operations": [item["operation"] for item in diagnostics]}


def verify_excel_and_rest():
    items = [{"type": "message", "role": "user", "content": "First"},
             {"type": "reasoning", "id": "rs_one", "summary": [], "encrypted_content": "opaque"},
             {"type": "message", "role": "assistant", "content": "Done"}]
    body = {"model": "gpt-5.6-sol-excel", "instructions": "Stable", "input": items,
            "tools": [], "prompt_cache_key": "synthetic-thread"}
    first = excel_upstream.prepare_responses_body(copy.deepcopy(body))
    follow_up = {**body, "input": [*items, {"type": "message", "role": "user", "content": "Next"}]}
    second = excel_upstream.prepare_responses_body(copy.deepcopy(follow_up))
    assert first["input"] == second["input"][:len(first["input"])]
    assert any(item.get("encrypted_content") == "opaque" for item in second["input"])
    for native in (False, True):
        old = format_translation.sanitize_input(items, native_responses_passthrough=native)
        new = format_translation.sanitize_input(follow_up["input"], native_responses_passthrough=native)
        assert new[:len(old)] == old
        assert new[1]["encrypted_content"] == "opaque"


async def verify_chat_passthrough():
    import proxy
    from fastapi.responses import JSONResponse

    for model in ("gpt-5.6-sol", "gpt-4.1"):
        body = {"model": model, "messages": [
            {"role": "system", "content": "Stable system"},
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
        ], "tools": [{"type": "function", "function": {
            "name": "inspect", "parameters": {"type": "object", "properties": {}},
        }}]}
        with patch.object(proxy, "parse_json_request", return_value=body), \
             patch.object(proxy.auth, "get_api_base", return_value="https://example.invalid"), \
             patch.object(proxy, "_prepare_upstream_request", return_value=(SimpleNamespace(), None)) as prepare, \
             patch.object(proxy, "_post_non_streaming_request", return_value=JSONResponse({})):
            await proxy.chat_completions(None)
            assert prepare.call_args.kwargs["body"] == body
            previous = copy.deepcopy(body)
            body["messages"].extend([
                {"role": "assistant", "content": "Done"}, {"role": "user", "content": "Next"},
            ])
            await proxy.chat_completions(None)
            forwarded = prepare.call_args.kwargs["body"]
            assert forwarded["messages"][:2] == previous["messages"]
            assert forwarded["tools"] == previous["tools"]


async def main(root):
    results = []
    for model, reasoning in (("gpt-5.6-sol", True), ("gpt-5.6-luna", True), ("gpt-4.1", False)):
        for old in (True, False):
            directory = root / f"{model}-{'before' if old else 'after'}"
            directory.mkdir(parents=True, exist_ok=True)
            results.append(await replay(model, reasoning, old, directory))
    verify_excel_and_rest()
    await verify_chat_passthrough()
    print(json.dumps({"replays": results, "excel_and_rest_prefix_checks": "passed",
                      "chat_completions_passthrough_checks": "passed",
                      "live_cache_measurement": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        asyncio.run(main(args.output.resolve()))
    else:
        with tempfile.TemporaryDirectory(prefix="sdk-cache-replay-") as directory:
            asyncio.run(main(Path(directory)))
