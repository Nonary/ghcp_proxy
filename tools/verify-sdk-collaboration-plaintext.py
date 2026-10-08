#!/usr/bin/env python
"""Opt-in live probe: one SDK turn, no Codex agents or tool execution.

Run with the repository virtualenv and --live. State is isolated in a temporary
directory, and the probe's SDK session is disconnected/deleted before exit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copilot_sdk_upstream as sdk


async def verify(model: str) -> dict:
    message = "Synthetic plaintext assignment: reply with PROXY_CANARY_RECEIVED."
    body = {
        "model": model,
        "stream": True,
        "instructions": "You are an isolated protocol probe. Only use the single declared tool.",
        "tools": [{
            "type": "namespace", "name": "collaboration", "tools": [{
                "type": "function", "name": "followup_task", "defer_loading": False,
                "description": "Capture a synthetic follow-up assignment. No real agent receives this probe.",
                "parameters": {
                    "type": "object", "properties": {
                        "target": {"type": "string"},
                        "message": {"type": "string", "encrypted": True},
                    }, "required": ["target", "message"], "additionalProperties": False,
                },
            }],
        }],
        "input": [{"role": "user", "content": (
            "Call followup_task once with target /root/proxy_canary and the exact message:\n" + message
        )}],
    }
    session = None
    client = None
    with tempfile.TemporaryDirectory(prefix="ghcp-collaboration-probe-") as state_dir:
        sdk._SDK_STATE_DIR = state_dir
        try:
            registration = sdk.build_tool_registration(body)
            assert "encrypted" not in registration.tools[0].parameters["properties"]["message"]
            session, dispatch = await sdk._open_session(body, registration)
            client = sdk._client
            outcome = await asyncio.wait_for(sdk._wait_for_outcome(session, dispatch, registration), 120)
            assert len(outcome.calls) == 1, "Expected exactly one captured follow-up call"
            item = sdk._tool_item(session.session_id, outcome.calls[0])
            assert (item.get("namespace"), item["name"]) == ("collaboration", "followup_task")
            arguments = json.loads(item["arguments"])
            assert arguments["message"] == message, "Live SDK did not return the exact plaintext assignment"
            assert arguments["target"] == "/root/proxy_canary"
            assert item["encrypted_function_args"] == []
            return {
                "model": model, "live_sdk_message_plaintext": True,
                "exact_assignment_preserved": True, "encrypted_function_args": [],
                "native_agents_spawned": 0, "tools_executed": 0,
                "usage": outcome.usage,
            }
        finally:
            try:
                await sdk._evict_all_live_sessions()
                if client is not None and session is not None:
                    await client.delete_session(session.session_id)
            finally:
                await sdk.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize one real Copilot model turn")
    parser.add_argument("--model", default="gpt-6-luna", help="Copilot SDK model ID")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this probe makes a real model request")
    print(json.dumps(asyncio.run(verify(args.model)), indent=2))


if __name__ == "__main__":
    main()
