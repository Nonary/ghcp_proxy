import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from copilot.session_events import (
    AssistantMessageDeltaData,
    AssistantReasoningData,
    AssistantReasoningDeltaData,
    AssistantUsageData,
    ExternalToolRequestedData,
    SessionIdleData,
)

import copilot_sdk_upstream as sdk


class _FakeSession:
    def __init__(self):
        self.handlers = []
        self.session_id = "session-1"
        self.disconnected = False
        self.aborted = False

    def on(self, handler):
        self.handlers.append(handler)

        def unsubscribe():
            self.handlers.remove(handler)

        return unsubscribe

    def emit(self, event_type, data):
        event = SimpleNamespace(type=SimpleNamespace(value=event_type), data=data)
        for handler in list(self.handlers):
            handler(event)

    async def disconnect(self):
        self.disconnected = True

    async def abort(self):
        self.aborted = True


class _ConnectedRequest:
    async def is_disconnected(self):
        return False


class CopilotSdkTranslationTests(unittest.TestCase):
    def test_sdk_is_default_and_rest_is_explicit_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sdk.responses_upstream(), "sdk")
        with patch.dict(os.environ, {sdk.RESPONSES_UPSTREAM_ENV: "rest"}):
            self.assertEqual(sdk.responses_upstream(), "rest")

    def test_function_and_custom_tools_are_declaration_only(self):
        registration = sdk.build_tool_registration(
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "mcp.server/read",
                        "description": "Read a value",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                        },
                    },
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Apply a patch",
                    },
                ]
            }
        )

        self.assertEqual([tool.name for tool in registration.tools], ["mcp_server_read", "apply_patch"])
        self.assertTrue(all(tool.handler is None for tool in registration.tools))
        self.assertTrue(all(tool.skip_permission for tool in registration.tools))
        self.assertTrue(all(tool.defer == "never" for tool in registration.tools))
        self.assertEqual(
            registration.tools[1].parameters["properties"]["input"]["type"],
            "string",
        )

    def test_call_id_round_trip_carries_resume_and_original_tool_metadata(self):
        call_id = sdk._encode_call_id(
            "session-1",
            "request-2",
            tool_name="apply_patch",
            tool_type="custom",
        )

        self.assertEqual(
            sdk._decode_call_id(call_id),
            {"s": "session-1", "r": "request-2", "n": "apply_patch", "t": "custom"},
        )
        self.assertIsNone(sdk._decode_call_id("call_from_another_provider"))

    def test_only_trailing_tool_outputs_resume_a_session(self):
        old = sdk._encode_call_id("session-1", "old", tool_name="one", tool_type="function")
        new = sdk._encode_call_id("session-1", "new", tool_name="two", tool_type="function")
        value = [
            {"type": "function_call_output", "call_id": old, "output": "old result"},
            {"role": "assistant", "content": "continued"},
            {"type": "function_call", "call_id": new, "name": "two", "arguments": "{}"},
            {"type": "function_call_output", "call_id": new, "output": "new result"},
        ]

        session_id, results = sdk.resolve_tool_continuation(value)
        self.assertEqual(session_id, "session-1")
        self.assertEqual([(item.request_id, item.output) for item in results], [("new", "new result")])
        self.assertIsNone(
            sdk.resolve_tool_continuation(value + [{"role": "user", "content": "new turn"}])
        )

    def test_prompt_preserves_roles_and_omits_opaque_reasoning(self):
        prompt = sdk.input_to_prompt(
            [
                {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                {"type": "reasoning", "encrypted_content": "opaque"},
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            ]
        )
        self.assertEqual(prompt, "User: hello\n\nAssistant: hi")

    def test_response_payload_preserves_custom_tool_shape(self):
        outcome = sdk.TurnOutcome(
            calls=[sdk.ToolCall("request-1", "apply_patch", "custom", {"input": "*** patch"})],
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )
        payload = sdk._response_payload({"model": "gpt-test"}, "session-1", outcome, "resp_1")

        item = payload["output"][0]
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["input"], "*** patch")
        self.assertEqual(sdk._decode_call_id(item["call_id"])["s"], "session-1")


class CopilotSdkEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_streaming_event_translation_collects_text_and_usage(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="SDK", message_id="message-1"),
            )
            session.emit(
                "assistant.usage",
                AssistantUsageData(model="gpt-test", input_tokens=4, output_tokens=2),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())

        self.assertEqual(outcome.text, "SDK")
        self.assertEqual(outcome.usage["total_tokens"], 6)

    async def test_external_tool_request_suspends_and_returns_to_caller(self):
        session = _FakeSession()
        registration = sdk.build_tool_registration(
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            }
        )

        async def dispatch():
            session.emit(
                "external_tool.requested",
                ExternalToolRequestedData(
                    request_id="request-1",
                    session_id="session-1",
                    tool_call_id="runtime-call-1",
                    tool_name="lookup",
                    arguments={"key": "value"},
                ),
            )

        outcome = await sdk._wait_for_outcome(session, dispatch, registration)

        self.assertEqual(len(outcome.calls), 1)
        self.assertEqual(outcome.calls[0].name, "lookup")
        self.assertEqual(json.loads(sdk._arguments_json(outcome.calls[0])), {"key": "value"})

    async def test_stream_emits_responses_event_sequence(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="hello", message_id="message-1"),
            )
            session.emit("session.idle", SessionIdleData())

        chunks = [
            chunk.decode()
            async for chunk in sdk._stream_turn(
                _ConnectedRequest(),
                {"model": "gpt-test"},
                session,
                dispatch,
                sdk.ToolRegistration(),
            )
        ]
        wire = "".join(chunks)

        expected = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        positions = [wire.index(f"event: {name}") for name in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(session.disconnected)

    async def test_wait_for_outcome_collects_reasoning_without_duplication(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.reasoning_delta",
                AssistantReasoningDeltaData(delta_content="thinking step", reasoning_id="r-1"),
            )
            session.emit(
                "assistant.reasoning",
                AssistantReasoningData(content="thinking step", reasoning_id="r-1"),
            )
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="answer", message_id="m-1"),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())
        self.assertEqual(outcome.reasoning, "thinking step")
        self.assertEqual(outcome.text, "answer")

    async def test_stream_emits_reasoning_and_message_with_stable_ids_and_indices(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.reasoning_delta",
                AssistantReasoningDeltaData(delta_content="thought", reasoning_id="r-1"),
            )
            session.emit(
                "assistant.reasoning",
                AssistantReasoningData(content="thought", reasoning_id="r-1"),
            )
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="reply", message_id="m-1"),
            )
            session.emit("session.idle", SessionIdleData())

        events: list[tuple[str, dict]] = []
        async for chunk in sdk._stream_turn(
            _ConnectedRequest(),
            {"model": "gpt-test"},
            session,
            dispatch,
            sdk.ToolRegistration(),
        ):
            text = chunk.decode()
            for block in text.strip().split("\n\n"):
                if not block.strip():
                    continue
                lines = block.splitlines()
                ev_name = lines[0].replace("event: ", "").strip()
                data = json.loads(lines[1].replace("data: ", ""))
                events.append((ev_name, data))

        event_names = [name for name, _ in events]
        expected_names = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        self.assertEqual(event_names, expected_names)

        # Verify reasoning item has output_index 0 and message has output_index 1
        reasoning_added = next(d for name, d in events if name == "response.output_item.added" and d["item"]["type"] == "reasoning")
        message_added = next(d for name, d in events if name == "response.output_item.added" and d["item"]["type"] == "message")
        self.assertEqual(reasoning_added["output_index"], 0)
        self.assertEqual(message_added["output_index"], 1)

        # Verify response.completed matches item IDs and ordering
        completed = events[-1][1]["response"]
        self.assertEqual(len(completed["output"]), 2)
        self.assertEqual(completed["output"][0]["id"], reasoning_added["item"]["id"])
        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][1]["id"], message_added["item"]["id"])
        self.assertEqual(completed["output"][1]["type"], "message")


if __name__ == "__main__":
    unittest.main()
