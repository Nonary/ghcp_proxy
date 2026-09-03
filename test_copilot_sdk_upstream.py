import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantReasoningData,
    AssistantReasoningDeltaData,
    AssistantUsageData,
    ExternalToolRequestedData,
    SessionIdleData,
    SubagentCompletedData,
    SubagentFailedData,
    SubagentStartedData,
)

import copilot_sdk_upstream as sdk
import format_translation


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

    async def test_usage_event_subtracts_cached_tokens_from_fresh(self):
        """AssistantUsageData.input_tokens is the *total* (fresh + cached).
        _usage_from_event must subtract cache_read_tokens so that
        fresh_input_tokens and cached_input_tokens are reported correctly."""
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.usage",
                AssistantUsageData(
                    model="gpt-test",
                    input_tokens=300,   # total = 100 fresh + 200 cached
                    output_tokens=50,
                    cache_read_tokens=200,
                    reasoning_tokens=10,
                ),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())

        u = outcome.usage
        self.assertEqual(u["input_tokens"], 300)
        self.assertEqual(u["cached_input_tokens"], 200)
        self.assertEqual(u["fresh_input_tokens"], 100)
        self.assertEqual(u["pricing_fresh_input_tokens"], 100)
        self.assertEqual(u["pricing_cached_input_tokens"], 200)
        self.assertEqual(u["output_tokens"], 50)
        self.assertEqual(u["reasoning_output_tokens"], 10)
        self.assertEqual(u["total_tokens"], 350)

    async def test_usage_event_no_cached_tokens_fresh_equals_total(self):
        """When cache_read_tokens is zero, fresh_input_tokens == input_tokens."""
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.usage",
                AssistantUsageData(model="gpt-test", input_tokens=80, output_tokens=20),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())

        u = outcome.usage
        self.assertEqual(u["input_tokens"], 80)
        self.assertEqual(u["cached_input_tokens"], 0)
        self.assertEqual(u["fresh_input_tokens"], 80)
        self.assertEqual(u["output_tokens"], 20)

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

    def test_input_to_prompt_compaction_window_and_decoding(self):
        fake_enc = format_translation.encode_fake_compaction("Previous conversation summary")
        items = [
            {"type": "message", "role": "user", "content": "Old message 1"},
            {"type": "message", "role": "assistant", "content": "Old reply 1"},
            {"type": "compaction", "encrypted_content": fake_enc},
            {"type": "message", "role": "user", "content": "Current question"},
        ]
        prompt = sdk.input_to_prompt(items)
        self.assertNotIn("Old message 1", prompt)
        self.assertNotIn("Old reply 1", prompt)
        self.assertIn("[Compacted conversation summary]\nPrevious conversation summary", prompt)
        self.assertIn("User: Current question", prompt)

    def test_input_to_prompt_skips_subagent_notification_messages(self):
        items = [
            {"type": "message", "role": "user", "content": "<subagent_notification>\n{\"status\": \"completed\"}\n</subagent_notification>"},
            {"type": "message", "role": "user", "content": "Real user instruction"},
        ]
        prompt = sdk.input_to_prompt(items)
        self.assertNotIn("<subagent_notification>", prompt)
        self.assertIn("Real user instruction", prompt)

    def test_to_compaction_payload(self):
        outcome = sdk.TurnOutcome(text="Summary of earlier code work.")
        payload = sdk.to_compaction_payload({"model": "gpt-5.6-luna"}, "sess-1", outcome, "resp-1")
        self.assertEqual(payload["id"], "resp-1")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output_text"], "Summary of earlier code work.")
        self.assertEqual(len(payload["output"]), 1)
        self.assertEqual(payload["output"][0]["type"], "compaction")
        enc = payload["output"][0]["encrypted_content"]
        self.assertEqual(format_translation.decode_fake_compaction(enc), "Summary of earlier code work.")

    def test_responses_to_compaction_response(self):
        responses_payload = {
            "id": "resp-test",
            "model": "gpt-5.6-luna",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "This is the generated summary."}],
                }
            ],
            "usage": {"total_tokens": 42},
        }
        compact = format_translation.responses_to_compaction_response(responses_payload)
        self.assertEqual(compact["output_text"], "This is the generated summary.")
        self.assertEqual(compact["output"][0]["type"], "compaction")
        self.assertEqual(
            format_translation.decode_fake_compaction(compact["output"][0]["encrypted_content"]),
            "This is the generated summary.",
        )

    async def test_subagent_events_and_parent_tool_call_id_route_to_reasoning(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "subagent.started",
                SubagentStartedData(agent_description="Reviews code", agent_name="reviewer", agent_display_name="Code Reviewer", tool_call_id="call-sub"),
            )
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="internal check", message_id="msg-sub", parent_tool_call_id="call-sub"),
            )
            session.emit(
                "subagent.completed",
                SubagentCompletedData(agent_name="reviewer", agent_display_name="Code Reviewer", tool_call_id="call-sub"),
            )
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="final answer", message_id="msg-main", parent_tool_call_id=None),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())
        self.assertEqual(outcome.text, "final answer")
        self.assertIn("Code Reviewer", outcome.reasoning)
        self.assertIn("internal check", outcome.reasoning)
        self.assertIn("completed", outcome.reasoning)

    async def test_stream_turn_routes_subagent_delta_to_reasoning_stream(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="subagent thought", message_id="msg-sub", parent_tool_call_id="call-sub"),
            )
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="final output", message_id="msg-main", parent_tool_call_id=None),
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

        # Check that subagent delta was emitted as reasoning summary delta
        reasoning_deltas = [d.get("delta") for name, d in events if name == "response.reasoning_summary_text.delta"]
        self.assertIn("subagent thought", reasoning_deltas)

        # Check that main delta was emitted as output text delta
        text_deltas = [d.get("delta") for name, d in events if name == "response.output_text.delta"]
        self.assertIn("final output", text_deltas)

    async def test_handle_responses_sets_session_id_on_plan_and_triggers_finish(self):
        session = _FakeSession()
        session.session_id = "custom-sdk-session-id"

        async def fake_open(body, registration):
            async def dispatch():
                session.emit(
                    "assistant.message_delta",
                    AssistantMessageDeltaData(delta_content="done", message_id="msg-1"),
                )
                session.emit("session.idle", SessionIdleData())
            return session, dispatch

        finished_events = []
        def finish_callback(plan, status_code, **kwargs):
            finished_events.append((plan, status_code, kwargs))

        plan = SimpleNamespace(usage_event={"request_id": "req-1"})

        with patch.object(sdk, "_open_session", side_effect=fake_open), \
             patch.object(sdk, "_delete_owned_session") as mock_delete:
            resp = await sdk.handle_responses(
                _ConnectedRequest(),
                {"input": "test"},
                plan=plan,
                finish_usage_callback=finish_callback,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(plan.usage_event.get("session_id"), "custom-sdk-session-id")
            self.assertEqual(len(finished_events), 1)
            self.assertEqual(finished_events[0][1], 200)
            # Verify session was NOT deleted
            mock_delete.assert_not_called()

    def test_scan_session_state(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sess_dir = os.path.join(temp_dir, "session-state", "test-sess-1")
            os.makedirs(sess_dir, exist_ok=True)
            events_file = os.path.join(sess_dir, "events.jsonl")
            with open(events_file, "w") as f:
                f.write(json.dumps({"type": "session.start", "timestamp": "2026-09-02T12:00:00Z", "data": {"selectedModel": "gpt-5.6-luna", "startTime": "2026-09-02T12:00:00Z"}}) + "\n")
                f.write(json.dumps({"type": "assistant.message", "timestamp": "2026-09-02T12:00:05Z", "data": {"model": "gpt-5.6-luna", "outputTokens": 100}}) + "\n")
                f.write(json.dumps({
                    "type": "session.shutdown",
                    "timestamp": "2026-09-02T12:00:10Z",
                    "data": {
                        "totalApiDurationMs": 5000,
                        "tokenDetails": {
                            "input": {"tokenCount": 50},
                            "cache_read": {"tokenCount": 200},
                            "output": {"tokenCount": 100},
                        },
                        "modelMetrics": {
                            "gpt-5.6-luna": {
                                "usage": {"reasoningTokens": 20},
                            }
                        }
                    }
                }) + "\n")

            recorded = []
            with patch.object(sdk, "_SDK_STATE_DIR", temp_dir), \
                 patch.object(sdk, "_SESSION_STATE_DIR", os.path.join(temp_dir, "session-state")), \
                 patch.object(sdk, "_INGEST_CURSOR_FILE", os.path.join(temp_dir, "session-cursor.json")):
                count = sdk.scan_session_state(recorded.append)
                self.assertEqual(count, 1)
                self.assertEqual(len(recorded), 1)
                ev = recorded[0]
                self.assertEqual(ev["session_id"], "test-sess-1")
                self.assertEqual(ev["requested_model"], "gpt-5.6-luna")
                self.assertEqual(ev["duration_ms"], 5000)
                self.assertEqual(ev["usage"]["input_tokens"], 250)
                self.assertEqual(ev["usage"]["cached_input_tokens"], 200)
                self.assertEqual(ev["usage"]["fresh_input_tokens"], 50)
                self.assertEqual(ev["usage"]["output_tokens"], 100)
                self.assertEqual(ev["usage"]["reasoning_output_tokens"], 20)

                # Re-scan should skip due to cursor
                count_again = sdk.scan_session_state(recorded.append)
                self.assertEqual(count_again, 0)
                self.assertEqual(len(recorded), 1)

    def test_scan_session_state_multi_turn(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sess_dir = os.path.join(temp_dir, "session-state", "test-multi-turn")
            os.makedirs(sess_dir, exist_ok=True)
            events_file = os.path.join(sess_dir, "events.jsonl")
            with open(events_file, "w") as f:
                f.write(json.dumps({"type": "session.start", "timestamp": "2026-09-02T12:00:00Z", "data": {"selectedModel": "gpt-5.6-luna"}}) + "\n")
                # Turn 1
                f.write(json.dumps({"type": "assistant.turn_start", "timestamp": "2026-09-02T12:00:01Z", "data": {"turnId": "0", "interactionId": "turn-1"}}) + "\n")
                f.write(json.dumps({"type": "assistant.message", "timestamp": "2026-09-02T12:00:05Z", "data": {"model": "gpt-5.6-luna", "outputTokens": 100}}) + "\n")
                f.write(json.dumps({
                    "type": "session.shutdown",
                    "timestamp": "2026-09-02T12:00:06Z",
                    "data": {
                        "totalApiDurationMs": 5000,
                        "tokenDetails": {"input": {"tokenCount": 50}, "cache_read": {"tokenCount": 200}, "cache_write": {"tokenCount": 0}, "output": {"tokenCount": 100}},
                        "modelMetrics": {"gpt-5.6-luna": {"usage": {"reasoningTokens": 20}}}
                    }
                }) + "\n")
                # Turn 2
                f.write(json.dumps({"type": "assistant.turn_start", "timestamp": "2026-09-02T12:00:07Z", "data": {"turnId": "0", "interactionId": "turn-2"}}) + "\n")
                f.write(json.dumps({"type": "assistant.message", "timestamp": "2026-09-02T12:00:10Z", "data": {"model": "gpt-5.6-luna", "outputTokens": 50}}) + "\n")
                f.write(json.dumps({
                    "type": "session.shutdown",
                    "timestamp": "2026-09-02T12:00:11Z",
                    "data": {
                        "totalApiDurationMs": 8000,
                        "tokenDetails": {"input": {"tokenCount": 60}, "cache_read": {"tokenCount": 500}, "cache_write": {"tokenCount": 0}, "output": {"tokenCount": 150}},
                        "modelMetrics": {"gpt-5.6-luna": {"usage": {"reasoningTokens": 30}}}
                    }
                }) + "\n")

            recorded = []
            with patch.object(sdk, "_SDK_STATE_DIR", temp_dir), \
                 patch.object(sdk, "_SESSION_STATE_DIR", os.path.join(temp_dir, "session-state")), \
                 patch.object(sdk, "_INGEST_CURSOR_FILE", os.path.join(temp_dir, "session-cursor.json")):
                count = sdk.scan_session_state(recorded.append)
                self.assertEqual(count, 2)
                self.assertEqual(len(recorded), 2)

                # Check Turn 1
                t1 = recorded[0]
                self.assertEqual(t1["request_id"], "copilot-sdk:test-multi-turn:turn-1")
                self.assertEqual(t1["duration_ms"], 5000)
                self.assertEqual(t1["usage"]["input_tokens"], 250)
                self.assertEqual(t1["usage"]["output_tokens"], 100)
                self.assertEqual(t1["usage"]["cached_input_tokens"], 200)
                self.assertEqual(t1["usage"]["fresh_input_tokens"], 50)

                # Check Turn 2 (deltas!)
                t2 = recorded[1]
                self.assertEqual(t2["request_id"], "copilot-sdk:test-multi-turn:turn-2")
                self.assertEqual(t2["duration_ms"], 3000)  # 8000 - 5000
                self.assertEqual(t2["usage"]["input_tokens"], 310)  # 560 - 250
                self.assertEqual(t2["usage"]["output_tokens"], 50)  # 150 - 100
                self.assertEqual(t2["usage"]["cached_input_tokens"], 300)  # 500 - 200
                self.assertEqual(t2["usage"]["fresh_input_tokens"], 10)  # 60 - 50
                self.assertEqual(t2["usage"]["reasoning_output_tokens"], 10)  # 30 - 20

    async def test_stream_turn_emits_keepalive_comments_on_timeout(self):
        session = _FakeSession()

        async def dispatch():
            # Delay emitting an event to trigger keepalive
            await asyncio.sleep(0.05)
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="delayed message", message_id="msg-1"),
            )
            session.emit("session.idle", SessionIdleData())

        chunks: list[bytes] = []
        with patch.object(sdk, "_KEEPALIVE_INTERVAL_SECONDS", 0.01), \
             patch.object(sdk, "_TURN_TIMEOUT_SECONDS", 1.0):
            async for chunk in sdk._stream_turn(
                _ConnectedRequest(),
                {"model": "gpt-test"},
                session,
                dispatch,
                sdk.ToolRegistration(),
            ):
                chunks.append(chunk)

        # Verify that at least one keep-alive comment chunk was yielded
        self.assertIn(b": keep-alive\n\n", chunks)
        all_text = b"".join(chunks).decode()
        self.assertIn("delayed message", all_text)

    async def test_stream_turn_captures_usage_from_session_shutdown(self):
        session = _FakeSession()
        session.session_id = "test-shutdown-usage"

        async def dispatch():
            session.emit(
                "assistant.message",
                AssistantMessageData(content="Hello world", message_id="msg-1"),
            )
            # Emit session.shutdown with token details
            fake_shutdown = {
                "totalApiDurationMs": 1500,
                "tokenDetails": {
                    "input": {"tokenCount": 25},
                    "cache_read": {"tokenCount": 100},
                    "cache_write": {"tokenCount": 0},
                    "output": {"tokenCount": 15},
                },
                "modelMetrics": {
                    "gpt-test": {
                        "usage": {"reasoningTokens": 5},
                    }
                }
            }
            session.emit("session.shutdown", fake_shutdown)
            session.emit("session.idle", SessionIdleData())

        recorded_usage = []
        def fake_finish(plan, status_code, **kwargs):
            recorded_usage.append(kwargs.get("usage"))

        chunks = []
        plan = unittest.mock.MagicMock()
        async for chunk in sdk._stream_turn(
            _ConnectedRequest(),
            {"model": "gpt-test"},
            session,
            dispatch,
            sdk.ToolRegistration(),
            plan=plan,
            finish_usage_callback=fake_finish,
        ):
            chunks.append(chunk)

        self.assertEqual(len(recorded_usage), 1)
        u = recorded_usage[0]
        self.assertEqual(u["input_tokens"], 125)
        self.assertEqual(u["output_tokens"], 15)
        self.assertEqual(u["cached_input_tokens"], 100)
        self.assertEqual(u["fresh_input_tokens"], 25)
        self.assertEqual(u["reasoning_output_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
