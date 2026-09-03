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

    def test_sdk_sessions_enable_automatic_context_compaction(self):
        options = sdk._session_options({"model": "gpt-test"}, sdk.ToolRegistration())
        self.assertEqual(options["infinite_sessions"], {"enabled": True})

    def test_recognizes_terminal_in_band_compaction_trigger(self):
        self.assertTrue(
            sdk.is_compaction_request(
                {
                    "input": [
                        {"type": "message", "role": "user", "content": "history"},
                        {"type": "compaction_trigger"},
                    ]
                }
            )
        )
        self.assertFalse(
            sdk.is_compaction_request(
                {"input": [{"type": "message", "role": "user", "content": "normal"}]}
            )
        )

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

    def test_response_payload_unwraps_luna_apply_patch_arguments(self):
        patch_text = "*** Begin Patch\n*** Update File: example.txt\n@@\n-old\n+new\n*** End Patch"
        observed_arguments = [
            patch_text,
            {"input": patch_text},
            json.dumps({"input": patch_text}),
            {"patch": patch_text},
            json.dumps({"patch": patch_text}),
        ]

        for arguments in observed_arguments:
            with self.subTest(arguments=arguments):
                call = sdk.ToolCall("request-1", "apply_patch", "custom", arguments)
                self.assertEqual(sdk._arguments_json(call), patch_text)

    def test_custom_tool_does_not_unwrap_unknown_structured_arguments(self):
        call = sdk.ToolCall("request-1", "other_custom_tool", "custom", {"patch": "value"})
        self.assertEqual(json.loads(sdk._arguments_json(call)), {"patch": "value"})

    def test_response_payload_includes_input_and_output_tokens_details(self):
        outcome = sdk.TurnOutcome(
            usage={
                "input_tokens": 1000,
                "output_tokens": 150,
                "cached_input_tokens": 800,
                "cache_creation_input_tokens": 50,
                "fresh_input_tokens": 200,
                "reasoning_output_tokens": 40,
                "total_tokens": 1150,
            },
        )
        payload = sdk._response_payload({"model": "gpt-5.6-luna"}, "session-1", outcome, "resp_1")
        usage = payload["usage"]
        self.assertEqual(usage["input_tokens"], 1000)
        self.assertEqual(usage["output_tokens"], 150)
        self.assertEqual(usage["cached_input_tokens"], 800)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 800)
        self.assertEqual(usage["input_tokens_details"]["cache_creation_input_tokens"], 50)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], 40)



class _IsolatedSdkState:
    """Redirect SDK state files at a temp dir for the duration of a test."""

    def __enter__(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self._patch = patch.object(sdk, "_SDK_STATE_DIR", self._dir.name)
        self._patch.start()
        return self._dir.name

    def __exit__(self, *exc):
        self._patch.stop()
        self._dir.cleanup()
        return False


class CopilotSdkEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._sdk_state = _IsolatedSdkState()
        self._sdk_state.__enter__()
        self.addCleanup(self._sdk_state.__exit__, None, None, None)

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

    async def test_usage_events_are_accumulated_across_model_calls_in_one_turn(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.usage",
                AssistantUsageData(model="gpt-test", input_tokens=100, output_tokens=10),
            )
            session.emit(
                "assistant.usage",
                AssistantUsageData(
                    model="gpt-test",
                    input_tokens=40,
                    output_tokens=20,
                    cache_read_tokens=30,
                    cache_write_tokens=2,
                    reasoning_tokens=5,
                ),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())

        self.assertEqual(outcome.usage["input_tokens"], 140)
        self.assertEqual(outcome.usage["output_tokens"], 30)
        self.assertEqual(outcome.usage["cached_input_tokens"], 30)
        self.assertEqual(outcome.usage["fresh_input_tokens"], 110)
        self.assertEqual(outcome.usage["cache_creation_input_tokens"], 2)
        self.assertEqual(outcome.usage["reasoning_output_tokens"], 5)
        self.assertEqual(outcome.usage["total_tokens"], 170)

    async def test_shutdown_usage_wins_over_per_call_usage_events(self):
        """The two sources describe the same tokens; exactly one must win."""
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "assistant.usage",
                AssistantUsageData(model="gpt-test", input_tokens=100, output_tokens=10),
            )
            session.emit(
                "session.shutdown",
                SimpleNamespace(
                    token_details=None,
                    model_metrics={
                        "gpt-test": SimpleNamespace(
                            usage=SimpleNamespace(
                                input_tokens=500,
                                cache_read_tokens=100,
                                cache_write_tokens=50,
                                output_tokens=40,
                                reasoning_tokens=7,
                            )
                        )
                    },
                ),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())

        # Shutdown totals, not the 100/10 from the per-call event, and not a sum.
        self.assertEqual(outcome.usage["input_tokens"], 500)
        self.assertEqual(outcome.usage["output_tokens"], 40)
        self.assertEqual(outcome.usage["fresh_input_tokens"], 400)
        self.assertEqual(outcome.event_usage["input_tokens"], 100)

    async def test_shutdown_baseline_survives_a_proxy_restart(self):
        """A resumed session must not bill its whole history to one request."""
        def shutdown(total_input, output):
            return SimpleNamespace(
                token_details=None,
                model_metrics={
                    "gpt-test": SimpleNamespace(
                        usage=SimpleNamespace(
                            input_tokens=total_input,
                            cache_read_tokens=0,
                            cache_write_tokens=0,
                            output_tokens=output,
                            reasoning_tokens=0,
                        )
                    )
                },
            )

        async def run_turn(total_input, output):
            session = _FakeSession()

            async def dispatch():
                session.emit("session.shutdown", shutdown(total_input, output))
                session.emit("session.idle", SessionIdleData())

            return await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())

        first = await run_turn(1000, 50)
        self.assertEqual(first.usage["input_tokens"], 1000)

        # Session-cumulative totals keep climbing; the second turn is the delta.
        second = await run_turn(3000, 120)
        self.assertEqual(second.usage["input_tokens"], 2000)
        self.assertEqual(second.usage["output_tokens"], 70)

        # Simulate a proxy restart: only the on-disk baseline is left.
        self.assertIsNotNone(sdk._shutdown_baseline("session-1"))
        third = await run_turn(3500, 130)
        self.assertEqual(third.usage["input_tokens"], 500)
        self.assertEqual(third.usage["output_tokens"], 10)

    async def test_open_session_resumes_alias_and_sends_only_new_input(self):
        """The fix for replaying a whole transcript into a fresh session."""
        created = []
        resumed = []
        sent = []

        class _Session(_FakeSession):
            def __init__(self, session_id):
                super().__init__()
                self.session_id = session_id

            async def send(self, prompt):
                sent.append(prompt)

        class _Client:
            async def create_session(self, **options):
                created.append(options)
                return _Session("sdk-session-1")

            async def resume_session(self, session_id, **options):
                resumed.append(session_id)
                return _Session(session_id)

        first_body = {"input": [{"role": "user", "content": "hello"}], "session_id": "thread-A"}
        with patch.object(sdk, "_get_client", return_value=_Client()):
            session, dispatch = await sdk._open_session(first_body, sdk.ToolRegistration())
            await dispatch()
        self.assertEqual(len(created), 1)
        self.assertEqual(sent, ["User: hello"])

        # The turn succeeded, so the watermark becomes durable.
        sdk._commit_alias_watermark(session.session_id, success=True)

        second_body = {
            "input": [
                {"role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "hi"},
                {"role": "user", "content": "next question"},
            ],
            "session_id": "thread-A",
        }
        with patch.object(sdk, "_get_client", return_value=_Client()):
            _, dispatch = await sdk._open_session(second_body, sdk.ToolRegistration())
            await dispatch()

        # Resumed rather than recreated, and only the new user turn was sent.
        self.assertEqual(resumed, ["sdk-session-1"])
        self.assertEqual(len(created), 1)
        self.assertEqual(sent[1], "User: next question")

    async def test_open_session_does_not_commit_watermark_for_a_failed_turn(self):
        sdk._pending_alias_watermark["sdk-session-9"] = ("thread-B", ["abc"])
        sdk._commit_alias_watermark("sdk-session-9", success=False)
        self.assertIsNone(sdk._session_for_alias("thread-B"))

    async def test_tool_continuation_falls_back_to_session_send_when_pending_tool_call_rejected(self):
        sent = []
        handled = []

        class _RpcTools:
            async def handle_pending_tool_call(self, req):
                handled.append(req)
                return SimpleNamespace(success=False)

        class _Rpc:
            tools = _RpcTools()

        class _Session(_FakeSession):
            def __init__(self, session_id):
                super().__init__()
                self.session_id = session_id
                self.rpc = _Rpc()

            async def send(self, prompt):
                sent.append(prompt)

        class _Client:
            async def resume_session(self, session_id, **options):
                return _Session(session_id)

        call_id = sdk._encode_call_id("sdk-sess-1", "req-1", tool_name="bash", tool_type="function")
        body = {
            "input": [
                {"role": "user", "content": "run ls"},
                {"type": "function_call", "call_id": call_id, "name": "bash", "arguments": '{"cmd":"ls"}'},
                {"type": "function_call_output", "call_id": call_id, "output": "file1.txt"},
            ],
            "session_id": "thread-C",
        }
        sdk._remember_session("sdk-sess-1")
        with patch.object(sdk, "_get_client", return_value=_Client()):
            session, dispatch = await sdk._open_session(body, sdk.ToolRegistration())
            await dispatch()

        self.assertEqual(len(handled), 1)
        self.assertEqual(handled[0].request_id, "req-1")
        self.assertEqual(len(sent), 1)
        self.assertIn("Tool result for bash: file1.txt", sent[0])

    async def test_tool_continuation_falls_back_to_create_session_when_resume_fails(self):
        created = []
        sent = []

        class _Session(_FakeSession):
            def __init__(self, session_id):
                super().__init__()
                self.session_id = session_id

            async def send(self, prompt):
                sent.append(prompt)

        class _Client:
            async def resume_session(self, session_id, **options):
                raise RuntimeError("Session state missing on disk")

            async def create_session(self, **options):
                created.append(options)
                return _Session("new-sdk-sess")

        call_id = sdk._encode_call_id("lost-sess-1", "req-1", tool_name="bash", tool_type="function")
        body = {
            "input": [
                {"role": "user", "content": "run ls"},
                {"type": "function_call", "call_id": call_id, "name": "bash", "arguments": '{"cmd":"ls"}'},
                {"type": "function_call_output", "call_id": call_id, "output": "file1.txt"},
            ],
            "session_id": "thread-D",
        }
        sdk._remember_session("lost-sess-1")
        with patch.object(sdk, "_get_client", return_value=_Client()):
            session, dispatch = await sdk._open_session(body, sdk.ToolRegistration())
            await dispatch()

        self.assertEqual(len(created), 1)
        self.assertEqual(session.session_id, "new-sdk-sess")
        self.assertEqual(len(sent), 1)
        self.assertIn("Tool result: file1.txt", sent[0])


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

    async def test_tool_call_captures_session_shutdown_usage(self):
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
            session.emit(
                "session.shutdown",
                SimpleNamespace(
                    token_details=None,
                    model_metrics={
                        "gpt-test": SimpleNamespace(
                            usage=SimpleNamespace(
                                input_tokens=400,
                                cache_read_tokens=300,
                                cache_write_tokens=0,
                                output_tokens=50,
                                reasoning_tokens=0,
                            )
                        )
                    },
                ),
            )

        outcome = await sdk._wait_for_outcome(session, dispatch, registration)
        self.assertEqual(len(outcome.calls), 1)
        self.assertEqual(outcome.usage["input_tokens"], 400)
        self.assertEqual(outcome.usage["cached_input_tokens"], 300)
        self.assertEqual(outcome.usage["output_tokens"], 50)

    async def test_stream_turn_tool_call_captures_session_shutdown_usage(self):
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
            session.emit(
                "session.shutdown",
                SimpleNamespace(
                    token_details=None,
                    model_metrics={
                        "gpt-test": SimpleNamespace(
                            usage=SimpleNamespace(
                                input_tokens=600,
                                cache_read_tokens=450,
                                cache_write_tokens=0,
                                output_tokens=35,
                                reasoning_tokens=0,
                            )
                        )
                    },
                ),
            )

        chunks = [
            chunk.decode()
            async for chunk in sdk._stream_turn(
                _ConnectedRequest(),
                {"model": "gpt-test"},
                session,
                dispatch,
                registration,
            )
        ]
        completed = [c for c in chunks if "response.completed" in c]
        self.assertEqual(len(completed), 1)
        data = json.loads(completed[0].replace("event: response.completed\ndata: ", "").strip())
        usage = data["response"]["usage"]
        self.assertEqual(usage["input_tokens"], 600)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 450)
        self.assertEqual(usage["output_tokens"], 35)

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

    async def test_wait_for_outcome_does_not_expose_internal_compaction_summary(self):
        session = _FakeSession()

        async def dispatch():
            session.emit(
                "session.compaction_complete",
                SimpleNamespace(summary_content="internal SDK summary"),
            )
            session.emit(
                "assistant.message",
                AssistantMessageData(content="visible answer", message_id="m-1"),
            )
            session.emit("session.idle", SessionIdleData())

        outcome = await sdk._wait_for_outcome(session, dispatch, sdk.ToolRegistration())
        self.assertEqual(outcome.text, "visible answer")

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

    def test_extract_shutdown_usage_prefers_token_details_over_model_metrics(self):
        """tokenDetails is the session-wide superset; modelMetrics undercounts."""
        usage = sdk._extract_shutdown_usage({
            "tokenDetails": {
                "input": {"tokenCount": 282},
                "cache_read": {"tokenCount": 3076421},
                "cache_write": {"tokenCount": 255349},
                "output": {"tokenCount": 38342},
            },
            "modelMetrics": {
                "gpt-5.6-luna": {
                    "usage": {
                        "inputTokens": 3006170,
                        "outputTokens": 33717,
                        "cacheReadTokens": 2752601,
                        "cacheWriteTokens": 253437,
                        "reasoningTokens": 26418,
                    }
                }
            },
        })
        self.assertEqual(usage["input_tokens"], 282 + 3076421 + 255349)
        self.assertEqual(usage["cached_input_tokens"], 3076421)
        self.assertEqual(usage["cache_creation_input_tokens"], 255349)
        self.assertEqual(usage["fresh_input_tokens"], 282 + 255349)
        self.assertEqual(usage["output_tokens"], 38342)
        # Cache creation is fresh input; only cache reads are excluded.
        self.assertEqual(usage["pricing_fresh_input_tokens"], 282 + 255349)
        # Reasoning only ever appears under modelMetrics.
        self.assertEqual(usage["reasoning_output_tokens"], 26418)

    def test_dashboard_keeps_cached_sdk_tokens_in_total_volume(self):
        """Fresh input is a cost bucket, not the request's token total."""
        import dashboard

        event = {
            "requested_model": "gpt-5.6-luna",
            "finished_at": "2026-09-03T19:31:00+00:00",
            "usage": {
                "input_tokens": 65_087,
                "cached_input_tokens": 54_898,
                "cache_creation_input_tokens": 10_186,
                "fresh_input_tokens": 10_189,
                "output_tokens": 936,
                "total_tokens": 66_023,
            },
        }

        prepared = dashboard._prepare_usage_event(event)
        self.assertEqual(prepared["input_tokens"], 10_189)
        self.assertEqual(prepared["total_tokens"], 66_023)

    def test_cache_creation_is_part_of_fresh_input_but_not_double_billed(self):
        import util

        usage = sdk._extract_shutdown_usage({
            "tokenDetails": {
                "input": {"tokenCount": 10},
                "cache_read": {"tokenCount": 70},
                "cache_write": {"tokenCount": 20},
                "output": {"tokenCount": 0},
            },
        })
        # Total input contains direct input, cache reads, and cache writes.
        # Fresh input is total input less only cache reads.
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["fresh_input_tokens"], 30)
        self.assertEqual(usage["cached_input_tokens"], 70)

        breakdown = util._usage_event_cost_breakdown("gpt-5.6-luna", usage)
        # 10 direct fresh tokens at $0.20/M, plus 20 cache writes at $0.25/M.
        self.assertAlmostEqual(breakdown["input_fresh"], 10 * 0.20 / 1_000_000)
        self.assertAlmostEqual(breakdown["cache_creation"], 20 * 0.25 / 1_000_000)

    def test_resume_delta_returns_only_new_user_segments(self):
        first = [{"role": "user", "content": "hello"}]
        segments = sdk._render_input_segments(first)
        seen = sdk._segment_fingerprints(segments)

        follow_up = first + [
            {"type": "message", "role": "assistant", "content": "hi there"},
            {"type": "function_call", "name": "ls", "arguments": "{}"},
            {"type": "function_call_output", "output": "a.py"},
            {"role": "user", "content": "now what?"},
        ]
        segments2 = sdk._render_input_segments(follow_up)
        delta = sdk._resume_delta(segments2, sdk._segment_fingerprints(segments2), seen)

        # Only the new user turn: the session already holds its own reply and
        # the tool call/result it executed.
        self.assertEqual(delta, ["User: now what?"])

    def test_resume_delta_bails_out_when_history_diverges(self):
        segments = sdk._render_input_segments([{"role": "user", "content": "hello"}])
        stale = sdk._segment_fingerprints(
            sdk._render_input_segments([{"role": "user", "content": "different"}])
        )
        self.assertEqual(sdk._resume_delta(segments, sdk._segment_fingerprints(segments), stale), [])

        # Nothing new to send is also a bail-out.
        fresh = sdk._segment_fingerprints(segments)
        self.assertEqual(sdk._resume_delta(segments, fresh, fresh), [])
        # No watermark at all means a full replay.
        self.assertEqual(sdk._resume_delta(segments, fresh, []), [])

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

    def test_input_to_prompt_preserves_preamble_and_task_for_codex_compaction(self):
        fake_enc = format_translation.encode_fake_compaction("Investigation focused on RTSS limiter.")
        items = [
            {"type": "message", "role": "developer", "content": "<skills_instructions>\n## Skills\nAvailable skills...</skills_instructions>"},
            {"type": "message", "role": "user", "content": "<environment_context>\n  <cwd>/Users/chasepayne/sources/vibeshine</cwd>\n</environment_context>"},
            {"type": "message", "role": "user", "content": "Getting reports that the windows version frame limiter is broken"},
            {"type": "compaction", "encrypted_content": fake_enc},
        ]
        prompt = sdk.input_to_prompt(items)
        self.assertIn("Developer: <skills_instructions>", prompt)
        self.assertIn("User: <environment_context>", prompt)
        self.assertIn("User: Getting reports that the windows version frame limiter is broken", prompt)
        self.assertIn("Assistant: [Compacted conversation summary]\nInvestigation focused on RTSS limiter.", prompt)
        self.assertIn("User: Please continue and complete your response", prompt)

    def test_input_to_prompt_preserves_preamble_when_intermediate_turns_compacted(self):
        fake_enc = format_translation.encode_fake_compaction("Summary of earlier work")
        items = [
            {"type": "message", "role": "developer", "content": "<skills_instructions>skills</skills_instructions>"},
            {"type": "message", "role": "user", "content": "<environment_context><cwd>/repo</cwd></environment_context>"},
            {"type": "message", "role": "user", "content": "Old user turn"},
            {"type": "message", "role": "assistant", "content": "Old assistant turn"},
            {"type": "compaction", "encrypted_content": fake_enc},
            {"type": "message", "role": "user", "content": "New user follow-up"},
        ]
        prompt = sdk.input_to_prompt(items)
        self.assertIn("Developer: <skills_instructions>skills</skills_instructions>", prompt)
        self.assertIn("User: <environment_context><cwd>/repo</cwd></environment_context>", prompt)
        self.assertNotIn("Old user turn", prompt)
        self.assertNotIn("Old assistant turn", prompt)
        self.assertIn("Assistant: [Compacted conversation summary]\nSummary of earlier work", prompt)
        self.assertIn("User: New user follow-up", prompt)
        self.assertNotIn("Please continue and complete your response", prompt)

    def test_input_to_prompt_preserves_task_without_post_compaction_user_message(self):
        fake_enc = format_translation.encode_fake_compaction("Summary of ongoing work")
        items = [
            {"type": "message", "role": "developer", "content": "Developer instructions"},
            {"type": "message", "role": "user", "content": "Active task to solve"},
            {"type": "message", "role": "assistant", "content": "Initial assistant thought"},
            {"type": "compaction", "encrypted_content": fake_enc},
        ]
        prompt = sdk.input_to_prompt(items)
        self.assertIn("Developer: Developer instructions", prompt)
        self.assertIn("User: Active task to solve", prompt)
        self.assertNotIn("Initial assistant thought", prompt)
        self.assertIn("Assistant: [Compacted conversation summary]\nSummary of ongoing work", prompt)
        self.assertIn("User: Please continue and complete your response", prompt)

    def test_input_to_prompt_skips_subagent_notification_messages(self):
        items = [
            {"type": "message", "role": "user", "content": "<subagent_notification>\n{\"status\": \"completed\"}\n</subagent_notification>"},
            {"type": "message", "role": "user", "content": "Real user instruction"},
        ]
        prompt = sdk.input_to_prompt(items)
        self.assertNotIn("<subagent_notification>", prompt)
        self.assertIn("Real user instruction", prompt)

    def test_to_compaction_payload(self):
        outcome = sdk.TurnOutcome(
            text="Summary of earlier code work.",
            usage={
                "input_tokens": 500,
                "output_tokens": 100,
                "cached_input_tokens": 400,
                "cache_creation_input_tokens": 20,
                "reasoning_output_tokens": 15,
            },
        )
        payload = sdk.to_compaction_payload({"model": "gpt-5.6-luna"}, "sess-1", outcome, "resp-1")
        self.assertEqual(payload["id"], "resp-1")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output_text"], "Summary of earlier code work.")
        self.assertEqual(len(payload["output"]), 1)
        self.assertEqual(payload["output"][0]["type"], "compaction")
        enc = payload["output"][0]["encrypted_content"]
        self.assertEqual(format_translation.decode_fake_compaction(enc), "Summary of earlier code work.")
        self.assertEqual(payload["usage"]["input_tokens"], 500)
        self.assertEqual(payload["usage"]["input_tokens_details"]["cached_tokens"], 400)
        self.assertEqual(payload["usage"]["input_tokens_details"]["cache_creation_input_tokens"], 20)
        self.assertEqual(payload["usage"]["output_tokens_details"]["reasoning_tokens"], 15)


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
            self.assertEqual(plan.usage_event.get("session_id_origin"), "copilot_sdk")
            self.assertEqual(len(finished_events), 1)
            self.assertEqual(finished_events[0][1], 200)
            # Verify session was NOT deleted
            mock_delete.assert_not_called()

    async def test_stream_records_finished_turn_before_cancelled_disconnect(self):
        class _CancelledDisconnectSession(_FakeSession):
            async def disconnect(self):
                raise asyncio.CancelledError

        session = _CancelledDisconnectSession()

        async def dispatch():
            session.emit(
                "assistant.message_delta",
                AssistantMessageDeltaData(delta_content="done", message_id="msg-1"),
            )
            session.emit("session.idle", SessionIdleData())

        finished_events = []
        async for _chunk in sdk._stream_turn(
            _ConnectedRequest(),
            {"model": "gpt-test"},
            session,
            dispatch,
            sdk.ToolRegistration(),
            plan=SimpleNamespace(),
            finish_usage_callback=lambda plan, status, **kwargs: finished_events.append(
                (status, kwargs)
            ),
        ):
            pass

        self.assertEqual(len(finished_events), 1)
        self.assertEqual(finished_events[0][0], 200)

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

    def test_scan_session_state_skips_sessions_owned_by_request_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sess_dir = os.path.join(temp_dir, "session-state", "owned-session")
            os.makedirs(sess_dir, exist_ok=True)
            with open(os.path.join(sess_dir, "events.jsonl"), "w") as f:
                f.write(json.dumps({
                    "type": "assistant.turn_start",
                    "timestamp": "2026-09-02T12:00:01Z",
                    "data": {"interactionId": "turn-1"},
                }) + "\n")
                f.write(json.dumps({
                    "type": "session.shutdown",
                    "timestamp": "2026-09-02T12:00:02Z",
                    "data": {"modelMetrics": {"model": {"usage": {"inputTokens": 100}}}},
                }) + "\n")

            recorded = []
            with patch.object(sdk, "_SDK_STATE_DIR", temp_dir), \
                 patch.object(sdk, "_SESSION_STATE_DIR", os.path.join(temp_dir, "session-state")), \
                 patch.object(sdk, "_INGEST_CURSOR_FILE", os.path.join(temp_dir, "session-cursor.json")), \
                 patch.object(sdk, "_owns_session", return_value=True):
                self.assertEqual(sdk.scan_session_state(recorded.append), 0)
            self.assertEqual(recorded, [])

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
        self.assertEqual(u["input_tokens_details"]["cached_tokens"], 100)
        self.assertEqual(u["output_tokens_details"]["reasoning_tokens"], 5)



if __name__ == "__main__":
    unittest.main()
