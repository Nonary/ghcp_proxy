"""Focused, offline checks for Codex <-> Copilot SDK tool round trips."""

import asyncio
import base64
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from copilot.rpc import ExternalToolTextResultForLlm
from copilot.session_events import AssistantMessageData, ExternalToolRequestedData, SessionIdleData

import copilot_sdk_upstream as sdk
import format_translation
from codex_sdk_adapter import CodexSdkAdapter, decode_call_id, encode_call_id


def _event(tool_name, arguments, request_id="request-1"):
    return SimpleNamespace(tool_name=tool_name, arguments=arguments, request_id=request_id)


class CodexSdkAdapterTests(unittest.TestCase):
    def test_codex_v2_collaboration_encryption_is_adapted_to_explicit_plaintext(self):
        for name in ["spawn_agent", "send_message", "followup_task"]:
            with self.subTest(name=name):
                body = {"input": [{"type": "additional_tools", "tools": [
                    {"type": "namespace", "name": "collaboration", "tools": [{
                        "type": "function", "name": name, "parameters": {
                            "type": "object", "properties": {
                                "message": {"type": "string", "encrypted": True, "description": "Assignment text"},
                                "target": {"type": "string"},
                            }, "required": ["message"], "additionalProperties": False,
                        },
                    }]},
                ]}]}
                original = copy.deepcopy(body)
                adapter = CodexSdkAdapter.from_request(body)
                message_schema = adapter.tool_options[0]["parameters"]["properties"]["message"]
                self.assertNotIn("encrypted", message_schema)
                self.assertEqual(message_schema["description"], "Assignment text")
                message = "Run the second assignment and report the result."
                call = adapter.tool_call(_event("collaboration" + name, {"message": message, "target": "/root/worker"}))
                for completed in [False, True]:
                    item = adapter.tool_item("session-1", call, completed=completed)
                    self.assertEqual(item["encrypted_function_args"], [])
                    self.assertEqual(item["namespace"], "collaboration")
                    self.assertEqual(json.loads(item["arguments"])["message"], message)
                self.assertEqual(body, original)

    def test_unrelated_tool_encryption_is_not_disabled_or_marked_plaintext(self):
        for namespace, name in [("documents", "send_message"), ("collaboration", "other")]:
            with self.subTest(namespace=namespace, name=name):
                adapter = CodexSdkAdapter.from_request({"tools": [{
                    "type": "namespace", "name": namespace, "tools": [{
                        "type": "function", "name": name,
                        "parameters": {"type": "object", "properties": {"message": {"type": "string", "encrypted": True}}},
                    }],
                }]})
                self.assertTrue(adapter.tool_options[0]["parameters"]["properties"]["message"]["encrypted"])
                item = adapter.tool_item("session-1", adapter.tool_call(_event(namespace + name, {"message": "test"})))
                self.assertNotIn("encrypted_function_args", item)

    def test_agent_messages_are_caller_input_during_a_pending_tool_continuation(self):
        call_id = encode_call_id("session-1", "request-1", tool_name="read", tool_type="function")
        assignment = {"type": "agent_message", "author": "/root", "recipient": "/root/worker", "content": [
            {"type": "input_text", "text": "Continue with the second assignment."},
        ]}
        value = [{"type": "function_call_output", "call_id": call_id, "output": "ok"}, assignment]
        self.assertTrue(sdk._is_caller_message(assignment))
        self.assertEqual(CodexSdkAdapter.continuation(value)[0], "session-1")
        CodexSdkAdapter.validate_new_agent_message(value)

    def test_old_opaque_history_does_not_block_a_new_plaintext_assignment(self):
        old = {"type": "agent_message", "author": "/root", "recipient": "/root/worker", "content": [
            {"type": "input_text", "text": "Message Type: NEW_TASK\nTask name: /root/worker\nSender: /root\nPayload:\n"},
            {"type": "encrypted_content", "encrypted_content": "opaque-historical-assignment"},
        ]}
        previous = [old, {"type": "message", "role": "assistant", "content": "Completed"}]
        assignment = {"type": "agent_message", "author": "/root", "recipient": "/root/worker", "content": [
            {"type": "input_text", "text": "New plaintext follow-up assignment."},
        ]}
        value = previous + [assignment]
        CodexSdkAdapter.validate_new_agent_message(value)
        seen = sdk._segment_fingerprints(sdk._render_input_segments(previous))
        segments = sdk._render_input_segments(value)
        self.assertEqual(sdk._resume_delta(segments, sdk._segment_fingerprints(segments), seen),
                         ["User: New plaintext follow-up assignment."])

    def test_legacy_and_lite_catalogs_produce_the_same_sdk_tools(self):
        tools = [{"type": "namespace", "name": "docs", "description": "Read-only tools", "tools": [
            {"type": "function", "name": "read", "parameters": {"type": "object", "properties": {}}},
            {"type": "custom", "name": "query", "format": {"type": "text"}},
        ]}]
        standard = {"tools": tools}
        lite = {"input": [{"type": "additional_tools", "tools": tools}]}
        original = copy.deepcopy(lite)
        standard_adapter = CodexSdkAdapter.from_request(standard)
        lite_adapter = CodexSdkAdapter.from_request(lite)
        self.assertEqual(standard_adapter.tool_options, lite_adapter.tool_options)
        self.assertEqual(standard_adapter.names, lite_adapter.names)
        self.assertIn("Read-only tools", lite_adapter.tool_options[0]["description"])
        self.assertEqual(lite, original)

    def test_lite_updates_replace_old_schemas_without_reordering_aliases(self):
        older = [{"type": "function", "name": "read", "parameters": {"type": "object"}},
                 {"type": "function", "name": "write"}]
        newer = {"type": "function", "name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
        }}
        adapter = CodexSdkAdapter.from_request({"input": [
            {"type": "additional_tools", "tools": older},
            {"type": "additional_tools", "tools": [newer]},
        ]})
        self.assertEqual(list(adapter.names), ["read", "write"])
        self.assertEqual(adapter.tool_options[0]["parameters"], newer["parameters"])

    def test_tool_search_loaded_definitions_are_registered(self):
        adapter = CodexSdkAdapter.from_request({"input": [
            {"type": "tool_search_output", "tools": [{"type": "namespace", "name": "docs", "tools": [
                {"type": "function", "name": "search"},
            ]}]},
        ]})
        self.assertEqual(adapter.names["docssearch"].namespace, "docs")

    def test_current_standard_catalog_takes_precedence_over_replayed_lite_catalogs(self):
        current = {"type": "function", "name": "read", "description": "Current schema"}
        adapter = CodexSdkAdapter.from_request({"tools": [current], "input": [
            {"type": "additional_tools", "tools": [{"type": "function", "name": "read", "description": "Old schema"}]},
        ]})
        self.assertEqual(adapter.tool_options[0]["description"], "Current schema")

    def test_colliding_runtime_aliases_restore_distinct_client_identities(self):
        tools = [
            {"type": "namespace", "name": "ab", "tools": [{"type": "function", "name": "c"}]},
            {"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "bc"}]},
            {"type": "function", "name": "abc"},
        ]
        adapter = CodexSdkAdapter.from_request({"tools": tools})
        self.assertEqual(list(adapter.names), ["abc", "abc_1", "abc_2"])
        items = [adapter.tool_item("session-1", adapter.tool_call(_event(name, {}))) for name in adapter.names]
        self.assertEqual([(item.get("namespace"), item["name"]) for item in items],
                         [("ab", "c"), ("a", "bc"), (None, "abc")])
        self.assertEqual([decode_call_id(item["call_id"]).get("ns") for item in items], ["ab", "a", None])

    def test_namespaced_history_does_not_confuse_tools_with_the_same_leaf_name(self):
        items = [{"type": "function_call", "name": "read", "namespace": namespace, "arguments": "{}"}
                 for namespace in ["docs", "files", "functions"]]
        prompt = sdk.input_to_prompt(items)
        self.assertIn("Assistant tool call docs.read:", prompt)
        self.assertIn("Assistant tool call files.read:", prompt)
        self.assertIn("Assistant tool call read:", prompt)

    def test_custom_grammar_and_raw_arguments_survive_the_adapter(self):
        grammar = {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}
        adapter = CodexSdkAdapter.from_request({"tools": [{"type": "namespace", "name": "functions", "tools": [
            {"type": "custom", "name": "apply_patch", "format": grammar},
        ]}]})
        self.assertIn(grammar["definition"], adapter.tool_options[0]["description"])
        raw = "*** Begin Patch\n*** End Patch"
        for arguments in [raw, {"input": raw}, json.dumps({"input": raw}), {"patch": raw}]:
            with self.subTest(arguments=arguments):
                item = adapter.tool_item("session-1", adapter.tool_call(_event("ghcp_custom_apply_patch", arguments)))
                self.assertEqual(item["type"], "custom_tool_call")
                self.assertEqual(item["input"], raw)
                self.assertNotIn("namespace", item)

    def test_legacy_call_ids_remain_compatible(self):
        call_id = encode_call_id("session-1", "request-1", tool_name="read", tool_type="function")
        self.assertEqual(decode_call_id(call_id), {"s": "session-1", "r": "request-1", "n": "read", "t": "function"})
        result = CodexSdkAdapter.continuation([{"type": "function_call_output", "call_id": call_id, "output": "ok"}])
        self.assertEqual(result[1][0].output, "ok")
        self.assertIsNone(result[1][0].namespace)
        for invalid in ["other", "ghcpsdk_a", "ghcpsdk_!!!!", None]:
            self.assertIsNone(decode_call_id(invalid))

    def test_updated_catalog_and_steering_do_not_hide_pending_results(self):
        call_id = encode_call_id("session-1", "request-1", tool_name="read", tool_type="function", namespace="docs")
        value = [
            {"type": "function_call_output", "call_id": call_id, "output": "ok"},
            {"type": "additional_tools", "role": "developer", "tools": []},
            {"type": "tool_search_output", "tools": []},
            {"type": "message", "role": "user", "content": "Keep going"},
        ]
        session_id, results = CodexSdkAdapter.continuation(value)
        self.assertEqual(session_id, "session-1")
        self.assertEqual(results[0].namespace, "docs")

    def test_mismatched_output_types_and_namespaces_are_rejected(self):
        call_id = encode_call_id("session-1", "request-1", tool_name="read", tool_type="function", namespace="docs")
        for item in [
            {"type": "custom_tool_call_output", "call_id": call_id, "output": "ok"},
            {"type": "function_call_output", "call_id": call_id, "namespace": "files", "output": "ok"},
            {"type": "function_call_output", "call_id": call_id, "name": "write", "output": "ok"},
        ]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                CodexSdkAdapter.continuation([item])

        result = CodexSdkAdapter.continuation([
            {"type": "function_call_output", "call_id": call_id, "namespace": None, "name": None, "output": "ok"},
        ])
        self.assertEqual(result[1][0].namespace, "docs")

    def test_inline_image_results_reach_the_real_sdk_rpc_schema(self):
        image = base64.b64encode(b"synthetic-image").decode("ascii")
        call_id = encode_call_id("session-1", "request-1", tool_name="screenshot", tool_type="function")
        output = [
            {"type": "input_text", "text": "Before image"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
            {"type": "input_text", "text": "After image"},
        ]
        result = CodexSdkAdapter.continuation([{"type": "function_call_output", "call_id": call_id, "output": output}])[1][0]
        rpc_result = ExternalToolTextResultForLlm.from_dict(result.sdk_payload())
        self.assertEqual(rpc_result.to_dict()["contents"], [
            {"type": "text", "text": "Before image"},
            {"type": "image", "data": image, "mimeType": "image/png"},
            {"type": "text", "text": "After image"},
        ])
        self.assertEqual(result.image_attachments()[0]["data"], image)

    def test_images_are_preserved_only_for_the_sdk_sanitizer(self):
        item = {"type": "function_call_output", "call_id": "example", "content": [], "output": [
            {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
        ]}
        original = copy.deepcopy(item)
        sdk_item = format_translation.sanitize_input([item], preserve_tool_output_images=True)[0]
        rest_item = format_translation.sanitize_input([item])[0]
        self.assertEqual(sdk_item["output"], item["output"])
        self.assertNotIn("content", sdk_item)
        self.assertEqual(rest_item["output"][0]["type"], "input_text")
        self.assertEqual(item, original)

    def test_unsupported_images_fail_explicitly_without_fetching_urls(self):
        call_id = encode_call_id("session-1", "request-1", tool_name="read", tool_type="function")
        for url in ["https://example.invalid/image.png", "data:image/png;base64,!!!", "data:image/png;base64,"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                CodexSdkAdapter.continuation([{"type": "function_call_output", "call_id": call_id, "output": [
                    {"type": "input_image", "image_url": url},
                ]}])


class _Session:
    def __init__(self):
        self.session_id = "adapter-session"
        self.handlers = []
        self.pending_requests = []
        self.send_calls = []
        self.rpc = SimpleNamespace(tools=SimpleNamespace(handle_pending_tool_call=AsyncMock(side_effect=self.complete)))

    def on(self, handler):
        self.handlers.append(handler)
        return lambda: self.handlers.remove(handler)

    def emit(self, name, data):
        for handler in list(self.handlers):
            handler(SimpleNamespace(type=SimpleNamespace(value=name), data=data))

    async def send(self, prompt, **kwargs):
        self.send_calls.append((prompt, kwargs))
        for name, arguments, request_id in [
            ("docsread", {"path": "example"}, "function-request"),
            ("ghcp_custom_apply_patch", {"input": "*** Begin Patch\n*** End Patch"}, "custom-request"),
        ]:
            self.emit("external_tool.requested", ExternalToolRequestedData(
                request_id=request_id, session_id=self.session_id, tool_call_id=request_id,
                tool_name=name, arguments=arguments,
            ))

    async def complete(self, request):
        self.pending_requests.append(request)
        if len(self.pending_requests) == 2:
            self.emit("assistant.message", AssistantMessageData(content="Round trip completed", message_id="reply"))
            self.emit("session.idle", SessionIdleData())
        return SimpleNamespace(success=True)

    async def disconnect(self):
        pass


class _ConnectedRequest:
    async def is_disconnected(self):
        return False


class CodexSdkRoundTripTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_opaque_agent_assignment_is_rejected_instead_of_sending_a_blank_payload(self):
        body = {"input": [{"type": "agent_message", "author": "/root", "recipient": "/root/worker", "content": [
            {"type": "input_text", "text": "Message Type: NEW_TASK\nTask name: /root/worker\nSender: /root\nPayload:\n"},
            {"type": "encrypted_content", "encrypted_content": "opaque-assignment-from-the-old-transport"},
        ]}]}
        with patch.object(sdk, "_get_client", AsyncMock()) as get_client:
            response = await sdk.handle_responses(_ConnectedRequest(), body)
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot read an encrypted agent-message", json.loads(response.body)["error"]["message"])
        get_client.assert_not_awaited()

    async def test_plaintext_followup_reaches_a_completed_agent_in_json_and_sse(self):
        assignment = "Follow-up canary: inspect the new assignment and reply CANARY_RECEIVED."

        class ParentSession(_Session):
            async def send(self, prompt, **kwargs):
                self.send_calls.append((prompt, kwargs))
                self.emit("external_tool.requested", ExternalToolRequestedData(
                    request_id="followup-request", session_id=self.session_id, tool_call_id="followup-call",
                    tool_name="collaborationfollowup_task", arguments={"target": "/root/worker", "message": assignment},
                ))

        class CompletedChildSession(_Session):
            async def send(self, prompt, **kwargs):
                self.send_calls.append((prompt, kwargs))
                self.emit("assistant.message", AssistantMessageData(content="CANARY_RECEIVED", message_id="child-reply"))
                self.emit("session.idle", SessionIdleData())

        for stream in [False, True]:
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as state_dir:
                parent = ParentSession()
                child = CompletedChildSession()
                child.session_id = "completed-child"
                client = SimpleNamespace(create_session=AsyncMock(return_value=parent), resume_session=AsyncMock())
                parent_body = {"model": "gpt-test", "stream": stream, "tools": [{
                    "type": "namespace", "name": "collaboration", "tools": [{
                        "type": "function", "name": "followup_task", "parameters": {
                            "type": "object", "properties": {
                                "target": {"type": "string"}, "message": {"type": "string", "encrypted": True},
                            }, "required": ["target", "message"],
                        },
                    }],
                }], "input": [{"role": "user", "content": "Send the synthetic follow-up"}]}
                prior_child_input = [{"role": "user", "content": "First assignment"},
                                     {"role": "assistant", "content": "First assignment completed"}]
                child_body = {"model": "gpt-test", "stream": stream,
                              "client_metadata": {"thread_id": "completed-child-thread"}, "input": prior_child_input}
                fingerprints = sdk._segment_fingerprints(sdk._render_input_segments(prior_child_input))
                with patch.object(sdk, "_get_client", AsyncMock(return_value=client)), \
                     patch.object(sdk, "_SDK_STATE_DIR", state_dir), \
                     patch.object(sdk, "_live_sessions", {}), \
                     patch.object(sdk, "_pending_alias_watermark", {}), \
                     patch.object(sdk, "_session_for_alias", return_value=(child.session_id, fingerprints)), \
                     patch.object(sdk, "_PARALLEL_TOOL_SETTLE_SECONDS", 0.001):
                    try:
                        await sdk._track_live_session(child, options=sdk._session_options(child_body, sdk.ToolRegistration()),
                                                      fingerprints=fingerprints)
                        await sdk._release_session(child, sdk.TurnOutcome(text="First assignment completed"), completed=True)
                        payload, events = await self._payload(await sdk.handle_responses(_ConnectedRequest(), parent_body))
                        call = payload["output"][0]
                        self.assertEqual(call["encrypted_function_args"], [])
                        self.assertNotIn("encrypted", client.create_session.call_args.kwargs["tools"][0].parameters["properties"]["message"])
                        # Codex's current ToolCall::direct_source selects the
                        # plaintext AgentMessage path for this exact marker.
                        self.assertEqual((call["namespace"], call["name"]), ("collaboration", "followup_task"))
                        child_body["input"] = prior_child_input + [{
                            "type": "agent_message", "author": "/root", "recipient": "/root/worker",
                            "content": [{"type": "input_text", "text": json.loads(call["arguments"])["message"]}],
                        }]
                        result, _ = await self._payload(await sdk.handle_responses(_ConnectedRequest(), child_body))
                        self.assertEqual(child.send_calls, [("User: " + assignment, {})])
                        self.assertEqual(result["output"][0]["content"][0]["text"], "CANARY_RECEIVED")
                        client.resume_session.assert_not_awaited()
                        client.create_session.assert_awaited_once()
                        for event in events:
                            if event["type"] in {"response.output_item.added", "response.output_item.done"}:
                                self.assertEqual(event["item"]["encrypted_function_args"], [])
                    finally:
                        await sdk._evict_all_live_sessions()
                        await asyncio.sleep(0)

    async def test_lost_session_fallback_preserves_namespaced_history_and_images(self):
        image = base64.b64encode(b"synthetic-image").decode("ascii")
        call_id = encode_call_id("lost-session", "request-1", tool_name="read", tool_type="function", namespace="docs")
        body = {"input": [
            {"type": "function_call", "name": "read", "namespace": "docs", "call_id": call_id, "arguments": "{}"},
            {"type": "function_call_output", "call_id": call_id, "output": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
            ]},
        ]}
        session = SimpleNamespace(session_id="replacement-session", send=AsyncMock())
        client = SimpleNamespace(create_session=AsyncMock(return_value=session),
                                 resume_session=AsyncMock(side_effect=RuntimeError("Session lost")))
        with patch.object(sdk, "_get_client", AsyncMock(return_value=client)), \
             patch.object(sdk, "_reuse_live_session", AsyncMock(return_value=None)), \
             patch.object(sdk, "_owns_session", return_value=True), \
             patch.object(sdk, "_track_live_session", AsyncMock()), \
             patch.object(sdk, "_remember_session"), \
             patch.object(sdk, "_session_alias", return_value=None):
            _, dispatch = await sdk._open_session(body, sdk.ToolRegistration())
            await dispatch()
        self.assertIn("Assistant tool call docs.read:", session.send.call_args.args[0])
        self.assertEqual(session.send.call_args.kwargs["attachments"][0]["data"], image)

    async def test_invalid_tool_results_return_400_before_opening_an_sdk_client(self):
        call_id = encode_call_id("session-1", "request-1", tool_name="read", tool_type="function")
        body = {"input": [{"type": "custom_tool_call_output", "call_id": call_id, "output": "ok"}]}
        with patch.object(sdk, "_get_client", AsyncMock()) as get_client:
            response = await sdk.handle_responses(_ConnectedRequest(), body)
        self.assertEqual(response.status_code, 400)
        get_client.assert_not_awaited()

    async def test_json_and_sse_use_the_same_adapter_and_resume_the_same_session(self):
        for stream in [False, True]:
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as state_dir:
                session = _Session()
                client = SimpleNamespace(create_session=AsyncMock(return_value=session), resume_session=AsyncMock())
                body = {"model": "gpt-test", "stream": stream, "input": [
                    {"type": "additional_tools", "tools": [
                        {"type": "namespace", "name": "docs", "tools": [{"type": "function", "name": "read"}]},
                        {"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "apply_patch"}]},
                    ]},
                    {"type": "message", "role": "user", "content": "Run the synthetic tools"},
                ]}
                with patch.object(sdk, "_get_client", AsyncMock(return_value=client)), \
                     patch.object(sdk, "_SDK_STATE_DIR", state_dir), \
                     patch.object(sdk, "_live_sessions", {}), \
                     patch.object(sdk, "_pending_alias_watermark", {}), \
                     patch.object(sdk, "_PARALLEL_TOOL_SETTLE_SECONDS", 0.001):
                    try:
                        first, first_events = await self._payload(await sdk.handle_responses(_ConnectedRequest(), body))
                        calls = first["output"]
                        self.assertEqual([item["type"] for item in calls], ["function_call", "custom_tool_call"])
                        self.assertEqual((calls[0]["name"], calls[0]["namespace"]), ("read", "docs"))
                        self.assertEqual(calls[1]["name"], "apply_patch")
                        self.assertEqual(calls[1]["input"], "*** Begin Patch\n*** End Patch")
                        image = base64.b64encode(b"synthetic-image").decode("ascii")
                        follow_up = copy.deepcopy(body)
                        follow_up["input"].extend(calls + [
                            {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": [
                                {"type": "input_text", "text": "Read succeeded"},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
                            ]},
                            {"type": "custom_tool_call_output", "call_id": calls[1]["call_id"], "output": "Patch succeeded"},
                            {"type": "additional_tools", "tools": []},
                        ])
                        second, _ = await self._payload(await sdk.handle_responses(_ConnectedRequest(), follow_up))
                        self.assertEqual(second["output"][0]["content"][0]["text"], "Round trip completed")
                        self.assertEqual([r.request_id for r in session.pending_requests], ["function-request", "custom-request"])
                        self.assertEqual(session.pending_requests[0].result.to_dict()["contents"][1]["data"], image)
                        client.create_session.assert_awaited_once()
                        client.resume_session.assert_not_awaited()
                        self.assertEqual(len(session.send_calls), 1)
                        if stream:
                            for event in first_events:
                                if event["type"] == "response.output_item.done":
                                    self.assertIn(event["item"], calls)
                            self.assertIn("response.custom_tool_call_input.delta", [e["type"] for e in first_events])
                            self.assertIn("response.function_call_arguments.delta", [e["type"] for e in first_events])
                    finally:
                        await sdk._evict_all_live_sessions()
                        await asyncio.sleep(0)

    async def _payload(self, response):
        if hasattr(response, "body_iterator"):
            chunks = [chunk async for chunk in response.body_iterator]
            events = [json.loads(block.split("data: ", 1)[1])
                      for block in b"".join(chunks).decode().split("\n\n") if "data: " in block]
            completed = [event["response"] for event in events if event["type"] == "response.completed"]
            self.assertEqual(len(completed), 1)
            return completed[0], events
        self.assertEqual(response.status_code, 200, response.body)
        return json.loads(response.body), []


if __name__ == "__main__":
    unittest.main()
