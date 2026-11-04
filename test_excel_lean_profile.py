"""Offline coverage for the external client's smaller gateway profile."""

import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import excel_upstream as excel
from test_excel_tool_disclosure import source_catalog, complete_disclosure


def native_response(envelope, *, name=None, call_id="call_lean_test"):
    return {"output": [{
        "type": "function_call", "id": "fc_lean_test",
        "call_id": call_id, "status": "completed",
        "name": name or excel.LEAN_CLIENT_TRANSPORT_NAME,
        "arguments": json.dumps(excel._lean_transport_arguments(envelope)),
    }]}


class ExcelLeanProfileTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("LEAN_EXTERNAL_PROFILE", True), ("LAZY_TOOL_CATALOG", True),
                            ("CATALOG_AT_PROMPT_END", False)):
            flag = patch.object(excel, name, value)
            flag.start()
            self.addCleanup(flag.stop)
        self.source = source_catalog()

    def store(self):
        store = excel.ExcelSessionStore()
        store.configure({
            "authorization": "Bearer test-token", "chatgpt-account-id": "test-account",
            "user-agent": "Office test",
            "x-openai-internal-basispoints-browser-name": "Chrome",
        }, persist=False)
        return store

    def test_only_external_requests_drop_office_identity_not_authentication(self):
        store = self.store()
        office = store.request_headers(stream=True)
        lean = store.request_headers(stream=True, external_client=True)
        self.assertEqual(office["x-openai-internal-basispoints-client-agent-profile"], "excel")
        for name in ("authorization", "chatgpt-account-id", "x-openai-account-id",
                     "x-basispoints-auth-mode", "origin", "accept", "content-type"):
            self.assertEqual(lean[name], office[name])
        self.assertNotIn("user-agent", lean)
        self.assertFalse(any(key.startswith(("x-openai-internal-basispoints-", "x-stainless-"))
                             for key in lean))
        self.assertEqual(store.request_headers(stream=True), office)

    def test_legacy_optout_preserves_headers_and_tools_version(self):
        store = self.store()
        version = "tools-excel-core-2026-06-16-3af59f22"
        with patch.object(excel, "LEAN_EXTERNAL_PROFILE", False):
            self.assertEqual(store.request_headers(stream=False, external_client=True),
                             store.request_headers(stream=False))
            body = excel.prepare_responses_body(self.source, tools_version_id=version)
        self.assertEqual(body["metadata"][excel.TOOLS_VERSION_METADATA_KEY], version)
        self.assertFalse(any(item.get("name") == "list_connectors" for item in body["input"]))

    def test_default_drops_both_captured_and_caller_excel_tools_pin(self):
        self.source["metadata"] = {excel.TOOLS_VERSION_METADATA_KEY: "caller-pin", "keep": "yes"}
        body = excel.prepare_responses_body(self.source, tools_version_id="tools-excel-core-2026-06-16-3af59f22")
        self.assertNotIn(excel.TOOLS_VERSION_METADATA_KEY, body["metadata"])
        self.assertEqual(body["metadata"]["keep"], "yes")

    def test_catalog_is_local_exact_marker_and_not_a_remote_lookup(self):
        body = excel.prepare_responses_body(self.source)
        call, output = body["input"][2:4]
        self.assertEqual(call["name"], "list_connectors")
        self.assertEqual(call["call_id"], output["call_id"])
        catalog = json.loads(output["output"])
        self.assertEqual(catalog["source"], "ghcp_proxy_local_catalog")
        self.assertIs(catalog["remote_connectors_fetched"], False)
        self.assertIsNone(catalog["next_cursor"])
        self.assertEqual(len(catalog["actions"]), 1)
        action = catalog["actions"][0]
        self.assertEqual(action["action_ref"], excel.LEAN_CLIENT_ACTION_REF)
        self.assertTrue(excel._value_matches_schema({"name": "exec_command", "arguments": {"cmd": "pwd"}},
                                                   action["schema"]))
        self.assertEqual(body["input"][-1]["role"], "user")

    def test_prefix_and_task_identity_stay_stable_without_mutating_source(self):
        original = copy.deepcopy(self.source)
        first = excel.prepare_responses_body(self.source)
        self.assertEqual(self.source, original)
        self.assertEqual(first, excel.prepare_responses_body(self.source))
        with patch.object(excel, "LEAN_EXTERNAL_PROFILE", False):
            old = excel.prepare_responses_body(self.source)
        for key in ("task_id", "turn_id", "agent_iteration"):
            self.assertEqual(first["metadata"][key], old["metadata"][key])
        next_turn = copy.deepcopy(self.source)
        next_turn["input"] = [first["input"][-1], {"role": "user", "content": "next"}]
        second = excel.prepare_responses_body(next_turn)
        self.assertEqual(first["input"][:-1], second["input"][:-2])
        self.assertEqual(first["metadata"]["task_id"], second["metadata"]["task_id"])

    def test_suffix_escape_hatch_keeps_terminal_compaction_last(self):
        trigger = {"type": "compaction_trigger"}
        self.source["input"] = [{"role": "user", "content": "hi"}, trigger]
        with patch.object(excel, "CATALOG_AT_PROMPT_END", True):
            body = excel.prepare_responses_body(self.source)
        self.assertEqual(body["input"][-1], trigger)
        self.assertEqual(body["input"][0]["name"], "list_connectors")

    def test_small_or_empty_catalogs_do_not_force_lazy_schema_loading(self):
        source = {"input": "hi"}
        body = excel.prepare_responses_body(source)
        self.assertFalse(any(item.get("name") == "list_connectors" for item in body["input"]))
        source["tools"] = self.source["tools"][:1]
        text = excel._client_tool_protocol_instructions(source)
        self.assertIn("exec_command", text)
        self.assertNotIn("Deferred tools", text)

    def test_function_envelope_and_native_identity_roundtrip(self):
        envelope = {"name": "exec_command", "arguments": {"cmd": "pwd"}}
        response = native_response(envelope)
        call = excel.extract_native_client_tool_call(response, self.source)
        self.assertEqual(call["name"], "exec_command")
        self.assertEqual(json.loads(call["arguments"]), envelope["arguments"])
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": "/tmp"}
        replay = excel.translate_input_items([call, output])
        self.assertEqual(replay[0], response["output"][0])
        self.assertEqual(replay[1], {**output, "id": "fc_call_lean_test"})

    def test_namespace_alias_and_original_schema_validation(self):
        for name in ("run_connector_action", "functions.run_connector_action"):
            valid = native_response({"name": "exec_command", "arguments": {"cmd": "pwd"}}, name=name)
            self.assertIsNotNone(excel.extract_native_client_tool_call(valid, self.source))
        for arguments in ({}, {"cmd": 12}, [], None):
            self.assertIsNone(excel.extract_native_client_tool_call(
                native_response({"name": "exec_command", "arguments": arguments}), self.source))

    def test_malformed_reserved_envelopes_never_become_real_connector_calls(self):
        source = {"tools": [{"type": "function", "name": "run_connector_action",
                             "parameters": {"type": "object"}}]}
        for params in (None, [], "{\"name\":\"exec_command\"}", 1):
            response = native_response({})
            response["output"][0]["arguments"] = json.dumps({
                "action_ref": excel.LEAN_CLIENT_ACTION_REF, "params": params})
            self.assertIsNone(excel.extract_native_client_tool_call(response, source))
        response = native_response({})
        response["output"][0]["name"] = []
        self.assertIsNone(excel.extract_native_client_tool_call(response, self.source))

    def test_real_registered_client_connector_tool_is_not_recursive_transport(self):
        for name in ("run_connector_action", "list_connectors"):
            source = {"tools": [{"type": "function", "name": name,
                                 "parameters": {"type": "object"}}]}
            envelope = {"name": name, "arguments": {"action_ref": "real-client-action", "params": {}}}
            call = excel.extract_native_client_tool_call(native_response(envelope), source)
            self.assertEqual(call["name"], name)
            self.assertEqual(json.loads(call["arguments"]), envelope["arguments"])

    def test_lazy_lookup_and_namespaced_action_replay(self):
        response = native_response({"name": excel.TOOL_SCHEMA_LOOKUP_NAME,
                                    "arguments": {"names": ["mcp__demo.tool_0"]}})
        call = excel.extract_native_client_tool_call(response, self.source)
        self.assertEqual(call["namespace"], "mcp__node_repl")
        completed = complete_disclosure(self.source, call)
        action = excel.extract_native_client_tool_call(native_response(
            {"name": "mcp__demo.tool_0", "arguments": {"key": "abc"}}, call_id="call_after_lookup"), completed)
        self.assertEqual(action["name"], "tool_0")
        self.assertEqual(action["namespace"], "mcp__demo")
        replay = excel.translate_input_items(completed["input"])
        self.assertEqual(replay[-2], response["output"][0])

    def test_custom_tool_output_and_restart_fallback_are_native_function_items(self):
        self.source["tools"] = [{"type": "custom", "name": "apply_patch"}]
        response = native_response({"name": "apply_patch", "input": "*** patch ***"})
        call = excel.extract_native_client_tool_call(response, self.source)
        self.assertEqual(call["input"], "*** patch ***")
        output = {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "done"}
        with patch.object(excel, "_remembered_native_call", return_value=None):
            replay = excel.translate_input_items([call, output])
        self.assertEqual(replay[0]["name"], "run_connector_action")
        self.assertEqual(excel._transport_envelope(replay[0]), {"name": "apply_patch", "input": "*** patch ***"})
        self.assertEqual(replay[1]["type"], "function_call_output")
        self.assertEqual(replay[1]["call_id"], replay[0]["call_id"])

    def test_legacy_office_history_migrates_only_decodable_relay(self):
        envelope = {"name": "exec_command", "arguments": {"cmd": "pwd"}}
        native = native_response(envelope)["output"][0]
        native["name"] = "run_officejs"
        native["arguments"] = json.dumps({"code": json.dumps(envelope)})
        call = excel.extract_native_client_tool_call({"output": [native]}, self.source)
        replay = excel.translate_input_items([call])[0]
        self.assertEqual(replay["name"], "run_connector_action")
        for key in ("id", "call_id", "status"):
            self.assertEqual(replay[key], native[key])
        self.assertEqual(excel._transport_envelope(replay), envelope)
        self.assertEqual(excel._remembered_native_call(call["call_id"]), native)
        output = {"type": "function_call_output", "call_id": call["call_id"],
                  "output": "unsupported call: run_officejs"}
        retry = excel.translate_input_items([call, output])[1]["output"]
        self.assertIn("native run_connector_action", retry)
        self.assertIn("retry only once", retry)

    def test_streamed_native_relay_emits_client_tool_with_original_usage(self):
        import proxy
        import format_translation
        response = native_response({"name": "exec_command", "arguments": {"cmd": "pwd"}})
        response.update({"id": "resp_lean", "status": "completed",
                         "usage": {"input_tokens": 18000, "output_tokens": 40}})
        async def collect():
            async def raw():
                yield format_translation.sse_encode("response.completed", {"type": "response.completed", "response": response})
            return b"".join([chunk async for chunk in proxy._excel_tool_stream_transform(self.source)(raw())])
        wire = asyncio.run(collect()).decode()
        events = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]
        completed = [event for event in events if event["type"] == "response.completed"][-1]["response"]
        self.assertEqual(completed["output"][0]["name"], "exec_command")
        self.assertEqual(completed["usage"], response["usage"])
