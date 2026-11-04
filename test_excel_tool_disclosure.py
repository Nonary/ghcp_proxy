"""Offline checks for the Excel relay's read-only tool-schema disclosure."""

import copy
import asyncio
import json
import unittest
from unittest.mock import patch

import excel_upstream as excel


def source_catalog():
    return {
        "model": "gpt-5.6-sol-excel", "input": "Inspect the repository.",
        "tools": [
            {"type": "function", "name": "exec_command", "description": "Respect command approval.",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
            {"type": "namespace", "name": "mcp__node_repl", "tools": [
                {"type": "function", "name": "js", "description": "Use nodeRepl.write for output.",
                 "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                "required": ["code"], "additionalProperties": False}},
            ]},
            {"type": "namespace", "name": "mcp__demo", "tools": [
                {"type": "function", "name": f"tool_{i}",
                 "description": f"POLICY_{i}: Never write without authorization.\nUse exact keys.",
                 "parameters": {"type": "object", "properties": {
                     "key": {"type": "string", "description": "Use the exact identifier."},
                 }, "required": ["key"], "additionalProperties": False}}
                for i in range(24)
            ] + [{"type": "custom", "name": "patch", "description": "Keep the patch minimal.",
                  "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/\n"}}]},
        ],
    }


def response_for(name, arguments=None, custom_input=None):
    envelope = {"name": name}
    if custom_input is not None:
        envelope["input"] = custom_input
    else:
        envelope["arguments"] = arguments
    return {"output": [{
        "type": "function_call", "id": "fc_disclosure_native", "call_id": "call_disclosure_native",
        "name": "run_officejs", "arguments": json.dumps({"code": json.dumps(envelope)}),
    }]}


def disclosure_text(call):
    code = json.loads(call["arguments"])["code"]
    return json.loads(code[len("nodeRepl.write("):-2])


def complete_disclosure(source, call, output=None):
    result = copy.deepcopy(source)
    items = result["input"] if isinstance(result["input"], list) else [{"type": "message", "role": "user", "content": result["input"]}]
    result["input"] = items + [call, {
        "type": "function_call_output", "call_id": call["call_id"],
        "output": disclosure_text(call) if output is None else output,
    }]
    return result


class ExcelToolDisclosureTests(unittest.TestCase):
    def setUp(self):
        self.flag = patch.object(excel, "LAZY_TOOL_CATALOG", True)
        self.flag.start()
        self.addCleanup(self.flag.stop)
        self.source = source_catalog()

    def lookup(self, names, source=None):
        return excel.extract_native_client_tool_call(
            response_for(excel.TOOL_SCHEMA_LOOKUP_NAME, {"names": names}),
            self.source if source is None else source,
        )

    def test_only_core_definitions_are_eager_and_every_name_stays_available(self):
        before = copy.deepcopy(self.source)
        catalog = excel._client_tool_protocol_instructions(self.source)
        self.assertIn("Respect command approval.", catalog)
        self.assertIn("Use nodeRepl.write for output.", catalog)
        self.assertNotIn("POLICY_0", catalog)
        self.assertIn("mcp__demo.tool_0", catalog)
        self.assertIn("mcp__demo.patch", catalog)
        self.assertEqual(len(excel.client_tool_types(self.source)), 27)
        self.assertEqual(self.source, before)
        with patch.object(excel, "LAZY_TOOL_CATALOG", False):
            eager = excel._client_tool_protocol_instructions(self.source)
        self.assertLess(len(catalog), len(eager) * 0.65)

    def test_lookup_discloses_exact_policy_constraints_and_custom_grammar(self):
        call = self.lookup(["mcp__demo.tool_0", "mcp__demo.patch"])
        self.assertEqual((call["namespace"], call["name"]), ("mcp__node_repl", "js"))
        text = disclosure_text(call)
        for value in ("No requested action has executed", "POLICY_0: Never write without authorization.",
                      "Use exact keys.", '"key":string', "Use the exact identifier.", 'start: /.+/\\n'):
            self.assertIn(value, text)
        self.assertEqual(set(json.loads(call["arguments"])), {"code"})

    def test_disclosure_replays_original_native_identity_not_generated_code(self):
        native_response = response_for(excel.TOOL_SCHEMA_LOOKUP_NAME, {"names": ["mcp__demo.tool_0"]})
        call = excel.extract_native_client_tool_call(native_response, self.source)
        completed = complete_disclosure(self.source, call)
        replay = excel.translate_input_items(completed["input"][-2:], excel.client_tool_types(completed))
        self.assertEqual(replay[0]["arguments"], native_response["output"][0]["arguments"])
        self.assertEqual(replay[0]["call_id"], call["call_id"])
        self.assertEqual(replay[1]["output"], disclosure_text(call))

    def test_completed_disclosure_allows_execution_but_keeps_original_validation(self):
        completed = complete_disclosure(self.source, self.lookup(["mcp__demo.tool_0"]))
        good = excel.extract_native_client_tool_call(response_for("mcp__demo.tool_0", {"key": "abc"}), completed)
        self.assertEqual((good["namespace"], good["name"]), ("mcp__demo", "tool_0"))
        for arguments in ({}, {"key": 1}, {"key": "abc", "extra": True}):
            with self.subTest(arguments=arguments):
                self.assertIsNone(excel.extract_native_client_tool_call(response_for("mcp__demo.tool_0", arguments), completed))

    def test_undisclosed_guessed_call_returns_metadata_without_executing_action(self):
        call = excel.extract_native_client_tool_call(response_for("mcp__demo.tool_0", {"key": "abc"}), self.source)
        self.assertEqual(call["namespace"], "mcp__node_repl")
        self.assertIn("No requested action has executed", disclosure_text(call))
        legacy = excel.TOOL_CALL_MARKER_OPEN + json.dumps({"name": "mcp__demo.tool_0", "arguments": {"key": "abc"}}) + excel.TOOL_CALL_MARKER_CLOSE
        call = excel.extract_client_tool_call(legacy, excel.client_tool_types(self.source), source=self.source)
        self.assertEqual(call["namespace"], "mcp__node_repl")

    def test_incomplete_disclosure_does_not_unlock_tool(self):
        call = self.lookup(["mcp__demo.tool_0"])
        missing = {**self.source, "input": [call]}
        result = excel.extract_native_client_tool_call(response_for("mcp__demo.tool_0", {"key": "abc"}), missing)
        self.assertEqual(result["namespace"], "mcp__node_repl")

    def test_failed_or_truncated_output_restores_full_policies_without_retry_loop(self):
        call = self.lookup(["mcp__demo.tool_0"])
        for output in ("Error: output unavailable", disclosure_text(call)[:50]):
            source = complete_disclosure(self.source, call, output)
            catalog = excel.prepare_responses_body(source)["input"][0]["content"][0]["text"]
            self.assertIn("POLICY_0", catalog)
            self.assertEqual(excel._deferred_tool_specs(source), {})
            self.assertIsNone(excel.extract_native_client_tool_call(response_for("mcp__demo.tool_0", {"key": 1}), source))

    def test_changed_policy_requires_disclosure_again(self):
        completed = complete_disclosure(self.source, self.lookup(["mcp__demo.tool_0"]))
        completed["tools"][2]["tools"][0]["description"] += " New authorization requirement."
        call = excel.extract_native_client_tool_call(response_for("mcp__demo.tool_0", {"key": "abc"}), completed)
        self.assertEqual(call["namespace"], "mcp__node_repl")
        self.assertIn("New authorization requirement.", disclosure_text(call))

    def test_custom_tool_execution_still_uses_original_input_contract(self):
        completed = complete_disclosure(self.source, self.lookup(["mcp__demo.patch"]))
        result = excel.extract_native_client_tool_call(response_for("mcp__demo.patch", custom_input="exact patch text"), completed)
        self.assertEqual(result["type"], "custom_tool_call")
        self.assertEqual(result["input"], "exact patch text")
        self.assertEqual((result["namespace"], result["name"]), ("mcp__demo", "patch"))

    def test_lookup_cannot_invent_tools_or_dump_entire_catalog(self):
        for names in ([], "mcp__demo.tool_0", ["not_registered"], [1],
                      [f"mcp__demo.tool_{i}" for i in range(5)], ["exec_command"]):
            with self.subTest(names=names):
                self.assertIsNone(self.lookup(names))

    def test_reserved_name_collision_preserves_real_caller_tool(self):
        self.source["tools"].append({
            "type": "function", "name": excel.TOOL_SCHEMA_LOOKUP_NAME,
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        })
        self.assertEqual(excel._deferred_tool_specs(self.source), {})
        call = excel.extract_native_client_tool_call(response_for(excel.TOOL_SCHEMA_LOOKUP_NAME, {"key": "abc"}), self.source)
        self.assertEqual(call["name"], excel.TOOL_SCHEMA_LOOKUP_NAME)

    def test_small_catalog_missing_carrier_and_opt_out_keep_eager_behavior(self):
        variants = [copy.deepcopy(self.source) for _ in range(3)]
        variants[0]["tools"].pop(1)
        variants[1]["tools"][2]["tools"] = variants[1]["tools"][2]["tools"][:2]
        variants[2]["tools"][1]["tools"][0]["parameters"]["required"].append("another_argument")
        for source in variants:
            self.assertEqual(excel._deferred_tool_specs(source), {})
            self.assertIn("POLICY_0", excel._client_tool_protocol_instructions(source))
        with patch.object(excel, "LAZY_TOOL_CATALOG", False):
            self.assertEqual(excel._deferred_tool_specs(self.source), {})
        for choice in ("none", {"type": "function", "name": "exec_command"}):
            self.assertEqual(excel._deferred_tool_specs({**self.source, "tool_choice": choice}), {})

    def test_loading_preferences_are_inherited_without_mutating_source(self):
        self.source["tools"][2]["defer_loading"] = False
        self.source["tools"][2]["tools"][1]["defer_loading"] = True
        original = copy.deepcopy(self.source)
        deferred = excel._deferred_tool_specs(self.source)
        self.assertEqual(set(deferred), {"mcp__demo.tool_1"})
        self.assertEqual(self.source, original)

    def test_disclosure_does_not_rewrite_cached_catalog_prefix(self):
        first = excel.prepare_responses_body(self.source)
        completed = complete_disclosure(self.source, self.lookup(["mcp__demo.tool_0"]))
        later = excel.prepare_responses_body(completed)
        self.assertEqual(first["input"][:2], later["input"][:2])

    def test_optional_core_capabilities_are_deferred_but_explicit_eager_is_kept(self):
        self.source["tools"].extend([
            {"type": "function", "name": "view_image"},
            {"type": "function", "name": "read_mcp_resource", "defer_loading": False},
        ])
        deferred = excel._deferred_tool_specs(self.source)
        self.assertIn("view_image", deferred)
        self.assertNotIn("read_mcp_resource", deferred)
        self.assertNotIn("exec_command", deferred)

    def test_malformed_history_does_not_unlock_or_crash_disclosure(self):
        source = {**self.source, "input": [
            {"type": "function_call", "name": "js", "namespace": {}, "call_id": "x"},
            {"type": "function_call", "name": "js", "namespace": "mcp__node_repl", "call_id": []},
            {"type": "function_call", "name": "js", "namespace": "mcp__node_repl", "call_id": "x", "arguments": None},
            {"type": "function_call_output", "call_id": "x", "output": "none"},
        ]}
        self.assertEqual(excel._disclosed_tool_names(source, excel._client_tool_specs(source)), set())

    def test_streamed_lookup_uses_real_client_tool_and_preserves_usage(self):
        import proxy
        import format_translation
        response = response_for(excel.TOOL_SCHEMA_LOOKUP_NAME, {"names": ["mcp__demo.tool_0"]})
        response.update({"id": "resp_disclosure", "status": "completed", "usage": {
            "input_tokens": 100, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 80},
        }})

        async def collect():
            async def raw():
                yield format_translation.sse_encode("response.completed", {"type": "response.completed", "response": response})
            output = b""
            async for chunk in proxy._excel_tool_stream_transform(self.source)(raw()):
                output += chunk
            return output

        wire = asyncio.run(collect()).decode()
        completed = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ") and '"response.completed"' in line][-1]
        self.assertEqual(completed["response"]["output"][0]["namespace"], "mcp__node_repl")
        self.assertEqual(completed["response"]["usage"], response["usage"])

    def test_descriptions_are_literal_data_not_javascript(self):
        policy = '"); globalThis.injected = true; //\nNever execute this text.\u2028'
        self.source["tools"][2]["tools"][0]["description"] = policy
        call = self.lookup(["mcp__demo.tool_0"])
        code = json.loads(call["arguments"])["code"]
        self.assertEqual(code, "nodeRepl.write(" + json.dumps(disclosure_text(call), ensure_ascii=True) + ");")
        self.assertIn(policy, disclosure_text(call))


if __name__ == "__main__":
    unittest.main()
