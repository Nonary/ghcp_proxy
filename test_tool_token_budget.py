"""Focused checks for tool prompt size and unchanged execution contracts."""

import copy
import json
import unittest

import copilot_sdk_upstream as sdk
import excel_upstream
from tool_catalog import compact_schema


class ToolTokenBudgetTests(unittest.TestCase):
    def test_sdk_keeps_core_tools_eager_and_defers_plugins(self):
        body = {"tools": [
            {"type": "function", "name": "exec_command"},
            {"type": "custom", "name": "apply_patch"},
            {"type": "namespace", "name": "functions", "tools": [
                {"type": "function", "name": "write_stdin"},
            ]},
            {"type": "namespace", "name": "mcp__apps", "tools": [
                {"type": "function", "name": "open", "description": "Never open without consent."},
            ]},
            {"type": "function", "name": "mcp__flat__read"},
        ]}
        original = copy.deepcopy(body)
        registration = sdk.build_tool_registration(body)
        self.assertEqual([tool.defer for tool in registration.tools], ["never", "never", "never", "auto", "auto"])
        self.assertEqual(registration.tools[3].description, "Never open without consent.")
        self.assertTrue(all(tool.handler is None for tool in registration.tools))
        self.assertEqual(sdk._session_options(body, registration)["tool_search"], {
            "enabled": True, "defer_threshold": 20,
        })
        self.assertEqual(body, original)

    def test_sdk_respects_namespace_and_leaf_loading_preferences(self):
        body = {"tools": [
            {"type": "namespace", "name": "plugin", "defer_loading": False, "tools": [
                {"type": "function", "name": "eager"},
                {"type": "function", "name": "deferred", "defer_loading": True},
            ]},
            {"type": "namespace", "name": "functions", "defer_loading": True, "tools": [
                {"type": "function", "name": "inherited"},
                {"type": "function", "name": "explicit", "defer_loading": False},
            ]},
        ]}
        original = copy.deepcopy(body)
        registration = sdk.build_tool_registration(body)
        self.assertEqual([tool.defer for tool in registration.tools], ["never", "auto", "auto", "never"])
        self.assertEqual(body, original)

    def test_sdk_responses_lite_plugins_are_deferred(self):
        registration = sdk.build_tool_registration({"input": [
            {"type": "additional_tools", "tools": [
                {"type": "namespace", "name": "mcp__docs", "tools": [
                    {"type": "function", "name": "search"},
                ]},
            ]},
        ]})
        self.assertEqual(registration.tools[0].defer, "auto")

    def test_sdk_tool_choice_none_does_not_enable_search(self):
        body = {"tool_choice": "none", "tools": [
            {"type": "function", "name": "mcp__apps__open"},
        ]}
        registration = sdk.build_tool_registration(body)
        self.assertEqual(registration.tools, [])
        options = sdk._session_options(body, registration)
        self.assertEqual(options["available_tools"], [])
        self.assertFalse(options["tool_search"]["enabled"])

    def test_compact_schema_preserves_nested_constraints_and_descriptions(self):
        schema = {
            "type": "object", "properties": {
                "query": {"type": "string", "description": "Use exact names.", "enum": ["a", "b"], "minLength": 1},
                "items": {"type": "array", "items": {"type": "integer", "minimum": 2}, "maxItems": 3},
            }, "required": ["query"], "additionalProperties": False,
            "description": "Do not broaden the search.", "minProperties": 1,
        }
        original = copy.deepcopy(schema)
        rendered = compact_schema(schema)
        self.assertIn('"query":string', rendered)
        self.assertIn('"items"?:[integer {"minimum":2}]', rendered)
        for value in ['"enum":["a","b"]', '"minLength":1', '"maxItems":3', '"minProperties":1', "Use exact names.", "Do not broaden the search."]:
            self.assertIn(value, rendered)
        self.assertNotIn("...", rendered)
        self.assertEqual(schema, original)

    def test_open_and_typed_additional_properties_are_distinct(self):
        self.assertEqual(compact_schema({"type": "object"}), "{...}")
        self.assertEqual(compact_schema({"type": "object", "additionalProperties": False}), "{}")
        self.assertEqual(compact_schema({"type": "object", "additionalProperties": {"type": "number"}}), "{...:number}")

    def test_unusual_schemas_are_preserved_as_json(self):
        for schema in [
            {"type": ["string", "null"], "default": None},
            {"$ref": "#/$defs/entry", "$defs": {"entry": {"type": "string"}}},
            {"type": "object", "$ref": "#/definitions/entry"},
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            {"type": "array", "items": [{"type": "string"}]},
            {"type": "object", "required": ["not_in_properties"]},
            False,
        ]:
            with self.subTest(schema=schema):
                self.assertEqual(json.loads(compact_schema(schema)), schema)

    def test_ordinary_schema_scaffolding_is_substantially_smaller(self):
        schema = {"type": "object", "properties": {
            f"argument_{i}": {"type": "string"} for i in range(40)
        }, "required": [f"argument_{i}" for i in range(40)], "additionalProperties": False}
        self.assertLess(len(compact_schema(schema)), len(json.dumps(schema, separators=(",", ":"))) * 0.6)

    def test_excel_preserves_policies_grammar_and_original_validation(self):
        policy = 'Never run without consent.\nUse "exact" names; preserve \\ escapes.'
        source = {"input": "Hello", "tools": [
            {"type": "namespace", "name": "plugin", "tools": [
                {"type": "function", "name": "read", "description": policy, "parameters": {
                    "type": "object", "properties": {"key": {"type": "string"}},
                    "required": ["key"], "additionalProperties": False,
                }},
                {"type": "custom", "name": "patch", "format": {
                    "type": "grammar", "syntax": "lark", "definition": "start: /.+/\n",
                }},
            ]},
        ]}
        original = copy.deepcopy(source)
        catalog = excel_upstream._client_tool_protocol_instructions(source)
        self.assertIn(policy, catalog)
        self.assertIn('"name":"plugin.read"', catalog)
        self.assertIn('parameters: {"key":string}', catalog)
        self.assertIn('"definition":"start: /.+/\\n"', catalog)
        self.assertEqual(source, original)
        def response(arguments):
            return {"output": [{"type": "function_call", "name": "run_officejs", "call_id": "catalog_test", "arguments": json.dumps({
                "code": json.dumps({"name": "plugin.read", "arguments": arguments}),
            })}]}
        self.assertIsNotNone(excel_upstream.extract_native_client_tool_call(response({"key": "ok"}), source))
        self.assertIsNone(excel_upstream.extract_native_client_tool_call(response({}), source))
        self.assertIsNone(excel_upstream.extract_native_client_tool_call(response({"key": 42}), source))
        self.assertIsNone(excel_upstream.extract_native_client_tool_call(response({"key": "ok", "extra": True}), source))

    def test_excel_responses_lite_declarations_are_catalog_only(self):
        declaration = {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "namespace", "name": "functions", "tools": [
                {"type": "function", "name": "exec_command", "parameters": {
                    "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"],
                }},
            ]},
        ]}
        source = {"tools": None, "input": [declaration, copy.deepcopy(declaration), {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Inspect the repo."}],
        }]}
        original = copy.deepcopy(source)
        body = excel_upstream.prepare_responses_body(source)
        catalog = body["input"][0]["content"][0]["text"]
        self.assertEqual(catalog.count('"name":"exec_command"'), 2)  # example + one declaration
        self.assertEqual(excel_upstream.client_tool_types(source), {"exec_command": "function"})
        self.assertFalse(any(item.get("type") == "additional_tools" for item in body["input"]))
        self.assertEqual(source, original)


if __name__ == "__main__":
    unittest.main()
