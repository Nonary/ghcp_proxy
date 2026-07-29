import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest

import excel_upstream


def _jwt_with_exp(expiration: float) -> str:
    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': expiration})}."


class ExcelUpstreamTests(unittest.TestCase):
    def test_session_store_keeps_only_allowlisted_headers(self):
        store = excel_upstream.ExcelSessionStore()
        status = store.configure(
            {
                "Authorization": f"Bearer {_jwt_with_exp(time.time() + 600)}",
                "ChatGPT-Account-ID": "account-1",
                "Cookie": "must-not-be-forwarded",
            }
        )

        self.assertTrue(status["configured"])
        headers = store.request_headers(stream=True)
        self.assertNotIn("cookie", headers)
        self.assertEqual(headers["x-openai-account-id"], "account-1")
        self.assertEqual(headers["accept"], "text/event-stream")

    def test_expired_session_is_rejected(self):
        store = excel_upstream.ExcelSessionStore()
        with self.assertRaisesRegex(ValueError, "already expired"):
            store.configure(
                {
                    "authorization": f"Bearer {_jwt_with_exp(time.time() - 1)}",
                    "chatgpt-account-id": "account-1",
                }
            )

    def test_responses_body_uses_excel_wire_shape(self):
        source = {
            "model": "gpt-excel",
            "instructions": "Use the client tools.",
            "input": "Hello",
            "prompt_cache_key": "conversation-1",
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "text": {"verbosity": "low"},
            "max_output_tokens": 1234,
            "tools": [{"type": "function", "name": "demo"}],
        }
        body = excel_upstream.prepare_responses_body(source)

        self.assertEqual(body["model"], "gpt-5.5")
        self.assertFalse(body["store"])
        self.assertEqual(body["reasoning_effort"], "xhigh")
        self.assertEqual(body["prompt_cache_key"], "conversation-1")
        self.assertEqual(body["input"][0]["role"], "developer")
        self.assertEqual(
            body["input"][0]["content"][0]["text"], "Use the client tools."
        )
        self.assertEqual(body["input"][1]["role"], "developer")
        tool_prompt = body["input"][1]["content"][0]["text"]
        self.assertIn("external Codex Responses API client", tool_prompt)
        self.assertIn("<codex_tool_call>", tool_prompt)
        self.assertIn('"name":"demo"', tool_prompt)
        self.assertEqual(body["input"][2]["role"], "user")
        self.assertNotIn("tools", body)
        self.assertNotIn("include", body)
        self.assertNotIn("text", body)
        self.assertNotIn("max_output_tokens", body)
        self.assertNotIn("reasoning", body)

    def test_task_identity_is_stable_for_a_conversation(self):
        source = {
            "model": "gpt-excel",
            "input": "Hello",
            "prompt_cache_key": "conversation-1",
        }
        first = excel_upstream.prepare_responses_body(source)
        second = excel_upstream.prepare_responses_body(source)
        self.assertEqual(
            first["metadata"]["task_id"], second["metadata"]["task_id"]
        )
        other = excel_upstream.prepare_responses_body(
            {**source, "prompt_cache_key": "conversation-2"}
        )
        self.assertNotEqual(
            first["metadata"]["task_id"], other["metadata"]["task_id"]
        )

    def test_encrypted_reasoning_is_replayed_and_bare_reasoning_dropped(self):
        items = excel_upstream.translate_input_items(
            [
                {"type": "reasoning", "summary": [], "encrypted_content": "gAAA=="},
                {"type": "reasoning", "id": "rs_bare", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["type"], "reasoning")
        self.assertEqual(items[0]["encrypted_content"], "gAAA==")
        self.assertEqual(items[1]["type"], "message")

    def test_tool_history_is_replayed_natively(self):
        raw_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list files"}],
            },
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "shell_command",
                "arguments": '{"command":"ls"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "file.txt",
            },
        ]
        items = excel_upstream.translate_input_items(
            raw_input, {"shell_command": "function"}
        )

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["role"], "user")
        # Native items pass through untouched (bare reasoning is dropped).
        self.assertIs(items[1], raw_input[2])
        self.assertIs(items[2], raw_input[3])
        # Deterministic rendering keeps the upstream prompt-cache prefix stable.
        self.assertEqual(
            items,
            excel_upstream.translate_input_items(
                raw_input, {"shell_command": "function"}
            ),
        )

    def test_plan_outputs_carry_steering_directive(self):
        items = excel_upstream.translate_input_items(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "update_plan",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Plan updated",
                },
            ],
            {"update_plan": "function", "shell_command": "function"},
        )
        output = items[1]["output"]
        self.assertTrue(output.startswith("Plan updated\n"))
        self.assertIn(excel_upstream.PLAN_TOOL_OUTPUT_STEERING, output)
        self.assertIn("shell_command", output)

    def test_catalog_leads_and_only_a_compact_reminder_trails(self):
        body = excel_upstream.prepare_responses_body(
            {
                "model": "gpt-excel",
                "input": "Hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "demo",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )
        catalog = body["input"][0]["content"][0]["text"]
        self.assertIn('"name":"demo"', catalog)
        last = body["input"][-1]
        self.assertEqual(last["role"], "developer")
        reminder = last["content"][0]["text"]
        self.assertIn("<codex_tool_call>", reminder)
        self.assertIn("demo", reminder)
        # The trailing message re-bills on every turn, so it must stay small
        # relative to the catalog it replaces.
        self.assertLess(len(reminder), len(catalog) / 2)

    def test_catalog_is_the_only_message_without_tools(self):
        without_tools = excel_upstream.prepare_responses_body(
            {"model": "gpt-excel", "input": "Hello"}
        )
        self.assertEqual(
            without_tools["input"][0]["content"][0]["text"],
            excel_upstream.EXTERNAL_CLIENT_INSTRUCTIONS,
        )
        self.assertEqual(without_tools["input"][-1]["role"], "user")

    def test_growing_conversation_keeps_a_stable_cache_prefix(self):
        source = {
            "model": "gpt-excel",
            "instructions": "Be terse.",
            "prompt_cache_key": "conversation-1",
            "tools": [{"type": "function", "name": "shell_command"}],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "list files"}],
                }
            ],
        }
        first = excel_upstream.prepare_responses_body(source)
        second = excel_upstream.prepare_responses_body(
            {
                **source,
                "input": source["input"]
                + [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell_command",
                        "arguments": '{"command":"ls"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "file.txt",
                    },
                ],
            }
        )

        # Everything the first turn sent, minus its trailing reminder, must
        # still be a byte-identical prefix of the second turn: that prefix is
        # exactly what the upstream prompt cache can reuse.
        first_prefix = first["input"][:-1]
        self.assertEqual(second["input"][: len(first_prefix)], first_prefix)
        self.assertEqual(first["input"][-1], second["input"][-1])

    def test_catalog_position_escape_hatch_restores_suffix_layout(self):
        source = {
            "model": "gpt-excel",
            "input": "Hello",
            "tools": [{"type": "function", "name": "demo"}],
        }
        original = excel_upstream.CATALOG_AT_PROMPT_END
        excel_upstream.CATALOG_AT_PROMPT_END = True
        try:
            body = excel_upstream.prepare_responses_body(source)
        finally:
            excel_upstream.CATALOG_AT_PROMPT_END = original
        self.assertEqual(len(body["input"]), 2)
        self.assertEqual(body["input"][0]["role"], "user")
        self.assertIn('"name":"demo"', body["input"][-1]["content"][0]["text"])

    def test_empty_tool_output_is_rendered_as_explicit_success(self):
        items = excel_upstream.translate_input_items(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell_command",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "",
                },
            ],
            {"shell_command": "function"},
        )
        self.assertEqual(
            items[1]["output"], "(tool call succeeded with no output)"
        )

    def test_plan_update_churn_detection(self):
        def plan_call(call_id, arguments):
            return {
                "type": "function_call",
                "call_id": call_id,
                "name": "update_plan",
                "arguments": arguments,
            }

        def plan_output(call_id):
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "Plan updated",
            }

        create_args = '{"plan": [{"status": "in_progress", "step": "a"}]}'
        source = {"input": [plan_call("call_1", create_args), plan_output("call_1")]}
        progressed = {
            "type": "function_call",
            "name": "update_plan",
            "arguments": '{"plan":[{"step":"a","status":"completed"}]}',
        }
        repeat = {
            "type": "function_call",
            "name": "update_plan",
            "arguments": '{"plan":[{"step":"a","status":"in_progress"}]}',
        }
        reworded_repeat = {
            "type": "function_call",
            "name": "update_plan",
            "arguments": (
                '{"plan":[{"step":"a","status":"in_progress"}],'
                '"explanation":"Slightly different wording"}'
            ),
        }
        # A single follow-up plan update that progresses is legitimate
        # (create the plan, then mark the first step in progress).
        self.assertFalse(excel_upstream.is_plan_update_churn(progressed, source))
        # An identical repeat is churn even on the first follow-up.
        self.assertTrue(excel_upstream.is_plan_update_churn(repeat, source))
        # Rewording only the explanation does not make a repeat legitimate.
        self.assertTrue(
            excel_upstream.is_plan_update_churn(reworded_repeat, source)
        )
        self.assertFalse(
            excel_upstream.is_plan_update_churn(progressed, {"input": []})
        )

        # A third consecutive plan update is churn regardless of arguments.
        two_in_a_row = {
            "input": [
                plan_call("call_1", create_args),
                plan_output("call_1"),
                plan_call(
                    "call_2", '{"plan":[{"step":"a","status":"in_progress"}]}'
                ),
                plan_output("call_2"),
            ]
        }
        self.assertTrue(
            excel_upstream.is_plan_update_churn(progressed, two_in_a_row)
        )

        # A user message resets the churn window.
        after_user_nudge = {
            "input": [
                *two_in_a_row["input"],
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ]
        }
        self.assertFalse(
            excel_upstream.is_plan_update_churn(progressed, after_user_nudge)
        )

        # Substantive work between plan updates resets the chain.
        worked_since = {
            "input": [
                plan_call("call_1", create_args),
                plan_output("call_1"),
                {
                    "type": "function_call",
                    "call_id": "call_3",
                    "name": "shell_command",
                    "arguments": '{"command":"ls"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_3",
                    "output": "file.txt",
                },
            ]
        }
        self.assertFalse(
            excel_upstream.is_plan_update_churn(progressed, worked_since)
        )
        shell_call = {
            "type": "function_call",
            "name": "shell_command",
            "arguments": "{}",
        }
        self.assertFalse(excel_upstream.is_plan_update_churn(shell_call, source))

    def test_native_plan_status_aliases_are_normalized(self):
        source = {
            "tools": [
                {
                    "type": "function",
                    "name": "update_plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "in_progress",
                                                "completed",
                                            ],
                                        },
                                    },
                                    "required": ["step", "status"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "update_plan",
                    "arguments": json.dumps(
                        {
                            "plan": [
                                {"step": "a", "status": "done"},
                                {"step": "b", "status": "Not Started"},
                            ]
                        }
                    ),
                }
            ]
        }
        tool_call = excel_upstream.extract_native_client_tool_call(response, source)
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {
                "plan": [
                    {"step": "a", "status": "completed"},
                    {"step": "b", "status": "pending"},
                ]
            },
        )

    def test_unsupported_reasoning_effort_falls_back_to_medium(self):
        body = excel_upstream.prepare_responses_body(
            {
                "model": "gpt-excel",
                "input": "Hello",
                "reasoning": {"effort": "ultra"},
            }
        )
        self.assertEqual(body["reasoning_effort"], "medium")

    def test_function_tool_marker_is_converted_only_for_allowed_tool(self):
        marker = (
            '<codex_tool_call>{"name":"shell_command","arguments":'
            '{"command":"rg -n gpt-excel excel_upstream.py"}}</codex_tool_call>'
        )
        tool_call = excel_upstream.extract_client_tool_call(
            marker,
            {"shell_command": "function"},
        )
        self.assertEqual(tool_call["type"], "function_call")
        self.assertEqual(tool_call["name"], "shell_command")
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {"command": "rg -n gpt-excel excel_upstream.py"},
        )
        self.assertIsNone(
            excel_upstream.extract_client_tool_call(
                marker,
                {"update_plan": "function"},
            )
        )

    def test_custom_tool_marker_is_converted_to_custom_call(self):
        marker = (
            '<codex_tool_call>{"name":"apply_patch",'
            '"input":"*** Begin Patch\\n*** End Patch\\n"}</codex_tool_call>'
        )
        tool_call = excel_upstream.extract_client_tool_call(
            marker,
            {"apply_patch": "custom"},
        )
        self.assertEqual(tool_call["type"], "custom_tool_call")
        self.assertEqual(tool_call["name"], "apply_patch")
        self.assertEqual(
            tool_call["input"],
            "*** Begin Patch\n*** End Patch\n",
        )

    def test_response_payload_replaces_marker_text_with_tool_call(self):
        tool_call = excel_upstream.extract_client_tool_call(
            '<codex_tool_call>{"name":"demo","arguments":{"value":1}}</codex_tool_call>',
            {"demo": "function"},
        )
        payload = excel_upstream.response_payload_with_tool_call(
            {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [{"type": "message"}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
            tool_call,
        )
        self.assertEqual(payload["model"], "gpt-excel")
        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["usage"]["output_tokens"], 2)

    def test_native_excel_update_plan_is_normalized_to_client_schema(self):
        source = {
            "tools": [
                {
                    "type": "function",
                    "name": "update_plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "in_progress",
                                                "completed",
                                            ],
                                        },
                                    },
                                    "required": ["step", "status"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        response = {
            "output": [
                {"type": "reasoning"},
                {
                    "type": "function_call",
                    "name": "update_plan",
                    "arguments": json.dumps(
                        {
                            "summary": "Inspect repository",
                            "plan": [
                                {
                                    "id": "step1",
                                    "description": "Search the code",
                                    "status": "in_progress",
                                    "result": "",
                                }
                            ],
                        }
                    ),
                },
            ]
        }
        tool_call = excel_upstream.extract_native_client_tool_call(
            response,
            source,
        )
        self.assertEqual(tool_call["name"], "update_plan")
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {
                "plan": [
                    {
                        "step": "Search the code",
                        "status": "in_progress",
                    }
                ],
                "explanation": "Inspect repository",
            },
        )

    def test_unknown_native_excel_tool_is_not_forwarded(self):
        self.assertIsNone(
            excel_upstream.extract_native_client_tool_call(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "list_skills",
                            "arguments": "{}",
                        }
                    ]
                },
                {
                    "tools": [
                        {
                            "type": "function",
                            "name": "shell_command",
                            "parameters": {"type": "object"},
                        }
                    ]
                },
            )
        )

    def test_local_model_is_merged_once(self):
        payload = excel_upstream.merge_local_models_payload(
            {"object": "list", "data": [{"id": "gpt-5.5", "object": "model"}]}
        )
        payload = excel_upstream.merge_local_models_payload(payload)
        self.assertEqual(
            [item["id"] for item in payload["data"]],
            ["gpt-5.5", "gpt-excel"],
        )

class ExcelStreamTransformTests(unittest.TestCase):
    SOURCE_BODY = {
        "tools": [
            {
                "type": "function",
                "name": "shell_command",
                "parameters": {"type": "object"},
            }
        ]
    }

    @staticmethod
    def _sse(event: str, payload: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()

    def _collect(
        self,
        chunks: list[bytes],
        source_body: dict | None = None,
    ) -> list[tuple[str, dict]]:
        import proxy as proxy_module

        transform = proxy_module._excel_tool_stream_transform(
            source_body if source_body is not None else self.SOURCE_BODY
        )

        async def source():
            for chunk in chunks:
                yield chunk

        async def run():
            return [chunk async for chunk in transform(source())]

        raw = b"".join(asyncio.run(run())).decode()
        events = []
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            name = None
            data_lines = []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            data = "\n".join(data_lines)
            events.append((name, json.loads(data) if data != "[DONE]" else {}))
        return events

    def _delta_chunks(self, text: str, size: int = 9) -> list[bytes]:
        return [
            self._sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text[start : start + size],
                },
            )
            for start in range(0, len(text), size)
        ]

    def _stream(self, text: str) -> list[bytes]:
        return (
            [
                self._sse(
                    "response.created",
                    {"type": "response.created", "response": {"id": "resp_1"}},
                )
            ]
            + self._delta_chunks(text)
            + [
                self._sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": "msg_1",
                        "text": text,
                    },
                ),
                self._sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        },
                    },
                ),
                self._sse(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_1",
                            "object": "response",
                            "status": "completed",
                            "model": "gpt-5.5",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {"type": "output_text", "text": text}
                                    ],
                                }
                            ],
                            "usage": {"input_tokens": 5, "output_tokens": 7},
                        },
                    },
                ),
            ]
        )

    def test_marker_stream_is_converted_to_tool_call_events(self):
        marker = (
            '<codex_tool_call>{"name":"shell_command","arguments":'
            '{"command":"ls"}}</codex_tool_call>'
        )
        events = self._collect(self._stream(marker))
        names = [name for name, _ in events]

        self.assertNotIn("response.output_text.delta", names)
        self.assertNotIn("response.output_text.done", names)
        self.assertIn("response.function_call_arguments.done", names)
        completed = dict(events)[
            "response.completed"
        ]["response"]
        self.assertEqual(completed["model"], "gpt-excel")
        self.assertEqual(completed["output"][0]["type"], "function_call")
        self.assertEqual(completed["output"][0]["name"], "shell_command")
        self.assertEqual(completed["usage"]["output_tokens"], 7)

    def test_plain_text_streams_through_incrementally(self):
        text = "The answer is 42, see <codex spreadsheet notes for details."
        events = self._collect(self._stream(text))
        deltas = [
            payload["delta"]
            for name, payload in events
            if name == "response.output_text.delta"
        ]
        self.assertEqual("".join(deltas), text)
        completed = dict(events)["response.completed"]["response"]
        self.assertEqual(completed["output"][0]["type"], "message")
        done_payload = dict(events)["response.output_text.done"]
        self.assertEqual(done_payload["text"], text)

    def test_duplicate_plan_update_ends_turn_with_text(self):
        plan_arguments = json.dumps(
            {"plan": [{"step": "Inspect code", "status": "in_progress"}]}
        )
        source_body = {
            "tools": [
                {
                    "type": "function",
                    "name": "update_plan",
                    "parameters": {"type": "object"},
                }
            ],
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "update_plan",
                    "arguments": plan_arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Plan updated",
                },
            ],
        }
        marker = (
            '<codex_tool_call>{"name":"update_plan","arguments":'
            + plan_arguments
            + "}</codex_tool_call>"
        )
        events = self._collect(self._stream(marker), source_body)
        names = [name for name, _ in events]

        self.assertNotIn("response.function_call_arguments.done", names)
        completed = dict(events)["response.completed"]["response"]
        self.assertEqual(completed["output"][0]["type"], "message")
        self.assertEqual(
            completed["output"][0]["content"][0]["text"],
            excel_upstream.DUPLICATE_PLAN_UPDATE_TEXT,
        )

    def test_native_tool_call_is_normalized_and_not_leaked_raw(self):
        source_body = {
            "tools": [
                {
                    "type": "function",
                    "name": "update_plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "in_progress",
                                                "completed",
                                            ],
                                        },
                                    },
                                    "required": ["step", "status"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                    },
                }
            ],
            "input": [],
        }
        raw_arguments = json.dumps(
            {
                "plan": [
                    {
                        "id": "step1",
                        "description": "Inspect the repository",
                        "status": "in_progress",
                        "result": "",
                    }
                ]
            }
        )
        native_item = {
            "type": "function_call",
            "id": "fc_upstream",
            "call_id": "call_upstream",
            "name": "update_plan",
            "arguments": raw_arguments,
        }
        chunks = [
            self._sse(
                "response.created",
                {"type": "response.created", "response": {"id": "resp_1"}},
            ),
            self._sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**native_item, "arguments": ""},
                },
            ),
            self._sse(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_upstream",
                    "delta": raw_arguments,
                },
            ),
            self._sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_upstream",
                    "arguments": raw_arguments,
                },
            ),
            self._sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": native_item,
                },
            ),
            self._sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "model": "gpt-5.5",
                        "output": [native_item],
                        "usage": {"input_tokens": 5, "output_tokens": 7},
                    },
                },
            ),
        ]
        events = self._collect(chunks, source_body)

        item_done_payloads = [
            payload
            for name, payload in events
            if name == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        self.assertEqual(len(item_done_payloads), 1)
        arguments = json.loads(item_done_payloads[0]["item"]["arguments"])
        self.assertEqual(
            arguments,
            {"plan": [{"step": "Inspect the repository", "status": "in_progress"}]},
        )
        raw = json.dumps(events)
        self.assertNotIn("fc_upstream", raw)
        completed = dict(events)["response.completed"]["response"]
        self.assertEqual(completed["model"], "gpt-excel")
        self.assertEqual(completed["output"][0]["type"], "function_call")

    def test_marker_without_valid_tool_is_released_as_text(self):
        marker = (
            '<codex_tool_call>{"name":"unknown_tool","arguments":{}}'
            "</codex_tool_call>"
        )
        events = self._collect(self._stream(marker))
        deltas = [
            payload["delta"]
            for name, payload in events
            if name == "response.output_text.delta"
        ]
        self.assertEqual("".join(deltas), marker)
        completed = dict(events)["response.completed"]["response"]
        self.assertEqual(completed["output"][0]["type"], "message")


class ExcelSessionPersistenceTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows DPAPI test")
    def test_session_is_encrypted_and_reloaded_with_dpapi(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "excel-session.dpapi")
            token = _jwt_with_exp(time.time() + 600)
            store = excel_upstream.ExcelSessionStore(path)
            status = store.configure(
                {
                    "authorization": f"Bearer {token}",
                    "chatgpt-account-id": "account-1",
                }
            )

            self.assertTrue(status["persisted"])
            with open(path, "rb") as handle:
                protected = handle.read()
            self.assertNotIn(token.encode(), protected)
            self.assertNotIn(b"account-1", protected)

            restored = excel_upstream.ExcelSessionStore(path)
            restored_status = restored.load()
            self.assertTrue(restored_status["configured"])
            self.assertTrue(restored_status["persisted"])
            self.assertEqual(
                restored.request_headers(stream=False)["chatgpt-account-id"],
                "account-1",
            )


if __name__ == "__main__":
    unittest.main()
