import copy
import unittest
from types import SimpleNamespace

import format_translation
import initiator_policy
import proxy
import usage_tracking
from protocol_bridge import ProtocolBridgePlanner


class _RoutingConfigStub:
    def __init__(self, target_model=None):
        self.target_model = target_model

    def resolve_target_model(self, requested_model):
        return self.target_model or requested_model

    def resolve_approval_target_model(self, requested_model):
        del requested_model
        return None

    def resolve_compact_fallback_model(self, requested_model):
        del requested_model
        return None


def _count_cache_controls(value):
    if isinstance(value, dict):
        return int(isinstance(value.get("cache_control"), dict)) + sum(
            _count_cache_controls(item) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_count_cache_controls(item) for item in value)
    return 0


class CodexRequestCachePositioningTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()

    def test_responses_to_messages_keeps_tool_call_and_output_positions_cacheable(self):
        body = {
            "model": "claude-sonnet-4.6",
            "prompt_cache_key": "session-123",
            "instructions": "system prompt",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer prompt"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "read main.py"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I will inspect it."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "summarize it"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertEqual(out["system"][0]["text"], "system prompt\n\ndeveloper prompt")
        self.assertEqual(out["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out["tools"][0]["cache_control"], {"type": "ephemeral"})

        self.assertEqual([message["role"] for message in out["messages"]], ["user", "assistant", "user"])
        self.assertEqual(out["messages"][0]["content"], [{"type": "text", "text": "read main.py"}])

        assistant_blocks = out["messages"][1]["content"]
        self.assertEqual([block["type"] for block in assistant_blocks], ["text", "tool_use"])
        self.assertEqual(assistant_blocks[0]["text"], "I will inspect it.")
        self.assertEqual(assistant_blocks[1]["id"], "call_1")
        self.assertEqual(assistant_blocks[1]["name"], "Read")
        self.assertEqual(assistant_blocks[1]["input"], {"file": "main.py"})
        self.assertEqual(assistant_blocks[1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", assistant_blocks[0])

        user_blocks = out["messages"][2]["content"]
        self.assertEqual([block["type"] for block in user_blocks], ["tool_result", "text"])
        self.assertEqual(user_blocks[0]["tool_use_id"], "call_1")
        self.assertEqual(user_blocks[0]["content"], [{"type": "text", "text": "file contents"}])
        self.assertEqual(user_blocks[1]["text"], "summarize it")
        self.assertEqual(user_blocks[1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", user_blocks[0])

        self.assertEqual(_count_cache_controls(out), 4)

    def test_responses_to_messages_marks_trailing_input_not_prior_tool_result(self):
        body = {
            "model": "claude-opus-4.6",
            "promptCacheKey": "session-abc",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Bash",
                    "arguments": {"cmd": "git status"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "output_text", "text": "clean"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue with tests"}],
                },
            ],
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertEqual([message["role"] for message in out["messages"]], ["assistant", "user"])
        self.assertEqual(out["messages"][0]["content"][0]["type"], "tool_use")
        self.assertEqual(out["messages"][0]["content"][0]["cache_control"], {"type": "ephemeral"})

        trailing_user_blocks = out["messages"][1]["content"]
        self.assertEqual([block["type"] for block in trailing_user_blocks], ["tool_result", "text"])
        self.assertEqual(trailing_user_blocks[0]["tool_use_id"], "call_1")
        self.assertNotIn("cache_control", trailing_user_blocks[0])
        self.assertEqual(trailing_user_blocks[1]["text"], "continue with tests")
        self.assertEqual(trailing_user_blocks[1]["cache_control"], {"type": "ephemeral"})

    def test_responses_to_messages_never_moves_input_around_reasoning_or_custom_tool_history(self):
        body = {
            "model": "claude-sonnet-4.6",
            "prompt_cache_key": "stable-cache",
            "input": [
                {"type": "message", "role": "user", "content": "first"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                    "encrypted_content": "cipher",
                    "id": "rs_1",
                },
                {"type": "custom_tool_call", "call_id": "ct_1", "name": "shell", "input": "ls"},
                {"type": "custom_tool_call_output", "call_id": "ct_1", "output": "a.py"},
                {"type": "message", "role": "user", "content": "second"},
            ],
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertEqual([message["role"] for message in out["messages"]], ["user", "assistant", "user"])
        self.assertEqual(out["messages"][0]["content"][0]["text"], "first")
        self.assertEqual([block["type"] for block in out["messages"][1]["content"]], ["thinking", "text"])
        self.assertEqual(out["messages"][1]["content"][0]["signature"], "cipher@rs_1")
        self.assertIn("[Custom tool call (ct_1)] shell", out["messages"][1]["content"][1]["text"])
        self.assertIn("[Custom tool result (ct_1)]", out["messages"][2]["content"][0]["text"])
        self.assertEqual(out["messages"][2]["content"][1]["text"], "second")
        self.assertEqual(out["messages"][2]["content"][1]["cache_control"], {"type": "ephemeral"})

    def test_responses_prompt_cache_uses_last_four_breakpoints_only(self):
        body = {
            "model": "claude-sonnet-4.6",
            "prompt_cache_key": "cache-limit",
            "instructions": "system prompt",
            "input": [
                {"type": "message", "role": "user", "content": "turn 1"},
                {"type": "message", "role": "assistant", "content": "turn 2"},
                {"type": "message", "role": "user", "content": "turn 3"},
                {"type": "message", "role": "assistant", "content": "turn 4"},
                {"type": "message", "role": "user", "content": "turn 5"},
            ],
            "tools": [
                {"type": "function", "name": "First", "parameters": {"type": "object", "properties": {}}},
                {"type": "function", "name": "Last", "parameters": {"type": "object", "properties": {}}},
            ],
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertEqual(_count_cache_controls(out), 4)
        self.assertNotIn("cache_control", out["tools"][0])
        self.assertEqual(out["tools"][1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", out["messages"][0]["content"][0])
        self.assertNotIn("cache_control", out["messages"][1]["content"][0])
        self.assertNotIn("cache_control", out["messages"][2]["content"][0])
        self.assertEqual(out["messages"][3]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out["messages"][4]["content"][0]["cache_control"], {"type": "ephemeral"})

    def test_protocol_bridge_native_responses_sanitizes_local_cache_fields_without_reordering_input(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("gpt-5.4"))
        input_items = [
            {"type": "message", "role": "user", "content": "one"},
            {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "two"},
            {"type": "message", "role": "user", "content": "three"},
        ]
        body = {
            "model": "gpt-5.4",
            "input": copy.deepcopy(input_items),
            "prompt_cache_key": "local-cache-key",
            "promptCacheKey": "legacy-cache-key",
            "previous_response_id": "resp_prev",
            "service_tier": "priority",
            "client_metadata": {"session_id": "local-only"},
            "tool_choice": "auto",
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertEqual(plan.upstream_body["input"], input_items)
        self.assertEqual(body["input"], input_items)
        # Cache-lineage fields are now forwarded so upstream can reuse the
        # prompt-prefix cache across turns.
        for key in ("prompt_cache_key", "promptCacheKey", "previous_response_id"):
            self.assertEqual(plan.upstream_body.get(key), body[key])
        # Only fields known to be rejected by Copilot are stripped; other
        # request fields stay byte-stable for cache-sensitive follow-ups.
        self.assertNotIn("service_tier", plan.upstream_body)
        self.assertEqual(plan.upstream_body["client_metadata"], {"session_id": "local-only"})
        self.assertEqual(plan.upstream_body["tool_choice"], "auto")

        self.assertEqual(plan.diagnostics[0]["fields"], ["service_tier"])

    def test_responses_header_identity_ignores_prompt_cache_key_and_strips_aliases(self):
        first_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-fallback"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        first_body = {
            "model": "gpt-5.4",
            "input": "hello",
            "prompt_cache_key": " primary-cache ",
            "promptCacheKey": "secondary-cache",
        }

        first_headers = format_translation.build_responses_headers_for_request(
            first_request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        second_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "different-client"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        second_body = {
            "model": "gpt-5.4",
            "input": "continue",
            "prompt_cache_key": "primary-cache",
        }
        second_headers = format_translation.build_responses_headers_for_request(
            second_request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        third_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-fallback"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        third_body = {
            "model": "gpt-5.4",
            "input": "continue",
            "promptCacheKey": "secondary-cache",
        }
        third_headers = format_translation.build_responses_headers_for_request(
            third_request,
            third_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertNotEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertEqual(second_headers["x-interaction-id"], third_headers["x-interaction-id"])
        self.assertEqual(second_headers["x-agent-task-id"], third_headers["x-agent-task-id"])
        self.assertEqual(first_body.get("prompt_cache_key"), " primary-cache ")
        self.assertEqual(first_body.get("promptCacheKey"), "secondary-cache")
        self.assertEqual(second_body.get("prompt_cache_key"), "primary-cache")
        self.assertEqual(third_body.get("promptCacheKey"), "secondary-cache")
        self.assertNotIn("x-client-request-id", first_headers)
        self.assertNotIn("x-openai-subagent", first_headers)

    def test_responses_header_identity_ignores_session_and_client_request_id_for_native_responses(self):
        first_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-a"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        first_body = {"model": "gpt-5.4", "input": "hello", "sessionId": "session-123"}
        first_headers = format_translation.build_responses_headers_for_request(
            first_request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        second_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-b"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        second_body = {"model": "gpt-5.4", "input": "continue", "sessionId": "session-123"}
        second_headers = format_translation.build_responses_headers_for_request(
            second_request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        third_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-a"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        third_body = {"model": "gpt-5.4", "input": "continue"}
        third_headers = format_translation.build_responses_headers_for_request(
            third_request,
            third_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertNotEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertEqual(second_headers["x-interaction-id"], third_headers["x-interaction-id"])
        self.assertEqual(second_headers["x-agent-task-id"], third_headers["x-agent-task-id"])
        self.assertNotIn("session_id", first_headers)
        self.assertNotIn("x-client-request-id", first_headers)

    def test_responses_header_identity_ignores_camel_prompt_cache_key_alias(self):
        first_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-a"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        first_body = {"model": "gpt-5.4", "input": "hello", "promptCacheKey": "camel-cache"}
        first_headers = format_translation.build_responses_headers_for_request(
            first_request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        second_request = SimpleNamespace(
            headers={"x-openai-subagent": "worker", "x-client-request-id": "client-b"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        second_body = {"model": "gpt-5.4", "input": "continue", "promptCacheKey": " camel-cache "}
        second_headers = format_translation.build_responses_headers_for_request(
            second_request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertNotEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertEqual(first_body.get("promptCacheKey"), "camel-cache")
        self.assertEqual(second_body.get("promptCacheKey"), " camel-cache ")


class ResponsesToAnthropicMessagesUpstreamFormattingTests(unittest.TestCase):
    def test_rejects_non_dict_body_and_unsupported_roles_or_items(self):
        with self.assertRaisesRegex(ValueError, "Responses body must be a dict"):
            format_translation.responses_request_to_anthropic_messages("bad")

        with self.assertRaisesRegex(ValueError, "Unsupported Responses message role"):
            format_translation.responses_request_to_anthropic_messages(
                {"model": "claude-sonnet-4.6", "input": [{"type": "message", "role": "critic", "content": "bad"}]}
            )

        with self.assertRaisesRegex(ValueError, "Unsupported Responses input item type"):
            format_translation.responses_request_to_anthropic_messages(
                {"model": "claude-sonnet-4.6", "input": [{"type": "unknown"}]}
            )

    def test_content_block_conversion_handles_files_images_and_invalid_shapes(self):
        data_pdf = {"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,PDFDATA"}
        data_image = {"type": "input_image", "image_url": "data:image/png;base64,IMGDATA"}
        url_image = {"type": "input_image", "image_url": {"url": "https://example.invalid/a.png"}}

        self.assertEqual(format_translation._responses_input_text_to_anthropic_block({"type": "input_text", "text": "hi"}), {"type": "text", "text": "hi"})
        self.assertIsNone(format_translation._responses_input_text_to_anthropic_block({"type": "input_text", "text": 3}))
        self.assertEqual(format_translation._responses_input_text_to_anthropic_block(data_pdf)["source"]["data"], "PDFDATA")
        self.assertIsNone(format_translation._responses_input_text_to_anthropic_block({"type": "input_file", "file_data": "bad"}))
        self.assertIsNone(format_translation._responses_input_text_to_anthropic_block({"type": "input_file", "file_data": "data:bad"}))
        self.assertEqual(format_translation._responses_input_text_to_anthropic_block(data_image)["source"]["type"], "base64")
        self.assertEqual(format_translation._responses_input_text_to_anthropic_block(url_image)["source"]["type"], "url")
        self.assertIsNone(format_translation._responses_input_text_to_anthropic_block({"type": "input_image", "image_url": 3}))
        self.assertIsNone(format_translation._responses_input_text_to_anthropic_block({"type": "not_supported"}))

    def test_function_call_and_output_conversion_handles_json_and_errors(self):
        self.assertEqual(
            format_translation._responses_function_call_to_anthropic_block(
                {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "[1,2]"}
            )["input"],
            {"value": [1, 2]},
        )
        self.assertEqual(
            format_translation._responses_function_call_to_anthropic_block(
                {"type": "function_call", "id": "fc_1", "name": "Read", "arguments": "{bad"}
            )["input"],
            {"_raw": "{bad"},
        )
        self.assertEqual(
            format_translation._responses_function_call_to_anthropic_block(
                {"type": "function_call", "id": "fc_1", "name": "Read"}
            )["input"],
            {},
        )

        with self.assertRaisesRegex(ValueError, "include a name"):
            format_translation._responses_function_call_to_anthropic_block({"call_id": "call_1"})
        with self.assertRaisesRegex(ValueError, "include call_id"):
            format_translation._responses_function_call_to_anthropic_block({"name": "Read"})
        with self.assertRaisesRegex(ValueError, "function_call_output items must include call_id"):
            format_translation._responses_function_call_output_to_anthropic_block({"output": "x"})

        self.assertEqual(
            format_translation._responses_function_call_output_to_anthropic_block(
                {"call_id": "call_1", "output": {"ok": True}}
            )["content"],
            [{"type": "text", "text": '{"ok":true}'}],
        )
        self.assertEqual(
            format_translation._responses_function_call_output_to_anthropic_block(
                {"call_id": "call_1", "output": None}
            )["content"],
            [{"type": "text", "text": ""}],
        )

    def test_tool_conversion_and_choice_cover_safe_dangerous_and_parallel_cases(self):
        self.assertIsNone(format_translation._responses_tool_to_anthropic("bad"))
        self.assertIsNone(format_translation._responses_tool_to_anthropic({"type": "image_generation", "name": "draw"}))
        self.assertIsNone(format_translation._responses_tool_to_anthropic({"type": "function", "name": "mcp__ide__executeCode"}))
        with self.assertRaisesRegex(ValueError, "Responses tools must include a name"):
            format_translation._responses_tool_to_anthropic({"type": "function"})

        converted = format_translation._responses_tool_to_anthropic(
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "read files",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        )
        self.assertEqual(converted["name"], "Read")
        self.assertEqual(converted["description"], "read files")

        self.assertEqual(format_translation._responses_tool_choice_to_anthropic("auto"), {"type": "auto"})
        self.assertEqual(format_translation._responses_tool_choice_to_anthropic("required"), {"type": "any"})
        self.assertEqual(format_translation._responses_tool_choice_to_anthropic("none"), {"type": "none"})
        self.assertEqual(format_translation._responses_tool_choice_to_anthropic({"type": "required"}), {"type": "any"})
        self.assertIsNone(
            format_translation._responses_tool_choice_to_anthropic(
                {"type": "function", "name": "mcp__ide__executeCode"}
            )
        )
        with self.assertRaisesRegex(ValueError, "tool_choice type=function must include name"):
            format_translation._responses_tool_choice_to_anthropic({"type": "function"})
        with self.assertRaisesRegex(ValueError, "Unsupported Responses tool_choice value"):
            format_translation._responses_tool_choice_to_anthropic("bogus")

        out = format_translation.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": "hi",
                "tools": [{"type": "function", "name": "Read", "parameters": {"type": "object"}}],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
                "parallel_tool_calls": False,
            }
        )
        self.assertEqual(out["tool_choice"], {"type": "tool", "name": "Read", "disable_parallel_tool_use": True})

        out = format_translation.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": "hi",
                "tools": [{"type": "function", "name": "Read", "parameters": {"type": "object"}}],
                "parallel_tool_calls": False,
            }
        )
        self.assertEqual(out["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})

    def test_response_body_fields_clamp_and_reasoning_are_preserved_for_upstream(self):
        out = format_translation.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": [
                    "skip",
                    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
                    {"type": "message", "role": "system", "content": "sys"},
                    {"type": "message", "role": "user", "content": ""},
                ],
                "instructions": "instructions",
                "max_output_tokens": -1,
                "temperature": 2,
                "top_p": 0.5,
                "metadata": {"k": "v"},
                "stop": ["END"],
                "stream": 1,
                "reasoning": {"effort": "max"},
            }
        )

        self.assertEqual(out["system"], "instructions\n\ndev\n\nsys")
        self.assertEqual(out["max_tokens"], 4096)
        self.assertEqual(out["temperature"], 1)
        self.assertEqual(out["top_p"], 0.5)
        self.assertEqual(out["metadata"], {"k": "v"})
        self.assertEqual(out["stop_sequences"], ["END"])
        self.assertTrue(out["stream"])
        self.assertEqual(out["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(out["output_config"], {"effort": "max"})

        forced_tool = format_translation.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": "hi",
                "tools": [{"type": "function", "name": "Read", "parameters": {"type": "object"}}],
                "tool_choice": "required",
                "reasoning": {"effort": "max"},
                "temperature": -1,
                "stop_sequences": ["STOP"],
            }
        )
        self.assertNotIn("thinking", forced_tool)
        self.assertEqual(forced_tool["temperature"], 0)
        self.assertEqual(forced_tool["stop_sequences"], ["STOP"])

    def test_prompt_cache_helpers_ignore_malformed_payloads_and_existing_markers(self):
        self.assertFalse(format_translation._responses_body_requests_prompt_cache({"prompt_cache_key": " "}))
        self.assertTrue(format_translation._responses_body_requests_prompt_cache({"promptCacheKey": " cache "}))

        block = {"type": "text", "text": "already", "cache_control": {"type": "persist"}}
        format_translation._add_anthropic_cache_control(block)
        self.assertEqual(block["cache_control"], {"type": "persist"})

        self.assertIsNone(format_translation._last_cacheable_anthropic_content_block("bad"))
        self.assertEqual(
            format_translation._last_cacheable_anthropic_content_block(["bad", {"type": "TEXT", "text": "x"}])["text"],
            "x",
        )
        payload = {
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "persist"}}],
            "messages": ["bad", {"role": "user", "content": ["bad", {"type": "text", "text": "hi"}]}],
        }
        format_translation._apply_responses_prompt_cache_to_anthropic_messages(payload)
        self.assertEqual(payload["system"][0]["cache_control"], {"type": "persist"})
        self.assertEqual(payload["messages"][1]["content"][1]["cache_control"], {"type": "ephemeral"})


if __name__ == "__main__":
    unittest.main()
