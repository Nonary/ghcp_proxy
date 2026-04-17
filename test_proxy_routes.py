import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

import auth
import format_translation
import initiator_policy
import proxy
import usage_tracking


class ProxyRoutesTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()

    def test_premium_plan_status_api_returns_service_payload(self):
        expected = {"configured": True, "plan": "pro_plus"}

        with mock.patch.object(proxy.premium_plan_config_service, "config_payload", return_value=expected):
            response = proxy.asyncio.run(proxy.premium_plan_status_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), expected)

    def test_premium_plan_config_api_saves_settings(self):
        request = SimpleNamespace()
        expected = {"configured": True, "plan": "business", "synced_percent": 44.5}

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value={"plan": "business", "current_percent": 44.5})),
            mock.patch.object(proxy.premium_plan_config_service, "save_settings", return_value=expected) as save_settings,
        ):
            response = proxy.asyncio.run(proxy.premium_plan_config_api(request))

        save_settings.assert_called_once_with({"plan": "business", "current_percent": 44.5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), expected)

    def test_premium_plan_config_api_clears_settings(self):
        request = SimpleNamespace()
        expected = {"configured": False, "plan": ""}

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value={"clear": True})),
            mock.patch.object(proxy.premium_plan_config_service, "clear_settings", return_value=expected) as clear_settings,
        ):
            response = proxy.asyncio.run(proxy.premium_plan_config_api(request))

        clear_settings.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), expected)

    def test_anthropic_messages_route_uses_anthropic_headers_and_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-client-request-id": "req-123"},
        )
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }
        outbound = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        upstream = httpx.Response(
            400,
            json={
                "error": {
                    "message": "unsupported field",
                    "type": "invalid_request_error",
                }
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="claude-sonnet-4.6"),
            mock.patch.object(format_translation, "anthropic_request_to_chat", mock.AsyncMock(return_value=outbound)),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(format_translation, "build_anthropic_headers_for_request", return_value={"X-Initiator": "user"}) as build_headers,
            mock.patch.object(format_translation, "build_chat_headers_for_request", side_effect=AssertionError("unexpected chat headers")),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        build_headers.assert_called_once_with(
            request, body, "test-key",
            request_id=mock.ANY,
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            verdict_sink=mock.ANY,
        )
        self.assertEqual(post.await_args.args[1], "https://example.invalid/chat/completions")
        self.assertEqual(post.await_args.kwargs["headers"], {"X-Initiator": "user"})
        self.assertEqual(post.await_args.kwargs["json"], outbound)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"invalid_request_error","message":"unsupported field"}}',
        )

    def test_responses_route_invalid_json_returns_openai_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        with mock.patch.object(
            proxy,
            "parse_json_request",
            mock.AsyncMock(side_effect=proxy.HTTPException(status_code=400, detail="Invalid JSON body")),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Invalid JSON body","type":"invalid_request_error","param":null,"code":null}}',
        )

    def test_responses_route_upstream_connect_error_returns_openai_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4-mini",
            "input": "hello",
            "stream": False,
        }
        connect_error = httpx.ConnectError(
            "All connection attempts failed",
            request=httpx.Request("POST", "https://example.invalid/responses"),
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "build_responses_headers_for_request", return_value={"X-Initiator": "agent"}),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(side_effect=connect_error)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        finish_usage.assert_called_once_with(
            None,
            502,
            upstream=None,
            response_payload=None,
            response_text="Upstream connection failed",
            usage=None,
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Upstream connection failed","type":"server_error","param":null,"code":null}}',
        )

    def test_anthropic_messages_invalid_json_returns_anthropic_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )

        with mock.patch.object(
            proxy,
            "parse_json_request",
            mock.AsyncMock(side_effect=proxy.HTTPException(status_code=400, detail="Invalid JSON body")),
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"invalid_request_error","message":"Invalid JSON body"}}',
        )

    def test_anthropic_messages_upstream_connect_error_returns_anthropic_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }
        outbound = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        connect_error = httpx.ConnectError(
            "All connection attempts failed",
            request=httpx.Request("POST", "https://example.invalid/chat/completions"),
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "anthropic_request_to_chat", mock.AsyncMock(return_value=outbound)),
            mock.patch.object(format_translation, "build_anthropic_headers_for_request", return_value={"X-Initiator": "agent"}),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(side_effect=connect_error)),
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        finish_usage.assert_called_once_with(
            None,
            502,
            upstream=None,
            response_payload=None,
            response_text="Upstream connection failed",
            usage=None,
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"api_error","message":"Upstream connection failed"}}',
        )

    def test_responses_route_mapped_to_claude_uses_chat_upstream_and_returns_responses_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.3-codex",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            "stream": False,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "chatcmpl_123",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello from claude",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="claude-opus-4.6"),
            mock.patch.object(format_translation, "build_chat_headers_for_request", return_value={"X-Initiator": "user"}) as build_headers,
            mock.patch.object(format_translation, "build_responses_headers_for_request", side_effect=AssertionError("unexpected responses headers")),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None) as start_event,
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        build_headers.assert_called_once()
        start_event.assert_called_once()
        self.assertEqual(start_event.call_args.args[1], "gpt-5.3-codex")
        self.assertEqual(start_event.call_args.args[2], "claude-opus-4.6")
        self.assertEqual(post.await_args.args[1], "https://example.invalid/chat/completions")
        self.assertEqual(post.await_args.kwargs["headers"], {"X-Initiator": "user"})
        self.assertEqual(
            response.body,
            b'{"id":"chatcmpl_123","object":"response","created_at":null,"status":"completed","model":"claude-opus-4.6","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello from claude"}]}],"output_text":"hello from claude","usage":{"input_tokens":20,"output_tokens":4,"total_tokens":24}}',
        )

    def test_responses_route_mapped_to_claude_accepts_custom_tool_history(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.2",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "Exit code: 0\nSuccess.",
                },
            ],
            "stream": False,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "chatcmpl_123",
                "model": "claude-opus-4.6",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "bridged ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="claude-opus-4.6"),
            mock.patch.object(format_translation, "build_chat_headers_for_request", return_value={"X-Initiator": "user"}),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        outbound = post.await_args.kwargs["json"]
        self.assertEqual(outbound["messages"][0]["role"], "user")
        self.assertEqual(outbound["messages"][1]["role"], "assistant")
        self.assertIn("[Custom tool call (call_1)] apply_patch", outbound["messages"][1]["content"])
        self.assertEqual(outbound["messages"][2]["role"], "user")
        self.assertIn("[Custom tool result (call_1)]", outbound["messages"][2]["content"])
        self.assertEqual(response.status_code, 200)

    def test_anthropic_messages_route_mapped_to_codex_uses_responses_upstream_and_returns_anthropic_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        body = {
            "model": "claude-opus-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "stream": False,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello from codex"}],
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.4"),
            mock.patch.object(format_translation, "build_responses_headers_for_request", return_value={"X-Initiator": "user"}) as build_headers,
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None) as start_event,
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        build_headers.assert_called_once()
        start_event.assert_called_once()
        self.assertEqual(start_event.call_args.args[1], "claude-opus-4.6")
        self.assertEqual(start_event.call_args.args[2], "gpt-5.4")
        self.assertEqual(post.await_args.args[1], "https://example.invalid/responses")
        self.assertEqual(post.await_args.kwargs["headers"], {"X-Initiator": "user"})
        self.assertEqual(
            response.body,
            b'{"id":"resp_123","type":"message","role":"assistant","model":"gpt-5.4","content":[{"type":"text","text":"hello from codex"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}',
        )

    def test_responses_route_treats_local_compaction_as_handoff_boundary(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old context"}],
                },
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction("carry this forward"),
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue from here"}],
                },
            ],
            "stream": False,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "build_responses_headers_for_request", return_value={"X-Initiator": "agent"}),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_input = post.await_args.kwargs["json"]["input"]
        self.assertEqual(len(forwarded_input), 2)
        self.assertEqual(forwarded_input[0]["type"], "message")
        self.assertEqual(forwarded_input[0]["role"], "user")
        self.assertIn("carry this forward", forwarded_input[0]["content"][0]["text"])
        self.assertEqual(
            forwarded_input[1],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "continue from here"}],
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_responses_route_preserves_cache_affinity_fields(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "sessionId": "session-123",
            "promptCacheKey": "cache-123",
            "previous_response_id": "resp_prev",
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {
                    "input_tokens": 24,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 5,
                    "total_tokens": 29,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.4"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_headers = post.await_args.kwargs["headers"]
        forwarded_body = post.await_args.kwargs["json"]
        self.assertEqual(forwarded_headers["session_id"], "session-123")
        self.assertEqual(forwarded_headers["x-client-request-id"], "session-123")
        self.assertEqual(forwarded_body["sessionId"], "session-123")
        self.assertEqual(forwarded_body["prompt_cache_key"], "cache-123")
        self.assertNotIn("promptCacheKey", forwarded_body)
        self.assertEqual(forwarded_body["previous_response_id"], "resp_prev")
        self.assertEqual(forwarded_body["tools"], body["tools"])
        self.assertEqual(forwarded_body["include"], body["include"])
        self.assertTrue(forwarded_body["parallel_tool_calls"])
        self.assertEqual(forwarded_body["tool_choice"], "auto")
        self.assertFalse(forwarded_body["stream"])
        self.assertEqual(response.status_code, 200)
        response_payload = json.loads(response.body)
        self.assertEqual(response_payload["id"], "resp_123")
        self.assertEqual(response_payload["output"][0]["content"][0]["text"], "done")

    def test_responses_compact_preserves_cache_affinity_fields(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses/compact"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "sessionId": "session-123",
            "promptCacheKey": "cache-123",
            "previous_response_id": "resp_prev",
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "summary"}],
                    }
                ],
                "usage": {
                    "input_tokens": 24,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 5,
                    "total_tokens": 29,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.4"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses_compact(request))

        forwarded_headers = post.await_args.kwargs["headers"]
        forwarded_body = post.await_args.kwargs["json"]
        self.assertEqual(forwarded_headers["session_id"], "session-123")
        self.assertEqual(forwarded_headers["x-client-request-id"], "session-123")
        self.assertEqual(forwarded_body["sessionId"], "session-123")
        self.assertEqual(forwarded_body["prompt_cache_key"], "cache-123")
        self.assertNotIn("promptCacheKey", forwarded_body)
        self.assertEqual(forwarded_body["previous_response_id"], "resp_prev")
        self.assertEqual(forwarded_body["tools"], body["tools"])
        self.assertEqual(forwarded_body["include"], body["include"])
        self.assertTrue(forwarded_body["parallel_tool_calls"])
        self.assertEqual(forwarded_body["tool_choice"], "auto")
        self.assertFalse(forwarded_body["stream"])
        self.assertEqual(response.status_code, 200)
        response_payload = json.loads(response.body)
        self.assertEqual(response_payload["id"], "resp_123")
        self.assertEqual(response_payload["output"][0]["content"][0]["text"], "summary")

    def test_proxy_streaming_response_connect_error_returns_openai_error(self):
        request = httpx.Request("POST", "https://example.invalid/responses")

        class FakeClient:
            def __init__(self):
                self.aclose = mock.AsyncMock()

            def build_request(self, *args, **kwargs):
                return request

        fake_client = FakeClient()
        usage_event = {"request_id": "req-123"}
        connect_error = httpx.ConnectError("All connection attempts failed", request=request)

        with (
            mock.patch.object(proxy.httpx, "AsyncClient", return_value=fake_client),
            mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(side_effect=connect_error)),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
        ):
            response = proxy.asyncio.run(
                proxy.proxy_streaming_response(
                    "https://example.invalid/responses",
                    {"Authorization": "Bearer test"},
                    {"model": "gpt-5.4-mini", "stream": True},
                    usage_event=usage_event,
                )
            )

        finish_usage.assert_called_once_with(
            None,
            502,
            upstream=None,
            response_payload=None,
            response_text="Upstream connection failed",
            usage=None,
        )
        fake_client.aclose.assert_awaited_once()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Upstream connection failed","type":"server_error","param":null,"code":null}}',
        )

    def test_graceful_streaming_response_swallows_cancelled_error(self):
        response = proxy.GracefulStreamingResponse(iter(()))
        receive = mock.AsyncMock()
        send = mock.AsyncMock()

        with mock.patch.object(
            proxy.StreamingResponse,
            "__call__",
            mock.AsyncMock(side_effect=asyncio.CancelledError()),
        ) as parent_call:
            proxy.asyncio.run(response({}, receive, send))

        parent_call.assert_awaited_once_with({}, receive, send)

    def test_configured_upstream_timeout_seconds_defaults_to_300(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GHCP_UPSTREAM_TIMEOUT_SECONDS", None)
            self.assertEqual(proxy.configured_upstream_timeout_seconds(), 300)

    def test_configured_upstream_timeout_seconds_uses_env_override(self):
        with mock.patch.dict(os.environ, {"GHCP_UPSTREAM_TIMEOUT_SECONDS": "480"}, clear=False):
            self.assertEqual(proxy.configured_upstream_timeout_seconds(), 480)


if __name__ == "__main__":
    unittest.main()
