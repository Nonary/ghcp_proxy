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
        proxy._COPILOT_MODEL_CAPS_CACHE.update({"key": None, "ts": 0.0, "data": {}})
        proxy._CLIENT_PROXY_STARTUP_RESTORE_COMPLETE = False
        proxy._CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE = False

    def test_proxy_registers_copilot_native_root_aliases(self):
        paths = {route.path for route in proxy.app.routes}
        self.assertIn("/responses", paths)
        self.assertIn("/responses/compact", paths)
        self.assertIn("/chat/completions", paths)
        self.assertIn("/models", paths)
        self.assertIn("/v1/responses", paths)
        self.assertIn("/v1/chat/completions", paths)
        self.assertIn("/v1/models", paths)
        self.assertIn("/api/config/background-proxy", paths)

    def test_background_proxy_status_route_uses_manager(self):
        manager = SimpleNamespace(status_payload=mock.Mock(return_value={"startup_enabled": True}))
        with mock.patch.object(proxy, "background_proxy_manager", manager):
            response = proxy.asyncio.run(proxy.background_proxy_status_api())

        manager.status_payload.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {"startup_enabled": True})

    def test_models_route_proxies_upstream_copilot_models(self):
        upstream = httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "gpt-5.4",
                        "name": "GPT 5.4",
                        "version": "gpt-5.4-2026-01-01",
                        "model_picker_enabled": True,
                        "capabilities": {
                            "family": "gpt-5",
                            "limits": {
                                "max_context_window_tokens": 400000,
                                "max_output_tokens": 128000,
                                "max_prompt_tokens": 272000,
                            },
                            "supports": {
                                "streaming": True,
                                "tool_calls": True,
                            },
                        },
                    }
                ]
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "build_copilot_headers", return_value={"Authorization": "Bearer test-key"}) as build_headers,
            mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=upstream)) as send,
        ):
            response = proxy.asyncio.run(proxy.models())

        build_headers.assert_called_once_with("test-key")
        request = send.await_args.args[1]
        self.assertEqual(request.method, "GET")
        self.assertEqual(str(request.url), "https://example.invalid/models")
        self.assertEqual(request.headers.get("Authorization"), "Bearer test-key")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, upstream.content)

    def test_auth_status_api_disables_http_caching(self):
        payload = {"authenticated": False, "state": "unauthenticated"}

        with mock.patch.object(auth, "auth_status", return_value=payload) as auth_status:
            response = proxy.asyncio.run(proxy.auth_status_api())

        auth_status.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(json.loads(response.body), payload)

    def test_auth_device_api_starts_browser_auth_flow(self):
        payload = {"authenticated": False, "state": "pending", "user_code": "ABCD-EFGH"}

        with mock.patch.object(auth, "begin_device_flow", return_value=payload) as begin_device_flow:
            response = proxy.asyncio.run(proxy.auth_device_api())

        begin_device_flow.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(json.loads(response.body), payload)

    def test_fetch_copilot_model_capabilities_uses_prompt_window_for_all_400k_models(self):
        upstream = mock.Mock()
        upstream.raise_for_status.return_value = None
        upstream.json.return_value = {
            "data": [
                {
                    "id": "gpt-5.4",
                    "capabilities": {
                        "limits": {
                            "max_context_window_tokens": 400000,
                            "max_prompt_tokens": 272000,
                        },
                        "supports": {
                            "reasoning_effort": ["low", "medium", "high", "xhigh"],
                            "parallel_tool_calls": True,
                        },
                    },
                },
                {
                    "id": "gpt-5.4-mini",
                    "capabilities": {
                        "limits": {
                            "max_context_window_tokens": 400000,
                            "max_prompt_tokens": 272000,
                        },
                        "supports": {
                            "reasoning_effort": ["low", "medium", "high"],
                            "parallel_tool_calls": True,
                        },
                    },
                },
                {
                    "id": "gpt-5.3-codex",
                    "capabilities": {
                        "limits": {
                            "max_context_window_tokens": 400000,
                            "max_prompt_tokens": 272000,
                        },
                        "supports": {
                            "reasoning_effort": ["low", "medium", "high", "xhigh"],
                            "parallel_tool_calls": True,
                        },
                    },
                },
                {
                    "id": "gpt-4.1",
                    "capabilities": {
                        "limits": {
                            "max_context_window_tokens": 128000,
                        },
                        "supports": {
                            "parallel_tool_calls": True,
                        },
                    },
                },
            ]
        }
        client = mock.MagicMock()
        client.__enter__.return_value = client
        client.get.return_value = upstream

        with (
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "build_copilot_headers", return_value={"Authorization": "Bearer test-key"}),
            mock.patch.object(proxy.httpx, "Client", return_value=client),
        ):
            caps = proxy.fetch_copilot_model_capabilities()

        for model_name in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"):
            self.assertEqual(caps[model_name]["context_window"], 272000)
            self.assertEqual(caps[model_name]["max_context_window"], 400000)
        self.assertEqual(caps["gpt-4.1"]["context_window"], 128000)
        self.assertEqual(caps["gpt-4.1"]["max_context_window"], 128000)

    def test_model_routing_config_route_refreshes_codex_catalog(self):
        request = SimpleNamespace()
        payload = {"enabled": True, "mappings": [{"source_model": "gpt-5.4", "target_model": "claude-sonnet-4.6"}]}
        saved = {
            "enabled": True,
            "mappings": [{"source_model": "gpt-5.4", "target_model": "claude-sonnet-4.6"}],
            "approval_enabled": False,
            "approval_mappings": [],
            "available_models": [],
            "path": "ignored",
        }

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=payload)),
            mock.patch.object(proxy.model_routing_config_service, "save_settings", return_value=saved) as save_settings,
            mock.patch.object(proxy.client_proxy_config_service, "refresh_codex_model_catalog", return_value=True) as refresh_catalog,
        ):
            response = proxy.asyncio.run(proxy.model_routing_config_api(request))

        save_settings.assert_called_once_with(payload)
        refresh_catalog.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), saved)

    def test_client_proxy_status_route_includes_settings_payload(self):
        payload = {
            "clients": {
                "codex": {"client": "codex"},
                "claude": {"client": "claude"},
            },
            "settings": {"revert_on_shutdown": True, "path": "/tmp/client-proxy.json"},
        }

        with mock.patch.object(
            proxy.client_proxy_config_service,
            "proxy_client_status_payload",
            return_value=payload,
        ) as status_payload:
            response = proxy.asyncio.run(proxy.client_proxy_status_api())

        status_payload.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), payload)

    def test_client_proxy_settings_route_saves_settings(self):
        request = SimpleNamespace()
        payload = {"revert_on_shutdown": False}
        saved = {"revert_on_shutdown": False, "path": "/tmp/client-proxy.json"}

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=payload)),
            mock.patch.object(
                proxy.client_proxy_config_service,
                "save_client_proxy_settings",
                return_value=saved,
            ) as save_settings,
        ):
            response = proxy.asyncio.run(proxy.client_proxy_settings_api(request))

        save_settings.assert_called_once_with(payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), saved)

    def test_revert_client_proxy_configs_on_shutdown_is_idempotent(self):
        result = {
            "attempted": True,
            "reverted": True,
            "reason": "shutdown",
            "clients": {},
        }

        with mock.patch.object(
            proxy.client_proxy_config_service,
            "revert_proxy_configs_on_shutdown",
            return_value=result,
        ) as revert_configs:
            first = proxy.revert_client_proxy_configs_on_shutdown()
            second = proxy.revert_client_proxy_configs_on_shutdown()

        revert_configs.assert_called_once_with()
        self.assertEqual(first, result)
        self.assertEqual(second["reason"], "already-ran")

    def test_restore_client_proxy_configs_on_startup_is_idempotent(self):
        result = {
            "attempted": True,
            "restored": True,
            "reason": "startup",
            "clients": {},
        }

        with mock.patch.object(
            proxy.client_proxy_config_service,
            "restore_proxy_configs_on_startup",
            return_value=result,
        ) as restore_configs:
            first = proxy.restore_client_proxy_configs_on_startup()
            second = proxy.restore_client_proxy_configs_on_startup()

        restore_configs.assert_called_once_with()
        self.assertEqual(first, result)
        self.assertEqual(second["reason"], "already-ran")

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
            mock.patch.object(proxy, "model_supports_native_messages", return_value=False),
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
            reasoning_text=None,
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
            reasoning_text=None,
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
            mock.patch.object(proxy, "model_supports_native_messages", return_value=False),
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
            mock.patch.object(proxy, "model_supports_native_messages", return_value=False),
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

    def test_responses_route_strips_unsupported_image_generation_tool(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "tools": [
                {"type": "image_generation"},
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
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

        forwarded_body = post.await_args.kwargs["json"]
        self.assertEqual(
            forwarded_body["tools"],
            [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        )
        self.assertEqual(forwarded_body["tool_choice"], "auto")
        self.assertTrue(forwarded_body["parallel_tool_calls"])
        self.assertEqual(response.status_code, 200)

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

    def test_responses_compact_wraps_chat_summary_for_translated_model(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses/compact"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.2",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "please inspect this"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I found the issue."}],
                },
            ],
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
                            "content": "summary",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": 5,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="claude-opus-4.6"),
            mock.patch.object(proxy, "model_supports_native_messages", return_value=False),
            mock.patch.object(proxy.model_routing_config_service, "resolve_compact_fallback_model", return_value="gpt-5.2"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses_compact(request))

        forwarded_body = post.await_args.kwargs["json"]
        self.assertEqual(forwarded_body["model"], "claude-opus-4.6")
        self.assertEqual([message["role"] for message in forwarded_body["messages"]], ["user", "user", "user"])
        self.assertIn("please inspect this", forwarded_body["messages"][0]["content"])
        self.assertIn("[assistant message]\nI found the issue.", forwarded_body["messages"][1]["content"])
        self.assertIn("Please create a detailed summary", forwarded_body["messages"][2]["content"])
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["output"][0]["type"], "compaction")
        self.assertEqual(
            format_translation.decode_fake_compaction(payload["output"][0]["encrypted_content"]),
            "summary",
        )

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
            reasoning_text=None,
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

# ─── Native Anthropic Messages passthrough route tests ───────────────────────


from protocol_bridge import BridgeExecutionPlan as _BridgeExecutionPlan_for_msgs


class AnthropicMessagesPassthroughRouteTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()

    def _messages_request(self):
        return SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-client-request-id": "req-msg-1", "anthropic-beta": "context-management-2025-06-27"},
        )

    def test_messages_route_uses_messages_passthrough_upstream(self):
        request = self._messages_request()
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        # Build a synthetic plan that drives MessagesToMessages.
        plan = _BridgeExecutionPlan_for_msgs(
            strategy_name="messages_to_messages",
            inbound_protocol="messages",
            caller_protocol="anthropic",
            upstream_protocol="messages",
            header_kind="messages",
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            upstream_body=dict(body),
            stream=False,
            is_compact=False,
        )
        upstream = httpx.Response(
            200,
            json={
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4.6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            headers={"content-type": "application/json"},
        )

        async def fake_plan(*args, **kwargs):
            return plan

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.bridge_planner, "plan", side_effect=fake_plan),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.await_args.args[1], "https://example.invalid/v1/messages")
        sent_headers = post.await_args.kwargs["headers"]
        # Required: anthropic-beta and messages-proxy interaction type, NO copilot-integration-id
        self.assertIn("anthropic-beta", sent_headers)
        self.assertIn("context-management-2025-06-27", sent_headers["anthropic-beta"])
        self.assertEqual(sent_headers.get("x-interaction-type"), "messages-proxy")
        lower_keys = {k.lower() for k in sent_headers}
        self.assertNotIn("copilot-integration-id", lower_keys)
        # Body should be the upstream Anthropic body, unchanged shape
        self.assertEqual(post.await_args.kwargs["json"]["model"], "claude-sonnet-4.6")
        # Response is passthrough - same Anthropic shape
        payload = json.loads(response.body)
        self.assertEqual(payload["id"], "msg_x")
        self.assertEqual(payload["content"][0]["text"], "ok")

    def test_messages_passthrough_stream_preserves_cache_usage(self):
        chunks = [
            (
                'event: message_start\n'
                'data: {"type":"message_start","message":{"id":"msg_x","type":"message",'
                '"role":"assistant","usage":{"input_tokens":1200,"output_tokens":0,'
                '"cache_read_input_tokens":125000,"cache_creation_input_tokens":42}}}\n\n'
            ).encode("utf-8"),
            (
                'event: message_delta\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":321}}\n\n'
            ).encode("utf-8"),
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

        class FakeUpstream:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                for chunk in chunks:
                    yield chunk

            async def aclose(self):
                return None

        class FakeClient:
            def build_request(self, *args, **kwargs):
                return httpx.Request("POST", "https://example.invalid/v1/messages")

            async def aclose(self):
                return None

        usage_event = {"request_id": "req-stream-cache"}
        trace_plan = proxy.UpstreamRequestPlan(
            request_id="req-stream-cache",
            upstream_url="https://example.invalid/v1/messages",
            headers={},
            body={"stream": True},
            usage_event=usage_event,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
        )

        async def run_stream():
            with (
                mock.patch.object(proxy.httpx, "AsyncClient", return_value=FakeClient()),
                mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=FakeUpstream())),
                mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            ):
                response = await proxy.proxy_anthropic_passthrough_streaming_response(
                    "https://example.invalid/v1/messages",
                    {"Authorization": "Bearer test"},
                    {"model": "claude-sonnet-4.6", "stream": True},
                    "claude-sonnet-4.6",
                    usage_event=usage_event,
                    trace_plan=trace_plan,
                )
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                return body, finish_usage

        body, finish_usage = proxy.asyncio.run(run_stream())

        self.assertIn(b"cache_read_input_tokens", body)
        finish_usage.assert_called_once()
        usage = finish_usage.call_args.kwargs["usage"]
        self.assertEqual(usage["input_tokens"], 1200)
        self.assertEqual(usage["output_tokens"], 321)
        self.assertEqual(usage["cached_input_tokens"], 125000)
        self.assertEqual(usage["cache_creation_input_tokens"], 42)

    def test_responses_route_to_messages_translates_back_to_responses(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={"x-client-request-id": "req-msg-2"},
        )
        body = {
            "model": "claude-sonnet-4.6",
            "input": "hi",
            "stream": False,
        }
        translated_anthropic = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        plan = _BridgeExecutionPlan_for_msgs(
            strategy_name="responses_to_messages",
            inbound_protocol="responses",
            caller_protocol="responses",
            upstream_protocol="messages",
            header_kind="messages",
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            upstream_body=translated_anthropic,
            stream=False,
            is_compact=False,
        )
        upstream = httpx.Response(
            200,
            json={
                "id": "msg_y",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4.6",
                "content": [{"type": "text", "text": "translated"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            headers={"content-type": "application/json"},
        )

        async def fake_plan(*args, **kwargs):
            return plan

        translator_called = {"count": 0}

        def fake_translator(payload, *, fallback_model=None):
            translator_called["count"] += 1
            return {
                "id": "resp_translated",
                "object": "response",
                "model": fallback_model or payload.get("model"),
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "translated"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.bridge_planner, "plan", side_effect=fake_plan),
            mock.patch.object(format_translation, "anthropic_response_to_responses", create=True, side_effect=fake_translator),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.await_args.args[1], "https://example.invalid/v1/messages")
        self.assertEqual(translator_called["count"], 1)
        payload = json.loads(response.body)
        self.assertEqual(payload["id"], "resp_translated")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "translated")


    def test_responses_to_messages_upstream_error_returns_openai_error_shape(self):
        bridge_plan = SimpleNamespace(caller_protocol="responses", upstream_protocol="messages")
        upstream = httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad anthropic request",
                },
            },
            headers={"content-type": "application/json"},
        )

        response = proxy._bridge_error_response_from_upstream(bridge_plan, upstream)
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["message"], "bad anthropic request")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertNotEqual(payload.get("type"), "error")

    def test_prepare_upstream_request_reads_lowercase_initiator_header(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )

        with mock.patch.object(proxy.usage_tracker, "start_event", return_value=None) as start_event:
            plan, error = proxy._prepare_upstream_request(
                request,
                body={"model": "claude-sonnet-4.6", "messages": []},
                requested_model="claude-sonnet-4.6",
                resolved_model="claude-sonnet-4.6",
                upstream_path="/v1/messages",
                upstream_url="https://example.invalid/v1/messages",
                header_builder=lambda _api_key, _request_id: {"x-initiator": "agent"},
                error_response=format_translation.openai_error_response,
                api_key="test-key",
            )

        self.assertIsNone(error)
        self.assertIsNotNone(plan)
        self.assertEqual(start_event.call_args.args[3], "agent")


    def test_bridge_planner_end_to_end_dispatches_to_messages_url(self):
        """Regression: feed the *real* bridge_planner through to dispatch.

        Earlier tests synthesized BridgeExecutionPlan objects with
        ``header_kind="messages"`` directly, which masked a bug where the
        strategy classes returned ``"anthropic_messages"`` (mismatched with
        the dispatcher). This test exercises the full pipeline so a future
        regression in either side will fail loudly.
        """
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-client-request-id": "req-e2e-1"},
        )
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "msg_e2e",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4.6",
                "content": [{"type": "text", "text": "pong"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="claude-sonnet-4.6"),
            # Force native /v1/messages capability on (do NOT bypass the planner).
            mock.patch.object(proxy, "model_supports_native_messages", return_value=True),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        # The dispatcher must have selected the /v1/messages upstream URL,
        # which only happens when both the strategy header_kind and the
        # dispatcher branch agree on "messages".
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.await_args.args[1], "https://example.invalid/v1/messages")
        sent_headers = post.await_args.kwargs["headers"]
        self.assertEqual(sent_headers.get("x-interaction-type"), "messages-proxy")
        # Capability gating must drop the chat-style copilot-integration-id.
        lower_keys = {k.lower() for k in sent_headers}
        self.assertNotIn("copilot-integration-id", lower_keys)
