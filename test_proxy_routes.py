import asyncio
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
            response_text="Upstream connection failed: All connection attempts failed",
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Upstream connection failed: All connection attempts failed","type":"server_error","param":null,"code":null}}',
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
            response_text="Upstream connection failed: All connection attempts failed",
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"api_error","message":"Upstream connection failed: All connection attempts failed"}}',
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
            usage_event,
            502,
            response_text="Upstream connection failed: All connection attempts failed",
        )
        fake_client.aclose.assert_awaited_once()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Upstream connection failed: All connection attempts failed","type":"server_error","param":null,"code":null}}',
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
