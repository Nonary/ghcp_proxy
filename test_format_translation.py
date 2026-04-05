import unittest
from types import SimpleNamespace
from unittest import mock

import gzip
import compression.zstd as pyzstd

import format_translation
import proxy
import usage_tracking

import httpx


class FormatTranslationTests(unittest.TestCase):
    def test_parse_json_request_accepts_gzip_encoded_body(self):
        raw = gzip.compress(b'{"model":"gpt-5","input":"hello"}')
        request = SimpleNamespace(
            headers={"content-encoding": "gzip"},
            body=mock.AsyncMock(return_value=raw),
        )

        parsed = proxy.asyncio.run(proxy.parse_json_request(request))

        self.assertEqual(parsed["model"], "gpt-5")
        self.assertEqual(parsed["input"], "hello")

    def test_parse_json_request_accepts_zstd_encoded_body(self):
        raw = pyzstd.compress(b'{"model":"gpt-5","input":"hello"}')
        request = SimpleNamespace(
            headers={"content-encoding": "zstd"},
            body=mock.AsyncMock(return_value=raw),
        )

        parsed = proxy.asyncio.run(proxy.parse_json_request(request))

        self.assertEqual(parsed["model"], "gpt-5")
        self.assertEqual(parsed["input"], "hello")

    def test_anthropic_request_to_chat_translates_cache_control_to_copilot_cache_control(self):
        body = {
            "model": "claude-sonnet-4.6",
            "system": [
                {"type": "text", "text": "first"},
                {
                    "type": "text",
                    "text": "cached",
                    "cache_control": {"ephemeral": {"scope": "conversation"}},
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(outbound["model"], "claude-sonnet-4.6")
        self.assertEqual(outbound["thinking_budget"], 4096)
        self.assertEqual(outbound["messages"][0]["role"], "system")
        self.assertIsInstance(outbound["messages"][0]["content"], list)
        self.assertNotIn("stream_options", outbound)
        self.assertEqual(
            outbound["messages"][0]["content"][1]["copilot_cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(
            outbound["messages"][1]["content"][0]["copilot_cache_control"],
            {"type": "ephemeral"},
        )

    def test_anthropic_request_to_chat_preserves_cache_control_on_tool_results(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        tool_msg = [m for m in outbound["messages"] if m["role"] == "tool"][0]
        self.assertEqual(
            tool_msg["copilot_cache_control"],
            {"type": "ephemeral"},
        )

    def test_anthropic_request_to_chat_no_cache_control_on_tool_result_without_it(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        tool_msg = [m for m in outbound["messages"] if m["role"] == "tool"][0]
        self.assertNotIn("copilot_cache_control", tool_msg)

    def test_build_fake_compaction_request_preserves_openai_xhigh_reasoning(self):
        body = {
            "model": "openai/gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "reasoning": {"effort": "xhigh", "summary": "auto"},
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["reasoning"], {"effort": "xhigh", "summary": "auto"})
        self.assertEqual(body["reasoning"], {"effort": "xhigh", "summary": "auto"})

    def test_build_fake_compaction_request_preserves_anthropic_max_reasoning(self):
        body = {
            "model": "anthropic/claude-sonnet-4.6",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "reasoning": {"effort": "max"},
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["reasoning"], {"effort": "max"})

    def test_anthropic_request_to_chat_stream_requests_usage_chunks(self):
        body = {
            "model": "claude-sonnet-4.6",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(outbound["stream_options"], {"include_usage": True})

    def test_chat_completion_to_anthropic_maps_cached_usage_tokens(self):
        payload = {
            "id": "chatcmpl_123",
            "model": "claude-sonnet-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_input_tokens": 20,
                },
            },
        }

        translated = format_translation.chat_completion_to_anthropic(payload)

        self.assertEqual(translated["usage"]["input_tokens"], 20)
        self.assertEqual(translated["usage"]["output_tokens"], 25)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 80)
        self.assertEqual(translated["usage"]["cache_creation_input_tokens"], 0)

    def test_chat_completion_to_anthropic_subtracts_cache_reads_from_input_tokens(self):
        payload = {
            "id": "chatcmpl_456",
            "model": "claude-sonnet-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 135200,
                "completion_tokens": 690,
                "prompt_tokens_details": {
                    "cached_tokens": 125000,
                },
            },
        }

        translated = format_translation.chat_completion_to_anthropic(payload)

        self.assertEqual(translated["usage"]["input_tokens"], 10200)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 125000)
        self.assertEqual(translated["usage"]["cache_creation_input_tokens"], 0)

    def test_anthropic_stream_refreshes_message_start_usage_when_openai_usage_arrives_late(self):
        chunks = [
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-haiku-4.5","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-haiku-4.5","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        class FakeUpstream:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in chunks:
                    yield chunk

            async def aread(self):
                return b""

            async def aclose(self):
                return None

        class FakeClient:
            def build_request(self, *args, **kwargs):
                return object()

            async def aclose(self):
                return None

        async def collect_stream_body():
            response = await proxy.proxy_anthropic_streaming_response(
                "https://example.invalid/chat/completions",
                {"Authorization": "Bearer test"},
                {"stream": True},
                "claude-haiku-4.5",
            )
            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            return body.decode("utf-8")

        with (
            mock.patch.object(proxy.httpx, "AsyncClient", return_value=FakeClient()),
            mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=FakeUpstream())),
        ):
            body = proxy.asyncio.run(collect_stream_body())

        self.assertEqual(body.count("event: message_start"), 2)
        self.assertIn(
            '"usage":{"input_tokens":20,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":10}',
            body,
        )

    def test_upstream_request_error_status_and_message_maps_timeout_to_504(self):
        request = httpx.Request("POST", "https://example.invalid/responses")
        timeout_error = httpx.ReadTimeout("Timed out", request=request)

        self.assertEqual(
            format_translation.upstream_request_error_status_and_message(timeout_error),
            (504, "Upstream request timed out: Timed out"),
        )

    def test_anthropic_error_payload_from_openai_uses_anthropic_shape(self):
        payload = {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
            }
        }

        translated = format_translation.anthropic_error_payload_from_openai(payload, 400)

        self.assertEqual(
            translated,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad request",
                },
            },
        )

    def test_anthropic_error_response_from_upstream_translates_openai_error(self):
        upstream = httpx.Response(
            429,
            json={
                "error": {
                    "message": "rate limited",
                }
            },
            headers={"retry-after": "12"},
        )

        response = format_translation.anthropic_error_response_from_upstream(upstream)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "12")
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}',
        )

    def test_resolve_copilot_model_name_maps_dated_haiku_to_canonical_form(self):
        self.assertEqual(
            format_translation.resolve_copilot_model_name("claude-haiku-4-5-20251001"),
            "claude-haiku-4.5",
        )


if __name__ == "__main__":
    unittest.main()
