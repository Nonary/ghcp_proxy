import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

import proxy


class ProxyTracingTests(unittest.TestCase):
    def test_prepare_upstream_request_keeps_full_prompt_preview_for_user_initiator(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        long_system = "system-" + ("s" * 5000)
        long_user = "user-" + ("u" * 5000)
        source_body = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": long_user},
            ],
        }

        with (
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value={"request_id": "req_123"}) as start_event,
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
        ):
            plan, error_response = proxy._prepare_upstream_request(
                request,
                body=source_body,
                requested_model="gpt-5.4",
                resolved_model="gpt-5.4",
                upstream_path="/chat/completions",
                upstream_url="https://example.invalid/chat/completions",
                header_builder=lambda _api_key, _request_id: {"X-Initiator": "user"},
                error_response=proxy.format_translation.openai_error_response,
                api_key="test-key",
            )

        self.assertIsNone(error_response)
        self.assertIsNotNone(plan)

        prompt_preview = start_event.call_args.kwargs["prompt_preview"]
        self.assertEqual(prompt_preview["system"], long_system)
        self.assertEqual(prompt_preview["user"], long_user)
        self.assertNotIn("system_truncated", prompt_preview)
        self.assertNotIn("user_truncated", prompt_preview)

        trace_payload = append_trace.call_args.args[0]
        self.assertEqual(trace_payload["request_prompt"]["system"], long_system)
        self.assertEqual(trace_payload["request_prompt"]["user"], long_user)

    def test_extract_prompt_preview_truncates_agent_requests_normally(self):
        long_system = "system-" + ("s" * 5000)
        long_user = "user-" + ("u" * 5000)
        body = {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": long_user},
            ]
        }

        preview = proxy._extract_prompt_preview(body)

        self.assertTrue(preview["system_truncated"])
        self.assertTrue(preview["user_truncated"])
        self.assertIn("…[truncated; original ", preview["system"])
        self.assertIn("…[truncated; original ", preview["user"])

    def test_prepare_upstream_request_keeps_source_body_for_bridge_traces(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        source_body = {"model": "gpt-5.4-mini", "input": "hello"}
        upstream_body = {"model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "hello"}]}

        with (
            mock.patch.object(proxy, "request_tracing_enabled", return_value=False),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value={"request_id": "req_123"}),
        ):
            plan, error_response = proxy._prepare_upstream_request(
                request,
                body=upstream_body,
                source_body=source_body,
                requested_model="gpt-5.4-mini",
                resolved_model="claude-sonnet-4.6",
                upstream_path="/chat/completions",
                upstream_url="https://example.invalid/chat/completions",
                header_builder=lambda _api_key, _request_id: {"X-Initiator": "user"},
                error_response=proxy.format_translation.openai_error_response,
                api_key="test-key",
                trace_metadata={"bridge": True, "strategy_name": "responses_to_chat"},
            )

        self.assertIsNone(error_response)
        self.assertEqual(plan.source_body, source_body)
        self.assertEqual(plan.body, upstream_body)
        self.assertTrue(plan.trace_context["bridge"])
        self.assertEqual(plan.trace_context["strategy_name"], "responses_to_chat")

    def test_header_trace_subset_keeps_subagent_header(self):
        subset = proxy._header_trace_subset(
            {
                "X-OpenAI-Subagent": "collab_spawn",
                "X-Initiator": "agent",
                "Authorization": "secret",
            }
        )

        self.assertEqual(
            subset,
            {
                "X-OpenAI-Subagent": "collab_spawn",
                "X-Initiator": "agent",
            },
        )

    def test_finish_usage_and_trace_emits_failure_diagnostic_for_bridge_request(self):
        plan = proxy.UpstreamRequestPlan(
            request_id="req_123",
            upstream_url="https://example.invalid/chat/completions",
            headers={"X-Initiator": "user", "Authorization": "Bearer hidden"},
            body={"model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "hello"}]},
            usage_event={"request_id": "req_123"},
            requested_model="gpt-5.4-mini",
            resolved_model="claude-sonnet-4.6",
            source_body={"model": "gpt-5.4-mini", "input": "hello"},
            trace_context={
                "request_id": "req_123",
                "client_path": "/v1/responses",
                "upstream_path": "/chat/completions",
                "bridge": True,
                "strategy_name": "responses_to_chat",
            },
        )
        upstream = httpx.Response(400, text="unsupported field: tool_choice")
        response_payload = {"error": {"type": "invalid_request_error", "message": "unsupported field: tool_choice"}}

        with (
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "request_tracing_enabled", return_value=False),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
        ):
            proxy._finish_usage_and_trace(
                plan,
                400,
                upstream=upstream,
                response_payload=response_payload,
                response_text="unsupported field: tool_choice",
            )

        append_trace.assert_called_once()
        trace_payload = append_trace.call_args.args[0]
        trace_kwargs = append_trace.call_args.kwargs

        self.assertTrue(trace_kwargs["force"])
        self.assertEqual(trace_payload["requested_model"], "gpt-5.4-mini")
        self.assertEqual(trace_payload["resolved_model"], "claude-sonnet-4.6")
        self.assertEqual(trace_payload["source_body"], {"model": "gpt-5.4-mini", "input": "hello"})
        self.assertEqual(trace_payload["upstream_body"]["model"], "claude-sonnet-4.6")
        self.assertEqual(trace_payload["outbound_headers"], {"X-Initiator": "user"})
        self.assertEqual(trace_payload["response_text"], "unsupported field: tool_choice")


    def test_finish_usage_and_trace_records_reasoning_text_on_success(self):
        plan = proxy.UpstreamRequestPlan(
            request_id="req_999",
            upstream_url="https://example.invalid/chat/completions",
            headers={"X-Initiator": "agent"},
            body={"model": "claude-opus-4.7"},
            usage_event={"request_id": "req_999"},
            requested_model="claude-opus-4.7",
            resolved_model="claude-opus-4.7",
            source_body={"model": "claude-opus-4.7"},
            trace_context={"request_id": "req_999"},
        )
        upstream = httpx.Response(200, text="")

        captured: dict = {}

        def _capture(_event, _status, **kwargs):
            captured.update(kwargs)

        with (
            mock.patch.object(proxy.usage_tracker, "finish_event", side_effect=_capture),
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
        ):
            proxy._finish_usage_and_trace(
                plan,
                200,
                upstream=upstream,
                response_payload={"id": "resp_1"},
                response_text="answer",
                reasoning_text="thinking step by step",
            )

        self.assertEqual(captured["reasoning_text"], "thinking step by step")
        append_trace.assert_called_once()
        trace_payload = append_trace.call_args.args[0]
        self.assertTrue(trace_payload["reasoning_text_present"])
        self.assertEqual(trace_payload["reasoning_text"], "thinking step by step")


if __name__ == "__main__":
    unittest.main()
