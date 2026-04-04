import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import proxy


class ProxyInitiatorTests(unittest.TestCase):
    def test_responses_requests_default_to_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "hello",
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_plus_prefixed_responses_string_is_user_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "+hello",
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"], "hello")

    def test_only_latest_responses_user_item_can_opt_out_to_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "+old request"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "new request"},
            ],
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"][-1]["content"], "new request")

    def test_chat_plus_prefixed_user_message_is_user_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "assistant", "content": "prior work"},
            {"role": "user", "content": "+finish the task"},
        ]

        headers = proxy.build_chat_headers_for_request(request, messages, "gpt-4.1", "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(messages[-1]["content"], "finish the task")

    def test_anthropic_plus_prefixed_user_message_is_user_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "+hello"}],
                }
            ],
        }

        headers = proxy.build_anthropic_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["messages"][0]["content"][0]["text"], "hello")

    def test_prepare_anthropic_outbound_body_strips_nested_cache_control(self):
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

        outbound = proxy.prepare_anthropic_outbound_body(body, "claude-sonnet-4.6")

        self.assertNotIn("cache_control", outbound["system"][1])
        self.assertNotIn("cache_control", outbound["messages"][0]["content"][0])

    def test_forced_agent_responses_requests_stay_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "+hello",
        }

        headers = proxy.build_responses_headers_for_request(
            request,
            body,
            "test-key",
            force_initiator="agent",
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_month_key_for_source_rows(self):
        self.assertEqual(proxy._month_key_for_source_row("claude", {"month": "2026-04"}), "2026-04")
        self.assertEqual(proxy._month_key_for_source_row("codex", {"month": "Apr 2026"}), "2026-04")

    def test_build_dashboard_payload_combines_claude_and_codex(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        ccusage_payload = {
            "available": True,
            "sources": {
                "claude": {
                    "available": True,
                    "monthly": {
                        "monthly": [
                            {
                                "month": "2026-04",
                                "inputTokens": 100,
                                "outputTokens": 20,
                                "cacheCreationTokens": 10,
                                "cacheReadTokens": 30,
                                "totalTokens": 160,
                                "totalCost": 1.25,
                                "modelsUsed": ["claude-sonnet-4-6"],
                            }
                        ]
                    },
                    "sessions": {
                        "sessions": [
                            {
                                "sessionId": "claude-session",
                                "lastActivity": "2026-04-04T17:00:00Z",
                                "inputTokens": 100,
                                "outputTokens": 20,
                                "cacheCreationTokens": 10,
                                "cacheReadTokens": 30,
                                "totalTokens": 160,
                                "totalCost": 1.25,
                                "modelsUsed": ["claude-sonnet-4-6"],
                            }
                        ]
                    },
                },
                "codex": {
                    "available": True,
                    "monthly": {
                        "monthly": [
                            {
                                "month": "Apr 2026",
                                "inputTokens": 200,
                                "outputTokens": 40,
                                "cachedInputTokens": 50,
                                "reasoningOutputTokens": 12,
                                "totalTokens": 240,
                                "costUSD": 2.75,
                                "models": {"gpt-5.4": {"inputTokens": 200}},
                            }
                        ]
                    },
                    "sessions": {
                        "sessions": [
                            {
                                "sessionId": "codex-session",
                                "lastActivity": "2026-04-04T18:30:00Z",
                                "inputTokens": 200,
                                "outputTokens": 40,
                                "cachedInputTokens": 50,
                                "reasoningOutputTokens": 12,
                                "totalTokens": 240,
                                "costUSD": 2.75,
                                "models": {"gpt-5.4": {"inputTokens": 200}},
                            }
                        ]
                    },
                },
            },
            "errors": [],
        }

        usage_events = [
            {
                "started_at": "2026-04-04T17:50:00+00:00",
                "finished_at": "2026-04-04T17:51:00+00:00",
                "premium_requests": 1.0,
                "path": "/v1/responses",
            },
            {
                "started_at": "2026-04-04T17:55:00+00:00",
                "finished_at": "2026-04-04T17:56:00+00:00",
                "premium_requests": 0.33,
                "path": "/v1/messages",
            },
        ]

        with (
            mock.patch.object(proxy, "_utc_now", return_value=fixed_now),
            mock.patch.object(proxy, "_snapshot_usage_events", return_value=usage_events),
            mock.patch.object(proxy, "_load_api_key_payload", return_value={"sku": "plus_monthly_subscriber_quota"}),
            mock.patch.object(proxy, "_get_ccusage_payload", return_value=ccusage_payload),
            mock.patch.object(
                proxy,
                "_get_official_premium_payload",
                return_value={
                    "available": True,
                    "remaining": 1420,
                    "used": 80,
                    "included": 1500,
                    "reset_date": None,
                    "source": "github-rest-billing-api",
                    "raw": {},
                    "refreshing": False,
                    "error": None,
                },
            ),
        ):
            payload = proxy._build_dashboard_payload()

        self.assertEqual(payload["premium"]["included"], 1500)
        self.assertEqual(payload["premium"]["used"], 80)
        self.assertEqual(payload["premium"]["official_remaining"], 1420)
        self.assertEqual(payload["current_month"]["ccusage"]["cost_usd"], 4.0)
        self.assertEqual(payload["current_month"]["ccusage"]["total_tokens"], 400)
        self.assertEqual(payload["recent_sessions"][0]["source"], "codex")
        self.assertEqual(payload["recent_sessions"][1]["source"], "claude")


if __name__ == "__main__":
    unittest.main()
