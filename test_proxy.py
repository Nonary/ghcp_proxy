import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import initiator_policy
import proxy


class ProxyInitiatorTests(unittest.TestCase):
    def setUp(self):
        proxy._initiator_policy = initiator_policy.InitiatorPolicy()

    def test_responses_requests_default_to_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "hello",
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"], "hello")

    def test_underscore_prefixed_responses_string_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "_hello",
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_only_latest_responses_user_item_controls_agent_prefix(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "_old request"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "new request"},
            ],
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"][-1]["content"], "new request")

    def test_chat_underscore_prefixed_user_message_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "assistant", "content": "prior work"},
            {"role": "user", "content": "_finish the task"},
        ]

        headers = proxy.build_chat_headers_for_request(request, messages, "gpt-4.1", "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(messages[-1]["content"], "finish the task")

    def test_anthropic_underscore_prefixed_user_message_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "_hello"}],
                }
            ],
        }

        headers = proxy.build_anthropic_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
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
            "input": "hello",
        }

        headers = proxy.build_responses_headers_for_request(
            request,
            body,
            "test-key",
            force_initiator="agent",
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_haiku_requests_are_always_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "claude-haiku-4.5",
            "input": "hello",
        }

        headers = proxy.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_active_request_forces_following_user_request_to_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=5)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

    def test_request_resolution_with_request_id_marks_activity_for_other_requests(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 5, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5", request_id="req-1"), "user")

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=1)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5", request_id="req-2"), "agent")

    def test_recent_finished_request_forces_following_user_looking_request_to_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        finished_at = datetime(2026, 4, 4, 18, 10, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=finished_at.replace(second=0))
        policy.note_request_finished("req-1", finished_at=finished_at)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=finished_at.replace(second=10)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

    def test_guard_expires_15_seconds_after_request_finishes(self):
        policy = initiator_policy.InitiatorPolicy()
        base = datetime(2026, 4, 4, 18, 20, tzinfo=timezone.utc)

        policy.note_request_started("req-0", "user", started_at=base)
        policy.note_request_finished("req-0", finished_at=base.replace(second=5))

        policy.note_request_started("req-1", "agent", started_at=base.replace(second=10))
        policy.note_request_finished("req-1", finished_at=base.replace(second=12))

        with mock.patch.object(initiator_policy, "_utc_now", return_value=base.replace(second=25)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "_utc_now", return_value=base.replace(second=28)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_any_request_activity_refreshes_the_15_second_timer(self):
        policy = initiator_policy.InitiatorPolicy()
        first_finish = datetime(2026, 4, 4, 18, 40, 5, tzinfo=timezone.utc)
        second_start = datetime(2026, 4, 4, 18, 40, 14, tzinfo=timezone.utc)
        second_finish = datetime(2026, 4, 4, 18, 40, 18, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=first_finish.replace(second=0))
        policy.note_request_finished("req-1", finished_at=first_finish)

        policy.note_request_started("req-2", "agent", started_at=second_start)
        policy.note_request_finished("req-2", finished_at=second_finish)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=datetime(2026, 4, 4, 18, 40, 30, tzinfo=timezone.utc)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "_utc_now", return_value=datetime(2026, 4, 4, 18, 40, 34, tzinfo=timezone.utc)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_stream_like_request_stays_active_until_finished(self):
        policy = initiator_policy.InitiatorPolicy()
        started_at = datetime(2026, 4, 4, 18, 30, tzinfo=timezone.utc)
        finished_at = datetime(2026, 4, 4, 18, 31, tzinfo=timezone.utc)

        policy.note_request_started("stream-1", "user", started_at=started_at)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=finished_at):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        policy.note_request_finished("stream-1", finished_at=finished_at)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=finished_at.replace(second=10)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "_utc_now", return_value=finished_at.replace(second=16)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_safeguard_inactive_until_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "agent", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=10))

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=12)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_active_stream_does_not_block_user_before_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("stream-1", "agent", started_at=start)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=5)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_haiku_then_opus_user_prompt_is_user(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        policy.note_request_finished("haiku-1", finished_at=start.replace(second=2))

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=3)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-sonnet-4.6", request_id="opus-1"),
                "user",
            )

    def test_haiku_streaming_does_not_block_first_user_opus(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=1)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-sonnet-4.6", request_id="opus-1"),
                "user",
            )

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=2)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-sonnet-4.6"),
                "agent",
            )

    def test_safeguard_activates_after_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "agent", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=6)):
            self.assertEqual(
                policy.resolve_initiator("user", "gpt-5", request_id="req-2"),
                "user",
            )

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=7)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

    def test_safeguard_resets_after_expiry_so_haiku_cannot_reactivate(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start):
            policy.resolve_initiator("user", "gpt-5", request_id="req-1")

        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=10)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=25)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        policy.note_request_finished("haiku-1", finished_at=start.replace(second=26))

        with mock.patch.object(initiator_policy, "_utc_now", return_value=start.replace(second=27)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-opus-4.6", request_id="opus-1"),
                "user",
            )

    def test_seeded_user_history_does_not_pre_activate_safeguard(self):
        policy = initiator_policy.InitiatorPolicy()
        old_time = datetime(2026, 4, 4, 17, 0, tzinfo=timezone.utc)
        now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.seed_from_usage_events([
            {"finished_at": old_time.isoformat(), "initiator": "user"},
            {"finished_at": old_time.replace(second=5).isoformat(), "initiator": "agent"},
        ])

        with mock.patch.object(initiator_policy, "_utc_now", return_value=now):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        with mock.patch.object(initiator_policy, "_utc_now", return_value=now.replace(second=1)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-opus-4.6", request_id="opus-1"),
                "user",
            )

    def test_enabling_codex_proxy_is_idempotent_when_already_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(proxy.CODEX_PROXY_CONFIG + "\n", encoding="utf-8")

            with mock.patch.object(proxy, "CODEX_CONFIG_FILE", str(config_path)):
                status = proxy._write_codex_proxy_config()

            backups = list(Path(tmp).glob("config.toml.ghcp-proxy.bak.*"))
            self.assertTrue(status["configured"])
            self.assertEqual(status["status_message"], "proxy already enabled")
            self.assertEqual(backups, [])

    def test_disabling_codex_proxy_is_idempotent_when_already_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"

            with mock.patch.object(proxy, "CODEX_CONFIG_FILE", str(config_path)):
                status = proxy._disable_codex_proxy_config()

            self.assertFalse(status["configured"])
            self.assertEqual(status["status_message"], "proxy already disabled")
            self.assertIsNone(status["backup_path"])

    def test_disabling_codex_proxy_restores_latest_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            backup_path = Path(f"{config_path}.ghcp-proxy.bak.20260404_180000")
            config_path.write_text(proxy.CODEX_PROXY_CONFIG + "\n", encoding="utf-8")
            backup_contents = 'model_provider = "openai"\n'
            backup_path.write_text(backup_contents, encoding="utf-8")

            with mock.patch.object(proxy, "CODEX_CONFIG_FILE", str(config_path)):
                status = proxy._disable_codex_proxy_config()

            self.assertFalse(status["configured"])
            self.assertTrue(status["restored_from_backup"])
            self.assertFalse(backup_path.exists())
            self.assertEqual(config_path.read_text(encoding="utf-8"), backup_contents)

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
