import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
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
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_LINEAGE.clear()
        with proxy._PROMPT_CACHE_TRACE_LOCK:
            proxy._PROMPT_CACHE_LAST_INPUT_BY_LINEAGE.clear()
        with proxy._PROMPT_CACHE_AFFINITY_TRACE_LOCK:
            proxy._PROMPT_CACHE_LAST_AFFINITY_BY_LINEAGE.clear()
        proxy._CLIENT_PROXY_STARTUP_RESTORE_COMPLETE = False
        proxy._CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE = False

    def test_prompt_cache_prefix_diagnostics_uses_full_item_hashes(self):
        first_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "cache-key",
            "input": [
                {"type": "function_call", "call_id": "call-1", "name": "Read", "arguments": "{\"path\":\"a\"}"},
            ],
        }
        second_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "cache-key",
            "input": [
                {"type": "function_call", "call_id": "call-1", "name": "Read", "arguments": "{\"path\":\"b\"}"},
            ],
        }

        first = proxy._prompt_cache_prefix_diagnostics(
            request_id="req-1",
            upstream_body=first_body,
            resolved_model="gpt-5.5",
        )
        second = proxy._prompt_cache_prefix_diagnostics(
            request_id="req-2",
            upstream_body=second_body,
            resolved_model="gpt-5.5",
        )

        self.assertIsNone(first["previous_request_id"])
        self.assertEqual(second["previous_request_id"], "req-1")
        self.assertFalse(second["previous_is_prefix"])
        self.assertEqual(second["first_mismatch_index"], 0)
        self.assertNotEqual(second["previous_item_hash"], second["current_item_hash"])

    def test_prompt_cache_prefix_diagnostics_marks_append_only_prefix(self):
        first_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "cache-key",
            "input": [{"type": "message", "role": "user", "content": "one"}],
        }
        second_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "cache-key",
            "input": [
                {"type": "message", "role": "user", "content": "one"},
                {"type": "message", "role": "assistant", "content": "two"},
            ],
        }

        proxy._prompt_cache_prefix_diagnostics(
            request_id="req-1",
            upstream_body=first_body,
            resolved_model="gpt-5.5",
        )
        diagnostics = proxy._prompt_cache_prefix_diagnostics(
            request_id="req-2",
            upstream_body=second_body,
            resolved_model="gpt-5.5",
        )

        self.assertTrue(diagnostics["previous_is_prefix"])
        self.assertTrue(diagnostics["current_extends_previous"])
        self.assertEqual(diagnostics["common_prefix_items"], 1)

    def test_responses_cache_settle_waits_for_recent_same_lineage_finish(self):
        plan = proxy.UpstreamRequestPlan(
            request_id="req-1",
            upstream_url="https://example.invalid/responses",
            headers={"x-agent-task-id": "cache-task"},
            body={"model": "gpt-5.5", "prompt_cache_key": "cache-key", "input": "hello"},
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
        )

        with mock.patch.object(proxy.time, "monotonic", return_value=100.0):
            proxy._remember_responses_cache_settle_finish(plan, 200)

        sleep = mock.AsyncMock()
        with (
            mock.patch.object(proxy.time, "monotonic", return_value=100.5),
            mock.patch.object(proxy.asyncio, "sleep", sleep),
        ):
            proxy.asyncio.run(proxy._wait_for_responses_cache_settle(plan))

        sleep.assert_awaited_once()
        self.assertAlmostEqual(sleep.await_args.args[0], 1.5)

    def test_responses_cache_settle_does_not_wait_for_other_model(self):
        finished_plan = proxy.UpstreamRequestPlan(
            request_id="req-1",
            upstream_url="https://example.invalid/responses",
            headers={"x-agent-task-id": "cache-task"},
            body={"model": "gpt-5.5", "prompt_cache_key": "cache-key", "input": "hello"},
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
        )
        other_model_plan = proxy.UpstreamRequestPlan(
            request_id="req-2",
            upstream_url="https://example.invalid/responses",
            headers={"x-agent-task-id": "cache-task"},
            body={"model": "gpt-5.4", "prompt_cache_key": "cache-key", "input": "hello"},
            usage_event=None,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
        )

        with mock.patch.object(proxy.time, "monotonic", return_value=100.0):
            proxy._remember_responses_cache_settle_finish(finished_plan, 200)

        sleep = mock.AsyncMock()
        with mock.patch.object(proxy.asyncio, "sleep", sleep):
            proxy.asyncio.run(proxy._wait_for_responses_cache_settle(other_model_plan))

        sleep.assert_not_awaited()

    def _build_responses_plan(
        self,
        *,
        request_id: str = "req-1",
        cs_id: str | None,
        subagent: bool = False,
        prompt_cache_key: str = "019dd22d-34be-7da0-aaaa-aaaaaaaaaaaa",
        upstream_url: str = "https://example.invalid/responses",
    ):
        headers = {}
        if cs_id is not None:
            headers["x-client-session-id"] = cs_id
        if subagent:
            headers["x-interaction-type"] = "conversation-subagent"
            headers["x-parent-agent-id"] = "parent-task"
            headers["x-agent-task-id"] = "child-task"
        else:
            headers["x-agent-task-id"] = "parent-task"
        return proxy.UpstreamRequestPlan(
            request_id=request_id,
            upstream_url=upstream_url,
            headers=headers,
            body={
                "model": "gpt-5.5",
                "prompt_cache_key": prompt_cache_key,
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            },
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
        )

    def test_responses_plan_role_classifies_parent_and_subagent(self):
        import request_headers as request_headers_module
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        subagent = self._build_responses_plan(
            cs_id=request_headers_module._CLIENT_SESSION_ID,
            subagent=True,
            request_id="req-2",
        )
        chat_plan = proxy.UpstreamRequestPlan(
            request_id="req-3",
            upstream_url="https://example.invalid/chat/completions",
            headers={"x-client-session-id": request_headers_module._CLIENT_SESSION_ID},
            body={"model": "gpt-5.5", "prompt_cache_key": "019dd22d-34be-7da0-aaaa-aaaaaaaaaaaa"},
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
        )
        wrong_model = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID, request_id="req-4")
        wrong_model.resolved_model = "gpt-5.4"
        wrong_model.requested_model = "gpt-5.4"
        wrong_model.body["model"] = "gpt-5.4"
        self.assertEqual(proxy._responses_plan_role(parent), "parent")
        self.assertEqual(proxy._responses_plan_role(subagent), "subagent")
        self.assertIsNone(proxy._responses_plan_role(chat_plan))
        self.assertIsNone(proxy._responses_plan_role(wrong_model))

    def test_responses_plan_task_prefix_extracts_uuid_prefix(self):
        plan = self._build_responses_plan(cs_id="x", prompt_cache_key="019dd22d-34be-7da0-aaaa-aaaaaaaaaaaa")
        self.assertEqual(proxy._responses_plan_task_prefix(plan), "parent-task")
        plan_short = self._build_responses_plan(cs_id="x", prompt_cache_key="short-key")
        plan_short.headers.pop("x-agent-task-id", None)
        self.assertIsNone(proxy._responses_plan_task_prefix(plan_short))
        plan_none = self._build_responses_plan(cs_id="x")
        plan_none.body.pop("prompt_cache_key", None)
        plan_none.headers.pop("x-agent-task-id", None)
        self.assertIsNone(proxy._responses_plan_task_prefix(plan_none))

    def test_parent_keepalive_snapshot_stored_on_parent_finish(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        parent.body["previous_response_id"] = "should-be-stripped"
        with mock.patch.dict(proxy.os.environ, {"GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "1"}, clear=False):
            proxy._remember_parent_for_keepalive(parent, 200)
        snap = proxy._PARENT_KEEPALIVE_SNAPSHOTS.get("parent-task")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["upstream_url"], "https://example.invalid/responses")
        self.assertNotIn("previous_response_id", snap["body"])
        self.assertFalse(snap["warmer_in_flight"])

    def test_parent_keepalive_snapshot_not_stored_for_subagent(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        sub = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID, subagent=True)
        with mock.patch.dict(proxy.os.environ, {"GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "1"}, clear=False):
            proxy._remember_parent_for_keepalive(sub, 200)
        self.assertNotIn("parent-task", proxy._PARENT_KEEPALIVE_SNAPSHOTS)

    def test_parent_keepalive_skips_when_parent_recent(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        sub = self._build_responses_plan(
            cs_id=request_headers_module._CLIENT_SESSION_ID,
            subagent=True,
            request_id="req-2",
        )
        env = {
            "GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "1",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_IDLE_SECONDS": "6",
        }
        with mock.patch.dict(proxy.os.environ, env, clear=False):
            proxy._remember_parent_for_keepalive(parent, 200)
            with mock.patch.object(proxy.asyncio, "get_running_loop") as get_loop:
                proxy._maybe_fire_parent_keepalive(sub, 200)
                self.assertFalse(get_loop.called)
        self.assertFalse(proxy._PARENT_KEEPALIVE_SNAPSHOTS["parent-task"]["warmer_in_flight"])

    def test_parent_keepalive_fires_when_idle_threshold_exceeded(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        sub = self._build_responses_plan(
            cs_id=request_headers_module._CLIENT_SESSION_ID,
            subagent=True,
            request_id="req-2",
        )
        env = {
            "GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "1",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_IDLE_SECONDS": "0",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_INTERVAL_SECONDS": "0",
        }
        scheduled: list = []

        class FakeLoop:
            def create_task(self, coro):
                scheduled.append(coro)
                coro.close()
                return mock.Mock()

        with mock.patch.dict(proxy.os.environ, env, clear=False):
            proxy._remember_parent_for_keepalive(parent, 200)
            with mock.patch.object(proxy.asyncio, "get_running_loop", return_value=FakeLoop()):
                proxy._maybe_fire_parent_keepalive(sub, 200)
        self.assertEqual(len(scheduled), 1)

    def test_parent_keepalive_skips_when_disabled(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        sub = self._build_responses_plan(
            cs_id=request_headers_module._CLIENT_SESSION_ID,
            subagent=True,
            request_id="req-2",
        )
        with mock.patch.dict(proxy.os.environ, {"GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "0"}, clear=False):
            proxy._remember_parent_for_keepalive(parent, 200)
            self.assertNotIn("parent-task", proxy._PARENT_KEEPALIVE_SNAPSHOTS)
            with mock.patch.object(proxy.asyncio, "get_running_loop") as get_loop:
                proxy._maybe_fire_parent_keepalive(sub, 200)
                self.assertFalse(get_loop.called)

    def test_parent_keepalive_skips_when_already_in_flight(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        sub = self._build_responses_plan(
            cs_id=request_headers_module._CLIENT_SESSION_ID,
            subagent=True,
            request_id="req-2",
        )
        env = {
            "GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "1",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_IDLE_SECONDS": "0",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_INTERVAL_SECONDS": "0",
        }
        with mock.patch.dict(proxy.os.environ, env, clear=False):
            proxy._remember_parent_for_keepalive(parent, 200)
            proxy._PARENT_KEEPALIVE_SNAPSHOTS["parent-task"]["warmer_in_flight"] = True
            with mock.patch.object(proxy.asyncio, "get_running_loop") as get_loop:
                proxy._maybe_fire_parent_keepalive(sub, 200)
                self.assertFalse(get_loop.called)

    def test_parent_keepalive_drops_stale_snapshot_past_ttl(self):
        import request_headers as request_headers_module
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        parent = self._build_responses_plan(cs_id=request_headers_module._CLIENT_SESSION_ID)
        sub = self._build_responses_plan(
            cs_id=request_headers_module._CLIENT_SESSION_ID,
            subagent=True,
            request_id="req-2",
        )
        env = {
            "GHCP_PROXY_PARENT_KEEPALIVE_ENABLED": "1",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_IDLE_SECONDS": "0",
            "GHCP_PROXY_PARENT_KEEPALIVE_MIN_INTERVAL_SECONDS": "0",
            "GHCP_PROXY_PARENT_KEEPALIVE_SNAPSHOT_TTL_SECONDS": "1",
        }
        with mock.patch.dict(proxy.os.environ, env, clear=False):
            with mock.patch.object(proxy.time, "monotonic", return_value=100.0):
                proxy._remember_parent_for_keepalive(parent, 200)
            with mock.patch.object(proxy.time, "monotonic", return_value=200.0):
                with mock.patch.object(proxy.asyncio, "get_running_loop") as get_loop:
                    proxy._maybe_fire_parent_keepalive(sub, 200)
                    self.assertFalse(get_loop.called)
        self.assertNotIn("parent-task", proxy._PARENT_KEEPALIVE_SNAPSHOTS)

    def test_fire_parent_keepalive_posts_with_capped_max_output_tokens(self):
        proxy._PARENT_KEEPALIVE_SNAPSHOTS.clear()
        proxy._PARENT_KEEPALIVE_SNAPSHOTS["019dd22d"] = {
            "upstream_url": "https://example.invalid/responses",
            "headers": {"x-client-session-id": "sess-A"},
            "body": {"model": "gpt-5.5", "prompt_cache_key": "019dd22d-34be-7da0", "input": [], "previous_response_id": "old"},
            "finished_at": proxy.time.monotonic(),
            "last_warmer_at": 0.0,
            "warmer_in_flight": True,
        }
        captured: dict = {}

        class FakePost:
            def __init__(self, status: int):
                self.status_code = status

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def post(self, url, headers=None, json=None):
                captured["url"] = url
                captured["headers"] = headers
                captured["body"] = json
                return FakePost(200)

        env = {"GHCP_PROXY_PARENT_KEEPALIVE_MAX_OUTPUT_TOKENS": "8"}
        with mock.patch.dict(proxy.os.environ, env, clear=False):
            with mock.patch.object(proxy.httpx, "AsyncClient", FakeClient):
                proxy.asyncio.run(proxy._fire_parent_keepalive("019dd22d"))
        self.assertEqual(captured["url"], "https://example.invalid/responses")
        self.assertEqual(captured["body"]["max_output_tokens"], 8)
        self.assertFalse(captured["body"]["stream"])
        self.assertNotIn("previous_response_id", captured["body"])
        self.assertFalse(proxy._PARENT_KEEPALIVE_SNAPSHOTS["019dd22d"]["warmer_in_flight"])

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
        self.assertIn("/api/config/auto-update", paths)

    def test_background_proxy_status_route_uses_manager(self):
        manager = SimpleNamespace(status_payload=mock.Mock(return_value={"startup_enabled": True}))
        with mock.patch.object(proxy, "background_proxy_manager", manager):
            response = proxy.asyncio.run(proxy.background_proxy_status_api())

        manager.status_payload.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {"startup_enabled": True})

    def test_auto_update_status_route_uses_runtime_controller(self):
        controller = SimpleNamespace(status_payload=mock.Mock(return_value={"enabled": True, "last_result": {"reason": "up-to-date"}}))
        with mock.patch.object(proxy, "auto_update_runtime_controller", controller):
            response = proxy.asyncio.run(proxy.auto_update_status_api())

        controller.status_payload.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {"enabled": True, "last_result": {"reason": "up-to-date"}})

    def test_auto_update_config_route_sets_mode(self):
        request = SimpleNamespace()
        manager = SimpleNamespace(set_mode=mock.Mock(return_value={"mode": "developer"}))
        controller = SimpleNamespace(status_payload=mock.Mock(return_value={"enabled": True, "mode": "developer"}))

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value={"action": "set_mode", "mode": "developer"})),
            mock.patch.object(proxy, "auto_update_manager", manager),
            mock.patch.object(proxy, "auto_update_runtime_controller", controller),
        ):
            response = proxy.asyncio.run(proxy.auto_update_config_api(request))

        manager.set_mode.assert_called_once_with("developer")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["mode"], "developer")

    def test_auto_update_config_route_can_override_local_changes(self):
        request = SimpleNamespace()
        result = {"attempted": True, "updated": True, "reason": "updated", "discarded_local_changes": True}
        controller = SimpleNamespace(
            apply_update=mock.AsyncMock(return_value=result),
            status_payload=mock.Mock(return_value={"enabled": True, "last_result": result}),
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value={"action": "upgrade_anyway"})),
            mock.patch.object(proxy, "auto_update_runtime_controller", controller),
        ):
            response = proxy.asyncio.run(proxy.auto_update_config_api(request))

        controller.apply_update.assert_awaited_once_with(override_local_changes=True)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertTrue(payload["result"]["discarded_local_changes"])

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

    def test_model_routing_config_route_refreshes_client_model_metadata(self):
        request = SimpleNamespace()
        payload = {"enabled": True, "mappings": [{"source_model": "gpt-5.4", "target_model": "claude-sonnet-4.6"}]}
        saved = {
            "enabled": True,
            "mappings": [{"source_model": "gpt-5.4", "target_model": "claude-sonnet-4.6"}],
            "approval_enabled": False,
            "approval_mappings": [],
            "claude_code_defaults": {"opus_model": "", "sonnet_model": "", "haiku_model": ""},
            "available_models": [],
            "path": "ignored",
        }

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=payload)),
            mock.patch.object(proxy.model_routing_config_service, "save_settings", return_value=saved) as save_settings,
            mock.patch.object(proxy.client_proxy_config_service, "refresh_client_model_metadata", return_value={"codex": True, "claude": True}) as refresh_metadata,
        ):
            response = proxy.asyncio.run(proxy.model_routing_config_api(request))

        save_settings.assert_called_once_with(payload)
        refresh_metadata.assert_called_once_with()
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
        sent_headers = post.await_args.kwargs["headers"]
        self.assertEqual(sent_headers["X-Initiator"], "user")
        self.assertIn("x-request-id", sent_headers)
        self.assertEqual(
            response.body,
            b'{"id":"resp_123","type":"message","role":"assistant","model":"gpt-5.4","content":[{"type":"text","text":"hello from codex"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}',
        )

    def test_anthropic_messages_to_responses_reuses_claude_metadata_session_affinity(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        body1 = {
            "model": "claude-opus-4.6",
            "metadata": {"user_id": '{"session_id":"claude-meta-cache-session"}'},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "stream": False,
        }
        body2 = {
            "model": "claude-opus-4.6",
            "metadata": {"user_id": '{"session_id":"claude-meta-cache-session"}'},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "follow up"}],
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
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(side_effect=[body1, body2])),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.4"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            first_response = proxy.asyncio.run(proxy.anthropic_messages(request))
            second_response = proxy.asyncio.run(proxy.anthropic_messages(request))

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        first_headers = post.await_args_list[0].kwargs["headers"]
        second_headers = post.await_args_list[1].kwargs["headers"]
        self.assertEqual(first_headers["X-Initiator"], "user")
        self.assertEqual(second_headers["X-Initiator"], "user")
        self.assertEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertNotEqual(first_headers["x-request-id"], first_headers["x-agent-task-id"])
        self.assertNotEqual(second_headers["x-request-id"], second_headers["x-agent-task-id"])
        self.assertNotEqual(first_headers["x-request-id"], second_headers["x-request-id"])
        first_body = post.await_args_list[0].kwargs["json"]
        second_body = post.await_args_list[1].kwargs["json"]
        self.assertNotIn("prompt_cache_key", first_body)
        self.assertNotIn("prompt_cache_key", second_body)
        self.assertNotIn("metadata", first_body)

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

    def test_responses_route_uses_body_cache_affinity_fields_locally(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "sessionId": "session-123",
            "prompt_cache_key": "cache-123",
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
        self.assertIn("x-client-session-id", forwarded_headers)
        self.assertIn("x-interaction-id", forwarded_headers)
        self.assertIn("x-agent-task-id", forwarded_headers)
        self.assertIn("x-request-id", forwarded_headers)
        self.assertNotEqual(forwarded_headers["x-request-id"], forwarded_headers["x-agent-task-id"])
        self.assertNotIn("session_id", forwarded_headers)
        self.assertNotIn("x-client-request-id", forwarded_headers)
        self.assertNotIn("sessionId", forwarded_body)
        self.assertNotIn("prompt_cache_key", forwarded_body)
        self.assertNotIn("previous_response_id", forwarded_body)
        self.assertEqual(forwarded_body["tools"], body["tools"])
        self.assertEqual(forwarded_body["include"], body["include"])
        self.assertTrue(forwarded_body["parallel_tool_calls"])
        self.assertEqual(forwarded_body["tool_choice"], "auto")
        self.assertFalse(forwarded_body["stream"])
        self.assertEqual(response.status_code, 200)
        response_payload = json.loads(response.body)
        self.assertEqual(response_payload["id"], "resp_123")
        self.assertEqual(response_payload["output"][0]["content"][0]["text"], "done")

    def test_responses_route_session_affinity_is_deterministic_across_calls(self):
        body_template = {
            "model": "gpt-5.4",
            "sessionId": "session-123",
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
            },
            headers={"content-type": "application/json"},
        )
        forwarded_headers = []

        for index in range(2):
            request = SimpleNamespace(
                url=SimpleNamespace(path="/v1/responses"),
                method="POST",
                headers={},
            )
            body = json.loads(json.dumps(body_template))
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

            self.assertEqual(response.status_code, 200)
            forwarded_headers.append(post.await_args.kwargs["headers"])

        self.assertEqual(forwarded_headers[0]["x-interaction-id"], forwarded_headers[1]["x-interaction-id"])
        self.assertEqual(forwarded_headers[0]["x-agent-task-id"], forwarded_headers[1]["x-agent-task-id"])

    def test_responses_route_cache_key_affinity_is_stable_across_user_turns(self):
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )
        forwarded_headers = []

        for text in ("first", "follow up"):
            request = SimpleNamespace(
                url=SimpleNamespace(path="/v1/responses"),
                method="POST",
                headers={},
            )
            body = {
                "model": "gpt-5.5",
                "prompt_cache_key": "cache-user-turns",
                "stream": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                ],
            }
            with (
                mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
                mock.patch.object(auth, "get_api_key", return_value="test-key"),
                mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
                mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
                mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
                mock.patch.object(proxy.usage_tracker, "finish_event"),
                mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
            ):
                response = proxy.asyncio.run(proxy.responses(request))

            self.assertEqual(response.status_code, 200)
            forwarded_headers.append(post.await_args.kwargs["headers"])

        self.assertEqual(forwarded_headers[0]["x-interaction-id"], forwarded_headers[1]["x-interaction-id"])
        self.assertEqual(forwarded_headers[0]["x-agent-task-id"], forwarded_headers[1]["x-agent-task-id"])
        self.assertNotEqual(forwarded_headers[0]["x-request-id"], forwarded_headers[0]["x-agent-task-id"])
        self.assertNotEqual(forwarded_headers[0]["x-request-id"], forwarded_headers[1]["x-request-id"])

    def test_responses_route_returns_encrypted_reasoning_to_codex(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "include": ["reasoning.encrypted_content"],
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        }
        upstream_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [{"type": "summary_text", "text": "thought"}],
                    "encrypted_content": "fresh-upstream-ciphertext",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
            "output_text": "done",
            "usage": {
                "input_tokens": 24,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 5,
                "total_tokens": 29,
            },
        }
        upstream = httpx.Response(
            200,
            json=upstream_payload,
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(post.await_args.kwargs["json"]["include"], ["reasoning.encrypted_content"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), upstream_payload)

    def test_responses_route_returns_tripwire_message_for_large_uncached_request(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": "hello",
        }
        upstream_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "expensive answer"}],
                }
            ],
            "output_text": "expensive answer",
            "usage": {
                "input_tokens": 60000,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "total_tokens": 60005,
            },
        }
        upstream = httpx.Response(
            200,
            json=upstream_payload,
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertIn("Safety tripwire activated", payload["output_text"])
        self.assertIn("could burn through your Copilot session limit", payload["output_text"])
        self.assertIn("0 cached tokens", payload["output_text"])
        self.assertIn("60000 fresh input tokens", payload["output_text"])
        self.assertIn("30000 cached input tokens", payload["output_text"])
        self.assertIn("50000 token fresh/cache gap limit", payload["output_text"])
        self.assertIn("disable the tripwire in Settings", payload["output_text"])
        self.assertIn("debug the cache lineage", payload["output_text"])
        self.assertNotIn("expensive answer", payload["output_text"])
        finish_usage.assert_called_once()
        self.assertEqual(finish_usage.call_args.kwargs["usage"]["input_tokens"], 60000)
        self.assertEqual(finish_usage.call_args.kwargs["usage"]["cached_input_tokens"], 0)

    def test_responses_route_does_not_tripwire_when_cache_is_above_threshold(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": "hello",
        }
        upstream_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "cache acceptable"}],
                }
            ],
            "output_text": "cache acceptable",
            "usage": {
                "input_tokens": 107092,
                "input_tokens_details": {"cached_tokens": 90496},
                "output_tokens": 5,
                "total_tokens": 107097,
            },
        }
        upstream = httpx.Response(
            200,
            json=upstream_payload,
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["output_text"], "cache acceptable")
        self.assertNotIn("Safety tripwire activated", payload["output_text"])
        finish_usage.assert_called_once()

    def test_responses_route_does_not_tripwire_when_low_cache_gap_is_below_threshold(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": "hello",
        }
        upstream_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "low cache but tolerable"}],
                }
            ],
            "output_text": "low cache but tolerable",
            "usage": {
                "input_tokens": 36000,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "total_tokens": 36005,
            },
        }
        upstream = httpx.Response(
            200,
            json=upstream_payload,
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["output_text"], "low cache but tolerable")
        self.assertNotIn("Safety tripwire activated", payload["output_text"])
        finish_usage.assert_called_once()

    def test_responses_route_does_not_tripwire_when_disabled_in_settings(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": "hello",
        }
        upstream_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "tripwire disabled"}],
                }
            ],
            "output_text": "tripwire disabled",
            "usage": {
                "input_tokens": 60000,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "total_tokens": 60005,
            },
        }
        upstream = httpx.Response(
            200,
            json=upstream_payload,
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(
                proxy.client_proxy_config_service,
                "load_client_proxy_settings",
                return_value={"token_tripwire_enabled": False},
            ),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["output_text"], "tripwire disabled")
        self.assertNotIn("Safety tripwire activated", payload["output_text"])
        finish_usage.assert_called_once()

    def test_responses_route_does_not_tripwire_when_cache_is_above_threshold_even_with_high_fresh_tokens(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": "hello",
        }
        upstream_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "large but cached enough"}],
                }
            ],
            "output_text": "large but cached enough",
            "usage": {
                "input_tokens": 107092,
                "input_tokens_details": {"cached_tokens": 40000},
                "output_tokens": 5,
                "total_tokens": 107097,
            },
        }
        upstream = httpx.Response(
            200,
            json=upstream_payload,
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["output_text"], "large but cached enough")
        self.assertNotIn("Safety tripwire activated", payload["output_text"])
        finish_usage.assert_called_once()

    def test_responses_route_drops_reasoning_for_forked_context(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "parent developer"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "fork developer"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_parent",
                    "summary": [{"type": "summary_text", "text": "summary survives"}],
                    "encrypted_content": "foreign-parent-ciphertext",
                },
                {
                    "type": "reasoning",
                    "id": "rs_empty",
                    "encrypted_content": "ciphertext-only",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "subtask"}],
                },
            ],
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_input = post.await_args.kwargs["json"]["input"]
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("encrypted_content", json.dumps(forwarded_input))
        self.assertEqual([item for item in forwarded_input if item.get("type") == "reasoning"], [])

    def test_responses_route_strips_encrypted_reasoning_for_tool_history_without_lineage(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "spawn helpers"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_keep_summary",
                    "summary": [{"type": "summary_text", "text": "summary survives"}],
                    "encrypted_content": "old-ciphertext",
                },
                {
                    "type": "reasoning",
                    "id": "rs_drop_empty",
                    "encrypted_content": "ciphertext-only",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "agent id",
                },
            ],
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_input = post.await_args.kwargs["json"]["input"]
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("encrypted_content", json.dumps(forwarded_input))
        self.assertEqual([item for item in forwarded_input if item.get("type") == "reasoning"], [])
        request_started = next(
            call.args[0]
            for call in append_trace.call_args_list
            if call.args and call.args[0].get("event") == "request_started"
        )
        sanitization = request_started["trace"]["responses_input_sanitization"]
        self.assertEqual(sanitization["encrypted_content_preservation"], "disabled")
        self.assertEqual(sanitization["encrypted_content_strip_reason"], "tool_history_without_cache_lineage")
        self.assertEqual(sanitization["encrypted_content_items_dropped"], 2)
        self.assertTrue(sanitization["reasoning_items_dropped"])

    def test_responses_route_preserves_encrypted_reasoning_for_tool_history_with_cache_lineage(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "prompt_cache_key": "session-cache",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "spawn helpers"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_keep_summary",
                    "summary": [{"type": "summary_text", "text": "summary survives"}],
                    "encrypted_content": "old-ciphertext",
                },
                {
                    "type": "reasoning",
                    "id": "rs_keep_empty",
                    "encrypted_content": "ciphertext-only",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "agent id",
                },
            ],
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded = post.await_args.kwargs["json"]
        forwarded_input = forwarded["input"]
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("prompt_cache_key", forwarded)
        self.assertEqual(proxy._responses_input_encrypted_content_count(forwarded_input), 2)
        self.assertEqual([item.get("type") for item in forwarded_input], [
            "message",
            "message",
            "reasoning",
            "reasoning",
            "function_call",
            "function_call_output",
        ])

        request_started = next(
            call.args[0]
            for call in append_trace.call_args_list
            if call.args and call.args[0].get("event") == "request_started"
        )
        sanitization = request_started["trace"].get("responses_input_sanitization")
        self.assertEqual(sanitization["encrypted_content_preservation"], "preserved")
        self.assertIsNone(sanitization["encrypted_content_strip_reason"])
        self.assertEqual(sanitization["encrypted_content_items_dropped"], 0)
        self.assertFalse(sanitization["reasoning_items_dropped"])

    def test_responses_route_preserves_encrypted_reasoning_for_multiple_developers_with_cache_lineage(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "prompt_cache_key": "session-cache",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "base developer"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "summary 1"}],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "repo instructions"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "encrypted_content": "ciphertext-2",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "latest turn"}],
                },
            ],
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_input = post.await_args.kwargs["json"]["input"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(proxy._responses_input_encrypted_content_count(forwarded_input), 2)

        request_started = next(
            call.args[0]
            for call in append_trace.call_args_list
            if call.args and call.args[0].get("event") == "request_started"
        )
        sanitization = request_started["trace"].get("responses_input_sanitization")
        self.assertEqual(sanitization["encrypted_content_preservation"], "preserved")
        self.assertIsNone(sanitization["encrypted_content_strip_reason"])
        self.assertEqual(sanitization["encrypted_content_items_dropped"], 0)
        self.assertFalse(sanitization["reasoning_items_dropped"])

    def test_responses_route_preserves_all_encrypted_reasoning_without_size_trim(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        input_items = []
        for index in range(10):
            input_items.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{index}",
                    "summary": [{"type": "summary_text", "text": f"summary {index}"}],
                    "encrypted_content": f"ciphertext-{index}",
                }
            )
        for index in range(40):
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"turn {index}"}],
                }
            )
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "input": input_items,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_input = post.await_args.kwargs["json"]["input"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(proxy._responses_input_encrypted_content_count(forwarded_input), 10)

        request_started = next(
            call.args[0]
            for call in append_trace.call_args_list
            if call.args and call.args[0].get("event") == "request_started"
        )
        sanitization = request_started["trace"]["responses_input_sanitization"]
        self.assertEqual(sanitization["input_items_before"], 50)
        self.assertEqual(sanitization["input_items_after"], 50)
        self.assertEqual(sanitization["encrypted_content_items_before"], 10)
        self.assertEqual(sanitization["encrypted_content_items_after"], 10)
        self.assertEqual(sanitization["encrypted_content_items_dropped"], 0)
        self.assertEqual(sanitization["encrypted_content_preservation"], "preserved")
        self.assertIsNone(sanitization["encrypted_keep_last"])

    def test_responses_route_does_not_trim_encrypted_reasoning_with_cache_lineage(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        input_items = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "start"}],
            },
        ]
        for index in range(8):
            input_items.extend(
                [
                    {
                        "type": "reasoning",
                        "id": f"rs_{index}",
                        "summary": [{"type": "summary_text", "text": f"summary {index}"}],
                        "encrypted_content": f"ciphertext-{index}",
                    },
                    {
                        "type": "function_call",
                        "call_id": f"call_{index}",
                        "name": "exec_command",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": f"call_{index}",
                        "output": "ok",
                    },
                ]
            )
        for index in range(23):
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"turn {index}"}],
                }
            )
        self.assertGreaterEqual(len(input_items), 48)
        body = {
            "model": "gpt-5.5",
            "stream": False,
            "prompt_cache_key": "session-cache",
            "input": input_items,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="gpt-5.5"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        forwarded_input = post.await_args.kwargs["json"]["input"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(proxy._responses_input_encrypted_content_count(forwarded_input), 8)

        request_started = next(
            call.args[0]
            for call in append_trace.call_args_list
            if call.args and call.args[0].get("event") == "request_started"
        )
        sanitization = request_started["trace"]["responses_input_sanitization"]
        self.assertEqual(sanitization["input_items_before"], len(input_items))
        self.assertEqual(sanitization["input_items_after"], len(input_items))
        self.assertEqual(sanitization["encrypted_content_items_before"], 8)
        self.assertEqual(sanitization["encrypted_content_items_after"], 8)
        self.assertEqual(sanitization["encrypted_content_items_dropped"], 0)
        self.assertEqual(sanitization["encrypted_content_preservation"], "preserved")
        self.assertIsNone(sanitization["encrypted_keep_last"])

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

    def test_responses_compact_uses_body_cache_affinity_fields_locally(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses/compact"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "sessionId": "session-123",
            "prompt_cache_key": "cache-123",
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
        self.assertIn("x-client-session-id", forwarded_headers)
        self.assertIn("x-interaction-id", forwarded_headers)
        self.assertIn("x-agent-task-id", forwarded_headers)
        self.assertIn("x-request-id", forwarded_headers)
        self.assertNotEqual(forwarded_headers["x-request-id"], forwarded_headers["x-agent-task-id"])
        self.assertNotIn("session_id", forwarded_headers)
        self.assertNotIn("x-client-request-id", forwarded_headers)
        self.assertNotIn("sessionId", forwarded_body)
        self.assertNotIn("prompt_cache_key", forwarded_body)
        self.assertNotIn("previous_response_id", forwarded_body)
        self.assertEqual(forwarded_body["tools"], body["tools"])
        self.assertEqual(forwarded_body["include"], body["include"])
        self.assertTrue(forwarded_body["parallel_tool_calls"])
        self.assertEqual(forwarded_body["tool_choice"], "auto")
        self.assertFalse(forwarded_body["stream"])
        self.assertEqual(response.status_code, 200)
        response_payload = json.loads(response.body)
        self.assertEqual(response_payload["id"], "resp_123")
        self.assertEqual(response_payload["output"][0]["content"][0]["text"], "summary")

    def test_responses_compact_preserves_deferred_tools(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses/compact"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "stream": False,
            "tools": [
                {
                    "type": "namespace",
                    "name": "codex_app",
                    "tools": [
                        {
                            "type": "function",
                            "name": "automation_update",
                            "defer_loading": True,
                        }
                    ],
                }
            ],
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
                "usage": {"input_tokens": 24, "output_tokens": 5, "total_tokens": 29},
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
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.responses_compact(request))

        forwarded_body = post.await_args.kwargs["json"]
        self.assertTrue(forwarded_body["tools"][0]["tools"][0]["defer_loading"])
        self.assertEqual(response.status_code, 200)

        request_started = next(
            call.args[0]
            for call in append_trace.call_args_list
            if call.args and call.args[0].get("event") == "request_started"
        )
        self.assertEqual(request_started["request_body"]["deferred_tool_count"], 1)
        self.assertEqual(request_started["upstream_body"]["deferred_tool_count"], 1)
        diagnostics = request_started["trace"].get("sanitizer_diagnostics", [])
        self.assertEqual(diagnostics, [])

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

    def test_proxy_streaming_response_retries_without_prompt_cache_retention_when_rejected(self):
        class RecordingClient:
            def __init__(self):
                self.bodies = []
                self.aclose = mock.AsyncMock()

            def build_request(self, method, url, headers=None, json=None):
                self.bodies.append(json)
                return httpx.Request(method, url, headers=headers, json=json)

        first_client = RecordingClient()
        retry_client = RecordingClient()
        rejected = httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unknown parameter: prompt_cache_retention",
                    "param": "prompt_cache_retention",
                }
            },
            headers={"content-type": "application/json"},
        )
        accepted = httpx.Response(
            200,
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
        trace_plan = proxy.UpstreamRequestPlan(
            request_id="req-retention-stream-retry",
            upstream_url="https://example.invalid/responses",
            headers={"Authorization": "Bearer test"},
            body={
                "model": "gpt-5.4",
                "input": "hi",
                "stream": True,
                "prompt_cache_key": "cache-123",
                "prompt_cache_retention": "24h",
            },
            usage_event=None,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            trace_context={},
        )

        async def run_stream():
            with (
                mock.patch.object(proxy.httpx, "AsyncClient", side_effect=[first_client, retry_client]),
                mock.patch.object(
                    proxy,
                    "throttled_client_send",
                    mock.AsyncMock(side_effect=[rejected, accepted]),
                ) as send,
                mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            ):
                response = await proxy.proxy_streaming_response(
                    "https://example.invalid/responses",
                    {"Authorization": "Bearer test"},
                    trace_plan.body,
                    stream_type="responses",
                    trace_plan=trace_plan,
                )
                body_bytes = b""
                async for chunk in response.body_iterator:
                    body_bytes += chunk
                return response, body_bytes, send, finish_usage

        response, body, send, finish_usage = proxy.asyncio.run(run_stream())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"[DONE]", body)
        self.assertEqual(send.await_count, 2)
        self.assertEqual(first_client.bodies[0]["prompt_cache_retention"], "24h")
        self.assertNotIn("prompt_cache_retention", retry_client.bodies[0])
        self.assertEqual(retry_client.bodies[0]["prompt_cache_key"], "cache-123")
        self.assertEqual(
            trace_plan.trace_context["prompt_cache_retention_retry"],
            {"action": "drop_unsupported_field", "field": "prompt_cache_retention"},
        )
        first_client.aclose.assert_awaited_once()
        retry_client.aclose.assert_awaited_once()
        finish_usage.assert_called_once()

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


    def test_responses_route_returns_friendly_message_for_session_limit_429(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={"x-client-request-id": "req-limit-1"},
        )
        body = {"model": "gpt-5.4", "input": "hi", "stream": False}
        plan = _BridgeExecutionPlan_for_msgs(
            strategy_name="responses_to_responses",
            inbound_protocol="responses",
            caller_protocol="responses",
            upstream_protocol="responses",
            header_kind="responses",
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            upstream_body=body,
            stream=False,
            is_compact=False,
        )
        upstream = httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
            headers={
                "content-type": "application/json",
                "x-usage-ratelimit-session": "rem=0&rst=2026-04-25T20%3A00%3A00Z",
            },
        )

        async def fake_plan(*args, **kwargs):
            return plan

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.bridge_planner, "plan", side_effect=fake_plan),
            mock.patch.object(proxy.util, "utc_now", return_value=datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "completed")
        self.assertIn("5h usage limit", payload["output_text"])
        self.assertIn("2 hours", payload["output_text"])
        finish_usage.assert_called_once()
        self.assertEqual(finish_usage.call_args.args[1], 429)
        self.assertEqual(finish_usage.call_args.kwargs["response_text"], payload["output_text"])

    def test_responses_route_passes_through_unclassified_429(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={"x-client-request-id": "req-limit-2"},
        )
        body = {"model": "gpt-5.4", "input": "hi", "stream": False}
        plan = _BridgeExecutionPlan_for_msgs(
            strategy_name="responses_to_responses",
            inbound_protocol="responses",
            caller_protocol="responses",
            upstream_protocol="responses",
            header_kind="responses",
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            upstream_body=body,
            stream=False,
            is_compact=False,
        )
        upstream = httpx.Response(
            429,
            json={"error": {"message": "generic rate limited"}},
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
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 429)
        self.assertEqual(json.loads(response.body)["error"]["message"], "generic rate limited")

    def test_non_streaming_request_retries_without_prompt_cache_retention_when_rejected(self):
        rejected = httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unknown parameter: prompt_cache_retention",
                    "param": "prompt_cache_retention",
                }
            },
            headers={"content-type": "application/json"},
        )
        accepted = httpx.Response(
            200,
            json={"id": "resp_123", "output": [], "usage": {"input_tokens": 10, "output_tokens": 1}},
            headers={"content-type": "application/json"},
        )
        plan = proxy.UpstreamRequestPlan(
            request_id="req-retention-retry",
            upstream_url="https://example.invalid/responses",
            headers={"Authorization": "Bearer test"},
            body={
                "model": "gpt-5.4",
                "input": "hi",
                "prompt_cache_key": "cache-123",
                "prompt_cache_retention": "24h",
            },
            usage_event=None,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            trace_context={},
        )

        with (
            mock.patch.object(
                proxy,
                "throttled_client_post",
                mock.AsyncMock(side_effect=[rejected, accepted]),
            ) as post,
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
        ):
            response = proxy.asyncio.run(
                proxy._post_non_streaming_request(
                    plan,
                    error_response=format_translation.openai_error_response,
                )
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.await_count, 2)
        first_body = post.await_args_list[0].kwargs["json"]
        retry_body = post.await_args_list[1].kwargs["json"]
        self.assertEqual(first_body["prompt_cache_retention"], "24h")
        self.assertNotIn("prompt_cache_retention", retry_body)
        self.assertEqual(retry_body["prompt_cache_key"], "cache-123")
        self.assertEqual(
            plan.trace_context["prompt_cache_retention_retry"],
            {"action": "drop_unsupported_field", "field": "prompt_cache_retention"},
        )
        finish_usage.assert_called_once()

    def test_chat_streaming_returns_friendly_message_for_weekly_limit_429(self):
        upstream = httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
            headers={
                "content-type": "application/json",
                "x-usage-ratelimit-weekly": "rem=0&rst=2026-04-27T00%3A00%3A00Z",
            },
        )

        class FakeClient:
            def build_request(self, *args, **kwargs):
                return httpx.Request("POST", "https://example.invalid/chat/completions")

            async def aclose(self):
                return None

        trace_plan = proxy.UpstreamRequestPlan(
            request_id="req-chat-limit",
            upstream_url="https://example.invalid/chat/completions",
            headers={},
            body={"stream": True},
            usage_event=None,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
        )

        async def run_stream():
            with (
                mock.patch.object(proxy.httpx, "AsyncClient", return_value=FakeClient()),
                mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=upstream)),
                mock.patch.object(proxy.util, "utc_now", return_value=datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)),
                mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            ):
                response = await proxy.proxy_streaming_response(
                    "https://example.invalid/chat/completions",
                    {},
                    {"stream": True},
                    stream_type="chat",
                    trace_plan=trace_plan,
                )
                body_bytes = b""
                async for chunk in response.body_iterator:
                    body_bytes += chunk
                return response, body_bytes.decode("utf-8"), finish_usage

        response, body, finish_usage = proxy.asyncio.run(run_stream())
        self.assertEqual(response.status_code, 200)
        self.assertIn("weekly usage limit", body)
        self.assertIn("The weekly limit resets", body)
        self.assertIn("1 day 6 hours", body)
        self.assertIn("[DONE]", body)
        finish_usage.assert_called_once()
        self.assertEqual(finish_usage.call_args.args[1], 429)

    def test_responses_streaming_returns_tripwire_message_for_large_uncached_request(self):
        chunks = [
            (
                'event: response.created\n'
                'data: {"type":"response.created","response":{"id":"resp_123","status":"in_progress"}}\n\n'
            ).encode("utf-8"),
            (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"expensive answer"}\n\n'
            ).encode("utf-8"),
            (
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"id":"resp_123","object":"response",'
                '"status":"completed","output":[],"usage":{"input_tokens":60000,'
                '"input_tokens_details":{"cached_tokens":0},"output_tokens":5,"total_tokens":60005}}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
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
                return httpx.Request("POST", "https://example.invalid/responses")

            async def aclose(self):
                return None

        usage_event = {"request_id": "req-tripwire-stream"}
        trace_plan = proxy.UpstreamRequestPlan(
            request_id="req-tripwire-stream",
            upstream_url="https://example.invalid/responses",
            headers={},
            body={"stream": True},
            usage_event=usage_event,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
        )

        async def run_stream():
            with (
                mock.patch.object(proxy.httpx, "AsyncClient", return_value=FakeClient()),
                mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=FakeUpstream())),
                mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            ):
                response = await proxy.proxy_streaming_response(
                    "https://example.invalid/responses",
                    {},
                    {"stream": True},
                    stream_type="responses",
                    usage_event=usage_event,
                    trace_plan=trace_plan,
                )
                body_bytes = b""
                async for chunk in response.body_iterator:
                    body_bytes += chunk
                return response, body_bytes.decode("utf-8"), finish_usage

        response, body, finish_usage = proxy.asyncio.run(run_stream())
        self.assertEqual(response.status_code, 200)
        self.assertIn("Safety tripwire activated", body)
        self.assertIn("could burn through your Copilot session limit", body)
        self.assertIn("disable the tripwire in Settings", body)
        self.assertIn("[DONE]", body)
        self.assertNotIn("expensive answer", body)
        finish_usage.assert_called_once()
        usage = finish_usage.call_args.kwargs["usage"]
        self.assertEqual(usage["input_tokens"], 60000)
        self.assertEqual(usage["cached_input_tokens"], 0)

    def test_messages_route_forwards_claude_session_affinity_for_cache(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={
                "x-claude-code-session-id": "claude-session-1",
                "anthropic-beta": "context-management-2025-06-27",
            },
        )
        body = {
            "model": "claude-sonnet-4.6",
            "metadata": {"user_id": '{"session_id":"claude-session-1"}'},
            "system": "system prompt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "stream": False,
        }
        upstream = httpx.Response(
            200,
            json={
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4.6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 42,
                    "cache_read_input_tokens": 100,
                },
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(proxy.model_routing_config_service, "resolve_target_model", return_value="claude-sonnet-4.6"),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "_persist_event"),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        self.assertEqual(response.status_code, 200)
        sent_headers = post.await_args.kwargs["headers"]
        self.assertEqual(sent_headers["x-interaction-id"], "claude-session-1")
        self.assertIn("x-agent-task-id", sent_headers)
        self.assertIn("x-request-id", sent_headers)
        self.assertEqual(sent_headers["x-agent-task-id"], sent_headers["x-request-id"])
        self.assertNotEqual(sent_headers["x-agent-task-id"], sent_headers["x-interaction-id"])
        sent_body = post.await_args.kwargs["json"]
        self.assertEqual(sent_body["system"], "system prompt")
        self.assertEqual(sent_body["messages"][-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
        payload = json.loads(response.body)
        self.assertEqual(payload["usage"]["input_tokens"], 47)
        self.assertEqual(payload["usage"]["cache_creation_input_tokens"], 42)
        self.assertEqual(payload["usage"]["cache_read_input_tokens"], 100)

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
        self.assertIn(b'"input_tokens":1242', body)
        self.assertIn(b'"cache_read_input_tokens":125000', body)
        self.assertIn(b'"cache_creation_input_tokens":42', body)
        self.assertNotIn(b'"input_tokens":1200', body)
        finish_usage.assert_called_once()
        usage = finish_usage.call_args.kwargs["usage"]
        self.assertEqual(usage["input_tokens"], 1242)
        self.assertEqual(usage["output_tokens"], 321)
        self.assertEqual(usage["cached_input_tokens"], 125000)
        self.assertEqual(usage["cache_creation_input_tokens"], 42)
        self.assertEqual(usage["pricing_fresh_input_tokens"], 1200)
        self.assertEqual(usage["pricing_cached_input_tokens"], 125000)
        self.assertEqual(usage["pricing_cache_creation_input_tokens"], 42)

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

    def test_responses_route_to_messages_tracks_anthropic_cache_usage(self):
        """Codex (responses) requests routed to Anthropic Messages upstream
        should record the Anthropic-shape pricing buckets (cache writes/reads
        and fresh input). Without this, the Responses-shape usage from the
        translated payload would lose cache_creation_input_tokens entirely
        and miscompute fresh_input_tokens."""
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={"x-client-request-id": "req-cache-1"},
        )
        body = {"model": "claude-sonnet-4.6", "input": "hi", "stream": False}
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
                "id": "msg_z",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4.6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 1200,
                    "output_tokens": 321,
                    "cache_read_input_tokens": 125000,
                    "cache_creation_input_tokens": 42,
                },
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
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        # Codex/Responses-facing usage must remain Responses-shaped: gross
        # input includes cache reads, while cached input is separately reported
        # in input_tokens_details. This is the inverse of copilot-api's
        # Responses->Anthropic subtraction.
        self.assertEqual(payload["usage"]["input_tokens"], 126200)
        self.assertEqual(payload["usage"]["output_tokens"], 321)
        self.assertEqual(payload["usage"]["total_tokens"], 126521)
        self.assertEqual(
            payload["usage"]["input_tokens_details"],
            {"cached_tokens": 125000, "cache_creation_input_tokens": 42},
        )
        finish_usage.assert_called_once()
        usage = finish_usage.call_args.kwargs["usage"]
        # Anthropic-shape cache writes survive into tracking shape and are
        # rolled into input_tokens (so the dashboard's input bucket reflects
        # cache writes); cache reads remain a separate bucket.
        self.assertEqual(usage["input_tokens"], 1242)
        self.assertEqual(usage["output_tokens"], 321)
        self.assertEqual(usage["cached_input_tokens"], 125000)
        self.assertEqual(usage["cache_read_input_tokens"], 125000)
        self.assertEqual(usage["cache_creation_input_tokens"], 42)
        self.assertEqual(usage["pricing_fresh_input_tokens"], 1200)
        self.assertEqual(usage["pricing_cached_input_tokens"], 125000)
        self.assertEqual(usage["pricing_cache_creation_input_tokens"], 42)

    def test_responses_from_anthropic_stream_tracks_anthropic_cache_usage(self):
        """Streaming responses-from-anthropic translator must surface raw
        Anthropic cache_read/cache_creation/input_tokens to the proxy so
        finish_event records the pricing buckets correctly. Without this,
        the responses-shape usage on response.completed would lose the
        cache_creation bucket and miscompute fresh_input_tokens."""
        chunks = [
            (
                'event: message_start\n'
                'data: {"type":"message_start","message":{"id":"msg_x","type":"message",'
                '"role":"assistant","model":"claude-sonnet-4.6","usage":{"input_tokens":1200,'
                '"output_tokens":0,"cache_read_input_tokens":125000,'
                '"cache_creation_input_tokens":42}}}\n\n'
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

        usage_event = {"request_id": "req-resp-stream-cache"}
        trace_plan = proxy.UpstreamRequestPlan(
            request_id="req-resp-stream-cache",
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
                response = await proxy.proxy_responses_from_anthropic_streaming_response(
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

        self.assertIn(b'"input_tokens":126200', body)
        self.assertIn(b'"cached_tokens":125000', body)
        self.assertIn(b'"cache_creation_input_tokens":42', body)
        finish_usage.assert_called_once()
        usage = finish_usage.call_args.kwargs["usage"]
        self.assertEqual(usage["input_tokens"], 1242)
        self.assertEqual(usage["output_tokens"], 321)
        self.assertEqual(usage["cached_input_tokens"], 125000)
        self.assertEqual(usage["cache_creation_input_tokens"], 42)
        self.assertEqual(usage["pricing_fresh_input_tokens"], 1200)
        self.assertEqual(usage["pricing_cached_input_tokens"], 125000)
        self.assertEqual(usage["pricing_cache_creation_input_tokens"], 42)


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

    def test_prepare_upstream_request_preserves_responses_affinity_request_id_after_tracking(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        plan, error = proxy._prepare_upstream_request(
            request,
            body={"model": "gpt-5.5", "input": "hello"},
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
            upstream_path="/responses",
            upstream_url="https://example.invalid/responses",
            header_builder=lambda _api_key, _request_id: {
                "X-Initiator": "user",
                "x-interaction-type": "conversation-user",
                "x-agent-task-id": "stable-task",
                "x-request-id": "stable-task",
            },
            error_response=format_translation.openai_error_response,
            api_key="test-key",
        )

        self.assertIsNone(error)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.usage_event["initiator"], "user")
        self.assertEqual(plan.headers["X-Initiator"], "user")
        self.assertEqual(plan.headers["x-interaction-type"], "conversation-user")
        self.assertEqual(plan.headers["x-agent-task-id"], "stable-task")
        self.assertEqual(plan.headers["x-request-id"], "stable-task")
        self.assertNotIn("x-github-request-id", plan.headers)


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
