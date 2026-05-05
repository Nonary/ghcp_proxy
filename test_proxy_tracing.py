import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import httpx

import proxy
import trace_prompt_security


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

    def test_write_request_trace_line_recreates_deleted_trace_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "deleted", "request-trace.jsonl")

            proxy._write_request_trace_line(trace_path, '{"event":"request_started"}\n')

            with open(trace_path, encoding="utf-8") as f:
                self.assertEqual(f.readline(), '{"event":"request_started"}\n')

    def test_append_request_trace_schedules_file_write(self):
        class RecordingExecutor:
            def __init__(self):
                self.calls = []

            def submit(self, *args):
                self.calls.append(args)

        executor = RecordingExecutor()
        with (
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy, "request_trace_log_path", return_value="trace.jsonl"),
            mock.patch.object(proxy, "_get_request_trace_executor", return_value=executor),
        ):
            proxy._append_request_trace({"event": "request_started"})

        self.assertEqual(len(executor.calls), 1)
        self.assertIs(executor.calls[0][0], proxy._write_request_trace_line)
        self.assertEqual(executor.calls[0][1], "trace.jsonl")
        self.assertEqual(executor.calls[0][2], '{"event":"request_started"}\n')

    def test_prompt_cache_affinity_diagnostics_reports_header_drift(self):
        body = {"model": "gpt-5.5", "input": "hi", "prompt_cache_key": "cache-123"}
        with proxy._PROMPT_CACHE_AFFINITY_TRACE_LOCK:
            proxy._PROMPT_CACHE_LAST_AFFINITY_BY_LINEAGE.clear()

        first = proxy._prompt_cache_affinity_diagnostics(
            request_id="req_1",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers={
                "x-agent-task-id": "task-a",
                "x-interaction-id": "interaction-a",
                "x-client-session-id": "client-a",
            },
        )
        second = proxy._prompt_cache_affinity_diagnostics(
            request_id="req_2",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers={
                "x-agent-task-id": "task-a",
                "x-interaction-id": "interaction-a",
                "x-client-session-id": "client-a",
            },
        )
        drift = proxy._prompt_cache_affinity_diagnostics(
            request_id="req_3",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers={
                "x-agent-task-id": "task-b",
                "x-interaction-id": "interaction-b",
                "x-client-session-id": "client-a",
            },
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(drift["previous_request_id"], "req_2")
        self.assertEqual(drift["changed_fields"], ["x_agent_task_id", "x_interaction_id"])
        self.assertEqual(drift["previous"]["x_agent_task_id"], "task-a")
        self.assertEqual(drift["current"]["x_agent_task_id"], "task-b")

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

    def test_finish_usage_and_trace_encrypts_in_flight_plaintext_prompt_after_toggle(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        plan = proxy.UpstreamRequestPlan(
            request_id="req_inflight",
            upstream_url="https://example.invalid/responses",
            headers={"X-Initiator": "user"},
            body={"model": "gpt-5.4"},
            usage_event={"request_id": "req_inflight", "request_prompt": {"user": "plain"}},
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            source_body={"model": "gpt-5.4"},
            trace_context={"request_id": "req_inflight", "request_prompt": {"user": "plain"}},
        )

        captured_event = {}

        def _capture_finish(event, _status, **_kwargs):
            captured_event.update(event)

        with (
            mock.patch.object(proxy.client_proxy_config_service, "load_client_proxy_settings", return_value={"trace_prompt_logging_enabled": True}),
            mock.patch.object(proxy.usage_tracker, "finish_event", side_effect=_capture_finish),
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
        ):
            proxy._set_trace_prompt_active_key(key, salt)
            try:
                proxy._finish_usage_and_trace(plan, 200, response_payload={"id": "resp_1"})
            finally:
                proxy._set_trace_prompt_active_key(None, None)

        self.assertTrue(trace_prompt_security.is_encrypted_payload(captured_event["request_prompt"]))
        self.assertEqual(trace_prompt_security.decrypt_payload(captured_event["request_prompt"], key=key), {"user": "plain"})
        trace_payload = append_trace.call_args.args[0]
        self.assertTrue(trace_prompt_security.is_encrypted_payload(trace_payload["request_prompt"]))
        self.assertEqual(trace_prompt_security.decrypt_payload(trace_payload["request_prompt"], key=key), {"user": "plain"})

    def test_write_request_body_dump_encrypts_late_plaintext_snapshot(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "req_1.json")
            snapshot = {
                "request_body": {"input": "plain"},
                "upstream_body": {"messages": [{"role": "user", "content": "plain"}]},
                "upstream_body_wire": '{"input":"plain"}',
            }

            with mock.patch.object(
                proxy.client_proxy_config_service,
                "load_client_proxy_settings",
                return_value={"trace_prompt_logging_enabled": True},
            ):
                proxy._set_trace_prompt_active_key(key, salt)
                try:
                    proxy._write_request_body_dump(out_path, tmp, snapshot)
                finally:
                    proxy._set_trace_prompt_active_key(None, None)

            with open(out_path, encoding="utf-8") as f:
                stored = json.load(f)
            self.assertTrue(trace_prompt_security.is_encrypted_payload(stored["request_body"]))
            self.assertEqual(trace_prompt_security.decrypt_payload(stored["request_body"], key=key), {"input": "plain"})

    def test_configured_trace_prompt_logging_can_be_enabled_without_unlock_password(self):
        verifier = {"_encrypted": "ghcp_proxy.aesgcm.v1", "ciphertext": "abc"}
        existing = {
            "trace_prompt_logging_enabled": False,
            "trace_prompt_logging_salt": "salt",
            "trace_prompt_logging_verifier": verifier,
        }
        saved_payload = {
            "trace_prompt_logging_enabled": True,
            "trace_prompt_logging_configured": True,
        }

        with (
            mock.patch.object(proxy, "_trace_prompt_logging_settings", return_value=existing),
            mock.patch.object(proxy.client_proxy_config_service, "save_client_proxy_settings", return_value=saved_payload) as save_settings,
            mock.patch.object(proxy, "_migrate_existing_prompt_trace_logs") as migrate,
        ):
            result = proxy._save_client_proxy_settings_with_trace_prompt_logging(
                {"trace_prompt_logging_enabled": True}
            )

        self.assertTrue(result["trace_prompt_logging_enabled"])
        self.assertFalse(result["trace_prompt_logging_unlocked"])
        save_settings.assert_called_once()
        migrate.assert_not_called()

    def test_trace_prompt_unlock_verifies_password_and_refreshes_dashboard_status(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        verifier = trace_prompt_security.make_password_verifier(key, salt)
        settings = {
            "trace_prompt_logging_enabled": True,
            "trace_prompt_logging_salt": salt,
            "trace_prompt_logging_verifier": verifier,
        }
        payload = {
            "trace_prompt_logging_enabled": True,
            "trace_prompt_logging_configured": True,
        }

        with (
            mock.patch.object(proxy, "_trace_prompt_logging_settings", return_value=settings),
            mock.patch.object(proxy.client_proxy_config_service, "client_proxy_settings_payload", return_value=payload),
            mock.patch.object(proxy.dashboard_service, "notify_dashboard_stream_listeners") as notify,
        ):
            try:
                result = proxy._unlock_trace_prompt_logging("pw")
            finally:
                proxy._set_trace_prompt_active_key(None, None)

        self.assertTrue(result["trace_prompt_logging_unlocked"])
        notify.assert_called_once()

    def test_prompt_trace_migration_encrypts_existing_prompt_files(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            usage_path = os.path.join(tmp, "usage-log.jsonl")
            trace_path = os.path.join(tmp, "request-trace.jsonl")
            body_dir = os.path.join(tmp, "request-bodies")
            os.makedirs(body_dir)
            with open(usage_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"request_id": "req_1", "request_prompt": {"user": "hello"}}) + "\n")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"event": "request_started", "request_prompt": {"system": "rules"}}) + "\n")
                f.write(json.dumps({"event": "request_finished", "source_body": {"input": "secret"}}) + "\n")
            body_path = os.path.join(body_dir, "req_1.json")
            with open(body_path, "w", encoding="utf-8") as f:
                json.dump({"request_body": {"input": "hello"}, "upstream_body_wire": '{"input":"hello"}'}, f)

            salt = trace_prompt_security.new_salt()
            key = trace_prompt_security.derive_key("pw", salt)

            with (
                mock.patch.object(proxy.usage_tracker, "usage_log_file", usage_path),
                mock.patch.dict(
                    os.environ,
                    {"GHCP_TRACE_LOG_FILE": trace_path, "GHCP_REQUEST_BODY_DUMP_DIR": body_dir},
                ),
            ):
                stats = proxy._migrate_existing_prompt_trace_logs(key, salt)

            self.assertEqual(stats["usage_log"]["encrypted_fields"], 1)
            self.assertEqual(stats["request_trace"]["encrypted_fields"], 2)
            self.assertEqual(stats["request_bodies"]["encrypted_fields"], 2)

            with open(usage_path, encoding="utf-8") as f:
                usage_row = json.loads(f.readline())
            with open(trace_path, encoding="utf-8") as f:
                trace_rows = [json.loads(line) for line in f if line.strip()]
            with open(body_path, encoding="utf-8") as f:
                body_row = json.load(f)

            self.assertTrue(trace_prompt_security.is_encrypted_payload(usage_row["request_prompt"]))
            self.assertEqual(trace_prompt_security.decrypt_payload(usage_row["request_prompt"], key=key), {"user": "hello"})
            self.assertEqual(trace_prompt_security.decrypt_payload(trace_rows[0]["request_prompt"], key=key), {"system": "rules"})
            self.assertEqual(trace_prompt_security.decrypt_payload(trace_rows[1]["source_body"], key=key), {"input": "secret"})
            self.assertEqual(trace_prompt_security.decrypt_payload(body_row["request_body"], key=key), {"input": "hello"})
            self.assertEqual(trace_prompt_security.decrypt_payload(body_row["upstream_body_wire"], key=key), '{"input":"hello"}')

    def test_prompt_trace_jsonl_migration_uses_supplied_lock(self):
        class RecordingLock:
            def __init__(self):
                self.entered = 0
                self.exited = 0

            def __enter__(self):
                self.entered += 1

            def __exit__(self, exc_type, exc, tb):
                self.exited += 1

        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "request-trace.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"request_prompt": {"user": "hello"}}) + "\n")
            salt = trace_prompt_security.new_salt()
            key = trace_prompt_security.derive_key("pw", salt)
            lock = RecordingLock()

            proxy._migrate_prompt_jsonl_file(path, ("request_prompt",), key, salt, lock=lock)

            self.assertEqual(lock.entered, 1)
            self.assertEqual(lock.exited, 1)


if __name__ == "__main__":
    unittest.main()
