import unittest
from concurrent.futures import ThreadPoolExecutor
import json
import os
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

import httpx

import proxy
import trace_prompt_security


class ProxyTracingTests(unittest.TestCase):
    def setUp(self):
        proxy._set_trace_prompt_active_key(None, None)
        proxy._set_trace_prompt_enable_pending(False)
        proxy._reset_debug_detail_capture_state()
        self._settings_patcher = mock.patch.object(
            proxy.client_proxy_config_service,
            "load_client_proxy_settings",
            return_value={
                "trace_prompt_logging_enabled": False,
                "trace_prompt_logging_salt": "",
                "trace_prompt_logging_verifier": None,
                "trace_prompt_logging_public_key": "",
                "trace_prompt_logging_private_key": None,
            },
        )
        self._settings_patcher.start()
        self.addCleanup(self._settings_patcher.stop)
        self.addCleanup(proxy._set_trace_prompt_enable_pending, False)
        self.addCleanup(proxy._set_trace_prompt_active_key, None, None)

    def test_prepare_upstream_request_keeps_full_prompt_preview_for_user_initiator(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
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
            proxy._set_trace_prompt_active_key(key, salt)
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
            proxy._set_trace_prompt_active_key(None, None)

        self.assertIsNone(error_response)
        self.assertIsNotNone(plan)

        prompt_preview = start_event.call_args.kwargs["prompt_preview"]
        prompt_preview = trace_prompt_security.decrypt_payload(prompt_preview, key=key)
        self.assertEqual(prompt_preview["system"], long_system)
        self.assertEqual(prompt_preview["user"], long_user)
        self.assertNotIn("system_truncated", prompt_preview)
        self.assertNotIn("user_truncated", prompt_preview)

        trace_payloads = [call.args[0] for call in append_trace.call_args_list]
        request_started = next(payload for payload in trace_payloads if payload["event"] == "request_started")
        self.assertEqual(
            trace_prompt_security.decrypt_payload(request_started["request_prompt"], key=key)["system"],
            long_system,
        )
        detail = next(payload for payload in trace_payloads if payload["event"] == "request_debug_detail")
        self.assertEqual(detail["debug_detail_capture"]["reasons"], ["user_initiated"])
        self.assertEqual(
            trace_prompt_security.decrypt_payload(detail["source_body"], key=key),
            source_body,
        )

    def test_prepare_upstream_request_omits_prompt_for_basic_agent_trace(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "do not retain me"},
                {"role": "user", "content": "secret user prompt"},
            ],
        }

        with (
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value={"request_id": "req_agent"}) as start_event,
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "_dump_outbound_request_body") as dump_body,
        ):
            plan, error_response = proxy._prepare_upstream_request(
                request,
                body=body,
                requested_model="gpt-5.4",
                resolved_model="gpt-5.4",
                upstream_path="/chat/completions",
                upstream_url="https://example.invalid/chat/completions",
                header_builder=lambda _api_key, _request_id: {"X-Initiator": "agent"},
                error_response=proxy.format_translation.openai_error_response,
                api_key="test-key",
            )

        self.assertIsNone(error_response)
        self.assertIsNotNone(plan)
        self.assertIsNone(start_event.call_args.kwargs["prompt_preview"])
        dump_body.assert_not_called()
        self.assertEqual(append_trace.call_count, 1)
        trace_payload = append_trace.call_args.args[0]
        self.assertEqual(trace_payload["event"], "request_started")
        self.assertNotIn("request_prompt", trace_payload)
        self.assertNotIn("debug_detail_capture", trace_payload)

    def test_safeguarded_agent_request_gets_encrypted_debug_detail(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "safeguarded prompt"}],
        }

        with (
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value={"request_id": "req_safe"}) as start_event,
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "_dump_outbound_request_body"),
        ):
            proxy._set_trace_prompt_active_key(key, salt)
            try:
                plan, error_response = proxy._prepare_upstream_request(
                    request,
                    body=body,
                    requested_model="gpt-5.4",
                    resolved_model="gpt-5.4",
                    upstream_path="/chat/completions",
                    upstream_url="https://example.invalid/chat/completions",
                    header_builder=lambda _api_key, _request_id: {"X-Initiator": "agent"},
                    error_response=proxy.format_translation.openai_error_response,
                    api_key="test-key",
                    trace_metadata={
                        "initiator_verdict": {
                            "candidate_initiator": "user",
                            "resolved_initiator": "agent",
                            "safeguard_reason": "cooldown",
                        }
                    },
                )
            finally:
                proxy._set_trace_prompt_active_key(None, None)

        self.assertIsNone(error_response)
        self.assertIsNotNone(plan)
        self.assertEqual(
            trace_prompt_security.decrypt_payload(start_event.call_args.kwargs["prompt_preview"], key=key),
            {"user": "safeguarded prompt"},
        )
        detail_events = [
            call.args[0]
            for call in append_trace.call_args_list
            if call.args[0].get("event") == "request_debug_detail"
        ]
        self.assertEqual(len(detail_events), 1)
        self.assertEqual(detail_events[0]["debug_detail_capture"]["reasons"], ["safeguarded"])
        self.assertEqual(trace_prompt_security.decrypt_payload(detail_events[0]["source_body"], key=key), body)

    def test_cache_bust_prompt_shrink_flushes_before_and_after_debug_detail(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        first_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage",
            "input": [{"type": "message", "role": "user", "content": "A" * 240}],
        }
        shrunk_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage",
            "input": [{"type": "message", "role": "user", "content": "short and different"}],
        }
        after_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage",
            "input": [{"type": "message", "role": "user", "content": "after context"}],
        }

        with (
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "_dump_outbound_request_body"),
            mock.patch.object(proxy, "CACHE_BUST_DRIFT_PREVIOUS_TOKEN_THRESHOLD", 10),
            mock.patch.object(proxy, "CACHE_BUST_DRIFT_DROP_TOKEN_THRESHOLD", 5),
        ):
            proxy._set_trace_prompt_active_key(key, salt)
            try:
                proxy._emit_request_trace_start(
                    request_id="req_before",
                    request=request,
                    upstream_url="https://example.invalid/responses",
                    upstream_path="/v1/responses",
                    requested_model="gpt-5.5",
                    resolved_model="gpt-5.5",
                    request_body=first_body,
                    upstream_body=first_body,
                    outbound_headers={"X-Initiator": "agent"},
                )
                proxy._emit_request_trace_start(
                    request_id="req_trigger",
                    request=request,
                    upstream_url="https://example.invalid/responses",
                    upstream_path="/v1/responses",
                    requested_model="gpt-5.5",
                    resolved_model="gpt-5.5",
                    request_body=shrunk_body,
                    upstream_body=shrunk_body,
                    outbound_headers={"X-Initiator": "agent"},
                )
                proxy._emit_request_trace_start(
                    request_id="req_after",
                    request=request,
                    upstream_url="https://example.invalid/responses",
                    upstream_path="/v1/responses",
                    requested_model="gpt-5.5",
                    resolved_model="gpt-5.5",
                    request_body=after_body,
                    upstream_body=after_body,
                    outbound_headers={"X-Initiator": "agent"},
                )
            finally:
                proxy._set_trace_prompt_active_key(None, None)

        detail_events = [
            call.args[0]
            for call in append_trace.call_args_list
            if call.args[0].get("event") == "request_debug_detail"
        ]
        phases = [event["debug_detail_capture"].get("phase") for event in detail_events]
        self.assertIn("before", phases)
        self.assertIn("trigger", phases)
        self.assertIn("after", phases)
        before_event = next(event for event in detail_events if event["debug_detail_capture"].get("phase") == "before")
        self.assertEqual(before_event["request_id"], "req_before")
        self.assertEqual(
            trace_prompt_security.decrypt_payload(before_event["source_body"], key=key),
            first_body,
        )
        trigger_event = next(event for event in detail_events if event["debug_detail_capture"].get("phase") == "trigger")
        self.assertIn("cache_bust_drift", trigger_event["debug_detail_capture"]["reasons"])
        after_event = next(event for event in detail_events if event["debug_detail_capture"].get("phase") == "after")
        self.assertEqual(after_event["request_id"], "req_after")

    def test_post_response_cache_bust_flushes_chain_when_cached_tokens_collapse(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        # Two requests in the same lineage where the prompt keeps growing
        # — the local prompt-shrink detector cannot fire on this trace —
        # but the upstream evicts the cache between them.
        first_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage-a",
            "input": [{"type": "message", "role": "user", "content": "A" * 2000}],
        }
        bust_body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage-a",
            "input": [
                {"type": "message", "role": "user", "content": "A" * 2000},
                {"type": "message", "role": "user", "content": "B" * 200},
            ],
        }
        outbound_headers = {"X-Initiator": "agent"}

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            with (
                mock.patch.object(proxy, "_append_request_trace"),
                mock.patch.object(proxy, "_dump_outbound_request_body"),
            ):
                proxy._emit_request_trace_start(
                    request_id="req_first",
                    request=request,
                    upstream_url="https://example.invalid/responses",
                    upstream_path="/v1/responses",
                    requested_model="gpt-5.5",
                    resolved_model="gpt-5.5",
                    request_body=first_body,
                    upstream_body=first_body,
                    outbound_headers=outbound_headers,
                )
                proxy._emit_request_trace_start(
                    request_id="req_bust",
                    request=request,
                    upstream_url="https://example.invalid/responses",
                    upstream_path="/v1/responses",
                    requested_model="gpt-5.5",
                    resolved_model="gpt-5.5",
                    request_body=bust_body,
                    upstream_body=bust_body,
                    outbound_headers=outbound_headers,
                )

            # First request had a strong cache hit, second collapses.
            proxy._post_response_cache_bust_diagnostics(
                request_id="req_first",
                upstream_body=first_body,
                resolved_model="gpt-5.5",
                outbound_headers=outbound_headers,
                usage={"input_tokens": 87000, "cached_input_tokens": 85000},
            )
            bust_pair = proxy._post_response_cache_bust_diagnostics(
                request_id="req_bust",
                upstream_body=bust_body,
                resolved_model="gpt-5.5",
                outbound_headers=outbound_headers,
                usage={"input_tokens": 88000, "cached_input_tokens": 22000},
            )

            self.assertIsNotNone(bust_pair)
            bust_lineage, bust_diagnostics = bust_pair
            self.assertEqual(bust_diagnostics["previous_request_id"], "req_first")
            self.assertEqual(bust_diagnostics["previous_cached_input_tokens"], 85000)
            self.assertEqual(bust_diagnostics["current_cached_input_tokens"], 22000)

            detail_events = proxy._register_post_response_cache_bust_capture(
                "req_bust", bust_lineage, bust_diagnostics
            )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        phases = [event["debug_detail_capture"]["phase"] for event in detail_events]
        self.assertEqual(phases.count("before"), 1)
        self.assertEqual(phases.count("trigger"), 1)
        before_event = next(e for e in detail_events if e["debug_detail_capture"]["phase"] == "before")
        trigger_event = next(e for e in detail_events if e["debug_detail_capture"]["phase"] == "trigger")
        self.assertEqual(before_event["request_id"], "req_first")
        self.assertEqual(trigger_event["request_id"], "req_bust")
        self.assertIn("cache_bust_usage_drift", trigger_event["debug_detail_capture"]["reasons"])
        self.assertEqual(
            trigger_event["debug_detail_capture"]["cache_bust_usage_diagnostics"],
            bust_diagnostics,
        )
        self.assertEqual(
            trace_prompt_security.decrypt_payload(trigger_event["source_body"], key=key),
            bust_body,
        )
        # No `_lineage` scratch field should leak into emitted events.
        self.assertNotIn("_lineage", before_event)
        self.assertNotIn("_lineage", trigger_event)

    def test_post_response_cache_bust_caps_full_detail_to_ten_same_session_requests(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        def body_for(session_id: str, content: str) -> dict:
            return {
                "model": "gpt-5.5",
                "session_id": session_id,
                "input": [{"type": "message", "role": "user", "content": content}],
            }

        def emit(request_id: str, body: dict) -> None:
            proxy._emit_request_trace_start(
                request_id=request_id,
                request=request,
                upstream_url="https://example.invalid/responses",
                upstream_path="/v1/responses",
                requested_model="gpt-5.5",
                resolved_model="gpt-5.5",
                request_body=body,
                upstream_body=body,
                outbound_headers={"X-Initiator": "agent"},
            )

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            with (
                mock.patch.object(proxy, "_append_request_trace"),
                mock.patch.object(proxy, "_dump_outbound_request_body"),
            ):
                for index in range(10):
                    emit(
                        f"req_unrelated_prior_{index}",
                        body_for(f"unrelated-prior-{index}", f"unrelated prior {index}"),
                    )
                    emit(
                        f"req_prior_{index}",
                        body_for("target-session", f"prior {index}"),
                    )
                buster_body = body_for("target-session", "buster")
                emit("req_bust", buster_body)
                for index in range(10):
                    emit(
                        f"req_after_{index}",
                        body_for("target-session", f"after {index}"),
                    )
                    emit(
                        f"req_unrelated_after_{index}",
                        body_for(f"unrelated-after-{index}", f"unrelated after {index}"),
                    )

            diagnostics = {
                "session_key": "session:target-session",
                "previous_request_id": "req_previous_usage",
                "previous_cached_input_tokens": 85_000,
                "current_cached_input_tokens": 20_000,
                "cache_drop": 65_000,
            }
            detail_events = proxy._register_post_response_cache_bust_capture(
                "req_bust",
                "session:target-session",
                diagnostics,
            )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        phases = [event["debug_detail_capture"]["phase"] for event in detail_events]
        self.assertEqual(len(detail_events), 10)
        self.assertEqual(phases.count("before"), 9)
        self.assertEqual(phases.count("trigger"), 1)
        self.assertEqual(phases.count("after"), 0)
        self.assertEqual(
            [event["request_id"] for event in detail_events if event["debug_detail_capture"]["phase"] == "before"],
            [f"req_prior_{index}" for index in range(1, 10)],
        )
        trigger_event = next(event for event in detail_events if event["debug_detail_capture"]["phase"] == "trigger")
        self.assertEqual(trigger_event["request_id"], "req_bust")
        self.assertEqual(
            trace_prompt_security.decrypt_payload(trigger_event["source_body"], key=key),
            buster_body,
        )
        for event in detail_events:
            self.assertEqual(
                event["debug_detail_capture"]["cache_bust_usage_diagnostics"],
                diagnostics,
            )
            self.assertNotIn("_lineage", event)

    def test_post_response_cache_bust_future_capture_records_remaining_same_session_slots(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        def body_for(session_id: str, content: str) -> dict:
            return {
                "model": "gpt-5.5",
                "session_id": session_id,
                "input": [{"type": "message", "role": "user", "content": content}],
            }

        def emit(request_id: str, body: dict) -> None:
            proxy._emit_request_trace_start(
                request_id=request_id,
                request=request,
                upstream_url="https://example.invalid/responses",
                upstream_path="/v1/responses",
                requested_model="gpt-5.5",
                resolved_model="gpt-5.5",
                request_body=body,
                upstream_body=body,
                outbound_headers={"X-Initiator": "agent"},
            )

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            with (
                mock.patch.object(proxy, "_append_request_trace") as append_trace,
                mock.patch.object(proxy, "_dump_outbound_request_body"),
            ):
                emit("req_before", body_for("target-session", "before"))
                emit("req_bust", body_for("target-session", "buster"))
                diagnostics = {
                    "session_key": "session:target-session",
                    "previous_request_id": "req_before",
                    "previous_cached_input_tokens": 11,
                    "current_cached_input_tokens": 10,
                    "cache_drop": 1,
                }
                initial_events = proxy._register_post_response_cache_bust_capture(
                    "req_bust",
                    "session:target-session",
                    diagnostics,
                )
                self.assertEqual(
                    [event["debug_detail_capture"]["phase"] for event in initial_events],
                    ["before", "trigger"],
                )
                append_trace.reset_mock()

                for index in range(12):
                    emit(
                        f"req_unrelated_after_{index}",
                        body_for("unrelated-session", f"unrelated {index}"),
                    )
                    emit(
                        f"req_after_{index}",
                        body_for("target-session", f"after {index}"),
                    )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        after_detail_events = [
            call.args[0]
            for call in append_trace.call_args_list
            if call.args[0].get("event") == "request_debug_detail"
            and call.args[0].get("debug_detail_capture", {}).get("phase") == "after"
        ]
        self.assertEqual(
            [event["request_id"] for event in after_detail_events],
            [f"req_after_{index}" for index in range(8)],
        )
        for event in after_detail_events:
            self.assertEqual(
                event["debug_detail_capture"]["cache_bust_usage_diagnostics"],
                diagnostics,
            )

    def test_repeated_post_response_cache_bust_dedupes_full_detail_per_session(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        def body_for(content: str) -> dict:
            return {
                "model": "gpt-5.5",
                "session_id": "target-session",
                "input": [{"type": "message", "role": "user", "content": content}],
            }

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            with (
                mock.patch.object(proxy, "_append_request_trace"),
                mock.patch.object(proxy, "_dump_outbound_request_body"),
            ):
                for index in range(12):
                    body = body_for(f"request {index}")
                    proxy._emit_request_trace_start(
                        request_id=f"req_{index}",
                        request=request,
                        upstream_url="https://example.invalid/responses",
                        upstream_path="/v1/responses",
                        requested_model="gpt-5.5",
                        resolved_model="gpt-5.5",
                        request_body=body,
                        upstream_body=body,
                        outbound_headers={"X-Initiator": "agent"},
                    )

            diagnostics = {
                "session_key": "session:target-session",
                "previous_cached_input_tokens": 85_000,
                "current_cached_input_tokens": 20_000,
                "cache_drop": 65_000,
            }
            first_events = proxy._register_post_response_cache_bust_capture(
                "req_10",
                "session:target-session",
                diagnostics,
            )
            second_events = proxy._register_post_response_cache_bust_capture(
                "req_11",
                "session:target-session",
                diagnostics,
            )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        all_events = first_events + second_events
        self.assertEqual(len(all_events), 10)
        self.assertEqual(len({event["request_id"] for event in all_events}), 10)
        self.assertEqual(second_events, [])

    def test_debug_detail_full_capture_limit_survives_concurrent_future_starts(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        trace_events = []
        trace_events_lock = threading.Lock()

        def append_trace(event: dict) -> None:
            with trace_events_lock:
                trace_events.append(event)

        def body_for(content: str) -> dict:
            return {
                "model": "gpt-5.5",
                "session_id": "target-session",
                "input": [{"type": "message", "role": "user", "content": content}],
            }

        def emit(request_id: str, content: str) -> None:
            body = body_for(content)
            proxy._emit_request_trace_start(
                request_id=request_id,
                request=request,
                upstream_url="https://example.invalid/responses",
                upstream_path="/v1/responses",
                requested_model="gpt-5.5",
                resolved_model="gpt-5.5",
                request_body=body,
                upstream_body=body,
                outbound_headers={"X-Initiator": "agent"},
            )

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            with (
                mock.patch.object(proxy, "_append_request_trace", side_effect=append_trace),
                mock.patch.object(proxy, "_dump_outbound_request_body"),
            ):
                emit("req_bust", "buster")
                diagnostics = {
                    "session_key": "session:target-session",
                    "previous_cached_input_tokens": 85_000,
                    "current_cached_input_tokens": 20_000,
                    "cache_drop": 65_000,
                }
                initial_events = proxy._register_post_response_cache_bust_capture(
                    "req_bust",
                    "session:target-session",
                    diagnostics,
                )

                with ThreadPoolExecutor(max_workers=12) as executor:
                    list(
                        executor.map(
                            lambda index: emit(f"req_after_{index}", f"after {index}"),
                            range(30),
                        )
                    )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        future_detail_events = [
            event
            for event in trace_events
            if event.get("event") == "request_debug_detail"
            and event.get("debug_detail_capture", {}).get("phase") == "after"
        ]
        all_detail_events = initial_events + future_detail_events
        self.assertEqual(len(all_detail_events), 10)
        self.assertEqual(len({event["request_id"] for event in all_detail_events}), 10)
        self.assertEqual(len(future_detail_events), 9)
        self.assertEqual(
            proxy._DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS["session:target-session"],
            {event["request_id"] for event in all_detail_events},
        )

    def test_debug_detail_session_lru_keeps_at_most_six_session_buffers(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            with (
                mock.patch.object(proxy, "_append_request_trace"),
                mock.patch.object(proxy, "_dump_outbound_request_body"),
            ):
                for index in range(7):
                    body = {
                        "model": "gpt-5.5",
                        "session_id": f"session-{index}",
                        "input": [{"type": "message", "role": "user", "content": str(index)}],
                    }
                    proxy._emit_request_trace_start(
                        request_id=f"req_{index}",
                        request=request,
                        upstream_url="https://example.invalid/responses",
                        upstream_path="/v1/responses",
                        requested_model="gpt-5.5",
                        resolved_model="gpt-5.5",
                        request_body=body,
                        upstream_body=body,
                        outbound_headers={"X-Initiator": "agent"},
                    )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        self.assertEqual(len(proxy._DEBUG_DETAIL_SESSION_RECENT_REQUESTS), 6)
        self.assertNotIn("session:session-0", proxy._DEBUG_DETAIL_SESSION_RECENT_REQUESTS)
        self.assertEqual(
            list(proxy._DEBUG_DETAIL_SESSION_RECENT_REQUESTS.keys()),
            [f"session:session-{index}" for index in range(1, 7)],
        )

    def test_post_response_cache_bust_triggers_on_any_cached_token_decrease(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
        body = {
            "model": "gpt-5.5",
            "session_id": "small-decrease-session",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        outbound_headers = {"X-Initiator": "agent"}

        proxy._set_trace_prompt_active_key(key, salt)
        try:
            first = proxy._post_response_cache_bust_diagnostics(
                request_id="req_first",
                upstream_body=body,
                resolved_model="gpt-5.5",
                outbound_headers=outbound_headers,
                usage={"input_tokens": 100, "cached_input_tokens": 10},
            )
            second = proxy._post_response_cache_bust_diagnostics(
                request_id="req_second",
                upstream_body=body,
                resolved_model="gpt-5.5",
                outbound_headers=outbound_headers,
                usage={"input_tokens": 101, "cached_input_tokens": 9},
            )
        finally:
            proxy._set_trace_prompt_active_key(None, None)

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        session_key, diagnostics = second
        self.assertEqual(session_key, "session:small-decrease-session")
        self.assertEqual(diagnostics["cache_drop"], 1)
        self.assertEqual(diagnostics["previous_cached_input_tokens"], 10)
        self.assertEqual(diagnostics["current_cached_input_tokens"], 9)

    def test_post_response_cache_bust_skips_when_encryption_not_set_up(self):
        # No active key, no public key — encryption is not set up, so prompt
        # logging is not permitted. Detection must skip rather than capture
        # plaintext or write a locked placeholder.
        proxy._set_trace_prompt_active_key(None, None)
        body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage-locked",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        outbound_headers = {"X-Initiator": "agent"}
        first = proxy._post_response_cache_bust_diagnostics(
            request_id="req_first",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers=outbound_headers,
            usage={"input_tokens": 87000, "cached_input_tokens": 85000},
        )
        second = proxy._post_response_cache_bust_diagnostics(
            request_id="req_bust",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers=outbound_headers,
            usage={"input_tokens": 88000, "cached_input_tokens": 22000},
        )
        self.assertIsNone(first)
        self.assertIsNone(second)

    def test_user_initiated_capture_skips_when_encryption_not_set_up(self):
        # Without a key, an X-Initiator: user request must not produce a
        # debug-detail event or a stored prompt preview.
        proxy._set_trace_prompt_active_key(None, None)
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "secret prompt"}],
        }

        with (
            mock.patch.object(proxy, "request_tracing_enabled", return_value=True),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value={"request_id": "req_user"}) as start_event,
            mock.patch.object(proxy, "_append_request_trace") as append_trace,
            mock.patch.object(proxy, "_dump_outbound_request_body") as dump_body,
        ):
            plan, _ = proxy._prepare_upstream_request(
                request,
                body=body,
                requested_model="gpt-5.4",
                resolved_model="gpt-5.4",
                upstream_path="/chat/completions",
                upstream_url="https://example.invalid/chat/completions",
                header_builder=lambda _api_key, _request_id: {"X-Initiator": "user"},
                error_response=proxy.format_translation.openai_error_response,
                api_key="test-key",
            )

        self.assertIsNotNone(plan)
        self.assertIsNone(start_event.call_args.kwargs.get("prompt_preview"))
        events = [call.args[0] for call in append_trace.call_args_list]
        self.assertFalse(any(event.get("event") == "request_debug_detail" for event in events))
        request_started = next(event for event in events if event.get("event") == "request_started")
        self.assertNotIn("request_prompt", request_started)
        dump_body.assert_not_called()

    def test_post_response_cache_bust_ignores_stable_cache_hit_rate(self):
        body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "lineage-stable",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        outbound_headers = {"X-Initiator": "agent"}
        proxy._post_response_cache_bust_diagnostics(
            request_id="req_first",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers=outbound_headers,
            usage={"input_tokens": 50000, "cached_input_tokens": 49000},
        )
        result = proxy._post_response_cache_bust_diagnostics(
            request_id="req_second",
            upstream_body=body,
            resolved_model="gpt-5.5",
            outbound_headers=outbound_headers,
            usage={"input_tokens": 51000, "cached_input_tokens": 49500},
        )
        self.assertIsNone(result)

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
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key("pw", salt)
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
            proxy._set_trace_prompt_active_key(key, salt)
            proxy._finish_usage_and_trace(
                plan,
                400,
                upstream=upstream,
                response_payload=response_payload,
                response_text="unsupported field: tool_choice",
            )
            proxy._set_trace_prompt_active_key(None, None)

        append_trace.assert_called_once()
        trace_payload = append_trace.call_args.args[0]
        trace_kwargs = append_trace.call_args.kwargs

        self.assertTrue(trace_kwargs["force"])
        self.assertEqual(trace_payload["requested_model"], "gpt-5.4-mini")
        self.assertEqual(trace_payload["resolved_model"], "claude-sonnet-4.6")
        self.assertEqual(
            trace_prompt_security.decrypt_payload(trace_payload["source_body"], key=key),
            {"model": "gpt-5.4-mini", "input": "hello"},
        )
        self.assertEqual(
            trace_prompt_security.decrypt_payload(trace_payload["upstream_body"], key=key)["model"],
            "claude-sonnet-4.6",
        )
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

    def test_prompt_trace_protection_encrypts_with_public_key_without_unlock(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        key_pair = trace_prompt_security.generate_envelope_key_pair()
        settings = {
            "trace_prompt_logging_enabled": True,
            "trace_prompt_logging_public_key": key_pair["public_key"],
        }

        with mock.patch.object(proxy.client_proxy_config_service, "load_client_proxy_settings", return_value=settings):
            proxy._set_trace_prompt_active_key(None, None)
            protected = proxy._protect_prompt_trace_value({"input": "plain"})

        self.assertEqual(protected.get("_encrypted"), trace_prompt_security.ENVELOPE_PAYLOAD_MARKER)
        self.assertEqual(
            trace_prompt_security.decrypt_payload(protected, private_key_pem=key_pair["private_key"]),
            {"input": "plain"},
        )

    def test_write_request_body_dump_uses_public_key_without_password(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        key_pair = trace_prompt_security.generate_envelope_key_pair()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "req_1.json")
            snapshot = {
                "request_body": {"input": "plain"},
                "upstream_body": {"messages": [{"role": "user", "content": "plain"}]},
            }
            settings = {
                "trace_prompt_logging_enabled": True,
                "trace_prompt_logging_public_key": key_pair["public_key"],
            }

            with mock.patch.object(proxy.client_proxy_config_service, "load_client_proxy_settings", return_value=settings):
                proxy._set_trace_prompt_active_key(None, None)
                proxy._write_request_body_dump(out_path, tmp, snapshot)

            with open(out_path, encoding="utf-8") as f:
                stored = json.load(f)
            self.assertEqual(stored["request_body"].get("_encrypted"), trace_prompt_security.ENVELOPE_PAYLOAD_MARKER)
            self.assertNotEqual(stored["request_body"].get("locked"), True)
            self.assertEqual(
                trace_prompt_security.decrypt_payload(stored["request_body"], private_key_pem=key_pair["private_key"]),
                {"input": "plain"},
            )

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
        key_pair = trace_prompt_security.generate_envelope_key_pair()
        settings = {
            "trace_prompt_logging_enabled": True,
            "trace_prompt_logging_salt": salt,
            "trace_prompt_logging_verifier": verifier,
            "trace_prompt_logging_public_key": key_pair["public_key"],
            "trace_prompt_logging_private_key": trace_prompt_security.encrypt_payload(key_pair["private_key"], key, salt),
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

    def test_trace_prompt_unlock_upgrades_legacy_password_settings_with_envelope_keypair(self):
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
            "trace_prompt_logging_public_key": "",
            "trace_prompt_logging_private_key": None,
        }
        saved_payload = {
            "trace_prompt_logging_enabled": True,
            "trace_prompt_logging_configured": True,
            "trace_prompt_logging_public_key_configured": True,
        }

        with (
            mock.patch.object(proxy, "_trace_prompt_logging_settings", return_value=settings),
            mock.patch.object(proxy.client_proxy_config_service, "save_client_proxy_settings", return_value=saved_payload) as save_settings,
            mock.patch.object(proxy.client_proxy_config_service, "client_proxy_settings_payload", return_value=saved_payload),
            mock.patch.object(proxy.dashboard_service, "notify_dashboard_stream_listeners"),
        ):
            try:
                result = proxy._unlock_trace_prompt_logging("pw")
            finally:
                proxy._set_trace_prompt_active_key(None, None)

        self.assertTrue(result["trace_prompt_logging_unlocked"])
        saved = save_settings.call_args.args[0]
        self.assertIn("BEGIN PUBLIC KEY", saved["trace_prompt_logging_public_key"])
        self.assertTrue(trace_prompt_security.is_encrypted_payload(saved["trace_prompt_logging_private_key"]))
        private_key = trace_prompt_security.decrypt_payload(saved["trace_prompt_logging_private_key"], key=key)
        self.assertIn("BEGIN PRIVATE KEY", private_key)

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
