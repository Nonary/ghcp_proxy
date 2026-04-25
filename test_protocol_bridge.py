import unittest

import proxy
from protocol_bridge import ProtocolBridgePlanner


class _RoutingConfigStub:
    def __init__(self, target_model=None, approval_target_model=None, compact_fallback_model=None):
        self.target_model = target_model
        self.approval_target_model = approval_target_model
        self.compact_fallback_model = compact_fallback_model

    def resolve_target_model(self, requested_model):
        del requested_model
        return self.target_model

    def resolve_approval_target_model(self, requested_model):
        del requested_model
        return self.approval_target_model

    def resolve_compact_fallback_model(self, requested_model):
        del requested_model
        return self.compact_fallback_model


class ProtocolBridgePlannerTests(unittest.TestCase):
    def setUp(self):
        # Reset the Copilot /models capability cache so cross-test
        # pollution cannot accidentally enable the native /v1/messages
        # bridge for plans that expect the legacy chat/responses path.
        cache = getattr(proxy, "_COPILOT_MODEL_CAPS_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()

    def test_planner_selects_native_responses_strategy_for_codex_models(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {"model": "gpt-5.4", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertEqual(plan.upstream_path, "/responses")
        self.assertEqual(plan.header_kind, "responses")
        self.assertEqual(plan.resolved_model, "gpt-5.4")

    def test_planner_strips_unsupported_image_generation_tools_for_native_responses(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "stream": False,
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
            "parallel_tool_calls": True,
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertNotIn("tools", plan.upstream_body)
        self.assertNotIn("tool_choice", plan.upstream_body)
        self.assertNotIn("parallel_tool_calls", plan.upstream_body)

    def test_planner_strips_service_tier_but_preserves_cache_affinity_for_native_responses(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "stream": False,
            "service_tier": "priority",
            "sessionId": "session-123",
            "prompt_cache_key": "cache-123",
            "previous_response_id": "resp_prev",
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertNotIn("service_tier", plan.upstream_body)
        self.assertEqual(plan.upstream_body["sessionId"], "session-123")
        self.assertEqual(plan.upstream_body["prompt_cache_key"], "cache-123")
        self.assertEqual(plan.upstream_body["previous_response_id"], "resp_prev")

    def test_planner_strips_invalid_deferred_tools_for_native_responses(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {
            "model": "gpt-5.5",
            "input": "hello",
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
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key", is_compact=True)
        )

        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertNotIn("defer_loading", plan.upstream_body["tools"][0]["tools"][0])
        self.assertEqual(plan.diagnostics[0]["action"], "strip_defer_loading")
        self.assertEqual(plan.diagnostics[0]["reason"], "deferred_tools_require_tool_search")

    def test_planner_selects_responses_to_chat_strategy_when_mapping_targets_claude(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("claude-opus-4.6"))
        body = {
            "model": "gpt-5.3-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_chat")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertEqual(plan.header_kind, "chat")
        self.assertEqual(plan.upstream_body["messages"][0]["role"], "user")
        self.assertEqual(plan.upstream_body["messages"][0]["content"], "hello")

    def test_planner_selects_responses_to_chat_strategy_when_mapping_targets_gemini(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("gemini-3.1-pro-preview"))
        body = {
            "model": "gpt-5.3-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_chat")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertEqual(plan.header_kind, "chat")
        self.assertEqual(plan.resolved_model, "gemini-3.1-pro-preview")
        self.assertEqual(plan.upstream_body["messages"][0]["role"], "user")
        self.assertEqual(plan.upstream_body["messages"][0]["content"], "hello")

    def test_planner_selects_messages_to_responses_strategy_when_mapping_targets_codex(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("gpt-5.4"))
        body = {
            "model": "claude-opus-4.6",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "messages_to_responses")
        self.assertEqual(plan.upstream_path, "/responses")
        self.assertEqual(plan.header_kind, "responses")
        self.assertEqual(plan.upstream_body["input"][0]["role"], "user")
        self.assertEqual(plan.upstream_body["input"][0]["content"][0]["text"], "hello")

    def test_planner_selects_messages_to_chat_strategy_when_mapping_targets_grok(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("grok-code-fast-1"))
        body = {
            "model": "claude-opus-4.6",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "messages_to_chat")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertEqual(plan.header_kind, "anthropic")
        self.assertEqual(plan.resolved_model, "grok-code-fast-1")
        self.assertEqual(plan.upstream_body["messages"][0]["role"], "user")
        self.assertEqual(plan.upstream_body["messages"][0]["content"], "hello")

    def test_planner_uses_approval_mapping_for_codex_subagent_request(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.6", approval_target_model="gpt-5.4-mini")
        )
        body = {"model": "gpt-5.4", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                subagent="guardian",
            )
        )

        self.assertEqual(plan.resolved_model, "gpt-5.4-mini")
        self.assertEqual(plan.strategy_name, "responses_to_responses")

    def test_planner_uses_approval_mapping_for_responses_security_monitor_system_prompt(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.6", approval_target_model="gpt-5.4-mini")
        )
        body = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": "You are a security monitor for autonomous AI coding agents.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "review this action",
                },
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.resolved_model, "gpt-5.4-mini")
        self.assertEqual(plan.strategy_name, "responses_to_responses")

    def test_planner_uses_approval_mapping_for_messages_security_monitor_system_prompt(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="gpt-5.4", approval_target_model="claude-haiku-4.5")
        )
        body = {
            "model": "claude-sonnet-4.6",
            "system": "You are a security monitor for autonomous AI coding agents.",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "review this action"}],
                }
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.resolved_model, "claude-haiku-4-5")
        self.assertEqual(plan.strategy_name, "messages_to_chat")

    def test_planner_falls_back_to_regular_mapping_when_no_approval_rule(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.6", approval_target_model=None)
        )
        body = {"model": "gpt-5.4", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                subagent="guardian",
            )
        )

        self.assertEqual(plan.resolved_model, "claude-opus-4.6")

    def test_planner_does_not_use_approval_mapping_for_review_subagent(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.7", approval_target_model="gpt-5.4-mini")
        )
        body = {"model": "claude-opus-4.7", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                subagent="review",
            )
        )

        self.assertEqual(plan.resolved_model, "claude-opus-4.7")

    def test_planner_does_not_use_approval_mapping_for_general_purpose_subagent(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.7", approval_target_model="gpt-5.4-mini")
        )
        body = {"model": "claude-opus-4.7", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                subagent="general-purpose",
            )
        )

        self.assertEqual(plan.resolved_model, "claude-opus-4.7")
    def test_planner_keeps_chat_target_on_compact_against_claude(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.6", compact_fallback_model="gpt-5.4")
        )
        body = {"model": "gpt-5.3-codex", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=True,
            )
        )

        self.assertEqual(plan.strategy_name, "responses_to_chat")
        self.assertEqual(plan.resolved_model, "claude-opus-4.6")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertTrue(plan.is_compact)

    def test_planner_keeps_direct_claude_compact_on_chat(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {"model": "claude-opus-4.7", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=True,
            )
        )

        self.assertEqual(plan.strategy_name, "responses_to_chat")
        self.assertEqual(plan.resolved_model, "claude-opus-4.7")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertTrue(plan.is_compact)

    def test_planner_keeps_chat_target_on_compact_against_gemini(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="gemini-3.1-pro-preview", compact_fallback_model="gpt-5.4")
        )
        body = {"model": "gpt-5.3-codex", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=True,
            )
        )

        self.assertEqual(plan.strategy_name, "responses_to_chat")
        self.assertEqual(plan.resolved_model, "gemini-3.1-pro-preview")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertTrue(plan.is_compact)

    def test_planner_keeps_chat_target_on_compact_against_grok(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="grok-code-fast-1", compact_fallback_model="gpt-5.4")
        )
        body = {"model": "gpt-5.3-codex", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=True,
            )
        )

        self.assertEqual(plan.strategy_name, "responses_to_chat")
        self.assertEqual(plan.resolved_model, "grok-code-fast-1")
        self.assertEqual(plan.upstream_path, "/chat/completions")
        self.assertTrue(plan.is_compact)

    def test_planner_does_not_swap_on_compact_when_target_is_codex(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="gpt-5.4", compact_fallback_model=None)
        )
        body = {"model": "gpt-5.3-codex", "input": "hello", "stream": False}

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=True,
            )
        )

        self.assertEqual(plan.resolved_model, "gpt-5.4")
        self.assertEqual(plan.strategy_name, "responses_to_responses")

    def test_planner_keeps_claude_on_non_compact_request(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub(target_model="claude-opus-4.6", compact_fallback_model="gpt-5.4")
        )
        body = {
            "model": "gpt-5.3-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan(
                "responses",
                body,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=False,
            )
        )

        self.assertEqual(plan.resolved_model, "claude-opus-4.6")
        self.assertEqual(plan.strategy_name, "responses_to_chat")


class NativeMessagesBridgeTests(unittest.TestCase):
    def _planner(self, target_model, *, supports=True):
        resolver = lambda model: bool(supports) and (model or "").startswith("claude")
        return ProtocolBridgePlanner(
            _RoutingConfigStub(target_model=target_model),
            capability_resolver=resolver,
        )

    def test_messages_to_messages_matches_when_capability_supported(self):
        planner = self._planner("claude-sonnet-4.6", supports=True)
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "stream": False,
        }
        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="k")
        )
        self.assertEqual(plan.strategy_name, "messages_to_messages")
        self.assertEqual(plan.upstream_protocol, "messages")
        self.assertEqual(plan.upstream_path, "/v1/messages")
        self.assertEqual(plan.header_kind, "messages")
        self.assertEqual(plan.caller_protocol, "anthropic")
        self.assertEqual(plan.upstream_body["model"], "claude-sonnet-4.6")

    def test_messages_to_messages_falls_back_when_capability_unsupported(self):
        planner = self._planner("claude-sonnet-4.6", supports=False)
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "stream": False,
        }
        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="k")
        )
        # Falls through to the existing chat bridge for Claude targets.
        self.assertEqual(plan.strategy_name, "messages_to_chat")
        self.assertEqual(plan.upstream_path, "/chat/completions")

    def test_responses_to_messages_matches_for_codex_inbound_claude_target(self):
        planner = self._planner("claude-opus-4.6", supports=True)
        body = {
            "model": "gpt-5.3-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            "stream": False,
        }
        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="k")
        )
        self.assertEqual(plan.strategy_name, "responses_to_messages")
        self.assertEqual(plan.upstream_protocol, "messages")
        self.assertEqual(plan.upstream_path, "/v1/messages")
        self.assertEqual(plan.header_kind, "messages")
        self.assertEqual(plan.caller_protocol, "responses")

    def test_responses_to_messages_preserves_prompt_cache_with_cache_control(self):
        planner = self._planner("claude-opus-4.6", supports=True)
        body = {
            "model": "gpt-5.3-codex",
            "prompt_cache_key": "session-123",
            "instructions": "system prompt",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "again"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "mcp__ide__executeCode",
                    "description": "Execute code",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="k")
        )

        self.assertEqual(plan.strategy_name, "responses_to_messages")
        self.assertEqual([tool["name"] for tool in plan.upstream_body["tools"]], ["read"])
        self.assertEqual(plan.upstream_body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(plan.upstream_body["tools"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(
            plan.upstream_body["messages"][-2]["content"][-1]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(
            plan.upstream_body["messages"][-1]["content"][-1]["cache_control"],
            {"type": "ephemeral"},
        )

    def test_responses_to_messages_does_not_match_for_gpt5_target(self):
        planner = self._planner("gpt-5", supports=True)
        body = {"model": "gpt-5", "input": "hi", "stream": False}
        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="k")
        )
        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertEqual(plan.upstream_path, "/responses")

    def test_upstream_path_returns_v1_messages_for_messages_protocol(self):
        from protocol_bridge import BridgeExecutionPlan
        plan = BridgeExecutionPlan(
            strategy_name="x",
            inbound_protocol="messages",
            caller_protocol="anthropic",
            upstream_protocol="messages",
            header_kind="messages",
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            upstream_body={},
            stream=False,
        )
        self.assertEqual(plan.upstream_path, "/v1/messages")


class ProxyNativeMessagesCapabilityTests(unittest.TestCase):
    def setUp(self):
        self._clear_capability_cache()

    def tearDown(self):
        self._clear_capability_cache()

    def _clear_capability_cache(self):
        cache = getattr(proxy, "_COPILOT_MODEL_CAPS_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()

    def test_proxy_native_messages_supports_opus_47_without_models_cache(self):
        self.assertTrue(proxy.model_supports_native_messages("claude-opus-4.7"))

    def test_proxy_native_messages_allowlist_overrides_stale_negative_cache(self):
        cache = getattr(proxy, "_COPILOT_MODEL_CAPS_CACHE", None)
        self.assertIsInstance(cache, dict)
        cache.update(
            {
                "key": "https://example.invalid",
                "data": {
                    "claude-opus-4.7": {
                        "messages_endpoint_supported": False,
                        "supported_endpoints": ["/chat/completions"],
                    }
                },
            }
        )

        self.assertTrue(proxy.model_supports_native_messages("claude-opus-4.7"))

    def test_runtime_planner_prefers_messages_bridge_for_opus_47_responses(self):
        planner = ProtocolBridgePlanner(
            _RoutingConfigStub("claude-opus-4.7"),
            capability_resolver=lambda model: proxy.model_supports_native_messages(model) if model else False,
        )
        body = {
            "model": "gpt-5.3-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_messages")
        self.assertEqual(plan.upstream_path, "/v1/messages")
        self.assertEqual(plan.header_kind, "messages")

if __name__ == "__main__":
    unittest.main()
