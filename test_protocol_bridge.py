import unittest
from unittest import mock

import protocol_bridge
import proxy
from protocol_bridge import BridgeExecutionPlan, ProtocolBridgePlanner, ProtocolBridgeStrategy


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

    def test_default_capability_resolver_requires_explicit_models_cache_support(self):
        cache = getattr(proxy, "_COPILOT_MODEL_CAPS_CACHE", None)
        self.assertIsInstance(cache, dict)

        self.assertFalse(protocol_bridge._default_capability_resolver(None))
        self.assertFalse(protocol_bridge._default_capability_resolver("claude-sonnet-4.6"))

        cache.update(
            {
                "data": {
                    "claude-sonnet-4.6": {"messages_endpoint_supported": True},
                    "claude-opus-4.6": {"messages_endpoint_supported": False},
                    "claude-haiku-4.5": "not-a-record",
                }
            }
        )

        self.assertTrue(protocol_bridge._default_capability_resolver("claude-sonnet-4.6"))
        self.assertFalse(protocol_bridge._default_capability_resolver("claude-opus-4.6"))
        self.assertFalse(protocol_bridge._default_capability_resolver("claude-haiku-4.5"))

    def test_default_cache_resolvers_treat_proxy_import_failure_as_unknown(self):
        real_import = __import__

        def fail_proxy_import(name, *args, **kwargs):
            if name == "proxy":
                raise ImportError("proxy unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fail_proxy_import):
            self.assertFalse(protocol_bridge._default_capability_resolver("claude-sonnet-4.6"))
            self.assertIsNone(protocol_bridge._resolve_reasoning_efforts("gpt-5.4"))
            self.assertFalse(protocol_bridge._default_adaptive_thinking_resolver("claude-sonnet-4.6"))

    def test_resolve_reasoning_efforts_reads_only_string_entries_from_cache(self):
        cache = getattr(proxy, "_COPILOT_MODEL_CAPS_CACHE", None)
        self.assertIsInstance(cache, dict)

        self.assertIsNone(protocol_bridge._resolve_reasoning_efforts(None))
        self.assertIsNone(protocol_bridge._resolve_reasoning_efforts("gpt-5.4"))

        cache.update(
            {
                "data": {
                    "gpt-5.4": {"reasoning_efforts": ["low", "xhigh", 7]},
                    "gpt-5.4-mini": {"reasoning_efforts": []},
                    "gpt-5.3-codex": "not-a-record",
                }
            }
        )

        self.assertEqual(protocol_bridge._resolve_reasoning_efforts("gpt-5.4"), ["low", "xhigh"])
        self.assertIsNone(protocol_bridge._resolve_reasoning_efforts("gpt-5.4-mini"))
        self.assertIsNone(protocol_bridge._resolve_reasoning_efforts("gpt-5.3-codex"))

        cache.clear()
        cache.update({"data": ["not-a-dict"]})
        self.assertIsNone(protocol_bridge._resolve_reasoning_efforts("gpt-5.4"))

    def test_default_adaptive_thinking_resolver_handles_cache_and_offline_allowlist(self):
        cache = getattr(proxy, "_COPILOT_MODEL_CAPS_CACHE", None)
        self.assertIsInstance(cache, dict)

        self.assertFalse(protocol_bridge._default_adaptive_thinking_resolver(None))
        self.assertTrue(protocol_bridge._default_adaptive_thinking_resolver("anthropic/claude-sonnet-4.6"))
        self.assertTrue(protocol_bridge._default_adaptive_thinking_resolver("claude-opus-4.6"))
        self.assertFalse(protocol_bridge._default_adaptive_thinking_resolver("gpt-5.4"))

        cache.update(
            {
                "data": {
                    "claude-sonnet-4.6": {"adaptive_thinking_supported": True},
                    "claude-opus-4.6": {"adaptive_thinking_supported": False},
                    "claude-haiku-4.5": "not-a-record",
                }
            }
        )

        self.assertTrue(protocol_bridge._default_adaptive_thinking_resolver("claude-sonnet-4.6"))
        self.assertFalse(protocol_bridge._default_adaptive_thinking_resolver("claude-opus-4.6"))
        self.assertFalse(protocol_bridge._default_adaptive_thinking_resolver("claude-haiku-4.5"))

        cache.clear()
        cache.update({"data": ["not-a-dict"]})
        self.assertTrue(protocol_bridge._default_adaptive_thinking_resolver("claude-sonnet-4.6"))

    def test_bridge_execution_plan_upstream_path_defaults_to_chat(self):
        plan = BridgeExecutionPlan(
            strategy_name="dummy",
            inbound_protocol="responses",
            caller_protocol="responses",
            upstream_protocol="chat",
            header_kind="chat",
            requested_model="gpt-5.4",
            resolved_model="claude-sonnet-4.6",
            upstream_body={},
            stream=False,
        )

        self.assertEqual(plan.upstream_path, "/chat/completions")

    def test_planner_pins_exact_execution_plan_metadata_for_each_route(self):
        cases = [
            {
                "name": "responses_to_responses",
                "planner": ProtocolBridgePlanner(_RoutingConfigStub()),
                "protocol": "responses",
                "body": {"model": "gpt-5.4", "input": "hello", "stream": True},
                "expected": {
                    "strategy_name": "responses_to_responses",
                    "inbound_protocol": "responses",
                    "caller_protocol": "responses",
                    "upstream_protocol": "responses",
                    "header_kind": "responses",
                    "requested_model": "gpt-5.4",
                    "resolved_model": "gpt-5.4",
                    "stream": True,
                    "is_compact": False,
                    "upstream_path": "/responses",
                    "diagnostics": (),
                },
            },
            {
                "name": "responses_to_chat",
                "planner": ProtocolBridgePlanner(_RoutingConfigStub("claude-opus-4.6"), capability_resolver=lambda _model: False),
                "protocol": "responses",
                "body": {"model": "gpt-5.4", "input": "hello", "stream": False},
                "expected": {
                    "strategy_name": "responses_to_chat",
                    "inbound_protocol": "responses",
                    "caller_protocol": "responses",
                    "upstream_protocol": "chat",
                    "header_kind": "chat",
                    "requested_model": "gpt-5.4",
                    "resolved_model": "claude-opus-4.6",
                    "stream": False,
                    "is_compact": False,
                    "upstream_path": "/chat/completions",
                    "diagnostics": (),
                },
            },
            {
                "name": "responses_to_messages",
                "planner": ProtocolBridgePlanner(_RoutingConfigStub("claude-sonnet-4.6"), capability_resolver=lambda _model: True),
                "protocol": "responses",
                "body": {"model": "gpt-5.4", "input": "hello", "stream": False},
                "expected": {
                    "strategy_name": "responses_to_messages",
                    "inbound_protocol": "responses",
                    "caller_protocol": "responses",
                    "upstream_protocol": "messages",
                    "header_kind": "messages",
                    "requested_model": "gpt-5.4",
                    "resolved_model": "claude-sonnet-4.6",
                    "stream": False,
                    "is_compact": False,
                    "upstream_path": "/v1/messages",
                    "diagnostics": (),
                },
            },
            {
                "name": "messages_to_messages",
                "planner": ProtocolBridgePlanner(_RoutingConfigStub("claude-sonnet-4.6"), capability_resolver=lambda _model: True),
                "protocol": "messages",
                "body": {
                    "model": "claude-opus-4.6",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                    "stream": True,
                },
                "expected": {
                    "strategy_name": "messages_to_messages",
                    "inbound_protocol": "messages",
                    "caller_protocol": "anthropic",
                    "upstream_protocol": "messages",
                    "header_kind": "messages",
                    "requested_model": "claude-opus-4.6",
                    "resolved_model": "claude-sonnet-4.6",
                    "stream": True,
                    "is_compact": False,
                    "upstream_path": "/v1/messages",
                    "diagnostics": (),
                },
            },
            {
                "name": "messages_to_chat",
                "planner": ProtocolBridgePlanner(_RoutingConfigStub("claude-sonnet-4.6"), capability_resolver=lambda _model: False),
                "protocol": "messages",
                "body": {
                    "model": "claude-opus-4.6",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                    "stream": False,
                },
                "expected": {
                    "strategy_name": "messages_to_chat",
                    "inbound_protocol": "messages",
                    "caller_protocol": "anthropic",
                    "upstream_protocol": "chat",
                    "header_kind": "anthropic",
                    "requested_model": "claude-opus-4.6",
                    "resolved_model": "claude-sonnet-4.6",
                    "stream": False,
                    "is_compact": False,
                    "upstream_path": "/chat/completions",
                    "diagnostics": (),
                },
            },
            {
                "name": "messages_to_responses",
                "planner": ProtocolBridgePlanner(_RoutingConfigStub("gpt-5.4")),
                "protocol": "messages",
                "body": {
                    "model": "claude-opus-4.6",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                    "stream": True,
                },
                "expected": {
                    "strategy_name": "messages_to_responses",
                    "inbound_protocol": "messages",
                    "caller_protocol": "anthropic",
                    "upstream_protocol": "responses",
                    "header_kind": "responses",
                    "requested_model": "claude-opus-4.6",
                    "resolved_model": "gpt-5.4",
                    "stream": True,
                    "is_compact": False,
                    "upstream_path": "/responses",
                    "diagnostics": (),
                },
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                plan = proxy.asyncio.run(
                    case["planner"].plan(
                        case["protocol"],
                        case["body"],
                        api_base="https://example.invalid",
                        api_key="test-key",
                    )
                )
                self.assertEqual(
                    {
                        "strategy_name": plan.strategy_name,
                        "inbound_protocol": plan.inbound_protocol,
                        "caller_protocol": plan.caller_protocol,
                        "upstream_protocol": plan.upstream_protocol,
                        "header_kind": plan.header_kind,
                        "requested_model": plan.requested_model,
                        "resolved_model": plan.resolved_model,
                        "stream": plan.stream,
                        "is_compact": plan.is_compact,
                        "upstream_path": plan.upstream_path,
                        "diagnostics": plan.diagnostics,
                    },
                    case["expected"],
                )

    def test_base_strategy_matches_tuple_and_scalar_families(self):
        class DummyStrategy(ProtocolBridgeStrategy):
            strategy_name = "dummy"
            inbound_protocol = "responses"
            target_family = ("claude", "gemini")
            upstream_protocol = "chat"
            header_kind = "chat"
            caller_protocol = "responses"

            async def build_plan(self, body, *, requested_model, resolved_model, api_base, api_key, is_compact=False):
                del body, requested_model, resolved_model, api_base, api_key, is_compact
                raise NotImplementedError

        strategy = DummyStrategy()

        self.assertTrue(strategy.matches("responses", "claude"))
        self.assertTrue(strategy.matches("responses", "gemini"))
        self.assertFalse(strategy.matches("messages", "claude"))
        self.assertFalse(strategy.matches("responses", "codex"))

    def test_native_responses_strategy_maps_reasoning_effort_before_upstream(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "stream": True,
            "reasoning": {"effort": "max", "summary": "auto"},
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "responses_to_responses")
        self.assertTrue(plan.stream)
        self.assertEqual(plan.upstream_body["reasoning"], {"effort": "xhigh", "summary": "auto"})

    def test_native_responses_strategy_leaves_unmapped_reasoning_effort_unchanged(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "reasoning": {"effort": "unrecognized", "summary": "auto"},
        }

        plan = proxy.asyncio.run(
            planner.plan("responses", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.upstream_body["reasoning"], {"effort": "unrecognized", "summary": "auto"})

    def test_abstract_strategy_build_plan_raises_when_called_directly(self):
        async def call_base_build_plan():
            return await ProtocolBridgeStrategy.build_plan(
                object(),
                {},
                requested_model=None,
                resolved_model=None,
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=False,
            )

        with self.assertRaises(NotImplementedError):
            proxy.asyncio.run(call_base_build_plan())

    def test_planner_rejects_unknown_mapped_model_family(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("unknown-family-model"))

        with self.assertRaisesRegex(ValueError, "Unsupported mapped model family"):
            proxy.asyncio.run(
                planner.plan(
                    "responses",
                    {"model": "gpt-5.4", "input": "hello"},
                    api_base="https://example.invalid",
                    api_key="test-key",
                )
            )

    def test_strategy_for_rejects_unsupported_protocol_family_pair(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub())

        with self.assertRaisesRegex(ValueError, "No bridge strategy for messages -> unsupported"):
            planner._strategy_for("messages", "unsupported", "gpt-5.4")

    def test_translation_strategies_inject_resolved_model_before_translation(self):
        responses_body = {"model": "gpt-5.3-codex", "input": "hello"}
        messages_body = {
            "model": "claude-opus-4.6",
            "messages": [{"role": "user", "content": "hello"}],
        }

        with mock.patch.object(
            protocol_bridge.format_translation,
            "responses_request_to_chat",
            side_effect=lambda body: {"model": body.get("model"), "stream": body.get("stream", False)},
        ) as responses_to_chat:
            plan = proxy.asyncio.run(
                protocol_bridge.ResponsesToChatStrategy().build_plan(
                    responses_body,
                    requested_model="gpt-5.3-codex",
                    resolved_model="claude-sonnet-4.6",
                    api_base="https://example.invalid",
                    api_key="test-key",
                    is_compact=False,
                )
            )

        self.assertEqual(responses_to_chat.call_args.args[0]["model"], "claude-sonnet-4.6")
        self.assertEqual(plan.upstream_body["model"], "claude-sonnet-4.6")
        self.assertFalse(plan.is_compact)

        with mock.patch.object(
            protocol_bridge.format_translation,
            "anthropic_request_to_chat",
            new=mock.AsyncMock(return_value={"model": "claude-sonnet-4.6", "stream": False}),
        ) as messages_to_chat:
            plan = proxy.asyncio.run(
                protocol_bridge.MessagesToChatStrategy().build_plan(
                    messages_body,
                    requested_model="claude-opus-4.6",
                    resolved_model="claude-sonnet-4.6",
                    api_base="https://api-base.invalid",
                    api_key="api-key",
                    is_compact=True,
                )
            )

        call_body, call_api_base, call_api_key = messages_to_chat.call_args.args
        self.assertEqual(call_body["model"], "claude-sonnet-4.6")
        self.assertEqual(call_api_base, "https://api-base.invalid")
        self.assertEqual(call_api_key, "api-key")
        self.assertTrue(plan.is_compact)

        with mock.patch.object(
            protocol_bridge.format_translation,
            "responses_request_to_anthropic_messages",
            side_effect=lambda body: {
                "model": body.get("model"),
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as responses_to_messages:
            plan = proxy.asyncio.run(
                protocol_bridge.ResponsesToMessagesStrategy().build_plan(
                    responses_body,
                    requested_model="gpt-5.3-codex",
                    resolved_model="claude-sonnet-4.6",
                    api_base="https://example.invalid",
                    api_key="test-key",
                    is_compact=False,
                )
            )

        self.assertEqual(responses_to_messages.call_args.args[0]["model"], "claude-sonnet-4.6")
        self.assertEqual(plan.upstream_body["model"], "claude-sonnet-4.6")
        self.assertFalse(plan.is_compact)

    def test_messages_to_messages_defaults_absent_stream_to_false(self):
        body = {
            "model": "claude-opus-4.6",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }

        plan = proxy.asyncio.run(
            protocol_bridge.MessagesToMessagesStrategy().build_plan(
                body,
                requested_model="claude-opus-4.6",
                resolved_model="claude-sonnet-4.6",
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=False,
            )
        )

        self.assertFalse(plan.stream)
        self.assertFalse(plan.is_compact)

        compact_plan = proxy.asyncio.run(
            protocol_bridge.MessagesToMessagesStrategy().build_plan(
                body,
                requested_model="claude-opus-4.6",
                resolved_model="claude-sonnet-4.6",
                api_base="https://example.invalid",
                api_key="test-key",
                is_compact=True,
            )
        )

        self.assertTrue(compact_plan.is_compact)

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

    def test_planner_uses_cache_affinity_fields_locally_for_native_responses(self):
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
        self.assertNotIn("sessionId", plan.upstream_body)
        # Cache lineage stays on the upstream body so the Copilot prefix cache
        # can match across turns.
        self.assertEqual(plan.upstream_body["prompt_cache_key"], "cache-123")
        self.assertEqual(plan.upstream_body["previous_response_id"], "resp_prev")
        self.assertEqual(plan.upstream_body["input"], "hello")
        self.assertEqual(
            plan.diagnostics[0]["fields"],
            ["service_tier", "sessionId"],
        )

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

    def test_planner_uses_native_messages_to_responses_cache_shape(self):
        planner = ProtocolBridgePlanner(_RoutingConfigStub("gpt-5.4"))
        body = {
            "model": "claude-opus-4.6",
            "system": "system prompt",
            "metadata": {"user_id": '{"session_id":"claude-cache-session"}'},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "stream": False,
        }

        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="test-key")
        )

        self.assertEqual(plan.strategy_name, "messages_to_responses")
        self.assertNotIn("prompt_cache_key", plan.upstream_body)
        self.assertNotIn("instructions", plan.upstream_body)
        self.assertEqual(plan.upstream_body["text"], {"format": {"type": "text"}, "verbosity": "low"})
        self.assertEqual(plan.upstream_body["input"][0]["role"], "developer")
        self.assertEqual(plan.upstream_body["input"][0]["content"][0]["text"], "system prompt")
        self.assertEqual(plan.upstream_body["input"][1]["role"], "user")
        self.assertNotIn("metadata", plan.upstream_body)

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

    def test_messages_to_messages_preserves_native_cache_markers_without_expansion(self):
        planner = self._planner("claude-sonnet-4.6", supports=True)
        body = {
            "model": "claude-sonnet-4.6",
            "system": "system prompt",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "first"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tool_1", "name": "Read", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool_1", "content": "ok"},
                        {
                            "type": "text",
                            "text": "<system-reminder>cache here</system-reminder>",
                            "cache_control": {"type": "ephemeral", "scope": "session"},
                        },
                    ],
                },
            ],
            "tools": [{"name": "Read", "description": "Read", "input_schema": {"type": "object"}}],
        }

        plan = proxy.asyncio.run(
            planner.plan("messages", body, api_base="https://example.invalid", api_key="k")
        )

        self.assertEqual(plan.strategy_name, "messages_to_messages")
        self.assertEqual(plan.upstream_body["system"], "system prompt")
        self.assertNotIn("cache_control", plan.upstream_body["tools"][0])
        self.assertNotIn("cache_control", plan.upstream_body["messages"][-2]["content"][-1])
        merged_result = plan.upstream_body["messages"][-1]["content"][0]
        self.assertEqual(merged_result["type"], "tool_result")
        self.assertEqual(merged_result["cache_control"], {"type": "ephemeral"})

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
