import unittest

import proxy
from protocol_bridge import ProtocolBridgePlanner


class _RoutingConfigStub:
    def __init__(self, target_model=None):
        self.target_model = target_model

    def resolve_target_model(self, requested_model):
        del requested_model
        return self.target_model


class ProtocolBridgePlannerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
