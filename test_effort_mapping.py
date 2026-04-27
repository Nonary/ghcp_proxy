import unittest
from unittest import mock

import effort_mapping


class EffortMappingTests(unittest.TestCase):
    def test_canonical_passthrough(self):
        for effort in ("low", "medium", "high"):
            self.assertEqual(
                effort_mapping.map_effort_for_model("gpt-5.4", effort),
                effort,
            )

    def test_max_maps_to_xhigh_for_gpt_models(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("gpt-5.4", "max"),
            "xhigh",
        )

    def test_openai_prefixed_gpt_models_use_gpt_effort_mapping(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("openai/gpt-5.4", "max"),
            "xhigh",
        )

    def test_model_normalization_contract(self):
        self.assertEqual(effort_mapping._normalize_model(None), "")
        self.assertEqual(effort_mapping._normalize_model("  OpenAI/GPT-5.4/preview  "), "gpt-5.4/preview")
        self.assertEqual(effort_mapping._normalize_model("  anthropic/Claude-Sonnet-4.6/beta  "), "claude-sonnet-4.6/beta")

    def test_xhigh_is_preserved_for_gpt_models(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("gpt-5.4", "xhigh"),
            "xhigh",
        )

    def test_xhigh_alias_maps_to_max_for_claude_models(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("claude-sonnet-4.6", "xhigh"),
            "max",
        )
        self.assertEqual(
            effort_mapping.map_effort_for_model("claude-sonnet-4.6", "x-high"),
            "max",
        )

    def test_minimal_and_none_aliases_map_to_low(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("gpt-5.4", "minimal"),
            "low",
        )
        self.assertEqual(
            effort_mapping.map_effort_for_model("gpt-5.4", "none"),
            "low",
        )

    def test_opus_47_clamps_to_medium(self):
        for effort in ("low", "medium", "high", "max", "xhigh"):
            self.assertEqual(
                effort_mapping.map_effort_for_model("claude-opus-4.7", effort),
                "medium",
            )

    def test_opus_47_with_anthropic_prefix(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("anthropic/claude-opus-4.7", "high"),
            "medium",
        )

    def test_haiku_45_omits_reasoning_effort(self):
        for effort in ("low", "medium", "high", "max", "xhigh"):
            self.assertIsNone(
                effort_mapping.map_effort_for_model("claude-haiku-4.5", effort)
            )

    def test_haiku_45_with_anthropic_prefix_omits_reasoning_effort(self):
        self.assertIsNone(
            effort_mapping.map_effort_for_model("anthropic/claude-haiku-4.5", "high")
        )

    def test_other_claude_sku_not_clamped(self):
        self.assertEqual(
            effort_mapping.map_effort_for_model("claude-opus-4.6", "high"),
            "high",
        )
        self.assertEqual(
            effort_mapping.map_effort_for_model("claude-sonnet-4.6", "max"),
            "max",
        )

    def test_unknown_effort_returns_none(self):
        self.assertIsNone(effort_mapping.map_effort_for_model("gpt-5.4", None))
        self.assertIsNone(effort_mapping.map_effort_for_model("gpt-5.4", ""))
        self.assertIsNone(effort_mapping.map_effort_for_model("gpt-5.4", "bogus"))
        self.assertIsNone(effort_mapping.map_effort_for_model("claude-opus-4.7", "bogus"))

    def test_base_strategy_methods_are_abstract_sentinels(self):
        class Concrete(effort_mapping.ModelEffortStrategy):
            def matches(self, normalized_model):
                return super().matches(normalized_model)

            def map(self, canonical_effort):
                return super().map(canonical_effort)

        strategy = Concrete()
        with self.assertRaises(NotImplementedError):
            strategy.matches("gpt-5.4")
        with self.assertRaises(NotImplementedError):
            strategy.map("high")

    def test_strategy_for_falls_back_to_last_strategy_if_none_match(self):
        class NoMatch(effort_mapping.ModelEffortStrategy):
            def matches(self, normalized_model):
                return False

            def map(self, canonical_effort):
                return f"mapped:{canonical_effort}"

        fallback = NoMatch()
        with mock.patch.object(effort_mapping, "_STRATEGIES", [fallback]):
            self.assertIs(effort_mapping._strategy_for("gpt-5.4"), fallback)

    def test_passthrough_strategy_matches_any_model(self):
        self.assertTrue(effort_mapping.PassthroughStrategy().matches("unknown-model"))


if __name__ == "__main__":
    unittest.main()
