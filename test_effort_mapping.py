import unittest

import effort_mapping


class EffortMappingTests(unittest.TestCase):
    def test_canonical_passthrough(self):
        for effort in ("low", "medium", "high", "max"):
            self.assertEqual(
                effort_mapping.map_effort_for_model("gpt-5.4", effort),
                effort,
            )

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


if __name__ == "__main__":
    unittest.main()
