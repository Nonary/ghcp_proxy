import unittest

from constants import MODEL_PRICING
from util import _usage_event_cost_breakdown, normalize_usage_payload


class TerraPricingTests(unittest.TestCase):
    def test_cached_input_without_explicit_fresh_tokens_is_not_double_billed(self):
        usage = normalize_usage_payload(
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 800,
                "output_tokens": 100,
            }
        )

        self.assertEqual(usage["fresh_input_tokens"], 200)
        breakdown = _usage_event_cost_breakdown("gpt-5.6-terra-excel", usage)
        self.assertEqual(breakdown["input_fresh"], 0.0005)
        self.assertEqual(breakdown["cached_input"], 0.0002)
        self.assertEqual(breakdown["output"], 0.0015)

    def test_terra_has_standard_and_long_context_cache_write_rates(self):
        for model_name in ("gpt-5.6-terra", "gpt-5.6-terra-excel"):
            pricing = MODEL_PRICING[model_name]
            self.assertEqual(pricing["input_per_million"], 2.50)
            self.assertEqual(pricing["cached_input_per_million"], 0.25)
            self.assertEqual(pricing["cache_write_per_million"], 3.125)
            self.assertEqual(pricing["output_per_million"], 15.00)
            self.assertEqual(pricing["long_context_input_per_million"], 5.00)
            self.assertEqual(pricing["long_context_cached_input_per_million"], 0.50)
            self.assertEqual(pricing["long_context_cache_write_per_million"], 6.25)
            self.assertEqual(pricing["long_context_output_per_million"], 22.50)

    def test_terra_cache_creation_uses_cache_write_rate(self):
        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 1_000,
            "output_tokens": 0,
        }
        self.assertEqual(
            _usage_event_cost_breakdown("gpt-5.6-terra", usage)["cache_creation"],
            0.003125,
        )

    def test_terra_long_context_cache_creation_uses_long_context_rate(self):
        usage = {
            "input_tokens": 272_001,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
            "output_tokens": 0,
        }
        self.assertEqual(
            _usage_event_cost_breakdown("gpt-5.6-terra", usage)["cache_creation"],
            6.25,
        )


if __name__ == "__main__":
    unittest.main()
