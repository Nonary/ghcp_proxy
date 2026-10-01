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


class Gpt6PricingTests(unittest.TestCase):
    def test_gpt_6_1_sol_uses_official_standard_rates(self):
        pricing = MODEL_PRICING["gpt-6.1-sol"]
        self.assertEqual(pricing["input_per_million"], 2.00)
        self.assertEqual(pricing["cached_input_per_million"], 0.10)
        self.assertEqual(pricing["cache_write_per_million"], 2.50)
        self.assertEqual(pricing["output_per_million"], 10.00)
        self.assertNotIn("long_context_threshold", pricing)

        usage = {
            "input_tokens": 500_000,
            "cached_input_tokens": 200_000,
            "cache_creation_input_tokens": 100_000,
            "output_tokens": 100_000,
        }
        breakdown = _usage_event_cost_breakdown("gpt-6.1-sol", usage)
        self.assertEqual(breakdown["input_fresh"], 0.8)
        self.assertEqual(breakdown["cached_input"], 0.02)
        self.assertEqual(breakdown["cache_creation"], 0.25)
        self.assertEqual(breakdown["output"], 1.0)

    def test_gpt_6_luna_pricing_is_already_configured(self):
        pricing = MODEL_PRICING["gpt-6-luna"]
        self.assertEqual(pricing["input_per_million"], 0.10)
        self.assertEqual(pricing["cached_input_per_million"], 0.01)
        self.assertEqual(pricing["cache_write_per_million"], 0.125)
        self.assertEqual(pricing["output_per_million"], 0.50)
        self.assertEqual(pricing["long_context_input_per_million"], 0.20)
        self.assertEqual(pricing["long_context_cached_input_per_million"], 0.02)
        self.assertEqual(pricing["long_context_cache_write_per_million"], 0.25)
        self.assertEqual(pricing["long_context_output_per_million"], 0.75)


if __name__ == "__main__":
    unittest.main()
