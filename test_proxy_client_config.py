import unittest

from proxy_client_config import ProxyClientConfigService


class ReasoningLevelTests(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(ProxyClientConfigService)

    def _effort_names(self, model_name, raw_efforts):
        levels, _ = self.service._resolve_reasoning_levels(
            "gpt", raw_efforts, model_name=model_name
        )
        return [level["effort"] for level in levels]

    def test_excel_models_never_expose_max(self):
        raw_efforts = ["low", "medium", "high", "xhigh", "max"]
        for model_name in (
            "gpt-5.6-luna-excel",
            "gpt-5.6-terra-excel",
            "gpt-5.6-sol-excel",
        ):
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    self._effort_names(model_name, raw_efforts),
                    ["low", "medium", "high", "xhigh"],
                )

    def test_non_excel_gpt_56_still_exposes_max(self):
        self.assertEqual(
            self._effort_names(
                "gpt-5.6-sol", ["low", "medium", "high", "xhigh"]
            ),
            ["low", "medium", "high", "xhigh", "max"],
        )

    def test_gpt_6_luna_and_gpt_61_sol_expose_upstream_max_effort(self):
        raw_efforts = ["low", "medium", "high", "max"]
        for model_name in ("gpt-6-luna", "gpt-6.1-sol"):
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    self._effort_names(model_name, raw_efforts),
                    raw_efforts,
                )

    def test_gpt_6_models_without_max_support_do_not_expose_max(self):
        self.assertEqual(
            self._effort_names("gpt-6-sol", ["low", "medium", "high", "max"]),
            ["low", "medium", "high"],
        )

    def test_empty_capability_fallback_does_not_expose_excel_aliases(self):
        model_names = self.service._sorted_catalog_model_names(set(), {})

        self.assertFalse(any(name.endswith("-excel") for name in model_names))


if __name__ == "__main__":
    unittest.main()
