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


if __name__ == "__main__":
    unittest.main()
