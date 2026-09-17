import unittest

from dashboard import _prepare_usage_event
from util import _usage_event_model_name


class UsageModelIdentityTests(unittest.TestCase):
    def test_excel_alias_wins_over_base_response_model(self):
        for base_model in (
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6-sol",
        ):
            excel_model = f"{base_model}-excel"
            event = {
                "requested_model": excel_model,
                "resolved_model": excel_model,
                "response_model": base_model,
                "finished_at": "2026-09-17T12:00:00Z",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            }

            with self.subTest(excel_model=excel_model):
                self.assertEqual(_usage_event_model_name(event), excel_model)
                self.assertEqual(_prepare_usage_event(event)["model_name"], excel_model)

    def test_non_credit_model_keeps_response_model_precedence(self):
        event = {
            "requested_model": "gpt-5.4",
            "resolved_model": "gpt-5.4",
            "response_model": "gpt-5.5",
        }

        self.assertEqual(_usage_event_model_name(event), "gpt-5.5")


if __name__ == "__main__":
    unittest.main()
