import os
import tempfile
import unittest

import excel_upstream
from model_routing_config import ModelRoutingConfig, ModelRoutingConfigService


class ExcelRoutingChoicesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = ModelRoutingConfig(config_file=os.path.join(self.directory.name, "routing.json"))

    def excel_choices(self, service):
        return {
            row["model"] for row in service.config_payload()["available_models"]
            if row["model"].endswith("-excel")
        }

    def test_only_discovered_aliases_are_offered(self):
        service = ModelRoutingConfigService(
            self.config,
            model_capabilities_provider=lambda: excel_upstream.merge_local_model_capabilities({}, excel_models={
                "gpt-5.6-sol": {},
                "gpt-6-astra": {"model_picker_enabled": False},
            }),
        )
        self.assertEqual(self.excel_choices(service), {"gpt-5.6-sol-excel"})

    def test_empty_capabilities_do_not_invent_routing_choices(self):
        service = ModelRoutingConfigService(self.config, model_capabilities_provider=lambda: {})
        self.assertEqual(self.excel_choices(service), set())
        self.assertEqual(service.config_payload()["available_models"], [])

    def test_missing_provider_does_not_offer_excel_aliases(self):
        self.assertEqual(self.excel_choices(ModelRoutingConfigService(self.config)), set())

    def test_failed_discovery_does_not_offer_excel_aliases(self):
        def unavailable():
            raise RuntimeError("discovery failed")
        service = ModelRoutingConfigService(self.config, model_capabilities_provider=unavailable)
        self.assertEqual(self.excel_choices(service), set())

    def test_choices_refresh_when_availability_changes(self):
        capabilities = {"gpt-5.6-sol-excel": {"model_picker_enabled": True}}
        service = ModelRoutingConfigService(self.config, model_capabilities_provider=lambda: capabilities)
        self.assertEqual(self.excel_choices(service), {"gpt-5.6-sol-excel"})
        capabilities["gpt-5.6-sol-excel"]["model_picker_enabled"] = False
        self.assertEqual(self.excel_choices(service), set())

    def test_dynamic_excel_choices_are_offered_and_saved_without_a_static_registry(self):
        service = ModelRoutingConfigService(
            self.config, model_capabilities_provider=lambda: {"gpt-7-new-model-excel": {}},
        )
        self.assertEqual(self.excel_choices(service), {"gpt-7-new-model-excel"})
        service.save_settings({
            "enabled": True, "mappings": [{"source_model": "gpt-5.6-sol", "target_model": "gpt-7-new-model-excel"}],
        })
        reloaded = ModelRoutingConfigService(self.config)
        self.assertEqual(reloaded.load_settings()["mappings"][0]["target_model"], "gpt-7-new-model-excel")

    def test_discovery_failure_does_not_delete_saved_mappings(self):
        service = ModelRoutingConfigService(self.config, model_capabilities_provider=lambda: {})
        payload = service.save_settings({
            "enabled": True,
            "mappings": [{"source_model": "gpt-5.6-sol", "target_model": "gpt-6-astra-excel"}],
        })
        self.assertEqual(payload["mappings"][0]["target_model"], "gpt-6-astra-excel")
        self.assertEqual(self.excel_choices(service), set())
        self.assertEqual(service.load_settings()["mappings"], payload["mappings"])


if __name__ == "__main__":
    unittest.main()
