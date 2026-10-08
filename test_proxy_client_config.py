import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import excel_upstream

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
            "gpt-6-sol-excel",
            "gpt-6-luna-excel",
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

    def test_empty_discovery_does_not_expose_pricing_metadata_as_models(self):
        model_names = self.service._sorted_catalog_model_names(set(), {})

        self.assertEqual(model_names, [])

    def test_empty_discovered_catalog_remains_a_valid_managed_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "models.json")
            with open(path, "w") as stream:
                json.dump({"models": []}, stream)
            self.service._config = SimpleNamespace(codex_model_catalog_file=path)
            self.assertTrue(self.service._codex_model_catalog_is_valid())

    def test_generated_catalog_uses_the_independent_backend_union(self):
        self.service._config = SimpleNamespace(codex_model_context_window=272_000, codex_model_auto_compact_token_limit=180_000)
        self.service._model_capabilities_provider = lambda: excel_upstream.merge_local_model_capabilities(
            {"gpt-6.1-sol": {}}, excel_models={"gpt-6-sol": {}},
        )
        self.service._model_routing_settings_provider = lambda: {}
        slugs = {row["slug"] for row in self.service._build_codex_model_catalog_payload()["models"]}
        self.assertEqual(slugs, {"gpt-6.1-sol", "gpt-6-sol-excel"})
        self.assertEqual(slugs, set(self.service.codex_model_catalog_model_names()))

    def test_generated_catalog_selects_v2_for_fresh_threads_and_children(self):
        self.service._config = SimpleNamespace(
            codex_model_context_window=272_000,
            codex_model_auto_compact_token_limit=180_000,
        )
        self.service._model_capabilities_provider = lambda: {
            "gpt-6.1-sol": {},
            "gpt-6-luna": {},
            "gpt-6-sol-excel": {},
        }
        self.service._model_routing_settings_provider = lambda: {
            "enabled": True,
            "mappings": [
                {"source_model": "gpt-5.3-codex", "target_model": "gpt-6.1-sol"},
            ],
        }

        models = self.service._build_codex_model_catalog_payload()["models"]

        self.assertEqual(
            {model["slug"] for model in models},
            {"gpt-6.1-sol", "gpt-6-luna", "gpt-6-sol-excel", "gpt-5.3-codex"},
        )
        for model in models:
            with self.subTest(model=model["slug"]):
                self.assertEqual(model.get("multi_agent_version"), "v2")

    def test_refresh_adds_v2_metadata_without_rewriting_agent_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = os.path.join(directory, "models.json")
            primary_path = os.path.join(directory, "config.toml")
            primary = '[agents]\nenabled = false\nmax_threads = 3\n'
            with open(primary_path, "w") as stream:
                stream.write(primary)
            with open(catalog_path, "w") as stream:
                json.dump({"models": [{"slug": "gpt-6.1-sol"}]}, stream)
            self.service._config = SimpleNamespace(
                codex_model_catalog_file=catalog_path,
                codex_primary_config_file=primary_path,
                codex_config_dir=directory,
                codex_model_context_window=272_000,
                codex_model_auto_compact_token_limit=180_000,
            )
            self.service._model_capabilities_provider = lambda: {"gpt-6.1-sol": {}}
            self.service._model_routing_settings_provider = lambda: {}

            self.assertTrue(self.service.refresh_codex_model_catalog())

            with open(catalog_path) as stream:
                self.assertEqual(json.load(stream)["models"][0]["multi_agent_version"], "v2")
            with open(primary_path) as stream:
                self.assertEqual(stream.read(), primary)

    def test_routing_can_target_a_discovered_excel_model_without_static_pricing(self):
        self.service._model_capabilities_provider = lambda: {"gpt-7-new-model-excel": {}}
        self.service._model_routing_settings_provider = lambda: {
            "enabled": True, "mappings": [{"source_model": "gpt-5.3-codex", "target_model": "gpt-7-new-model-excel"}],
        }
        self.assertEqual(set(self.service.codex_model_catalog_model_names()), {"gpt-5.3-codex", "gpt-7-new-model-excel"})

    def test_reenabling_proxy_refreshes_a_stale_saved_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "models.json")
            with open(path, "w") as stream:
                json.dump({"models": [{"slug": "gpt-6.1-sol-excel"}]}, stream)
            self.service._config = SimpleNamespace(
                codex_model_catalog_file=path, codex_config_dir=directory,
                codex_managed_config_file=os.path.join(directory, "config.toml"),
                codex_model_context_window=272_000, codex_model_auto_compact_token_limit=180_000,
            )
            self.service._model_capabilities_provider = lambda: {"gpt-6-sol-excel": {}}
            self.service._model_routing_settings_provider = lambda: {}
            with mock.patch.object(self.service, "codex_proxy_status", return_value={"configured": True}), \
                 mock.patch.object(self.service, "_latest_backup_path", return_value=None):
                self.service.write_codex_proxy_config()
            with open(path) as stream:
                self.assertEqual([row["slug"] for row in json.load(stream)["models"]], ["gpt-6-sol-excel"])

    def test_remapping_does_not_expose_an_unavailable_excel_alias(self):
        model_names = self.service._sorted_catalog_model_names(
            {"gpt-5.6-sol"},
            {"gpt-6-astra-excel": "gpt-5.6-sol", "gpt-5.3-codex": "gpt-5.6-sol"},
        )
        self.assertIn("gpt-5.3-codex", model_names)
        self.assertNotIn("gpt-6-astra-excel", model_names)

    def test_catalog_model_names_match_entitlement_visibility_and_routing(self):
        self.service._model_capabilities_provider = lambda: {
            "gpt-6-sol-excel": {"model_picker_enabled": True},
            "gpt-6-luna-excel": {"model_picker_enabled": True},
            "gpt-6-luna": {"model_picker_enabled": True},
            "gpt-5.6-sol": {"model_picker_enabled": True},
            "gpt-6-sol": {"model_picker_enabled": False},
        }
        self.service._model_routing_settings_provider = lambda: {
            "enabled": True,
            "mappings": [
                {"source_model": "gpt-5.3-codex", "target_model": "gpt-5.6-sol"}
            ],
        }

        model_names = self.service.codex_model_catalog_model_names()

        self.assertIn("gpt-6-sol-excel", model_names)
        self.assertIn("gpt-6-luna-excel", model_names)
        self.assertIn("gpt-6-luna", model_names)
        self.assertIn("gpt-5.6-sol", model_names)
        self.assertIn("gpt-5.3-codex", model_names)
        self.assertNotIn("gpt-6-sol", model_names)
        self.assertNotIn("gpt-5.4", model_names)


class SelectedExcelModelMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.primary_path = os.path.join(self.directory.name, "config.toml")
        self.catalog_path = os.path.join(self.directory.name, "models.json")
        self.service = object.__new__(ProxyClientConfigService)
        self.service._config = SimpleNamespace(
            codex_primary_config_file=self.primary_path,
            codex_model_catalog_file=self.catalog_path,
            codex_config_dir=self.directory.name,
            codex_model_context_window=272_000,
            codex_model_auto_compact_token_limit=180_000,
        )
        self.service._model_capabilities_provider = lambda: {"gpt-6-sol-excel": {}}
        self.service._model_routing_settings_provider = lambda: {}
        self.catalog = {"models": [{"slug": "gpt-6-sol-excel"}]}

    def write_primary(self, model="gpt-6.1-sol-excel", provider="custom", base_url="http://127.0.0.1:8000/v1"):
        content = (
            f'model = "{model}" # preserve this comment\n'
            f'model_provider = "{provider}"\n'
            f'model_catalog_json = {json.dumps(self.catalog_path)}\n'
            'model_reasoning_effort = "high"\n'
            '[model_providers.custom]\n'
            f'base_url = "{base_url}"\n'
            '[profiles.other]\n'
            'model = "gpt-6.1-sol-excel"\n'
        )
        with open(self.primary_path, "w") as stream:
            stream.write(content)
        return content

    def read_primary(self):
        with open(self.primary_path) as stream:
            return stream.read()

    def test_only_the_stale_top_level_selection_changes_and_is_backed_up(self):
        original = self.write_primary()
        self.assertTrue(self.service._repair_legacy_codex_excel_selection(self.catalog))
        self.assertEqual(self.read_primary(), original.replace('model = "gpt-6.1-sol-excel"', 'model = "gpt-6-sol-excel"', 1))
        backups = [name for name in os.listdir(self.directory.name) if ".model-selection.bak." in name]
        self.assertEqual(len(backups), 1)
        with open(os.path.join(self.directory.name, backups[0])) as stream:
            self.assertEqual(stream.read(), original)
        self.assertIsNone(self.service._latest_backup_path(self.primary_path))

    def test_migration_requires_positive_replacement_availability(self):
        original = self.write_primary()
        self.assertFalse(self.service._repair_legacy_codex_excel_selection({"models": []}))
        self.assertEqual(self.read_primary(), original)

    def test_a_now_available_61_excel_model_is_not_migrated(self):
        original = self.write_primary()
        catalog = {"models": [{"slug": "gpt-6-sol-excel"}, {"slug": "gpt-6.1-sol-excel"}]}
        self.assertFalse(self.service._repair_legacy_codex_excel_selection(catalog))
        self.assertEqual(self.read_primary(), original)

    def test_a_valid_copilot_selection_is_not_migrated(self):
        original = self.write_primary(model="gpt-6.1-sol")
        self.assertFalse(self.service._repair_legacy_codex_excel_selection(self.catalog))
        self.assertEqual(self.read_primary(), original)

    def test_native_or_other_provider_config_is_not_changed(self):
        for settings in ({"provider": "openai"}, {"base_url": "https://other.example/v1"}):
            with self.subTest(settings=settings):
                original = self.write_primary(**settings)
                self.assertFalse(self.service._repair_legacy_codex_excel_selection(self.catalog))
                self.assertEqual(self.read_primary(), original)

    def test_catalog_refresh_repairs_the_default_once(self):
        self.write_primary()
        with open(self.catalog_path, "w") as stream:
            json.dump({"models": [{"slug": "gpt-6.1-sol-excel"}]}, stream)
        self.assertTrue(self.service.refresh_codex_model_catalog())
        self.assertEqual(self.service._parse_toml_values(self.read_primary())["model"], "gpt-6-sol-excel")
        self.assertTrue(self.service.refresh_codex_model_catalog())
        self.assertEqual(len([name for name in os.listdir(self.directory.name) if ".model-selection.bak." in name]), 1)

    def test_invalid_toml_is_preserved(self):
        with open(self.primary_path, "w") as stream:
            stream.write('model = "unterminated\n')
        self.assertFalse(self.service._repair_legacy_codex_excel_selection(self.catalog))
        self.assertEqual(self.read_primary(), 'model = "unterminated\n')


if __name__ == "__main__":
    unittest.main()
