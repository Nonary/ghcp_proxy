import unittest
import json
import shutil
import tomllib
from pathlib import Path
from uuid import uuid4

from constants import (
    CLAUDE_MAX_CONTEXT_TOKENS,
    CLAUDE_MAX_OUTPUT_TOKENS,
    CLAUDE_PROXY_SETTINGS,
    CODEX_MANAGED_CONFIG_FILE,
    CODEX_PROXY_CONFIG,
    CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT,
    CODEX_PROXY_MODEL_CATALOG_FILE,
    CODEX_PROXY_MODEL_CONTEXT_WINDOW,
)
from proxy_client_config import ProxyClientConfig, ProxyClientConfigService


class ClientConfigTests(unittest.TestCase):
    def _make_temp_dir(self, prefix: str) -> Path:
        path = Path.cwd() / f"{prefix}{uuid4().hex}"

        def _cleanup():
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass

        self.addCleanup(_cleanup)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_client_proxy_service(
        self,
        *,
        codex_managed_config_file: str | None = None,
        codex_model_catalog_file: str | None = None,
        claude_settings_file: str | None = None,
        model_capabilities_provider=None,
        model_routing_settings_provider=None,
    ) -> ProxyClientConfigService:
        return ProxyClientConfigService(
            ProxyClientConfig(
                codex_managed_config_file=codex_managed_config_file or CODEX_MANAGED_CONFIG_FILE,
                codex_model_catalog_file=codex_model_catalog_file or CODEX_PROXY_MODEL_CATALOG_FILE,
                codex_proxy_config=CODEX_PROXY_CONFIG,
                codex_model_context_window=CODEX_PROXY_MODEL_CONTEXT_WINDOW,
                codex_model_auto_compact_token_limit=CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT,
                claude_settings_file=claude_settings_file or "",
                claude_proxy_settings=CLAUDE_PROXY_SETTINGS,
                claude_max_context_tokens=CLAUDE_MAX_CONTEXT_TOKENS,
                claude_max_output_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
            ),
            model_capabilities_provider=model_capabilities_provider,
            model_routing_settings_provider=model_routing_settings_provider,
        )

    def test_enabling_codex_proxy_is_idempotent_when_already_enabled(self):
        temp_dir = self._make_temp_dir("codex-config-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
        )
        service.write_codex_proxy_config()

        status = service.write_codex_proxy_config()

        backups = list(managed_config_path.parent.glob(f"{managed_config_path.name}.ghcp-proxy.bak.*"))
        self.assertTrue(status["configured"])
        self.assertEqual(status["status_message"], "proxy already enabled")
        self.assertEqual(backups, [])

    def test_disabling_codex_proxy_is_idempotent_when_already_disabled(self):
        temp_dir = self._make_temp_dir("codex-disabled-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
        )
        status = service.disable_codex_proxy_config()

        self.assertFalse(status["configured"])
        self.assertEqual(status["status_message"], "proxy already disabled")
        self.assertIsNone(status["backup_path"])

    def test_disabling_codex_proxy_restores_managed_config_backup(self):
        temp_dir = self._make_temp_dir("codex-restore-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        original_managed_config = 'model = "gpt-5.4-mini"\napproval_policy = "on-request"\n'
        managed_config_path.write_text(original_managed_config, encoding="utf-8")
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
        )
        enable_status = service.write_codex_proxy_config()

        status = service.disable_codex_proxy_config()

        self.assertFalse(status["configured"])
        self.assertTrue(status["restored_from_backup"])
        self.assertTrue(enable_status["backup_path"])
        self.assertEqual(managed_config_path.read_text(encoding="utf-8"), original_managed_config)
        self.assertFalse(catalog_path.exists())

    def test_disable_codex_proxy_removes_partial_proxy_artifacts(self):
        temp_dir = self._make_temp_dir("codex-partial-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        managed_config_path.write_text(CODEX_PROXY_CONFIG + "\n", encoding="utf-8")
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
        )

        status = service.disable_codex_proxy_config()

        self.assertFalse(status["configured"])
        self.assertEqual(status["status_message"], "removed proxy-managed Codex files")
        self.assertFalse(managed_config_path.exists())
        self.assertFalse(catalog_path.exists())

    def test_write_codex_proxy_config_writes_managed_files_without_touching_user_config(self):
        temp_dir = self._make_temp_dir("codex-write-")
        user_config_path = temp_dir / "config.toml"
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        user_config_contents = (
            'model = "gpt-5.4-mini"\n'
            'approval_policy = "on-request"\n'
            '\n'
            "[projects.'D:\\sources\\ghcp_proxy']\n"
            'trust_level = "trusted"\n'
        )
        user_config_path.write_text(user_config_contents, encoding="utf-8")
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
        )

        status = service.write_codex_proxy_config()

        managed_config = managed_config_path.read_text(encoding="utf-8")
        managed_config_parsed = tomllib.loads(managed_config)
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.assertTrue(status["configured"])
        self.assertEqual(status["status_message"], "installed proxy config")
        self.assertEqual(user_config_path.read_text(encoding="utf-8"), user_config_contents)
        self.assertIn('model_provider = "custom"', managed_config)
        self.assertEqual(managed_config_parsed["model_catalog_json"], str(catalog_path))
        self.assertIn(f"model_context_window = {CODEX_PROXY_MODEL_CONTEXT_WINDOW}", managed_config)
        self.assertIn(
            f"model_auto_compact_token_limit = {CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT}",
            managed_config,
        )
        self.assertIn('[model_providers.custom]', managed_config)
        self.assertIn('base_url = "http://localhost:8000/v1"', managed_config)
        self.assertTrue(catalog["models"])
        self.assertEqual(catalog["models"][0]["slug"], "gpt-5.4")
        self.assertEqual(catalog["models"][0]["context_window"], CODEX_PROXY_MODEL_CONTEXT_WINDOW)

    def test_codex_catalog_uses_provider_capabilities_per_model(self):
        temp_dir = self._make_temp_dir("codex-caps-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        capabilities = {
            "gpt-5.4": {
                "context_window": 400000,
                "max_context_window": 400000,
                "reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "vision": True,
                "parallel_tool_calls": True,
                "input_modalities": ["text", "image"],
            },
            "claude-sonnet-4.6": {
                "context_window": 200000,
                "max_context_window": 200000,
                "reasoning_efforts": ["low", "medium", "high"],
                "vision": True,
                "parallel_tool_calls": False,
                "input_modalities": ["text", "image"],
            },
        }
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
            model_capabilities_provider=lambda: capabilities,
        )

        service.write_codex_proxy_config()

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        models = {entry["slug"]: entry for entry in catalog["models"]}
        self.assertIn("gpt-5.4", models)
        self.assertEqual(models["gpt-5.4"]["context_window"], 400000)
        self.assertEqual(models["gpt-5.4"]["max_context_window"], 400000)
        gpt_efforts = [lvl["effort"] for lvl in models["gpt-5.4"]["supported_reasoning_levels"]]
        self.assertIn("xhigh", gpt_efforts)
        self.assertGreater(
            models["gpt-5.4"]["auto_compact_token_limit"],
            CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT,
        )
        self.assertIn("claude-sonnet-4.6", models)
        self.assertEqual(models["claude-sonnet-4.6"]["context_window"], 200000)
        claude_efforts = [lvl["effort"] for lvl in models["claude-sonnet-4.6"]["supported_reasoning_levels"]]
        self.assertNotIn("xhigh", claude_efforts)
        self.assertIn("high", claude_efforts)
        # Claude models must advertise reasoning summary support so Codex (a)
        # builds reasoning-aware requests and (b) renders the relayed
        # `response.reasoning_summary_text.delta` events in the TUI.
        self.assertTrue(models["claude-sonnet-4.6"]["supports_reasoning_summaries"])
        self.assertEqual(models["claude-sonnet-4.6"]["default_reasoning_summary"], "auto")
        self.assertTrue(models["gpt-5.4"]["supports_reasoning_summaries"])

    def test_codex_catalog_projects_remapped_target_metadata_onto_source_model(self):
        temp_dir = self._make_temp_dir("codex-remap-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        capabilities = {
            "gpt-5.4": {
                "context_window": 400000,
                "max_context_window": 400000,
                "reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "parallel_tool_calls": True,
                "input_modalities": ["text", "image"],
            },
            "claude-sonnet-4.6": {
                "context_window": 200000,
                "max_context_window": 200000,
                "reasoning_efforts": ["low", "medium", "high"],
                "parallel_tool_calls": False,
                "input_modalities": ["text"],
            },
        }
        routing_settings = {
            "enabled": True,
            "mappings": [
                {
                    "source_model": "gpt-5.4",
                    "target_model": "claude-sonnet-4.6",
                }
            ],
        }
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
            model_capabilities_provider=lambda: capabilities,
            model_routing_settings_provider=lambda: routing_settings,
        )

        service.write_codex_proxy_config()

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        models = {entry["slug"]: entry for entry in catalog["models"]}
        remapped = models["gpt-5.4"]
        self.assertEqual(remapped["display_name"], "gpt-5.4")
        self.assertEqual(remapped["context_window"], 200000)
        self.assertEqual(remapped["max_context_window"], 200000)
        self.assertIn("Routed to claude-sonnet-4.6 (Anthropic)", remapped["description"])
        self.assertFalse(remapped["supports_parallel_tool_calls"])
        self.assertFalse(remapped["support_verbosity"])
        self.assertEqual(remapped["input_modalities"], ["text"])
        remapped_efforts = [lvl["effort"] for lvl in remapped["supported_reasoning_levels"]]
        self.assertEqual(remapped_efforts, ["low", "medium", "high"])

    def test_refresh_codex_model_catalog_rewrites_existing_catalog_for_updated_remaps(self):
        temp_dir = self._make_temp_dir("codex-refresh-")
        managed_config_path = temp_dir / "managed_config.toml"
        catalog_path = temp_dir / "ghcp-proxy-models.json"
        routing_settings = {"enabled": False, "mappings": []}
        capabilities = {
            "gpt-5.4": {
                "context_window": 400000,
                "max_context_window": 400000,
                "reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "parallel_tool_calls": True,
                "input_modalities": ["text", "image"],
            },
            "claude-sonnet-4.6": {
                "context_window": 200000,
                "max_context_window": 200000,
                "reasoning_efforts": ["low", "medium", "high"],
                "parallel_tool_calls": False,
                "input_modalities": ["text"],
            },
        }
        service = self._make_client_proxy_service(
            codex_managed_config_file=str(managed_config_path),
            codex_model_catalog_file=str(catalog_path),
            model_capabilities_provider=lambda: capabilities,
            model_routing_settings_provider=lambda: routing_settings,
        )

        service.write_codex_proxy_config()
        initial_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        initial_models = {entry["slug"]: entry for entry in initial_catalog["models"]}
        self.assertEqual(initial_models["gpt-5.4"]["context_window"], 400000)

        routing_settings["enabled"] = True
        routing_settings["mappings"] = [
            {
                "source_model": "gpt-5.4",
                "target_model": "claude-sonnet-4.6",
            }
        ]

        refreshed = service.refresh_codex_model_catalog()

        self.assertTrue(refreshed)
        updated_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        updated_models = {entry["slug"]: entry for entry in updated_catalog["models"]}
        self.assertEqual(updated_models["gpt-5.4"]["context_window"], 200000)
        self.assertIn("Routed to claude-sonnet-4.6 (Anthropic)", updated_models["gpt-5.4"]["description"])

    def test_claude_proxy_status_requires_token_caps(self):
        temp_dir = self._make_temp_dir("claude-status-")
        settings_path = temp_dir / "settings.json"
        settings_path.write_text(
            (
                '{\n'
                '  "env": {\n'
                '    "ANTHROPIC_BASE_URL": "http://localhost:8000",\n'
                '    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",\n'
                '    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"\n'
                '  }\n'
                '}\n'
            ),
            encoding="utf-8",
        )
        service = self._make_client_proxy_service(claude_settings_file=str(settings_path))

        status = service.claude_proxy_status()

        self.assertFalse(status["configured"])
        self.assertEqual(status["status_message"], "proxy configured, missing context cap and output cap")

    def test_write_claude_proxy_settings_preserves_existing_keys_and_adds_context_cap(self):
        temp_dir = self._make_temp_dir("claude-write-")
        settings_path = temp_dir / "settings.json"
        settings_path.write_text(
            (
                '{\n'
                '  "env": {\n'
                '    "ANTHROPIC_BASE_URL": "http://localhost:8000",\n'
                '    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",\n'
                '    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"\n'
                '  },\n'
                '  "skipDangerousModePermissionPrompt": true,\n'
                '  "model": "opus"\n'
                '}\n'
            ),
            encoding="utf-8",
        )
        service = self._make_client_proxy_service(claude_settings_file=str(settings_path))

        status = service.write_claude_proxy_settings()

        written = json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertTrue(status["configured"])
        self.assertEqual(
            written["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"],
            CLAUDE_MAX_CONTEXT_TOKENS,
        )
        self.assertEqual(
            written["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"],
            CLAUDE_MAX_OUTPUT_TOKENS,
        )
        self.assertTrue(written["skipDangerousModePermissionPrompt"])
        self.assertEqual(written["model"], "opus")
        self.assertEqual(written["effortLevel"], "medium")


if __name__ == "__main__":
    unittest.main()
