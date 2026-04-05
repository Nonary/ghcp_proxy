import unittest
from pathlib import Path
from uuid import uuid4

import proxy
from proxy_client_config import ProxyClientConfig, ProxyClientConfigService


class ClientConfigTests(unittest.TestCase):
    def _make_temp_file_path(self, prefix: str, suffix: str) -> Path:
        path = Path.cwd() / f"{prefix}{uuid4().hex}{suffix}"
        def _cleanup():
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass
            for backup_path in path.parent.glob(f"{path.name}.ghcp-proxy.bak.*"):
                try:
                    backup_path.unlink(missing_ok=True)
                except PermissionError:
                    pass
        self.addCleanup(_cleanup)
        return path

    def _make_client_proxy_service(
        self,
        *,
        codex_config_file: str | None = None,
        claude_settings_file: str | None = None,
    ) -> ProxyClientConfigService:
        return ProxyClientConfigService(
            ProxyClientConfig(
                codex_config_file=codex_config_file or proxy.CODEX_CONFIG_FILE,
                codex_proxy_config=proxy.CODEX_PROXY_CONFIG,
                claude_settings_file=claude_settings_file or proxy.CLAUDE_SETTINGS_FILE,
                claude_proxy_settings=proxy.CLAUDE_PROXY_SETTINGS,
                claude_max_context_tokens=proxy.CLAUDE_MAX_CONTEXT_TOKENS,
                claude_max_output_tokens=proxy.CLAUDE_MAX_OUTPUT_TOKENS,
            )
        )

    def test_enabling_codex_proxy_is_idempotent_when_already_enabled(self):
        config_path = self._make_temp_file_path("codex-config-", ".toml")
        config_path.write_text(proxy.CODEX_PROXY_CONFIG + "\n", encoding="utf-8")
        service = self._make_client_proxy_service(codex_config_file=str(config_path))

        status = service.write_codex_proxy_config()

        backups = list(config_path.parent.glob(f"{config_path.name}.ghcp-proxy.bak.*"))
        self.assertTrue(status["configured"])
        self.assertEqual(status["status_message"], "proxy already enabled")
        self.assertEqual(backups, [])

    def test_disabling_codex_proxy_is_idempotent_when_already_disabled(self):
        config_path = self._make_temp_file_path("codex-config-", ".toml")
        service = self._make_client_proxy_service(codex_config_file=str(config_path))
        status = service.disable_codex_proxy_config()

        self.assertFalse(status["configured"])
        self.assertEqual(status["status_message"], "proxy already disabled")
        self.assertIsNone(status["backup_path"])

    def test_disabling_codex_proxy_restores_latest_backup(self):
        config_path = self._make_temp_file_path("codex-config-", ".toml")
        backup_path = Path(f"{config_path}.ghcp-proxy.bak.20260404_180000")
        config_path.write_text(proxy.CODEX_PROXY_CONFIG + "\n", encoding="utf-8")
        backup_contents = 'model_provider = "openai"\n'
        backup_path.write_text(backup_contents, encoding="utf-8")
        service = self._make_client_proxy_service(codex_config_file=str(config_path))

        status = service.disable_codex_proxy_config()

        self.assertFalse(status["configured"])
        self.assertTrue(status["restored_from_backup"])
        self.assertEqual(config_path.read_text(encoding="utf-8"), backup_contents)

    def test_claude_proxy_status_requires_token_caps(self):
        settings_path = self._make_temp_file_path("claude-settings-", ".json")
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
        settings_path = self._make_temp_file_path("claude-settings-", ".json")
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

        written = proxy.json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertTrue(status["configured"])
        self.assertEqual(
            written["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"],
            proxy.CLAUDE_MAX_CONTEXT_TOKENS,
        )
        self.assertEqual(
            written["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"],
            proxy.CLAUDE_MAX_OUTPUT_TOKENS,
        )
        self.assertTrue(written["skipDangerousModePermissionPrompt"])
        self.assertEqual(written["model"], "opus")
        self.assertEqual(written["effortLevel"], "medium")


if __name__ == "__main__":
    unittest.main()
