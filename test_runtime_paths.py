import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import app_paths
import migrate_runtime_paths


class AppPathsTests(unittest.TestCase):
    def test_windows_paths_use_appdata_locations(self):
        env = {
            "APPDATA": r"C:\Users\me\AppData\Roaming",
            "LOCALAPPDATA": r"C:\Users\me\AppData\Local",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(app_paths.sys, "platform", "win32"):
            self.assertEqual(app_paths.user_config_dir(), os.path.join(env["APPDATA"], "ghcp_proxy"))
            self.assertEqual(app_paths.user_state_dir(), os.path.join(env["LOCALAPPDATA"], "ghcp_proxy"))
            self.assertEqual(app_paths.user_cache_dir(), os.path.join(env["LOCALAPPDATA"], "ghcp_proxy", "Cache"))

    def test_macos_paths_use_library_locations(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(app_paths.sys, "platform", "darwin"):
            self.assertTrue(app_paths.user_config_dir().endswith("Library/Application Support/ghcp_proxy"))
            self.assertTrue(app_paths.user_state_dir().endswith("Library/Application Support/ghcp_proxy"))
            self.assertTrue(app_paths.user_cache_dir().endswith("Library/Caches/ghcp_proxy"))

    def test_environment_overrides_win(self):
        env = {
            "GHCP_CONFIG_DIR": "/config",
            "GHCP_STATE_DIR": "/state",
            "GHCP_CACHE_DIR": "/cache",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(app_paths.user_config_dir(), "/config")
            self.assertEqual(app_paths.user_state_dir(), "/state")
            self.assertEqual(app_paths.user_cache_dir(), "/cache")


class RuntimeMigrationTests(unittest.TestCase):
    def test_migration_copies_legacy_files_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "legacy"
            state = root / "state"
            config = root / "config"
            cache = root / "cache"
            legacy.mkdir()
            state.mkdir()
            config.mkdir()
            cache.mkdir()

            (legacy / "access-token").write_text("legacy-token", encoding="utf-8")
            (legacy / "model-routing.json").write_text('{"legacy":true}', encoding="utf-8")
            (legacy / ".ghcp_proxy-cache-v2.sqlite3").write_text("legacy-cache", encoding="utf-8")
            existing_api_key = state / "api-key.json"
            existing_api_key.write_text('{"existing":true}', encoding="utf-8")
            (legacy / "api-key.json").write_text('{"legacy":true}', encoding="utf-8")

            patches = [
                mock.patch.object(migrate_runtime_paths, "LEGACY_TOKEN_DIR", str(legacy)),
                mock.patch.object(migrate_runtime_paths, "TOKEN_DIR", str(state)),
                mock.patch.object(migrate_runtime_paths, "ACCESS_TOKEN_FILE", str(state / "access-token")),
                mock.patch.object(migrate_runtime_paths, "API_KEY_FILE", str(existing_api_key)),
                mock.patch.object(migrate_runtime_paths, "MODEL_ROUTING_CONFIG_FILE", str(config / "model-routing.json")),
                mock.patch.object(migrate_runtime_paths, "CLIENT_PROXY_SETTINGS_FILE", str(config / "client-proxy.json")),
                mock.patch.object(migrate_runtime_paths, "LEGACY_PREMIUM_PLAN_CONFIG_FILE", str(config / "premium-plan.json")),
                mock.patch.object(migrate_runtime_paths, "LEGACY_BILLING_TOKEN_FILE", str(config / "billing-token")),
                mock.patch.object(migrate_runtime_paths, "SAFEGUARD_CONFIG_FILE", str(config / "safeguard.json")),
                mock.patch.object(migrate_runtime_paths, "USAGE_LOG_FILE", str(state / "usage-log.jsonl")),
                mock.patch.object(migrate_runtime_paths, "REQUEST_ERROR_LOG_FILE", str(state / "request-errors.log")),
                mock.patch.object(migrate_runtime_paths, "REQUEST_TRACE_LOG_FILE", str(state / "request-trace.jsonl")),
                mock.patch.object(migrate_runtime_paths, "CURSOR_FILE", str(state / "codex-native-cursor.json")),
                mock.patch.object(migrate_runtime_paths, "SQLITE_CACHE_FILE", str(cache / ".ghcp_proxy-cache-v2.sqlite3")),
                mock.patch.object(migrate_runtime_paths, "_CACHE_ENV_OVERRIDDEN", False),
            ]
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                migrated = migrate_runtime_paths.migrate_legacy_runtime_files()

            self.assertEqual((state / "access-token").read_text(encoding="utf-8"), "legacy-token")
            self.assertEqual((config / "model-routing.json").read_text(encoding="utf-8"), '{"legacy":true}')
            self.assertEqual((cache / ".ghcp_proxy-cache-v2.sqlite3").read_text(encoding="utf-8"), "legacy-cache")
            self.assertEqual(existing_api_key.read_text(encoding="utf-8"), '{"existing":true}')
            self.assertEqual(len(migrated), 3)


if __name__ == "__main__":
    unittest.main()
