import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi import HTTPException

import proxy
from constants import MODEL_ROUTING_CONFIG_FILE
from model_routing_config import ModelRoutingConfig, ModelRoutingConfigService


class ModelRoutingConfigTests(unittest.TestCase):
    def _make_temp_file_path(self, prefix: str, suffix: str) -> Path:
        path = Path.cwd() / f"{prefix}{uuid4().hex}{suffix}"

        def _cleanup():
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass

        self.addCleanup(_cleanup)
        return path

    def _make_service(self, config_file: str | None = None) -> ModelRoutingConfigService:
        return ModelRoutingConfigService(
            ModelRoutingConfig(
                config_file=config_file or MODEL_ROUTING_CONFIG_FILE,
            )
        )

    def test_load_settings_defaults_when_file_is_missing(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        payload = service.config_payload()

        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["mappings"], [])
        self.assertEqual(
            payload["claude_code_defaults"],
            {"opus_model": "", "sonnet_model": "", "haiku_model": ""},
        )
        self.assertTrue(any(row["model"] == "claude-opus-4.6" for row in payload["available_models"]))
        self.assertTrue(any(row["model"] == "gpt-5.3-codex" for row in payload["available_models"]))
        self.assertTrue(any(row["model"] == "gemini-3.1-pro-preview" for row in payload["available_models"]))
        self.assertTrue(any(row["model"] == "gemini-3-flash-preview" for row in payload["available_models"]))
        self.assertTrue(any(row["model"] == "grok-code-fast-1" for row in payload["available_models"]))

    def test_save_settings_accepts_gemini_and_grok_models(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        payload = service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {
                        "source_model": "gemini-3.1-pro-preview",
                        "target_model": "grok-code-fast-1",
                    },
                    {
                        "source_model": "gemini-3-flash-preview",
                        "target_model": "gpt-5.4-mini",
                    },
                ],
            }
        )

        self.assertEqual(
            payload["mappings"],
            [
                {
                    "source_model": "gemini-3.1-pro-preview",
                    "source_provider": "gemini",
                    "target_model": "grok-code-fast-1",
                    "target_provider": "grok",
                },
                {
                    "source_model": "gemini-3-flash-preview",
                    "source_provider": "gemini",
                    "target_model": "gpt-5.4-mini",
                    "target_provider": "codex",
                },
            ],
        )
        self.assertEqual(
            payload["claude_code_defaults"],
            {"opus_model": "", "sonnet_model": "", "haiku_model": ""},
        )

    def test_save_settings_normalizes_models_and_resolves_target(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        payload = service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {
                        "source_model": "GPT 5.3 Codex",
                        "target_model": "Claude Opus 4.6",
                    }
                ],
            }
        )

        self.assertTrue(payload["enabled"])
        self.assertEqual(
            payload["mappings"],
            [
                {
                    "source_model": "gpt-5.3-codex",
                    "source_provider": "codex",
                    "target_model": "claude-opus-4.6",
                    "target_provider": "claude",
                }
            ],
        )
        self.assertEqual(
            service.resolve_target_model("openai/gpt-5.3-codex"),
            "claude-opus-4.6",
        )
        written = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(written["enabled"])

    def test_approval_mappings_round_trip_and_resolve(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        payload = service.save_settings(
            {
                "enabled": False,
                "mappings": [],
                "approval_enabled": True,
                "approval_mappings": [
                    {"source_model": "gpt-5.4", "target_model": "gpt-5.4-mini"}
                ],
            }
        )

        self.assertTrue(payload["approval_enabled"])
        self.assertEqual(
            payload["approval_mappings"],
            [
                {
                    "source_model": "gpt-5.4",
                    "source_provider": "codex",
                    "target_model": "gpt-5.4-mini",
                    "target_provider": "codex",
                }
            ],
        )
        self.assertEqual(service.resolve_approval_target_model("gpt-5.4"), "gpt-5.4-mini")
        # Regular mapping unaffected
        self.assertIsNone(service.resolve_target_model("gpt-5.4"))

    def test_claude_code_defaults_round_trip(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        payload = service.save_settings(
            {
                "enabled": True,
                "mappings": [],
                "claude_code_defaults": {
                    "opus_model": "Claude Opus 4.7",
                    "sonnet_model": "gpt-5.4",
                    "haiku_model": "gpt-5.4-mini",
                },
            }
        )

        self.assertEqual(
            payload["claude_code_defaults"],
            {
                "opus_model": "claude-opus-4.7",
                "sonnet_model": "gpt-5.4",
                "haiku_model": "gpt-5.4-mini",
            },
        )
        written = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(written["claude_code_defaults"], payload["claude_code_defaults"])

    def test_claude_code_defaults_reject_invalid_shape(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        with self.assertRaises(HTTPException) as exc:
            service.save_settings({"claude_code_defaults": ["gpt-5.4"]})

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("claude_code_defaults", str(exc.exception.detail))

    def test_claude_code_defaults_reject_unsupported_model(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        with self.assertRaises(HTTPException) as exc:
            service.save_settings({"claude_code_defaults": {"opus_model": "not-a-model"}})

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("claude_code_defaults.opus_model", str(exc.exception.detail))

    def test_approval_mappings_skip_when_disabled(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        service.save_settings(
            {
                "approval_enabled": False,
                "approval_mappings": [
                    {"source_model": "gpt-5.4", "target_model": "gpt-5.4-mini"}
                ],
            }
        )
        self.assertIsNone(service.resolve_approval_target_model("gpt-5.4"))

    def test_save_settings_rejects_duplicate_approval_source_models(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        with self.assertRaises(HTTPException) as exc:
            service.save_settings(
                {
                    "approval_enabled": True,
                    "approval_mappings": [
                        {"source_model": "gpt-5.4", "target_model": "gpt-5.4-mini"},
                        {"source_model": "GPT 5.4", "target_model": "claude-haiku-4.5"},
                    ],
                }
            )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("Duplicate approval mapping", str(exc.exception.detail))

    def test_save_settings_rejects_duplicate_source_models(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        with self.assertRaises(HTTPException) as exc:
            service.save_settings(
                {
                    "enabled": True,
                    "mappings": [
                        {"source_model": "gpt-5.3-codex", "target_model": "claude-opus-4.6"},
                        {"source_model": "openai/gpt-5.3-codex", "target_model": "claude-sonnet-4.6"},
                    ],
                }
            )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("Duplicate mapping", str(exc.exception.detail))

    def test_model_routing_config_api_saves_payload(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/api/config/model-routing"),
            method="POST",
            headers={},
        )
        payload = {"enabled": True, "mappings": []}

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=payload)),
            mock.patch.object(proxy.model_routing_config_service, "save_settings", return_value={"enabled": True, "mappings": [], "approval_enabled": False, "approval_mappings": [], "claude_code_defaults": {"opus_model": "", "sonnet_model": "", "haiku_model": ""}, "available_models": [], "path": "x"}) as save_settings,
            mock.patch.object(proxy.client_proxy_config_service, "refresh_client_model_metadata", return_value={"codex": False, "claude": False}),
        ):
            response = proxy.asyncio.run(proxy.model_routing_config_api(request))

        save_settings.assert_called_once_with(payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.body,
            b'{"enabled":true,"mappings":[],"approval_enabled":false,"approval_mappings":[],"claude_code_defaults":{"opus_model":"","sonnet_model":"","haiku_model":""},"available_models":[],"path":"x"}',
        )

    def test_model_routing_status_api_returns_current_payload(self):
        with mock.patch.object(
            proxy.model_routing_config_service,
            "config_payload",
            return_value={"enabled": False, "mappings": [], "approval_enabled": False, "approval_mappings": [], "claude_code_defaults": {"opus_model": "", "sonnet_model": "", "haiku_model": ""}, "available_models": [], "path": "x"},
        ) as config_payload:
            response = proxy.asyncio.run(proxy.model_routing_status_api())

        config_payload.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.body,
            b'{"enabled":false,"mappings":[],"approval_enabled":false,"approval_mappings":[],"claude_code_defaults":{"opus_model":"","sonnet_model":"","haiku_model":""},"available_models":[],"path":"x"}',
        )

    def test_compact_fallback_defaults_to_gpt_5_4_for_claude_target(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {"source_model": "gpt-5.3-codex", "target_model": "claude-opus-4.6"},
                ],
            }
        )

        self.assertEqual(
            service.resolve_compact_fallback_model("gpt-5.3-codex"),
            "gpt-5.4",
        )

    def test_compact_fallback_defaults_to_gpt_5_4_for_gemini_target(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {"source_model": "gpt-5.3-codex", "target_model": "gemini-3.1-pro-preview"},
                ],
            }
        )

        self.assertEqual(
            service.resolve_compact_fallback_model("gpt-5.3-codex"),
            "gpt-5.4",
        )

    def test_compact_fallback_defaults_to_gpt_5_4_for_grok_target(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {"source_model": "gpt-5.3-codex", "target_model": "grok-code-fast-1"},
                ],
            }
        )

        self.assertEqual(
            service.resolve_compact_fallback_model("gpt-5.3-codex"),
            "gpt-5.4",
        )

    def test_compact_fallback_respects_explicit_override(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        payload = service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {
                        "source_model": "gpt-5.3-codex",
                        "target_model": "claude-opus-4.6",
                        "compact_fallback_model": "gpt-5.2-codex",
                    },
                ],
            }
        )

        self.assertEqual(
            payload["mappings"][0]["compact_fallback_model"],
            "gpt-5.2-codex",
        )
        self.assertEqual(
            service.resolve_compact_fallback_model("gpt-5.3-codex"),
            "gpt-5.2-codex",
        )
        written = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            written["mappings"][0]["compact_fallback_model"],
            "gpt-5.2-codex",
        )

    def test_compact_fallback_rejects_non_gpt_override(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        with self.assertRaises(HTTPException) as exc:
            service.save_settings(
                {
                    "enabled": True,
                    "mappings": [
                        {
                            "source_model": "gpt-5.3-codex",
                            "target_model": "claude-opus-4.6",
                            "compact_fallback_model": "claude-sonnet-4.6",
                        },
                    ],
                }
            )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("must be a GPT model", str(exc.exception.detail))

    def test_compact_fallback_none_for_codex_target(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        service.save_settings(
            {
                "enabled": True,
                "mappings": [
                    {"source_model": "gpt-5.3-codex", "target_model": "gpt-5.4"},
                ],
            }
        )

        self.assertIsNone(service.resolve_compact_fallback_model("gpt-5.3-codex"))

    def test_compact_fallback_none_when_routing_disabled(self):
        config_path = self._make_temp_file_path("model-routing-", ".json")
        service = self._make_service(str(config_path))

        service.save_settings(
            {
                "enabled": False,
                "mappings": [
                    {"source_model": "gpt-5.3-codex", "target_model": "claude-opus-4.6"},
                ],
            }
        )

        self.assertIsNone(service.resolve_compact_fallback_model("gpt-5.3-codex"))


if __name__ == "__main__":
    unittest.main()
