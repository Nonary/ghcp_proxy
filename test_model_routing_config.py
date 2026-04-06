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
        self.assertTrue(any(row["model"] == "claude-opus-4.6" for row in payload["available_models"]))
        self.assertTrue(any(row["model"] == "gpt-5.3-codex" for row in payload["available_models"]))

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
            mock.patch.object(proxy.model_routing_config_service, "save_settings", return_value={"enabled": True, "mappings": [], "available_models": [], "path": "x"}) as save_settings,
        ):
            response = proxy.asyncio.run(proxy.model_routing_config_api(request))

        save_settings.assert_called_once_with(payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.body,
            b'{"enabled":true,"mappings":[],"available_models":[],"path":"x"}',
        )

    def test_model_routing_status_api_returns_current_payload(self):
        with mock.patch.object(
            proxy.model_routing_config_service,
            "config_payload",
            return_value={"enabled": False, "mappings": [], "available_models": [], "path": "x"},
        ) as config_payload:
            response = proxy.asyncio.run(proxy.model_routing_status_api())

        config_payload.assert_called_once_with()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.body,
            b'{"enabled":false,"mappings":[],"available_models":[],"path":"x"}',
        )


if __name__ == "__main__":
    unittest.main()
