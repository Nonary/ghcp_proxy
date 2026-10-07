import unittest
from unittest import mock

import httpx

import excel_model_catalog as catalog


def access_payload(models=None, restricted=None, allowed=True):
    return {
        "allowed": allowed,
        "model_catalog": {
            "models": models if models is not None else [{
                "id": "gpt-6-sol", "label": "GPT-6 Sol",
                "efforts": [{"value": "none"}, {"value": "medium"}, {"value": "high"}],
            }],
            "restricted_models": restricted or [],
            "default_model": "gpt-6-astra",
        },
    }


class ExcelModelCatalogTests(unittest.TestCase):
    def setUp(self):
        for target, kwargs in (
            ("_CACHE", {"new": {"key": None, "ts": 0.0, "data": {}}}),
        ):
            patcher = mock.patch.object(catalog, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("refresh_macos_excel_session", "refresh_windows_excel_session"):
            patcher = mock.patch.object(catalog.excel_session_capture, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = mock.Mock()
        self.store.request_headers.return_value = {
            "authorization": "test-session-token", "x-openai-account-id": "test-account",
        }
        self.response = httpx.Response(200, json=access_payload(), request=httpx.Request("GET", catalog.MODELS_URL))
        self.client = mock.Mock()
        self.client.get.side_effect = lambda *args, **kwargs: self.response
        patcher = mock.patch.object(catalog.httpx, "Client")
        self.factory = patcher.start()
        self.addCleanup(patcher.stop)
        self.factory.return_value.__enter__.return_value = self.client

    def test_official_access_response_supplies_ids_labels_and_efforts(self):
        models = catalog.parse_model_catalog(access_payload())
        self.assertEqual(set(models), {"gpt-6-sol"})
        self.assertEqual(models["gpt-6-sol"]["display_name"], "GPT-6 Sol")
        self.assertEqual(models["gpt-6-sol"]["reasoning_efforts"], ["medium", "high"])

    def test_restricted_and_disabled_models_are_not_available(self):
        payload = access_payload([
            {"id": "gpt-6-astra"}, {"id": "gpt-6-sol"},
            {"id": "gpt-6-luna", "model_picker_enabled": False},
        ], restricted=["gpt-6-astra"])
        self.assertEqual(set(catalog.parse_model_catalog(payload)), {"gpt-6-sol"})

    def test_unknown_new_models_are_discovered_without_a_static_allowlist(self):
        self.assertEqual(set(catalog.parse_model_catalog(access_payload([{"id": "gpt-7-new-model"}]))), {"gpt-7-new-model"})

    def test_missing_or_denied_access_does_not_invent_models(self):
        for payload in (None, {}, {"allowed": True}, access_payload(allowed=False), access_payload(models=[])):
            with self.subTest(payload=payload):
                self.assertEqual(catalog.parse_model_catalog(payload), {})

    def test_bad_model_entries_are_ignored(self):
        payload = access_payload([None, {}, {"id": []}, {"id": ""}, {"id": "not-a-model"}])
        self.assertEqual(catalog.parse_model_catalog(payload), {})

    def test_fetch_uses_excel_session_and_enterprise_tls_context(self):
        verify = object()
        models = catalog.fetch_model_capabilities(verify=verify, session_store=self.store)
        self.assertEqual(set(models), {"gpt-6-sol"})
        self.factory.assert_called_once_with(timeout=5.0, verify=verify, trust_env=True)
        self.client.get.assert_called_once_with(catalog.MODELS_URL, headers=self.store.request_headers.return_value)
        self.assertIn("/responses/access?include_models=true", catalog.MODELS_URL)
        self.assertNotIn("test-session-token", repr(catalog._CACHE))

    def test_cached_result_is_copied_and_scoped_to_session_and_account(self):
        models = catalog.fetch_model_capabilities(session_store=self.store)
        models["gpt-6-sol"]["reasoning_efforts"].clear()
        self.assertEqual(catalog.fetch_model_capabilities(session_store=self.store)["gpt-6-sol"]["reasoning_efforts"], ["medium", "high"])
        self.assertEqual(self.client.get.call_count, 1)
        self.store.request_headers.return_value = {"authorization": "second-token", "x-openai-account-id": "another-account"}
        self.response = httpx.Response(200, json=access_payload(models=[]), request=httpx.Request("GET", catalog.MODELS_URL))
        self.assertEqual(catalog.fetch_model_capabilities(session_store=self.store), {})
        self.assertEqual(self.client.get.call_count, 2)

    def test_missing_or_expired_session_does_not_reuse_a_cached_catalog(self):
        catalog.fetch_model_capabilities(session_store=self.store)
        self.store.request_headers.side_effect = RuntimeError("expired session")
        self.assertEqual(catalog.fetch_model_capabilities(session_store=self.store), {})
        self.assertEqual(self.client.get.call_count, 1)

    def test_http_failure_does_not_add_models(self):
        self.response = httpx.Response(403, request=httpx.Request("GET", catalog.MODELS_URL))
        self.assertEqual(catalog.fetch_model_capabilities(session_store=self.store), {})

    def test_network_failure_does_not_add_models(self):
        self.client.get.side_effect = httpx.ConnectError("unavailable")
        self.assertEqual(catalog.fetch_model_capabilities(session_store=self.store), {})

    def test_invalid_json_does_not_add_models(self):
        self.response = httpx.Response(200, content=b"not JSON", request=httpx.Request("GET", catalog.MODELS_URL))
        self.assertEqual(catalog.fetch_model_capabilities(session_store=self.store), {})


if __name__ == "__main__":
    unittest.main()
