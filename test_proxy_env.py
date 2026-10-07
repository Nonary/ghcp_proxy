import json
import os
import unittest
from unittest import mock

import background_proxy
import httpx
import proxy


class ProxyModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(proxy, "fetch_excel_model_capabilities", return_value={})
        self.excel_catalog = patcher.start()
        self.addCleanup(patcher.stop)
    async def test_no_auth_does_not_advertise_excel_models(self):
        with mock.patch.object(proxy.auth, "get_api_key", side_effect=RuntimeError("not signed in")):
            response = await proxy._proxy_models_request()
        self.assertEqual(json.loads(response.body)["data"], [])

    async def test_excel_only_models_survive_missing_copilot_auth(self):
        self.excel_catalog.return_value = {"gpt-6-sol": {}}
        with mock.patch.object(proxy.auth, "get_api_key", side_effect=RuntimeError("not signed in")):
            response = await proxy._proxy_models_request()
        self.assertEqual([item["id"] for item in json.loads(response.body)["data"]], ["gpt-6-sol-excel"])

    async def test_union_uses_independent_model_ids(self):
        self.excel_catalog.return_value = {"gpt-6-sol": {}}
        upstream = httpx.Response(200, json={"data": [{"id": "gpt-6.1-sol"}]})
        async with httpx.AsyncClient() as client:
            with mock.patch.object(proxy.auth, "get_api_key", return_value="token"), \
                 mock.patch.object(proxy.auth, "get_api_base", return_value="https://copilot.example"), \
                 mock.patch.object(proxy, "_get_upstream_client", return_value=client), \
                 mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=upstream)):
                response = await proxy._proxy_models_request()
        self.assertEqual({item["id"] for item in json.loads(response.body)["data"]}, {"gpt-6.1-sol", "gpt-6-sol-excel"})

    async def test_copilot_http_failure_does_not_hide_excel_models(self):
        self.excel_catalog.return_value = {"gpt-6-sol": {}}
        async with httpx.AsyncClient() as client:
            with mock.patch.object(proxy.auth, "get_api_key", return_value="token"), \
                 mock.patch.object(proxy.auth, "get_api_base", return_value="https://copilot.example"), \
                 mock.patch.object(proxy, "_get_upstream_client", return_value=client), \
                 mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=httpx.Response(503))):
                response = await proxy._proxy_models_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in json.loads(response.body)["data"]], ["gpt-6-sol-excel"])

    async def test_live_models_payload_only_adds_available_aliases(self):
        self.excel_catalog.return_value = {"gpt-5.6-sol": {}}
        upstream = httpx.Response(200, json={"data": [
            {"id": "gpt-5.6-sol"},
            {"id": "gpt-6-astra", "model_picker_enabled": False},
            {"id": "gpt-6-luna", "policy": {"state": "disabled"}},
        ]})
        async with httpx.AsyncClient() as client:
            with mock.patch.object(proxy.auth, "get_api_key", return_value="token"), \
                 mock.patch.object(proxy.auth, "get_api_base", return_value="https://copilot.example"), \
                 mock.patch.object(proxy, "_get_upstream_client", return_value=client), \
                 mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=upstream)):
                response = await proxy._proxy_models_request()
        ids = {item["id"] for item in json.loads(response.body)["data"]}
        self.assertEqual(ids & set(proxy.excel_upstream.MODEL_IDS), {"gpt-5.6-sol-excel"})


class ProxyEnvironmentTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(proxy, "fetch_excel_model_capabilities", return_value={})
        self.excel_catalog = patcher.start()
        self.addCleanup(patcher.stop)

    def test_excel_capabilities_survive_copilot_discovery_failure(self):
        self.excel_catalog.return_value = {"gpt-6-sol": {}}
        with mock.patch.object(proxy.auth, "get_api_base", side_effect=RuntimeError("not signed in")):
            self.assertEqual(set(proxy.fetch_available_model_capabilities()), {"gpt-6-sol-excel"})
    def test_model_capability_fetch_uses_runtime_enterprise_ca_context(self):
        self.excel_catalog.return_value = {"gpt-5.6-luna": {}}
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [
                    {"id": "gpt-5.6-luna"},
                    {"id": "gpt-6-astra", "model_picker_enabled": False},
                    {"id": "gpt-5.6-sol", "policy": {"state": "disabled"}},
                ]}

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, _url, *, headers):
                self.headers = headers
                return FakeResponse()

        runtime_context = object()
        with mock.patch.object(proxy, "_COPILOT_MODEL_CAPS_CACHE", {"key": None, "ts": 0.0, "data": {}}), \
             mock.patch.object(proxy.auth, "get_api_base", return_value="https://copilot.example"), \
             mock.patch.object(proxy.auth, "get_api_key", return_value="token"), \
             mock.patch.object(proxy.format_translation, "build_copilot_headers", return_value={}), \
             mock.patch.object(proxy.copilot_sdk_upstream, "_runtime_ssl_context", return_value=runtime_context), \
             mock.patch.object(proxy.httpx, "Client", side_effect=FakeClient) as client_factory:
            capabilities = proxy.fetch_available_model_capabilities()

        self.assertIs(client_factory.call_args.kwargs["verify"], runtime_context)
        self.assertTrue(client_factory.call_args.kwargs["trust_env"])
        self.assertIn("gpt-5.6-luna-excel", capabilities)
        self.assertEqual(capabilities["gpt-5.6-luna-excel"]["provider"], "OpenAI Excel")
        self.assertNotIn("gpt-6-sol-excel", capabilities)
        self.assertNotIn("gpt-6-luna-excel", capabilities)
        self.assertNotIn("gpt-6-astra-excel", capabilities)
        self.assertNotIn("gpt-5.6-sol-excel", capabilities)
        self.assertNotIn("gpt-5.6-sol", capabilities)

    def test_model_capability_fetch_without_auth_does_not_advertise_excel(self):
        with mock.patch.object(proxy.auth, "get_api_base", side_effect=RuntimeError("not signed in")):
            self.assertEqual(proxy.fetch_available_model_capabilities(), {})

    def test_model_capability_fetch_failure_does_not_advertise_excel(self):
        with mock.patch.object(proxy, "_COPILOT_MODEL_CAPS_CACHE", {"key": None, "ts": 0.0, "data": {}}), \
             mock.patch.object(proxy.auth, "get_api_base", return_value="https://copilot.example"), \
             mock.patch.object(proxy.auth, "get_api_key", return_value="token"), \
             mock.patch.object(proxy.httpx, "Client", side_effect=RuntimeError("upstream unavailable")):
            self.assertEqual(proxy.fetch_available_model_capabilities(), {})

    def test_model_capability_cache_does_not_restore_unavailable_aliases(self):
        self.excel_catalog.return_value = {"gpt-5.6-sol": {}}
        cache = {
            "key": "https://copilot.example", "ts": proxy.time.monotonic(),
            "data": {"gpt-5.6-sol": {}, "gpt-6-astra": {"model_picker_enabled": False}},
        }
        with mock.patch.object(proxy, "_COPILOT_MODEL_CAPS_CACHE", cache), \
             mock.patch.object(proxy.auth, "get_api_base", return_value="https://copilot.example"), \
             mock.patch.object(proxy.httpx, "Client") as client_factory:
            capabilities = proxy.fetch_available_model_capabilities()
        client_factory.assert_not_called()
        self.assertIn("gpt-5.6-sol-excel", capabilities)
        self.assertNotIn("gpt-6-astra-excel", capabilities)

    def test_apply_upstream_proxy_env_aliases_sets_standard_proxy_keys(self):
        with mock.patch.dict(
            os.environ,
            {
                "GHCP_UPSTREAM_PROXY": " http://proxy.example:8080 ",
                "GHCP_NO_PROXY": "localhost,127.0.0.1",
            },
            clear=True,
        ):
            applied = proxy._apply_upstream_proxy_env_aliases()

            self.assertIn("HTTPS_PROXY", applied)
            self.assertIn("HTTP_PROXY", applied)
            self.assertIn("NO_PROXY", applied)
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.example:8080")
            self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy.example:8080")
            self.assertEqual(os.environ["NO_PROXY"], "localhost,127.0.0.1")

    def test_apply_upstream_proxy_env_aliases_does_not_override_existing_proxy(self):
        with mock.patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://already-set:80",
                "GHCP_UPSTREAM_PROXY": "http://new-proxy:80",
            },
            clear=True,
        ):
            applied = proxy._apply_upstream_proxy_env_aliases()

            self.assertNotIn("HTTPS_PROXY", applied)
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://already-set:80")

    def test_upstream_tls_and_http2_defaults_for_proxy_environment(self):
        with mock.patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.example:80",
            },
            clear=True,
        ):
            self.assertTrue(proxy._upstream_proxy_configured())
            self.assertEqual(
                proxy._configured_upstream_tls_verify(True),
                (False, "proxy_default"),
            )
            self.assertEqual(
                proxy._configured_upstream_http2(True),
                (False, "proxy_default"),
            )

    def test_upstream_tls_and_http2_explicit_overrides(self):
        with mock.patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.example:80",
                "GHCP_UPSTREAM_TLS_VERIFY": "1",
                "GHCP_UPSTREAM_HTTP2": "true",
            },
            clear=True,
        ):
            self.assertEqual(
                proxy._configured_upstream_tls_verify(True),
                (True, "GHCP_UPSTREAM_TLS_VERIFY"),
            )
            self.assertEqual(
                proxy._configured_upstream_http2(True),
                (True, "GHCP_UPSTREAM_HTTP2"),
            )


class BackgroundProxyLaunchAgentEnvironmentTests(unittest.TestCase):
    def test_macos_launch_agent_includes_proxy_environment_variables(self):
        manager = background_proxy.BackgroundProxyManager(
            repo_dir="/tmp/repo",
            python_executable="/tmp/repo/.venv/bin/python",
            platform="darwin",
        )
        with mock.patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.example:80",
                "NO_PROXY": "localhost,127.0.0.1",
            },
            clear=True,
        ):
            plist = manager._macos_launch_agent()

        self.assertIn("<key>EnvironmentVariables</key>", plist)
        self.assertIn("<key>HTTPS_PROXY</key>", plist)
        self.assertIn("<string>http://proxy.example:80</string>", plist)
        self.assertIn("<key>NO_PROXY</key>", plist)
