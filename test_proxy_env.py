import os
import unittest
from unittest import mock

import background_proxy
import proxy


class ProxyEnvironmentTests(unittest.TestCase):
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
