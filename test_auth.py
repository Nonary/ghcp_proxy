import unittest

import httpx

import auth


class AuthTests(unittest.TestCase):
    def test_friendly_auth_failure_message_explains_dns_lookup_errors(self):
        error = httpx.ConnectError(
            "[Errno 8] nodename nor servname provided, or not known",
            request=httpx.Request("POST", "https://github.com/login/device/code"),
        )

        message = auth._friendly_auth_failure_message(error)

        self.assertIn("could not look up GitHub's network name", message)
        self.assertIn("curl -I https://github.com", message)

    def test_friendly_auth_failure_message_explains_timeouts(self):
        error = httpx.ReadTimeout(
            "timed out",
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )

        message = auth._friendly_auth_failure_message(error)

        self.assertIn("timed out", message)
        self.assertIn("start-ghproxy", message)

    def test_friendly_auth_failure_message_preserves_request_url_for_other_network_errors(self):
        error = httpx.ConnectError(
            "All connection attempts failed",
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )

        message = auth._friendly_auth_failure_message(error)

        self.assertIn("https://api.github.com/copilot_internal/v2/token", message)
