import json
import os
import tempfile
import unittest
from unittest import mock

import trace_prompt_security
from tools import trace_prompt_decryption


class TracePromptDecryptionTests(unittest.TestCase):
    def _key_pair_or_skip(self):
        try:
            return trace_prompt_security.generate_envelope_key_pair()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def test_from_environment_loads_private_key_and_decrypts_envelope_payload(self):
        key_pair = self._key_pair_or_skip()
        password = "pw"
        salt = trace_prompt_security.new_salt()
        key = trace_prompt_security.derive_key(password, salt)
        verifier = trace_prompt_security.make_password_verifier(key, salt)
        private_key_payload = trace_prompt_security.encrypt_payload(key_pair["private_key"], key, salt)
        envelope_payload = trace_prompt_security.encrypt_payload_with_public_key(
            {"input": "plain"},
            key_pair["public_key"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            settings_path = os.path.join(tmp, "client-proxy-settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "trace_prompt_logging_salt": salt,
                        "trace_prompt_logging_verifier": verifier,
                        "trace_prompt_logging_public_key": key_pair["public_key"],
                        "trace_prompt_logging_private_key": private_key_payload,
                    },
                    f,
                )

            with mock.patch.dict(
                os.environ,
                {
                    "GHCP_TRACE_PROMPT_PASSWORD": password,
                    "GHCP_CLIENT_PROXY_SETTINGS_FILE": settings_path,
                },
                clear=False,
            ):
                decryptor = trace_prompt_decryption.TracePromptDecryptor.from_environment()
                self.assertEqual(decryptor.decrypt(envelope_payload), {"input": "plain"})

        self.assertEqual(decryptor.stats.encrypted, 1)
        self.assertEqual(decryptor.stats.decrypted, 1)


if __name__ == "__main__":
    unittest.main()
