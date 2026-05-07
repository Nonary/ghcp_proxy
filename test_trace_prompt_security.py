import unittest

import trace_prompt_security


class TracePromptSecurityTests(unittest.TestCase):
    def _key_pair_or_skip(self):
        try:
            return trace_prompt_security.generate_envelope_key_pair()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def test_envelope_payload_round_trips_with_private_key(self):
        key_pair = self._key_pair_or_skip()

        encrypted = trace_prompt_security.encrypt_payload_with_public_key(
            {"input": "plain"},
            key_pair["public_key"],
        )

        self.assertEqual(encrypted["_encrypted"], trace_prompt_security.ENVELOPE_PAYLOAD_MARKER)
        self.assertIn("encrypted_key", encrypted)
        self.assertEqual(
            trace_prompt_security.decrypt_payload(encrypted, private_key_pem=key_pair["private_key"]),
            {"input": "plain"},
        )

    def test_envelope_payload_requires_private_key_to_decrypt(self):
        key_pair = self._key_pair_or_skip()
        encrypted = trace_prompt_security.encrypt_payload_with_public_key(
            {"input": "plain"},
            key_pair["public_key"],
        )

        with self.assertRaises(ValueError):
            trace_prompt_security.decrypt_payload(encrypted)


if __name__ == "__main__":
    unittest.main()
