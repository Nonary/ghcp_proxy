import json
import unittest

from codex_agent_compat import codex_subagent_identity, codex_subagent_role
from initiator_policy import is_approval_agent_request


class ApprovalRoutingTests(unittest.TestCase):
    def test_current_codex_guardian_metadata_is_detected(self):
        body = {
            "client_metadata": {
                "x-codex-turn-metadata": json.dumps(
                    {
                        "thread_source": "subagent",
                        "agent_role": "guardian",
                    }
                )
            }
        }

        self.assertEqual(codex_subagent_identity(body), "codex:guardian")
        self.assertEqual(codex_subagent_role(body), "guardian")
        self.assertTrue(
            is_approval_agent_request(
                subagent=codex_subagent_identity(body),
                inbound_protocol="responses",
                body=body,
            )
        )

    def test_regular_codex_subagent_is_not_approval_agent(self):
        body = {
            "client_metadata": {
                "thread_source": "subagent",
                "agent_role": "review",
            }
        }

        self.assertFalse(
            is_approval_agent_request(
                subagent=codex_subagent_identity(body),
                inbound_protocol="responses",
                body=body,
            )
        )

    def test_nested_approval_metadata_is_detected(self):
        body = {
            "client_metadata": {
                "turn": json.dumps(
                    {
                        "metadata": {
                            "thread_source": "subagent",
                            "agent_type": "approval",
                        }
                    }
                )
            }
        }

        self.assertEqual(codex_subagent_identity(body), "codex:approval")
        self.assertEqual(codex_subagent_role(body), "approval")
        self.assertTrue(
            is_approval_agent_request(
                subagent=codex_subagent_identity(body),
                inbound_protocol="responses",
                body=body,
            )
        )

    def test_qualified_guardian_identity_is_detected(self):
        self.assertTrue(
            is_approval_agent_request(
                subagent="codex:guardian",
                inbound_protocol="responses",
            )
        )


if __name__ == "__main__":
    unittest.main()
