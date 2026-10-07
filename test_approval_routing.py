import json
import unittest

from codex_agent_compat import (
    codex_subagent_identity,
    codex_subagent_role,
    normalize_codex_agent_tools,
)
from initiator_policy import is_approval_agent_request


class ApprovalRoutingTests(unittest.TestCase):
    def test_spawn_agent_compat_uses_active_model_catalog_for_overrides(self):
        body = {
            "tools": [
                {
                    "name": "spawn_agent",
                    "description": "Spawn a worker.",
                    "parameters": {
                        "properties": {
                            "fork_context": {"type": "boolean"},
                            "model": {"type": "string", "description": "Model id."},
                        }
                    },
                }
            ]
        }

        normalized = normalize_codex_agent_tools(
            body,
            model_slugs=["gpt-6-luna", "gpt-5.6-sol", "invalid\nmodel", "gpt-6-luna"],
        )
        tool = normalized["tools"][0]

        self.assertIn("active GHCP Proxy/Codex model catalog", tool["description"])
        model_description = tool["parameters"]["properties"]["model"]["description"]
        self.assertIn("active GHCP Proxy/Codex model catalog", model_description)
        self.assertIn("Available model slugs in the active catalog", model_description)
        self.assertIn("`gpt-6-luna`", model_description)
        self.assertIn("`gpt-5.6-sol`", model_description)
        self.assertEqual(model_description.count("`gpt-6-luna`"), 1)
        self.assertNotIn("invalid", model_description)

        normalized_again = normalize_codex_agent_tools(
            normalized,
            model_slugs=["gpt-6-astra"],
        )
        self.assertEqual(normalized_again, normalized)

    def test_model_catalog_provider_is_not_called_without_spawn_agent(self):
        body = {"tools": [{"name": "lookup", "parameters": {}}]}

        normalized = normalize_codex_agent_tools(
            body,
            model_slugs_provider=lambda: self.fail("catalog should not be loaded"),
        )

        self.assertIs(normalized, body)

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
