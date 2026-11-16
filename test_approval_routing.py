import json
import unittest

from codex_agent_compat import (
    codex_subagent_identity,
    codex_subagent_role,
    normalize_codex_agent_tools,
)
from initiator_policy import is_approval_agent_request


class ApprovalRoutingTests(unittest.TestCase):
    def test_spawn_agent_compat_accepts_any_catalog_model_or_inheritance(self):
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

        normalized = normalize_codex_agent_tools(body)
        tool = normalized["tools"][0]

        self.assertIn("active GHCP Proxy/Codex catalog", tool["description"])
        model_description = tool["parameters"]["properties"]["model"]["description"]
        self.assertIn("active GHCP Proxy/Codex model catalog", model_description)
        self.assertIn("Any model id in the active", model_description)
        self.assertIn("omit to inherit", model_description)
        self.assertNotIn("gpt-6-luna", str(tool))
        self.assertIn("set this false or omit it", tool["parameters"]["properties"]["fork_context"]["description"])

        normalized_again = normalize_codex_agent_tools(normalized)
        self.assertEqual(normalized_again, normalized)

    def test_collaboration_spawn_preserves_any_model_and_fork_none(self):
        body = {
            "tools": [{
                "namespace": "collaboration",
                "tools": [{
                    "name": "spawn_agent",
                    "description": (
                        "Spawn a child with a model override.\n\n"
                        "Available model overrides (optional; inherited parent model is preferred):\n"
                        "- `gpt-5.6-luna-excel`: An example.\n\n"
                        "Child agents inherit settings by default."
                    ),
                    "parameters": {"properties": {
                        "fork_turns": {"type": "string", "enum": ["all", "none"]},
                        "model": {"type": "string", "description": "Optional override."},
                        "reasoning_effort": {"type": "string"},
                    }},
                }],
            }],
        }
        normalized = normalize_codex_agent_tools(body)
        tool = normalized["tools"][0]["tools"][0]
        self.assertIn("exact id of any model", tool["description"])
        self.assertIn("Set `reasoning_effort` separately", tool["description"])
        self.assertIn("Child agents inherit settings by default.", tool["description"])
        self.assertNotIn("Available model overrides", tool["description"])
        self.assertNotIn("An example", tool["description"])
        self.assertIn("omit to inherit", tool["parameters"]["properties"]["model"]["description"])
        self.assertIn("use `none`", tool["parameters"]["properties"]["fork_turns"]["description"])
        self.assertNotIn("Available model slugs", str(tool))
        self.assertIn("Available model overrides", body["tools"][0]["tools"][0]["description"])

    def test_unrelated_tool_is_not_changed(self):
        body = {"tools": [{"name": "lookup", "parameters": {}}]}
        self.assertIs(normalize_codex_agent_tools(body), body)

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
