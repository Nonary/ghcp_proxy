import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "tools" / "analyze_cache_regressions.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("analyze_cache_regressions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyzeCacheRegressionsTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()

    def test_classify_reports_header_body_sequence_and_encrypted_reasoning_signals(self):
        prev = {
            "id": "req_prev",
            "cached": 52000,
            "fresh": 2000,
            "input_tokens": 54000,
            "status": 200,
            "cache_key_name": "prompt_cache_key",
            "cache_key": "lineage-a",
            "prompt_cache_key": "lineage-a",
            "resolved_model": "gpt-5.5",
            "upstream_fingerprints": {"tools": "tools-a", "reasoning": "reasoning-a"},
            "upstream_keys": ["input", "model", "tools"],
            "sequence": [{"item_hash": "a"}, {"item_hash": "b"}],
            "input_summary": {"kind": "list", "count": 2, "encrypted_reasoning_items": 1},
            "x_agent_task_id": "task-a",
            "x_interaction_id": "interaction-a",
        }
        cur = {
            **prev,
            "id": "req_cur",
            "cached": 1000,
            "fresh": 53000,
            "input_tokens": 54000,
            "upstream_fingerprints": {"tools": "tools-b", "reasoning": "reasoning-a"},
            "sequence": [{"item_hash": "a"}, {"item_hash": "changed"}],
            "input_summary": {"kind": "list", "count": 2, "encrypted_reasoning_items": 0},
            "x_agent_task_id": "task-b",
            "sanitization": {
                "encrypted_content_items_dropped": 1,
                "encrypted_content_preservation": False,
                "encrypted_content_strip_reason": "foreign_lineage",
            },
        }

        reasons = self.tool.classify(prev, cur)

        self.assertIn("usage:cached_tokens_dropped:51000", reasons)
        self.assertIn("headers:x_agent_task_id:task-a->task-b", reasons)
        self.assertIn("body_config:tools_changed", reasons)
        self.assertIn("prompt_sequence:mismatch@1", reasons)
        self.assertIn("encrypted_reasoning:dropped:foreign_lineage", reasons)

    def test_full_body_delta_reports_append_projection_and_prompt_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            body_dir = os.path.join(tmp, "request-bodies")
            os.mkdir(body_dir)
            prev_artifact = {
                "upstream_body": {
                    "model": "gpt-5.5",
                    "input": [{"type": "message", "role": "user", "content": "hello"}],
                },
                "outbound_headers": {"x-agent-task-id": "task-a"},
                "upstream_body_wire_sha256": "wire-a",
            }
            cur_artifact = {
                "upstream_body": {
                    "model": "gpt-5.5",
                    "input": [
                        {"type": "message", "role": "user", "content": "hello"},
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": "By the way, an update is available for GHCP Proxy.",
                        },
                    ],
                },
                "outbound_headers": {"x-agent-task-id": "task-b"},
                "upstream_body_wire_sha256": "wire-b",
            }
            with open(os.path.join(body_dir, "req_prev.json"), "w", encoding="utf-8") as f:
                json.dump(prev_artifact, f)
            with open(os.path.join(body_dir, "req_cur.json"), "w", encoding="utf-8") as f:
                json.dump(cur_artifact, f)

            delta = self.tool._full_body_delta(body_dir, "req_prev", "req_cur")
            header_delta = self.tool._safe_header_delta(prev_artifact, cur_artifact)

        self.assertTrue(delta["append_friendly_previous_is_prefix"])
        self.assertFalse(delta["append_friendly_config_changed"])
        self.assertEqual(delta["input_common_prefix_items"], 1)
        self.assertTrue(delta["proxy_notice_in_current"])
        self.assertEqual(header_delta, [{"header": "x-agent-task-id", "previous": "task-a", "current": "task-b"}])


if __name__ == "__main__":
    unittest.main()
