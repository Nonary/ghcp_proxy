import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import trace_prompt_security


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

    def test_full_body_delta_decrypts_encrypted_body_artifacts_from_env_file(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, ".env")
            body_dir = os.path.join(tmp, "request-bodies")
            os.mkdir(body_dir)
            salt = trace_prompt_security.new_salt()
            key = trace_prompt_security.derive_key("pw", salt)
            prev_body = {
                "model": "gpt-5.5",
                "input": [{"type": "message", "role": "user", "content": "hello"}],
            }
            cur_body = {
                "model": "gpt-5.5",
                "input": [
                    {"type": "message", "role": "user", "content": "hello"},
                    {"type": "message", "role": "assistant", "content": "answer"},
                ],
            }
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("GHCP_TRACE_PROMPT_PASSWORD=pw\n")
            with open(os.path.join(body_dir, "req_prev.json"), "w", encoding="utf-8") as f:
                json.dump({"upstream_body": trace_prompt_security.encrypt_payload(prev_body, key, salt)}, f)
            with open(os.path.join(body_dir, "req_cur.json"), "w", encoding="utf-8") as f:
                json.dump({"upstream_body": trace_prompt_security.encrypt_payload(cur_body, key, salt)}, f)

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(self.tool.trace_prompt_decryption.os, "getcwd", return_value=tmp),
            ):
                self.tool._TRACE_DECRYPTOR = None
                delta = self.tool._full_body_delta(body_dir, "req_prev", "req_cur")
                status = self.tool._trace_decryptor().status_line()

        self.assertTrue(delta["append_friendly_previous_is_prefix"])
        self.assertEqual(delta["input_common_prefix_items"], 1)
        self.assertIn("decrypted=2", status)

    def test_parse_decrypts_trace_request_body_from_env_file(self):
        try:
            trace_prompt_security._crypto_primitives()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, ".env")
            trace_path = os.path.join(tmp, "request-trace.jsonl")
            salt = trace_prompt_security.new_salt()
            key = trace_prompt_security.derive_key("pw", salt)
            request_body = {"prompt_cache_key": "lineage-a", "model": "gpt-5.5"}
            started = {
                "event": "request_started",
                "request_id": "req_1",
                "time": "2026-05-04T00:00:00Z",
                "resolved_model": "gpt-5.5",
                "request_body": trace_prompt_security.encrypt_payload(request_body, key, salt),
            }
            finished = {
                "event": "request_finished",
                "request_id": "req_1",
                "time": "2026-05-04T00:00:01Z",
                "response": {"usage": {"input_tokens": 100, "cached_input_tokens": 80}},
            }
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("GHCP_TRACE_PROMPT_PASSWORD=pw\n")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(started) + "\n")
                f.write(json.dumps(finished) + "\n")

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(self.tool.trace_prompt_decryption.os, "getcwd", return_value=tmp),
            ):
                self.tool._TRACE_DECRYPTOR = None
                rows, _keepalives = self.tool.parse(trace_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cache_key_name"], "prompt_cache_key")
        self.assertEqual(rows[0]["cache_key"], "lineage-a")


if __name__ == "__main__":
    unittest.main()
