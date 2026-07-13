import json
import os
import tempfile
import unittest

import proxy
import util


class RequestPromptArchiveTests(unittest.TestCase):
    def setUp(self):
        self._old_archive_dir = os.environ.get("GHCP_REQUEST_PROMPT_ARCHIVE_DIR")
        self._temp_dir = tempfile.TemporaryDirectory()
        os.environ["GHCP_REQUEST_PROMPT_ARCHIVE_DIR"] = self._temp_dir.name

    def tearDown(self):
        if self._old_archive_dir is None:
            os.environ.pop("GHCP_REQUEST_PROMPT_ARCHIVE_DIR", None)
        else:
            os.environ["GHCP_REQUEST_PROMPT_ARCHIVE_DIR"] = self._old_archive_dir
        with proxy._REQUEST_PROMPT_LOCK:
            proxy._REQUEST_PROMPT_ACTIVE_IDS.clear()
        self._temp_dir.cleanup()

    def test_extract_request_prompt_text_formats_readable_transcript(self):
        body = {
            "instructions": "Be concise.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Explain the latest error."}],
                },
                {
                    "type": "custom_tool_call",
                    "name": "Read",
                    "input": {"path": "/tmp/example.py"},
                },
            ],
        }

        transcript = util.extract_request_prompt_text(body)

        self.assertIn("INSTRUCTIONS:\nBe concise.", transcript)
        self.assertIn("USER:\nExplain the latest error.", transcript)
        self.assertIn("TOOL CALL READ:\n/tmp/example.py", transcript)

    def test_request_prompt_api_falls_back_to_archived_prompt_text(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Archive this exact prompt."}],
                }
            ]
        }
        proxy._save_request_prompt_record("request/with unsafe chars", "/v1/responses", body)

        response = proxy.asyncio.run(proxy.request_prompt_api("request/with unsafe chars"))
        payload = json.loads(response.body)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["path"], "/v1/responses")
        self.assertIn("Archive this exact prompt.", payload["prompt_text"])
        self.assertIn("Archive this exact prompt.", payload["request_prompt"]["user"])

    def test_request_prompt_api_resolves_client_request_id_to_archived_prompt(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Resolve this client request."}],
                }
            ]
        }
        proxy._save_request_prompt_record("server-request-id", "/v1/responses", body)
        original_snapshot = proxy.usage_tracker.snapshot_usage_events
        proxy.usage_tracker.snapshot_usage_events = lambda: [
            {
                "request_id": "server-request-id",
                "client_request_id": "client-request-id",
            }
        ]
        try:
            response = proxy.asyncio.run(proxy.request_prompt_api("client-request-id"))
        finally:
            proxy.usage_tracker.snapshot_usage_events = original_snapshot

        payload = json.loads(response.body)
        self.assertTrue(payload["available"])
        self.assertIn("Resolve this client request.", payload["prompt_text"])

    def test_prune_request_prompt_archive_keeps_only_recent_request_ids(self):
        proxy._save_request_prompt_record("keep-me", "/v1/responses", {"input": "keep"})
        proxy._save_request_prompt_record("drop-me", "/v1/responses", {"input": "drop"})
        unrelated_json = os.path.join(self._temp_dir.name, "unrelated-settings.json")
        with open(unrelated_json, "w", encoding="utf-8") as handle:
            json.dump({"owner": "not request prompt archive"}, handle)

        proxy._prune_request_prompt_archive({"keep-me"})

        self.assertIsNotNone(proxy._load_request_prompt_record("keep-me"))
        self.assertIsNone(proxy._load_request_prompt_record("drop-me"))
        self.assertTrue(os.path.exists(unrelated_json))
