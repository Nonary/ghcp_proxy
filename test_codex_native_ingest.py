"""Tests for codex_native_ingest.

Builds a synthetic Codex rollout JSONL fixture, runs the ingestor against
it, and asserts that the emitted events:
- carry the `native_source = "codex_native"` marker,
- normalize Codex's `last_token_usage` into our usage shape,
- compute a non-zero cost for known GPT models,
- are deduplicated across repeated scans via the cursor file,
- are classified as source `"codex_native"` by `_usage_event_source`.
"""

from __future__ import annotations

import json
import os
import unittest
import unittest.mock
from pathlib import Path
from uuid import uuid4

import codex_native_ingest
from util import _usage_event_source


def _write_rollout(path: Path, *, session_id: str, model: str = "gpt-5.4", model_provider: str = "openai") -> None:
    lines = [
        {
            "timestamp": "2026-04-18T15:15:34.519Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": "2026-04-18T15:15:05.154Z",
                "cwd": "D:/example",
                "originator": "codex-tui",
                "cli_version": "0.121.0",
                "source": "cli",
                "model_provider": model_provider,
            },
        },
        {
            "timestamp": "2026-04-18T15:15:34.520Z",
            "type": "turn_context",
            "payload": {"model": model, "effort": "high"},
        },
        {
            "timestamp": "2026-04-18T15:15:39.000Z",
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": f"{session_id}-turn-1",
                "started_at": 1776525339,
            },
        },
        # token_count with info=null (session start) — should be skipped.
        {
            "timestamp": "2026-04-18T15:15:35.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": None,
                "rate_limits": {"plan_type": "pro", "limit_id": "codex"},
            },
        },
        # First real turn.
        {
            "timestamp": "2026-04-18T15:15:40.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 1000, "cached_input_tokens": 200,
                        "output_tokens": 50, "reasoning_output_tokens": 10,
                        "total_tokens": 1050,
                    },
                    "last_token_usage": {
                        "input_tokens": 1000, "cached_input_tokens": 200,
                        "output_tokens": 50, "reasoning_output_tokens": 10,
                        "total_tokens": 1050,
                    },
                    "model_context_window": 258400,
                },
                "rate_limits": {"plan_type": "pro", "limit_id": "codex"},
            },
        },
        # Second turn (delta against cumulative).
        {
            "timestamp": "2026-04-18T15:15:50.000Z",
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": f"{session_id}-turn-2",
                "started_at": 1776525350,
            },
        },
        {
            "timestamp": "2026-04-18T15:15:50.100Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 1500, "cached_input_tokens": 800,
                        "output_tokens": 80, "reasoning_output_tokens": 15,
                        "total_tokens": 1580,
                    },
                    "last_token_usage": {
                        "input_tokens": 500, "cached_input_tokens": 600,
                        "output_tokens": 30, "reasoning_output_tokens": 5,
                        "total_tokens": 530,
                    },
                    "model_context_window": 258400,
                },
                "rate_limits": {"plan_type": "pro", "limit_id": "codex"},
            },
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for record in lines:
            f.write(json.dumps(record))
            f.write("\n")


class CodexNativeIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path.cwd() / f".tmp_codex_native_{uuid4().hex}"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.tmp_root / "sessions" / "2026" / "04" / "18"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.cursor_path = self.tmp_root / "cursor.json"

        self._orig_sessions = codex_native_ingest.CODEX_SESSIONS_DIR
        self._orig_cursor = codex_native_ingest.CURSOR_FILE
        codex_native_ingest.CODEX_SESSIONS_DIR = str(self.tmp_root / "sessions")
        codex_native_ingest.CURSOR_FILE = str(self.cursor_path)

    def tearDown(self):
        codex_native_ingest.CODEX_SESSIONS_DIR = self._orig_sessions
        codex_native_ingest.CURSOR_FILE = self._orig_cursor
        # Best-effort cleanup; on Windows leave files if locked.
        for path in self.tmp_root.rglob("*"):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
        for path in sorted(self.tmp_root.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            self.tmp_root.rmdir()
        except OSError:
            pass

    def test_emits_one_event_per_real_turn(self):
        rollout = self.sessions_dir / "rollout-2026-04-18T10-15-05-sess-1.jsonl"
        _write_rollout(rollout, session_id="sess-1")
        emitted: list[dict] = []
        with unittest.mock.patch(
            "codex_native_ingest._codex_logs_service_tiers",
            return_value={
                "requested": "priority",
                "requested_source": "codex_logs_request",
                "effective": "default",
                "effective_source": "codex_logs_response_completed",
            },
        ):
            codex_native_ingest.scan_once(emitted.append)
        self.assertEqual(len(emitted), 2, "expected 2 turns (info=null skipped)")

        first = emitted[0]
        self.assertEqual(first["native_source"], "codex_native")
        self.assertEqual(first["session_id"], "sess-1")
        self.assertEqual(first["server_request_id"], "sess-1")
        self.assertEqual(first["native_requested_service_tier"], "priority")
        self.assertEqual(first["native_requested_service_tier_source"], "codex_logs_request")
        self.assertEqual(first["native_service_tier"], "default")
        self.assertEqual(first["native_service_tier_source"], "codex_logs_response_completed")
        self.assertEqual(first["native_turn_id"], "sess-1-turn-1")
        self.assertEqual(first["requested_model"], "gpt-5.4")
        self.assertEqual(first["initiator"], "user")
        # Codex reports input_tokens as the total including cached. Ingestor
        # subtracts cached so the proxy's cost math (which bills input + cached
        # separately) doesn't double-count the cached portion.
        self.assertEqual(first["usage"]["input_tokens"], 800)  # 1000 total - 200 cached
        self.assertEqual(first["usage"]["cached_input_tokens"], 200)
        self.assertEqual(first["usage"]["output_tokens"], 50)
        self.assertEqual(first["usage"]["reasoning_output_tokens"], 10)
        self.assertGreater(first["cost_usd"], 0.0, "should compute non-zero cost from gpt-5.4 pricing")
        self.assertEqual(first["native_plan_type"], "pro")
        self.assertEqual(first["native_origin"], "codex-tui")

        # Second turn uses per-turn delta, not cumulative. Fixture has
        # input=500, cached=600 — cached clamps to input, fresh=0.
        self.assertEqual(emitted[1]["usage"]["input_tokens"], 0)
        self.assertEqual(emitted[1]["usage"]["cached_input_tokens"], 500)
        self.assertEqual(emitted[1]["usage"]["output_tokens"], 30)

    def test_classified_as_codex_native_source(self):
        rollout = self.sessions_dir / "rollout-2026-04-18T10-15-05-sess-2.jsonl"
        _write_rollout(rollout, session_id="sess-2")
        emitted: list[dict] = []
        codex_native_ingest.scan_once(emitted.append)
        self.assertTrue(emitted)
        for ev in emitted:
            self.assertEqual(_usage_event_source(ev), "codex_native")

    def test_cursor_dedupes_repeat_scans(self):
        rollout = self.sessions_dir / "rollout-2026-04-18T10-15-05-sess-3.jsonl"
        _write_rollout(rollout, session_id="sess-3")
        emitted_first: list[dict] = []
        codex_native_ingest.scan_once(emitted_first.append)
        self.assertEqual(len(emitted_first), 2)

        emitted_second: list[dict] = []
        codex_native_ingest.scan_once(emitted_second.append)
        self.assertEqual(emitted_second, [], "second scan should emit nothing")

    def test_appended_turns_are_picked_up(self):
        rollout = self.sessions_dir / "rollout-2026-04-18T10-15-05-sess-4.jsonl"
        _write_rollout(rollout, session_id="sess-4")
        emitted: list[dict] = []
        codex_native_ingest.scan_once(emitted.append)
        self.assertEqual(len(emitted), 2)

        # Append a third turn.
        extra = {
            "timestamp": "2026-04-18T15:16:00.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 100, "cached_input_tokens": 0,
                        "output_tokens": 5, "reasoning_output_tokens": 0,
                        "total_tokens": 105,
                    },
                },
                "rate_limits": {"plan_type": "pro", "limit_id": "codex"},
            },
        }
        with rollout.open("a", encoding="utf-8") as f:
            f.write(json.dumps(extra))
            f.write("\n")

        more: list[dict] = []
        codex_native_ingest.scan_once(more.append)
        self.assertEqual(len(more), 1)
        # input=100, cached=0 — fresh stays at 100 with no clamping.
        self.assertEqual(more[0]["usage"]["input_tokens"], 100)
        self.assertEqual(more[0]["usage"]["cached_input_tokens"], 0)

    def test_cost_does_not_double_count_cached_tokens(self):
        """Regression: input_tokens reported by Codex is fresh+cached. The
        proxy bills input + cached separately, so we must subtract cached
        from input or the cached portion gets billed twice."""
        from constants import MODEL_PRICING
        from util import _usage_event_cost

        # 1000 total input where 800 is cached, 50 output, gpt-5.4.
        usage_after_normalize = codex_native_ingest._normalize_token_count_payload({
            "info": {
                "last_token_usage": {
                    "input_tokens": 1000, "cached_input_tokens": 800,
                    "output_tokens": 50, "reasoning_output_tokens": 0,
                    "total_tokens": 1050,
                },
            },
        })
        self.assertEqual(usage_after_normalize["input_tokens"], 200)
        self.assertEqual(usage_after_normalize["cached_input_tokens"], 800)

        entry = MODEL_PRICING["gpt-5.4"]
        expected = (
            (200 * entry["input_per_million"]) / 1_000_000
            + (800 * entry["cached_input_per_million"]) / 1_000_000
            + (50 * entry["output_per_million"]) / 1_000_000
        )
        actual = _usage_event_cost("gpt-5.4", usage_after_normalize)
        self.assertAlmostEqual(actual, expected, places=8)

    def test_fast_service_tier_doubles_estimated_cost(self):
        rollout = self.sessions_dir / "rollout-2026-04-18T10-15-05-sess-fast.jsonl"
        _write_rollout(rollout, session_id="sess-fast")

        emitted: list[dict] = []
        with unittest.mock.patch(
            "codex_native_ingest._codex_logs_service_tiers",
            return_value={
                "requested": "priority",
                "requested_source": "codex_logs_request",
                "effective": "default",
                "effective_source": "codex_logs_response_completed",
            },
        ):
            codex_native_ingest.scan_once(emitted.append)

        self.assertTrue(emitted)
        fast_event = emitted[0]
        self.assertEqual(fast_event["native_reasoning_effort"], "high")
        self.assertEqual(fast_event["native_requested_service_tier"], "priority")
        self.assertEqual(fast_event["native_service_tier"], "default")
        self.assertGreater(fast_event["cost_usd"], 0.0)

        fresh_cursor = self.tmp_root / "cursor-fresh.json"
        codex_native_ingest.CURSOR_FILE = str(fresh_cursor)
        baseline: list[dict] = []
        with unittest.mock.patch("codex_native_ingest._codex_logs_service_tiers", return_value={}):
            codex_native_ingest.scan_once(baseline.append)
        self.assertTrue(baseline)
        self.assertAlmostEqual(fast_event["cost_usd"], baseline[0]["cost_usd"] * 2, places=6)


    def test_skips_proxied_sessions(self):
        rollout = self.sessions_dir / "rollout-2026-04-18T10-15-05-sess-custom.jsonl"
        _write_rollout(rollout, session_id="sess-custom", model_provider="custom")
        
        emitted: list[dict] = []
        codex_native_ingest.scan_once(emitted.append)
        self.assertEqual(len(emitted), 0, "expected 0 turns because model_provider is custom")

if __name__ == "__main__":
    unittest.main()


