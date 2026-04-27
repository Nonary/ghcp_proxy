"""Focused regression tests for remaining preprocess/header mutants."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import messages_preprocess as mp
import request_headers as rh


def _mergeable_turn(text: str = "reminder") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output"},
            {"type": "text", "text": text},
        ],
    }


def test_interaction_id_uses_trimmed_session_or_generated_uuid():
    assert rh._interaction_id_for_session(" session-1 ") == "session-1"

    with mock.patch.object(rh.uuid, "uuid4", return_value="generated-id"):
        assert rh._interaction_id_for_session(None) == "generated-id"
        assert rh._interaction_id_for_session("   ") == "generated-id"


def test_apply_forwarded_headers_forwards_resolved_session_by_default():
    headers = {}
    request = SimpleNamespace(headers={})
    body = {"model": "gpt-5.4"}

    session_id = rh._apply_forwarded_request_headers(
        headers,
        request,
        body,
        session_id_resolver=lambda req, request_body: "resolved-session",
    )

    assert session_id == "resolved-session"
    assert headers["session_id"] == "resolved-session"
    assert headers["x-client-request-id"] == "resolved-session"


def test_chat_headers_use_user_interaction_type_when_policy_resolves_user():
    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/chat/completions"),
        headers={},
    )

    class UserPolicy:
        def resolve_chat_messages(self, *args, **kwargs):
            return "user"

    with mock.patch.object(rh.uuid, "uuid4", return_value="agent-task-id"):
        headers = rh.build_chat_headers_for_request(
            request,
            [{"role": "user", "content": "hello"}],
            "gpt-4.1",
            "test-key",
            initiator_policy=UserPolicy(),
            session_id_resolver=lambda req, body_arg=None: "session-id",
        )

    assert headers["X-Initiator"] == "user"
    assert headers["x-interaction-type"] == "conversation-user"
    assert headers["x-interaction-id"] == "session-id"


def test_merge_tool_result_defaults_to_processing_last_message():
    body = {"messages": [_mergeable_turn()]}

    mp.merge_tool_result_with_reminder(body)

    assert body["messages"][0]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output\n\nreminder"}
    ]


def test_prepare_messages_defaults_to_non_compact_merge_behavior():
    cleaned = mp.prepare_messages_passthrough_payload(
        {"messages": [_mergeable_turn("default merge")]},
        model_supports_adaptive=False,
    )

    assert cleaned["messages"][0]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output\n\ndefault merge"}
    ]


def test_merge_tool_result_noops_for_truthy_non_list_messages():
    body = {"messages": 1}

    assert mp.merge_tool_result_with_reminder(body) is body
    assert body == {"messages": 1}


def test_merge_tool_result_continues_after_skipped_message_shapes():
    body = {
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "skip"}]},
            {"role": "user", "content": "not-a-list"},
            _mergeable_turn("after skips"),
        ]
    }

    mp.merge_tool_result_with_reminder(body)

    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output\n\nafter skips"}
    ]


def test_merge_tool_result_skip_last_processes_all_prior_user_messages():
    body = {
        "messages": [
            _mergeable_turn("first"),
            _mergeable_turn("second"),
            _mergeable_turn("last"),
        ]
    }

    mp.merge_tool_result_with_reminder(body, skip_last_message=True)

    assert body["messages"][0]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output\n\nfirst"}
    ]
    assert body["messages"][1]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output\n\nsecond"}
    ]
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool output"},
        {"type": "text", "text": "last"},
    ]


def test_strip_tool_reference_boundary_continues_after_user_turn_without_reference():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Tool loaded."}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": [{"type": "tool_reference", "id": "ref"}]},
                    {"type": "text", "text": " Tool loaded. "},
                    {"type": "text", "text": "keep"},
                ],
            },
        ]
    }

    mp.strip_tool_reference_turn_boundary(body)

    assert body["messages"][0]["content"] == [{"type": "text", "text": "Tool loaded."}]
    assert body["messages"][1]["content"] == [
        {"type": "tool_result", "content": [{"type": "tool_reference", "id": "ref"}]},
        {"type": "text", "text": "keep"},
    ]


def test_detect_compact_type_ignores_truthy_non_list_messages():
    assert mp.detect_compact_type({"messages": 1}) == 0


def test_detect_compact_type_does_not_probe_candidate_without_last_message():
    with mock.patch.object(mp, "_is_compact_summary_request", wraps=mp._is_compact_summary_request) as summary:
        with mock.patch.object(mp, "_is_compact_auto_continue", wraps=mp._is_compact_auto_continue) as auto:
            assert mp.detect_compact_type({"messages": []}) == 0

    summary.assert_not_called()
    auto.assert_not_called()


def test_apply_adaptive_thinking_passes_normalized_incoming_effort_to_mapper():
    body = {
        "model": "claude-sonnet-4.5",
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": " Medium "},
    }

    with mock.patch.object(mp, "map_effort_for_model", side_effect=lambda model, effort: effort) as mapper:
        mp.apply_adaptive_thinking(body, supports_adaptive=True)

    mapper.assert_called_once_with("claude-sonnet-4.5", "medium")
    assert body["output_config"]["effort"] == "medium"


def test_apply_adaptive_thinking_passes_lowercase_default_effort_to_mapper():
    body = {"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]}

    with mock.patch.object(mp, "map_effort_for_model", side_effect=lambda model, effort: effort) as mapper:
        mp.apply_adaptive_thinking(body, supports_adaptive=True)

    mapper.assert_called_once_with("claude-sonnet-4.5", "high")
    assert body["output_config"]["effort"] == "high"


def test_apply_adaptive_thinking_normalizes_none_and_minimal_before_mapping():
    for effort in ("none", "minimal"):
        body = {
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": effort},
        }

        with mock.patch.object(mp, "map_effort_for_model", side_effect=lambda model, mapped: mapped) as mapper:
            mp.apply_adaptive_thinking(body, supports_adaptive=True)

        mapper.assert_called_once_with("claude-sonnet-4.5", "low")
        assert body["output_config"]["effort"] == "low"
