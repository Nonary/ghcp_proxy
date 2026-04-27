import pytest

import format_translation


def test_normalize_upstream_model_name_strips_only_provider_prefix_once():
    assert (
        format_translation.normalize_upstream_model_name("  Anthropic/Claude/Sonnet/Test  ")
        == "claude/sonnet/test"
    )


def test_resolve_copilot_model_name_special_cases_and_fallbacks_are_exact():
    assert format_translation.resolve_copilot_model_name("claude-opus-4.6") == "claude-opus-4.6"
    assert format_translation.resolve_copilot_model_name("claude-opus-4.7") == "claude-opus-4.7"
    assert format_translation.resolve_copilot_model_name("claude-sonnet-4.6") == "claude-sonnet-4.6"
    assert format_translation.resolve_copilot_model_name("claude-haiku-4.5") == "claude-haiku-4.5"
    assert format_translation.resolve_copilot_model_name("anthropic/experimental-opus") == "claude-opus-4.7"


def test_anthropic_effort_helper_accepts_medium_exactly():
    assert format_translation._anthropic_effort_to_reasoning_effort({"effort": " medium "}) == "medium"


def test_anthropic_thinking_requires_integer_positive_budget():
    assert (
        format_translation._anthropic_thinking_to_reasoning_effort(
            {"type": "enabled", "budget_tokens": "10000"}
        )
        == format_translation.CLAUDE_DEFAULT_REASONING_EFFORT
    )

    assert (
        format_translation._anthropic_thinking_to_reasoning_effort(
            {"type": "enabled", "budget_tokens": 10000.0}
        )
        == format_translation.CLAUDE_DEFAULT_REASONING_EFFORT
    )


def test_normalize_anthropic_cache_control_prefers_non_empty_type():
    assert format_translation._normalize_anthropic_cache_control({"type": "ephemeral"}) == {
        "type": "ephemeral"
    }
    assert format_translation._normalize_anthropic_cache_control({"type": "ttl", "ttl": "5m"}) == {
        "type": "ttl"
    }


@pytest.mark.parametrize(
    ("item_type", "expected"),
    [
        ("text", True),
        ("input_text", True),
        ("output_text", True),
        ("image", False),
    ],
)
def test_response_content_text_type_contract(item_type, expected):
    assert format_translation._response_content_text_type(item_type) is expected
