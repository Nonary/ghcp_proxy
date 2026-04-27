import pytest

import format_translation


def test_response_content_text_type_accepts_plain_text_exactly():
    assert format_translation._response_content_text_type("text") is True
    assert format_translation._response_content_text_type("TEXT") is False


def test_response_content_item_to_chat_preserves_text_field_fallbacks():
    assert format_translation._response_content_item_to_chat(
        {"type": "input_text", "input_text": "input only"}
    ) == {"type": "text", "text": "input only"}
    assert format_translation._response_content_item_to_chat(
        {"type": "output_text", "output_text": "output only"}
    ) == {"type": "text", "text": "output only"}


def test_response_content_item_to_chat_parses_pdf_data_url_parameters_exactly():
    assert format_translation._response_content_item_to_chat(
        {
            "type": "input_file",
            "filename": "with-params.pdf",
            "file_data": "data:application/pdf;name=ignored.pdf;base64,PDFDATA",
        }
    ) == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "PDFDATA",
        },
        "title": "with-params.pdf",
    }


def test_response_content_item_to_chat_requires_complete_image_base64_pair():
    with pytest.raises(ValueError) as missing_media_type:
        format_translation._response_content_item_to_chat(
            {"type": "input_image", "image_base64": "AAAA"}
        )
    assert (
        str(missing_media_type.value)
        == "Responses input_image blocks must include image_url or image_base64/media_type"
    )

    with pytest.raises(ValueError) as missing_image_base64:
        format_translation._response_content_item_to_chat(
            {"type": "input_image", "media_type": "image/png"}
        )
    assert (
        str(missing_image_base64.value)
        == "Responses input_image blocks must include image_url or image_base64/media_type"
    )


def test_response_output_items_to_anthropic_content_tolerates_missing_message_content():
    assert format_translation._response_output_items_to_anthropic_content(
        [
            {"type": "message", "role": "assistant"},
            {"type": "message", "role": "user", "content": [{"text": "ignored"}]},
        ]
    ) == [{"type": "text", "text": ""}]


def test_responses_request_to_chat_skips_non_dict_input_and_keeps_following_items():
    translated = format_translation.responses_request_to_chat(
        {
            "model": "openai/gpt-5.4",
            "input": [
                "skip me",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "kept"}],
                },
            ],
        }
    )

    assert translated["messages"] == [{"role": "user", "content": "kept"}]


def test_responses_request_to_chat_preserves_string_arguments_exactly():
    translated = format_translation.responses_request_to_chat(
        {
            "model": "openai/gpt-5.4",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_json",
                    "name": "Lookup",
                    "arguments": '{"already":"json"}',
                }
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_json",
                    "type": "function",
                    "function": {
                        "name": "Lookup",
                        "arguments": '{"already":"json"}',
                    },
                }
            ],
        }
    ]


def test_responses_request_to_chat_json_arguments_are_compact_utf8_and_default_empty():
    translated = format_translation.responses_request_to_chat(
        {
            "model": "openai/gpt-5.4",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_unicode",
                    "name": "Lookup",
                    "arguments": {"query": "café", "items": [1, 2]},
                },
                {
                    "type": "function_call",
                    "call_id": "call_missing",
                    "name": "Read",
                },
            ],
        }
    )

    assert translated["messages"][0]["tool_calls"][0]["function"]["arguments"] == (
        '{"query":"café","items":[1,2]}'
    )
    assert translated["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_responses_tool_helpers_raise_exact_errors_for_invalid_shapes():
    with pytest.raises(ValueError) as missing_name:
        format_translation._responses_tool_to_chat({"type": "function", "parameters": {}})
    assert str(missing_name.value) == "Responses function tools must include a name"

    with pytest.raises(ValueError) as bad_choice:
        format_translation.responses_tool_choice_to_chat({"type": "function", "function": {}})
    assert str(bad_choice.value) == "Unsupported Responses tool_choice value"


def test_responses_request_to_chat_defers_until_all_pending_tool_calls_finish():
    translated = format_translation.responses_request_to_chat(
        {
            "model": "openai/gpt-5.4",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": {},
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "Search",
                    "arguments": {},
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "after both"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "first",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "second",
                },
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "Search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "first"},
        {"role": "tool", "tool_call_id": "call_2", "content": "second"},
        {"role": "user", "content": "after both"},
    ]


def test_responses_request_to_chat_joins_list_tool_output_without_separator():
    translated = format_translation.responses_request_to_chat(
        {
            "model": "openai/gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_parts",
                    "output": [
                        {"type": "output_text", "text": "left"},
                        {"type": "output_text", "text": "right"},
                    ],
                }
            ],
        }
    )

    assert translated["messages"] == [
        {"role": "tool", "tool_call_id": "call_parts", "content": "leftright"}
    ]


def test_responses_request_to_chat_invalid_function_items_use_exact_errors():
    with pytest.raises(ValueError) as call_error:
        format_translation.responses_request_to_chat(
            {
                "model": "openai/gpt-5.4",
                "input": [{"type": "function_call", "name": 123}],
            }
        )
    assert str(call_error.value) == "Responses function_call items must include a name"

    with pytest.raises(ValueError) as output_error:
        format_translation.responses_request_to_chat(
            {
                "model": "openai/gpt-5.4",
                "input": [{"type": "function_call_output", "call_id": 123}],
            }
        )
    assert str(output_error.value) == "Responses function_call_output items must include call_id"


def test_chat_tool_to_responses_exact_shape_and_dangerous_boundary():
    assert format_translation._chat_tool_to_responses({"type": "web_search"}) is None
    assert (
        format_translation._chat_tool_to_responses(
            {
                "type": "function",
                "function": {
                    "name": "mcp__ide__executeCode",
                    "description": "do not forward",
                    "parameters": {"type": "object"},
                },
            }
        )
        is None
    )
    assert format_translation._chat_tool_to_responses(
        {
            "type": "FUNCTION",
            "function": {
                "name": "Read",
                "description": 42,
                "parameters": "bad",
            },
        }
    ) == {
        "type": "function",
        "name": "Read",
        "description": "",
        "parameters": {"type": "object", "properties": {}},
    }
    with pytest.raises(ValueError) as missing_name_error:
        format_translation._chat_tool_to_responses(
            {"type": "function", "function": {"name": 123}}
        )
    assert str(missing_name_error.value) == "Chat function tools must include a name"


def test_chat_content_item_to_response_content_requires_image_url_object():
    assert format_translation._chat_content_item_to_response_content(
        {"type": "TEXT", "text": "hello"}
    ) == {"type": "input_text", "text": "hello"}
    assert format_translation._chat_content_item_to_response_content(
        {"type": "text", "text": "hello"}, role="assistant"
    ) == {"type": "output_text", "text": "hello", "annotations": []}
    assert format_translation._chat_content_item_to_response_content(
        {
            "type": "text",
            "text": "cached",
            "copilot_cache_control": {"type": "ephemeral"},
        }
    ) == {"type": "input_text", "text": "cached"}
    assert (
        format_translation._chat_content_item_to_response_content(
            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}
        )
        == {"type": "input_image", "image_url": "https://example.invalid/a.png"}
    )
    assert (
        format_translation._chat_content_item_to_response_content(
            {
                "type": "image_url",
                "text": "not text content",
                "image_url": {"url": "https://example.invalid/with-text.png"},
            }
        )
        == {"type": "input_image", "image_url": "https://example.invalid/with-text.png"}
    )
    assert (
        format_translation._chat_content_item_to_response_content(
            {"type": "image_url", "image_url": "https://example.invalid/a.png"}
        )
        is None
    )
    assert (
        format_translation._chat_content_item_to_response_content({"type": "image_url"})
        is None
    )
