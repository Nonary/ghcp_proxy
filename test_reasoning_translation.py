import asyncio
import json
import unittest

import format_translation
import proxy


class ReasoningTranslationTests(unittest.IsolatedAsyncioTestCase):
    def test_ensure_codex_reasoning_header(self):
        self.assertEqual(
            format_translation.ensure_codex_reasoning_header("Analyzing codebase"),
            "**Thinking**\n\nAnalyzing codebase",
        )
        self.assertEqual(
            format_translation.ensure_codex_reasoning_header("**Thinking**\n\nAlready bold"),
            "**Thinking**\n\nAlready bold",
        )
        self.assertEqual(
            format_translation.ensure_codex_reasoning_header("# Header\n\nSome text"),
            "# Header\n\nSome text",
        )
        self.assertEqual(
            format_translation.ensure_codex_reasoning_header(""),
            "",
        )

    def test_normalize_reasoning_item_for_client(self):
        # Item with only content (e.g. from standard OpenAI / Copilot)
        item = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Step 1 reasoning"}],
        }
        format_translation.normalize_reasoning_item_for_client(item)
        self.assertEqual(
            item["summary"],
            [{"type": "summary_text", "text": "**Thinking**\n\nStep 1 reasoning"}],
        )
        self.assertEqual(
            item["content"],
            [{"type": "reasoning_text", "text": "**Thinking**\n\nStep 1 reasoning"}],
        )

        # Item with encrypted_content but empty summary (e.g. gpt-excel)
        encrypted_item = {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [],
            "encrypted_content": "gAAAAABk...",
        }
        format_translation.normalize_reasoning_item_for_client(encrypted_item)
        self.assertEqual(
            encrypted_item["summary"],
            [{"type": "summary_text", "text": "**Thinking**\n\n*Thinking process completed.*"}],
        )

    def test_chat_completion_to_response_extracts_reasoning(self):
        payload = {
            "id": "chatcmpl-1",
            "created": 12345,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here is the solution.",
                        "reasoning_content": "Let me think about how to solve this.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        response = format_translation.chat_completion_to_response(payload)
        output = response.get("output", [])
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(
            output[0]["summary"],
            [{"type": "summary_text", "text": "**Thinking**\n\nLet me think about how to solve this."}],
        )
        self.assertEqual(output[1]["type"], "message")
        self.assertEqual(output[1]["content"][0]["text"], "Here is the solution.")

    def test_anthropic_response_to_responses_formats_reasoning(self):
        payload = {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Evaluating approach."},
                {"type": "text", "text": "Plan ready."},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 15},
        }
        response = format_translation.anthropic_response_to_responses(payload)
        output = response.get("output", [])
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(
            output[0]["summary"],
            [{"type": "summary_text", "text": "**Thinking**\n\nEvaluating approach."}],
        )
        self.assertEqual(
            output[0]["content"],
            [{"type": "reasoning_text", "text": "**Thinking**\n\nEvaluating approach."}],
        )

    async def test_responses_reasoning_stream_transform(self):
        raw_events = [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "reasoning", "id": "rs_upstream", "summary": [], "content": []},
                },
            ),
            format_translation.sse_encode(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": "rs_upstream",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Thinking token 1. ",
                },
            ),
            format_translation.sse_encode(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": "rs_upstream",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Thinking token 2.",
                },
            ),
            format_translation.sse_encode(
                "response.reasoning_text.done",
                {
                    "type": "response.reasoning_text.done",
                    "item_id": "rs_upstream",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "Thinking token 1. Thinking token 2.",
                },
            ),
            format_translation.sse_encode(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "reasoning",
                        "id": "rs_upstream",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "Thinking token 1. Thinking token 2."}],
                    },
                },
            ),
            b"data: [DONE]\n\n",
        ]

        async def byte_generator():
            for ev in raw_events:
                yield ev

        transform = proxy._responses_reasoning_stream_transform()
        transformed_events = []
        async for chunk in transform(byte_generator()):
            for block in chunk.decode().strip().split("\n\n"):
                if not block.strip() or block == "data: [DONE]":
                    continue
                lines = block.splitlines()
                ev_name = lines[0].replace("event: ", "").strip()
                data = json.loads(lines[1].replace("data: ", ""))
                transformed_events.append((ev_name, data))

        event_names = [name for name, _ in transformed_events]
        self.assertIn("response.reasoning_summary_part.added", event_names)
        self.assertIn("response.reasoning_summary_text.delta", event_names)
        self.assertIn("response.reasoning_summary_text.done", event_names)

        # Check that header delta was sent
        summary_deltas = [
            d["delta"] for name, d in transformed_events if name == "response.reasoning_summary_text.delta"
        ]
        self.assertEqual(summary_deltas[0], "**Thinking**\n\n")
        self.assertEqual(summary_deltas[1], "Thinking token 1. ")

        # Check output_item.done has populated summary
        done_item = next(d["item"] for name, d in transformed_events if name == "response.output_item.done")
        self.assertEqual(
            done_item["summary"],
            [{"type": "summary_text", "text": "**Thinking**\n\nThinking token 1. Thinking token 2."}],
        )


if __name__ == "__main__":
    unittest.main()
