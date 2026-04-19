import unittest

from anthropic_stream import AnthropicStreamTranslator
from bridge_streams import ChatToResponsesStreamTranslator, ResponsesToAnthropicStreamTranslator
import proxy


class BridgeStreamTranslatorTests(unittest.TestCase):
    def test_chat_to_responses_stream_translator_emits_response_events(self):
        chunks = [
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"gpt-5.4","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"gpt-5.4","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Read","arguments":"{\\"file\\":\\"main.py\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            translator = ChatToResponsesStreamTranslator("claude-opus-4.6")
            body = b""
            async for event in translator.translate(byte_iter()):
                body += event if isinstance(event, bytes) else event.encode("utf-8")
            return body.decode("utf-8"), translator.build_response_payload()

        body, payload = proxy.asyncio.run(collect())

        self.assertIn("event: response.created", body)
        self.assertIn("event: response.output_text.delta", body)
        self.assertIn("event: response.function_call_arguments.delta", body)
        self.assertIn("event: response.completed", body)
        self.assertEqual(payload["output_text"], "hello")
        self.assertEqual(payload["model"], "claude-opus-4.6")
        # Reasoning absent: first output is the message, second is the tool call.
        self.assertEqual(payload["output"][0]["type"], "message")
        self.assertEqual(payload["output"][1]["type"], "function_call")
        self.assertEqual(payload["usage"]["input_tokens"], 12)

    def test_chat_to_responses_stream_translator_relays_thinking_as_reasoning(self):
        chunks = [
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-opus-4.6","choices":[{"delta":{"thinking":"let me "},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-opus-4.6","choices":[{"delta":{"thinking":"think"},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-opus-4.6","choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            translator = ChatToResponsesStreamTranslator("claude-opus-4.6")
            body = b""
            async for event in translator.translate(byte_iter()):
                body += event if isinstance(event, bytes) else event.encode("utf-8")
            return body.decode("utf-8"), translator.build_response_payload()

        body, payload = proxy.asyncio.run(collect())

        self.assertIn("event: response.reasoning_summary_text.delta", body)
        self.assertIn("event: response.reasoning_summary_text.done", body)
        # Reasoning item is added before the message item.
        self.assertEqual(payload["output"][0]["type"], "reasoning")
        # The proxy prepends a synthetic "**Thinking**" bold header so Codex's
        # `new_reasoning_summary_block` renders the cell instead of marking it
        # transcript-only. The original reasoning text follows the header.
        self.assertEqual(
            payload["output"][0]["summary"],
            [{"type": "summary_text", "text": "**Thinking**\n\nlet me think"}],
        )
        self.assertEqual(payload["output"][1]["type"], "message")
        self.assertEqual(payload["output_text"], "done")

    def test_responses_to_anthropic_stream_translator_emits_message_events(self):
        chunks = [
            (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hello"}\n\n'
            ).encode("utf-8"),
            (
                'event: response.output_item.added\n'
                'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{\\"file\\":\\"main.py\\"}"}}\n\n'
            ).encode("utf-8"),
            (
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-5.4","usage":{"input_tokens":12,"output_tokens":3,"input_tokens_details":{"cached_tokens":4}}}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            translator = ResponsesToAnthropicStreamTranslator("gpt-5.4")
            body = b""
            async for event in translator.translate(byte_iter()):
                body += event if isinstance(event, bytes) else event.encode("utf-8")
            return body.decode("utf-8"), translator.build_response_payload()

        body, payload = proxy.asyncio.run(collect())

        self.assertIn("event: message_start", body)
        self.assertIn("event: content_block_delta", body)
        self.assertIn("event: message_stop", body)
        self.assertEqual(payload["id"], "resp_123")
        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertEqual(payload["content"][0], {"type": "text", "text": "hello"})
        self.assertEqual(payload["content"][1]["type"], "tool_use")
        self.assertEqual(payload["usage"]["input_tokens"], 8)
        self.assertEqual(payload["usage"]["cache_read_input_tokens"], 4)

    def test_responses_to_anthropic_stream_translator_relays_reasoning(self):
        chunks = [
            (
                'event: response.reasoning_summary_text.delta\n'
                'data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"thinking "}\n\n'
            ).encode("utf-8"),
            (
                'event: response.reasoning_summary_text.delta\n'
                'data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"hard"}\n\n'
            ).encode("utf-8"),
            (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"answer"}\n\n'
            ).encode("utf-8"),
            (
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"id":"resp_xyz","model":"gpt-5.4","usage":{"input_tokens":3,"output_tokens":4}}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            translator = ResponsesToAnthropicStreamTranslator("gpt-5.4")
            body = b""
            async for event in translator.translate(byte_iter()):
                body += event if isinstance(event, bytes) else event.encode("utf-8")
            return body.decode("utf-8"), translator.build_response_payload()

        body, payload = proxy.asyncio.run(collect())

        self.assertIn('"type":"thinking"', body)
        self.assertIn('"type":"thinking_delta"', body)
        # Thinking block must be closed before the text block opens.
        first_thinking_stop = body.index("content_block_stop")
        first_text_start = body.index('"type":"text"')
        self.assertLess(first_thinking_stop, first_text_start)
        self.assertEqual(payload["content"][0], {"type": "thinking", "thinking": "thinking hard"})
        self.assertEqual(payload["content"][1], {"type": "text", "text": "answer"})

    def test_anthropic_stream_translator_relays_thinking_blocks(self):
        chunks = [
            (
                'event: message\n'
                'data: {"id":"chatcmpl_xyz","model":"claude-opus-4.6","choices":[{"delta":{"thinking":"step 1 "},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_xyz","model":"claude-opus-4.6","choices":[{"delta":{"reasoning_content":"step 2"},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_xyz","model":"claude-opus-4.6","choices":[{"delta":{"content":"final"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            translator = AnthropicStreamTranslator("claude-opus-4.6")
            body = b""
            async for event in translator.translate(byte_iter()):
                body += event if isinstance(event, bytes) else event.encode("utf-8")
            return body.decode("utf-8"), translator.build_response_payload()

        body, payload = proxy.asyncio.run(collect())

        self.assertIn('"type":"thinking"', body)
        self.assertIn('"type":"thinking_delta"', body)
        self.assertIn('"thinking":"step 1 "', body)
        self.assertIn('"thinking":"step 2"', body)
        self.assertIn("event: message_stop", body)
        # Final response payload still surfaces the visible text.
        self.assertEqual(payload["content"], [{"type": "text", "text": "final"}])



class ReasoningTextExtractionTests(unittest.TestCase):
    """Regression coverage for the ``delta.reasoning_text`` shape.

    Copilot's Anthropic-fronted ``/chat/completions`` endpoint emits Opus
    extended-thinking deltas as ``delta.reasoning_text`` (singular string),
    not ``delta.thinking`` / ``delta.reasoning_content`` / ``delta.reasoning``.
    Without this branch, every Opus thought was silently dropped before the
    SSE translator could relay it to Codex.
    """

    def test_extract_reasoning_from_chat_delta_handles_reasoning_text(self):
        from format_translation import extract_reasoning_from_chat_delta
        self.assertEqual(extract_reasoning_from_chat_delta({"reasoning_text": " hmm"}), " hmm")

    def test_chat_to_responses_translator_relays_reasoning_text(self):
        from bridge_streams import ChatToResponsesStreamTranslator

        chunks = [
            (
                'data: {"id":"chatcmpl_xyz","model":"claude-opus-4.7",'
                '"choices":[{"delta":{"role":"assistant","reasoning_text":" Let me"}}]}\n\n'
            ).encode("utf-8"),
            (
                'data: {"id":"chatcmpl_xyz","model":"claude-opus-4.7",'
                '"choices":[{"delta":{"reasoning_text":" think."}}]}\n\n'
            ).encode("utf-8"),
            (
                'data: {"id":"chatcmpl_xyz","model":"claude-opus-4.7",'
                '"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            translator = ChatToResponsesStreamTranslator("claude-opus-4.7")
            body = b""
            async for event in translator.translate(byte_iter()):
                body += event if isinstance(event, bytes) else event.encode("utf-8")
            return body.decode("utf-8"), translator.build_response_payload(), translator.reasoning_text

        body, payload, reasoning_text = proxy.asyncio.run(collect())

        # The reasoning summary stream must include both deltas with summary_index 0.
        self.assertIn('"type":"response.reasoning_summary_text.delta"', body)
        self.assertIn('"summary_index":0', body)
        self.assertIn('"delta":" Let me"', body)
        self.assertIn('"delta":" think."', body)
        # The synthetic "**Thinking**" header must precede the model's
        # reasoning so Codex's `new_reasoning_summary_block` renders the cell
        # rather than dropping it as transcript-only.
        self.assertIn('"delta":"**Thinking**\\n\\n"', body)
        thinking_pos = body.index('"delta":"**Thinking**\\n\\n"')
        first_real_delta = body.index('"delta":" Let me"')
        self.assertLess(thinking_pos, first_real_delta)
        # Final aggregated reasoning text is preserved end-to-end (with the
        # synthetic Codex-rendering bold header prepended).
        self.assertEqual(reasoning_text, "**Thinking**\n\n Let me think.")
        # Build_response_payload should expose the reasoning summary first.
        self.assertEqual(payload["output"][0]["type"], "reasoning")
        self.assertEqual(
            payload["output"][0]["summary"][0]["text"],
            "**Thinking**\n\n Let me think.",
        )
        # Codex's `ResponseItem::Reasoning` deserializer requires `id` and
        # `encrypted_content` to be present on every reasoning item it sees in
        # `response.output_item.added` / `response.output_item.done` — without
        # them serde silently drops the item, the turn never has an active
        # reasoning item, and every subsequent `ReasoningSummaryDelta` is
        # discarded by `error_or_panic("ReasoningSummaryDelta without active
        # item")`. That is the exact failure mode that hides Opus 4.7 thoughts
        # in Codex even though the upstream stream relays the deltas.
        import json as _json
        self.assertIn('"type":"response.output_item.added"', body)
        added_lines = [
            line for line in body.splitlines()
            if line.startswith('data: {') and 'response.output_item.added' in line
        ]
        self.assertTrue(added_lines, "expected response.output_item.added event")
        added_payload = _json.loads(added_lines[0][len("data: "):])
        added_item = added_payload["item"]
        self.assertEqual(added_item["type"], "reasoning")
        self.assertIsInstance(added_item.get("id"), str)
        self.assertTrue(added_item["id"], "reasoning item id must be non-empty")
        self.assertIn("encrypted_content", added_item)
        # build_response_payload must also carry the same shape so that Codex
        # can replay the rollout.
        self.assertIn("id", payload["output"][0])
        self.assertIn("encrypted_content", payload["output"][0])


if __name__ == "__main__":
    unittest.main()
