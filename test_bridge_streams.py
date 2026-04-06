import unittest

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
        self.assertEqual(payload["output"][1]["type"], "function_call")
        self.assertEqual(payload["usage"]["input_tokens"], 12)

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


if __name__ == "__main__":
    unittest.main()
