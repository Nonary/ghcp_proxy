import asyncio
import json
import unittest

import update_notice


class UpdateNoticeTests(unittest.TestCase):
    def _collect(self, chunks, protocol, notice="Update available."):
        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            body = b""
            async for event in update_notice.inject_text_notice(byte_iter(), protocol, notice):
                body += event
            return body.decode("utf-8")

        return asyncio.run(collect())

    def test_chat_notice_is_inserted_before_terminal_chunk(self):
        body = self._collect(
            [
                b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            "chat",
        )

        self.assertLess(body.index("answer"), body.index("Update available."))
        self.assertLess(body.index("Update available."), body.index('"finish_reason":"stop"'))
        self.assertLess(body.index('"finish_reason":"stop"'), body.index("[DONE]"))

    def test_chat_notice_is_inserted_before_terminal_chunk_with_usage_after_finish(self):
        body = self._collect(
            [
                b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
                b"data: [DONE]\n\n",
            ],
            "chat",
        )

        self.assertLess(body.index("answer"), body.index("Update available."))
        self.assertLess(body.index("Update available."), body.index('"finish_reason":"stop"'))
        self.assertLess(body.index('"finish_reason":"stop"'), body.index('"usage"'))
        self.assertLess(body.index('"usage"'), body.index("[DONE]"))

    def test_responses_notice_is_inserted_before_completed_event(self):
        body = self._collect(
            [
                b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n\n',
                b'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}\n\n',
                b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"message","content":[{"type":"output_text","text":"answer"}]}],"output_text":"answer"}}\n\n',
                b"data: [DONE]\n\n",
            ],
            "responses",
        )

        self.assertLess(body.index("Update available."), body.index("response.completed"))
        completed_line = [line for line in body.splitlines() if line.startswith("data: {") and "response.completed" in line][-1]
        payload = json.loads(completed_line.removeprefix("data: "))
        self.assertTrue(payload["response"]["output_text"].endswith("Update available.\n\n"))
        self.assertEqual(payload["response"]["output"][-1]["content"][0]["text"], "Update available.\n\n")

    def test_anthropic_notice_is_inserted_before_message_stop(self):
        body = self._collect(
            [
                b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","content":[]}}\n\n',
                b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}\n\n',
                b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
                b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n',
                b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ],
            "anthropic",
        )

        self.assertLess(body.index("answer"), body.index("Update available."))
        self.assertLess(body.index("Update available."), body.index("message_stop"))


if __name__ == "__main__":
    unittest.main()
