"""Cache continuity for Copilot SDK sessions resumed from disk.

Item shapes follow the runtime's real model wire (probed against Copilot):
Sol chains ``previous_response_id`` over a WebSocket and only reveals its
reasoning in response output; a disk resume resends the whole history
without any reasoning items.
"""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anyio
import httpx
from copilot import copilot_request_handler
from copilot.session_events import SessionIdleData

import copilot_sdk_upstream as sdk
from sdk_reasoning_ledger import ReasoningLedger, anchor_key

MODEL = "gpt-5.6-sol"


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def call(call_id):
    return {"type": "function_call", "call_id": call_id, "name": "inspect", "arguments": "{}"}


def output(call_id):
    return {"type": "function_call_output", "call_id": call_id, "output": "42"}


def reasoning(name):
    # A response item carries a server id and no status beyond these fields.
    return {"content": [], "encrypted_content": f"opaque-{name}", "id": f"rs-{name}", "summary": [], "type": "reasoning"}


def answer(text, *, item_id="msg_runtime"):
    return {"type": "message", "id": item_id, "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


# One turn with two tool calls, then a new user turn, as the live session
# resends it after a WebSocket reconnect ...
LIVE_HISTORY = [user("first"), call("c1"), output("c1"), reasoning("a"), call("c2"), output("c2"),
                reasoning("b"), answer("done"), user("second")]
# ... and as the runtime resends it after a disk resume.
RESUMED_HISTORY = [item for item in LIVE_HISTORY if item["type"] != "reasoning"]


def record_turn_from_responses(ledger, session_id="s1"):
    """Record the turn the way Sol's WebSocket reveals it."""
    ledger.record_output(session_id, MODEL, anchor_key(user("first")), [call("c1")])
    ledger.record_output(session_id, MODEL, anchor_key(output("c1")), [reasoning("a"), {**call("c2"), "id": "fc-server"}])
    ledger.record_output(session_id, MODEL, anchor_key(output("c2")), [reasoning("b"), answer("done", item_id="server-id")])


class ReasoningLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ledger = self.new_ledger()

    def new_ledger(self, **kwargs):
        return ReasoningLedger(lambda: self.directory.name, **kwargs)

    def test_response_output_restores_the_live_history_after_a_disk_resume(self):
        record_turn_from_responses(self.ledger)
        restored, count = self.ledger.restore("s1", MODEL, RESUMED_HISTORY)
        self.assertEqual(count, 2)
        self.assertEqual(restored, LIVE_HISTORY)

    def test_request_input_records_http_reasoning(self):
        self.ledger.record_input("s1", MODEL, LIVE_HISTORY)
        restored, count = self.ledger.restore("s1", MODEL, RESUMED_HISTORY)
        self.assertEqual((restored, count), (LIVE_HISTORY, 2))

    def test_runs_survive_a_restart(self):
        record_turn_from_responses(self.ledger)
        restored, count = self.new_ledger().restore("s1", MODEL, RESUMED_HISTORY)
        self.assertEqual((restored, count), (LIVE_HISTORY, 2))

    def test_repeated_requests_do_not_grow_the_file(self):
        for _ in range(5):
            self.ledger.record_input("s1", MODEL, LIVE_HISTORY)
        with open(os.path.join(self.directory.name, "s1.jsonl"), encoding="utf-8") as handle:
            self.assertEqual(len(handle.readlines()), 2)

    def test_other_models_sessions_and_complete_histories_are_untouched(self):
        record_turn_from_responses(self.ledger)
        for session_id, model, items in (
            ("s1", "gpt-5.6-luna", RESUMED_HISTORY),
            ("s2", MODEL, RESUMED_HISTORY),
            ("s1", MODEL, LIVE_HISTORY),
            (None, MODEL, RESUMED_HISTORY),
        ):
            with self.subTest(session_id=session_id, model=model):
                restored, count = self.ledger.restore(session_id, model, items)
                self.assertIs(restored, items)
                self.assertEqual(count, 0)

    def test_unrelated_neighbours_do_not_receive_reasoning(self):
        record_turn_from_responses(self.ledger)
        # c2 now follows a different item (a steering message): no match.
        items = [user("first"), call("c1"), output("c1"), user("steer"), call("c2")]
        self.assertEqual(self.ledger.restore("s1", MODEL, items), (items, 0))

    def test_reasoning_without_encrypted_content_is_not_recorded(self):
        bare = {"type": "reasoning", "id": "rs-bare", "summary": []}
        self.ledger.record_output("s1", MODEL, anchor_key(output("c1")), [bare, call("c2")])
        self.assertEqual(self.ledger.restore("s1", MODEL, RESUMED_HISTORY)[1], 0)

    def test_disable_and_forget(self):
        record_turn_from_responses(self.ledger)
        self.ledger.disable("s1", MODEL)
        self.assertEqual(self.ledger.restore("s1", MODEL, RESUMED_HISTORY)[1], 0)
        self.ledger.forget({"s1"})
        self.assertFalse(os.path.exists(os.path.join(self.directory.name, "s1.jsonl")))
        self.assertEqual(self.new_ledger().restore("s1", MODEL, RESUMED_HISTORY)[1], 0)

    def test_memory_budget_reloads_trimmed_sessions_from_disk(self):
        ledger = self.new_ledger(memory_budget_bytes=1)
        record_turn_from_responses(ledger, "s1")
        record_turn_from_responses(ledger, "s2")
        self.assertEqual(ledger.restore("s1", MODEL, RESUMED_HISTORY)[1], 2)

    def test_prune_removes_stale_files(self):
        record_turn_from_responses(self.ledger)
        path = os.path.join(self.directory.name, "s1.jsonl")
        os.utime(path, (1, 1))
        self.ledger.prune_older_than(2)
        self.assertFalse(os.path.exists(path))

    def test_unsafe_session_ids_are_kept_in_memory_only(self):
        record_turn_from_responses(self.ledger, "../escape")
        self.assertEqual(os.listdir(self.directory.name), [])
        self.assertEqual(self.ledger.restore("../escape", MODEL, RESUMED_HISTORY)[1], 2)


class _Upstream:
    """An open upstream WebSocket that never sends on its own."""

    def __init__(self):
        self.sent = []
        self.state = SimpleNamespace(name="OPEN")
        self._closed = asyncio.Event()

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        self.state = SimpleNamespace(name="CLOSED")
        self._closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration


class _Bridge:
    def __init__(self):
        self.written = []

    async def write(self, data):
        self.written.append(json.loads(data))

    async def end(self):
        pass

    async def error(self, *args):
        pass


class CopilotSdkCacheContinuityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.ledger = ReasoningLedger(lambda: directory.name)
        self.entry = sdk._LiveSession(session=SimpleNamespace(), diagnostics={})
        for name, value in (
            ("_reasoning_ledger", self.ledger),
            ("_INJECT_PROMPT_CACHE_KEY", True),
            ("_live_sessions", {"s1": self.entry}),
            ("_websocket_pool", sdk.OrderedDict()),
        ):
            patcher = patch.object(sdk, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.sent = []
        self.reject = False

        def respond(request):
            self.sent.append(json.loads(request.content))
            if self.reject and len(self.sent) % 2:
                return httpx.Response(400, json={"error": {"message": "bad request"}})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: {}\n\n")

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(client.aclose)
        patcher = patch.object(copilot_request_handler, "_get_shared_http_client", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.handler = sdk._UpstreamRequestHandler()

    async def post(self, body, session_id="s1"):
        request = httpx.Request("POST", "https://api.githubcopilot.com/responses", json=body)
        response = await self.handler.send_request(request, SimpleNamespace(session_id=session_id))
        await response.aread()
        return response

    def websocket(self, session_id="s1", upstream=None):
        socket = sdk._UpstreamWebSocket(
            SimpleNamespace(session_id=session_id, _bridge=_Bridge()), self.handler,
        )
        socket._upstream = upstream or _Upstream()
        return socket

    async def pooled_websocket(self, session_id="s1"):
        socket = sdk._UpstreamWebSocket(
            SimpleNamespace(session_id=session_id, _bridge=_Bridge()), self.handler,
        )
        await socket.open()
        self.addAsyncCleanup(socket.aclose)
        return socket

    async def exchange(self, socket, body, response_output, usage=None, response_id=None):
        await socket.send_request_message(json.dumps({"type": "response.create", **body}))
        sent = socket._upstream.sent[-1]
        await socket.send_response_message(json.dumps({"type": "response.created", "response": {"error": None}}))
        await socket.send_response_message(json.dumps({
            "type": "response.completed",
            "response": {"id": response_id, "output": response_output, "usage": usage or {}},
        }))
        return sent

    async def live_turn(self):
        """The recorded turn on one live connection, left idle."""
        socket = self.websocket()
        await self.exchange(socket, {"model": MODEL, "input": [user("first")]}, [call("c1")], response_id="r1")
        await self.exchange(socket, {"model": MODEL, "previous_response_id": "r1", "input": [output("c1")]},
                            [reasoning("a"), call("c2")], response_id="r2")
        await self.exchange(socket, {"model": MODEL, "previous_response_id": "r2", "input": [output("c2")]},
                            [reasoning("b"), answer("done")], response_id="r3")
        return socket

    async def test_disk_resume_continues_the_kept_upstream_chain(self):
        live = await self.live_turn()
        upstream = live._upstream
        await live.close()
        self.assertFalse(upstream._closed.is_set())

        resumed = await self.pooled_websocket()
        self.assertIs(resumed._upstream, upstream)
        sent = await self.exchange(resumed, {"model": MODEL, "input": RESUMED_HISTORY}, [answer("again")],
                                   response_id="r4")
        self.assertEqual(sent["previous_response_id"], "r3")
        self.assertEqual(sent["input"], [user("second")])
        self.assertEqual(self.entry.diagnostics["model_calls"][-1]["resumed_chain_items"], 1)
        # The chain keeps extending: a later resend continues from r4.
        self.assertEqual(resumed._chain.last_response_id, "r4")

    async def test_rejected_continuation_falls_back_to_the_full_request(self):
        live = await self.live_turn()
        await live.close()
        resumed = await self.pooled_websocket()
        await resumed.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": RESUMED_HISTORY}))
        await resumed.send_response_message(json.dumps({"type": "error", "error": {
            "type": "invalid_request_error", "code": "previous_response_not_found"}}))
        full = resumed._upstream.sent[-1]
        self.assertNotIn("previous_response_id", full)
        self.assertEqual(full["input"], LIVE_HISTORY)
        self.assertEqual(resumed.context._bridge.written, [])
        self.assertTrue(self.entry.diagnostics["model_calls"][-1]["resumed_chain_rejected"])

    async def test_connections_with_a_response_in_flight_are_not_kept(self):
        live = await self.live_turn()
        await live.send_request_message(json.dumps({"type": "response.create", "model": MODEL,
                                                    "previous_response_id": "r3", "input": [user("second")]}))
        upstream = live._upstream
        await live.close()
        self.assertTrue(upstream._closed.is_set())
        self.assertEqual(len(sdk._websocket_pool), 0)

    async def test_a_diverged_history_is_sent_in_full(self):
        live = await self.live_turn()
        await live.close()
        resumed = await self.pooled_websocket()
        edited = [user("first"), call("c1"), output("c1"), user("steer")]
        await resumed.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": edited}))
        self.assertNotIn("previous_response_id", resumed._upstream.sent[-1])

    async def test_non_request_errors_keep_the_changes(self):
        socket = self.websocket()
        await socket.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": [user("x")]}))
        await socket.send_response_message(json.dumps({"type": "error", "error": {"type": "rate_limit_error"}}))
        await socket.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": [user("x")]}))
        self.assertEqual(socket._upstream.sent[-1]["prompt_cache_key"], "s1")

    async def test_http_resume_restores_reasoning_and_keys_the_session(self):
        await self.post({"model": MODEL, "input": LIVE_HISTORY})
        await self.post({"model": MODEL, "input": RESUMED_HISTORY})
        self.assertEqual(self.sent[1]["input"], LIVE_HISTORY)
        self.assertEqual(self.sent[1]["prompt_cache_key"], "s1")
        self.assertEqual(self.entry.diagnostics["model_calls"][1]["restored_reasoning"], 2)

    async def test_http_rejection_retries_as_sent_and_stops_the_changes(self):
        self.ledger.record_input("s1", MODEL, LIVE_HISTORY)
        self.reject = True
        response = await self.post({"model": MODEL, "input": RESUMED_HISTORY})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.sent[1], {"model": MODEL, "input": RESUMED_HISTORY})
        self.reject = False
        self.sent.clear()
        await self.post({"model": MODEL, "input": RESUMED_HISTORY})
        self.assertEqual(self.sent, [{"model": MODEL, "input": RESUMED_HISTORY}])

    async def test_requests_without_a_session_are_forwarded_unchanged(self):
        body = {"model": MODEL, "input": RESUMED_HISTORY}
        request = httpx.Request("POST", "https://api.githubcopilot.com/responses", json=body)
        original = request.read()
        await self.handler.send_request(request, SimpleNamespace(session_id=None))
        self.assertEqual(self.sent, [json.loads(original)])

    async def test_websocket_records_output_and_restores_the_next_connection(self):
        live = self.websocket()
        await self.exchange(live, {"model": MODEL, "input": [user("first")]}, [call("c1")])
        await self.exchange(live, {"model": MODEL, "previous_response_id": "r1", "input": [output("c1")]},
                            [reasoning("a"), call("c2")])
        continued = await self.exchange(
            live, {"model": MODEL, "previous_response_id": "r2", "input": [output("c2")]},
            [reasoning("b"), answer("done")], usage={"input_tokens": 900, "input_tokens_details": {"cached_tokens": 512}},
        )
        self.assertEqual(continued["input"], [output("c2")])
        self.assertEqual(continued["prompt_cache_key"], "s1")
        self.assertEqual(self.entry.diagnostics["model_calls"][2]["cached_tokens"], 512)

        # A disk resume opens a new connection and resends everything.
        resumed = self.websocket()
        sent = await self.exchange(resumed, {"model": MODEL, "input": RESUMED_HISTORY}, [answer("again")])
        self.assertEqual(sent["input"], LIVE_HISTORY)
        self.assertEqual(sent["type"], "response.create")
        self.assertEqual(self.entry.diagnostics["model_calls"][3]["restored_reasoning"], 2)

    async def test_compaction_turn_calls_keep_tools_with_tool_choice_none(self):
        tools = [{"type": "function", "name": "inspect", "parameters": {"type": "object", "properties": {}}}]
        self.entry.tool_choice = "none"
        socket = self.websocket()
        sent = await self.exchange(socket, {"model": MODEL, "tools": tools, "input": [user("summarize")]},
                                   [answer("summary")], response_id="r1")
        self.assertEqual(sent["tools"], tools)
        self.assertEqual(sent["tool_choice"], "none")
        self.assertEqual(self.entry.diagnostics["model_calls"][-1]["tool_choice"], "none")
        # Other turns keep the runtime's tool_choice.
        self.entry.tool_choice = None
        sent = await self.exchange(socket, {"model": MODEL, "previous_response_id": "r1", "tools": tools,
                                            "input": [user("next")]}, [answer("ok")])
        self.assertNotIn("tool_choice", sent)

    async def test_rejected_tool_choice_is_recorded_and_not_sent_again(self):
        tools = [{"type": "function", "name": "inspect"}]
        self.entry.tool_choice = "none"
        socket = self.websocket()
        create = json.dumps({"type": "response.create", "model": MODEL, "tools": tools, "input": [user("x")]})
        await socket.send_request_message(create)
        await socket.send_response_message(json.dumps({"type": "error", "error": {
            "type": "invalid_request_error", "message": "tool_choice is not supported"}}))
        await socket.send_request_message(create)
        self.assertNotIn("tool_choice", socket._upstream.sent[-1])
        self.assertEqual(self.entry.diagnostics["model_calls"][0]["error"],
                         {"type": "invalid_request_error", "message": "tool_choice is not supported"})

    async def test_websocket_calls_record_copilots_request_ids(self):
        socket = self.websocket()
        await socket.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": [user("x")]}))
        await socket.send_response_message(json.dumps({
            "type": "response.completed",
            "headers": {"X-Copilot-Service-Request-Id": "svc-1", "X-Copilot-WebSocket-Session-Id": "0123456789abcdef"},
            "response": {"id": "r1", "output": [answer("y")], "usage": {}},
        }))
        record = self.entry.diagnostics["model_calls"][-1]
        self.assertEqual(record["service_request_id"], "svc-1")
        self.assertEqual(record["copilot_websocket_session"], "01234567")

    async def test_calls_between_requests_are_kept_for_the_next_trace(self):
        self.entry.diagnostics = None
        await self.post({"model": MODEL, "input": [user("background")]})
        self.assertEqual(len(self.entry.background_calls), 1)
        self.assertEqual(self.entry.background_calls[0]["transport"], "http")

    async def test_websocket_error_after_a_change_disables_it(self):
        record_turn_from_responses(self.ledger)
        socket = self.websocket()
        await socket.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": RESUMED_HISTORY}))
        await socket.send_response_message(json.dumps({"type": "error", "error": {
            "type": "invalid_request_error", "message": "rejected"}}))
        await socket.send_request_message(json.dumps({"type": "response.create", "model": MODEL, "input": RESUMED_HISTORY}))
        self.assertEqual(socket._upstream.sent[-1], {"type": "response.create", "model": MODEL, "input": RESUMED_HISTORY})


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class CopilotSdkHttpCallTraceTests(unittest.IsolatedAsyncioTestCase):
    """HTTP model calls record their usage and errors without altering them."""

    def setUp(self):
        self.entry = sdk._LiveSession(session=SimpleNamespace(), diagnostics={})
        for name, value in (("_INJECT_PROMPT_CACHE_KEY", True), ("_live_sessions", {"s1": self.entry})):
            patcher = patch.object(sdk, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.handler = sdk._UpstreamRequestHandler()

    async def send(self, response):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
        self.addAsyncCleanup(client.aclose)
        with patch.object(copilot_request_handler, "_get_shared_http_client", return_value=client):
            request = httpx.Request("POST", "https://api.githubcopilot.com/responses",
                                    json={"model": MODEL, "input": [user("x")]})
            sent = await self.handler.send_request(request, SimpleNamespace(session_id="s1", interaction_type="conversation-agent"))
            return sent, await sent.aread()

    async def test_usage_is_read_from_the_stream_as_it_passes(self):
        completed = json.dumps({"type": "response.completed", "response": {"usage": {
            "input_tokens": 900, "input_tokens_details": {"cached_tokens": 512}}}}).encode()
        chunks = [b"event: response.created\ndata: {}\n\nevent: response.completed\ndata: ", completed[:20],
                  completed[20:] + b"\n\n"]
        _, body = await self.send(httpx.Response(200, headers={"content-type": "text/event-stream"},
                                                 stream=_ChunkedStream(chunks)))
        self.assertEqual(body, b"".join(chunks))
        record = self.entry.diagnostics["model_calls"][0]
        self.assertEqual((record["input_tokens"], record["cached_tokens"]), (900, 512))
        self.assertEqual(record["interaction"], "conversation-agent")

    async def test_error_status_and_body_are_recorded_and_forwarded(self):
        response, body = await self.send(httpx.Response(503, json={"error": {"message": "overloaded"}}))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(body), {"error": {"message": "overloaded"}})
        record = self.entry.diagnostics["model_calls"][0]
        self.assertEqual(record["status"], 503)
        self.assertIn("overloaded", record["error"])


class ProxyCompactionRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_compaction_turn_keeps_the_callers_tools(self):
        import format_translation
        import proxy
        from starlette.requests import Request

        tools = [{"type": "function", "name": "inspect", "parameters": {"type": "object", "properties": {}}}]
        body = {"model": MODEL, "tools": tools, "tool_choice": "auto",
                "input": [user("first"), {"type": "compaction_trigger"}]}
        captured = {}

        async def handle_responses(request, sdk_body, **kwargs):
            captured.update(body=sdk_body, **kwargs)

        request = Request({"type": "http", "method": "POST", "path": "/v1/responses", "headers": []})
        with patch.object(proxy, "_prepare_upstream_request", return_value=(None, None)), \
                patch.object(sdk, "handle_responses", handle_responses):
            await proxy._handle_copilot_sdk_responses(
                request, format_translation.build_fake_compaction_request(body), source_body=body, is_compact=True,
            )
        self.assertTrue(captured["is_compact"])
        self.assertEqual(captured["body"]["tool_choice"], "auto")
        self.assertEqual([tool.name for tool in sdk.build_tool_registration(captured["body"]).tools], ["inspect"])


class _AbortingSession:
    def __init__(self, *, settles):
        self.session_id = "s1"
        self.handlers = []
        self.settles = settles
        self.disconnected = False

    def on(self, handler):
        self.handlers.append(handler)
        return lambda: self.handlers.remove(handler)

    async def abort(self):
        if self.settles:
            event = SimpleNamespace(type=SimpleNamespace(value="session.idle"), data=SessionIdleData())
            for handler in list(self.handlers):
                handler(event)

    async def disconnect(self):
        self.disconnected = True


class CopilotSdkInterruptedTurnTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for name, value in (("_live_sessions", {}), ("_ABORT_SETTLE_SECONDS", 0.2)):
            patcher = patch.object(sdk, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addAsyncCleanup(sdk._evict_all_live_sessions)

    async def interrupt(self, session):
        await sdk._track_live_session(session)

        async def dispatch():
            return None

        async def connected():
            return False

        async def consume():
            async for _chunk in sdk._stream_turn(
                SimpleNamespace(is_disconnected=connected), {"model": MODEL}, session, dispatch,
                sdk.ToolRegistration(),
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_settled_interrupt_keeps_the_live_session(self):
        session = _AbortingSession(settles=True)
        await self.interrupt(session)
        await asyncio.gather(*sdk._interrupted_turns)
        self.assertFalse(session.disconnected)
        self.assertFalse(sdk._live_sessions["s1"].in_use)
        self.assertFalse(sdk._live_sessions["s1"].pending_calls)

    async def test_unsettled_interrupt_still_disconnects(self):
        session = _AbortingSession(settles=False)
        await self.interrupt(session)
        await asyncio.gather(*sdk._interrupted_turns)
        self.assertTrue(session.disconnected)
        self.assertNotIn("s1", sdk._live_sessions)

    async def test_interrupt_through_starlettes_cancel_scope_keeps_the_live_session(self):
        # Starlette cancels a streaming response by cancelling an anyio task
        # group, which cancels every later await inside the generator too.
        session = _AbortingSession(settles=True)
        await sdk._track_live_session(session)

        async def dispatch():
            return None

        async def connected():
            return False

        async def consume():
            async for _chunk in sdk._stream_turn(
                SimpleNamespace(is_disconnected=connected), {"model": MODEL}, session, dispatch,
                sdk.ToolRegistration(),
            ):
                pass

        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await asyncio.sleep(0.05)
            group.cancel_scope.cancel()
        await asyncio.gather(*sdk._interrupted_turns)
        self.assertFalse(session.disconnected)
        self.assertFalse(sdk._live_sessions["s1"].in_use)

    async def test_next_request_waits_for_an_interrupted_turn_to_settle(self):
        session = _AbortingSession(settles=True)
        entry = await sdk._track_live_session(session)
        settled = asyncio.Event()

        async def settle():
            await settled.wait()
            await sdk._release_session(session, None, completed=False, park=True)

        entry.settling = asyncio.create_task(settle())
        reuse = asyncio.create_task(sdk._reuse_live_session("s1", allow_pending=False, options=None))
        await asyncio.sleep(0.05)
        self.assertFalse(reuse.done())
        settled.set()
        self.assertIs(await reuse, session)


if __name__ == "__main__":
    unittest.main()
