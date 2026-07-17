import asyncio
import json

import httpx
from starlette.requests import Request

import proxy
from codex_agent_compat import codex_session_id, codex_thread_id, codex_turn_id
from initiator_policy import InitiatorPolicy, utc_now
from proxy import (
    _ACTIVE_RESPONSES_STREAMS,
    _ACTIVE_RESPONSES_STREAMS_LOCK,
    UpstreamRequestPlan,
    _DownstreamDisconnectedBeforeResponse,
    _ManagedResponsesStreamBody,
    _complete_active_responses_teardown,
    _open_streaming_upstream,
    _register_active_responses_stream,
    _responses_cache_settle_identity,
)
from request_headers import build_responses_headers_for_request


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/v1/responses", "headers": []})


def _body(
    *,
    session: str,
    thread: str,
    turn: str,
    source: str = "user",
    parent: str | None = None,
    agent_tail: bool = False,
) -> dict:
    turn_metadata = {
        "session_id": session,
        "thread_id": thread,
        "turn_id": turn,
        "thread_source": source,
    }
    if parent:
        turn_metadata["parent_thread_id"] = parent
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    if agent_tail:
        input_items.extend(
            [
                {"type": "function_call", "call_id": "call", "name": "tool", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call", "output": "ok"},
            ]
        )
    return {
        "model": "gpt-5.5",
        "prompt_cache_key": thread,
        "input": input_items,
        "client_metadata": {
            "session_id": session,
            "thread_id": thread,
            "turn_id": turn,
            "x-codex-turn-metadata": json.dumps(turn_metadata),
        },
    }


def _headers(body: dict, policy: InitiatorPolicy, request_id: str) -> dict:
    return build_responses_headers_for_request(
        _request(),
        body,
        "token",
        initiator_policy=policy,
        request_id=request_id,
    )


def test_turns_rotate_identity_and_children_get_distinct_interactions():
    policy = InitiatorPolicy(request_finish_guard_seconds=600)
    root_one = _headers(
        _body(session="session", thread="root", turn="turn-one"),
        policy,
        "root-one",
    )
    policy.note_request_finished("root-one", finished_at=utc_now())
    root_two = _headers(
        _body(session="session", thread="root", turn="turn-two"),
        policy,
        "root-two",
    )
    child_one = _headers(
        _body(
            session="session",
            thread="child-one",
            turn="child-turn-one",
            source="subagent",
            parent="root",
            agent_tail=True,
        ),
        policy,
        "child-one",
    )
    child_two = _headers(
        _body(
            session="session",
            thread="child-two",
            turn="child-turn-two",
            source="subagent",
            parent="root",
            agent_tail=True,
        ),
        policy,
        "child-two",
    )

    assert root_one["x-client-session-id"] == root_two["x-client-session-id"]
    assert root_one["x-agent-task-id"] != root_two["x-agent-task-id"]
    assert root_one["x-interaction-id"] != root_two["x-interaction-id"]
    assert root_two["x-initiator"] == "user"
    assert root_two["x-interaction-type"] == "conversation-user"
    assert child_one["x-client-session-id"] == root_two["x-client-session-id"]
    assert child_two["x-client-session-id"] == root_two["x-client-session-id"]
    assert child_one["x-parent-agent-id"] == root_two["x-agent-task-id"]
    assert child_two["x-parent-agent-id"] == root_two["x-agent-task-id"]
    assert child_one["x-interaction-id"] != child_two["x-interaction-id"]
    assert child_one["x-agent-task-id"] != child_two["x-agent-task-id"]


def test_nested_turn_metadata_wins_over_stale_outer_cache_fields():
    body = _body(session="current-session", thread="current-thread", turn="current-turn")
    body["client_metadata"].update(
        {
            "session_id": "stale-session",
            "thread_id": "stale-thread",
            "turn_id": "stale-turn",
        }
    )

    assert codex_session_id(body) == "current-session"
    assert codex_thread_id(body) == "current-thread"
    assert codex_turn_id(body) == "current-turn"


def test_same_turn_steering_reuses_identity():
    policy = InitiatorPolicy(request_finish_guard_seconds=600)
    first = _headers(
        _body(session="steer-session", thread="steer-root", turn="same-turn"),
        policy,
        "steer-one",
    )
    policy.note_request_finished("steer-one", finished_at=utc_now())
    steered = _headers(
        _body(session="steer-session", thread="steer-root", turn="same-turn"),
        policy,
        "steer-two",
    )

    assert steered["x-agent-task-id"] == first["x-agent-task-id"]
    assert steered["x-interaction-id"] == first["x-interaction-id"]
    assert steered["x-initiator"] == "user"
    assert steered["x-interaction-type"] == "conversation-user"

    continuation = _headers(
        _body(
            session="steer-session",
            thread="steer-root",
            turn="same-turn",
            agent_tail=True,
        ),
        policy,
        "steer-continuation",
    )
    assert continuation["x-agent-task-id"] == first["x-agent-task-id"]
    assert continuation["x-interaction-id"] == first["x-interaction-id"]
    assert continuation["x-initiator"] == "agent"
    assert continuation["x-interaction-type"] == "conversation-agent"


def test_siblings_use_distinct_cache_settle_families():
    policy = InitiatorPolicy()
    root = _headers(
        _body(session="cache-session", thread="cache-root", turn="cache-turn"),
        policy,
        "cache-root",
    )
    children = []
    for name in ("one", "two"):
        body = _body(
            session="cache-session",
            thread=f"cache-child-{name}",
            turn=f"cache-child-turn-{name}",
            source="subagent",
            parent="cache-root",
            agent_tail=True,
        )
        headers = _headers(body, policy, f"cache-child-{name}")
        children.append(
            UpstreamRequestPlan(
                request_id=f"cache-child-{name}",
                upstream_url="https://example.invalid/responses",
                headers=headers,
                body=body,
                usage_event=None,
                requested_model="gpt-5.5",
                resolved_model="gpt-5.5",
            )
        )

    first_identity = _responses_cache_settle_identity(children[0])
    second_identity = _responses_cache_settle_identity(children[1])
    assert children[0].headers["x-parent-agent-id"] == root["x-agent-task-id"]
    assert children[1].headers["x-parent-agent-id"] == root["x-agent-task-id"]
    assert first_identity is not None
    assert second_identity is not None
    assert first_identity[2] != second_identity[2]


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.wait = asyncio.Event()

    async def __aiter__(self):
        await self.wait.wait()
        yield b"never"

    async def aclose(self):
        self.closed = True


def test_disconnect_closes_an_unstarted_upstream_stream():
    async def scenario():
        stream = _BlockingStream()
        response = httpx.Response(
            200,
            stream=stream,
            extensions={"http_version": b"HTTP/1.1"},
        )
        owner = _ManagedResponsesStreamBody(
            upstream=response,
            body={},
            headers={},
            usage_event=None,
            stream_type="responses",
            trace_plan=None,
            active_stream=None,
        )
        pending = asyncio.create_task(owner.__anext__())
        await asyncio.sleep(0)
        await owner.aclose()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        assert stream.closed
        assert owner._finalized

    asyncio.run(scenario())


def test_unconfirmed_finished_stream_does_not_poison_its_lineage():
    async def scenario():
        body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "durable-thread",
            "input": [],
        }
        plan = UpstreamRequestPlan(
            request_id="transport-error",
            upstream_url="https://example.invalid/responses",
            headers={"x-agent-task-id": "durable-task"},
            body=body,
            source_body=body,
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
            trace_context={"upstream_path": "/responses"},
        )
        entry = _register_active_responses_stream(plan)
        assert entry is not None
        assert entry.identity in _ACTIVE_RESPONSES_STREAMS

        _complete_active_responses_teardown(
            entry,
            transport_cancel="pre_response_request_error_unconfirmed",
            confirmed=False,
        )

        assert entry.teardown_complete.is_set()
        assert not entry.teardown_confirmed
        assert entry.identity not in _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()


def test_disconnect_before_headers_waits_for_a_wire_cancel(monkeypatch):
    async def scenario():
        stream = _BlockingStream()
        response = httpx.Response(
            200,
            stream=stream,
            extensions={"http_version": b"HTTP/1.1"},
        )
        response_ready = asyncio.Event()

        async def fake_send(_client, _request, *, stream):
            assert stream is True
            await response_ready.wait()
            return response

        async def receive():
            return {"type": "http.disconnect"}

        monkeypatch.setattr(proxy, "throttled_client_send", fake_send)
        downstream = Request(
            {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []},
            receive=receive,
        )
        opening = asyncio.create_task(
            _open_streaming_upstream(
                object(),
                httpx.Request("POST", "https://example.invalid/responses"),
                trace_plan=None,
                downstream_request=downstream,
            )
        )
        await asyncio.sleep(0)
        assert not opening.done()
        response_ready.set()
        try:
            await opening
        except _DownstreamDisconnectedBeforeResponse as exc:
            assert exc.transport_close == "http1_connection_close"
        else:
            raise AssertionError("disconnect should stop the pre-response stream")
        assert stream.closed

    asyncio.run(scenario())
