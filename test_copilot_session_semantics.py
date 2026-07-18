import asyncio
import json

import httpx
from starlette.requests import Request

import proxy
import request_headers
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


def test_user_steering_with_agent_history_keeps_the_user_interaction_type():
    body = _body(
        session="history-session",
        thread="history-root",
        turn="history-turn",
    )
    body["input"] = [
        body["input"][0],
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "earlier answer"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "steer it differently"}],
        },
    ]

    headers = _headers(body, InitiatorPolicy(), "history-steering")

    assert headers["x-initiator"] == "user"
    assert headers["x-interaction-type"] == "conversation-user"


def test_existing_child_keeps_its_spawn_parent_after_the_root_advances():
    policy = InitiatorPolicy()
    root_one = _headers(
        _body(session="pin-session", thread="pin-root", turn="pin-root-one"),
        policy,
        "pin-root-one",
    )
    child_one = _headers(
        _body(
            session="pin-session",
            thread="pin-child",
            turn="pin-child-one",
            source="subagent",
            parent="pin-root",
            agent_tail=True,
        ),
        policy,
        "pin-child-one",
    )
    root_two = _headers(
        _body(session="pin-session", thread="pin-root", turn="pin-root-two"),
        policy,
        "pin-root-two",
    )
    child_continuation = _headers(
        _body(
            session="pin-session",
            thread="pin-child",
            turn="pin-child-one",
            source="subagent",
            parent="pin-root",
            agent_tail=True,
        ),
        policy,
        "pin-child-continuation",
    )
    child_two = _headers(
        _body(
            session="pin-session",
            thread="pin-child",
            turn="pin-child-two",
            source="subagent",
            parent="pin-root",
            agent_tail=True,
        ),
        policy,
        "pin-child-two",
    )
    later_sibling = _headers(
        _body(
            session="pin-session",
            thread="pin-sibling",
            turn="pin-sibling-one",
            source="subagent",
            parent="pin-root",
            agent_tail=True,
        ),
        policy,
        "pin-sibling-one",
    )

    assert root_one["x-agent-task-id"] != root_two["x-agent-task-id"]
    assert child_one["x-parent-agent-id"] == root_one["x-agent-task-id"]
    assert child_continuation["x-parent-agent-id"] == child_one["x-parent-agent-id"]
    assert child_continuation["x-agent-task-id"] == child_one["x-agent-task-id"]
    assert child_continuation["x-interaction-id"] == child_one["x-interaction-id"]
    assert child_two["x-parent-agent-id"] == root_one["x-agent-task-id"]
    assert child_two["x-agent-task-id"] != child_one["x-agent-task-id"]
    assert child_two["x-interaction-id"] != child_one["x-interaction-id"]
    assert later_sibling["x-parent-agent-id"] == root_two["x-agent-task-id"]


def test_child_parent_turn_lookup_is_isolated_by_codex_session():
    with request_headers._CODEX_ROOT_TURN_SCOPE_LOCK:
        request_headers._CODEX_ROOT_TURN_SCOPE_BY_THREAD.clear()
        request_headers._CODEX_CHILD_PARENT_TURN_SCOPE_BY_LINEAGE.clear()
    try:
        policy = InitiatorPolicy()
        root_a = _headers(
            _body(
                session="lookup-session-a",
                thread="shared-root-thread",
                turn="lookup-turn-a",
            ),
            policy,
            "lookup-root-a",
        )
        _headers(
            _body(
                session="lookup-session-b",
                thread="shared-root-thread",
                turn="lookup-turn-b",
            ),
            policy,
            "lookup-root-b",
        )
        child_a = _headers(
            _body(
                session="lookup-session-a",
                thread="lookup-child-a",
                turn="lookup-child-turn-a",
                source="subagent",
                parent="shared-root-thread",
                agent_tail=True,
            ),
            policy,
            "lookup-child-a",
        )

        assert child_a["x-parent-agent-id"] == root_a["x-agent-task-id"]
    finally:
        with request_headers._CODEX_ROOT_TURN_SCOPE_LOCK:
            request_headers._CODEX_ROOT_TURN_SCOPE_BY_THREAD.clear()
            request_headers._CODEX_CHILD_PARENT_TURN_SCOPE_BY_LINEAGE.clear()


def test_child_parent_pin_survives_more_than_the_old_lru_limit():
    with request_headers._CODEX_ROOT_TURN_SCOPE_LOCK:
        request_headers._CODEX_ROOT_TURN_SCOPE_BY_THREAD.clear()
        request_headers._CODEX_CHILD_PARENT_TURN_SCOPE_BY_LINEAGE.clear()
    try:
        request_headers._remember_codex_root_turn_scope(
            "long-root",
            "long-root-turn-one",
            "long-session",
        )
        for index in range(2050):
            request_headers._remember_codex_root_turn_scope(
                f"filler-root-{index}",
                f"filler-root-turn-{index}",
                f"filler-session-{index}",
            )
        original_key = ("long-session", "long-root", "thread:long-child")
        original_parent = request_headers._codex_child_parent_turn_scope(
            original_key,
            "long-root",
            "long-session",
        )
        for index in range(4100):
            request_headers._codex_child_parent_turn_scope(
                ("long-session", "long-root", f"thread:filler-{index}"),
                "long-root",
                "long-session",
            )
        request_headers._remember_codex_root_turn_scope(
            "long-root",
            "long-root-turn-two",
            "long-session",
        )

        assert request_headers._codex_child_parent_turn_scope(
            original_key,
            "long-root",
            "long-session",
        ) == original_parent
        assert original_parent == "long-root-turn-one"
    finally:
        with request_headers._CODEX_ROOT_TURN_SCOPE_LOCK:
            request_headers._CODEX_ROOT_TURN_SCOPE_BY_THREAD.clear()
            request_headers._CODEX_CHILD_PARENT_TURN_SCOPE_BY_LINEAGE.clear()


def test_subagent_identity_encoding_has_no_delimiter_collisions():
    first_task = request_headers._responses_subagent_task_id("a:b", "c", "d")
    second_task = request_headers._responses_subagent_task_id("a", "b:c", "d")
    first_scope = request_headers._responses_subagent_affinity_scope("a:b", "c")
    second_scope = request_headers._responses_subagent_affinity_scope("a", "b:c")

    assert first_task is not None
    assert second_task is not None
    assert first_task != second_task
    assert first_scope != second_scope


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


def test_same_child_turns_rotate_lineage_but_share_the_settle_family(monkeypatch):
    async def scenario():
        policy = InitiatorPolicy()
        _headers(
            _body(
                session="child-family-session",
                thread="child-family-root",
                turn="child-family-root-turn",
            ),
            policy,
            "child-family-root",
        )
        plans = []
        for turn in ("one", "two"):
            source_body = _body(
                session="child-family-session",
                thread="child-family-worker",
                turn=f"child-family-turn-{turn}",
                source="subagent",
                parent="child-family-root",
                agent_tail=True,
            )
            plans.append(
                UpstreamRequestPlan(
                    request_id=f"child-family-{turn}",
                    upstream_url="https://example.invalid/responses",
                    headers=_headers(source_body, policy, f"child-family-{turn}"),
                    body=source_body,
                    source_body=source_body,
                    usage_event=None,
                    requested_model="gpt-5.5",
                    resolved_model="gpt-5.5",
                    trace_context={
                        "upstream_path": "/responses",
                        "initiator_verdict": {"candidate_initiator": "agent"},
                    },
                )
            )

        first_identity = _responses_cache_settle_identity(plans[0])
        second_identity = _responses_cache_settle_identity(plans[1])
        assert first_identity is not None
        assert second_identity is not None
        assert first_identity[1] != second_identity[1]
        assert first_identity[2] == second_identity[2]

        proxy._remember_responses_cache_settle_finish(plans[0], 200)
        await proxy._wait_for_responses_cache_settle(plans[1])
        settle = plans[1].trace_context.get("prompt_cache_settle")
        assert settle is not None
        assert settle["same_lineage"] is False

    monkeypatch.setattr(proxy, "_prompt_cache_settle_delay_seconds", lambda _plan: 0.01)
    with proxy._PROMPT_CACHE_SETTLE_LOCK:
        proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()
    try:
        asyncio.run(scenario())
    finally:
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()


def _cache_settle_plan(*, request_id: str, model: str, steering: bool) -> UpstreamRequestPlan:
    body = {
        "model": model,
        "prompt_cache_key": "durable-cache-thread",
        "input": [],
    }
    return UpstreamRequestPlan(
        request_id=request_id,
        upstream_url="https://example.invalid/responses",
        headers={
            "x-agent-task-id": "durable-cache-task",
            "x-interaction-id": "durable-cache-interaction",
        },
        body=body,
        source_body=body,
        usage_event=None,
        requested_model=model,
        resolved_model=model,
        trace_context={
            "upstream_path": "/responses",
            "initiator_verdict": {
                "candidate_initiator": "user" if steering else "agent",
            },
        },
    )


def test_all_models_use_one_cache_settle_default(monkeypatch):
    monkeypatch.delenv("GHCP_PROXY_RESPONSES_CACHE_SETTLE_DELAY_SECONDS", raising=False)
    model = "future-responses-model"
    continuation = _cache_settle_plan(
        request_id="cache-continuation",
        model=model,
        steering=False,
    )
    steering = _cache_settle_plan(
        request_id="cache-steering",
        model=model,
        steering=True,
    )

    assert proxy._prompt_cache_settle_delay_seconds(continuation) == 3.0
    assert proxy._prompt_cache_settle_delay_seconds(steering) == 3.0


def test_same_lineage_steering_does_not_wait_for_cache_settle(monkeypatch):
    async def fail_sleep(_delay):
        raise AssertionError("native same-turn steering must not be delayed")

    async def scenario():
        model = "future-responses-model"
        previous = _cache_settle_plan(
            request_id="cache-previous",
            model=model,
            steering=False,
        )
        steering = _cache_settle_plan(
            request_id="cache-steering",
            model=model,
            steering=True,
        )
        proxy._remember_responses_cache_settle_finish(previous, 200)

        await proxy._wait_for_responses_cache_settle(steering)

        assert "prompt_cache_settle" not in steering.trace_context

    monkeypatch.setattr(proxy, "_prompt_cache_settle_delay_seconds", lambda _plan: 5.0)
    monkeypatch.setattr(proxy.asyncio, "sleep", fail_sleep)
    with proxy._PROMPT_CACHE_SETTLE_LOCK:
        proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()
    try:
        asyncio.run(scenario())
    finally:
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()


def test_same_lineage_agent_continuation_waits_for_cache_settle(monkeypatch):
    async def scenario():
        previous = _cache_settle_plan(
            request_id="cache-previous",
            model="future-responses-model",
            steering=False,
        )
        continuation = _cache_settle_plan(
            request_id="cache-continuation",
            model="future-responses-model",
            steering=False,
        )
        proxy._remember_responses_cache_settle_finish(previous, 200)

        await proxy._wait_for_responses_cache_settle(continuation)

        settle = continuation.trace_context.get("prompt_cache_settle")
        assert settle is not None
        assert settle["same_lineage"] is True
        assert settle["steering"] is False
        assert settle["configured_delay_seconds"] == 0.01

    monkeypatch.setattr(proxy, "_prompt_cache_settle_delay_seconds", lambda _plan: 0.01)
    with proxy._PROMPT_CACHE_SETTLE_LOCK:
        proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()
    try:
        asyncio.run(scenario())
    finally:
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()


def test_fresh_root_turns_share_a_durable_cache_settle_family(monkeypatch):
    async def scenario():
        policy = InitiatorPolicy()
        first_body = _body(
            session="durable-session",
            thread="durable-root",
            turn="root-turn-one",
        )
        second_body = _body(
            session="durable-session",
            thread="durable-root",
            turn="root-turn-two",
        )
        first = UpstreamRequestPlan(
            request_id="root-one",
            upstream_url="https://example.invalid/responses",
            headers=_headers(first_body, policy, "root-one"),
            body=first_body,
            source_body=first_body,
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
            trace_context={"upstream_path": "/responses"},
        )
        second = UpstreamRequestPlan(
            request_id="root-two",
            upstream_url="https://example.invalid/responses",
            headers=_headers(second_body, policy, "root-two"),
            body=second_body,
            source_body=second_body,
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
            trace_context={
                "upstream_path": "/responses",
                "initiator_verdict": {"candidate_initiator": "user"},
            },
        )

        first_identity = _responses_cache_settle_identity(first)
        second_identity = _responses_cache_settle_identity(second)
        assert first_identity is not None
        assert second_identity is not None
        assert first_identity[1] != second_identity[1]
        assert first_identity[2] == second_identity[2]

        proxy._remember_responses_cache_settle_finish(first, 200)
        await proxy._wait_for_responses_cache_settle(second)
        settle = second.trace_context.get("prompt_cache_settle")
        assert settle is not None
        assert settle["same_lineage"] is False

    monkeypatch.setattr(proxy, "_prompt_cache_settle_delay_seconds", lambda _plan: 0.01)
    with proxy._PROMPT_CACHE_SETTLE_LOCK:
        proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()
    try:
        asyncio.run(scenario())
    finally:
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()


def test_no_affinity_request_has_no_cross_lineage_settle_family():
    policy = InitiatorPolicy()
    body = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "unrelated request"}],
            }
        ],
    }
    plan = UpstreamRequestPlan(
        request_id="no-affinity",
        upstream_url="https://example.invalid/responses",
        headers=_headers(body, policy, "no-affinity"),
        body=body,
        source_body=body,
        usage_event=None,
        requested_model="gpt-5.5",
        resolved_model="gpt-5.5",
        trace_context={"upstream_path": "/responses"},
    )

    assert _responses_cache_settle_identity(plan) is None


def test_resolver_only_session_affinity_gets_a_cache_settle_family():
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"x-session-affinity", b"resolver-only-session")],
        }
    )
    body = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }
    headers = build_responses_headers_for_request(
        request,
        body,
        "token",
        initiator_policy=InitiatorPolicy(),
        session_id_resolver=proxy.usage_tracking.request_session_id,
        request_id="resolver-only",
    )
    plan = UpstreamRequestPlan(
        request_id="resolver-only",
        upstream_url="https://example.invalid/responses",
        headers=headers,
        body=body,
        source_body=body,
        usage_event=None,
        requested_model="gpt-5.5",
        resolved_model="gpt-5.5",
        request_affinity=proxy.usage_tracking.request_session_id(request, body),
        trace_context={"upstream_path": "/responses"},
    )

    identity = _responses_cache_settle_identity(plan)

    assert "session_id" not in headers
    assert plan.request_affinity == "resolver-only-session"
    assert identity is not None
    assert "resolver-only-session" in identity[2]


def test_cache_settle_family_encoding_cannot_collide_on_delimiters():
    policy = InitiatorPolicy()

    def plan_for(session: str, thread: str, request_id: str) -> UpstreamRequestPlan:
        body = _body(session=session, thread=thread, turn=request_id)
        return UpstreamRequestPlan(
            request_id=request_id,
            upstream_url="https://example.invalid/responses",
            headers=_headers(body, policy, request_id),
            body=body,
            source_body=body,
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
            trace_context={"upstream_path": "/responses"},
        )

    first = _responses_cache_settle_identity(plan_for("a:b", "c", "colon-one"))
    second = _responses_cache_settle_identity(plan_for("a", "b:c", "colon-two"))

    assert first is not None
    assert second is not None
    assert first[2] != second[2]


def test_cache_settle_uses_source_body_codex_affinity_after_translation():
    source_body = _body(
        session="source-only-session",
        thread="source-only-thread",
        turn="source-only-turn",
    )
    upstream_body = {
        "model": "gpt-5.5",
        "input": source_body["input"],
    }
    plan = UpstreamRequestPlan(
        request_id="source-only",
        upstream_url="https://example.invalid/responses",
        headers=_headers(source_body, InitiatorPolicy(), "source-only"),
        body=upstream_body,
        source_body=source_body,
        usage_event=None,
        requested_model="gpt-5.5",
        resolved_model="gpt-5.5",
        trace_context={"upstream_path": "/responses"},
    )

    identity = _responses_cache_settle_identity(plan)

    assert identity is not None
    assert "source-only-session" in identity[2]
    assert "source-only-thread" in identity[2]


def test_rollout_memory_writer_does_not_share_the_interactive_settle_family():
    policy = InitiatorPolicy()
    interactive_body = _body(
        session="rollout-session",
        thread="rollout-thread",
        turn="rollout-turn",
    )
    rollout_body = _body(
        session="rollout-session",
        thread="rollout-thread",
        turn="rollout-memory-turn",
    )
    rollout_body["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Analyze this rollout and produce JSON\n"
                        "rollout_context: isolated\n"
                        "rendered conversation follows"
                    ),
                }
            ],
        }
    ]

    def plan_for(body: dict, request_id: str) -> UpstreamRequestPlan:
        return UpstreamRequestPlan(
            request_id=request_id,
            upstream_url="https://example.invalid/responses",
            headers=_headers(body, policy, request_id),
            body=body,
            source_body=body,
            usage_event=None,
            requested_model="gpt-5.5",
            resolved_model="gpt-5.5",
            trace_context={"upstream_path": "/responses"},
        )

    interactive = _responses_cache_settle_identity(
        plan_for(interactive_body, "rollout-interactive")
    )
    rollout = _responses_cache_settle_identity(
        plan_for(rollout_body, "rollout-memory")
    )

    assert interactive is not None
    assert rollout is not None
    assert interactive[2] != rollout[2]
    assert "codex-rollout-memory:" in rollout[2]


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.wait = asyncio.Event()

    async def __aiter__(self):
        await self.wait.wait()
        yield b"never"

    async def aclose(self):
        self.closed = True


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class _BuildOnlyClient:
    def build_request(self, method, url, *, headers, json):
        return httpx.Request(method, url, headers=headers, json=json)


def _responses_sse_chunk(payload) -> bytes:
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"data: {json.dumps(payload)}\n\n".encode()


def _managed_bridge_plan(request_id: str) -> UpstreamRequestPlan:
    body = {
        "model": "gpt-5.5",
        "prompt_cache_key": f"{request_id}-thread",
        "input": [],
    }
    return UpstreamRequestPlan(
        request_id=request_id,
        upstream_url="https://example.invalid/responses",
        headers={"x-agent-task-id": f"{request_id}-task"},
        body=body,
        source_body=body,
        usage_event=None,
        requested_model="gpt-5.5",
        resolved_model="gpt-5.5",
        trace_context={"upstream_path": "/responses"},
    )


def test_anthropic_responses_bridge_uses_one_managed_lifecycle(monkeypatch):
    async def scenario():
        stream = _ChunkStream(
            [
                _responses_sse_chunk(
                    {
                        "type": "response.created",
                        "response": {"id": "resp_managed", "model": "gpt-5.5"},
                    }
                ),
                _responses_sse_chunk(
                    {"type": "response.output_text.delta", "delta": "hello"}
                ),
                _responses_sse_chunk(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_managed",
                            "model": "gpt-5.5",
                            "usage": {
                                "input_tokens": 12,
                                "output_tokens": 2,
                                "total_tokens": 14,
                            },
                        },
                    }
                ),
                _responses_sse_chunk("[DONE]"),
            ]
        )
        upstream = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=httpx.Request("POST", "https://example.invalid/responses"),
            extensions={"http_version": b"HTTP/1.1"},
        )
        downstream = _request()
        seen = {}
        finishes = []

        async def fake_open(
            _client,
            _request_value,
            *,
            trace_plan,
            downstream_request,
            active_stream,
        ):
            seen["trace_plan"] = trace_plan
            seen["downstream_request"] = downstream_request
            if active_stream is not None:
                active_stream.send_started = True
                active_stream.upstream = upstream
            return upstream

        def fake_finish(plan, status_code, **details):
            finishes.append((plan, status_code, details))

        monkeypatch.setattr(proxy, "_get_upstream_client", lambda: _BuildOnlyClient())
        monkeypatch.setattr(proxy, "_open_streaming_upstream", fake_open)
        monkeypatch.setattr(proxy, "_finish_usage_and_trace", fake_finish)
        plan = _managed_bridge_plan("managed-bridge")

        response = await proxy.proxy_anthropic_from_responses_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            "gpt-5.5",
            trace_plan=plan,
            downstream_request=downstream,
        )
        output = b""
        async for chunk in response.body_iterator:
            output += chunk

        assert seen == {
            "trace_plan": plan,
            "downstream_request": downstream,
        }
        assert response.headers["content-type"].startswith("text/event-stream")
        assert b'"type":"message_stop"' in output
        assert b'"text":"hello"' in output
        assert len(finishes) == 1
        assert finishes[0][1] == 200
        assert finishes[0][2]["response_text"] == "hello"
        assert finishes[0][2]["response_payload"]["type"] == "message"
        assert finishes[0][2]["usage"]["input_tokens"] == 12
        lifecycle = plan.trace_context["responses_stream_lifecycle"]
        assert lifecycle["termination_cause"] == "source_eof"
        assert lifecycle["completed_event_seen"] is True
        assert lifecycle["teardown_confirmed"] is True
        assert stream.closed
        assert not _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()


def test_anthropic_responses_bridge_preserves_terminal_semantics(monkeypatch):
    async def scenario():
        cases = [
            (
                "failed-terminal",
                [
                    _responses_sse_chunk(
                        {
                            "type": "response.failed",
                            "response": {
                                "error": {
                                    "code": "server_error",
                                    "type": "server_error",
                                    "message": "generation failed",
                                }
                            },
                        }
                    )
                ],
            ),
            (
                "incomplete-terminal",
                [
                    _responses_sse_chunk(
                        {"type": "response.output_text.delta", "delta": "partial"}
                    ),
                    _responses_sse_chunk(
                        {
                            "type": "response.incomplete",
                            "response": {
                                "incomplete_details": {
                                    "reason": "max_output_tokens"
                                }
                            },
                        }
                    ),
                ],
            ),
            ("bare-done-terminal", [_responses_sse_chunk("[DONE]")]),
        ]
        upstreams = []
        streams = []
        for _name, chunks in cases:
            stream = _ChunkStream(chunks)
            streams.append(stream)
            upstreams.append(
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=stream,
                    request=httpx.Request(
                        "POST", "https://example.invalid/responses"
                    ),
                    extensions={"http_version": b"HTTP/1.1"},
                )
            )
        finishes = []

        async def fake_open(
            _client,
            _request_value,
            *,
            trace_plan,
            downstream_request,
            active_stream,
        ):
            upstream = upstreams.pop(0)
            if active_stream is not None:
                active_stream.send_started = True
                active_stream.upstream = upstream
            return upstream

        def record_finish(plan, status_code, **details):
            finishes.append((plan, status_code, details))
            proxy._remember_responses_cache_settle_finish(plan, status_code)

        monkeypatch.setattr(proxy, "_get_upstream_client", lambda: _BuildOnlyClient())
        monkeypatch.setattr(proxy, "_open_streaming_upstream", fake_open)
        monkeypatch.setattr(
            proxy,
            "_finish_usage_and_trace",
            record_finish,
        )

        outputs = {}
        for name, _chunks in cases:
            plan = _managed_bridge_plan(name)
            response = await proxy.proxy_anthropic_from_responses_streaming_response(
                plan.upstream_url,
                plan.headers,
                plan.body,
                "gpt-5.5",
                trace_plan=plan,
            )
            output = b""
            async for chunk in response.body_iterator:
                output += chunk
            outputs[name] = output

        failed_output = outputs["failed-terminal"]
        assert b"event: error" in failed_output
        assert b'"type":"error"' in failed_output
        assert b'"type":"api_error"' in failed_output
        assert b'"type":"server_error"' not in failed_output
        assert b"generation failed" in failed_output
        assert b"message_stop" not in failed_output

        incomplete_output = outputs["incomplete-terminal"]
        assert b'"stop_reason":"max_tokens"' in incomplete_output
        assert b'"type":"message_stop"' in incomplete_output
        assert b"event: error" not in incomplete_output

        bare_done_output = outputs["bare-done-terminal"]
        assert b"event: error" in bare_done_output
        assert b"without a terminal event" in bare_done_output
        assert b"message_stop" not in bare_done_output

        assert [status for _plan, status, _details in finishes] == [502, 200, 502]
        assert finishes[0][2]["response_payload"]["type"] == "error"
        assert finishes[0][2]["response_text"] == "generation failed"
        assert finishes[1][2]["response_payload"]["stop_reason"] == "max_tokens"
        assert finishes[2][2]["response_payload"]["type"] == "error"
        incomplete_lifecycle = finishes[1][0].trace_context[
            "responses_stream_lifecycle"
        ]
        assert incomplete_lifecycle["source_loop_completed"] is False
        assert incomplete_lifecycle["presentation_loop_completed"] is True
        assert incomplete_lifecycle["terminal_eof"] is False
        incomplete_identity = _responses_cache_settle_identity(finishes[1][0])
        assert incomplete_identity is not None
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            assert (
                incomplete_identity[0],
                incomplete_identity[2],
            ) in proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY
        assert all(stream.closed for stream in streams)
        assert not _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    with proxy._PROMPT_CACHE_SETTLE_LOCK:
        proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()


def test_managed_transform_failure_after_completed_is_not_traced_as_success(
    monkeypatch,
):
    async def scenario():
        stream = _ChunkStream(
            [
                _responses_sse_chunk(
                    {
                        "type": "response.completed",
                        "response": {
                            "usage": {
                                "input_tokens": 5,
                                "output_tokens": 1,
                                "total_tokens": 6,
                            }
                        },
                    }
                )
            ]
        )
        upstream = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=httpx.Request("POST", "https://example.invalid/responses"),
            extensions={"http_version": b"HTTP/1.1"},
        )
        finishes = []

        async def fake_open(
            _client,
            _request_value,
            *,
            trace_plan,
            downstream_request,
            active_stream,
        ):
            if active_stream is not None:
                active_stream.send_started = True
                active_stream.upstream = upstream
            return upstream

        async def fail_after_raw_chunk(source_iter):
            async for _chunk in source_iter:
                raise RuntimeError("translator failed")
            if False:
                yield b""

        monkeypatch.setattr(proxy, "_get_upstream_client", lambda: _BuildOnlyClient())
        monkeypatch.setattr(proxy, "_open_streaming_upstream", fake_open)
        monkeypatch.setattr(
            proxy,
            "_finish_usage_and_trace",
            lambda plan, status_code, **details: finishes.append(
                (plan, status_code, details)
            ),
        )
        plan = _managed_bridge_plan("transform-failure")
        response = await proxy.proxy_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            trace_plan=plan,
            stream_transform=fail_after_raw_chunk,
            sync_replay_ids=False,
        )

        try:
            await response.body_iterator.__anext__()
        except RuntimeError as exc:
            assert str(exc) == "translator failed"
        else:
            raise AssertionError("the presentation transform should fail")

        assert len(finishes) == 1
        assert finishes[0][1] == 502
        assert finishes[0][2]["usage"]["input_tokens"] == 5
        assert finishes[0][2]["usage"]["output_tokens"] == 1
        lifecycle = plan.trace_context["responses_stream_lifecycle"]
        assert lifecycle["completed_event_seen"] is True
        assert lifecycle["termination_cause"] == "upstream_error"
        assert lifecycle["upstream_error_type"] == "RuntimeError"
        assert stream.closed
        assert not _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()


def test_closing_transformed_stream_after_raw_completion_is_client_cancelled(
    monkeypatch,
):
    async def scenario():
        stream = _ChunkStream(
            [
                _responses_sse_chunk(
                    {
                        "type": "response.completed",
                        "response": {
                            "usage": {
                                "input_tokens": 9,
                                "output_tokens": 2,
                                "total_tokens": 11,
                            }
                        },
                    }
                )
            ]
        )
        upstream = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=httpx.Request("POST", "https://example.invalid/responses"),
            extensions={"http_version": b"HTTP/1.1"},
        )
        finishes = []

        async def fake_open(
            _client,
            _request_value,
            *,
            trace_plan,
            downstream_request,
            active_stream,
        ):
            if active_stream is not None:
                active_stream.send_started = True
                active_stream.upstream = upstream
            return upstream

        async def translated_with_tail(source_iter):
            async for _chunk in source_iter:
                yield b"translated-prefix"
            yield b"translated-tail"

        monkeypatch.setattr(proxy, "_get_upstream_client", lambda: _BuildOnlyClient())
        monkeypatch.setattr(proxy, "_open_streaming_upstream", fake_open)
        monkeypatch.setattr(
            proxy,
            "_finish_usage_and_trace",
            lambda plan, status_code, **details: finishes.append(
                (plan, status_code, details)
            ),
        )
        plan = _managed_bridge_plan("close-after-completed")
        response = await proxy.proxy_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            trace_plan=plan,
            stream_transform=translated_with_tail,
            trace_details_factory=lambda: {
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            },
            sync_replay_ids=False,
        )

        assert await response.body_iterator.__anext__() == b"translated-prefix"
        await response.body_iterator.aclose()

        assert len(finishes) == 1
        assert finishes[0][1] == 499
        assert finishes[0][2]["usage"]["input_tokens"] == 9
        assert finishes[0][2]["usage"]["output_tokens"] == 2
        lifecycle = plan.trace_context["responses_stream_lifecycle"]
        assert lifecycle["completed_event_seen"] is True
        assert lifecycle["presentation_loop_completed"] is False
        assert lifecycle["termination_cause"] == "downstream_closed"
        identity = _responses_cache_settle_identity(plan)
        assert identity is not None
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            assert (identity[0], identity[2]) in proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY
        assert stream.closed
        assert not _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    with proxy._PROMPT_CACHE_SETTLE_LOCK:
        proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()
        with proxy._PROMPT_CACHE_SETTLE_LOCK:
            proxy._PROMPT_CACHE_LAST_FINISH_BY_FAMILY.clear()


def test_anthropic_responses_bridge_closes_before_first_iteration(monkeypatch):
    async def scenario():
        stream = _BlockingStream()
        upstream = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=httpx.Request("POST", "https://example.invalid/responses"),
            extensions={"http_version": b"HTTP/1.1"},
        )
        finishes = []

        async def fake_open(
            _client,
            _request_value,
            *,
            trace_plan,
            downstream_request,
            active_stream,
        ):
            if active_stream is not None:
                active_stream.send_started = True
                active_stream.upstream = upstream
            return upstream

        monkeypatch.setattr(proxy, "_get_upstream_client", lambda: _BuildOnlyClient())
        monkeypatch.setattr(proxy, "_open_streaming_upstream", fake_open)
        monkeypatch.setattr(
            proxy,
            "_finish_usage_and_trace",
            lambda plan, status_code, **details: finishes.append(
                (plan, status_code, details)
            ),
        )
        plan = _managed_bridge_plan("unstarted-bridge")

        response = await proxy.proxy_anthropic_from_responses_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            "gpt-5.5",
            trace_plan=plan,
        )
        await response.body_iterator.aclose()

        assert len(finishes) == 1
        assert finishes[0][1] == 499
        lifecycle = plan.trace_context["responses_stream_lifecycle"]
        assert lifecycle["termination_cause"] == "downstream_closed"
        assert lifecycle["source_loop_completed"] is False
        assert lifecycle["transport_cancel_confirmed"] is True
        assert lifecycle["teardown_confirmed"] is True
        assert stream.closed
        assert not _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()


def test_anthropic_responses_bridge_renders_request_errors_as_anthropic(monkeypatch):
    async def scenario():
        downstream = _request()

        async def fail_open(
            _client,
            request,
            *,
            trace_plan,
            downstream_request,
            active_stream,
        ):
            assert downstream_request is downstream
            raise httpx.ConnectError("bridge unavailable", request=request)

        monkeypatch.setattr(proxy, "_get_upstream_client", lambda: _BuildOnlyClient())
        monkeypatch.setattr(proxy, "_open_streaming_upstream", fail_open)
        monkeypatch.setattr(proxy, "_finish_usage_and_trace", lambda *args, **kwargs: None)
        plan = _managed_bridge_plan("bridge-connect-error")

        response = await proxy.proxy_anthropic_from_responses_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            "gpt-5.5",
            trace_plan=plan,
            downstream_request=downstream,
        )
        payload = json.loads(response.body)

        assert response.status_code == 502
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "api_error"
        assert not _ACTIVE_RESPONSES_STREAMS

    with _ACTIVE_RESPONSES_STREAMS_LOCK:
        _ACTIVE_RESPONSES_STREAMS.clear()
    try:
        asyncio.run(scenario())
    finally:
        with _ACTIVE_RESPONSES_STREAMS_LOCK:
            _ACTIVE_RESPONSES_STREAMS.clear()


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
