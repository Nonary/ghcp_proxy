# Repository Notes

## Python Environment

Use the repo-local virtualenv for all Python tests and tooling:

```sh
./.venv/bin/pytest ...
```

Do not use bare `pytest` or `python3 -m pytest` unless you have first verified
they resolve inside `./.venv`. The system Python on this machine may not have
project dependencies such as `pytest`, `httpx`, or `fastapi` installed.

## Test Commands

Run focused tests with explicit files or node ids, for example:

```sh
./.venv/bin/pytest -q test_proxy_routes.py::ProxyRoutesTests::test_responses_route_preserves_encrypted_reasoning_for_tool_history_with_cache_lineage
```

Run the normal full suite with:

```sh
./.venv/bin/pytest -q --ignore=mutants
```

The `mutants/` directory is a generated mutation-testing workspace that mirrors
the source tree. Do not include it in normal test collection, searches, or
manual edits unless the task is specifically about mutation testing.

## Cache Debugging

Runtime request traces live under:

```sh
~/Library/Application\ Support/ghcp_proxy/request-trace.jsonl
```

For GPT Responses cache misses, compare `request_started` and
`request_finished` rows for:

- `prompt_cache_key`, `promptCacheKey`, and `previous_response_id`
- `x-interaction-id` and `x-agent-task-id`
- `responses_input_sanitization.encrypted_content_preservation`
- `responses_input_sanitization.encrypted_content_items_dropped`
- `response.usage.cached_input_tokens`

Normal same-lineage GPT Responses turns with a cache lineage hint should
preserve reasoning `encrypted_content`, even when tool history is present.
Only fork/subagent-like foreign-lineage contexts should strip encrypted
reasoning replay.
