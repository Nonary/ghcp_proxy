# Repository Notes

## Python Environment

Use the repo-local virtualenv for all Python tests and tooling:

```sh
./.venv/bin/pytest ...
```

Do not use bare `pytest` or `python3 -m pytest` unless you have first verified
they resolve inside `./.venv`. The system Python on this machine may not have
project dependencies such as `pytest`, `httpx`, or `fastapi` installed.

## Tests

The top-level pytest suite has been removed. Use syntax checks or targeted
manual verification for changes.

The `mutants/` directory is a generated mutation-testing workspace that mirrors
the source tree. Do not include it in normal searches or manual edits unless
the task is specifically about mutation testing.

## Prompt Debugging

Runtime request traces live under:

```sh
~/Library/Application\ Support/ghcp_proxy/request-trace.jsonl
```

Full prompt and request-body traces are only written when
`debug_prompt_logging_enabled` is true in the client proxy settings. It defaults
to false.
