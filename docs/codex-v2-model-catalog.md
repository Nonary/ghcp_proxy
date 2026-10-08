# Codex v2 backend selection

The proxy-generated Codex model catalog declares `multi_agent_version: "v2"`
for every model entry, including Copilot models, Excel aliases, and remapped
aliases. This is separate from adapting v2 tool calls and plaintext assignments.

Codex resolves the backend from a configuration override, then model metadata,
then feature defaults. A parent thread can already be using v2 while a fresh
child, which does not inherit the parent's per-thread selection, falls back to
disabled or v1 behavior when its model catalog omits the selector. Missing
collaboration tools in a child's request therefore do not, by themselves,
exonerate the proxy's generated configuration.

The model selector does not change user concurrency limits or override an
explicit agent-disable setting. Existing threads may retain their previously
selected backend. After loading the proxy change, refresh the Codex integration's
model metadata and reload the client before retrying with fresh coordinators.
Do not interrupt active agents merely to reload configuration.

## Existing threads and explicit v2 enablement

Catalog metadata alone does not prove an already-open thread switched backends.
Check the incoming tool namespace: `collaboration` is the v2 surface;
`multi_agent_v1` means the retry is still using v1.

For an explicit v2 override, enable the feature in the client configuration:

```toml
[features.multi_agent_v2]
enabled = true
max_concurrent_threads_per_session = 20 # Example; preserve your chosen limit.
```

An existing table that sets only `max_concurrent_threads_per_session` does not
enable the feature. Verify the effective setting using the same Codex executable
that runs the client:

```sh
/absolute/path/to/codex features list
```

`multi_agent_v2` should report `true`. Fully reload the Codex client/app-server
when it was started before the setting changed; restarting only the proxy does
not reload the client's captured settings. Then retry with a freshly initialized
coordinator and verify its incoming catalog exposes `collaboration.spawn_agent`.
Feature-status and model-list probes still do not prove a completed live nested
spawn; the final check is an actual child-created grandchild.

## Focused verification

```sh
./.venv/bin/python -m unittest test_proxy_client_config test_codex_sdk_adapter test_approval_routing -q
./.venv/bin/python tools/verify-codex-v2-model-catalog.py --codex /absolute/path/to/codex
```

The probe uses a temporary `CODEX_HOME` and an unreachable model endpoint. It
sends only `initialize` and `model/list`; it creates no chats, spawns no agents,
and makes no model calls. The installed Codex `0.162.0-alpha.2` returned an
omitted backend for the old catalog and `v2` for the patched catalog. This proves
catalog loading and selection metadata, not a completed live nested-agent run.
