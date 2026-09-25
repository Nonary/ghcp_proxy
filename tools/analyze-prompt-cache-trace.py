"""Compare consecutive request fingerprints without exposing prompt text.

Accepts a request-trace.jsonl or ZIP containing it. Outputs CSV; token counts
come from usage, prefix lengths from recorded item fingerprints. SDK adapter
inputs are not the runtime's model-wire requests, and are labelled accordingly;
the runtime's model calls are summarized from ``copilot_sdk_session.model_calls``.

``input_shortfall`` is input_tokens - previous input_tokens - previous
output_tokens.  Near zero, the upstream saw the previous prompt and output
intact, so a cache miss is upstream routing or expiry.  Clearly negative, the
history was rewritten (typically encrypted reasoning dropped by a disk resume).

Items are compared without ``item_hash``, which also covers caller metadata
the SDK adapter does not forward; the content fingerprints still must match.

``miss_cause`` explains a request that read less than the conversation's
previous input (see docs/prompt-cache-investigation-2026-09-24.md):
compaction; idle_over_30m (Copilot's cache lives 30 minutes); settings_changed
(model, effort, instructions or tools); retried_after_failed_call;
session_resumed (the SDK session was not live, with the reason); failed.
Losses under 1,024 tokens are cache-block rounding.  On a live chain, Copilot can also serve
a call from its 512-token block cache (block_cache), from an older entry
(stale_cache) or from nothing (cold_cache); these also happen with the
runtime's own transport and no proxy code in the path.  ``--summary`` prints
the lost tokens per cause.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


@contextmanager
def trace_file(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            with archive.open("request-trace.jsonl") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                    yield handle
    else:
        with path.open(encoding="utf-8") as handle:
            yield handle


def _usage(finished_event):
    return (finished_event.get("response") or {}).get("usage") or {}


def _shortfall(usage, previous_usage):
    try:
        return usage["input_tokens"] - previous_usage["input_tokens"] - previous_usage["output_tokens"]
    except (KeyError, TypeError):
        return ""


def _call_cached(calls):
    return ";".join(
        f"{call.get('transport', '')[:2]}:{call['cached_tokens']}/{call['input_tokens']}" for call in calls
        if call.get("input_tokens") is not None
    )


def _call_error(call):
    error = call.get("error")
    if isinstance(error, dict):
        return error.get("code") or error.get("type") or error.get("event") or "error"
    return str(call.get("status") or "error")


def _model_calls(finished_event):
    session = finished_event.get("copilot_sdk_session") or {}
    calls = session.get("model_calls") or []
    background = session.get("background_model_calls") or []
    return {
        "sdk_model_calls": len(calls) if calls else "",
        "sdk_full_resends": sum(1 for call in calls if not call.get("continuation")) if calls else "",
        "sdk_resumed_chain": sum(1 for call in calls if call.get("resumed_chain_items")) if calls else "",
        "sdk_restored_reasoning": sum(call.get("restored_reasoning") or 0 for call in calls) if calls else "",
        "sdk_call_cached": _call_cached(calls),
        "sdk_call_errors": ";".join(_call_error(call) for call in calls if call.get("error") or call.get("status")),
        "sdk_background_calls": _call_cached(background) or (len(background) if background else ""),
        "sdk_interrupted": bool(session.get("interrupted")) or "",
        "sdk_error": (session.get("error") or "")[:120],
    }


def _same_item(left, right):
    return {k: v for k, v in left.items() if k != "item_hash"} == {k: v for k, v in right.items() if k != "item_hash"}


# Copilot's runtime reports a 1,800 s cache TTL; a prefix idle for 12 minutes
# still hit, one idle for 31 minutes did not.
_CACHE_TTL_SECONDS = 1800
_PROMPT_SETTINGS = {"instructions", "tools", "reasoning", "model", "text"}


def _seconds(timestamp):
    return datetime.fromisoformat(timestamp).timestamp()


def _miss_cause(event, usage, last, changed, sdk_session):
    """Why a request read less than the conversation's previous input."""
    total, cached = usage.get("input_tokens"), usage.get("cached_input_tokens")
    if last is None:
        return ""
    if not total:
        return "failed"
    # Less than a cache block is rounding, not a miss.
    if cached is None or cached >= last["input"] - 1024:
        return ""
    if "compact" in (event.get("upstream_path") or "") or total < 0.7 * last["input"]:
        return "compaction"
    if last["finished"] and _seconds(event["time"]) - _seconds(last["finished"]) > _CACHE_TTL_SECONDS:
        return "idle_over_30m"
    if changed & _PROMPT_SETTINGS:
        return "settings_changed"
    calls = sdk_session.get("model_calls") or []
    if any(call.get("error") or call.get("status") for call in calls) or any(
        call.get("input_tokens") is None for call in calls[:-1]
    ):
        return "retried_after_failed_call"
    operation = sdk_session.get("operation") or ""
    if operation.startswith(("resume", "create")):
        return f"session_resumed:{sdk_session.get('reuse_miss') or operation}"
    if cached == 0:
        return "cold_cache"
    return "block_cache" if cached % 512 == 0 else "stale_cache"


def compare(events):
    finished = {r["request_id"]: r for r in events if r.get("event") == "request_finished"}
    previous = {}
    previous_usage = {}
    # The conversation's last successful request, across /responses and
    # /responses/compact.
    last_ok = {}
    for event in events:
        if event.get("event") != "request_started":
            continue
        body = event.get("request_body") or {}
        sequence = body.get("input", {}).get("sequence", [])
        key = (body.get("prompt_cache_key_fingerprint") or body.get("session_id") or event["request_id"],
               event.get("resolved_model"), event.get("upstream_path"))
        old = previous.get(key)
        old_sequence = old.get("input", {}).get("sequence", []) if old else []
        shared = 0
        for left, right in zip(old_sequence, sequence):
            if not _same_item(left, right):
                break
            shared += 1
        difference = ""
        if shared < min(len(old_sequence), len(sequence)):
            left, right = old_sequence[shared], sequence[shared]
            difference = f"input[{shared}]: " + ",".join(sorted(
                field for field in set(left) | set(right) if left.get(field) != right.get(field)
            ))
        elif old:
            difference = "append_only" if shared == len(old_sequence) else "input_shortened"
        finished_event = finished.get(event["request_id"], {})
        usage = _usage(finished_event)
        sdk_session = finished_event.get("copilot_sdk_session") or {}
        appended = sequence[len(old_sequence):] if old and difference == "append_only" else []
        source = event.get("source_body", {})
        changed = {
            k.removesuffix("_fingerprint") for k, v in body.items()
            if old and k.endswith("_fingerprint") and k != "body_fingerprint" and old.get(k) != v
        }
        conversation = (key[0], key[1])
        last = last_ok.get(conversation)
        cause = _miss_cause(event, usage, last, changed, sdk_session)
        if usage.get("input_tokens"):
            last_ok[conversation] = {"input": usage["input_tokens"], "finished": finished_event.get("time")}
        yield {
            "time": event["time"], "request_id": event["request_id"],
            "model": event.get("resolved_model"), "conversation_hash": key[0],
            "representation": "sdk_adapter_input" if event.get("trace", {}).get("sdk") else "responses_source",
            "trace_body_bytes": source.get("original_bytes", ""),
            "input_tokens": usage.get("input_tokens", ""),
            "cached_tokens": usage.get("cached_input_tokens", ""),
            "cache_write_tokens": usage.get("cache_creation_input_tokens", ""),
            "miss_cause": cause,
            "lost_tokens": (last["input"] - (usage.get("cached_input_tokens") or 0)) if cause and cause != "failed" else "",
            "input_items": len(sequence), "previous_items": len(old_sequence),
            "identical_prefix_items": shared, "first_item_difference": difference,
            "changed_parameters": ",".join(sorted(changed)),
            "last_input_type": sequence[-1].get("type", "") if sequence else "",
            "last_input_role": sequence[-1].get("role", "") if sequence else "",
            "turn_boundary": any(
                item.get("type") == "message" and item.get("role") == "user" for item in appended
            ),
            "input_shortfall": _shortfall(usage, previous_usage.get(key)) if old else "",
            "sdk_operation": sdk_session.get("operation", "unrecorded"),
            "sdk_reuse_miss": sdk_session.get("reuse_miss") or "",
            **_model_calls(finished_event),
        }
        previous[key] = body
        previous_usage[key] = usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--summary", action="store_true", help="print lost tokens per miss cause instead of rows")
    args = parser.parse_args()
    with trace_file(args.trace) as handle:
        rows = list(compare([json.loads(line) for line in handle if line.strip()]))
    if args.summary:
        totals = {}
        for row in rows:
            if row["miss_cause"]:
                count, lost = totals.get(row["miss_cause"], (0, 0))
                totals[row["miss_cause"]] = (count + 1, lost + (row["lost_tokens"] or 0))
        for cause, (count, lost) in sorted(totals.items(), key=lambda item: -item[1][1]):
            print(f"{cause:40} {count:5} requests {lost:12,} tokens")
    elif rows:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
