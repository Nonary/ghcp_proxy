"""Compare consecutive request fingerprints without exposing prompt text.

Accepts a request-trace.jsonl or ZIP containing it. Outputs CSV; token counts
come from usage, prefix lengths from recorded item fingerprints. SDK adapter
inputs are not the runtime's model-wire requests, and are labelled accordingly;
the runtime's model calls are summarized from ``copilot_sdk_session.model_calls``.

``input_shortfall`` is input_tokens - previous input_tokens - previous
output_tokens.  Near zero, the upstream saw the previous prompt and output
intact, so a cache miss is upstream routing or expiry.  Clearly negative, the
history was rewritten (typically encrypted reasoning dropped by a disk resume).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from contextlib import contextmanager
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


def _model_calls(finished_event):
    calls = (finished_event.get("copilot_sdk_session") or {}).get("model_calls") or []
    return {
        "sdk_model_calls": len(calls) if calls else "",
        "sdk_full_resends": sum(1 for call in calls if not call.get("continuation")) if calls else "",
        "sdk_resumed_chain": sum(1 for call in calls if call.get("resumed_chain_items")) if calls else "",
        "sdk_restored_reasoning": sum(call.get("restored_reasoning") or 0 for call in calls) if calls else "",
        "sdk_call_cached": ";".join(
            f"{call['cached_tokens']}/{call['input_tokens']}" for call in calls
            if call.get("input_tokens") is not None
        ),
    }


def compare(events):
    finished = {r["request_id"]: r for r in events if r.get("event") == "request_finished"}
    previous = {}
    previous_usage = {}
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
            if left != right:
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
        yield {
            "time": event["time"], "request_id": event["request_id"],
            "model": event.get("resolved_model"), "conversation_hash": key[0],
            "representation": "sdk_adapter_input" if event.get("trace", {}).get("sdk") else "responses_source",
            "trace_body_bytes": source.get("original_bytes", ""),
            "input_tokens": usage.get("input_tokens", ""),
            "cached_tokens": usage.get("cached_input_tokens", ""),
            "cache_write_tokens": usage.get("cache_creation_input_tokens", ""),
            "input_items": len(sequence), "previous_items": len(old_sequence),
            "identical_prefix_items": shared, "first_item_difference": difference,
            "changed_parameters": ",".join(
                k.removesuffix("_fingerprint") for k, v in body.items()
                if old and k.endswith("_fingerprint") and k != "body_fingerprint" and old.get(k) != v
            ),
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
    args = parser.parse_args()
    with trace_file(args.trace) as handle:
        rows = list(compare([json.loads(line) for line in handle if line.strip()]))
    if rows:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
