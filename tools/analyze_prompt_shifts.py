"""Detect prompt-cache-sensitive request-body shifts in ghcp_proxy traces.

This script compares consecutive requests within each prompt-cache lineage using
the full request-body artifacts written under ``request-bodies/*.json``.  Unlike
trace-level item-prefix summaries, it builds an append-friendly canonical
projection of the upstream prompt material and flags any mutation that appears
inside the first N percent of the previous prompt.

Default usage:

    python tools/analyze_prompt_shifts.py

Useful focused run:

    python tools/analyze_prompt_shifts.py --min-drop 10000 --show 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


NOTICE_PATTERNS = (
    "By the way, an update is available for GHCP Proxy.",
    "5h-usage reminder",
    "Weekly limit:",
)


def default_trace_paths() -> list[str]:
    home = os.path.expanduser("~")
    paths: list[str] = []
    local = os.environ.get("LOCALAPPDATA")
    appdata = os.environ.get("APPDATA")
    if local:
        paths.append(os.path.join(local, "ghcp_proxy", "request-trace.jsonl"))
    if appdata:
        paths.append(os.path.join(appdata, "ghcp_proxy", "request-trace.jsonl"))
    paths.append(os.path.join(home, ".config", "ghcp_proxy", "request-trace.jsonl"))
    paths.append(os.path.join(home, "Library", "Application Support", "ghcp_proxy", "request-trace.jsonl"))
    return paths


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()[:16]


def first_diff_offset(left: str, right: str) -> int | None:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    if len(left) == len(right):
        return None
    return limit


def coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return default


def header_value(headers: dict | None, wanted: str) -> str | None:
    if not isinstance(headers, dict):
        return None
    wanted = wanted.lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == wanted and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def cache_key(start: dict) -> tuple[str | None, str | None]:
    body = start.get("request_body") if isinstance(start.get("request_body"), dict) else {}
    for key in ("prompt_cache_key", "promptCacheKey", "session_id", "sessionId", "previous_response_id"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return key, value.strip()
    headers = start.get("outbound_headers") if isinstance(start.get("outbound_headers"), dict) else {}
    for key in ("x-agent-task-id", "x-parent-agent-id", "x-interaction-id", "x-client-session-id"):
        value = header_value(headers, key)
        if value:
            return key, value
    return None, None


@dataclass
class Row:
    request_id: str
    started_at: str | None
    finished_at: str | None
    model: str
    cache_key_name: str
    cache_key: str
    cached: int
    input_tokens: int
    fresh: int
    x_client_session_id: str | None
    x_interaction_id: str | None
    x_agent_task_id: str | None
    x_parent_agent_id: str | None


def parse_trace(path: str) -> list[Row]:
    starts: dict[str, dict] = {}
    order: list[str] = []
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = obj.get("event")
            request_id = obj.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                continue
            if event == "request_started":
                starts[request_id] = obj
                order.append(request_id)
            elif event == "request_finished" and request_id in starts:
                starts[request_id]["finished"] = obj

    for request_id in order:
        start = starts.get(request_id) or {}
        finish = start.get("finished")
        if not isinstance(finish, dict):
            continue
        usage = (((finish.get("response") or {}).get("usage")) if isinstance(finish.get("response"), dict) else None)
        if not isinstance(usage, dict):
            continue
        key_name, key = cache_key(start)
        model = str(start.get("resolved_model") or start.get("requested_model") or "").strip().lower()
        if not key_name or not key or not model:
            continue
        headers = start.get("outbound_headers") if isinstance(start.get("outbound_headers"), dict) else {}
        input_tokens = coerce_int(usage.get("input_tokens"))
        cached = coerce_int(usage.get("cached_input_tokens"))
        fresh = coerce_int(usage.get("fresh_input_tokens"), default=max(0, input_tokens - cached))
        rows.append(
            Row(
                request_id=request_id,
                started_at=start.get("time"),
                finished_at=finish.get("time"),
                model=model,
                cache_key_name=key_name,
                cache_key=key,
                cached=cached,
                input_tokens=input_tokens,
                fresh=fresh,
                x_client_session_id=header_value(headers, "x-client-session-id"),
                x_interaction_id=header_value(headers, "x-interaction-id"),
                x_agent_task_id=header_value(headers, "x-agent-task-id"),
                x_parent_agent_id=header_value(headers, "x-parent-agent-id"),
            )
        )
    return rows


def artifact_dir_for_trace(trace_path: str, explicit: str | None = None) -> str:
    if explicit:
        return os.path.expanduser(explicit)
    return os.path.join(os.path.dirname(os.path.abspath(trace_path)), "request-bodies")


def load_artifact(body_dir: str, request_id: str) -> dict | None:
    path = os.path.join(body_dir, f"{request_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def prompt_projection(upstream_body: dict) -> tuple[str, list[str], str]:
    """Return append-friendly canonical prompt material.

    The raw JSON body cannot be prefix-compared directly because appending to an
    array changes the old closing bracket/comma structure.  This projection keeps
    non-input config in a stable first line and then serializes each input item as
    one appendable record.  If the current request is a pure append to the prior
    prompt, ``current_projection.startswith(previous_projection)`` is true.
    """
    config = {key: value for key, value in upstream_body.items() if key != "input"}
    config_line = "CONFIG\t" + canonical(config) + "\n"
    input_value = upstream_body.get("input")
    item_lines: list[str] = []
    if isinstance(input_value, list):
        for item in input_value:
            item_lines.append("INPUT\t" + canonical(item) + "\n")
    else:
        item_lines.append("INPUT_SCALAR\t" + canonical(input_value) + "\n")
    return config_line + "".join(item_lines), item_lines, config_line


def text_from_node(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_from_node(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "input_text", "output_text"):
            item = value.get(key)
            if isinstance(item, str):
                parts.append(item)
        for key in ("content", "output"):
            if key in value:
                parts.append(text_from_node(value.get(key)))
        return "".join(parts)
    return ""


def short_text(value: Any, limit: int = 180) -> str:
    text = text_from_node(value).replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def item_brief(item: Any) -> dict:
    if not isinstance(item, dict):
        return {"type": type(item).__name__, "hash": digest(item), "text": short_text(item)}
    text = text_from_node(item)
    result = {
        "type": item.get("type"),
        "role": item.get("role"),
        "name": item.get("name"),
        "status": item.get("status"),
        "hash": digest(item),
    }
    if text:
        result["text_chars"] = len(text)
        result["text"] = short_text(item)
        notices = [pattern for pattern in NOTICE_PATTERNS if pattern in text]
        if notices:
            result["proxy_notice_matches"] = notices
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        result["encrypted_content_chars"] = len(encrypted)
        result["encrypted_content_hash"] = digest(encrypted)
    return {key: value for key, value in result.items() if value is not None}


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a,b],[c,d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    col1 = a + c

    def prob(x: int) -> float:
        return (math.comb(col1, x) * math.comb(n - col1, row1 - x)) / math.comb(n, row1)

    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-15))


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    # Haldane-Anscombe correction keeps zero cells finite.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def analyze_pair(prev: Row, cur: Row, body_dir: str, threshold: float) -> dict | None:
    prev_art = load_artifact(body_dir, prev.request_id)
    cur_art = load_artifact(body_dir, cur.request_id)
    if not prev_art or not cur_art:
        return None
    prev_body = prev_art.get("upstream_body")
    cur_body = cur_art.get("upstream_body")
    if not isinstance(prev_body, dict) or not isinstance(cur_body, dict):
        return None

    prev_projection, prev_lines, prev_config = prompt_projection(prev_body)
    cur_projection, cur_lines, cur_config = prompt_projection(cur_body)
    diff = first_diff_offset(prev_projection, cur_projection)
    append_only = cur_projection.startswith(prev_projection)
    diff_ratio = None if diff is None else diff / max(1, len(prev_projection))
    first75_shift = diff is not None and diff_ratio is not None and diff_ratio < threshold

    config_changed = prev_config != cur_config
    item_common = 0
    for prev_line, cur_line in zip(prev_lines, cur_lines):
        if prev_line != cur_line:
            break
        item_common += 1
    existing_item_changed = item_common < min(len(prev_lines), len(cur_lines))
    item_change_ratio = item_common / max(1, len(prev_lines))
    first75_item_shift = existing_item_changed and item_change_ratio < threshold

    prev_input = prev_body.get("input") if isinstance(prev_body.get("input"), list) else []
    cur_input = cur_body.get("input") if isinstance(cur_body.get("input"), list) else []
    first_mismatch = None
    if existing_item_changed and item_common < len(prev_input) and item_common < len(cur_input):
        first_mismatch = {
            "index": item_common,
            "index_ratio": item_change_ratio,
            "previous": item_brief(prev_input[item_common]),
            "current": item_brief(cur_input[item_common]),
        }
    appended_items = [item_brief(item) for item in cur_input[item_common:]] if item_common == len(prev_input) else []
    prompt_json = canonical(cur_body)
    proxy_notice = any(pattern in prompt_json for pattern in NOTICE_PATTERNS)

    return {
        "prev_projection_chars": len(prev_projection),
        "cur_projection_chars": len(cur_projection),
        "append_only": append_only,
        "first_diff_offset": diff,
        "first_diff_ratio": diff_ratio,
        "first75_shift": first75_shift,
        "config_changed": config_changed,
        "existing_item_changed": existing_item_changed,
        "item_common_prefix": item_common,
        "prev_items": len(prev_lines),
        "cur_items": len(cur_lines),
        "item_change_ratio": item_change_ratio,
        "first75_item_shift": first75_item_shift,
        "first_mismatch": first_mismatch,
        "appended_items": appended_items,
        "proxy_notice_in_current": proxy_notice,
        "wire_sha_changed": prev_art.get("upstream_body_wire_sha256") != cur_art.get("upstream_body_wire_sha256"),
        "prev_wire_sha256": prev_art.get("upstream_body_wire_sha256"),
        "cur_wire_sha256": cur_art.get("upstream_body_wire_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?", help="request-trace.jsonl path")
    parser.add_argument("--body-dir", help="directory containing request-bodies/*.json")
    parser.add_argument("--threshold", type=float, default=0.75, help="previous prompt fraction to flag")
    parser.add_argument("--min-drop", type=int, default=10_000, help="cached-token drop threshold for detailed output")
    parser.add_argument("--show", type=int, default=25)
    parser.add_argument("--model", default="gpt-5.5")
    args = parser.parse_args()

    trace_path = args.trace or next((p for p in default_trace_paths() if os.path.exists(p)), None)
    if not trace_path:
        raise SystemExit("No trace file found; pass a trace path.")
    body_dir = artifact_dir_for_trace(trace_path, args.body_dir)
    if not os.path.isdir(body_dir):
        raise SystemExit(f"Body artifact directory not found: {body_dir}")

    rows = [row for row in parse_trace(trace_path) if row.model == args.model.lower()]
    by_lineage: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        by_lineage[(row.model, row.cache_key_name, row.cache_key)].append(row)

    records: list[dict] = []
    missing = 0
    for key, lineage_rows in by_lineage.items():
        prev = None
        for cur in lineage_rows:
            if prev is not None:
                delta = analyze_pair(prev, cur, body_dir, args.threshold)
                if delta is None:
                    missing += 1
                else:
                    drop = prev.cached - cur.cached
                    records.append({"key": key, "prev": prev, "cur": cur, "drop": drop, **delta})
            prev = cur

    print(f"Trace: {trace_path}")
    print(f"Body artifacts: {body_dir}")
    print(f"Model: {args.model}; lineages: {len(by_lineage)}; comparable pairs: {len(records)}; missing artifacts: {missing}")

    drop_thresholds = [1, 1024, args.min_drop, 50_000, 100_000]
    seen_thresholds = []
    for threshold in drop_thresholds:
        if threshold in seen_thresholds:
            continue
        seen_thresholds.append(threshold)
        drops = [record for record in records if record["drop"] >= threshold]
        nondrops = [record for record in records if record["drop"] < threshold]
        a = sum(1 for record in drops if record["first75_shift"])
        b = len(drops) - a
        c = sum(1 for record in nondrops if record["first75_shift"])
        d = len(nondrops) - c
        print(
            f"drop>={threshold}: drops={len(drops)} first75={a}; "
            f"non_drops={len(nondrops)} first75={c}; "
            f"odds_ratio={odds_ratio(a,b,c,d):.3g}; fisher_p={fisher_two_sided(a,b,c,d):.3g}"
        )

    print()
    counters = Counter()
    for record in records:
        if record["drop"] >= args.min_drop:
            if record["append_only"]:
                counters["append_only"] += 1
            if record["first75_shift"]:
                counters["first75_shift"] += 1
            if record["config_changed"]:
                counters["config_changed"] += 1
            if record["existing_item_changed"]:
                counters["existing_item_changed"] += 1
            if record["proxy_notice_in_current"]:
                counters["proxy_notice_in_current"] += 1
    print(f"Signals among drops >= {args.min_drop}:")
    for key, value in counters.most_common():
        print(f"  {value:4} {key}")

    print()
    print(f"Flagged first-{int(args.threshold*100)}% shifts:")
    shown = 0
    for record in sorted(records, key=lambda r: (not r["first75_shift"], -(r["drop"] or 0), r["cur"].started_at or "")):
        if not record["first75_shift"] and record["drop"] < args.min_drop:
            continue
        if shown >= args.show:
            remaining = max(0, sum(1 for r in records if r["first75_shift"] or r["drop"] >= args.min_drop) - shown)
            if remaining:
                print(f"... {remaining} more")
            break
        shown += 1
        prev: Row = record["prev"]
        cur: Row = record["cur"]
        print("=" * 110)
        print(
            f"{cur.started_at} {cur.request_id} key={record['key'][1]}:{record['key'][2]} "
            f"cached {prev.cached}->{cur.cached} drop={record['drop']} "
            f"input {prev.input_tokens}->{cur.input_tokens}"
        )
        print(
            f"append_only={record['append_only']} first75_shift={record['first75_shift']} "
            f"first_diff_ratio={record['first_diff_ratio']} "
            f"config_changed={record['config_changed']} existing_item_changed={record['existing_item_changed']} "
            f"item_common={record['item_common_prefix']}/{record['prev_items']}->{record['cur_items']}"
        )
        print(
            f"headers session={cur.x_client_session_id} interaction={cur.x_interaction_id} "
            f"task={cur.x_agent_task_id} parent={cur.x_parent_agent_id}"
        )
        if record["first_mismatch"]:
            print("first_mismatch:", json.dumps(record["first_mismatch"], ensure_ascii=False, sort_keys=True))
        appended = record.get("appended_items") or []
        if appended:
            print("appended_items:")
            for item in appended[:8]:
                print(" -", json.dumps(item, ensure_ascii=False, sort_keys=True))
            if len(appended) > 8:
                print(f" ... {len(appended) - 8} more")


if __name__ == "__main__":
    main()
