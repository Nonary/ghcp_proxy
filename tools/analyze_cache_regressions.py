import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict, Counter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import trace_prompt_decryption

DEFAULT_MIN_DROP = 10_000
DEFAULT_CONTEXT_ROWS = 3
PROMPT_TRACE_JSONL_FIELDS = ("request_body", "upstream_body", "source_body", "request_prompt")
PROMPT_TRACE_BODY_FIELDS = ("request_body", "upstream_body", "upstream_body_wire")
_TRACE_DECRYPTOR = None

PROXY_NOTICE_PATTERNS = (
    "By the way, an update is available for GHCP Proxy.",
    "5h-usage reminder",
    "Weekly limit:",
)

HEADER_FIELDS = (
    ("content-type", "content_type"),
    ("accept", "accept"),
    ("accept-encoding", "accept_encoding"),
    ("x-client-session-id", "x_client_session_id"),
    ("x-client-request-id", "x_client_request_id"),
    ("x-interaction-id", "x_interaction_id"),
    ("x-agent-task-id", "x_agent_task_id"),
    ("x-request-id", "x_request_id"),
    ("x-github-request-id", "x_github_request_id"),
    ("x-parent-agent-id", "x_parent_agent_id"),
    ("x-interaction-type", "x_interaction_type"),
    ("x-openai-subagent", "subagent"),
    ("X-Initiator", "initiator"),
    ("x-initiator", "x_initiator"),
    ("Openai-Intent", "openai_intent"),
    ("Copilot-Integration-Id", "copilot_integration_id"),
    ("User-Agent", "user_agent"),
    ("x-client-machine-id", "x_client_machine_id"),
    ("x-copilot-client-exp-assignment-context", "x_copilot_exp_context"),
    ("x-stainless-retry-count", "x_stainless_retry_count"),
    ("x-stainless-lang", "x_stainless_lang"),
    ("x-stainless-package-version", "x_stainless_package_version"),
    ("x-stainless-os", "x_stainless_os"),
    ("x-stainless-arch", "x_stainless_arch"),
    ("x-stainless-runtime", "x_stainless_runtime"),
    ("x-stainless-runtime-version", "x_stainless_runtime_version"),
    ("Copilot-Vision-Request", "copilot_vision_request"),
    ("anthropic-beta", "anthropic_beta"),
    ("x-github-api-version", "x_github_api_version"),
    ("accept-language", "accept_language"),
    ("sec-fetch-mode", "sec_fetch_mode"),
    ("session_id", "session_header"),
)

AFFINITY_FIELDS = (
    "x_client_session_id",
    "x_interaction_id",
    "x_agent_task_id",
    "x_parent_agent_id",
)

CAUSE_HEADER_FIELDS = {
    "content_type",
    "accept",
    "accept_encoding",
    "x_client_session_id",
    "x_client_request_id",
    "x_interaction_id",
    "x_agent_task_id",
    "x_request_id",
    "x_github_request_id",
    "x_parent_agent_id",
    "x_interaction_type",
    "subagent",
    "initiator",
    "x_initiator",
    "openai_intent",
    "copilot_integration_id",
    "user_agent",
    "x_client_machine_id",
    "x_copilot_exp_context",
    "copilot_vision_request",
    "anthropic_beta",
    "x_github_api_version",
    "session_header",
}

IMPORTANT_BODY_FIELDS = {
    "anthropic_version",
    "instructions",
    "messages",
    "metadata",
    "model",
    "parallel_tool_calls",
    "prompt_cache_key",
    "prompt_cache_retention",
    "promptCacheKey",
    "previous_response_id",
    "reasoning",
    "reasoning_effort",
    "session_id",
    "sessionId",
    "store",
    "stream",
    "system",
    "thinking",
    "tool_choice",
    "tools",
}

ROUTING_FIELDS = (
    "client_path",
    "upstream_path",
    "upstream_host",
    "requested_model",
    "resolved_model",
    "response_model",
    "strategy_name",
    "caller_protocol",
    "upstream_protocol",
    "header_kind",
    "bridge",
)

PROMPT_ID_FIELDS = (
    "prompt_cache_key",
    "previous_response_id",
    "session_id",
    "cache_key_name",
    "cache_key",
)

CONFIG_FINGERPRINT_FIELDS = (
    "instructions",
    "system",
    "tools",
    "tool_choice",
    "reasoning",
    "reasoning_effort",
    "thinking",
    "parallel_tool_calls",
    "metadata",
    "model",
    "stream",
    "store",
    "prompt_cache_retention",
)


def _trace_decryptor():
    global _TRACE_DECRYPTOR
    if _TRACE_DECRYPTOR is None:
        _TRACE_DECRYPTOR = trace_prompt_decryption.TracePromptDecryptor.from_environment()
    return _TRACE_DECRYPTOR


def _decrypt_prompt_trace_fields(payload, fields):
    if isinstance(payload, dict):
        trace_prompt_decryption.decrypt_mapping_fields(payload, fields, _trace_decryptor())
    return payload


def default_trace_paths():
    paths=[]
    local=os.environ.get('LOCALAPPDATA')
    app=os.environ.get('APPDATA')
    home=os.path.expanduser('~')
    if local: paths.append(os.path.join(local,'ghcp_proxy','request-trace.jsonl'))
    if app: paths.append(os.path.join(app,'ghcp_proxy','request-trace.jsonl'))
    paths.append(os.path.join(home,'.config','ghcp_proxy','request-trace.jsonl'))
    paths.append(os.path.join(home,'Library','Application Support','ghcp_proxy','request-trace.jsonl'))
    return paths


def hget(headers, name):
    if not isinstance(headers, dict): return None
    lname=name.lower()
    for k,v in headers.items():
        if isinstance(k,str) and k.lower()==lname:
            return v.strip() if isinstance(v,str) and v.strip() else v
    return None


def dig(d, *keys):
    cur=d
    for k in keys:
        if not isinstance(cur, dict): return None
        cur=cur.get(k)
    return cur


def coerce_int(value, default=0):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return default


def _summary_fingerprints(summary):
    if not isinstance(summary, dict):
        return {}
    out = {}
    for key, value in summary.items():
        if key.endswith("_fingerprint") and isinstance(value, str):
            out[key[: -len("_fingerprint")]] = value
    return out


def _first_diff_offset(left, right):
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    if len(left) == len(right):
        return None
    return limit


def cache_key(start):
    rb=start.get('request_body') if isinstance(start.get('request_body'),dict) else {}
    # Client body summary preserves prompt_cache_key/session ids even when upstream summary drops them.
    for k in ('prompt_cache_key','promptCacheKey','session_id','sessionId','previous_response_id'):
        v=rb.get(k)
        if isinstance(v,str) and v.strip():
            return k, v.strip()
    # fallback to headers
    hdr=start.get('outbound_headers') if isinstance(start.get('outbound_headers'),dict) else {}
    for k in ('x-agent-task-id','x-parent-agent-id','x-interaction-id','x-client-session-id'):
        v=hget(hdr,k)
        if isinstance(v,str) and v.strip(): return k, v.strip()
    return None, None


def seq(summary, side='upstream_body'):
    b=summary.get(side) if isinstance(summary.get(side),dict) else {}
    inp=b.get('input')
    if isinstance(inp,dict) and isinstance(inp.get('sequence'),list): return inp['sequence']
    return []


def item_sig(item):
    if not isinstance(item,dict): return None
    return item.get('item_hash') or item.get('content_hash') or item.get('output_hash')


def common_prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if item_sig(x)!=item_sig(y): break
        n+=1
    return n


def brief_item(item):
    if not isinstance(item, dict): return None
    out={k:item.get(k) for k in ('index','type','role','name','item_hash','content_hash','output_hash','encrypted_content_hash','call_id_hash','content_chars','output_chars','encrypted_content_chars') if k in item}
    return out


def parse(path):
    starts={}
    order=[]
    keepalives=[]
    with open(path,encoding='utf-8') as f:
        for lineno,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            try: o=json.loads(line)
            except Exception as e:
                print(f'bad json line {lineno}: {e}', file=sys.stderr); continue
            _decrypt_prompt_trace_fields(o, PROMPT_TRACE_JSONL_FIELDS)
            ev=o.get('event')
            rid=o.get('request_id')
            if ev=='request_started' and rid:
                starts[rid]=o
                order.append(rid)
            elif ev=='request_finished' and rid:
                st=starts.get(rid)
                if st is not None:
                    st['finished']=o
            elif ev and ev.startswith('parent_keepalive'):
                keepalives.append(o)
    rows=[]
    for rid in order:
        st=starts[rid]
        fin=st.get('finished')
        usage=dig(fin or {}, 'response','usage') or {}
        status=dig(fin or {}, 'response','status_code')
        if not usage: continue
        hdr=st.get('outbound_headers') if isinstance(st.get('outbound_headers'),dict) else {}
        rb=st.get('request_body') if isinstance(st.get('request_body'),dict) else {}
        ub=st.get('upstream_body') if isinstance(st.get('upstream_body'),dict) else {}
        ck_name, ck_val=cache_key(st)
        trace=st.get('trace') if isinstance(st.get('trace'),dict) else {}
        san=trace.get('responses_input_sanitization') if isinstance(trace.get('responses_input_sanitization'),dict) else {}
        pref=trace.get('prompt_cache_prefix') if isinstance(trace.get('prompt_cache_prefix'),dict) else {}
        aff = (
            trace.get('prompt_cache_affinity_drift')
            if isinstance(trace.get('prompt_cache_affinity_drift'), dict)
            else trace.get('prompt_cache_affinity')
            if isinstance(trace.get('prompt_cache_affinity'), dict)
            else None
        )
        response=fin.get('response') if isinstance(fin.get('response'),dict) else {}
        input_tokens=coerce_int(usage.get('input_tokens'), default=coerce_int(usage.get('prompt_tokens')))
        cached_tokens=coerce_int(
            usage.get('cached_input_tokens'),
            default=coerce_int(usage.get('cache_read_input_tokens')),
        )
        fresh_tokens=coerce_int(
            usage.get('fresh_input_tokens'),
            default=max(0, input_tokens-cached_tokens),
        )
        row={
            'id':rid, 'time':st.get('time'), 'status':status,
            'finish_time':fin.get('time'),
            'client_path':st.get('client_path'), 'upstream_path':st.get('upstream_path'),
            'upstream_host':st.get('upstream_host'),
            'requested_model':st.get('requested_model'), 'resolved_model':st.get('resolved_model') or rb.get('model') or ub.get('model'),
            'response_model':response.get('model'),
            'response_id':response.get('id'),
            'upstream_request_id':response.get('upstream_request_id'),
            'usage':usage,
            'input_tokens':input_tokens,
            'cached':cached_tokens,
            'fresh':fresh_tokens,
            'cache_key_name':ck_name, 'cache_key':ck_val,
            'prompt_cache_key': rb.get('prompt_cache_key') or rb.get('promptCacheKey'),
            'previous_response_id': rb.get('previous_response_id'),
            'session_id': rb.get('session_id') or rb.get('sessionId'),
            'body_fp': ub.get('body_fingerprint'),
            'request_body_fp': rb.get('body_fingerprint'),
            'instructions_fp': ub.get('instructions_fingerprint'),
            'tools_fp': ub.get('tools_fingerprint'),
            'reasoning_fp': ub.get('reasoning_fingerprint'),
            'upstream_keys': ub.get('keys') if isinstance(ub.get('keys'), list) else [],
            'request_keys': rb.get('keys') if isinstance(rb.get('keys'), list) else [],
            'upstream_fingerprints': _summary_fingerprints(ub),
            'request_fingerprints': _summary_fingerprints(rb),
            'input_summary': ub.get('input') if isinstance(ub.get('input'), dict) else None,
            'messages_summary': ub.get('messages') if isinstance(ub.get('messages'), dict) else None,
            'tool_count': ub.get('tool_count'),
            'deferred_tool_count': ub.get('deferred_tool_count'),
            'tool_search_present': ub.get('tool_search_present'),
            'reasoning_effort': ub.get('reasoning_effort'),
            'metadata_keys': ub.get('metadata_keys') if isinstance(ub.get('metadata_keys'), list) else [],
            'sanitization': san,
            'sanitizer_diagnostics': trace.get('sanitizer_diagnostics') if isinstance(trace.get('sanitizer_diagnostics'), list) else [],
            'prefix_diag': pref,
            'affinity_diag': aff,
            'prompt_cache_retention_retry': trace.get('prompt_cache_retention_retry') if isinstance(trace.get('prompt_cache_retention_retry'), dict) else None,
            'initiator_verdict': trace.get('initiator_verdict') if isinstance(trace.get('initiator_verdict'), dict) else None,
            'bridge': trace.get('bridge'),
            'strategy_name': trace.get('strategy_name'),
            'caller_protocol': trace.get('caller_protocol'),
            'upstream_protocol': trace.get('upstream_protocol'),
            'header_kind': trace.get('header_kind'),
            'sequence': seq(st,'upstream_body'),
            'rb_sequence': seq(st,'request_body'),
        }
        for header_name, field_name in HEADER_FIELDS:
            row[field_name] = hget(hdr, header_name)
        rows.append(row)
    return rows, keepalives


def _canonical_hash(value):
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()[:16]


def _artifact_body_dir(trace_path, explicit=None):
    if explicit:
        return os.path.expanduser(explicit)
    parent = os.path.dirname(os.path.abspath(trace_path))
    return os.path.join(parent, "request-bodies")


def _load_body_artifact(body_dir, request_id):
    if not body_dir or not request_id:
        return None
    path = os.path.join(body_dir, f"{request_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return _decrypt_prompt_trace_fields(payload, PROMPT_TRACE_BODY_FIELDS)


def _text_from_node(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text_from_node(item) for item in value)
    if isinstance(value, dict):
        chunks = []
        for key in ("text", "input_text", "output_text"):
            item = value.get(key)
            if isinstance(item, str):
                chunks.append(item)
        for key in ("content", "output"):
            nested = value.get(key)
            if nested is not None:
                chunks.append(_text_from_node(nested))
        return "".join(chunks)
    return ""


def _short_text(value, limit=220):
    text = _text_from_node(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _short_value(value, limit=120):
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def _value_brief(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {
            "type": "str",
            "chars": len(value),
            "preview": _short_value(value),
            "hash": _canonical_hash(value),
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "count": len(value),
            "hash": _canonical_hash(value),
            "first_types": [type(item).__name__ for item in value[:5]],
        }
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value.keys())
        return {
            "type": "dict",
            "key_count": len(keys),
            "keys": keys[:24],
            "hash": _canonical_hash(value),
        }
    return {
        "type": type(value).__name__,
        "preview": _short_value(value),
        "hash": _canonical_hash(value),
    }


def _prompt_projection(upstream_body):
    """Append-friendly projection for prefix comparisons.

    Raw JSON is awkward for prefix comparisons because appending to a list
    changes the old closing bracket/comma.  Keep non-input config in a stable
    first line, then serialize each input item as one appendable record.
    """
    config = {key: value for key, value in upstream_body.items() if key != "input"}
    config_line = "CONFIG\t" + json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str) + "\n"
    input_value = upstream_body.get("input")
    item_lines = []
    if isinstance(input_value, list):
        for item in input_value:
            item_lines.append("INPUT\t" + json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str) + "\n")
    else:
        item_lines.append("INPUT_SCALAR\t" + json.dumps(input_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str) + "\n")
    return config_line + "".join(item_lines), item_lines, config_line


def _safe_header_delta(prev_art, cur_art):
    prev_headers = prev_art.get("outbound_headers") if isinstance(prev_art.get("outbound_headers"), dict) else {}
    cur_headers = cur_art.get("outbound_headers") if isinstance(cur_art.get("outbound_headers"), dict) else {}
    interesting = {header.lower() for header, _ in HEADER_FIELDS}
    for headers in (prev_headers, cur_headers):
        for key in headers:
            if not isinstance(key, str):
                continue
            lower = key.lower()
            if lower.startswith("x-") or lower in {
                "user-agent",
                "openai-intent",
                "copilot-integration-id",
                "copilot-vision-request",
                "session_id",
            }:
                interesting.add(lower)
    changes = []
    for header in sorted(interesting):
        previous = hget(prev_headers, header)
        current = hget(cur_headers, header)
        if previous != current:
            changes.append({"header": header, "previous": previous, "current": current})
    return changes


def _item_brief(item):
    if not isinstance(item, dict):
        return {
            "type": type(item).__name__,
            "item_hash": _canonical_hash(item),
            "text": _short_text(item),
        }
    result = {
        "type": item.get("type"),
        "role": item.get("role"),
        "name": item.get("name"),
        "status": item.get("status"),
        "item_hash": _canonical_hash(item),
    }
    text = _short_text(item)
    full_text = _text_from_node(item)
    if text:
        result["text_chars"] = len(full_text)
        result["text"] = text
        notices = [pattern for pattern in PROXY_NOTICE_PATTERNS if pattern in full_text]
        if notices:
            result["proxy_notice_matches"] = notices
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        result["encrypted_content_chars"] = len(encrypted)
        result["encrypted_content_hash"] = _canonical_hash(encrypted)
    return {k: v for k, v in result.items() if v is not None}


def _full_body_delta(body_dir, prev_id, cur_id, side="upstream_body"):
    prev_art = _load_body_artifact(body_dir, prev_id)
    cur_art = _load_body_artifact(body_dir, cur_id)
    if not isinstance(prev_art, dict) or not isinstance(cur_art, dict):
        return None
    prev_body = prev_art.get(side)
    cur_body = cur_art.get(side)
    if not isinstance(prev_body, dict) or not isinstance(cur_body, dict):
        return None

    prev_projection, prev_lines, prev_config = _prompt_projection(prev_body)
    cur_projection, cur_lines, cur_config = _prompt_projection(cur_body)
    projection_first_diff = _first_diff_offset(prev_projection, cur_projection)
    projection_first_diff_ratio = (
        None
        if projection_first_diff is None
        else projection_first_diff / max(1, len(prev_projection))
    )

    top_level_changes = []
    for key in sorted(set(prev_body) | set(cur_body)):
        if key == "input":
            continue
        prev_hash = _canonical_hash(prev_body.get(key))
        cur_hash = _canonical_hash(cur_body.get(key))
        if prev_hash != cur_hash:
            top_level_changes.append(
                {
                    "key": key,
                    "previous_hash": prev_hash,
                    "current_hash": cur_hash,
                }
            )

    prev_input = prev_body.get("input")
    cur_input = cur_body.get("input")
    common = None
    first_mismatch = None
    appended = []
    if isinstance(prev_input, list) and isinstance(cur_input, list):
        common = 0
        for prev_item, cur_item in zip(prev_input, cur_input):
            if _canonical_hash(prev_item) != _canonical_hash(cur_item):
                break
            common += 1
        if common < len(prev_input) or common < len(cur_input):
            if common < len(prev_input) and common < len(cur_input):
                first_mismatch = {
                    "index": common,
                    "previous": _item_brief(prev_input[common]),
                    "current": _item_brief(cur_input[common]),
                }
            if common == len(prev_input):
                appended = [_item_brief(item) for item in cur_input[common:]]

    return {
        "side": side,
        "previous_body_hash": _canonical_hash(prev_body),
        "current_body_hash": _canonical_hash(cur_body),
        "previous_wire_sha256": prev_art.get("upstream_body_wire_sha256"),
        "current_wire_sha256": cur_art.get("upstream_body_wire_sha256"),
        "append_friendly_previous_is_prefix": cur_projection.startswith(prev_projection),
        "append_friendly_config_changed": prev_config != cur_config,
        "append_friendly_first_diff_offset": projection_first_diff,
        "append_friendly_first_diff_ratio": projection_first_diff_ratio,
        "previous_projection_chars": len(prev_projection),
        "current_projection_chars": len(cur_projection),
        "top_level_changes": top_level_changes,
        "input_common_prefix_items": common,
        "previous_input_items": len(prev_input) if isinstance(prev_input, list) else None,
        "current_input_items": len(cur_input) if isinstance(cur_input, list) else None,
        "first_mismatch": first_mismatch,
        "appended_items": appended,
        "proxy_notice_in_previous": any(pattern in json.dumps(prev_body, ensure_ascii=False) for pattern in PROXY_NOTICE_PATTERNS),
        "proxy_notice_in_current": any(pattern in json.dumps(cur_body, ensure_ascii=False) for pattern in PROXY_NOTICE_PATTERNS),
    }


def _changed(row_a, row_b, fields):
    return [field for field in fields if row_a.get(field) != row_b.get(field)]


def _fingerprint_changed(prev, cur, field):
    prev_fps = prev.get("upstream_fingerprints") if isinstance(prev.get("upstream_fingerprints"), dict) else {}
    cur_fps = cur.get("upstream_fingerprints") if isinstance(cur.get("upstream_fingerprints"), dict) else {}
    return prev_fps.get(field) != cur_fps.get(field)


def _compact_change(prev, cur, field):
    before = prev.get(field)
    after = cur.get(field)
    if before == after:
        return None
    return f"{field}:{_short_value(before, 70)}->{_short_value(after, 70)}"


def _reason(category, detail):
    return f"{category}:{detail}"


def classify(prev, cur):
    reasons=[]
    pfx=cur.get('prefix_diag') or {}
    san=cur.get('sanitization') or {}
    aff=cur.get('affinity_diag') or {}
    drop = (prev.get('cached') or 0) - (cur.get('cached') or 0)
    fresh_delta = (cur.get('fresh') or 0) - (prev.get('fresh') or 0)
    if drop > 0:
        reasons.append(_reason('usage', f'cached_tokens_dropped:{drop}'))
    if fresh_delta > 0:
        reasons.append(_reason('usage', f'fresh_tokens_increased:{fresh_delta}'))
    if (cur.get('cached') or 0) == 0 and (cur.get('input_tokens') or 0) > 0:
        reasons.append(_reason('usage', 'zero_cache_read'))
    if cur.get('status') and coerce_int(cur.get('status')) >= 400:
        reasons.append(_reason('response', f"status_{cur.get('status')}"))

    for detail in filter(None, (_compact_change(prev, cur, field) for field in PROMPT_ID_FIELDS)):
        reasons.append(_reason('lineage', detail))

    route_changes = [detail for detail in (_compact_change(prev, cur, field) for field in ROUTING_FIELDS) if detail]
    for detail in route_changes:
        reasons.append(_reason('routing', detail))

    changed_headers = _changed(prev, cur, CAUSE_HEADER_FIELDS)
    for field in changed_headers:
        reasons.append(_reason('headers', _compact_change(prev, cur, field)))

    previous_keys = set(prev.get('upstream_keys') or [])
    current_keys = set(cur.get('upstream_keys') or [])
    if previous_keys != current_keys:
        added = ",".join(sorted(current_keys - previous_keys)) or "-"
        removed = ",".join(sorted(previous_keys - current_keys)) or "-"
        reasons.append(_reason('body_keys', f"added={added} removed={removed}"))

    for field in CONFIG_FINGERPRINT_FIELDS:
        if _fingerprint_changed(prev, cur, field):
            reasons.append(_reason('body_config', f'{field}_changed'))
    for field in ("tool_count", "deferred_tool_count", "tool_search_present", "reasoning_effort", "metadata_keys"):
        detail = _compact_change(prev, cur, field)
        if detail:
            reasons.append(_reason('body_config', detail))

    if cur.get('prompt_cache_retention_retry'):
        reasons.append(_reason('body_config', 'prompt_cache_retention_retry_dropped'))

    if pfx:
        if pfx.get('previous_is_prefix') is True:
            reasons.append(_reason('prompt_prefix', 'outbound_extends_previous'))
        elif pfx.get('previous_is_prefix') is False:
            reasons.append(_reason('prompt_prefix', f"outbound_mismatch@{pfx.get('first_mismatch_index')}"))
    # independent sequence compare (helps when trace diag keyed by wrong lineage)
    cp=common_prefix(prev.get('sequence') or [], cur.get('sequence') or [])
    if prev.get('sequence') and cur.get('sequence'):
        if cp == len(prev['sequence']) and len(cur['sequence']) >= len(prev['sequence']):
            reasons.append(_reason('prompt_sequence', 'extends_previous'))
        else:
            reasons.append(_reason('prompt_sequence', f'mismatch@{cp}'))

    prev_input = prev.get("input_summary") if isinstance(prev.get("input_summary"), dict) else {}
    cur_input = cur.get("input_summary") if isinstance(cur.get("input_summary"), dict) else {}
    for field in ("kind", "count", "item_types", "roles", "has_compaction", "encrypted_reasoning_items"):
        if prev_input.get(field) != cur_input.get(field):
            reasons.append(_reason('input_summary', f"{field}:{_short_value(prev_input.get(field), 80)}->{_short_value(cur_input.get(field), 80)}"))

    if san.get('encrypted_content_items_dropped'):
        reasons.append(_reason('encrypted_reasoning', f"dropped:{san.get('encrypted_content_strip_reason')}"))
    if san.get('encrypted_content_preservation') is False:
        reasons.append(_reason('encrypted_reasoning', 'preservation_disabled'))
    if san.get('reasoning_items_dropped'):
        reasons.append(_reason('encrypted_reasoning', 'reasoning_items_dropped'))
    if aff:
        reasons.append(_reason('trace_affinity', ','.join(aff.get('changed_fields') or [])))
    if cur.get('body_fp') == prev.get('body_fp'):
        reasons.append(_reason('body', 'identical_upstream_body'))
    if not reasons:
        reasons.append(_reason('unknown', 'no_trace_signal'))
    return reasons


def reason_category(reason):
    if not isinstance(reason, str):
        return "unknown"
    return reason.split(":", 1)[0]


def _cache_rate(row):
    input_tokens = row.get("input_tokens") or 0
    if input_tokens <= 0:
        return 0.0
    return (row.get("cached") or 0) / input_tokens


def _lineage_key(row):
    return (str(row.get('resolved_model') or '').lower(), row.get('cache_key_name'), row.get('cache_key'))


def _build_lineages(rows):
    by=defaultdict(list)
    for r in rows:
        if not r.get('cache_key') or not r.get('resolved_model'):
            continue
        by[_lineage_key(r)].append(r)
    return by


def _is_interesting_pair(prev, cur, min_drop, fresh_spike, min_cached_before):
    drop=(prev.get('cached') or 0)-(cur.get('cached') or 0)
    fresh_delta=(cur.get('fresh') or 0)-(prev.get('fresh') or 0)
    if (prev.get('cached') or 0) < min_cached_before:
        return False
    return drop >= min_drop or (fresh_spike > 0 and fresh_delta >= fresh_spike)


def _build_regressions(rows, *, min_drop, fresh_spike, min_cached_before, global_order):
    regressions=[]
    if global_order:
        prev=None
        for cur in rows:
            if prev is not None and _is_interesting_pair(prev, cur, min_drop, fresh_spike, min_cached_before):
                key=_lineage_key(cur)
                regressions.append(((prev.get('cached') or 0)-(cur.get('cached') or 0), key, prev, cur, classify(prev,cur)))
            prev=cur
        return regressions

    for key, items in _build_lineages(rows).items():
        prev=None
        for cur in items:
            if prev is not None and _is_interesting_pair(prev, cur, min_drop, fresh_spike, min_cached_before):
                regressions.append(((prev.get('cached') or 0)-(cur.get('cached') or 0), key, prev, cur, classify(prev,cur)))
            prev=cur
    return regressions


def _json_row(row):
    keep = (
        "id",
        "time",
        "finish_time",
        "status",
        "client_path",
        "upstream_path",
        "upstream_host",
        "requested_model",
        "resolved_model",
        "response_model",
        "response_id",
        "upstream_request_id",
        "input_tokens",
        "cached",
        "fresh",
        "cache_key_name",
        "cache_key",
        "prompt_cache_key",
        "previous_response_id",
        "session_id",
        "body_fp",
        "instructions_fp",
        "tools_fp",
        "reasoning_fp",
        "input_summary",
        "messages_summary",
        "sanitization",
        "prefix_diag",
        "affinity_diag",
        "prompt_cache_retention_retry",
        "bridge",
        "strategy_name",
        "caller_protocol",
        "upstream_protocol",
        "header_kind",
    )
    out = {key: row.get(key) for key in keep if row.get(key) is not None}
    out["headers"] = {
        field: row.get(field)
        for _, field in HEADER_FIELDS
        if row.get(field) is not None
    }
    return out


def _json_regression(drop, key, prev, cur, reasons, body_dir=None):
    record = {
        "drop": drop,
        "fresh_delta": (cur.get("fresh") or 0) - (prev.get("fresh") or 0),
        "cache_rate_previous": _cache_rate(prev),
        "cache_rate_current": _cache_rate(cur),
        "lineage": {"model": key[0], "key_name": key[1], "key": key[2]},
        "previous": _json_row(prev),
        "current": _json_row(cur),
        "reasons": reasons,
        "reason_categories": sorted({reason_category(reason) for reason in reasons}),
    }
    if body_dir:
        delta = _full_body_delta(body_dir, prev["id"], cur["id"], side="upstream_body")
        if delta is not None:
            record["full_body_delta"] = delta
        prev_art = _load_body_artifact(body_dir, prev["id"])
        cur_art = _load_body_artifact(body_dir, cur["id"])
        if isinstance(prev_art, dict) and isinstance(cur_art, dict):
            record["full_header_delta"] = _safe_header_delta(prev_art, cur_art)
    return record


def _print_overview(rows, by, regressions, min_drop):
    total_input = sum(row.get("input_tokens") or 0 for row in rows)
    total_cached = sum(row.get("cached") or 0 for row in rows)
    total_fresh = sum(row.get("fresh") or 0 for row in rows)
    rate = (100.0 * total_cached / total_input) if total_input else 0.0
    zero_cache_large = [
        row
        for row in rows
        if (row.get("input_tokens") or 0) >= min_drop and (row.get("cached") or 0) == 0
    ]
    print(
        f"Usage: input={total_input:,} cached={total_cached:,} fresh={total_fresh:,} "
        f"cache_read_rate={rate:.1f}%"
    )
    print(
        f"Lineages: {len(by)}; comparable cache busts: {len(regressions)} "
        f"(threshold={min_drop:,}); large zero-cache requests: {len(zero_cache_large)}"
    )


def _print_top_counters(regressions):
    signal_counter=Counter()
    category_counter=Counter()
    header_counter=Counter()
    routing_counter=Counter()
    body_counter=Counter()
    for _,_,_,_,reasons in regressions:
        for reason in reasons:
            signal_counter[reason]+=1
            category_counter[reason_category(reason)]+=1
            if reason.startswith("headers:") and ":" in reason:
                header_counter[reason.split(":", 2)[1]] += 1
            if reason.startswith("routing:") and ":" in reason:
                routing_counter[reason.split(":", 2)[1]] += 1
            if reason.startswith("body_config:") and ":" in reason:
                body_counter[reason.split(":", 1)[1]] += 1
    print("Top categories:")
    for key,value in category_counter.most_common(12):
        print(f"  {value:3} {key}")
    print("Top signals:")
    for key,value in signal_counter.most_common(20):
        print(f"  {value:3} {key}")
    if header_counter:
        print("Header drift:")
        for key,value in header_counter.most_common(15):
            print(f"  {value:3} {key}")
    if routing_counter:
        print("Routing drift:")
        for key,value in routing_counter.most_common(10):
            print(f"  {value:3} {key}")
    if body_counter:
        print("Body config drift:")
        for key,value in body_counter.most_common(12):
            print(f"  {value:3} {key}")


def _print_lineage_health(by, regressions, limit=10):
    bust_counts=Counter()
    worst_drop=defaultdict(int)
    for drop,key,_,_,_ in regressions:
        bust_counts[key]+=1
        worst_drop[key]=max(worst_drop[key], drop)
    ranked=sorted(
        by.items(),
        key=lambda item: (bust_counts[item[0]], worst_drop[item[0]], len(item[1])),
        reverse=True,
    )
    print("Lineage health:")
    for key, items in ranked[:limit]:
        cached_values=[row.get("cached") or 0 for row in items]
        input_values=[row.get("input_tokens") or 0 for row in items]
        avg_cached=sum(cached_values)/len(cached_values) if cached_values else 0
        avg_input=sum(input_values)/len(input_values) if input_values else 0
        print(
            f"  busts={bust_counts[key]:2} rows={len(items):3} worst_drop={worst_drop[key]:7,} "
            f"avg_cached={avg_cached:8.0f} avg_input={avg_input:8.0f} "
            f"{key[0]} {key[1]}:{_short_value(key[2], 90)}"
        )


def _print_decryption_status():
    status = _trace_decryptor().status_line()
    if status:
        print(status, file=sys.stderr)


def main():
    ap=argparse.ArgumentParser(description='Find prompt-cache regressions in ghcp_proxy request-trace.jsonl')
    ap.add_argument('trace', nargs='?', help='request-trace.jsonl path')
    ap.add_argument('--min-drop', type=int, default=DEFAULT_MIN_DROP)
    ap.add_argument(
        '--fresh-spike',
        type=int,
        default=DEFAULT_MIN_DROP,
        help='also flag pairs where fresh input tokens increase by at least this much; set 0 to disable',
    )
    ap.add_argument('--min-cached-before', type=int, default=0, help='ignore pairs whose previous cached tokens were below this')
    ap.add_argument('--limit', type=int, default=80)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--model', action='append', help='only include resolved model; repeatable')
    ap.add_argument('--lineage', help='substring filter for cache key / prompt_cache_key / session id')
    ap.add_argument('--body-dir', help='directory containing full request-bodies/*.json artifacts')
    ap.add_argument('--format', choices=('text','json'), default='text')
    ap.add_argument('--lineage-limit', type=int, default=10)
    ap.add_argument(
        '--global-order',
        action='store_true',
        help='compare each completed request to the immediately previous completed request, not previous same lineage',
    )
    ns=ap.parse_args()
    path=ns.trace
    if not path:
        path=next((p for p in default_trace_paths() if os.path.exists(p)), None)
    if not path:
        raise SystemExit('No trace file found; pass a path')
    rows, keepalives=parse(path)
    if ns.model:
        wanted={model.lower() for model in ns.model}
        rows=[row for row in rows if str(row.get('resolved_model') or '').lower() in wanted]
    if ns.lineage:
        needle=ns.lineage.lower()
        rows=[
            row for row in rows
            if any(
                needle in str(row.get(field) or '').lower()
                for field in ('cache_key','prompt_cache_key','previous_response_id','session_id','x_agent_task_id','x_interaction_id')
            )
        ]

    by=_build_lineages(rows)
    regressions=_build_regressions(
        rows,
        min_drop=ns.min_drop,
        fresh_spike=ns.fresh_spike,
        min_cached_before=ns.min_cached_before,
        global_order=ns.global_order,
    )
    regressions.sort(key=lambda x: (x[3].get('time') or '', -x[0]))

    body_dir = _artifact_body_dir(path, ns.body_dir)
    if not os.path.isdir(body_dir):
        body_dir = None

    if ns.format == 'json':
        payload = {
            "trace": path,
            "body_dir": body_dir,
            "completed_usage_rows": len(rows),
            "parent_keepalive_events": len(keepalives),
            "lineages": len(by),
            "min_drop": ns.min_drop,
            "fresh_spike": ns.fresh_spike,
            "global_order": ns.global_order,
            "regressions": [
                _json_regression(drop, key, prev, cur, reasons, body_dir=body_dir)
                for drop, key, prev, cur, reasons in regressions
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        _print_decryption_status()
        return

    print(f'Trace: {path}')
    print(f'Completed usage rows: {len(rows)}; parent keepalive events: {len(keepalives)}')
    _print_overview(rows, by, regressions, ns.min_drop)

    prompt_rows = [r for r in rows if r.get("prompt_cache_key")]
    for field in ("x_client_session_id", "x_interaction_id", "x_agent_task_id", "x_parent_agent_id"):
        buckets = defaultdict(set)
        for row in prompt_rows:
            value = row.get(field)
            if isinstance(value, str) and value:
                buckets[(str(row.get("resolved_model") or "").lower(), value)].add(row["prompt_cache_key"])
        collisions = {key: vals for key, vals in buckets.items() if len(vals) > 1}
        if collisions:
            print(f'Affinity collisions on {field}: {len(collisions)}')
            for (model, value), keys in list(collisions.items())[:10]:
                joined = ", ".join(sorted(keys)[:5])
                suffix = "" if len(keys) <= 5 else f", ... +{len(keys) - 5}"
                print(f'  {model} {field}={value}: {joined}{suffix}')
        else:
            print(f'Affinity collisions on {field}: 0')

    if body_dir:
        print(f'Body artifacts: {body_dir}')
    else:
        print('Body artifacts: unavailable; pass --body-dir or enable GHCP_DUMP_REQUEST_BODIES for full diffs')

    _print_top_counters(regressions)
    _print_lineage_health(by, regressions, limit=ns.lineage_limit)
    print()
    if ns.limit <= 0 and not ns.all:
        _print_decryption_status()
        return
    shown=0
    for drop,key,prev,cur,reasons in regressions:
        shown+=1
        print('='*110)
        print(f"{cur['time']} {cur['id']} model={cur['resolved_model']} key={key[1]}:{key[2]}")
        fresh_delta=(cur.get('fresh') or 0)-(prev.get('fresh') or 0)
        print(
            f"cached {prev['cached']:,} -> {cur['cached']:,} (drop {drop:,}); "
            f"input {prev['input_tokens']:,} -> {cur['input_tokens']:,}; "
            f"fresh {prev['fresh']:,} -> {cur['fresh']:,} (delta {fresh_delta:,}); "
            f"cache_rate {_cache_rate(prev)*100:.1f}% -> {_cache_rate(cur)*100:.1f}%"
        )
        print(f"prev {prev['time']} {prev['id']}")
        print('reasons:', '; '.join(reasons) if reasons else '(none)')
        for label,row in [('prev',prev),('cur ',cur)]:
            print(f"  {label} hdr session={row.get('x_client_session_id')} interaction={row.get('x_interaction_id')} task={row.get('x_agent_task_id')} parent={row.get('x_parent_agent_id')} type={row.get('x_interaction_type')}")
            print(f"  {label} body_fp={row.get('body_fp')} instr={row.get('instructions_fp')} tools={row.get('tools_fp')} san={row.get('sanitization')}")
        cp=common_prefix(prev.get('sequence') or [], cur.get('sequence') or [])
        print(f"  sequence common_prefix={cp}/{len(prev.get('sequence') or [])} -> cur_count={len(cur.get('sequence') or [])}")
        if cp < len(prev.get('sequence') or []) and prev.get('sequence') and cur.get('sequence'):
            print('  prev mismatch item:', brief_item(prev['sequence'][cp]) if cp < len(prev['sequence']) else None)
            print('  cur  mismatch item:', brief_item(cur['sequence'][cp]) if cp < len(cur['sequence']) else None)
        if cur.get('prefix_diag'): print('  trace prefix_diag:', json.dumps(cur['prefix_diag'], sort_keys=True))
        if cur.get('affinity_diag'): print('  trace affinity_diag:', json.dumps(cur['affinity_diag'], sort_keys=True))
        if body_dir:
            prev_art = _load_body_artifact(body_dir, prev["id"])
            cur_art = _load_body_artifact(body_dir, cur["id"])
            if isinstance(prev_art, dict) and isinstance(cur_art, dict):
                header_changes = _safe_header_delta(prev_art, cur_art)
                if header_changes:
                    print("  full header changes:")
                    for change in header_changes[:18]:
                        print("   -", json.dumps(change, sort_keys=True, ensure_ascii=False))
                    if len(header_changes) > 18:
                        print(f"   ... {len(header_changes) - 18} more header changes")
            delta = _full_body_delta(body_dir, prev["id"], cur["id"], side="upstream_body")
            if isinstance(delta, dict):
                print(
                    "  full upstream body:",
                    f"hash {delta.get('previous_body_hash')} -> {delta.get('current_body_hash')};",
                    f"input common {delta.get('input_common_prefix_items')}/"
                    f"{delta.get('previous_input_items')} -> {delta.get('current_input_items')}",
                )
                print(
                    "  append-friendly prompt:",
                    f"previous_is_prefix={delta.get('append_friendly_previous_is_prefix')}",
                    f"config_changed={delta.get('append_friendly_config_changed')}",
                    f"first_diff_ratio={delta.get('append_friendly_first_diff_ratio')}",
                    f"projection_chars={delta.get('previous_projection_chars')}->{delta.get('current_projection_chars')}",
                )
                if delta.get("proxy_notice_in_previous") or delta.get("proxy_notice_in_current"):
                    print(
                        "  proxy notice text in prompt:",
                        f"previous={bool(delta.get('proxy_notice_in_previous'))}",
                        f"current={bool(delta.get('proxy_notice_in_current'))}",
                    )
                if delta.get("top_level_changes"):
                    print("  top-level prompt changes:", json.dumps(delta["top_level_changes"], sort_keys=True))
                if delta.get("first_mismatch"):
                    print("  first full-body mismatch:", json.dumps(delta["first_mismatch"], sort_keys=True))
                appended = delta.get("appended_items") or []
                if appended:
                    print("  appended prompt items:")
                    for item in appended[:8]:
                        print("   -", json.dumps(item, sort_keys=True, ensure_ascii=False))
                    if len(appended) > 8:
                        print(f"   ... {len(appended) - 8} more appended items")
        if shown>=ns.limit and not ns.all:
            remaining=len(regressions)-shown
            if remaining>0: print(f'... {remaining} more (use --all)')
            break
    _print_decryption_status()

if __name__=='__main__': main()
