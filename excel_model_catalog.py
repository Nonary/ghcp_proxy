"""Discover model access from the signed-in Excel add-in, not Copilot."""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time

import httpx

import excel_session_capture
import excel_upstream


MODELS_URL = os.environ.get(
    "GHCP_EXCEL_MODELS_URL",
    excel_upstream.RESPONSES_URL.rsplit("/", 1)[0] + "/responses/access?include_models=true",
).strip()
_CACHE: dict[str, object] = {"key": None, "ts": 0.0, "data": {}}
_LOCK = threading.Lock()
_TTL_SECONDS = 300.0
_FAILURE_TTL_SECONDS = 15.0


def parse_model_catalog(payload: object) -> dict[str, dict]:
    """Normalize the official add-in access response, excluding restricted IDs."""
    if not isinstance(payload, dict) or payload.get("allowed") is not True:
        return {}
    catalog = payload.get("model_catalog")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        return {}
    restricted = catalog.get("restricted_models")
    restricted_ids = {item for item in restricted if isinstance(item, str)} if isinstance(restricted, list) else set()
    result = {}
    for entry in catalog["models"]:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if (
            not isinstance(model_id, str)
            or not excel_upstream.excel_model_id(f"{model_id}-excel")
            or excel_upstream.excel_model_id(model_id) is not None
            or model_id in restricted_ids
            or not excel_upstream.model_record_available(entry)
        ):
            continue
        caps = {}
        label = entry.get("label")
        if isinstance(label, str) and label.strip():
            caps["display_name"] = label.strip()
        efforts = entry.get("efforts")
        if isinstance(efforts, list):
            caps["reasoning_efforts"] = list(dict.fromkeys(
                effort["value"] for effort in efforts
                if isinstance(effort, dict) and effort.get("value") in excel_upstream.EXCEL_REASONING_EFFORTS
            ))
        result[model_id] = caps
    return result


def fetch_model_capabilities(*, verify=True, session_store=None) -> dict[str, dict]:
    """Best-effort, session-scoped discovery with no persisted credentials."""
    store = session_store if session_store is not None else excel_upstream.excel_session_store
    excel_session_capture.refresh_macos_excel_session(store)
    excel_session_capture.refresh_windows_excel_session(store)
    try:
        headers = store.request_headers(stream=False)
    except RuntimeError:
        return {}
    identity = "\n".join((
        MODELS_URL, headers.get("authorization", ""),
        headers.get("x-openai-account-id", ""),
        headers.get("x-openai-account-user-id", ""),
        headers.get("x-basispoints-auth-mode", ""),
    ))
    key = hashlib.sha256(identity.encode()).hexdigest()
    with _LOCK:
        now = time.monotonic()
        ttl = _TTL_SECONDS if _CACHE.get("data") else _FAILURE_TTL_SECONDS
        if _CACHE.get("key") == key and now - float(_CACHE.get("ts", 0.0)) < ttl:
            return copy.deepcopy(_CACHE["data"])
        try:
            with httpx.Client(timeout=5.0, verify=verify, trust_env=True) as client:
                response = client.get(MODELS_URL, headers=headers)
                response.raise_for_status()
                result = parse_model_catalog(response.json())
        except (httpx.HTTPError, ValueError, OSError):
            result = {}
        _CACHE.update(key=key, ts=time.monotonic(), data=result)
        return copy.deepcopy(result)
