"""Migrate legacy ghcp_proxy runtime files into platform-specific locations."""

from __future__ import annotations

import os
import shutil
import sys

from constants import (
    ACCESS_TOKEN_FILE,
    API_KEY_FILE,
    CLIENT_PROXY_SETTINGS_FILE,
    LEGACY_BILLING_TOKEN_FILE,
    LEGACY_PREMIUM_PLAN_CONFIG_FILE,
    MODEL_ROUTING_CONFIG_FILE,
    REQUEST_ERROR_LOG_FILE,
    REQUEST_TRACE_LOG_FILE,
    SAFEGUARD_CONFIG_FILE,
    SQLITE_CACHE_FILE,
    TOKEN_DIR,
    USAGE_LOG_FILE,
)
from codex_native_ingest import CURSOR_FILE


LEGACY_TOKEN_DIR = os.path.expanduser("~/.config/ghcp_proxy")
_CACHE_ENV_OVERRIDDEN = bool(os.environ.get("GHCP_CACHE_DB_PATH"))


def _same_path(left: str, right: str) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)


def _copy_file_if_missing(source: str, destination: str) -> bool:
    if not os.path.isfile(source):
        return False
    if _same_path(source, destination) or os.path.exists(destination):
        return False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    return True


def migrate_legacy_runtime_files() -> list[tuple[str, str]]:
    """Copy legacy files forward without deleting or overwriting user data."""

    if not os.path.isdir(LEGACY_TOKEN_DIR) or _same_path(LEGACY_TOKEN_DIR, TOKEN_DIR):
        return []

    mappings = [
        ("access-token", ACCESS_TOKEN_FILE),
        ("api-key.json", API_KEY_FILE),
        ("model-routing.json", MODEL_ROUTING_CONFIG_FILE),
        ("client-proxy.json", CLIENT_PROXY_SETTINGS_FILE),
        ("premium-plan.json", LEGACY_PREMIUM_PLAN_CONFIG_FILE),
        ("billing-token", LEGACY_BILLING_TOKEN_FILE),
        ("safeguard.json", SAFEGUARD_CONFIG_FILE),
        ("usage-log.jsonl", USAGE_LOG_FILE),
        ("request-errors.log", REQUEST_ERROR_LOG_FILE),
        ("request-trace.jsonl", REQUEST_TRACE_LOG_FILE),
        ("codex-native-cursor.json", CURSOR_FILE),
    ]
    if not _CACHE_ENV_OVERRIDDEN:
        mappings.extend(
            [
                (".ghcp_proxy-cache-v2.sqlite3", SQLITE_CACHE_FILE),
                (".ghcp_proxy-cache-v2.sqlite3-wal", f"{SQLITE_CACHE_FILE}-wal"),
                (".ghcp_proxy-cache-v2.sqlite3-shm", f"{SQLITE_CACHE_FILE}-shm"),
                (".ghcp_proxy-cache.sqlite3", os.path.join(os.path.dirname(SQLITE_CACHE_FILE), ".ghcp_proxy-cache.sqlite3")),
            ]
        )

    migrated: list[tuple[str, str]] = []
    for filename, destination in mappings:
        source = os.path.join(LEGACY_TOKEN_DIR, filename)
        try:
            if _copy_file_if_missing(source, destination):
                migrated.append((source, destination))
        except OSError as exc:
            print(
                f"runtime migration: failed to copy {source!r} to {destination!r}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    return migrated
