"""Central runtime wiring for the proxy's extracted modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Thread
from typing import Callable

import auth as auth_module
import dashboard as dashboard_module
import usage_tracking as usage_tracking_module
from initiator_policy import InitiatorPolicy


@dataclass(frozen=True)
class ProxySettings:
    codex_config_file: str
    codex_proxy_config: str
    claude_settings_file: str
    claude_proxy_settings: dict
    claude_max_context_tokens: str
    claude_max_output_tokens: str


class ProxyRuntime:
    """Owns the compatibility wiring between extracted modules."""

    def __init__(
        self,
        *,
        settings_provider: Callable[[], ProxySettings],
        load_api_key_payload: Callable[[], dict],
        sqlite_cache_enabled: Callable[[], bool],
        sqlite_cache_error: Callable[[], str | None],
        sqlite_connect: Callable,
        sqlite_cache_put: Callable[[str, dict], None],
        collect_official_premium_payload: Callable[..., dict],
        get_official_premium_payload: Callable[..., dict],
        notify_dashboard_stream_listeners: Callable[[], None],
        utc_now: Callable,
        utc_now_iso: Callable[[], str],
        record_usage_event: Callable[[dict], None],
        rewrite_usage_log_locked: Callable[[list[dict]], None],
        snapshot_usage_events: Callable[[], list[dict]],
        snapshot_all_usage_events: Callable[[], list[dict]],
        usage_log_lock: Callable[[], object] | None = None,
        recent_usage_events: Callable[[], object] | None = None,
        archived_usage_events: Callable[[], object] | None = None,
        session_request_id_lock: Callable[[], object] | None = None,
        latest_server_request_ids_by_chain: Callable[[], object] | None = None,
        active_server_request_ids_by_request: Callable[[], object] | None = None,
        latest_claude_user_session_contexts: Callable[[], object] | None = None,
        thread_class: Callable[[], type] | type = Thread,
    ):
        self._settings_provider = settings_provider
        self._load_api_key_payload = load_api_key_payload
        self._sqlite_cache_enabled = sqlite_cache_enabled
        self._sqlite_cache_error = sqlite_cache_error
        self._sqlite_connect = sqlite_connect
        self._sqlite_cache_put = sqlite_cache_put
        self._collect_official_premium_payload = collect_official_premium_payload
        self._get_official_premium_payload = get_official_premium_payload
        self._notify_dashboard_stream_listeners = notify_dashboard_stream_listeners
        self._utc_now = utc_now
        self._utc_now_iso = utc_now_iso
        self._record_usage_event = record_usage_event
        self._rewrite_usage_log_locked = rewrite_usage_log_locked
        self._snapshot_usage_events = snapshot_usage_events
        self._snapshot_all_usage_events = snapshot_all_usage_events
        self._usage_log_lock = usage_log_lock
        self._recent_usage_events = recent_usage_events
        self._archived_usage_events = archived_usage_events
        self._session_request_id_lock = session_request_id_lock
        self._latest_server_request_ids_by_chain = latest_server_request_ids_by_chain
        self._active_server_request_ids_by_request = active_server_request_ids_by_request
        self._latest_claude_user_session_contexts = latest_claude_user_session_contexts
        self._thread_class = thread_class

    @property
    def dashboard_stream_version(self) -> int:
        return getattr(dashboard_module, "_dashboard_stream_version", 0)

    def initialize(self, initiator_policy: InitiatorPolicy):
        self.load_archived_usage_history()
        self.load_usage_history()
        initiator_policy.seed_from_usage_events(self._snapshot_usage_events())
        self.seed_cached_payloads_from_sqlite()

    def _apply_auth_settings(self):
        settings = self._settings_provider()
        auth_module.CODEX_CONFIG_FILE = settings.codex_config_file
        auth_module.CODEX_CONFIG_DIR = (
            os.path.dirname(settings.codex_config_file) or auth_module.CODEX_CONFIG_DIR
        )
        auth_module.CODEX_PROXY_CONFIG = settings.codex_proxy_config
        auth_module.CLAUDE_SETTINGS_FILE = settings.claude_settings_file
        auth_module.CLAUDE_CONFIG_DIR = (
            os.path.dirname(settings.claude_settings_file) or auth_module.CLAUDE_CONFIG_DIR
        )
        auth_module.CLAUDE_PROXY_SETTINGS = settings.claude_proxy_settings
        auth_module.CLAUDE_MAX_CONTEXT_TOKENS = settings.claude_max_context_tokens
        auth_module.CLAUDE_MAX_OUTPUT_TOKENS = settings.claude_max_output_tokens
        auth_module._load_api_key_payload = self._load_api_key_payload

    def _apply_dashboard_runtime(
        self,
        *,
        include_sqlite_state: bool = False,
        include_runtime_hooks: bool = False,
    ):
        if include_sqlite_state:
            dashboard_module._sqlite_cache_enabled = self._sqlite_cache_enabled()
            dashboard_module._sqlite_cache_error = self._sqlite_cache_error()
            dashboard_module._sqlite_connect = self._sqlite_connect
            dashboard_module._sqlite_cache_put = self._sqlite_cache_put

        if include_runtime_hooks:
            dashboard_module._collect_official_premium_payload = (
                self._collect_official_premium_payload
            )
            dashboard_module._get_official_premium_payload = self._get_official_premium_payload
            dashboard_module._notify_dashboard_stream_listeners = (
                self._notify_dashboard_stream_listeners
            )
            dashboard_module._sqlite_cache_put = self._sqlite_cache_put
            dashboard_module._utc_now = self._utc_now
            dashboard_module._utc_now_iso = self._utc_now_iso
            resolved_thread_class = self._thread_class
            if callable(resolved_thread_class) and not isinstance(resolved_thread_class, type):
                resolved_thread_class = resolved_thread_class()
            dashboard_module.Thread = resolved_thread_class

    def _apply_usage_tracking_runtime(self):
        if self._usage_log_lock is not None:
            usage_tracking_module._usage_log_lock = self._usage_log_lock()
        if self._recent_usage_events is not None:
            usage_tracking_module._recent_usage_events = self._recent_usage_events()
        if self._archived_usage_events is not None:
            usage_tracking_module._archived_usage_events = self._archived_usage_events()
        if self._session_request_id_lock is not None:
            usage_tracking_module._session_request_id_lock = self._session_request_id_lock()
        if self._latest_server_request_ids_by_chain is not None:
            usage_tracking_module._latest_server_request_ids_by_chain = (
                self._latest_server_request_ids_by_chain()
            )
        if self._active_server_request_ids_by_request is not None:
            usage_tracking_module._active_server_request_ids_by_request = (
                self._active_server_request_ids_by_request()
            )
        if self._latest_claude_user_session_contexts is not None:
            usage_tracking_module._latest_claude_user_session_contexts = (
                self._latest_claude_user_session_contexts()
            )
        usage_tracking_module._record_usage_event = self._record_usage_event
        usage_tracking_module._rewrite_usage_log_locked = self._rewrite_usage_log_locked
        usage_tracking_module._snapshot_usage_events = self._snapshot_usage_events
        usage_tracking_module._snapshot_all_usage_events = self._snapshot_all_usage_events

    def codex_proxy_status(self):
        self._apply_auth_settings()
        return auth_module._codex_proxy_status()

    def claude_proxy_status(self):
        self._apply_auth_settings()
        return auth_module._claude_proxy_status()

    def write_codex_proxy_config(self):
        self._apply_auth_settings()
        return auth_module._write_codex_proxy_config()

    def write_claude_proxy_settings(self):
        self._apply_auth_settings()
        return auth_module._write_claude_proxy_settings()

    def disable_codex_proxy_config(self):
        self._apply_auth_settings()
        return auth_module._disable_codex_proxy_config()

    def disable_claude_proxy_settings(self):
        self._apply_auth_settings()
        return auth_module._disable_claude_proxy_settings()

    def proxy_client_status_payload(self):
        self._apply_auth_settings()
        return auth_module._proxy_client_status_payload()

    def load_archived_usage_history(self):
        self._apply_dashboard_runtime(include_sqlite_state=True)
        return usage_tracking_module._load_archived_usage_history()

    def load_usage_history(self):
        self._apply_dashboard_runtime(include_sqlite_state=True)
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._load_usage_history()

    def compact_usage_history_if_needed(self):
        self._apply_dashboard_runtime(include_sqlite_state=True)
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._compact_usage_history_if_needed()

    def finish_usage_event(self, *args, **kwargs):
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._finish_usage_event(*args, **kwargs)

    def seed_cached_payloads_from_sqlite(self):
        self._apply_dashboard_runtime(include_sqlite_state=True)
        return dashboard_module._seed_cached_payloads_from_sqlite()

    def trigger_official_premium_refresh(self, force: bool = False):
        self._apply_dashboard_runtime(include_runtime_hooks=True)
        return dashboard_module._trigger_official_premium_refresh(force=force)

    def build_dashboard_payload(self, force_refresh: bool = False) -> dict:
        self._apply_auth_settings()
        self._apply_dashboard_runtime(include_runtime_hooks=True)
        self._apply_usage_tracking_runtime()
        return dashboard_module._build_dashboard_payload(force_refresh=force_refresh)
