"""Central runtime wiring for the proxy's extracted auth and usage modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AuthRuntimeBindings:
    load_api_key_payload: Callable[[], dict]


@dataclass(frozen=True)
class UsageTrackingRuntimeBindings:
    state_provider: Callable[[], usage_tracking_module.UsageTrackingState]


class ProxyRuntime:
    """Owns the compatibility wiring between extracted modules."""

    def __init__(
        self,
        *,
        settings_provider: Callable[[], ProxySettings],
        auth_runtime: AuthRuntimeBindings,
        usage_tracking_runtime: UsageTrackingRuntimeBindings,
    ):
        self._settings_provider = settings_provider
        self._auth_runtime = auth_runtime
        self._usage_tracking_runtime = usage_tracking_runtime

    @property
    def dashboard_stream_version(self) -> int:
        return getattr(dashboard_module, "_dashboard_stream_version", 0)

    def initialize(self, initiator_policy: InitiatorPolicy):
        self.load_archived_usage_history()
        self.load_usage_history()
        self._apply_usage_tracking_runtime()
        initiator_policy.seed_from_usage_events(
            list(self._usage_tracking_runtime.state_provider().recent_usage_events)
        )

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
        auth_module.load_api_key_payload = self._auth_runtime.load_api_key_payload

    def _apply_usage_tracking_runtime(self):
        state = self._usage_tracking_runtime.state_provider()
        archive_store = usage_tracking_module.UsageArchiveStore(
            init_storage=dashboard_module._init_sqlite_cache,
            lock=dashboard_module._sqlite_cache_lock,
            connect=dashboard_module._sqlite_connect,
            mark_unavailable=dashboard_module._set_sqlite_cache_unavailable,
        )
        usage_tracking_module.configure_usage_tracking(
            state=state,
            archive_store=archive_store,
        )

    def codex_proxy_status(self):
        self._apply_auth_settings()
        return auth_module.codex_proxy_status()

    def claude_proxy_status(self):
        self._apply_auth_settings()
        return auth_module.claude_proxy_status()

    def write_codex_proxy_config(self):
        self._apply_auth_settings()
        return auth_module.write_codex_proxy_config()

    def write_claude_proxy_settings(self):
        self._apply_auth_settings()
        return auth_module.write_claude_proxy_settings()

    def disable_codex_proxy_config(self):
        self._apply_auth_settings()
        return auth_module.disable_codex_proxy_config()

    def disable_claude_proxy_settings(self):
        self._apply_auth_settings()
        return auth_module.disable_claude_proxy_settings()

    def proxy_client_status_payload(self):
        self._apply_auth_settings()
        return auth_module.proxy_client_status_payload()

    def load_archived_usage_history(self):
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._load_archived_usage_history()

    def load_usage_history(self):
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._load_usage_history()

    def compact_usage_history_if_needed(self):
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._compact_usage_history_if_needed()

    def finish_usage_event(self, *args, **kwargs):
        self._apply_usage_tracking_runtime()
        return usage_tracking_module._finish_usage_event(*args, **kwargs)
