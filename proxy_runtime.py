"""Central runtime wiring for proxy auth compatibility."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import auth as auth_module


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


class ProxyRuntime:
    """Applies proxy settings to the extracted auth module on demand."""

    def __init__(
        self,
        *,
        settings_provider: Callable[[], ProxySettings],
        auth_runtime: AuthRuntimeBindings,
    ):
        self._settings_provider = settings_provider
        self._auth_runtime = auth_runtime

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
