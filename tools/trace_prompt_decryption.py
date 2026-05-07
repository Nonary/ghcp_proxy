"""Optional decryption support for encrypted prompt trace payloads."""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from typing import Any


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import trace_prompt_security
from constants import CLIENT_PROXY_SETTINGS_FILE


PASSWORD_ENV_NAME = "GHCP_TRACE_PROMPT_PASSWORD"

_LOADED_ENV_FILES: set[str] = set()


@dataclass
class DecryptionStats:
    encrypted: int = 0
    decrypted: int = 0
    missing_secret: int = 0
    locked: int = 0
    failed: int = 0


class TracePromptDecryptor:
    def __init__(self, *, password: str | None = None, private_key_pem: str | None = None):
        self.password = password
        self.private_key_pem = private_key_pem
        self.stats = DecryptionStats()

    @classmethod
    def from_environment(cls) -> "TracePromptDecryptor":
        load_dotenv_files()
        password = os.environ.get(PASSWORD_ENV_NAME) or None
        return cls(password=password, private_key_pem=_load_private_key_from_settings(password))

    def decrypt(self, value: Any) -> Any:
        if not trace_prompt_security.is_encrypted_payload(value):
            return value
        self.stats.encrypted += 1
        if value.get("locked") is True:
            self.stats.locked += 1
            return value
        if trace_prompt_security.is_envelope_payload(value):
            if not self.private_key_pem:
                self.stats.missing_secret += 1
                return value
        elif not self.password:
            self.stats.missing_secret += 1
            return value
        try:
            decrypted = trace_prompt_security.decrypt_payload(
                value,
                password=self.password,
                private_key_pem=self.private_key_pem,
            )
        except Exception:
            self.stats.failed += 1
            return value
        self.stats.decrypted += 1
        return decrypted

    def status_line(self) -> str | None:
        stats = self.stats
        if not stats.encrypted:
            return None
        secret = PASSWORD_ENV_NAME if self.password else "no env secret"
        return (
            f"Prompt trace decryption: encrypted={stats.encrypted} decrypted={stats.decrypted} "
            f"locked={stats.locked} missing_secret={stats.missing_secret} failed={stats.failed} "
            f"source={secret}"
        )


def load_dotenv_files(paths: list[str] | None = None) -> None:
    candidates: list[str] = []
    if paths:
        candidates.extend(paths)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    candidates.append(os.path.join(REPO_ROOT, ".env"))

    seen: set[str] = set()
    for candidate in candidates:
        path = os.path.abspath(os.path.expanduser(candidate))
        if path in seen:
            continue
        seen.add(path)
        if path in _LOADED_ENV_FILES or not os.path.isfile(path):
            continue
        _load_dotenv_file(path)
        _LOADED_ENV_FILES.add(path)


def _load_private_key_from_settings(password: str | None) -> str | None:
    if not password:
        return None
    settings_path = os.path.abspath(os.path.expanduser(os.environ.get("GHCP_CLIENT_PROXY_SETTINGS_FILE", CLIENT_PROXY_SETTINGS_FILE)))
    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(settings, dict):
        return None
    salt = settings.get("trace_prompt_logging_salt")
    verifier = settings.get("trace_prompt_logging_verifier")
    private_key_payload = settings.get("trace_prompt_logging_private_key")
    if not isinstance(salt, str) or not salt or not isinstance(verifier, dict) or not isinstance(private_key_payload, dict):
        return None
    try:
        key = trace_prompt_security.derive_key(password, salt)
    except Exception:
        return None
    if not trace_prompt_security.verify_password_payload(verifier, key):
        return None
    try:
        private_key = trace_prompt_security.decrypt_payload(private_key_payload, key=key)
    except Exception:
        return None
    return private_key if isinstance(private_key, str) and private_key.strip() else None


def decrypt_mapping_fields(payload: dict[str, Any], fields: tuple[str, ...], decryptor: TracePromptDecryptor) -> dict[str, Any]:
    for field in fields:
        if field in payload:
            payload[field] = decryptor.decrypt(payload[field])
    return payload


def _load_dotenv_file(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for raw_line in lines:
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_env_value(value.strip())


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    if value and value[0] not in ("'", '"') and " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value
