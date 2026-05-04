"""AES-GCM helpers for encrypted prompt trace payloads."""

from __future__ import annotations

import base64
import json
import os
from typing import Any


ENCRYPTED_PAYLOAD_MARKER = "ghcp_proxy.aesgcm.v1"
PASSWORD_VERIFIER_PLAINTEXT = "ghcp_proxy_trace_prompt_password_v1"
DEFAULT_KDF_ITERATIONS = 200_000


def _crypto_primitives():
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RuntimeError(
            "Encrypted prompt tracing requires the 'cryptography' package. "
            "Install project requirements before enabling it."
        ) from exc
    return AESGCM, PBKDF2HMAC, hashes


def new_salt() -> str:
    return base64.b64encode(os.urandom(16)).decode("ascii")


def derive_key(password: str, salt_b64: str, *, iterations: int = DEFAULT_KDF_ITERATIONS) -> bytes:
    if not isinstance(password, str) or not password:
        raise ValueError("AES password is required.")
    if not isinstance(salt_b64, str) or not salt_b64:
        raise ValueError("AES password salt is required.")
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("AES password salt is invalid.") from exc
    AESGCM, PBKDF2HMAC, hashes = _crypto_primitives()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode("utf-8"))


def is_encrypted_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("_encrypted") == ENCRYPTED_PAYLOAD_MARKER


def locked_payload() -> dict[str, Any]:
    return {
        "_encrypted": ENCRYPTED_PAYLOAD_MARKER,
        "locked": True,
        "reason": "password_required",
    }


def encrypt_payload(value: Any, key: bytes, salt_b64: str, *, iterations: int = DEFAULT_KDF_ITERATIONS) -> dict[str, Any]:
    if is_encrypted_payload(value):
        return value
    AESGCM, _PBKDF2HMAC, _hashes = _crypto_primitives()
    nonce = os.urandom(12)
    plaintext = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return {
        "_encrypted": ENCRYPTED_PAYLOAD_MARKER,
        "cipher": "AES-256-GCM",
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": iterations,
        "salt": salt_b64,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_payload(value: Any, password: str | None = None, *, key: bytes | None = None) -> Any:
    if not is_encrypted_payload(value):
        return value
    if value.get("locked") is True:
        raise ValueError("Encrypted prompt payload is locked because no AES password was loaded.")
    if key is None:
        key = derive_key(password or "", str(value.get("salt") or ""), iterations=int(value.get("iterations") or DEFAULT_KDF_ITERATIONS))
    try:
        nonce = base64.b64decode(str(value.get("nonce") or "").encode("ascii"), validate=True)
        ciphertext = base64.b64decode(str(value.get("ciphertext") or "").encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("Encrypted prompt payload is malformed.") from exc
    AESGCM, _PBKDF2HMAC, _hashes = _crypto_primitives()
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))


def make_password_verifier(key: bytes, salt_b64: str) -> dict[str, Any]:
    return encrypt_payload(PASSWORD_VERIFIER_PLAINTEXT, key, salt_b64)


def verify_password_payload(verifier: Any, key: bytes) -> bool:
    try:
        return decrypt_payload(verifier, key=key) == PASSWORD_VERIFIER_PLAINTEXT
    except Exception:
        return False
