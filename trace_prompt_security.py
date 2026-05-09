"""Encryption helpers for prompt trace payloads."""

from __future__ import annotations

import base64
import json
import os
import threading
from collections import OrderedDict
from typing import Any


ENCRYPTED_PAYLOAD_MARKER = "ghcp_proxy.aesgcm.v1"
ENVELOPE_PAYLOAD_MARKER = "ghcp_proxy.envelope.v1"
PASSWORD_VERIFIER_PLAINTEXT = "ghcp_proxy_trace_prompt_password_v1"
DEFAULT_KDF_ITERATIONS = 200_000

# RSA PEM parsing dominates envelope encrypt/decrypt cost (~5-10ms per parse for
# 3072-bit keys). The dashboard refreshes up to 5000 events per build and the AI
# hot path encrypts a prompt preview per request, so we memoize the parsed key
# objects keyed by the raw PEM bytes. Caches stay tiny in practice (one entry
# per active key) and rotate naturally when the PEM changes.
_PRIVATE_KEY_PARSE_CACHE: "OrderedDict[bytes, Any]" = OrderedDict()
_PRIVATE_KEY_PARSE_CACHE_LIMIT = 4
_PRIVATE_KEY_PARSE_CACHE_LOCK = threading.Lock()

_PUBLIC_KEY_PARSE_CACHE: "OrderedDict[bytes, tuple[Any, bytes]]" = OrderedDict()
_PUBLIC_KEY_PARSE_CACHE_LIMIT = 4
_PUBLIC_KEY_PARSE_CACHE_LOCK = threading.Lock()

# Decrypted envelope plaintexts are pure functions of their ciphertext, so a
# bounded LRU lets repeated dashboard refreshes skip RSA-OAEP entirely on cache
# hits. The cap is sized comfortably above DETAILED_REQUEST_HISTORY_LIMIT.
_ENVELOPE_DECRYPT_CACHE: "OrderedDict[tuple[str, str, str], Any]" = OrderedDict()
_ENVELOPE_DECRYPT_CACHE_LIMIT = 8192
_ENVELOPE_DECRYPT_CACHE_LOCK = threading.Lock()


def _load_private_key_cached(private_key_data: bytes):
    _AESGCM, _hashes, _padding, _rsa, serialization = _public_key_primitives()
    with _PRIVATE_KEY_PARSE_CACHE_LOCK:
        cached = _PRIVATE_KEY_PARSE_CACHE.get(private_key_data)
        if cached is not None:
            _PRIVATE_KEY_PARSE_CACHE.move_to_end(private_key_data)
            return cached
    parsed = serialization.load_pem_private_key(private_key_data, password=None)
    with _PRIVATE_KEY_PARSE_CACHE_LOCK:
        _PRIVATE_KEY_PARSE_CACHE[private_key_data] = parsed
        _PRIVATE_KEY_PARSE_CACHE.move_to_end(private_key_data)
        while len(_PRIVATE_KEY_PARSE_CACHE) > _PRIVATE_KEY_PARSE_CACHE_LIMIT:
            _PRIVATE_KEY_PARSE_CACHE.popitem(last=False)
    return parsed


def _load_public_key_cached(public_key_data: bytes) -> tuple[Any, bytes]:
    """Return (parsed_public_key, der_bytes) for a PEM-encoded public key."""
    _AESGCM, _hashes, _padding, _rsa, serialization = _public_key_primitives()
    with _PUBLIC_KEY_PARSE_CACHE_LOCK:
        cached = _PUBLIC_KEY_PARSE_CACHE.get(public_key_data)
        if cached is not None:
            _PUBLIC_KEY_PARSE_CACHE.move_to_end(public_key_data)
            return cached
    parsed = serialization.load_pem_public_key(public_key_data)
    der_bytes = parsed.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    entry = (parsed, der_bytes)
    with _PUBLIC_KEY_PARSE_CACHE_LOCK:
        _PUBLIC_KEY_PARSE_CACHE[public_key_data] = entry
        _PUBLIC_KEY_PARSE_CACHE.move_to_end(public_key_data)
        while len(_PUBLIC_KEY_PARSE_CACHE) > _PUBLIC_KEY_PARSE_CACHE_LIMIT:
            _PUBLIC_KEY_PARSE_CACHE.popitem(last=False)
    return entry


def _envelope_cache_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("encrypted_key") or ""),
        str(value.get("nonce") or ""),
        str(value.get("ciphertext") or ""),
    )


def _envelope_decrypt_cache_get(cache_key: tuple[str, str, str]) -> Any:
    with _ENVELOPE_DECRYPT_CACHE_LOCK:
        cached = _ENVELOPE_DECRYPT_CACHE.get(cache_key)
        if cached is not None:
            _ENVELOPE_DECRYPT_CACHE.move_to_end(cache_key)
        return cached


def _envelope_decrypt_cache_set(cache_key: tuple[str, str, str], plaintext: Any) -> None:
    with _ENVELOPE_DECRYPT_CACHE_LOCK:
        _ENVELOPE_DECRYPT_CACHE[cache_key] = plaintext
        _ENVELOPE_DECRYPT_CACHE.move_to_end(cache_key)
        while len(_ENVELOPE_DECRYPT_CACHE) > _ENVELOPE_DECRYPT_CACHE_LIMIT:
            _ENVELOPE_DECRYPT_CACHE.popitem(last=False)


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


def _public_key_primitives():
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RuntimeError(
            "Public-key prompt trace encryption requires the 'cryptography' package. "
            "Install project requirements before enabling it."
        ) from exc
    return AESGCM, hashes, padding, rsa, serialization


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
    return isinstance(value, dict) and value.get("_encrypted") in {
        ENCRYPTED_PAYLOAD_MARKER,
        ENVELOPE_PAYLOAD_MARKER,
    }


def is_envelope_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("_encrypted") == ENVELOPE_PAYLOAD_MARKER


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


def generate_envelope_key_pair() -> dict[str, str]:
    _AESGCM, _hashes, _padding, rsa, serialization = _public_key_primitives()
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    return {"public_key": public_pem, "private_key": private_pem}


def public_key_fingerprint(public_key_pem: str) -> str:
    if not isinstance(public_key_pem, str) or not public_key_pem.strip():
        raise ValueError("Public key PEM is required.")
    _AESGCM, hashes, _padding, _rsa, _serialization = _public_key_primitives()
    _public_key, public_der = _load_public_key_cached(public_key_pem.encode("utf-8"))
    digest = hashes.Hash(hashes.SHA256())
    digest.update(public_der)
    return base64.urlsafe_b64encode(digest.finalize()).decode("ascii").rstrip("=")


def encrypt_payload_with_public_key(value: Any, public_key_pem: str) -> dict[str, Any]:
    if is_encrypted_payload(value):
        return value
    AESGCM, hashes, padding, _rsa, _serialization = _public_key_primitives()
    public_key_data = public_key_pem.encode("utf-8")
    public_key, public_der = _load_public_key_cached(public_key_data)
    content_key = os.urandom(32)
    nonce = os.urandom(12)
    plaintext = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")
    ciphertext = AESGCM(content_key).encrypt(nonce, plaintext, None)
    encrypted_key = public_key.encrypt(
        content_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    fingerprint_digest = hashes.Hash(hashes.SHA256())
    fingerprint_digest.update(public_der)
    public_key_sha256 = base64.urlsafe_b64encode(fingerprint_digest.finalize()).decode("ascii").rstrip("=")
    return {
        "_encrypted": ENVELOPE_PAYLOAD_MARKER,
        "cipher": "AES-256-GCM",
        "key_cipher": "RSA-OAEP-SHA256",
        "public_key_sha256": public_key_sha256,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "encrypted_key": base64.b64encode(encrypted_key).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_payload_with_private_key(value: Any, private_key_pem: str | bytes) -> Any:
    if not is_envelope_payload(value):
        return value
    if not private_key_pem:
        raise ValueError("Private key PEM is required.")
    cache_key = _envelope_cache_key(value)
    if all(cache_key):
        cached_plaintext = _envelope_decrypt_cache_get(cache_key)
        if cached_plaintext is not None:
            return cached_plaintext
    AESGCM, hashes, padding, _rsa, _serialization = _public_key_primitives()
    private_key_data = private_key_pem if isinstance(private_key_pem, bytes) else private_key_pem.encode("utf-8")
    private_key = _load_private_key_cached(private_key_data)
    try:
        encrypted_key = base64.b64decode(cache_key[0].encode("ascii"), validate=True)
        nonce = base64.b64decode(cache_key[1].encode("ascii"), validate=True)
        ciphertext = base64.b64decode(cache_key[2].encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("Encrypted prompt envelope is malformed.") from exc
    content_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    plaintext = AESGCM(content_key).decrypt(nonce, ciphertext, None)
    decoded = json.loads(plaintext.decode("utf-8"))
    if all(cache_key):
        _envelope_decrypt_cache_set(cache_key, decoded)
    return decoded


def decrypt_payload(
    value: Any,
    password: str | None = None,
    *,
    key: bytes | None = None,
    private_key_pem: str | bytes | None = None,
) -> Any:
    if not is_encrypted_payload(value):
        return value
    if is_envelope_payload(value):
        return decrypt_payload_with_private_key(value, private_key_pem or b"")
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
