"""Read the cached ChatGPT Excel WebView session without observing traffic."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import urllib.parse


_STORAGE_KEY = "bps_auth_tokens"
_MACOS_WEBSITE_DATA = Path(
    os.environ.get(
        "GHCP_EXCEL_WEBKIT_WEBSITE_DATA_DIR",
        str(
            Path.home()
            / "Library/Containers/com.microsoft.Excel/Data/Library/WebKit/WebsiteData"
        ),
    )
).expanduser()
_WINDOWS_WEBVIEW_ROOT = Path(
    os.environ.get(
        "GHCP_EXCEL_WEBVIEW2_DATA_DIR",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Office"),
    )
).expanduser()
_MACOS_REFRESH_LOCK = threading.Lock()
_WINDOWS_REFRESH_LOCK = threading.Lock()
_LEVELDB_TABLE_MAGIC = 0xDB4775248B80FB57


def _jwt_payload(token: str) -> dict[str, object]:
    try:
        encoded = token.split(".", 2)[1]
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(decoded)
    except (IndexError, ValueError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cached_session_headers(payload: dict[str, object]) -> tuple[dict[str, str], float]:
    session_info = payload.get("sessionInfo")
    user_info = payload.get("userInfo")
    if not isinstance(session_info, dict) or not isinstance(user_info, dict):
        raise ValueError("the stored ChatGPT session is missing session or user data")
    access_token = session_info.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("the stored ChatGPT session has no access token")
    claims = _jwt_payload(access_token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if not isinstance(auth_claims, dict):
        auth_claims = {}
    account_id = user_info.get("chatgpt_account_id") or auth_claims.get(
        "chatgpt_account_id"
    )
    account_user_id = user_info.get("chatgpt_account_user_id") or auth_claims.get(
        "chatgpt_account_user_id"
    )
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("the stored ChatGPT session has no account ID")
    headers = {
        "authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "x-openai-account-id": account_id,
    }
    if isinstance(account_user_id, str) and account_user_id:
        headers["x-openai-account-user-id"] = account_user_id
    auth_mode = payload.get("authMode")
    if isinstance(auth_mode, str) and auth_mode:
        headers["x-basispoints-auth-mode"] = auth_mode
    expires_at = session_info.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = claims.get("exp")
    return headers, float(expires_at) if isinstance(expires_at, (int, float)) else 0.0


def _macos_database_paths(website_data: Path) -> list[Path]:
    return sorted(
        website_data.glob("Default/*/*/LocalStorage/localstorage.sqlite3"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def load_macos_excel_session(website_data: Path | None = None) -> dict[str, str]:
    root = website_data or _MACOS_WEBSITE_DATA
    candidates: list[tuple[float, dict[str, str]]] = []
    errors: list[str] = []
    for database in _macos_database_paths(root):
        try:
            uri = f"file:{urllib.parse.quote(str(database))}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=1)
            try:
                row = connection.execute(
                    "SELECT value FROM ItemTable WHERE key = ?", (_STORAGE_KEY,)
                ).fetchone()
            finally:
                connection.close()
            if row is None:
                continue
            value = row[0]
            text = value.decode("utf-16-le") if isinstance(value, bytes) else value
            if not isinstance(text, str):
                raise ValueError("the WebKit LocalStorage value has an unsupported type")
            headers, expires_at = _cached_session_headers(json.loads(text))
            candidates.append((expires_at, headers))
        except (OSError, sqlite3.Error, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{database}: {exc}")
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if errors:
        raise RuntimeError(errors[0])
    raise RuntimeError(
        "No signed-in ChatGPT Excel session was found. Open the ChatGPT task pane "
        "in Excel and sign in."
    )


def _decode_leveldb_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift <= 63:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid LevelDB varint")


def _decompress_snappy(data: bytes) -> bytes:
    expected_size, offset = _decode_leveldb_varint(data, 0)
    output = bytearray()
    while offset < len(data):
        tag = data[offset]
        offset += 1
        kind = tag & 0x03
        if kind == 0:
            length = tag >> 2
            if length < 60:
                length += 1
            else:
                width = length - 59
                if offset + width > len(data):
                    raise ValueError("truncated Snappy literal length")
                length = int.from_bytes(data[offset : offset + width], "little") + 1
                offset += width
            if offset + length > len(data):
                raise ValueError("truncated Snappy literal")
            output.extend(data[offset : offset + length])
            offset += length
            continue
        length = 4 + ((tag >> 2) & 0x07) if kind == 1 else 1 + (tag >> 2)
        if kind == 1:
            if offset >= len(data):
                raise ValueError("truncated Snappy copy")
            distance = ((tag & 0xE0) << 3) | data[offset]
            offset += 1
        else:
            width = 2 if kind == 2 else 4
            if offset + width > len(data):
                raise ValueError("truncated Snappy copy distance")
            distance = int.from_bytes(data[offset : offset + width], "little")
            offset += width
        if distance <= 0 or distance > len(output):
            raise ValueError("invalid Snappy copy distance")
        for _ in range(length):
            output.append(output[-distance])
    if len(output) != expected_size:
        raise ValueError("invalid Snappy output length")
    return bytes(output)


def _leveldb_block_entries(data: bytes):
    if len(data) < 4:
        raise ValueError("truncated LevelDB block")
    restart_count = int.from_bytes(data[-4:], "little")
    entries_end = len(data) - 4 - restart_count * 4
    if entries_end < 0:
        raise ValueError("invalid LevelDB restart array")
    offset = 0
    prior_key = b""
    while offset < entries_end:
        shared, offset = _decode_leveldb_varint(data, offset)
        unshared, offset = _decode_leveldb_varint(data, offset)
        value_length, offset = _decode_leveldb_varint(data, offset)
        if shared > len(prior_key) or offset + unshared + value_length > entries_end:
            raise ValueError("invalid LevelDB entry")
        key = prior_key[:shared] + data[offset : offset + unshared]
        offset += unshared
        value = data[offset : offset + value_length]
        offset += value_length
        prior_key = key
        yield key, value


def _leveldb_table_entries(path: Path):
    data = path.read_bytes()
    if len(data) < 48 or int.from_bytes(data[-8:], "little") != _LEVELDB_TABLE_MAGIC:
        raise ValueError("not a LevelDB table")
    footer = data[-48:-8]
    _, offset = _decode_leveldb_varint(footer, 0)
    _, offset = _decode_leveldb_varint(footer, offset)
    index_offset, offset = _decode_leveldb_varint(footer, offset)
    index_size, _ = _decode_leveldb_varint(footer, offset)

    def read_block(block_offset: int, block_size: int) -> bytes:
        trailer_offset = block_offset + block_size
        if block_offset < 0 or trailer_offset >= len(data):
            raise ValueError("invalid LevelDB block handle")
        compressed = data[block_offset:trailer_offset]
        compression_type = data[trailer_offset]
        if compression_type == 0:
            return compressed
        if compression_type == 1:
            return _decompress_snappy(compressed)
        raise ValueError("unsupported LevelDB compression")

    for _, encoded_handle in _leveldb_block_entries(read_block(index_offset, index_size)):
        block_offset, handle_offset = _decode_leveldb_varint(encoded_handle, 0)
        block_size, _ = _decode_leveldb_varint(encoded_handle, handle_offset)
        yield from _leveldb_block_entries(read_block(block_offset, block_size))


def _windows_leveldb_paths(webview_root: Path) -> list[Path]:
    return sorted(
        webview_root.glob("**/EBWebView/Default/Local Storage/leveldb"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def _decode_windows_storage_value(value: bytes) -> dict[str, object]:
    text = value.decode("utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError("the WebView2 LocalStorage value is not JSON")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise ValueError("the WebView2 LocalStorage token payload is not an object")
    return payload


def load_windows_excel_session(webview_root: Path | None = None) -> dict[str, str]:
    root = webview_root or _WINDOWS_WEBVIEW_ROOT
    candidates: list[tuple[float, float, dict[str, str]]] = []
    errors: list[str] = []
    for database in _windows_leveldb_paths(root):
        for table in sorted(database.glob("*.ldb"), reverse=True):
            try:
                for key, value in _leveldb_table_entries(table):
                    if _STORAGE_KEY.encode("utf-8") not in key:
                        continue
                    headers, expires_at = _cached_session_headers(
                        _decode_windows_storage_value(value)
                    )
                    candidates.append((table.stat().st_mtime, expires_at, headers))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{table}: {exc}")
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    if errors:
        raise RuntimeError(errors[0])
    raise RuntimeError(
        "No signed-in ChatGPT Excel session was found in the Windows WebView2 cache. "
        "Open the ChatGPT task pane in Excel and sign in."
    )


def refresh_macos_excel_session(session_store, *, force: bool = False, website_data: Path | None = None) -> dict[str, object]:
    if sys.platform != "darwin":
        return session_store.status()
    status = session_store.status()
    if not force and status.get("configured") and not status.get("expired"):
        return status
    with _MACOS_REFRESH_LOCK:
        status = session_store.status()
        if not force and status.get("configured") and not status.get("expired"):
            return status
        try:
            headers = load_macos_excel_session(website_data)
            return session_store.configure(headers, persist=False, allow_expired=True)
        except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError):
            return session_store.status()


def refresh_windows_excel_session(session_store, *, force: bool = False, webview_root: Path | None = None) -> dict[str, object]:
    if sys.platform != "win32":
        return session_store.status()
    status = session_store.status()
    if not force and status.get("configured") and not status.get("expired"):
        return status
    with _WINDOWS_REFRESH_LOCK:
        status = session_store.status()
        if not force and status.get("configured") and not status.get("expired"):
            return status
        try:
            headers = load_windows_excel_session(webview_root)
            return session_store.configure(headers, persist=True, allow_expired=True)
        except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError):
            return session_store.status()


def cached_session_reader_status() -> dict[str, object]:
    if sys.platform == "win32":
        return {
            "available": bool(_windows_leveldb_paths(_WINDOWS_WEBVIEW_ROOT)),
            "method": "webview2-localstorage-leveldb",
            "platform": sys.platform,
            "process_running": False,
            "capturing": False,
            "error": "",
            "message": "",
        }
    if sys.platform == "darwin":
        return {
            "available": bool(_macos_database_paths(_MACOS_WEBSITE_DATA)),
            "method": "webkit-localstorage-sqlite",
            "platform": sys.platform,
            "process_running": False,
            "capturing": False,
            "error": "",
            "message": "",
        }
    return {
        "available": False,
        "method": "unavailable",
        "platform": sys.platform,
        "process_running": False,
        "capturing": False,
        "error": "",
        "message": "",
    }
