"""Rate limiting for upstream GitHub Copilot API requests."""

import asyncio
import time
from collections import deque

import httpx

from constants import UPSTREAM_REQUESTS_PER_WINDOW, UPSTREAM_REQUEST_WINDOW_SECONDS

_upstream_rate_limit_lock = asyncio.Lock()
_upstream_request_timestamps = deque()
_TRANSIENT_UPSTREAM_RETRY_ATTEMPTS = 2
_TRANSIENT_UPSTREAM_RETRY_DELAY_SECONDS = 0.1

# These errors mean that httpx did not produce a response.  In particular,
# RemoteProtocolError is commonly raised when the upstream/load balancer has
# closed the pinned keep-alive HTTP/2 connection between requests.  Retrying
# once lets httpx discard that connection and establish a fresh one instead of
# surfacing a transient 502 to the client.
_TRANSIENT_UPSTREAM_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.CloseError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


def _prune_upstream_request_timestamps(now: float):
    cutoff = now - UPSTREAM_REQUEST_WINDOW_SECONDS
    while _upstream_request_timestamps and _upstream_request_timestamps[0] <= cutoff:
        _upstream_request_timestamps.popleft()


async def throttle_upstream_request(now: float | None = None):
    while True:
        async with _upstream_rate_limit_lock:
            current_time = time.monotonic() if now is None else now
            _prune_upstream_request_timestamps(current_time)

            if len(_upstream_request_timestamps) < UPSTREAM_REQUESTS_PER_WINDOW:
                _upstream_request_timestamps.append(current_time)
                return

            oldest = _upstream_request_timestamps[0]
            delay = max(0.0, (oldest + UPSTREAM_REQUEST_WINDOW_SECONDS) - current_time)

        await asyncio.sleep(delay)


async def _call_with_transient_retry(call):
    for attempt in range(_TRANSIENT_UPSTREAM_RETRY_ATTEMPTS):
        await throttle_upstream_request()
        try:
            return await call()
        except _TRANSIENT_UPSTREAM_ERRORS:
            if attempt + 1 >= _TRANSIENT_UPSTREAM_RETRY_ATTEMPTS:
                raise
            await asyncio.sleep(_TRANSIENT_UPSTREAM_RETRY_DELAY_SECONDS)


async def throttled_client_post(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    return await _call_with_transient_retry(lambda: client.post(url, **kwargs))


async def throttled_client_send(client: httpx.AsyncClient, request: httpx.Request, **kwargs) -> httpx.Response:
    return await _call_with_transient_retry(lambda: client.send(request, **kwargs))
