"""Rate limiting for upstream GitHub Copilot API requests."""

import asyncio
import time
from collections import deque

import httpx

from constants import UPSTREAM_REQUESTS_PER_WINDOW, UPSTREAM_REQUEST_WINDOW_SECONDS

_upstream_rate_limit_lock = asyncio.Lock()
_upstream_request_timestamps = deque()


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


async def throttled_client_post(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    await throttle_upstream_request()
    return await client.post(url, **kwargs)


async def throttled_client_send(client: httpx.AsyncClient, request: httpx.Request, **kwargs) -> httpx.Response:
    await throttle_upstream_request()
    return await client.send(request, **kwargs)
