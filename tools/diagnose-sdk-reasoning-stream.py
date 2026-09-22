#!/usr/bin/env python3
"""Show where Copilot SDK reasoning stalls on its way to Codex Desktop.

Runs one reasoning-heavy turn through the same Copilot SDK session options and
``_stream_turn`` translation the proxy uses, and timestamps every stage:

  upstream  model events the Copilot runtime receives (HTTP or WebSocket)
  sdk       session events the proxy consumes
  proxy     Responses SSE events the proxy sends to Codex
  desktop   the live thought heading Codex Desktop would display

By default the turn runs twice: once exactly as the SDK sends it, and once with
``stream_options.reasoning_summary_delivery=sequential_cutoff`` injected into the
upstream model request.  Codex sends that option to OpenAI so reasoning
summaries are generated while the model is still thinking; the SDK does not
forward it.  Comparing the runs shows whether Copilot streams summaries during
reasoning, and whether injecting the option changes that.

This does not talk to the running proxy; it imports the checkout's code.

Usage:
    ./.venv/bin/python tools/diagnose-sdk-reasoning-stream.py [--model gpt-5.6-luna]
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import certifi  # noqa: E402
import httpx  # noqa: E402
from copilot import CopilotClient  # noqa: E402
from copilot.copilot_request_handler import (  # noqa: E402
    CopilotRequestHandler,
    CopilotWebSocketForwarder,
)

import auth  # noqa: E402
import copilot_sdk_upstream as sdk  # noqa: E402

DEFAULT_PROMPT = (
    "Think this through carefully before answering. A 5x5 grid of lights starts "
    "all off. Pressing a light toggles it and its orthogonal neighbours. What is "
    "the minimum number of presses that turns every light on, and why? Keep the "
    "final answer to three sentences."
)
SEQUENTIAL = {"reasoning_summary_delivery": "sequential_cutoff"}


class Recorder:
    def __init__(self) -> None:
        self.t0: float | None = None
        self.rows: list[tuple[float, str, str, object, str]] = []
        self.requests: list[dict] = []

    def start(self) -> None:
        self.t0 = time.monotonic()

    def add(self, source: str, kind: str, index: object = None, text: str = "") -> None:
        if self.t0 is not None:
            self.rows.append((time.monotonic() - self.t0, source, kind, index, text or ""))


def _sse_events(buffer: str) -> tuple[list[dict], str]:
    events = []
    while "\n\n" in buffer:
        block, buffer = buffer.split("\n\n", 1)
        data = "".join(line[5:].lstrip() for line in block.splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, buffer


def _record_upstream(rec: Recorder, event: dict) -> None:
    kind = event.get("type")
    if not isinstance(kind, str):
        # Chat Completions chunk: classify by which delta fields carry text.
        choices = event.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        fields = sorted(k for k, v in delta.items() if isinstance(v, str) and v)
        if fields:
            rec.add("upstream", "chat.delta:" + "+".join(fields), text=next(
                (delta[k] for k in fields), ""))
        return
    if kind in {"response.output_item.added", "response.output_item.done"}:
        rec.add("upstream", kind, text=str((event.get("item") or {}).get("type", "")))
    elif kind == "content_block_delta":  # Anthropic Messages
        delta = event.get("delta") or {}
        rec.add("upstream", f"content_block_delta:{delta.get('type')}", event.get("index"),
                delta.get("thinking") or delta.get("text") or "")
    else:
        text = event.get("delta") if isinstance(event.get("delta"), str) else event.get("text")
        rec.add("upstream", kind, event.get("summary_index", event.get("content_index")),
                text if isinstance(text, str) else "")


class _TimedStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response, rec: Recorder) -> None:
        self._response = response
        self._rec = rec

    async def __aiter__(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""
        async for chunk in self._response.aiter_bytes():
            buffer = (buffer + decoder.decode(chunk)).replace("\r\n", "\n")
            events, buffer = _sse_events(buffer)
            for event in events:
                _record_upstream(self._rec, event)
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


class _TimedWebSocket(CopilotWebSocketForwarder):
    def __init__(self, ctx, handler: "TimingHandler") -> None:
        super().__init__(ctx)
        self._handler = handler

    async def send_request_message(self, data):
        try:
            message = json.loads(data)
        except (TypeError, ValueError):
            message = None
        if isinstance(message, dict) and message.get("type") == "response.create":
            body = message["response"] if isinstance(message.get("response"), dict) else message
            data = json.dumps(self._handler.prepare(body, "websocket", self.context.url, message)) \
                if self._handler.inject else data
        await super().send_request_message(data)

    async def send_response_message(self, data):
        try:
            event = json.loads(data)
        except (TypeError, ValueError):
            event = None
        if isinstance(event, dict):
            _record_upstream(self._handler.rec, event)
        await super().send_response_message(data)


class TimingHandler(CopilotRequestHandler):
    """Pass-through handler that timestamps model traffic and can inject options."""

    def __init__(self, rec: Recorder, inject: bool) -> None:
        self.rec = rec
        self.inject = inject

    def prepare(self, body: dict, transport: str, url: str, envelope: dict | None = None) -> dict:
        if self.inject and "input" in body:
            body["stream_options"] = {**(body.get("stream_options") or {}), **SEQUENTIAL}
        self.rec.requests.append({
            "transport": transport,
            "path": httpx.URL(url).path,
            "model": body.get("model"),
            "reasoning": body.get("reasoning"),
            "stream_options": body.get("stream_options"),
        })
        return envelope if envelope is not None else body

    async def send_request(self, request, ctx):
        if request.method != "POST" or self.rec.t0 is None:
            return await super().send_request(request, ctx)
        try:
            body = json.loads(await request.aread())
        except ValueError:
            body = None
        if not isinstance(body, dict) or "model" not in body:
            return await super().send_request(request, ctx)
        body = self.prepare(body, "http", str(request.url))
        if self.inject:
            headers = [(k, v) for k, v in request.headers.multi_items() if k.lower() != "content-length"]
            request = httpx.Request(request.method, request.url, headers=headers,
                                    content=json.dumps(body).encode())
        response = await super().send_request(request, ctx)
        self.rec.requests[-1]["status"] = response.status_code
        if "text/event-stream" not in response.headers.get("content-type", ""):
            return response
        headers = [(k, v) for k, v in response.headers.multi_items()
                   if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}]
        return httpx.Response(response.status_code, headers=headers,
                              stream=_TimedStream(response, self.rec), request=request)

    async def open_websocket(self, ctx):
        return _TimedWebSocket(ctx, self)


def _record_sdk(rec: Recorder, event) -> None:
    raw = getattr(event, "type", "")
    name = str(getattr(raw, "value", raw))
    data = getattr(event, "data", None)
    if name in {"assistant.reasoning_delta", "assistant.message_delta"}:
        rec.add("sdk", name, text=getattr(data, "delta_content", "") or "")
    elif name in {"assistant.reasoning", "assistant.message", "assistant.intent",
                  "external_tool.requested", "session.idle", "session.error"}:
        text = getattr(data, "content", None) or getattr(data, "intent", None) \
            or getattr(data, "message", None) or ""
        rec.add("sdk", name, text=str(text))


# Codex Desktop heading: the webview joins summary parts with a blank line
# (wrapping a non-bold first part in **) and shows the last non-empty line,
# unwrapping a line that is a single bold span.
def _desktop_heading(parts: list[str]) -> str | None:
    if not parts:
        return None
    first, rest = parts[0], parts[1:]
    if rest and not first.startswith("**"):
        first = f"**{first}**"
    joined = "\n\n".join([first, *rest]) if rest else first
    line = next((ln.strip() for ln in reversed(joined.rstrip().splitlines()) if ln.strip()), None)
    if line is None:
        return None
    bold = re.fullmatch(r"\*\*(.+?)\*\*", line)
    return bold.group(1).strip() if bold else line


class _Connected:
    async def is_disconnected(self) -> bool:
        return False


async def run_turn(args, inject: bool) -> Recorder:
    rec = Recorder()
    handler = TimingHandler(rec, inject)
    state_dir = tempfile.mkdtemp(prefix="ghcp-reasoning-diag-")
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    if args.provider_base_url:
        client = CopilotClient(use_logged_in_user=False, base_directory=state_dir,
                               log_level="error", mode="empty", request_handler=handler)
    else:
        # GITHUB_TOKEN allows a run without the proxy's saved login, e.g.
        # GITHUB_TOKEN=$(gh auth token).
        token = os.environ.get("GITHUB_TOKEN") or auth.load_access_token()
        client = CopilotClient(github_token=token, use_logged_in_user=token is None,
                               working_directory=str(REPO), base_directory=state_dir,
                               log_level="error", mode="empty", request_handler=handler)
    body = {
        "model": args.model,
        "stream": True,
        "reasoning": {"effort": args.effort, "summary": "detailed"},
        "stream_options": dict(SEQUENTIAL),
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": args.prompt}]}],
    }
    registration = sdk.ToolRegistration()
    session = None
    unsubscribe = None
    try:
        await client.start()
        # The proxy checks the Copilot model catalog; a custom provider has none.
        effort = args.effort if args.provider_base_url else \
            await sdk._reasoning_effort_for_client(body, client)
        options = sdk._session_options(body, registration, reasoning_effort=effort)
        if args.provider_base_url:
            options["provider"] = {"type": "openai", "wire_api": "responses",
                                   "base_url": args.provider_base_url,
                                   "api_key": os.environ.get("OPENAI_API_KEY", "unused")}
        session = await client.create_session(**options)
        unsubscribe = session.on(lambda event: _record_sdk(rec, event))

        async def dispatch() -> None:
            rec.start()
            await session.send(args.prompt)

        parts: list[str] = []
        reasoning_id = None
        async for chunk in sdk._stream_turn(_Connected(), body, session, dispatch, registration):
            events, _ = _sse_events(chunk.decode("utf-8", "replace"))
            for event in events:
                kind = event.get("type", "")
                if kind == "response.output_item.added" and event["item"]["type"] == "reasoning":
                    reasoning_id = event["item"]["id"]
                    rec.add("proxy", kind, text="reasoning")
                elif kind == "response.reasoning_summary_text.done":
                    rec.add("proxy", kind, event.get("summary_index"), event.get("text", ""))
                    if event.get("item_id") == reasoning_id:
                        index = event["summary_index"]
                        parts.extend([""] * (index + 1 - len(parts)))
                        parts[index] += event["text"]
                        rec.add("desktop", "heading", text=_desktop_heading(parts) or "")
                elif kind in {"response.output_item.added", "response.output_item.done"}:
                    rec.add("proxy", kind, text=event["item"]["type"])
                    if kind == "response.output_item.done" and event["item"]["id"] == reasoning_id:
                        reasoning_id = None
                elif kind in {"response.output_text.delta", "response.completed", "response.failed"}:
                    rec.add("proxy", kind, text=event.get("delta") or
                            json.dumps((event.get("response") or {}).get("error") or "")[:200])
    finally:
        if unsubscribe is not None:
            unsubscribe()
        if session is not None:
            try:
                await session.disconnect()
            except Exception:
                pass
        await client.stop()
        shutil.rmtree(state_dir, ignore_errors=True)
    return rec


REASONING_UPSTREAM = re.compile(
    r"reasoning_summary_text\.(delta|done)|reasoning_text\.delta|thinking_delta|chat\.delta:.*reasoning")
ANSWER_UPSTREAM = re.compile(
    r"output_text\.delta|chat\.delta:.*content|content_block_delta:text_delta"
    r"|function_call_arguments|custom_tool_call_input")


def _first(rows, source, pattern):
    return next((r[0] for r in rows if r[1] == source and re.search(pattern, r[2])), None)


def _last(rows, source, pattern):
    return next((r[0] for r in reversed(rows) if r[1] == source and re.search(pattern, r[2])), None)


def _fmt(value) -> str:
    return "   -  " if value is None else f"{value:6.2f}"


def report(label: str, rec: Recorder, *, timeline: bool) -> None:
    rows = sorted(rec.rows, key=lambda r: r[0])
    print(f"\n=== {label}")
    for request in rec.requests:
        print(f"    model request: {request}")
    if not rec.requests:
        print("    model request: none observed by the request handler")

    up_first = _first(rows, "upstream", REASONING_UPSTREAM.pattern)
    up_last = _last(rows, "upstream", REASONING_UPSTREAM.pattern)
    up_answer = _first(rows, "upstream", ANSWER_UPSTREAM.pattern)
    up_count = sum(1 for r in rows if r[1] == "upstream" and REASONING_UPSTREAM.search(r[2]))
    sdk_first = _first(rows, "sdk", r"assistant\.reasoning")
    sdk_last = _last(rows, "sdk", r"assistant\.reasoning_delta")
    sdk_count = sum(1 for r in rows if r[1] == "sdk" and r[2] == "assistant.reasoning_delta")
    proxy_first = _first(rows, "proxy", r"reasoning_summary_text\.done")
    proxy_count = sum(1 for r in rows if r[1] == "proxy" and r[2].endswith("summary_text.done"))
    answer = next((r[0] for r in rows if r[1] == "proxy" and (
        r[2] == "response.output_text.delta"
        or (r[2] == "response.output_item.added" and r[4] != "reasoning"))), None)

    print("    seconds after send:        first    last   count")
    print(f"    upstream reasoning text   {_fmt(up_first)}  {_fmt(up_last)}   {up_count}")
    print(f"    upstream answer/tool      {_fmt(up_answer)}")
    print(f"    sdk reasoning events      {_fmt(sdk_first)}  {_fmt(sdk_last)}   {sdk_count}")
    print(f"    proxy summary parts done  {_fmt(proxy_first)}            {proxy_count}")
    print(f"    proxy answer/tool start   {_fmt(answer)}")

    print("    Codex Desktop live headings:")
    headings = [(t, text) for t, source, _, _, text in rows if source == "desktop"]
    for t, text in headings:
        print(f"      {t:6.2f}s  {text[:100]!r}")
    if not headings:
        print("      (none)")

    # Copilot sends summary text in bursts; each finished burst should reach
    # the desktop promptly.
    bursts: list[list[float]] = []
    for t, source, kind, _, _ in rows:
        if source == "upstream" and REASONING_UPSTREAM.search(kind):
            if bursts and t - bursts[-1][1] <= 1.0:
                bursts[-1][1] = t
            else:
                bursts.append([t, t])
    print("    upstream reasoning bursts: " + (", ".join(
        f"{start:.2f}-{stop:.2f}s" for start, stop in bursts) or "none"))
    lag = max((next((t for t in (h[0] for h in headings) if t >= stop - 0.001), float("inf")) - stop
               for _, stop in bursts), default=0.0)

    failure = next((r[4] for r in rows if r[1] == "proxy" and r[2] == "response.failed"), None)
    end = up_answer if up_answer is not None else answer
    if failure is not None:
        verdict = f"The turn failed: {failure}"
    elif up_count == 0 and not rec.requests:
        verdict = "No model traffic was visible to the request handler; see the sdk rows."
    elif up_count == 0:
        verdict = ("Copilot sent no streamed reasoning text for this turn, so there is nothing "
                   "to show while the model thinks.")
    elif sdk_first is not None and up_first is not None and sdk_first - up_first > 1.0:
        verdict = f"The SDK runtime delayed reasoning by {sdk_first - up_first:.2f}s."
    elif lag > 2.0:
        verdict = (f"The proxy held finished reasoning text back for up to {lag:.2f}s "
                   "before the desktop could show it.")
    elif end is not None and up_first is not None and up_first > max(5.0, 0.25 * end):
        verdict = (f"Copilot sent its first reasoning text {up_first:.2f}s into a {end:.2f}s "
                   "thinking phase. The proxy cannot show thoughts before it receives them.")
    else:
        verdict = ("Reasoning streamed live end to end: every burst reached the desktop within "
                   f"{lag:.2f}s. If the app still shows nothing, confirm the running proxy was "
                   "restarted on this commit.")
    print(f"    verdict: {verdict}")

    if timeline:
        print("    timeline:")
        run = None
        for t, source, kind, index, text in rows + [(None, None, None, None, None)]:
            key = (source, kind)
            if run and run[0] == key and kind and kind.endswith(("delta", "reasoning_delta")):
                run[2] = t
                run[3] += 1
                continue
            if run:
                span = f"{run[1]:6.2f}" + (f"..{run[2]:6.2f}" if run[3] > 1 else "        ")
                count = f" x{run[3]}" if run[3] > 1 else ""
                print(f"      {span} {run[0][0]:8} {run[0][1]}{count} {run[4][:60]!r}")
            run = [key, t, t, 1, text] if source else None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--mode", choices=("both", "plain", "sequential"), default="both",
                        help="plain: upstream as the SDK sends it; sequential: inject "
                             "sequential_cutoff into the upstream model request")
    parser.add_argument("--timeline", action="store_true", help="print the full event timeline")
    parser.add_argument("--provider-base-url",
                        help="use an OpenAI-compatible Responses endpoint instead of Copilot")
    args = parser.parse_args()
    print(f"checkout: {REPO}  splitter present: {hasattr(sdk, '_ReasoningSummaryParts')}")
    if args.mode in ("both", "plain"):
        report("upstream as the SDK sends it", await run_turn(args, inject=False),
               timeline=args.timeline)
    if args.mode in ("both", "sequential"):
        report("upstream with sequential_cutoff injected", await run_turn(args, inject=True),
               timeline=args.timeline)


if __name__ == "__main__":
    asyncio.run(main())
