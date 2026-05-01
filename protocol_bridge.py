"""Protocol bridge planning using pluggable strategy objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable

import effort_mapping
import format_translation
from initiator_policy import is_approval_agent_request
from model_routing_config import ModelRoutingConfigService, model_provider_family, normalize_routing_model_name


def _default_capability_resolver(model: str | None) -> bool:
    """Cache-only capability check.

    The bridge defaults to enabling native ``/v1/messages`` only when the
    upstream Copilot ``/models`` cache explicitly advertises support for
    this model. The offline-dev allowlist used by
    ``proxy.model_supports_native_messages`` is intentionally NOT consulted
    here, so unit tests and offline boots keep their existing chat-bridge
    behavior unless a caller injects a custom resolver.
    """
    if not model:
        return False
    try:
        from proxy import _COPILOT_MODEL_CAPS_CACHE  # lazy import to avoid cycles
    except Exception:
        return False
    data = _COPILOT_MODEL_CAPS_CACHE.get("data") if isinstance(_COPILOT_MODEL_CAPS_CACHE, dict) else None
    if not isinstance(data, dict) or not data:
        return False
    record = data.get(model)
    if not isinstance(record, dict):
        return False
    return bool(record.get("messages_endpoint_supported"))


def _resolve_reasoning_efforts(model: str | None) -> list[str] | None:
    """Look up the per-model ``reasoning_efforts`` whitelist from the Copilot
    ``/models`` capability cache. Returns ``None`` when unknown so the caller
    can skip the clamp entirely.
    """
    if not model:
        return None
    try:
        from proxy import _COPILOT_MODEL_CAPS_CACHE  # lazy import to avoid cycles
    except Exception:
        return None
    data = _COPILOT_MODEL_CAPS_CACHE.get("data") if isinstance(_COPILOT_MODEL_CAPS_CACHE, dict) else None
    if not isinstance(data, dict) or not data:
        return None
    record = data.get(model)
    if not isinstance(record, dict):
        return None
    efforts = record.get("reasoning_efforts")
    if isinstance(efforts, list) and efforts:
        return [str(e) for e in efforts if isinstance(e, str)]
    return None


def _default_adaptive_thinking_resolver(model: str | None) -> bool:
    """Best-effort check whether ``model`` supports adaptive thinking.

    Reads from the Copilot ``/models`` capability cache when populated.
    Returns ``False`` for unknown models; the messages-preprocess pipeline
    will then leave the request's ``thinking`` field untouched.
    """
    if not model:
        return False
    try:
        from proxy import _COPILOT_MODEL_CAPS_CACHE  # lazy import to avoid cycles
    except Exception:
        return False
    data = _COPILOT_MODEL_CAPS_CACHE.get("data") if isinstance(_COPILOT_MODEL_CAPS_CACHE, dict) else None
    if not isinstance(data, dict) or not data:
        # Conservative: when the cache is empty (offline/test), assume any
        # Claude family model supports adaptive thinking. The offline
        # allowlist for native messages already covers these.
        normalized = str(model).lower()
        if normalized.startswith("anthropic/"):
            normalized = normalized.split("/", 1)[1]
        return normalized.startswith(("claude-sonnet-4", "claude-opus-4", "claude-haiku-4"))
    record = data.get(model)
    if not isinstance(record, dict):
        return False
    return bool(record.get("adaptive_thinking_supported"))


@dataclass(frozen=True)
class BridgeExecutionPlan:
    strategy_name: str
    inbound_protocol: str
    caller_protocol: str
    upstream_protocol: str
    header_kind: str
    requested_model: str | None
    resolved_model: str | None
    upstream_body: dict
    stream: bool
    is_compact: bool = False
    approval_agent: bool = False
    diagnostics: tuple[dict, ...] = ()

    @property
    def upstream_path(self) -> str:
        if self.upstream_protocol == "messages":
            return "/v1/messages"
        if self.upstream_protocol == "responses":
            return "/responses"
        return "/chat/completions"


class ProtocolBridgeStrategy(ABC):
    strategy_name: str
    inbound_protocol: str
    target_family: str | tuple[str, ...]
    upstream_protocol: str
    header_kind: str
    caller_protocol: str

    def matches(
        self,
        inbound_protocol: str,
        target_family: str,
        *,
        model: str | None = None,
        capability_resolver: Callable[[str | None], bool] | None = None,
    ) -> bool:
        del model, capability_resolver
        supported_families = (
            self.target_family if isinstance(self.target_family, tuple) else (self.target_family,)
        )
        return self.inbound_protocol == inbound_protocol and target_family in supported_families

    @abstractmethod
    async def build_plan(
        self,
        body: dict,
        *,
        requested_model: str | None,
        resolved_model: str | None,
        api_base: str,
        api_key: str,
        is_compact: bool,
    ) -> BridgeExecutionPlan:
        raise NotImplementedError


class ResponsesToResponsesStrategy(ProtocolBridgeStrategy):
    strategy_name = "responses_to_responses"
    inbound_protocol = "responses"
    target_family = "codex"
    upstream_protocol = "responses"
    header_kind = "responses"
    caller_protocol = "responses"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key, is_compact) -> BridgeExecutionPlan:
        del api_base, api_key
        diagnostics: list[dict] = []
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        incoming_reasoning = upstream_body.get("reasoning")
        if isinstance(incoming_reasoning, dict) and "effort" in incoming_reasoning:
            mapped_effort = effort_mapping.map_effort_for_model(
                resolved_model, incoming_reasoning.get("effort")
            )
            if mapped_effort is not None:
                upstream_body["reasoning"] = {**incoming_reasoning, "effort": mapped_effort}
        upstream_body = format_translation.normalize_responses_instructions_for_copilot(
            upstream_body,
            diagnostics=diagnostics,
        )
        upstream_body = format_translation.normalize_responses_input_for_copilot(
            upstream_body,
            diagnostics=diagnostics,
        )
        upstream_body = format_translation.sanitize_responses_tools_for_copilot(
            upstream_body,
            diagnostics=diagnostics,
        )
        upstream_body = format_translation.sanitize_responses_body_for_copilot(
            upstream_body,
            diagnostics=diagnostics,
        )
        return BridgeExecutionPlan(
            strategy_name=self.strategy_name,
            inbound_protocol=self.inbound_protocol,
            caller_protocol=self.caller_protocol,
            upstream_protocol=self.upstream_protocol,
            header_kind=self.header_kind,
            requested_model=requested_model,
            resolved_model=resolved_model,
            upstream_body=upstream_body,
            stream=bool(upstream_body.get("stream", False)),
            is_compact=is_compact,
            diagnostics=tuple(diagnostics),
        )


class ResponsesToChatStrategy(ProtocolBridgeStrategy):
    strategy_name = "responses_to_chat"
    inbound_protocol = "responses"
    target_family = ("claude", "gemini", "grok")
    upstream_protocol = "chat"
    header_kind = "chat"
    caller_protocol = "responses"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key, is_compact) -> BridgeExecutionPlan:
        del api_base, api_key
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        translated = format_translation.responses_request_to_chat(upstream_body)
        return BridgeExecutionPlan(
            strategy_name=self.strategy_name,
            inbound_protocol=self.inbound_protocol,
            caller_protocol=self.caller_protocol,
            upstream_protocol=self.upstream_protocol,
            header_kind=self.header_kind,
            requested_model=requested_model,
            resolved_model=resolved_model,
            upstream_body=translated,
            stream=bool(translated.get("stream", False)),
            is_compact=is_compact,
        )


class MessagesToChatStrategy(ProtocolBridgeStrategy):
    strategy_name = "messages_to_chat"
    inbound_protocol = "messages"
    target_family = ("claude", "gemini", "grok")
    upstream_protocol = "chat"
    header_kind = "anthropic"
    caller_protocol = "anthropic"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key, is_compact) -> BridgeExecutionPlan:
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        translated = await format_translation.anthropic_request_to_chat(upstream_body, api_base, api_key)
        return BridgeExecutionPlan(
            strategy_name=self.strategy_name,
            inbound_protocol=self.inbound_protocol,
            caller_protocol=self.caller_protocol,
            upstream_protocol=self.upstream_protocol,
            header_kind=self.header_kind,
            requested_model=requested_model,
            resolved_model=resolved_model,
            upstream_body=translated,
            stream=bool(translated.get("stream", False)),
            is_compact=is_compact,
        )


class MessagesToResponsesStrategy(ProtocolBridgeStrategy):
    strategy_name = "messages_to_responses"
    inbound_protocol = "messages"
    target_family = "codex"
    upstream_protocol = "responses"
    header_kind = "responses"
    caller_protocol = "anthropic"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key, is_compact) -> BridgeExecutionPlan:
        del api_base, api_key
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        translated = format_translation.anthropic_request_to_responses(upstream_body)
        diagnostics: list[dict] = []
        translated = format_translation.sanitize_responses_tools_for_copilot(
            translated, diagnostics=diagnostics
        )
        translated = format_translation.sanitize_responses_body_for_copilot(
            translated,
            diagnostics=diagnostics,
        )
        return BridgeExecutionPlan(
            strategy_name=self.strategy_name,
            inbound_protocol=self.inbound_protocol,
            caller_protocol=self.caller_protocol,
            upstream_protocol=self.upstream_protocol,
            header_kind=self.header_kind,
            requested_model=requested_model,
            resolved_model=resolved_model,
            upstream_body=translated,
            stream=bool(translated.get("stream", False)),
            is_compact=is_compact,
            diagnostics=tuple(diagnostics),
        )


class MessagesToMessagesStrategy(ProtocolBridgeStrategy):
    """Pass an Anthropic /v1/messages request straight through to upstream."""

    strategy_name = "messages_to_messages"
    inbound_protocol = "messages"
    target_family = "claude"
    upstream_protocol = "messages"
    header_kind = "messages"
    caller_protocol = "anthropic"

    def matches(
        self,
        inbound_protocol: str,
        target_family: str,
        *,
        model: str | None = None,
        capability_resolver: Callable[[str | None], bool] | None = None,
    ) -> bool:
        if inbound_protocol != self.inbound_protocol:
            return False
        if target_family != self.target_family:
            return False
        resolver = capability_resolver or _default_capability_resolver
        return bool(resolver(model))

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key, is_compact) -> BridgeExecutionPlan:
        del api_base, api_key
        import messages_preprocess  # local to avoid hard import cycle
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        upstream_body = messages_preprocess.prepare_messages_passthrough_payload(
            upstream_body,
            model_supports_adaptive=_default_adaptive_thinking_resolver(resolved_model),
        )
        stream_value = upstream_body["stream"] if "stream" in upstream_body else False
        return BridgeExecutionPlan(
            strategy_name=self.strategy_name,
            inbound_protocol=self.inbound_protocol,
            caller_protocol=self.caller_protocol,
            upstream_protocol=self.upstream_protocol,
            header_kind=self.header_kind,
            requested_model=requested_model,
            resolved_model=resolved_model,
            upstream_body=upstream_body,
            stream=bool(stream_value),
            is_compact=is_compact,
        )


class ResponsesToMessagesStrategy(ProtocolBridgeStrategy):
    """Translate inbound Codex /v1/responses to upstream Anthropic /v1/messages."""

    strategy_name = "responses_to_messages"
    inbound_protocol = "responses"
    target_family = "claude"
    upstream_protocol = "messages"
    header_kind = "messages"
    caller_protocol = "responses"

    def matches(
        self,
        inbound_protocol: str,
        target_family: str,
        *,
        model: str | None = None,
        capability_resolver: Callable[[str | None], bool] | None = None,
    ) -> bool:
        if inbound_protocol != self.inbound_protocol:
            return False
        if target_family != self.target_family:
            return False
        resolver = capability_resolver or _default_capability_resolver
        return bool(resolver(model))

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key, is_compact) -> BridgeExecutionPlan:
        del api_base, api_key
        import messages_preprocess  # local to avoid hard import cycle
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        translated = format_translation.responses_request_to_anthropic_messages(upstream_body)
        translated = messages_preprocess.prepare_messages_passthrough_payload(
            translated,
            model_supports_adaptive=_default_adaptive_thinking_resolver(resolved_model),
        )
        if format_translation._responses_body_requests_prompt_cache(upstream_body):
            format_translation._apply_responses_prompt_cache_to_anthropic_messages(translated)
        return BridgeExecutionPlan(
            strategy_name=self.strategy_name,
            inbound_protocol=self.inbound_protocol,
            caller_protocol=self.caller_protocol,
            upstream_protocol=self.upstream_protocol,
            header_kind=self.header_kind,
            requested_model=requested_model,
            resolved_model=resolved_model,
            upstream_body=translated,
            stream=bool(translated.get("stream", False)),
            is_compact=is_compact,
        )


class ProtocolBridgePlanner:
    def __init__(
        self,
        routing_config_service: ModelRoutingConfigService,
        *,
        capability_resolver: Callable[[str | None], bool] | None = None,
    ):
        self._routing_config_service = routing_config_service
        self._capability_resolver = capability_resolver or _default_capability_resolver
        # Native-messages strategies are registered FIRST so they win for
        # Claude targets when capability gating allows; otherwise the planner
        # falls through to the existing chat/responses bridges.
        self._strategies = [
            MessagesToMessagesStrategy(),
            ResponsesToMessagesStrategy(),
            ResponsesToResponsesStrategy(),
            ResponsesToChatStrategy(),
            MessagesToChatStrategy(),
            MessagesToResponsesStrategy(),
        ]

    async def plan(
        self,
        inbound_protocol: str,
        body: dict,
        *,
        api_base: str,
        api_key: str,
        subagent: str | None = None,
        is_compact: bool = False,
    ) -> BridgeExecutionPlan:
        requested_model = body.get("model") if isinstance(body, dict) else None
        mapped_model = None
        approval_agent = is_approval_agent_request(
            subagent=subagent,
            inbound_protocol=inbound_protocol,
            body=body if isinstance(body, dict) else None,
        )
        if approval_agent:
            mapped_model = self._routing_config_service.resolve_approval_target_model(requested_model)
        if mapped_model is None:
            mapped_model = self._routing_config_service.resolve_target_model(requested_model)
        resolved_model = normalize_routing_model_name(mapped_model or requested_model)
        target_family = model_provider_family(resolved_model)
        if target_family is None:
            raise ValueError(f"Unsupported mapped model family: {resolved_model}")

        # Compaction synthesis is built around the chat-completions fallback;
        # never route compaction through the native /v1/messages bridge.
        capability_override = (lambda _model: False) if is_compact else None
        strategy = self._strategy_for(
            inbound_protocol,
            target_family,
            resolved_model,
            capability_override=capability_override,
        )
        plan = await strategy.build_plan(
            body,
            requested_model=requested_model,
            resolved_model=resolved_model,
            api_base=api_base,
            api_key=api_key,
            is_compact=is_compact,
        )
        return replace(plan, approval_agent=approval_agent)

    def _strategy_for(
        self,
        inbound_protocol: str,
        target_family: str,
        resolved_model: str | None = None,
        *,
        capability_override: Callable[[str | None], bool] | None = None,
    ) -> ProtocolBridgeStrategy:
        resolver = capability_override or self._capability_resolver
        for strategy in self._strategies:
            if strategy.matches(
                inbound_protocol,
                target_family,
                model=resolved_model,
                capability_resolver=resolver,
            ):
                return strategy
        raise ValueError(f"No bridge strategy for {inbound_protocol} -> {target_family}")
