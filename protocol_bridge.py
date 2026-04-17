"""Protocol bridge planning using pluggable strategy objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import format_translation
from initiator_policy import is_approval_agent_request
from model_routing_config import ModelRoutingConfigService, model_provider_family, normalize_routing_model_name


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

    @property
    def upstream_path(self) -> str:
        return "/responses" if self.upstream_protocol == "responses" else "/chat/completions"


class ProtocolBridgeStrategy(ABC):
    strategy_name: str
    inbound_protocol: str
    target_family: str
    upstream_protocol: str
    header_kind: str
    caller_protocol: str

    def matches(self, inbound_protocol: str, target_family: str) -> bool:
        return self.inbound_protocol == inbound_protocol and self.target_family == target_family

    @abstractmethod
    async def build_plan(
        self,
        body: dict,
        *,
        requested_model: str | None,
        resolved_model: str | None,
        api_base: str,
        api_key: str,
    ) -> BridgeExecutionPlan:
        raise NotImplementedError


class ResponsesToResponsesStrategy(ProtocolBridgeStrategy):
    strategy_name = "responses_to_responses"
    inbound_protocol = "responses"
    target_family = "codex"
    upstream_protocol = "responses"
    header_kind = "responses"
    caller_protocol = "responses"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key) -> BridgeExecutionPlan:
        del api_base, api_key
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
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
        )


class ResponsesToChatStrategy(ProtocolBridgeStrategy):
    strategy_name = "responses_to_chat"
    inbound_protocol = "responses"
    target_family = "claude"
    upstream_protocol = "chat"
    header_kind = "chat"
    caller_protocol = "responses"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key) -> BridgeExecutionPlan:
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
        )


class MessagesToChatStrategy(ProtocolBridgeStrategy):
    strategy_name = "messages_to_chat"
    inbound_protocol = "messages"
    target_family = "claude"
    upstream_protocol = "chat"
    header_kind = "anthropic"
    caller_protocol = "anthropic"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key) -> BridgeExecutionPlan:
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
        )


class MessagesToResponsesStrategy(ProtocolBridgeStrategy):
    strategy_name = "messages_to_responses"
    inbound_protocol = "messages"
    target_family = "codex"
    upstream_protocol = "responses"
    header_kind = "responses"
    caller_protocol = "anthropic"

    async def build_plan(self, body: dict, *, requested_model, resolved_model, api_base, api_key) -> BridgeExecutionPlan:
        del api_base, api_key
        upstream_body = dict(body)
        upstream_body["model"] = resolved_model
        translated = format_translation.anthropic_request_to_responses(upstream_body)
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
        )


class ProtocolBridgePlanner:
    def __init__(self, routing_config_service: ModelRoutingConfigService):
        self._routing_config_service = routing_config_service
        self._strategies = [
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
    ) -> BridgeExecutionPlan:
        requested_model = body.get("model") if isinstance(body, dict) else None
        mapped_model = None
        if is_approval_agent_request(
            subagent=subagent,
            inbound_protocol=inbound_protocol,
            body=body if isinstance(body, dict) else None,
        ):
            mapped_model = self._routing_config_service.resolve_approval_target_model(requested_model)
        if mapped_model is None:
            mapped_model = self._routing_config_service.resolve_target_model(requested_model)
        resolved_model = normalize_routing_model_name(mapped_model or requested_model)
        target_family = model_provider_family(resolved_model)
        if target_family is None:
            raise ValueError(f"Unsupported mapped model family: {resolved_model}")

        strategy = self._strategy_for(inbound_protocol, target_family)
        return await strategy.build_plan(
            body,
            requested_model=requested_model,
            resolved_model=resolved_model,
            api_base=api_base,
            api_key=api_key,
        )

    def _strategy_for(self, inbound_protocol: str, target_family: str) -> ProtocolBridgeStrategy:
        for strategy in self._strategies:
            if strategy.matches(inbound_protocol, target_family):
                return strategy
        raise ValueError(f"No bridge strategy for {inbound_protocol} -> {target_family}")
