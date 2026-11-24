import asyncio
import gzip
import json
import os
import time
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

import copilot_fusion as fusion
import copilot_sdk_upstream as sdk
from effort_mapping import map_effort_for_model
from proxy_client_config import ProxyClientConfigService


def _catalog(*tiers):
    return {"auto": {"tiers": list(tiers)}}


def _tier(name, *, enabled=True, visible=True):
    return {
        "fusionTier": name,
        "status": {"enabled": enabled, "pickerVisible": visible},
    }


def _client(payload):
    return SimpleNamespace(
        rpc=SimpleNamespace(models=SimpleNamespace(
            list=AsyncMock(return_value=SimpleNamespace(to_dict=lambda: payload)),
        )),
        list_models=AsyncMock(return_value=[SimpleNamespace(id="auto")]),
    )


class CopilotFusionMappingTests(unittest.TestCase):
    def test_native_and_compatible_efforts_select_routing_tiers(self):
        for effort, expected in ((None, "standard"), ("standard", "standard"),
                                 ("medium", "standard"), ("high", "high")):
            for field in ("nested", "top_level"):
                with self.subTest(effort=effort, field=field):
                    body = {"model": "fusion"}
                    if effort is not None:
                        body.update({"reasoning": {"effort": effort}} if field == "nested"
                                    else {"reasoning_effort": effort})
                    options = sdk._session_options(body, sdk.ToolRegistration())
                    self.assertEqual(options["model"], "fusion")
                    self.assertEqual(options["capi"], {
                        "enable_web_socket_responses": False, "fusion_tier": expected,
                    })
                    self.assertNotIn("reasoning_effort", options)

    def test_top_level_tier_takes_precedence(self):
        self.assertEqual(fusion.tier_for_request({
            "reasoning_effort": "high", "reasoning": {"effort": "medium"},
        }), "high")

    def test_unsupported_efforts_do_not_silently_downgrade(self):
        for effort in ("low", "xhigh", "max", "", 10, ["high"]):
            with self.subTest(effort=effort), self.assertRaises(sdk.AdapterValidationError):
                sdk._session_options({"model": "fusion", "reasoning": {"effort": effort}}, sdk.ToolRegistration())

    def test_qualified_model_ids_are_normalized(self):
        options = sdk._session_options({
            "model": "github-copilot/Fusion", "reasoning_effort": "high",
        }, sdk.ToolRegistration())
        self.assertEqual(options["model"], "fusion")
        self.assertEqual(options["capi"]["fusion_tier"], "high")

    def test_plain_auto_and_concrete_model_options_are_unchanged(self):
        for model in ("auto", "gpt-6.1-sol"):
            with self.subTest(model=model):
                options = sdk._session_options({
                    "model": model, "reasoning": {"effort": "high"},
                }, sdk.ToolRegistration())
                self.assertEqual(options["reasoning_effort"], "high")
                self.assertNotIn("fusion_tier", options["capi"])

    def test_standard_alias_is_only_for_fusion_translation(self):
        self.assertEqual(map_effort_for_model("fusion", "standard"), "medium")
        self.assertEqual(map_effort_for_model("github-copilot/fusion", "standard"), "medium")
        self.assertIsNone(map_effort_for_model("gpt-6.1-sol", "standard"))

    def test_discovery_filters_ineligible_hidden_and_out_of_scope_tiers(self):
        caps = fusion.model_capabilities(_catalog(
            _tier("standard"), _tier("high", enabled=False), _tier("max"),
            _tier("high", visible=False), {"id": "fast"}, None,
        ))
        self.assertEqual(caps["fusion"]["reasoning_efforts"], ["medium"])
        self.assertEqual(caps["fusion"]["fusion_tiers"], ["standard"])

    def test_discovery_requires_positive_fusion_eligibility(self):
        for payload in ({}, {"auto": {}}, {"auto": {"tiers": None}},
                        _catalog(_tier("high", enabled=False)),
                        _catalog(_tier("max")), _catalog({"fusionTier": "high", "status": {}})):
            with self.subTest(payload=payload):
                self.assertEqual(fusion.model_capabilities(payload), {})

    def test_generated_picker_uses_discovered_medium_high_analogues(self):
        service = object.__new__(ProxyClientConfigService)
        service._config = SimpleNamespace(codex_model_context_window=272_000,
                                          codex_model_auto_compact_token_limit=180_000)
        service._model_capabilities_provider = lambda: fusion.model_capabilities(
            _catalog(_tier("high"), _tier("standard"), _tier("max", enabled=False)),
        )
        service._model_routing_settings_provider = lambda: {}
        model, = service._build_codex_model_catalog_payload()["models"]
        self.assertEqual(model["slug"], "fusion")
        self.assertEqual(model["display_name"], "Copilot Fusion")
        self.assertEqual(model["default_reasoning_level"], "medium")
        self.assertEqual([x["effort"] for x in model["supported_reasoning_levels"]], ["medium", "high"])
        self.assertIn("Standard", model["supported_reasoning_levels"][0]["description"])

    def test_tier_changes_are_visible_to_session_reuse_options(self):
        standard = sdk._session_options({"model": "fusion"}, sdk.ToolRegistration())
        high = sdk._session_options({"model": "fusion", "reasoning_effort": "high"}, sdk.ToolRegistration())
        self.assertNotEqual(standard, high)


class CopilotFusionContinuityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.entry = sdk._LiveSession(
            session=SimpleNamespace(session_id="conversation-1"),
            options={"model": "fusion"}, diagnostics={},
        )
        self.sessions = {"conversation-1": self.entry}
        for name, value in (("_live_sessions", self.sessions), ("_fusion_routes", OrderedDict())):
            patcher = patch.object(sdk, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.handler = sdk._UpstreamRequestHandler()
        self.upstream = AsyncMock(side_effect=self.route_response)
        patcher = patch.object(self.handler, "_send_upstream", self.upstream)
        patcher.start()
        self.addCleanup(patcher.stop)

    def route_response(self, request):
        if request.url.path == "/model/fusion":
            payload = {"steps": [{"role": "primary", "model_id": "gpt-6-luna"}],
                       "session": {"token": "synthetic-fusion-token", "expires_at": time.time() + 3600}}
        else:
            payload = {"selected_model": {"id": "gpt-6-luna"},
                       "session_token": "synthetic-fusion-token", "expires_at": time.time() + 3600}
        return httpx.Response(200, json=payload, request=request)

    async def route(self, path="/auto", *, session_id="conversation-1", parent_agent_id=None, **changes):
        body = {"prompt": "First question", **(
            {"fusion_mode": "hydrafusion", "plan_version": "v1"} if path == "/model/fusion"
            else {"tier": "balance", "has_image": False, "fusion": True}
        ), **changes}
        request = httpx.Request("POST", "https://example.invalid" + path, json=body)
        return await self.handler.send_request(
            request, SimpleNamespace(session_id=session_id, parent_agent_id=parent_agent_id),
        )

    async def test_followups_reuse_native_routes_and_billing_tokens_for_both_tiers(self):
        for path in ("/auto", "/model/fusion"):
            first = await self.route(path)
            followup = await self.route(path, prompt="Different, more complex question",
                                        multi_turn={"turns_since_anchor": 1},
                                        hydra_rl_multi_turn={"state_token": "new-state"})
            self.assertEqual(followup.json(), first.json())
        self.assertEqual(self.upstream.await_count, 2)
        self.assertEqual([x["cache"] for x in self.entry.diagnostics["fusion_routes"]],
                         ["miss", "hit", "miss", "hit"])
        self.assertNotIn("synthetic-fusion-token", json.dumps(self.entry.diagnostics))

    async def test_conversations_subagents_and_plain_auto_do_not_share_routes(self):
        await self.route()
        self.sessions["conversation-2"] = sdk._LiveSession(
            session=SimpleNamespace(session_id="conversation-2"), options={"model": "fusion"},
        )
        await self.route(session_id="conversation-2")
        await self.route(parent_agent_id="parent")
        self.entry.options = {"model": "auto"}
        await self.route()
        self.assertEqual(self.upstream.await_count, 4)

    async def test_changed_tier_or_vision_requirements_reroute(self):
        await self.route()
        await self.route(tier="hydrafusion")
        await self.route(has_image=True)
        await self.route()
        self.assertEqual(self.upstream.await_count, 4)

    async def test_token_is_refreshed_before_expiry(self):
        now = time.time()
        with patch.object(sdk.time, "time", return_value=now):
            await self.route()
        with patch.object(sdk.time, "time", return_value=now + 3300):
            await self.route()
        self.assertEqual(self.upstream.await_count, 2)

    async def test_compaction_and_model_errors_invalidate_routes(self):
        await self.route()
        sdk._set_compaction_state(self.entry, True)
        sdk._set_compaction_state(self.entry, False)
        await self.route()
        await sdk._observe_http_response(httpx.Response(403, content=b"model unavailable"), {},
                                         session_id="conversation-1")
        await self.route()
        self.assertEqual(self.upstream.await_count, 3)

    async def test_ordinary_disk_resume_retains_the_decision(self):
        first = await self.route()
        self.sessions["conversation-1"] = sdk._LiveSession(
            session=SimpleNamespace(session_id="conversation-1"), options={"model": "fusion"},
        )
        self.assertEqual((await self.route()).json(), first.json())
        self.assertEqual(self.upstream.await_count, 1)

    async def test_explicit_compaction_release_invalidates_the_decision(self):
        await self.route()
        self.entry.tool_choice = "none"
        await sdk._release_session(self.entry.session, sdk.TurnOutcome(), completed=True)
        self.entry.reaper.cancel()
        await self.route()
        self.assertEqual(self.upstream.await_count, 2)

    async def test_settled_interrupt_retains_the_decision(self):
        await self.route()
        await sdk._release_session(self.entry.session, None, completed=False, park=True)
        self.entry.reaper.cancel()
        await self.route(prompt="Continue after interrupt")
        self.assertEqual(self.upstream.await_count, 1)

    async def test_model_selection_changes_discard_the_old_route(self):
        await self.route()
        self.entry.in_use = False
        with patch.object(sdk, "_evict_live_session", AsyncMock()):
            await sdk._reuse_live_session(
                "conversation-1", allow_pending=False,
                options={"model": "fusion", "capi": {"fusion_tier": "high"}},
            )
        await self.route()
        self.assertEqual(self.upstream.await_count, 2)

    async def test_failed_or_invalid_router_responses_are_not_cached(self):
        for response in (
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(200, content=b"not json"),
            httpx.Response(200, json={"session_token": "invalid", "expires_at": time.time() + 3600}),
            httpx.Response(200, json={"selected_model": {"id": "gpt-6-luna"},
                                     "session_token": "invalid", "expires_at": True}),
        ):
            with self.subTest(status=response.status_code, content=response.content):
                sdk._invalidate_fusion_routes("conversation-1")
                self.upstream.side_effect = [response, self.route_response(httpx.Request("POST", "https://example.invalid/auto"))]
                self.upstream.reset_mock()
                await self.route()
                await self.route()
                self.assertEqual(self.upstream.await_count, 2)

    async def test_compressed_routing_responses_remain_readable_on_reuse(self):
        request = httpx.Request("POST", "https://example.invalid/auto")
        payload = self.route_response(request).content
        self.upstream.return_value = httpx.Response(
            200, headers={"content-type": "application/json", "content-encoding": "gzip"},
            content=gzip.compress(payload),
        )
        self.upstream.side_effect = None
        for _ in range(2):
            response = await self.route()
            forwarded = httpx.Response(200, headers=response.headers, content=await response.aread())
            self.assertEqual(forwarded.json()["session_token"], "synthetic-fusion-token")
        self.assertEqual(self.upstream.await_count, 1)

    async def test_invalidation_during_routing_does_not_restore_stale_decision(self):
        async def send(request):
            sdk._invalidate_fusion_routes("conversation-1")
            await asyncio.sleep(0)
            return self.route_response(request)
        self.upstream.side_effect = send
        await self.route()
        await self.route()
        self.assertEqual(self.upstream.await_count, 2)


class CopilotFusionDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(sdk, "_fusion_model_caps", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_native_alias_is_validated_against_account_metadata(self):
        client = _client(_catalog(_tier("standard"), _tier("high")))
        await sdk._validate_requested_model({"model": "fusion", "reasoning_effort": "high"}, client)
        self.assertEqual(sdk.get_fusion_model_capabilities()["fusion"]["fusion_tiers"], ["standard", "high"])
        client.list_models.assert_not_awaited()

    async def test_disabled_tiers_and_missing_metadata_are_rejected(self):
        for payload in ({}, _catalog(_tier("standard"), _tier("high", enabled=False))):
            with self.subTest(payload=payload), self.assertRaises(sdk.AdapterValidationError):
                await sdk._validate_requested_model(
                    {"model": "fusion", "reasoning_effort": "high"}, _client(payload),
                )

    async def test_discovery_failure_clears_stale_capabilities(self):
        client = _client(_catalog(_tier("standard"), _tier("high")))
        await sdk.refresh_fusion_model_capabilities(client)
        client.rpc.models.list.side_effect = RuntimeError("discovery offline")
        self.assertEqual(await sdk.refresh_fusion_model_capabilities(client), {})
        self.assertEqual(sdk.get_fusion_model_capabilities(), {})
        with self.assertRaises(sdk.AdapterValidationError):
            await sdk._validate_requested_model({"model": "fusion"}, client)

    async def test_capability_snapshots_cannot_mutate_cached_eligibility(self):
        await sdk.refresh_fusion_model_capabilities(_client(_catalog(_tier("standard"))))
        sdk.get_fusion_model_capabilities()["fusion"]["fusion_tiers"].append("high")
        self.assertEqual(sdk.get_fusion_model_capabilities()["fusion"]["fusion_tiers"], ["standard"])

    async def test_fusion_does_not_send_a_model_reasoning_effort(self):
        client = _client({})
        effort = await sdk._reasoning_effort_for_client(
            {"model": "fusion", "reasoning_effort": "high"}, client,
        )
        self.assertIsNone(effort)
        client.list_models.assert_not_awaited()

    async def test_models_endpoint_adds_only_discovered_fusion(self):
        client = _client(_catalog(_tier("standard"), _tier("high")))
        with patch.object(sdk, "_get_client", AsyncMock(return_value=client)), \
             patch.object(sdk, "_runtime_ssl_context", return_value=True), \
             patch.object(sdk.excel_model_catalog, "fetch_model_capabilities", return_value={}):
            response = await sdk.models_response()
        self.assertEqual([m["id"] for m in json.loads(response.body)["data"]], ["auto", "fusion"])


class CopilotFusionProxyCatalogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with patch.dict(os.environ, {
            "GHCP_CODEX_NATIVE_INGEST_INTERVAL": "0",
            "GHCP_COPILOT_SDK_INGEST_INTERVAL": "0",
        }):
            import proxy
        self.proxy = proxy

    async def test_fusion_union_does_not_contaminate_raw_http_catalog(self):
        proxy = self.proxy
        raw_catalog = {"gpt-6.1-sol": {}}
        fusion_caps = fusion.model_capabilities(_catalog(_tier("standard"), _tier("high")))
        with patch.object(proxy, "fetch_copilot_model_capabilities", return_value=raw_catalog), \
             patch.object(proxy, "fetch_excel_model_capabilities", return_value={"gpt-6-sol": {}}), \
             patch.object(sdk, "get_fusion_model_capabilities", return_value=fusion_caps), \
             patch.object(sdk, "enabled", return_value=True):
            result = proxy.fetch_available_model_capabilities()
            self.assertEqual(set(result), {"gpt-6.1-sol", "gpt-6-sol-excel", "fusion"})
            self.assertNotIn("fusion", raw_catalog)
            with patch.object(sdk, "enabled", return_value=False):
                self.assertNotIn("fusion", proxy.fetch_available_model_capabilities())
            with patch.object(sdk, "get_fusion_model_capabilities", return_value={}):
                self.assertNotIn("fusion", proxy.fetch_available_model_capabilities())

    async def test_startup_discovers_tiers_before_refreshing_the_picker(self):
        order = []

        async def discover():
            order.append("discover")

        def refresh():
            order.append("refresh")

        with patch.object(sdk, "enabled", return_value=True), \
             patch.object(sdk, "refresh_fusion_model_capabilities", side_effect=discover), \
             patch.object(self.proxy.client_proxy_config_service, "refresh_codex_model_catalog", side_effect=refresh):
            await self.proxy._refresh_sdk_client_model_catalog()
        self.assertEqual(order, ["discover", "refresh"])


if __name__ == "__main__":
    unittest.main()
