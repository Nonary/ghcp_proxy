#!/usr/bin/env python
"""Check Codex's model-catalog v2 selector without inference or spawning agents.

The app-server uses an isolated CODEX_HOME and an unreachable model endpoint.
Only initialize and model/list are sent; existing Codex settings are untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxy_client_config import ProxyClientConfigService


def read_models(binary: str, catalog: dict) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="ghcp-v2-catalog-probe-") as directory:
        home = Path(directory)
        (home / "models.json").write_text(json.dumps(catalog), encoding="utf-8")
        (home / "config.toml").write_text(
            'model = "gpt-6.1-sol"\nmodel_provider = "custom"\n'
            f'model_catalog_json = {json.dumps(str(home / "models.json"))}\n'
            '[model_providers.custom]\nname = "Catalog probe"\n'
            'base_url = "http://127.0.0.1:1/v1"\nwire_api = "responses"\n'
            'requires_openai_auth = false\n',
            encoding="utf-8",
        )
        with (home / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [binary, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, text=True, bufsize=1,
                env={**os.environ, "CODEX_HOME": directory},
            )
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)

            def rpc(identifier: int, method: str, params: dict) -> dict:
                process.stdin.write(json.dumps({
                    "id": identifier, "method": method, "params": params,
                }) + "\n")
                process.stdin.flush()
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if not selector.select(0.5):
                        continue
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError("Codex app-server closed before replying")
                    response = json.loads(line)
                    if response.get("id") != identifier:
                        continue
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    return response["result"]
                raise TimeoutError(f"Codex did not reply to {method}")

            try:
                rpc(1, "initialize", {
                    "clientInfo": {"name": "ghcp_v2_catalog_probe", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                })
                process.stdin.write(json.dumps({"method": "initialized", "params": {}}) + "\n")
                process.stdin.flush()
                return rpc(2, "model/list", {"includeHidden": True})["data"]
            finally:
                selector.close()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", default=shutil.which("codex"), help="Codex CLI executable")
    args = parser.parse_args()
    if not args.codex:
        parser.error("Codex was not found; supply --codex /absolute/path/to/codex")
    service = object.__new__(ProxyClientConfigService)
    service._config = SimpleNamespace(
        codex_model_context_window=272_000, codex_model_auto_compact_token_limit=180_000,
    )
    service._model_capabilities_provider = lambda: {"gpt-6.1-sol": {}, "gpt-6-sol-excel": {}}
    service._model_routing_settings_provider = lambda: {}
    catalog = service._build_codex_model_catalog_payload()
    before = copy.deepcopy(catalog)
    for model in before["models"]:
        model.pop("multi_agent_version", None)
    names = {model["slug"] for model in catalog["models"]}
    results = {}
    for label, payload in (("before", before), ("after", catalog)):
        results[label] = {
            model["model"]: model.get("multiAgentVersion")
            for model in read_models(args.codex, payload) if model["model"] in names
        }
        if set(results[label]) != names:
            raise RuntimeError(f"Codex did not return the complete {label} probe catalog")
    if any(value != "v2" for value in results["after"].values()):
        raise RuntimeError(f"Codex did not select v2: {results['after']}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
