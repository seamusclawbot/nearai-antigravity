#!/usr/bin/env python3
"""Offline unit and loopback-proxy tests for nearai_agy.py (stdlib only)."""

from __future__ import annotations

import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nearai_agy as n  # noqa: E402


class TestModalityAndFiltering(unittest.TestCase):
    def test_is_chat_model_excludes_class_b_and_c(self):
        # Chat models must have text in both inputs and outputs and no non-chat keyword
        self.assertTrue(n._is_chat_model("anthropic/claude-sonnet-4-5", ["text", "image"], ["text"]))
        self.assertTrue(n._is_chat_model("Qwen/Qwen3.8-27B", ["text"], ["text"]))

        # Class B / Class C non-chat models (missing text output or non-chat keyword)
        self.assertFalse(n._is_chat_model("BAAI/bge-large-en-v1.5", ["text"], ["embedding"]))
        self.assertFalse(n._is_chat_model("Qwen/Qwen3-Embedding-8B", ["text"], ["text"]))
        self.assertFalse(n._is_chat_model("Qwen/Qwen3-Reranker-8B", ["text"], ["text"]))
        self.assertFalse(n._is_chat_model("openai/whisper-large-v3", ["audio"], ["text"]))
        self.assertFalse(n._is_chat_model("near-ai/pii-privacy-filter-v2", ["text"], ["text"]))

    def test_route_check_requires_freshness(self):
        self.assertIn("Fail-closed", n.cmd_attest_route_check.__doc__)

        profile = {"attestation_required": True, "select": {"tools_only": True}}
        catalog = {"models": []}
        with self.assertRaises(SystemExit):
            n.resolve_profile_models(profile, catalog, {})

        raw_chat_tee = {
            "id": "Qwen/Qwen3.8-27B",
            "owned_by": "Qwen",
            "context_length": 262144,
            "supported_features": ["tools"],
            "is_ready": True,
            "architecture": {
                "inputModalities": ["text"],
                "outputModalities": ["text"],
            },
            "pricing": {"prompt": "0.00000035", "completion": "0.0000015"},
        }
        raw_ext_spoofed = {
            "id": "levanto/sage-2",
            "owned_by": "levanto",
            "context_length": 128000,
            "supported_features": ["tools"],
            "is_ready": True,
            "architecture": {
                "inputModalities": ["text"],
                "outputModalities": ["text"],
            },
        }
        raw_embed = {
            "id": "Qwen/Qwen3-Embedding-8B",
            "owned_by": "Qwen",
            "supported_features": [],
            "is_ready": True,
            "architecture": {
                "inputModalities": ["text"],
                "outputModalities": ["embedding"],
            },
        }

        privacy_map = {
            "Qwen/Qwen3.8-27B": {
                "provider_type": "vllm",
                "verifiable": True,
                "attestation_supported": True,
            },
            "levanto/sage-2": {
                "provider_type": "external",
                "verifiable": True,
                "attestation_supported": False,
            },
        }

        catalog = [
            n._normalize_model(raw_chat_tee, privacy_map["Qwen/Qwen3.8-27B"]),
            n._normalize_model(raw_ext_spoofed, privacy_map["levanto/sage-2"]),
            n._normalize_model(raw_embed, None),
        ]

        self.assertEqual(n.provider_trust_class({"provider_type": "vllm"}), "tee")
        self.assertEqual(n.provider_trust_class({"provider_type": "chutes"}), "tee")
        self.assertEqual(n.provider_trust_class({"provider_type": "tinfoil"}), "tee")
        for provider in ("", "unknown", "legacy", "sglang"):
            self.assertEqual(n.provider_trust_class({"provider_type": provider}), "unknown")
        self.assertEqual(n.provider_trust_class({"provider_type": "external"}), "external")

        # Default chat_only=True drops embedding model
        args_chat = types.SimpleNamespace(chat_only=True, tools_only=False, tee_only=False)
        filtered_chat = n.apply_filters(catalog, args_chat)
        self.assertEqual([m["id"] for m in filtered_chat], ["Qwen/Qwen3.8-27B", "levanto/sage-2"])

        # tee_only=True keeps only recognized TEE providers (drops external/unknown)
        args_tee = types.SimpleNamespace(chat_only=True, tools_only=True, tee_only=True)
        filtered_tee = n.apply_filters(catalog, args_tee)
        self.assertEqual([m["id"] for m in filtered_tee], ["Qwen/Qwen3.8-27B"])


class TestProbeClassificationAndEffort(unittest.TestCase):
    def test_agy_effort_injection_rules(self):
        self.assertEqual(n.agy_effort_for("anthropic/claude-sonnet-4-5"), "medium")
        self.assertEqual(n.agy_effort_for("google/gemini-3.1-pro-preview"), "medium")
        self.assertEqual(n.agy_effort_for("openai/o3-mini"), "low")
        self.assertIsNone(n.agy_effort_for("Qwen/Qwen3.8-27B"))

    def test_probe_reason_buckets(self):
        self.assertEqual(
            n.probe_reason({"ok": False, "code": 400, "error": "Unsupported parameter: reasoning_effort"}),
            "effort",
        )
        self.assertEqual(
            n.probe_reason({"ok": False, "code": 400, "error": "Please use /v1/systemone endpoint"}),
            "systemone",
        )
        self.assertEqual(
            n.probe_reason({"ok": False, "code": 429, "error": "Rate limit exceeded"}),
            "transient",
        )
        self.assertEqual(
            n.probe_reason({"ok": False, "code": 503, "error": "Service currently unavailable"}),
            "transient",
        )
        self.assertEqual(
            n.probe_reason({"ok": False, "code": 502, "error": "Upstream request failed with status 404"}),
            "route",
        )

    def test_probe_entry_ttl_adaptive(self):
        ok_entry = {"ok": True, "verdict": True}
        self.assertEqual(n.probe_entry_ttl(ok_entry), n.PROBE_TTL_OK)

        incompat_entry = {"ok": False, "verdict": True, "code": 400}
        self.assertEqual(n.probe_entry_ttl(incompat_entry), n.PROBE_TTL_FAIL)

        unk_entry = {"ok": False, "verdict": False, "code": 502}
        self.assertEqual(n.probe_entry_ttl(unk_entry), 0)
        self.assertEqual(n.probe_entry_ttl(None), 0)


class TestSlotBudgetAndLauncherRendering(unittest.TestCase):
    def test_plan_slots_enforces_32_cap(self):
        ids = [f"org/model-{i}" for i in range(35)]
        tiers = {"pro": "org/extra-pro", "flash": "org/model-0", "flash_lite": ""}
        plan = n.plan_slots(ids, tiers, cap=32)
        self.assertFalse(plan["fits"])
        self.assertEqual(len(plan["models"]), 31)
        self.assertEqual(plan["tiers"]["pro"], "org/extra-pro")
        self.assertEqual(len(plan["evicted_models"]), 4)

    def test_render_launcher_bash_syntax_and_no_leaked_keys(self):
        plan = n.plan_slots(
            ["anthropic/claude-sonnet-4-5", "Qwen/Qwen3.8-27B"],
            {"pro": "anthropic/claude-sonnet-4-5", "flash": "Qwen/Qwen3.8-27B", "flash_lite": "Qwen/Qwen3.8-27B"},
            cap=32,
        )
        for strip in (False, True):
            script = n.render_launcher(
                name="test",
                profile_ref="test",
                url="https://cloud-api.near.ai/v1",
                ttl=3600,
                cap=32,
                plan=plan,
                default_model="anthropic/claude-sonnet-4-5",
                strip_effort=strip,
            )
            self.assertIn('run test "$@"', script)
            self.assertIn('_MODELS_CSV="anthropic/claude-sonnet-4-5,Qwen/Qwen3.8-27B"', script)
            self.assertIsNone(re.search(r"sk-[A-Za-z0-9]{10,}", script))
            self.assertNotIn("/Users/", script)
            proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_security_validations_reject_unsafe_inputs(self):
        with self.assertRaises(ValueError):
            n._validate_gateway_url("file:///etc/passwd")
        with self.assertRaises(ValueError):
            n._validate_gateway_url("http://remote-plaintext.example.com/v1")
        with self.assertRaises(ValueError):
            n._validate_gateway_url("https://cloud-api.near.ai/v1$(whoami)")
        self.assertEqual(n._validate_gateway_url("https://cloud-api.near.ai/v1"), "https://cloud-api.near.ai/v1")
        self.assertEqual(n._validate_gateway_url("http://127.0.0.1:8899/v1"), "http://127.0.0.1:8899/v1")

        bad_plan = {"models": ["org/good", "org/bad;rm -rf /"], "tiers": {"pro": "", "flash": "", "flash_lite": ""}}
        with self.assertRaises(ValueError):
            n.render_launcher("test", "test", "https://cloud-api.near.ai/v1", 3600, 32, bad_plan, "org/good")


class TestMockGatewayAndWhatIfProbe(unittest.TestCase):
    def test_probe_one_what_if_against_local_mock(self):
        seen_bodies = []

        class MockUpstream(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                seen_bodies.append(body)
                if str(body.get("model", "")).startswith("anthropic/") and "reasoning_effort" in body:
                    payload = json.dumps({"error": {"message": "Unsupported parameter: reasoning_effort"}}).encode()
                    self.send_response(400)
                else:
                    payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        srv = ThreadedServer(("127.0.0.1", 0), MockUpstream)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            gw = f"http://127.0.0.1:{port}/v1"
            # Default probe sends reasoning_effort="medium" for anthropic/* -> 400
            res_default = n.probe_one(gw, "dummy-key", "anthropic/claude-sonnet-4-5", strip_effort=False)
            self.assertFalse(res_default["ok"])
            self.assertEqual(n.probe_reason(res_default), "effort")

            # What-if probe strips reasoning_effort for anthropic/* -> 200
            res_what_if = n.probe_one(gw, "dummy-key", "anthropic/claude-sonnet-4-5", strip_effort=True)
            self.assertTrue(res_what_if["ok"])

            # Non-anthropic model (google/*) keeps reasoning_effort even with strip_effort=True
            res_google = n.probe_one(gw, "dummy-key", "google/gemini-3.1-pro-preview", strip_effort=True)
            self.assertTrue(res_google["ok"])
            self.assertEqual(seen_bodies[-1].get("reasoning_effort"), "medium")
        finally:
            srv.shutdown()
            srv.server_close()


class TestCustomModelsSync(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.tmpdir.name) / "settings.json"
        self.state_path = Path(self.tmpdir.name) / "jetski_state.pbtxt"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sync_creates_and_populates_settings_and_state(self):
        catalog = {
            "models": [
                {"id": "Qwen/Qwen3.8-27B", "context": 262144, "max_output": 32768},
                {"id": "anthropic/claude-sonnet-4-5", "context": 200000, "max_output": 8192},
            ]
        }
        count = n.sync_custom_models(
            catalog,
            ["Qwen/Qwen3.8-27B", "anthropic/claude-sonnet-4-5"],
            settings_path=self.settings_path,
            state_path=self.state_path,
        )
        self.assertEqual(count, 2)
        self.assertTrue(self.settings_path.is_file())
        self.assertTrue(self.state_path.is_file())

        with open(self.settings_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # customModels must be a JSON object mapping model IDs to config dicts
        self.assertIsInstance(data.get("customModels"), dict)
        self.assertIn("Qwen/Qwen3.8-27B", data["customModels"])
        qwen = data["customModels"]["Qwen/Qwen3.8-27B"]
        self.assertEqual(qwen["modelName"], "Qwen/Qwen3.8-27B")
        self.assertEqual(qwen["maxTokens"], 262144)
        self.assertEqual(qwen["maxOutputTokens"], 32768)

        claude = data["customModels"]["anthropic/claude-sonnet-4-5"]
        self.assertEqual(claude["maxTokens"], 200000)
        self.assertEqual(claude["maxOutputTokens"], 8192)

        # Check protobuf textpb state
        with open(self.state_path, "r", encoding="utf-8") as fh:
            pbtxt = fh.read()
        self.assertIn("custom_models_config: {", pbtxt)
        self.assertIn('key: "Qwen/Qwen3.8-27B"', pbtxt)
        self.assertIn('model_name: "Qwen/Qwen3.8-27B"', pbtxt)
        self.assertIn("max_tokens: 262144", pbtxt)
        self.assertIn("max_output_tokens: 32768", pbtxt)

    def test_sync_migrates_legacy_custom_models_dict_and_preserves_keys(self):
        initial = {
            "model": "Gemini 3.8 Flash (Medium)",
            "toolPermission": "always-proceed",
            "trustedWorkspaces": ["/path/one", "/path/two"],
            "customModels": True,
            "customModelsConfig": {
                "existing/model": {
                    "modelName": "existing/model",
                    "label": "Existing",
                    "maxTokens": 100000,
                    "maxOutputTokens": 4096,
                }
            },
        }
        with open(self.settings_path, "w", encoding="utf-8") as fh:
            json.dump(initial, fh)

        catalog = [{"id": "new/model", "context_length": 131072, "max_output_length": 8192}]
        n.sync_custom_models(catalog, ["new/model"], settings_path=self.settings_path, state_path=self.state_path)

        with open(self.settings_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertEqual(data["model"], "Gemini 3.8 Flash (Medium)")
        self.assertEqual(data["toolPermission"], "always-proceed")
        self.assertEqual(data["trustedWorkspaces"], ["/path/one", "/path/two"])
        self.assertIsInstance(data["customModels"], dict)
        self.assertIn("existing/model", data["customModels"])
        self.assertIn("new/model", data["customModels"])
        self.assertEqual(data["customModels"]["new/model"]["maxTokens"], 131072)
        self.assertNotIn("customModelsConfig", data)

    def test_sync_fallbacks_for_zero_or_missing_context(self):
        catalog = [{"id": "zero/model", "context": 0, "max_output": 0}]
        n.sync_custom_models(catalog, ["zero/model", "unknown/model"], settings_path=self.settings_path, state_path=self.state_path)

        with open(self.settings_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        zm = data["customModels"]["zero/model"]
        self.assertEqual(zm["maxTokens"], 131072)
        self.assertEqual(zm["maxOutputTokens"], 8192)

        um = data["customModels"]["unknown/model"]
        self.assertEqual(um["maxTokens"], 131072)
        self.assertEqual(um["maxOutputTokens"], 8192)

    def test_sync_jetski_state_preserves_surrounding_protobuf_fields(self):
        initial_pbtxt = (
            'post_onboarding: {\n  completed_steps: POST_ONBOARDING_STEP_TYPE_MANAGER_WELCOME\n}\n'
            'installation_uuid: "uuid-1234"\n'
        )
        with open(self.state_path, "w", encoding="utf-8") as fh:
            fh.write(initial_pbtxt)

        catalog = [{"id": "model/alpha", "context": 128000, "max_output": 4096}]
        n.sync_jetski_state(catalog, ["model/alpha"], state_path=self.state_path)

        with open(self.state_path, "r", encoding="utf-8") as fh:
            pbtxt = fh.read()

        self.assertIn('installation_uuid: "uuid-1234"', pbtxt)
        self.assertIn("POST_ONBOARDING_STEP_TYPE_MANAGER_WELCOME", pbtxt)
        self.assertIn("custom_models_config: {", pbtxt)
        self.assertIn('key: "model/alpha"', pbtxt)

        # Idempotent update: add model/beta
        catalog.append({"id": "model/beta", "context": 64000, "max_output": 2048})
        n.sync_jetski_state(catalog, ["model/alpha", "model/beta"], state_path=self.state_path)

        with open(self.state_path, "r", encoding="utf-8") as fh:
            pbtxt2 = fh.read()

        self.assertIn('installation_uuid: "uuid-1234"', pbtxt2)
        self.assertEqual(pbtxt2.count("custom_models_config: {"), 1)
        self.assertIn('key: "model/alpha"', pbtxt2)
        self.assertIn('key: "model/beta"', pbtxt2)

    def test_safe_backup_creates_bak_file_on_sync(self):
        initial_settings = {"customModels": {"initial/model": {"maxTokens": 1000}}}
        with open(self.settings_path, "w", encoding="utf-8") as fh:
            json.dump(initial_settings, fh)

        initial_state = 'installation_uuid: "uuid-orig"\n'
        with open(self.state_path, "w", encoding="utf-8") as fh:
            fh.write(initial_state)

        catalog = [{"id": "new/model", "context": 128000, "max_output": 4096}]
        n.sync_custom_models(catalog, ["new/model"], settings_path=self.settings_path, state_path=self.state_path)

        settings_bak = self.settings_path.with_name(self.settings_path.name + ".bak")
        state_bak = self.state_path.with_name(self.state_path.name + ".bak")

        self.assertTrue(settings_bak.is_file())
        with open(settings_bak, "r", encoding="utf-8") as fh:
            bak_data = json.load(fh)
        self.assertIn("initial/model", bak_data["customModels"])
        self.assertNotIn("new/model", bak_data["customModels"])

        self.assertTrue(state_bak.is_file())
        with open(state_bak, "r", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), initial_state)

    def test_validate_custom_models_configured(self):
        # 1. Non-existent settings file reports all models missing
        missing = n.validate_custom_models_configured(["model/a", "model/b"], settings_path=self.settings_path, state_path=self.state_path)
        self.assertEqual(missing, ["model/a", "model/b"])

        # 2. Configured models with positive maxTokens
        settings_data = {
            "customModels": {
                "model/a": {"maxTokens": 131072, "maxOutputTokens": 8192},
                "model/zero": {"maxTokens": 0, "maxOutputTokens": 8192},
            }
        }
        with open(self.settings_path, "w", encoding="utf-8") as fh:
            json.dump(settings_data, fh)

        # When state_path doesn't exist yet, only settings.json is checked
        self.assertEqual(n.validate_custom_models_configured(["model/a"], settings_path=self.settings_path, state_path=self.state_path), [])
        self.assertEqual(n.validate_custom_models_configured(["model/zero"], settings_path=self.settings_path, state_path=self.state_path), ["model/zero"])
        self.assertEqual(n.validate_custom_models_configured(["model/a", "model/b"], settings_path=self.settings_path, state_path=self.state_path), ["model/b"])

        # 3. When state_path exists, it must also contain protobuf entries
        with open(self.state_path, "w", encoding="utf-8") as fh:
            fh.write('custom_models_config: {\n  custom_models: {\n    key: "model/a"\n  }\n}\n')

        self.assertEqual(n.validate_custom_models_configured(["model/a"], settings_path=self.settings_path, state_path=self.state_path), [])
        # model/c not in state
        settings_data["customModels"]["model/c"] = {"maxTokens": 131072, "maxOutputTokens": 8192}
        with open(self.settings_path, "w", encoding="utf-8") as fh:
            json.dump(settings_data, fh)
        self.assertEqual(n.validate_custom_models_configured(["model/c"], settings_path=self.settings_path, state_path=self.state_path), ["model/c"])

    def test_cmd_sync_settings_check_and_dry_run(self):
        settings_data = {
            "customModels": {
                "model/ok": {"maxTokens": 131072, "maxOutputTokens": 8192},
            }
        }
        with open(self.settings_path, "w", encoding="utf-8") as fh:
            json.dump(settings_data, fh)

        with mock.patch.object(n, "AGY_SETTINGS_PATH", self.settings_path), \
             mock.patch.object(n, "AGY_STATE_PATH", self.state_path), \
             mock.patch.object(n, "load_catalog", return_value={"models": [{"id": "model/ok", "is_chat": True}]}):
            
            # --check passed
            args_check_ok = types.SimpleNamespace(
                models="model/ok", all_catalog=False, check=True, dry_run=False,
                refresh=False, offline=True
            )
            self.assertEqual(n.cmd_sync_settings(args_check_ok), 0)

            # --check failed
            args_check_fail = types.SimpleNamespace(
                models="model/missing", all_catalog=False, check=True, dry_run=False,
                refresh=False, offline=True
            )
            self.assertEqual(n.cmd_sync_settings(args_check_fail), 1)

            # --dry-run returns 0 and does not write
            args_dry = types.SimpleNamespace(
                models="model/missing", all_catalog=False, check=False, dry_run=True,
                refresh=False, offline=True
            )
            self.assertEqual(n.cmd_sync_settings(args_dry), 0)
            with open(self.settings_path, "r", encoding="utf-8") as fh:
                curr = json.load(fh)
            self.assertNotIn("model/missing", curr["customModels"])


class TestLauncherSettingsSync(unittest.TestCase):
    def test_launcher_forwarder_and_runtime_settings_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            settings_file = tmp_path / "settings.json"
            state_file = tmp_path / "jetski_state.pbtxt"

            # Create state file with some initial content
            state_file.write_text('installation_uuid: "uuid-test"\n', encoding="utf-8")

            catalog = {
                "models": [
                    {"id": "Qwen/Qwen3.8-27B", "context": 262144, "max_output": 32768},
                ]
            }
            plan = n.plan_slots(["Qwen/Qwen3.8-27B"], {"pro": "", "flash": "", "flash_lite": ""}, cap=32)
            script = n.render_launcher("test", "test", "https://cloud-api.near.ai/v1", 3600, 32, plan, "Qwen/Qwen3.8-27B")
            self.assertIn("nearai_agy.py\" run test", script)
            self.assertFalse(hasattr(n, "RESOLVER"))

            synced = n.sync_custom_models(catalog, ["Qwen/Qwen3.8-27B"], settings_path=settings_file, state_path=state_file)
            self.assertEqual(synced, 1)

            self.assertTrue(settings_file.is_file())
            with open(settings_file, "r", encoding="utf-8") as fh:
                sdata = json.load(fh)
            self.assertIsInstance(sdata.get("customModels"), dict)
            self.assertIn("Qwen/Qwen3.8-27B", sdata["customModels"])
            self.assertEqual(sdata["customModels"]["Qwen/Qwen3.8-27B"]["maxTokens"], 262144)

            # Check state_file was updated while preserving existing fields
            pbtxt = state_file.read_text(encoding="utf-8")
            self.assertIn('installation_uuid: "uuid-test"', pbtxt)
            self.assertIn("custom_models_config: {", pbtxt)
            self.assertIn('key: "Qwen/Qwen3.8-27B"', pbtxt)
            self.assertIn("max_tokens: 262144", pbtxt)


class TestGatewayProxy(unittest.TestCase):
    def test_gateway_proxy_auth_and_parameter_stripping(self):
        import urllib.request
        received_requests = []

        class MockUpstream(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                received_requests.append(("GET", self.path, self.headers.get("Authorization"), None))
                payload = json.dumps({"models": [{"id": "test-m"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                received_requests.append(("POST", self.path, self.headers.get("Authorization"), body))
                if self.path.endswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b"data: chunk1\n\n")
                    self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        upstream = ThreadedHTTPServer(("127.0.0.1", 0), MockUpstream)
        u_port = upstream.server_address[1]
        u_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        u_thread.start()

        upstream_url = f"http://127.0.0.1:{u_port}/v1"
        real_key = "sk-test-real-key-12345"

        try:
            with n.GatewayProxy(upstream_url, real_key) as proxy:
                proxy_base = f"http://127.0.0.1:{proxy.port}"

                # 1. Test unauthorized request (no token or invalid token -> 403)
                req_unauth = urllib.request.Request(f"{proxy_base}/v1/models")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req_unauth)
                self.assertEqual(ctx.exception.code, 403)

                # 2. Test authorized GET request
                auth_header = {"Authorization": f"Bearer {proxy.session_token}"}
                req_get = urllib.request.Request(f"{proxy_base}/v1/models", headers=auth_header)
                with urllib.request.urlopen(req_get) as resp:
                    self.assertEqual(resp.status, 200)
                    data = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(data["models"][0]["id"], "test-m")
                self.assertEqual(received_requests[-1][2], f"Bearer {real_key}")

                # 3. Test POST parameter stripping for anthropic/*
                post_body_anthropic = {
                    "model": "anthropic/claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "medium",
                }
                req_post1 = urllib.request.Request(
                    f"{proxy_base}/v1/chat/completions",
                    data=json.dumps(post_body_anthropic).encode("utf-8"),
                    headers=dict(auth_header, **{"Content-Type": "application/json"}),
                    method="POST",
                )
                with urllib.request.urlopen(req_post1) as resp:
                    self.assertEqual(resp.status, 200)
                last_body = received_requests[-1][3]
                self.assertNotIn("reasoning_effort", last_body)
                self.assertEqual(last_body["model"], "anthropic/claude-sonnet-4-5")

                # 4. Test POST non-anthropic preserves reasoning_effort
                post_body_google = {
                    "model": "google/gemini-3.1-pro-preview",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "medium",
                }
                req_post2 = urllib.request.Request(
                    f"{proxy_base}/v1/chat/completions",
                    data=json.dumps(post_body_google).encode("utf-8"),
                    headers=dict(auth_header, **{"Content-Type": "application/json"}),
                    method="POST",
                )
                with urllib.request.urlopen(req_post2) as resp:
                    self.assertEqual(resp.status, 200)
                last_body2 = received_requests[-1][3]
                self.assertEqual(last_body2.get("reasoning_effort"), "medium")

                # 5. Test streaming response
                req_stream = urllib.request.Request(
                    f"{proxy_base}/v1/stream",
                    data=json.dumps({"model": "test"}).encode("utf-8"),
                    headers=dict(auth_header, **{"Content-Type": "application/json"}),
                    method="POST",
                )
                with urllib.request.urlopen(req_stream) as resp:
                    self.assertEqual(resp.status, 200)
                    stream_data = resp.read().decode("utf-8")
                    self.assertIn("data: chunk1", stream_data)
                    self.assertIn("data: [DONE]", stream_data)
        finally:
            upstream.shutdown()
            upstream.server_close()


class TestCmdRun(unittest.TestCase):
    def test_cmd_run_validation_and_resolution(self):
        # 1. Missing profile exits with 1
        args_missing = types.SimpleNamespace(profile="non_existent_profile_xyz_123", agy_args=[])
        rc = n.cmd_run(args_missing)
        self.assertEqual(rc, 1)

        # 2. Test resolve_profile_models on qwen profile
        profile = n._load_profile("qwen")
        mock_catalog = {
            "models": [
                {
                    "id": "Qwen/Qwen3.8-27B",
                    "family": "qwen",
                    "inputs": ["text"],
                    "outputs": ["text"],
                    "is_chat": True,
                    "has_tools": True,
                    "context": 262144,
                    "max_output": 32768,
                },
                {
                    "id": "qwen/qwen3.7-max",
                    "family": "qwen",
                    "inputs": ["text"],
                    "outputs": ["text"],
                    "is_chat": True,
                    "has_tools": True,
                    "context": 262144,
                    "max_output": 32768,
                },
            ]
        }
        cfg = {"model_cap": 32}
        plan, default_model, strip_effort, _ = n.resolve_profile_models(profile, mock_catalog, cfg)
        self.assertIn("Qwen/Qwen3.8-27B", plan["models"])
        self.assertEqual(default_model, "Qwen/Qwen3.8-27B")
        self.assertEqual(plan["tiers"]["pro"], "Qwen/Qwen3.8-27B")


class TestToolCallingVerification(unittest.TestCase):
    def test_validate_tool_response_success(self):
        valid_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/workspace/config.json"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        ok, err = n.validate_tool_response(valid_response)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_validate_tool_response_failures(self):
        # Plain text without tool calls
        plain_response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "I cannot read files."},
                    "finish_reason": "stop",
                }
            ]
        }
        ok, err = n.validate_tool_response(plain_response)
        self.assertFalse(ok)
        self.assertIn("no tool_calls emitted", err)

        # Invalid JSON in arguments
        bad_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{bad-json:",
                                }
                            }
                        ],
                    }
                }
            ]
        }
        ok, err = n.validate_tool_response(bad_json)
        self.assertFalse(ok)
        self.assertIn("invalid JSON arguments", err)

        # Missing path argument
        missing_path = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"other_key": "val"}',
                                }
                            }
                        ],
                    }
                }
            ]
        }
        ok, err = n.validate_tool_response(missing_path)
        self.assertFalse(ok)
        self.assertIn("missing 'path' argument", err)

    def test_probe_one_validate_tools_against_mock(self):
        class MockToolUpstream(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                model = body.get("model", "")
                if model == "model/tool-success":
                    resp_data = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path": "/workspace/config.json"}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                    status = 200
                elif model == "model/tool-fail":
                    resp_data = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "I do not use tools.",
                                }
                            }
                        ]
                    }
                    status = 200
                else:
                    resp_data = {"error": {"message": "tools unsupported"}}
                    status = 400

                payload = json.dumps(resp_data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        srv = ThreadedServer(("127.0.0.1", 0), MockToolUpstream)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            gw = f"http://127.0.0.1:{port}/v1"
            res_success = n.probe_one(gw, "key", "model/tool-success", validate_tools=True)
            self.assertTrue(res_success["ok"])
            self.assertTrue(res_success["tool_call_ok"])

            res_fail = n.probe_one(gw, "key", "model/tool-fail", validate_tools=True)
            self.assertTrue(res_fail["ok"])
            self.assertFalse(res_fail["tool_call_ok"])
            self.assertIn("no tool_calls emitted", res_fail["tool_call_error"])

            res_error = n.probe_one(gw, "key", "model/tool-error", validate_tools=True)
            self.assertFalse(res_error["ok"])
            self.assertFalse(res_error["tool_call_ok"])
        finally:
            srv.shutdown()
            srv.server_close()


class TestSystemsTest(unittest.TestCase):
    def test_systems_test_requires_api_key(self):
        args = types.SimpleNamespace(
            refresh=False, offline=True, profile=None, models="test/model",
            all_catalog=False, skip_tools=False, workers=1, json_path=None,
            update_cache=False, limit=None
        )
        with mock.patch.object(n, "resolve_api_key", return_value=""):
            rc = n.cmd_systems_test(args)
            self.assertEqual(rc, 1)

    def test_systems_test_end_to_end_mock(self):
        class MockSystemsGateway(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                model = body.get("model", "")

                if model == "agent/model":
                    if "tools" in body:
                        resp_data = {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "id": "c1",
                                                "function": {
                                                    "name": "read_file",
                                                    "arguments": '{"path": "file.txt"}',
                                                },
                                            }
                                        ],
                                    }
                                }
                            ]
                        }
                    else:
                        resp_data = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
                    code = 200
                elif model == "chat/model":
                    resp_data = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
                    code = 200
                else:
                    resp_data = {"error": {"message": "incompatible model"}}
                    code = 400

                payload = json.dumps(resp_data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        srv = ThreadedServer(("127.0.0.1", 0), MockSystemsGateway)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_report = Path(tmpdir) / "report.json"
            cfg = {"gateway_url": f"http://127.0.0.1:{port}/v1", "api_key": "test-key"}
            mock_catalog = {
                "models": [
                    {"id": "agent/model", "inputs": ["text"], "outputs": ["text"], "context": 128000},
                    {"id": "chat/model", "inputs": ["text"], "outputs": ["text"], "context": 64000},
                    {"id": "error/model", "inputs": ["text"], "outputs": ["text"], "context": 32000},
                ]
            }

            args = types.SimpleNamespace(
                refresh=False, offline=False, profile=None, models="agent/model,chat/model,error/model",
                all_catalog=False, skip_tools=False, workers=2, json_path=str(json_report),
                update_cache=False, limit=None
            )

            try:
                with mock.patch.object(n, "load_config", return_value=cfg), \
                     mock.patch.object(n, "resolve_api_key", return_value="test-key"), \
                     mock.patch.object(n, "load_catalog", return_value=mock_catalog):
                    rc = n.cmd_systems_test(args)
                    self.assertEqual(rc, 0)

                self.assertTrue(json_report.is_file())
                with open(json_report, "r", encoding="utf-8") as fh:
                    report = json.load(fh)

                self.assertEqual(report["summary"]["agent_ready"], 1)
                self.assertEqual(report["summary"]["chat_only"], 1)
                self.assertEqual(report["summary"]["incompatible"], 1)
                self.assertEqual(report["models"]["agent/model"]["status"], "AGENT_READY")
                self.assertEqual(report["models"]["chat/model"]["status"], "CHAT_ONLY")
                self.assertEqual(report["models"]["error/model"]["status"], "INCOMPATIBLE")
            finally:
                srv.shutdown()
                srv.server_close()


class TestAgyProPlugin(unittest.TestCase):
    def test_render_subagent_md_and_manifest(self):
        spec = {
            "model": "Qwen/Qwen3.8-27B",
            "description": "Hardware-attested TEE specialist for confidential code auditing",
            "tee": True,
            "persona": "NEAR AI Private TEE Specialist",
            "instructions": "Follow zero-trust directives.",
        }
        md = n.render_subagent_md("nearai-private", spec)
        self.assertIn("name: nearai-private", md)
        self.assertIn("description: \"Hardware-attested TEE specialist for confidential code auditing\"", md)
        self.assertIn("subagent: true", md)
        self.assertIn("run_command", md)
        # Ensure legacy and modern file-tool aliases are not both declared (causes executor tool name conflict)
        self.assertIn("  - view_file\n", md)
        self.assertNotIn("  - read_file\n", md)
        self.assertNotIn("  - edit_file\n", md)
        self.assertNotIn("  - write_file\n", md)
        self.assertIn("# NEAR AI Private TEE Specialist", md)
        self.assertIn("Follow zero-trust directives.", md)

        manifest_str = n.render_plugin_json()
        manifest = json.loads(manifest_str)
        self.assertEqual(manifest["name"], "nearai-pro")
        self.assertIn("keywords", manifest)
        self.assertIn("tee", manifest["keywords"])

    def test_pro_setup_dry_run_and_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "plugin"
            mock_launcher_dir = Path(tmpdir) / "bin"
            settings_path = Path(tmpdir) / "settings.json"
            state_path = Path(tmpdir) / "jetski_state.pbtxt"
            mock_catalog = {
                "models": [
                    {"id": "Qwen/Qwen3.8-27B", "context": 262144, "max_output": 8192},
                    {"id": "anthropic/claude-sonnet-4-5", "context": 200000, "max_output": 8192},
                ]
            }

            # 1. Test Dry Run
            args_dry = types.SimpleNamespace(
                workspace=False, path=str(plugin_dir), dry_run=True, force=False,
                models=None, offline=True
            )
            with mock.patch.object(n, "load_config", return_value={}), \
                 mock.patch.object(n, "load_catalog", return_value=mock_catalog), \
                 mock.patch.object(n, "LAUNCHER_DIR", mock_launcher_dir):
                rc = n.cmd_pro_setup(args_dry)
                self.assertEqual(rc, 0)
                self.assertFalse(plugin_dir.exists())

            # 2. Test Live Setup
            args_live = types.SimpleNamespace(
                workspace=False, path=str(plugin_dir), dry_run=False, force=False,
                models=None, offline=True
            )
            with mock.patch.object(n, "load_config", return_value={}), \
                 mock.patch.object(n, "load_catalog", return_value=mock_catalog), \
                 mock.patch.object(n, "AGY_SETTINGS_PATH", settings_path), \
                 mock.patch.object(n, "AGY_STATE_PATH", state_path), \
                 mock.patch.object(n, "LAUNCHER_DIR", mock_launcher_dir):
                rc = n.cmd_pro_setup(args_live)
                self.assertEqual(rc, 0)

            self.assertTrue((plugin_dir / "plugin.json").is_file())
            self.assertTrue((plugin_dir / "agents" / "nearai-private.md").is_file())
            self.assertTrue((plugin_dir / "agents" / "nearai-claude.md").is_file())
            self.assertTrue((plugin_dir / "rules" / "AGENTS.md").is_file())
            self.assertTrue((plugin_dir / "skills" / "nearai-delegation" / "SKILL.md").is_file())
            self.assertTrue((plugin_dir / "skills" / "nearai-audit" / "SKILL.md").is_file())
            self.assertTrue((plugin_dir / "skills" / "nearai-redteam" / "SKILL.md").is_file())
            self.assertTrue((plugin_dir / "skills" / "nearai-refactor" / "SKILL.md").is_file())
            self.assertTrue((plugin_dir / "skills" / "nearai-optimize" / "SKILL.md").is_file())
            self.assertTrue((plugin_dir / "bin" / "nearai-agent-runner").is_file())
            self.assertTrue((mock_launcher_dir / "nearai-agent-runner").is_file())

            # Check that settings.json was populated
            self.assertTrue(settings_path.is_file())
            with open(settings_path, "r", encoding="utf-8") as fh:
                sdata = json.load(fh)
            self.assertIn("Qwen/Qwen3.8-27B", sdata.get("customModels", {}))
            self.assertGreater(sdata["customModels"]["Qwen/Qwen3.8-27B"]["maxTokens"], 0)

            # 3. Test List
            args_list = types.SimpleNamespace(workspace=False, path=str(plugin_dir))
            with mock.patch.object(n, "AGY_SETTINGS_PATH", settings_path):
                rc_list = n.cmd_pro_list(args_list)
                self.assertEqual(rc_list, 0)

            # 4. Test Uninstall Dry-Run
            args_uninst_dry = types.SimpleNamespace(workspace=False, path=str(plugin_dir), dry_run=True)
            with mock.patch.object(n, "LAUNCHER_DIR", mock_launcher_dir):
                rc_uninst_dry = n.cmd_pro_uninstall(args_uninst_dry)
                self.assertEqual(rc_uninst_dry, 0)
                self.assertTrue(plugin_dir.exists())

            # 5. Test Uninstall Live
            args_uninst_live = types.SimpleNamespace(workspace=False, path=str(plugin_dir), dry_run=False)
            with mock.patch.object(n, "LAUNCHER_DIR", mock_launcher_dir):
                rc_uninst = n.cmd_pro_uninstall(args_uninst_live)
                self.assertEqual(rc_uninst, 0)
                self.assertFalse(plugin_dir.exists())
                self.assertFalse((mock_launcher_dir / "nearai-agent-runner").exists())

    def test_pro_exec_mock_gateway(self):
        class MockBridgeGateway(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_len).decode("utf-8")
                req_data = json.loads(raw_body)

                auth = self.headers.get("Authorization", "")
                if auth != "Bearer secret-key":
                    self.send_response(401)
                    self.end_headers()
                    return

                if req_data.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunks = [
                        'data: {"choices":[{"delta":{"content":"Mocked "}}]}\n\n',
                        'data: {"choices":[{"delta":{"content":"streaming "}}]}\n\n',
                        'data: {"choices":[{"delta":{"content":"success"}}]}\n\n',
                        'data: [DONE]\n\n'
                    ]
                    for chunk in chunks:
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp_data = {
                        "id": "mock-id",
                        "choices": [{
                            "message": {"role": "assistant", "content": "Mocked non-stream output"}
                        }]
                    }
                    self.wfile.write(json.dumps(resp_data).encode("utf-8"))

        class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        srv = ThreadedServer(("127.0.0.1", 0), MockBridgeGateway)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        cfg = {"gateway_url": f"http://127.0.0.1:{port}/v1", "api_key": "secret-key"}

        try:
            # 1. Test streaming exec
            args_stream = types.SimpleNamespace(
                model="Qwen/Qwen3.8-27B", prompt="Test prompt", input_file=None,
                system="System prompt", json_out=False, stream=True, timeout=10
            )
            with mock.patch.object(n, "load_config", return_value=cfg), \
                 mock.patch.object(n, "resolve_api_key", return_value="secret-key"), \
                 mock.patch("sys.stdout.write") as mock_stdout:
                rc = n.cmd_pro_exec(args_stream)
                self.assertEqual(rc, 0)
                written = "".join(call[0][0] for call in mock_stdout.call_args_list)
                self.assertIn("Mocked streaming success", written)

            # 2. Test non-streaming json exec
            args_json = types.SimpleNamespace(
                model="Qwen/Qwen3.8-27B", prompt="Test json", input_file=None,
                system=None, json_out=True, stream=False, timeout=10
            )
            with mock.patch.object(n, "load_config", return_value=cfg), \
                 mock.patch.object(n, "resolve_api_key", return_value="secret-key"), \
                 mock.patch("builtins.print") as mock_print:
                rc = n.cmd_pro_exec(args_json)
                self.assertEqual(rc, 0)
                output = mock_print.call_args[0][0]
                parsed = json.loads(output)
                self.assertEqual(parsed["id"], "mock-id")

        finally:
            srv.shutdown()
            srv.server_close()

    def test_pro_add_remove_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "nearai_plugin"
            ledger_file = Path(tmpdir) / "test_ledger.jsonl"
            mock_catalog = {
                "models": [{
                    "id": "moonshotai/kimi-k3",
                    "context": 262144,
                    "max_output": 8192,
                    "verifiable": True,
                    "prompt_price": "0.000003",
                    "completion_price": "0.000015",
                }]
            }

            # 1. Test pro add
            args_add = types.SimpleNamespace(
                name="nearai-kimi", model="moonshotai/kimi-k3", role="Kimi Specialist",
                description="Long context research", workspace=False, path=str(plugin_dir)
            )
            with mock.patch.object(n, "load_catalog", return_value=mock_catalog), \
                 mock.patch.object(n, "sync_custom_models", return_value=1):
                rc = n.cmd_pro_add(args_add)
                self.assertEqual(rc, 0)
                spec_file = plugin_dir / "agents" / "nearai-kimi.md"
                self.assertTrue(spec_file.is_file())
                content = spec_file.read_text(encoding="utf-8")
                self.assertIn("moonshotai/kimi-k3", content)
                self.assertIn("Kimi Specialist", content)

            # 2. Test pro remove
            args_rm = types.SimpleNamespace(
                name="nearai-kimi", workspace=False, path=str(plugin_dir)
            )
            rc = n.cmd_pro_remove(args_rm)
            self.assertEqual(rc, 0)
            self.assertFalse(spec_file.exists())

            # 3. Test ledger recording and display
            with mock.patch.object(n, "LEDGER_PATH", ledger_file), \
                 mock.patch.object(n, "load_catalog", return_value=mock_catalog):
                entry = n.record_ledger_entry("moonshotai/kimi-k3", 100, 50, caller="unit test")
                self.assertTrue(ledger_file.is_file())
                self.assertAlmostEqual(entry["cost_usd"], 0.00105)

                # Test pro ledger display
                args_led = types.SimpleNamespace(json_out=False, clear=False)
                with mock.patch("builtins.print") as mock_print:
                    rc_led = n.cmd_pro_ledger(args_led)
                    self.assertEqual(rc_led, 0)

                # Test pro ledger json
                args_led_json = types.SimpleNamespace(json_out=True, clear=False)
                with mock.patch("builtins.print") as mock_print:
                    rc_led_json = n.cmd_pro_ledger(args_led_json)
                    self.assertEqual(rc_led_json, 0)
                    out = mock_print.call_args[0][0]
                    records = json.loads(out)
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["model"], "moonshotai/kimi-k3")

                # Test pro ledger clear
                args_led_clear = types.SimpleNamespace(json_out=False, clear=True)
                rc_clear = n.cmd_pro_ledger(args_led_clear)
                self.assertEqual(rc_clear, 0)
                self.assertFalse(ledger_file.exists())


class TestAgyProBenchmarkAndRecommend(unittest.TestCase):
    def test_benchmark_evaluators_and_cost(self):
        # 1. JSON Schema evaluation
        valid_json = '```json\n{"version": "1.0", "highlights": ["speed"], "breaking_change": false}\n```'
        invalid_json = '{"version": "1.0"}'
        self.assertTrue(n._evaluate_json_schema(valid_json, ["version", "highlights", "breaking_change"]))
        self.assertFalse(n._evaluate_json_schema(invalid_json, ["version", "highlights", "breaking_change"]))
        self.assertFalse(n._evaluate_json_schema(None, ["version"]))

        # 2. Cost calculation
        meta = {"prompt_price": "0.000001", "completion_price": "0.000002"}
        cost = n._calc_task_cost(100, 50, meta)
        self.assertAlmostEqual(cost, 0.0002)

        # 3. Domain Evaluators
        sec_audit_pass = "Vulnerable to timing attacks. Use hmac.compare_digest(user_token, expected_token)."
        sec_audit_fail = "The function has a syntax error in the loop."
        self.assertTrue(n.BENCHMARK_TASKS["sec_audit"]["evaluator"](sec_audit_pass))
        self.assertFalse(n.BENCHMARK_TASKS["sec_audit"]["evaluator"](sec_audit_fail))

        algo_pass = "Optimize to O(N) time and O(1) space with iterative loop."
        algo_fail = "Keep recursion with memoization."
        self.assertTrue(n.BENCHMARK_TASKS["algo_math"]["evaluator"](algo_pass))
        self.assertFalse(n.BENCHMARK_TASKS["algo_math"]["evaluator"](algo_fail))

    def test_pro_recommend_scoring(self):
        # Test privacy routing
        with mock.patch("builtins.print") as mock_print:
            args = types.SimpleNamespace(query="audit our private RSA key and credentials")
            rc = n.cmd_pro_recommend(args)
            self.assertEqual(rc, 0)
            printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("PRIVACY", printed)
            self.assertIn("nearai-private", printed)

        # Test security routing
        with mock.patch("builtins.print") as mock_print:
            args = types.SimpleNamespace(query="craft format string exploit payload for fuzz testing")
            rc = n.cmd_pro_recommend(args)
            self.assertEqual(rc, 0)
            printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("SECURITY", printed)
            self.assertIn("nearai-uncensored", printed)

        # Test refactor routing
        with mock.patch("builtins.print") as mock_print:
            args = types.SimpleNamespace(query="refactor architecture to async modules")
            rc = n.cmd_pro_recommend(args)
            self.assertEqual(rc, 0)
            printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("REFACTOR", printed)
            self.assertIn("nearai-claude", printed)

        # Test json output
        with mock.patch("builtins.print") as mock_print:
            args = types.SimpleNamespace(query="optimize fibonacci recurrence", as_json=True)
            rc = n.cmd_pro_recommend(args)
            self.assertEqual(rc, 0)
            printed = "".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            parsed = json.loads(printed)
            self.assertEqual(parsed["subagent"], "nearai-deepseek")
            self.assertEqual(parsed["detected_domain"], "algo")

    def test_pro_benchmark_mock_gateway(self):
        class MockBenchmarkGateway(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_len).decode("utf-8")
                req_data = json.loads(raw_body)

                resp_data = {
                    "id": "bench-1",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Vulnerable to timing attacks. Use hmac.compare_digest(a, b)."
                        }
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20}
                }
                payload = json.dumps(resp_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        srv = ThreadedServer(("127.0.0.1", 0), MockBenchmarkGateway)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        cfg = {"gateway_url": f"http://127.0.0.1:{port}/v1", "api_key": "test-key"}
        mock_catalog = {
            "models": [{
                "id": "Qwen/Qwen3.8-27B", "prompt_price": "0.00000035", "completion_price": "0.0000015",
                "verifiable": True
            }]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_report = Path(tmpdir) / "bench.json"
            cache_file = Path(tmpdir) / "matrix.json"
            args = types.SimpleNamespace(
                models="Qwen/Qwen3.8-27B", tasks="sec_audit", json_path=str(json_report),
                timeout=10, offline=True
            )

            try:
                with mock.patch.object(n, "load_config", return_value=cfg), \
                     mock.patch.object(n, "resolve_api_key", return_value="test-key"), \
                     mock.patch.object(n, "load_catalog", return_value=mock_catalog), \
                     mock.patch.object(n, "BENCHMARK_CACHE_PATH", cache_file):
                    rc = n.cmd_pro_benchmark(args)
                    self.assertEqual(rc, 0)

                self.assertTrue(json_report.is_file())
                self.assertTrue(cache_file.is_file())
                with open(json_report, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.assertIn("Qwen/Qwen3.8-27B", data["models"])
                self.assertTrue(data["models"]["Qwen/Qwen3.8-27B"]["tasks"]["sec_audit"]["passed"])
            finally:
                srv.shutdown()
                srv.server_close()


class TestIdeAndDesktopTargets(unittest.TestCase):
    def test_launcher_filename_and_render_for_ide_and_app(self):
        self.assertEqual(n.launcher_filename("qwen", "cli"), "nearai-qwen")
        self.assertEqual(n.launcher_filename("qwen", "ide"), "nearai-ide-qwen")
        self.assertEqual(n.launcher_filename("qwen", "app"), "nearai-app-qwen")

        plan = n.plan_slots(["Qwen/Qwen3.8-27B"], {"pro": "Qwen/Qwen3.8-27B", "flash": "", "flash_lite": ""}, cap=32)
        for target in ("ide", "app"):
            script = n.render_launcher(
                name="qwen",
                profile_ref="qwen",
                url="https://cloud-api.near.ai/v1",
                ttl=3600,
                cap=32,
                plan=plan,
                default_model="Qwen/Qwen3.8-27B",
                target=target,
            )
            self.assertIn(f"run qwen --target {target} \"$@\"", script)
            self.assertIn(f"nearai-{target}-qwen", script)
            self.assertNotIn("/Users/", script)
            self.assertIsNone(re.search(r"sk-[A-Za-z0-9]{10,}", script))
            proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_build_target_command_ide_and_app(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(n, "CONFIG_DIR", Path(tmpdir)):
                # CLI injects --model
                cli_cmd = n.build_target_command(
                    "cli", "/usr/bin/agy", False, [], default_model="Qwen/Qwen3.8-27B"
                )
                self.assertEqual(cli_cmd, ["/usr/bin/agy", "--model", "Qwen/Qwen3.8-27B"])

                # IDE does NOT inject --model, adds --user-data-dir & --new-window when isolated
                ide_cmd = n.build_target_command(
                    "ide", "/Applications/Antigravity IDE.app/Contents/MacOS/Electron",
                    False, ["/tmp/repo"], default_model="Qwen/Qwen3.8-27B",
                    isolated=True, profile_name="qwen",
                )
                self.assertNotIn("--model", ide_cmd)
                self.assertIn("--new-window", ide_cmd)
                self.assertTrue(any(a.startswith("--user-data-dir=") for a in ide_cmd))
                self.assertEqual(ide_cmd[-1], "/tmp/repo")

                # CLI wrapper fallback for IDE appends --wait so proxy stays attached
                ide_wrap_cmd = n.build_target_command(
                    "ide", "/usr/local/bin/antigravity-ide", True, ["."], isolated=False
                )
                self.assertIn("--wait", ide_wrap_cmd)

                # Desktop App isolated mode passes --user-data-dir without --new-window
                app_cmd = n.build_target_command(
                    "app", "/Applications/Antigravity.app/Contents/MacOS/Antigravity",
                    False, [], isolated=True, profile_name="private",
                )
                self.assertTrue(any(a.startswith("--user-data-dir=") for a in app_cmd))
                self.assertNotIn("--new-window", app_cmd)

    def test_cmd_run_ide_already_running_points_to_isolated(self):
        args = types.SimpleNamespace(
            profile="qwen", target="ide", isolated=False, direct=False,
            refresh=False, offline=True, agy_args=[],
        )
        stderr_buf = []
        with mock.patch.object(n, "load_config", return_value={"gateway_url": "https://cloud-api.near.ai/v1"}), \
             mock.patch.object(n, "resolve_api_key", return_value="test-key"), \
             mock.patch.object(n, "resolve_target_executable", return_value=("/Applications/Antigravity IDE.app/Contents/MacOS/Electron", False)), \
             mock.patch.object(n, "is_gui_target_running", return_value=True), \
             mock.patch.object(sys.stderr, "write", side_effect=stderr_buf.append):
            rc = n.cmd_run(args)
        self.assertEqual(rc, 1)
        msg = "".join(stderr_buf)
        self.assertIn("already running", msg)
        self.assertIn("--isolated", msg)

    def test_cmd_run_ide_isolated_launches_with_proxy_and_env(self):
        mock_catalog = {
            "models": [n._normalize_model({
                "id": "Qwen/Qwen3.8-27B", "owned_by": "Qwen", "context_length": 262144,
                "supported_features": ["tools"], "is_ready": True,
                "architecture": {"inputModalities": ["text"], "outputModalities": ["text"]},
            }, {"provider_type": "vllm", "verifiable": True, "attestation_supported": True})]
        }
        captured = {}

        def fake_call(cmd, env=None):
            captured["cmd"] = cmd
            captured["env"] = env
            return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_file = Path(tmpdir) / "qwen.json"
            profile_file.write_text(json.dumps({"select": {"family": ["qwen"]}}), encoding="utf-8")
            args = types.SimpleNamespace(
                profile=str(profile_file), target="ide", isolated=False, direct=False,
                refresh=False, offline=True, agy_args=["--isolated", "/tmp/workspace"],
            )
            with mock.patch.object(n, "CONFIG_DIR", Path(tmpdir)), \
                 mock.patch.object(n, "load_config", return_value={"gateway_url": "https://cloud-api.near.ai/v1"}), \
                 mock.patch.object(n, "resolve_api_key", return_value="sk-secret-test"), \
                 mock.patch.object(n, "resolve_target_executable", return_value=("/Applications/Antigravity IDE.app/Contents/MacOS/Electron", False)), \
                 mock.patch.object(n, "is_gui_target_running", return_value=True), \
                 mock.patch.object(n, "load_catalog", return_value=mock_catalog), \
                 mock.patch.object(n, "validate_custom_models_configured", return_value=[]), \
                 mock.patch.object(subprocess, "call", side_effect=fake_call):
                rc = n.cmd_run(args)

        self.assertEqual(rc, 0)
        self.assertTrue(any(a.startswith("--user-data-dir=") for a in captured["cmd"]))
        self.assertEqual(captured["cmd"][-1], "/tmp/workspace")
        self.assertEqual(captured["env"]["AGY_LLM_GATEWAY_WIRE_PROTOCOL"], "openai")
        self.assertEqual(captured["env"]["AGY_LLM_GATEWAY_MODELS"], "Qwen/Qwen3.8-27B")
        self.assertTrue(captured["env"]["AGY_LLM_GATEWAY_URL"].startswith("http://127.0.0.1:"))
        self.assertNotIn("NEAR_AI_KEY", captured["env"])

    def test_default_subagents_have_clean_newlines(self):
        for name, spec in n.DEFAULT_SUBAGENTS.items():
            self.assertNotIn("\\n", spec["instructions"], msg=f"literal \\n found in {name}")
            self.assertIn("\n", spec["instructions"], msg=f"missing newlines in {name}")


class TestRemediationR1ToR5(unittest.TestCase):
    def test_r1_die_defined_and_deepseek_route_updated(self):
        import io
        self.assertEqual(n.DEFAULT_SUBAGENTS["nearai-deepseek"]["model"], "deepseek/deepseek-v4.1-flash-1017")
        fake_tok = "sk-" + "dummy" * 4
        stderr_buf = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                n.die(f"bad key Bearer {fake_tok}")
        self.assertEqual(ctx.exception.code, 1)
        self.assertNotIn(fake_tok, stderr_buf.getvalue())
        self.assertIn("[REDACTED]", stderr_buf.getvalue())

        # cmd_pro_exec missing API key exits via die() (SystemExit, never NameError)
        args = types.SimpleNamespace(
            model="Qwen/Qwen3.8-27B", prompt="hi", input_file=None,
            system=None, stream=False, json_out=False, timeout=5,
        )
        with mock.patch.object(n, "load_config", return_value={"gateway_url": "https://cloud-api.near.ai/v1"}), \
             mock.patch.object(n, "resolve_api_key", return_value=""), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx2:
                n.cmd_pro_exec(args)
        self.assertEqual(ctx2.exception.code, 1)

    def test_r2_gateway_url_key_validation_and_config_hygiene(self):
        import io
        for bad in (
            "http://localhost.evil.com/v1",
            "http://127.0.0.1.evil.com/v1",
            "http://localhost:pass@evil.com/v1",
            "https://user:pass@cloud-api.near.ai/v1",
            "https://cloud-api.near.ai/v1#frag",
            "https://cloud-api.near.ai/v1\r\nX-Test: 1",
        ):
            with self.assertRaises(ValueError, msg=f"should reject {bad!r}"):
                n._validate_gateway_url(bad)

        # Valid loopback + query string with '&' allowed
        self.assertEqual(
            n._validate_gateway_url("https://cloud-api.near.ai/v1?a=1&b=2"),
            "https://cloud-api.near.ai/v1?a=1&b=2",
        )
        self.assertEqual(n._validate_gateway_url("http://[::1]:8080/v1"), "http://[::1]:8080/v1")

        # API key control-char rejection
        with self.assertRaises(ValueError):
            n._validate_api_key("sk-abc\r\nInjected: 1")
        with self.assertRaises(ValueError):
            n._validate_api_key("sk-abc def")

        # save_config atomic 0600 + cmd_init does not silently persist env NEAR_AI_KEY
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_dir = Path(tmpdir) / "cfg"
            cfg_path = cfg_dir / "config.json"
            prof_dir = Path(tmpdir) / "profiles"
            bin_dir = Path(tmpdir) / "bin"
            with mock.patch.object(n, "CONFIG_DIR", cfg_dir), \
                 mock.patch.object(n, "CONFIG_PATH", cfg_path), \
                 mock.patch.object(n, "PROFILE_DIR", prof_dir), \
                 mock.patch.object(n, "LAUNCHER_DIR", bin_dir), \
                 mock.patch.dict(os.environ, {"NEAR_AI_KEY": "sk-ephemeral-env-key"}, clear=False), \
                 mock.patch.object(sys, "stdout", io.StringIO()), \
                 mock.patch.object(sys, "stderr", io.StringIO()):
                rc = n.cmd_init(types.SimpleNamespace(save_key=False))
                self.assertEqual(rc, 0)
                stored = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.assertEqual(stored["api_key"], "")
                self.assertEqual(os.stat(cfg_path).st_mode & 0o777, 0o600)

                # Explicit --save-key persists it
                rc2 = n.cmd_init(types.SimpleNamespace(save_key=True))
                self.assertEqual(rc2, 0)
                stored2 = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.assertEqual(stored2["api_key"], "sk-ephemeral-env-key")

        # cmd_verify_chat validates --gateway override
        with mock.patch.object(n, "load_config", return_value={"gateway_url": "https://cloud-api.near.ai/v1"}), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc_bad = n.cmd_verify_chat(types.SimpleNamespace(gateway="http://evil.example.com/v1", model="m"))
        self.assertEqual(rc_bad, 1)

    def test_r3_gateway_proxy_security_hardening(self):
        import http.client
        import socket

        # Non-loopback bind host rejected; IPv6 host bracketed
        with self.assertRaises(ValueError):
            n.GatewayProxy("https://cloud-api.near.ai/v1", "sk-key", host="0.0.0.0")
        with self.assertRaises(ValueError):
            n.GatewayProxy("http://localhost.evil.com/v1", "sk-key")
        p_v6 = n.GatewayProxy("https://cloud-api.near.ai/v1", "sk-key", host="::1")
        p_v6.port = 9999
        self.assertEqual(p_v6.gateway_url, "http://[::1]:9999/v1")

        seen = []

        class MockUpstream(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a):
                pass
            def do_HEAD(self):
                seen.append(("HEAD", self.path, None))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "42")
                self.end_headers()
            def do_GET(self):
                seen.append(("GET", self.path, None))
                if self.path.endswith("/redirect"):
                    self.send_response(302)
                    self.send_header("Location", "http://127.0.0.1:1/exfil")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                cl = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(cl).decode("utf-8"))
                seen.append(("POST", self.path, payload))
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
        u_port = upstream.server_address[1]
        u_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        u_thread.start()

        try:
            with n.GatewayProxy(f"http://127.0.0.1:{u_port}/v1", "sk-upstream-secret") as proxy:
                auth = {"Authorization": f"Bearer {proxy.session_token}"}

                # 1. Path traversal rejected (400)
                for bad_path in ("/v1/../admin", "/v1/%2e%2e/admin", "/v1/foo\\bar"):
                    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
                    conn.request("GET", bad_path, headers=auth)
                    self.assertEqual(conn.getresponse().status, 400, msg=bad_path)
                    conn.close()

                # 2. 302 redirect NOT followed (returned as 302 to client)
                conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
                conn.request("GET", "/v1/redirect", headers=auth)
                self.assertEqual(conn.getresponse().status, 302)
                conn.close()

                # 3. HEAD returns headers with empty body
                conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
                conn.request("HEAD", "/v1/models", headers=auth)
                r_head = conn.getresponse()
                self.assertEqual(r_head.status, 200)
                self.assertEqual(r_head.read(), b"")
                conn.close()

                # 4. Duplicate Authorization header rejected (403)
                with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as s:
                    s.sendall(
                        f"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{proxy.port}\r\n"
                        f"Authorization: Bearer {proxy.session_token}\r\n"
                        f"Authorization: Bearer {proxy.session_token}\r\n\r\n".encode("utf-8")
                    )
                    self.assertIn(b"403", s.recv(512))

                # 5. Transfer-Encoding chunked rejected (400)
                with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as s:
                    s.sendall(
                        f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{proxy.port}\r\n"
                        f"Authorization: Bearer {proxy.session_token}\r\n"
                        f"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n".encode("utf-8")
                    )
                    self.assertIn(b"400", s.recv(512))

                # 6. Malformed JSON, duplicate keys, NaN rejected (400)
                for bad_json in (b"{not-json", b'{"model":"a","model":"b"}', b'{"temp":NaN}', b'[1,2]'):
                    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
                    conn.request("POST", "/v1/chat/completions", body=bad_json, headers=auth)
                    self.assertEqual(conn.getresponse().status, 400, msg=bad_json)
                    conn.close()

                # 7. Oversized body returns 413
                with mock.patch.object(n, "MAX_PROXY_REQUEST_BODY", 32):
                    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
                    conn.request("POST", "/v1/chat/completions", body=b'{"model":"' + b"a" * 64 + b'"}', headers=auth)
                    self.assertEqual(conn.getresponse().status, 413)
                    conn.close()

                # 8. Case-insensitive nested reasoning_effort stripped for anthropic/*
                nested_payload = json.dumps({
                    "model": "anthropic/claude-sonnet-4-5",
                    "Reasoning_Effort": "high",
                    "extra_body": {"reasoning_effort": "medium", "keep": 1},
                }).encode("utf-8")
                conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
                conn.request("POST", "/v1/chat/completions", body=nested_payload, headers=auth)
                self.assertEqual(conn.getresponse().status, 200)
                conn.close()
                last_post = seen[-1][2]
                self.assertNotIn("Reasoning_Effort", last_post)
                self.assertNotIn("reasoning_effort", last_post["extra_body"])
                self.assertEqual(last_post["extra_body"]["keep"], 1)
        finally:
            upstream.shutdown()
            upstream.server_close()

    def test_r5_plan_slots_edge_cases_and_sync_jetski_state_lexer(self):
        # plan_slots edge cases: cap=0, cap<0, duplicates, tier evicted when late in selected_ids
        p0 = n.plan_slots(["a", "a", "b"], {"pro": "c", "flash": "c"}, cap=0)
        self.assertEqual(p0["models"], [])
        self.assertEqual(p0["tiers"], {"pro": "", "flash": "", "flash_lite": ""})
        self.assertEqual(p0["evicted_models"], ["a", "b"])
        self.assertEqual(p0["evicted_tiers"], ["c"])
        self.assertFalse(p0["fits"])

        p_neg = n.plan_slots([], {}, cap=-1)
        self.assertFalse(p_neg["fits"])

        p_dup = n.plan_slots(["m1", "m2", "m3"], {"pro": "m3", "flash": "f1"}, cap=2)
        self.assertEqual(p_dup["models"], ["m1"])
        self.assertEqual(p_dup["tiers"]["flash"], "f1")
        self.assertEqual(p_dup["tiers"]["pro"], "")

        # sync_jetski_state: braces in comments/strings, colon-free blocks, repeated blocks, escaping, bad numeric
        with tempfile.TemporaryDirectory() as tmpdir:
            pb = Path(tmpdir) / "jetski_state.pbtxt"
            pb.write_text(
                '# comment mentioning custom_models_config: { fake }\n'
                'other_field: "has { braces } and custom_models_config: {"\n'
                'custom_models_config {\n'
                '  custom_models {\n'
                '    value {\n'
                '      max_tokens: 64000\n'
                '      model_name: "legacy/\\"quoted\\""\n'
                '      max_output_tokens: 4096\n'
                '    }\n'
                '    key: "legacy/quoted"\n'
                '  }\n'
                '}\n'
                'custom_models_config: {\n'
                '  custom_models: {\n'
                '    key: "second/block"\n'
                '    value: { model_name: "second/block" max_tokens: 32000 max_output_tokens: 2048 }\n'
                '  }\n'
                '}\n'
                'tail_flag: true\n',
                encoding="utf-8",
            )
            cat = [{"id": "new/model", "context": "not-a-number", "max_output": -5}]
            count = n.sync_jetski_state(cat, ["new/model"], state_path=pb)
            self.assertEqual(count, 3)
            out = pb.read_text(encoding="utf-8")
            self.assertIn('# comment mentioning custom_models_config: { fake }', out)
            self.assertIn('other_field: "has { braces } and custom_models_config: {"', out)
            self.assertIn('tail_flag: true', out)
            self.assertEqual(len(re.findall(r"^custom_models_config:\s*\{", out, re.MULTILINE)), 1)
            self.assertIn('key: "legacy/quoted"', out)
            self.assertIn('model_name: "legacy/\\"quoted\\""', out)
            self.assertIn('key: "second/block"', out)
            self.assertIn('key: "new/model"', out)
            self.assertIn("max_tokens: 131072", out)
            self.assertIn("max_output_tokens: 8192", out)

    def test_pre_release_security_hardening(self):
        import io
        import urllib.error

        # 1. err() helper is defined and redacts credentials
        buf = io.StringIO()
        fake_tok = "sk-" + "secretval" * 3
        with mock.patch.object(sys, "stderr", buf):
            n.err(f"failed with Bearer {fake_tok}")
        self.assertNotIn(fake_tok, buf.getvalue())
        self.assertIn("[REDACTED]", buf.getvalue())

        # 2. Shell injection and path traversal rejected in launcher/profile/subagent names
        plan = n.plan_slots(["Qwen/Qwen3.8-27B"], {"pro": "", "flash": "", "flash_lite": ""}, cap=32)
        for bad_name in ("../evil", "foo;id", "$(whoami)", "a b", "bad/slash", "-dash"):
            with self.assertRaises(ValueError, msg=bad_name):
                n.launcher_filename(bad_name)
            with self.assertRaises(ValueError, msg=bad_name):
                n.render_launcher(bad_name, "qwen", "https://cloud-api.near.ai/v1", 3600, 32, plan, "Qwen/Qwen3.8-27B")
            with self.assertRaises(ValueError, msg=bad_name):
                n.render_launcher("qwen", bad_name, "https://cloud-api.near.ai/v1", 3600, 32, plan, "Qwen/Qwen3.8-27B")
            with self.assertRaises(SystemExit, msg=bad_name):
                n._load_profile(bad_name)

        # cmd_pro_add and cmd_pro_remove reject traversal names cleanly (no NameError, no escape)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(sys, "stderr", io.StringIO()):
                rc_add = n.cmd_pro_add(types.SimpleNamespace(
                    name="../../escape", model="Qwen/Qwen3.8-27B", role=None, description=None,
                    workspace=False, path=tmpdir,
                ))
                rc_rm = n.cmd_pro_remove(types.SimpleNamespace(
                    name="../../escape", workspace=False, path=tmpdir,
                ))
            self.assertEqual(rc_add, 1)
            self.assertEqual(rc_rm, 1)

            # 3. cmd_pro_uninstall refuses to rmtree a non-plugin directory
            non_plugin = Path(tmpdir) / "important_dir"
            non_plugin.mkdir()
            sentinel = non_plugin / "keep.txt"
            sentinel.write_text("do not delete", encoding="utf-8")
            with mock.patch.object(sys, "stdout", io.StringIO()), \
                 mock.patch.object(sys, "stderr", io.StringIO()):
                rc_uninst = n.cmd_pro_uninstall(types.SimpleNamespace(
                    workspace=False, path=str(non_plugin), dry_run=False,
                ))
            self.assertEqual(rc_uninst, 1)
            self.assertTrue(sentinel.is_file())

        # 4. Portable agent-runner script embeds no local home path
        portable = n.render_agent_runner_script(None)
        self.assertNotIn("/Users/", portable)
        self.assertIn("NEARAI_AGY_PY", portable)

        # 5. Authenticated HTTP helpers do NOT follow 302 redirects (prevent Authorization leak)
        exfil_hits = []

        class RedirectHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass
            def do_GET(self):
                if self.path.startswith("/exfil"):
                    exfil_hits.append(self.headers.get("Authorization"))
                    self.send_response(200)
                    self.end_headers()
                    return
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/exfil")
                self.send_header("Content-Length", "0")
                self.end_headers()
            def do_POST(self):
                cl = int(self.headers.get("Content-Length") or 0)
                if cl:
                    self.rfile.read(cl)
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/exfil")
                self.send_header("Content-Length", "0")
                self.end_headers()

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            gw = f"http://127.0.0.1:{port}/v1"
            with self.assertRaises(urllib.error.HTTPError):
                n._fetch_catalog(gw, "sk-fake-redirect-check")
            probe_res = n.probe_one(gw, "sk-fake-redirect-check", "Qwen/Qwen3.8-27B")
            self.assertEqual(probe_res["code"], 302)
            self.assertFalse(exfil_hits, "Authorization header leaked across HTTP redirect")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_cmd_run_enforces_attestation_required_with_allow_unverified_override(self):
        import io

        catalog = {
            "models": [
                {
                    "id": "Qwen/Qwen3.8-27B",
                    "family": "qwen",
                    "owned_by": "qwen",
                    "context": 262144,
                    "max_output": 32768,
                    "is_chat": True,
                    "has_tools": True,
                    "is_ready": True,
                    "provider_type": "vllm",
                    "provider_trust": "tee_vllm",
                    "attestable": True,
                    "verifiable": True,
                }
            ]
        }
        fail_gate = {
            "mode": "opt_in_attestation_route_check",
            "status": "fail_closed",
            "required": ["gateway_quote", "model_execution"],
            "failed": ["gateway_quote", "model_execution"],
            "verification_summary": {"overall": "unverified"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            prof_dir = Path(tmpdir) / "profiles"
            prof_dir.mkdir()
            (prof_dir / "private.json").write_text(
                json.dumps({
                    "select": {"tools_only": True, "tee_only": True},
                    "attestation_required": True,
                    "launch": {"model": "Qwen/Qwen3.8-27B"},
                }),
                encoding="utf-8",
            )

            # 1. Without --allow-unverified: fails closed before launching agy
            err_buf = io.StringIO()
            with mock.patch.object(n, "PROFILE_DIR", prof_dir), \
                 mock.patch.object(n, "load_config", return_value={"gateway_url": "https://cloud-api.near.ai/v1"}), \
                 mock.patch.object(n, "resolve_api_key", return_value="sk-fake-key"), \
                 mock.patch.object(n, "resolve_target_executable", return_value=("/usr/bin/agy", False)), \
                 mock.patch.object(n, "load_catalog", return_value=catalog), \
                 mock.patch.object(n, "_evaluate_attestation_route", return_value=fail_gate), \
                 mock.patch.object(n.subprocess, "call") as mock_call, \
                 mock.patch.object(sys, "stderr", err_buf):
                rc = n.cmd_run(types.SimpleNamespace(
                    profile="private", target="cli", isolated=False, direct=False,
                    allow_unverified=False, agy_args=[], refresh=False, offline=True,
                ))
            self.assertEqual(rc, 1)
            mock_call.assert_not_called()
            self.assertIn("--allow-unverified", err_buf.getvalue())
            self.assertIn("nearai-private-check", err_buf.getvalue())

            # 2. With --allow-unverified passed via forwarder agy_args: warns, strips flag, launches
            err_buf2 = io.StringIO()
            with mock.patch.object(n, "PROFILE_DIR", prof_dir), \
                 mock.patch.object(n, "load_config", return_value={"gateway_url": "https://cloud-api.near.ai/v1"}), \
                 mock.patch.object(n, "resolve_api_key", return_value="sk-fake-key"), \
                 mock.patch.object(n, "resolve_target_executable", return_value=("/usr/bin/agy", False)), \
                 mock.patch.object(n, "load_catalog", return_value=catalog), \
                 mock.patch.object(n, "validate_custom_models_configured", return_value=[]), \
                 mock.patch.object(n, "_evaluate_attestation_route") as mock_gate, \
                 mock.patch.object(n.subprocess, "call", return_value=0) as mock_call2, \
                 mock.patch.object(sys, "stderr", err_buf2):
                rc2 = n.cmd_run(types.SimpleNamespace(
                    profile="private", target="cli", isolated=False, direct=False,
                    allow_unverified=False, agy_args=["--allow-unverified", "models"],
                    refresh=False, offline=True,
                ))
            self.assertEqual(rc2, 0)
            mock_gate.assert_not_called()
            launched_cmd = mock_call2.call_args.args[0]
            self.assertNotIn("--allow-unverified", launched_cmd)
            self.assertIn("models", launched_cmd)
            self.assertIn("--allow-unverified", err_buf2.getvalue())


if __name__ == "__main__":
    unittest.main()
