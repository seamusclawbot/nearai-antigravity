"""Tests for cryptographic attestation and response-signature verifiers."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import jwt
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from eth_account import Account
    from eth_account.messages import encode_defunct
except ImportError as exc:
    raise unittest.SkipTest(f"optional verifier dependencies not installed: {exc}") from exc

from attestation_evidence import build_verification_summary, verify_ohttp_signature
from verifiers.nvidia import verify_chutes_gpu_evidence, verify_nvidia_gpu_evidence
from verifiers.ohttp import verify_compose_manager_actions
from verifiers.signatures import hash_exact_bytes, verify_response_signature
from verifiers.tdx import compute_mr_config_id, replay_rtmr3, verify_tdx_quote

DCAP_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dcap"


class TestResponseSignatures(unittest.TestCase):
    def test_ecdsa_eip191_provider_tee_and_gateway(self):
        acct = Account.create()
        req = b'{"model":"deepseek-ai/DeepSeek-V3","messages":[]}'
        resp = b'data: {"id":"1"}\n\ndata: [DONE]\n\n'
        req_h = hash_exact_bytes(req)
        resp_h = hash_exact_bytes(resp)

        text = f"deepseek-ai/DeepSeek-V3:{req_h}:{resp_h}"
        sig = acct.sign_message(encode_defunct(text=text)).signature.hex()
        payload = {
            "signature_kind": "provider_tee",
            "text": text,
            "signature": "0x" + sig,
            "signing_address": acct.address,
            "signing_algo": "ecdsa",
        }
        out = verify_response_signature(
            payload,
            req,
            resp,
            requested_model="deepseek-ai/DeepSeek-V3",
            model_attestations=[{"signing_address": acct.address, "signing_algo": "ecdsa"}],
        )
        self.assertTrue(out["verified"])
        self.assertEqual(out["provenance"], "provider_tee")

        # Ambiguous model_attestations (2 matches) must fail closed
        dup = verify_response_signature(
            payload,
            req,
            resp,
            requested_model="deepseek-ai/DeepSeek-V3",
            model_attestations=[
                {"signing_address": acct.address, "signing_algo": "ecdsa"},
                {"signing_address": acct.address, "signing_algo": "ecdsa"},
            ],
        )
        self.assertFalse(dup["verified"])
        self.assertEqual(dup["status"], "failed")

        # Non-canonicalized JSON bytes: changing whitespace must fail
        req_reformatted = b'{"model": "deepseek-ai/DeepSeek-V3", "messages": []}'
        self.assertFalse(verify_response_signature(payload, req_reformatted, resp)["verified"])

        # Gateway signature keeps gateway_only provenance
        gw_text = f"{req_h}:{resp_h}"
        gw_sig = acct.sign_message(encode_defunct(text=gw_text)).signature.hex()
        gw_payload = {
            "signature_kind": "gateway",
            "text": gw_text,
            "signature": gw_sig,
            "signing_address": acct.address,
            "signing_algo": "ecdsa",
        }
        gw_out = verify_response_signature(
            gw_payload,
            req,
            resp,
            gateway_attestation={"signing_address": acct.address, "signing_algo": "ecdsa"},
        )
        self.assertTrue(gw_out["verified"])
        self.assertEqual(gw_out["provenance"], "gateway_only")

    def test_ed25519_and_fail_closed_cases(self):
        priv = ed25519.Ed25519PrivateKey.generate()
        pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        req = b'{"prompt":"hi"}'
        resp = b'{"ok":true}'
        text = f"{hash_exact_bytes(req)}:{hash_exact_bytes(resp)}"
        sig_hex = priv.sign(text.encode("utf-8")).hex()
        payload = {
            "signature_kind": "gateway",
            "text": text,
            "signature": sig_hex,
            "signing_address": pub_hex,
            "signing_algo": "ed25519",
        }
        self.assertTrue(verify_response_signature(payload, req, resp)["verified"])

        # 404, error_code, missing signature_kind -> unavailable
        self.assertEqual(verify_response_signature(payload, req, resp, http_status=404)["status"], "unavailable")
        self.assertEqual(verify_response_signature({"error_code": "NOT_FOUND", "message": "x"}, req, resp)["status"], "unavailable")
        bad_kind = dict(payload)
        del bad_kind["signature_kind"]
        self.assertEqual(verify_response_signature(bad_kind, req, resp)["status"], "unavailable")


class TestOhttpAndComposeManager(unittest.TestCase):
    def test_ohttp_cross_checks_and_compose_manager(self):
        priv = ed25519.Ed25519PrivateKey.generate()
        pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        cfg = b"\x01\x00\x20" + b"\xaa" * 32
        sig = priv.sign(cfg).hex()
        att = {"signing_key": pub_hex, "key_config": cfg.hex(), "signature": sig}

        self.assertTrue(
            verify_ohttp_signature(
                att,
                gateway_attestation={"signing_address": pub_hex, "signing_algo": "ed25519"},
                ohttp_key_config=cfg.hex(),
            )["verified"]
        )
        self.assertFalse(verify_ohttp_signature(att, ohttp_key_config="deadbeef")["verified"])
        self.assertFalse(
            verify_ohttp_signature(att, gateway_attestation={"signing_address": "11" * 32, "signing_algo": "ed25519"})["verified"]
        )
        # Fail-closed when gateway_attestation lacks signing_address or uses non-ed25519 algo
        self.assertFalse(
            verify_ohttp_signature(att, gateway_attestation={"signing_algo": "ed25519"})["verified"]
        )
        self.assertFalse(
            verify_ohttp_signature(att, gateway_attestation={"signing_address": "0x" + "11" * 20, "signing_algo": "ecdsa"})["verified"]
        )

        # _verify_collected_crypto must bind model verifiers to client nonce & caller TLS policy
        from attestation_evidence import _verify_collected_crypto
        client_nonce = "aa" * 32
        server_fake_nonce = "bb" * 32
        report = {
            "model_attestations": [{
                "model": "m1",
                "intel_quote": "deadbeef",
                "nvidia_payload": {"arch": "HOPPER"},
                "request_nonce": server_fake_nonce,
                "nonce": server_fake_nonce,
            }]
        }
        with patch("verifiers.tdx.verify_tdx_quote", return_value={"status": "verified", "verified": True}) as mock_tdx, \
             patch("verifiers.nvidia.verify_nvidia_gpu_evidence", return_value={"status": "verified", "verified": True}) as mock_gpu:
            _verify_collected_crypto(report, nonce=client_nonce, include_tls_fingerprint=True)
            self.assertEqual(mock_tdx.call_args.kwargs["nonce"], client_nonce)
            self.assertTrue(mock_tdx.call_args.kwargs["include_tls_fingerprint"])
            self.assertEqual(mock_gpu.call_args.args[1], client_nonce)

        actions = [{"b": "ü", "a": 1}]
        expected = hashlib.sha256('[{"a":1,"b":"ü"}]'.encode("utf-8")).hexdigest()
        self.assertTrue(verify_compose_manager_actions(actions, expected_hash=expected)["verified"])
        self.assertFalse(verify_compose_manager_actions(actions, expected_hash="00" * 32)["verified"])


class TestIntelTdxVerifier(unittest.TestCase):
    def test_public_dcap_fixture_and_negative_cases(self):
        raw_quote = (DCAP_FIXTURES / "tdx_quote").read_bytes()
        collateral = (DCAP_FIXTURES / "tdx_quote_collateral.json").read_text()
        now_secs = int(datetime(2025, 6, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp())

        res = verify_tdx_quote(raw_quote, collateral=collateral, now_secs=now_secs)
        self.assertTrue(res["verified"])
        self.assertEqual(res["tcb_status"], "UpToDate")

        # Tampered quote body must fail ECDSA signature verification
        tampered = bytearray(raw_quote)
        tampered[100] ^= 0xFF
        self.assertEqual(verify_tdx_quote(bytes(tampered), collateral=collateral, now_secs=now_secs)["status"], "failed")

        # Unacceptable TCB policy must fail
        rejected_tcb = verify_tdx_quote(
            raw_quote,
            collateral=collateral,
            now_secs=now_secs,
            accepted_tcb_statuses={"CustomOnly"},
        )
        self.assertEqual(rejected_tcb["status"], "failed")

        # Wrong app_compose -> mr_config_id mismatch
        self.assertEqual(
            verify_tdx_quote(raw_quote, collateral=collateral, now_secs=now_secs, app_compose="{}")["reason"],
            "mr_config_id mismatch with app_compose",
        )

        # Wrong nonce -> report_data mismatch
        self.assertEqual(
            verify_tdx_quote(
                raw_quote,
                collateral=collateral,
                now_secs=now_secs,
                nonce="00" * 32,
                include_tls_fingerprint=False,
            )["status"],
            "failed",
        )

        # Debug bit set -> reject
        class FakeReport:
            td_attributes = b"\x01" + b"\x00" * 7
        class FakeParsed:
            report = FakeReport()
        with patch("dcap_qvl.parse_quote", return_value=FakeParsed()):
            dbg = verify_tdx_quote(raw_quote, collateral=collateral, now_secs=now_secs)
            self.assertEqual(dbg["status"], "failed")
            self.assertIn("debug", dbg["reason"].lower())

    def test_mr_config_id_and_rtmr3_replay(self):
        app = '{"docker_compose_file":"services: {}"}'
        mr = compute_mr_config_id(app)
        self.assertEqual(len(mr), 48)
        self.assertEqual(mr[0], 1)
        self.assertEqual(mr[1:33], hashlib.sha256(app.encode("utf-8")).digest())
        self.assertEqual(mr[33:], b"\x00" * 15)

        payload_hex = "abcd"
        digest = hashlib.sha384(struct.pack("<I", 1) + b":boot:" + bytes.fromhex(payload_hex)).digest()
        entries = [{"imr": 3, "event_type": 1, "event": "boot", "event_payload": payload_hex, "digest": digest.hex()}]
        rtmr3 = replay_rtmr3(entries)
        self.assertEqual(rtmr3, hashlib.sha384((b"\x00" * 48) + digest).digest())
        self.assertEqual(replay_rtmr3(json.dumps(entries)), rtmr3)

        with self.assertRaises(ValueError):
            replay_rtmr3([{"imr": 3, "event_type": 1, "event": "boot", "event_payload": payload_hex, "digest": "00" * 48}])


class TestNvidiaGpuVerifier(unittest.TestCase):
    def test_nras_jwt_verification_and_chutes(self):
        priv = ec.generate_private_key(ec.SECP384R1())
        pub_jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
        pub_jwk["kid"] = "nras-test-key"
        jwks = {"keys": [pub_jwk]}

        now = int(time.time())
        nonce = "ab" * 32
        token = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "sub": "GPU-0",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "eat_nonce": nonce,
                "x-nvidia-overall-att-result": True,
            },
            priv,
            algorithm="ES384",
            headers={"kid": "nras-test-key"},
        )
        ok = verify_nvidia_gpu_evidence(
            json.dumps({"nonce": nonce, "evidence_list": []}),
            nonce,
            post_fn=lambda _u, _b: [["JWT", token]],
            jwks_fetch_fn=lambda _u: jwks,
        )
        self.assertTrue(ok["verified"])
        self.assertEqual(ok["gpu_count"], 1)

        # Nonce mismatch
        self.assertFalse(verify_nvidia_gpu_evidence({"nonce": "00" * 32}, nonce)["verified"])

        # att_result == False
        bad_token = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "eat_nonce": nonce,
                "x-nvidia-overall-att-result": False,
            },
            priv,
            algorithm="ES384",
            headers={"kid": "nras-test-key"},
        )
        self.assertFalse(
            verify_nvidia_gpu_evidence(
                {"nonce": nonce},
                nonce,
                post_fn=lambda _u, _b: [["JWT", bad_token]],
                jwks_fetch_fn=lambda _u: jwks,
            )["verified"]
        )

        # Chutes GPU challenge
        ch_nonce = "00ff"
        ch_pub = "cHVibGljLWtleQ=="
        ch_digest = hashlib.sha256((ch_nonce + ch_pub).encode("utf-8")).hexdigest()
        ch_item = {"nonce": ch_nonce, "e2e_pubkey": ch_pub, "report_data": ch_digest + "00" * 32, "gpu_evidence": [{}]}
        self.assertEqual(verify_chutes_gpu_evidence(ch_item)["status"], "unavailable")
        self.assertTrue(verify_chutes_gpu_evidence(ch_item, spdm_verifier=lambda _e, _c: {"status": "verified", "verified": True})["verified"])

    def test_nras_rejects_alg_confusion_missing_nonce_and_bad_per_gpu_claims(self):
        priv = ec.generate_private_key(ec.SECP384R1())
        pub_jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
        pub_jwk["kid"] = "nras-test-key"
        jwks = {"keys": [pub_jwk]}
        now = int(time.time())
        nonce = "cd" * 32

        # 1. Disallowed alg (e.g., alg=none or ES256) must be rejected even if signed
        es256_priv = ec.generate_private_key(ec.SECP256R1())
        es256_jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(es256_priv.public_key()))
        es256_jwk["kid"] = "k256"
        es256_tok = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "eat_nonce": nonce,
                "x-nvidia-overall-att-result": True,
            },
            es256_priv,
            algorithm="ES256",
            headers={"kid": "k256"},
        )
        alg_res = verify_nvidia_gpu_evidence(
            {"nonce": nonce},
            nonce,
            post_fn=lambda _u, _b: [["JWT", es256_tok]],
            jwks_fetch_fn=lambda _u: {"keys": [es256_jwk]},
        )
        self.assertFalse(alg_res["verified"])
        self.assertIn("alg", alg_res["reason"])

        # 2. Missing eat_nonce in JWT must fail closed (anti-replay)
        no_nonce_tok = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "x-nvidia-overall-att-result": True,
            },
            priv,
            algorithm="ES384",
            headers={"kid": "nras-test-key"},
        )
        no_nonce_res = verify_nvidia_gpu_evidence(
            {"nonce": nonce},
            nonce,
            post_fn=lambda _u, _b: [["JWT", no_nonce_tok]],
            jwks_fetch_fn=lambda _u: jwks,
        )
        self.assertFalse(no_nonce_res["verified"])

        # 3. Per-GPU JWT must require measres == "success" and fail closed on dbgstat/secboot
        overall_tok = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "eat_nonce": nonce,
                "x-nvidia-overall-att-result": True,
            },
            priv,
            algorithm="ES384",
            headers={"kid": "nras-test-key"},
        )
        bad_gpu_tok = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "eat_nonce": nonce,
                "measres": "fail",
            },
            priv,
            algorithm="ES384",
            headers={"kid": "nras-test-key"},
        )
        bad_gpu_res = verify_nvidia_gpu_evidence(
            {"nonce": nonce},
            nonce,
            post_fn=lambda _u, _b: [["JWT", overall_tok], {"GPU-0": bad_gpu_tok}],
            jwks_fetch_fn=lambda _u: jwks,
        )
        self.assertFalse(bad_gpu_res["verified"])
        self.assertIn("measres", bad_gpu_res["reason"])

        good_gpu_tok = jwt.encode(
            {
                "iss": "https://nras.attestation.nvidia.com",
                "iat": now - 10,
                "nbf": now - 10,
                "exp": now + 300,
                "eat_nonce": nonce,
                "measres": "success",
                "secboot": True,
                "dbgstat": "disabled",
            },
            priv,
            algorithm="ES384",
            headers={"kid": "nras-test-key"},
        )
        good_gpu_res = verify_nvidia_gpu_evidence(
            {"nonce": nonce},
            nonce,
            post_fn=lambda _u, _b: [["JWT", overall_tok], {"GPU-0": good_gpu_tok}],
            jwks_fetch_fn=lambda _u: jwks,
        )
        self.assertTrue(good_gpu_res["verified"])
        self.assertEqual(good_gpu_res["gpu_count"], 2)

        # 4. Non-HTTPS remote PCCS URL rejected in TDX verifier
        raw_quote = (DCAP_FIXTURES / "tdx_quote").read_bytes()
        bad_pccs = verify_tdx_quote(raw_quote, pccs_url="http://evil.example.com/pccs")
        self.assertEqual(bad_pccs["status"], "failed")


class TestSummaryOverallGuardrail(unittest.TestCase):
    def test_overall_remains_unverified_even_when_quote_verified(self):
        summary = build_verification_summary({
            "report": {"gateway_attestation": {"intel_quote": "abc"}},
            "validation": {"checks": {"request_nonce": True, "report_data": True, "tls_fingerprint": True}},
            "ohttp_verification": {"status": "verified"},
            "gateway_quote_verification": {"status": "verified", "verified": True},
        })
        self.assertEqual(summary["gateway_quote"], "verified")
        self.assertEqual(summary["overall"], "unverified")

    def test_cli_verify_crypto_flags_forwarded(self):
        import argparse
        import nearai_agy as n

        fake_result = {
            "label": "evidence",
            "status": "unverified",
            "errors": [],
            "freshness": {"fresh": True},
            "verification_summary": {
                "request_binding": "verified",
                "tls_binding": "verified",
                "ohttp_signature": "verified",
                "gateway_quote": "verified",
                "model_execution": "unverified",
                "overall": "unverified",
            },
        }
        with patch.object(n, "load_config", return_value={"gateway_url": "https://example.test"}), \
             patch.object(n, "collect_evidence", return_value=fake_result) as mock_collect:
            args = argparse.Namespace(
                gateway=None,
                model="m",
                provider="near",
                no_tls_fingerprint=False,
                verify_crypto=True,
                allow_out_of_date_tcb=True,
                json=True,
            )
            rc = n.cmd_attest_evidence(args)
            self.assertEqual(rc, 0)
            self.assertTrue(mock_collect.call_args.kwargs["verify_crypto"])
            self.assertEqual(mock_collect.call_args.kwargs["accepted_tcb_statuses"], {"UpToDate", "OutOfDate"})

    def test_verify_chat_completion_end_to_end_mock(self):
        import io
        import urllib.parse
        from attestation_evidence import verify_chat_completion

        acct = Account.create()
        model = "Qwen/Qwen3.8-27B"
        resp_bytes = b'{"id":"chatcmpl-123","choices":[{"message":{"content":"OK"}}]}'

        class _FakeResp:
            def __init__(self, body: bytes, status: int = 200):
                self._buf = io.BytesIO(body)
                self.status = status
            def read(self) -> bytes:
                return self._buf.read()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        captured_req = {}
        def fake_opener(req, timeout=20):
            url = req.full_url
            if "/attestation/report" in url:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                nonce = qs["nonce"][0]
                return _FakeResp(json.dumps({
                    "request_nonce": nonce,
                    "gateway_attestation": {"request_nonce": nonce},
                    "model_attestations": [{"model_name": model, "signing_address": acct.address, "signing_algo": "ecdsa"}],
                }).encode("utf-8"))
            if "/attestation/pins" in url or "/attestation/ita-token" in url:
                return _FakeResp(b"{}")
            if url.endswith("/chat/completions"):
                captured_req["body"] = req.data
                return _FakeResp(resp_bytes)
            if "/signature/chatcmpl-123" in url:
                req_h = hash_exact_bytes(captured_req["body"])
                resp_h = hash_exact_bytes(resp_bytes)
                text = f"{model}:{req_h}:{resp_h}"
                sig = acct.sign_message(encode_defunct(text=text)).signature.hex()
                return _FakeResp(json.dumps({
                    "signature_kind": "provider_tee",
                    "text": text,
                    "signature": "0x" + sig,
                    "signing_address": acct.address,
                    "signing_algo": "ecdsa",
                }).encode("utf-8"))
            raise AssertionError(f"unexpected url: {url}")

        out = verify_chat_completion(
            "https://example.test/v1",
            "sk-fake-example.invalid",
            model,
            include_tls_fingerprint=False,
            verify_crypto=False,
            opener=fake_opener,
        )
        self.assertEqual(out["chat_id"], "chatcmpl-123")
        self.assertTrue(out["signature_verification"]["verified"])
        self.assertEqual(out["signature_verification"]["provenance"], "provider_tee")
        self.assertEqual(out["verification_summary"]["overall"], "unverified")


if __name__ == "__main__":
    unittest.main()
