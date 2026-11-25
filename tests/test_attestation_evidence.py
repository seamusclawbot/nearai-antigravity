import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import urllib.error
from attestation_evidence import build_verification_summary, collect_evidence, correlate_gpu_evidence, gpu_challenge, parse_model_attestations, parse_signature_response, redact_secrets, validate_ita_metadata, validate_report, verify_ita_tokens, verify_ohttp_signature


class TestEvidence(unittest.TestCase):
    def test_structured_attestation_and_ita_signature_shapes(self):
        parsed = parse_model_attestations([{"model": "m", "provider": "chutes", "quote_b64": "q", "e2e_pubkey": "k", "gpu_evidence": [{}]}])
        self.assertEqual(parsed["status"], "present_unverified")
        self.assertEqual(parsed["items"][0]["quote"], "present")
        ita = validate_ita_metadata({"jwks_url": "https://example/jwks", "nonce": "n", "policy_ids": [], "policy_must_match": True, "models": [{"model": "m", "token": "t"}]})
        self.assertEqual(ita["status"], "present_unverified")
        sig = parse_signature_response({"signature_kind": "provider_tee", "text": "m:r:h", "signature": "s", "signing_address": "a", "signing_algo": "ed25519"}, "m")
        self.assertTrue(sig["valid"])

        self.assertEqual(
            gpu_challenge("00ff", "cHVibGljLWtleQ=="),
            hashlib.sha256("00ffcHVibGljLWtleQ==".encode()).hexdigest(),
        )

    def test_report_data_requires_documented_tls_prefix(self):
        nonce = "a" * 64
        address = "0x" + "11" * 20
        tls = "22" * 32
        prefix = hashlib.sha256(bytes.fromhex("11" * 20) + bytes.fromhex(tls)).digest()
        report = {
            "gateway_attestation": {
                "request_nonce": nonce,
                "report_data": (prefix + bytes.fromhex(nonce)).hex(),
                "signing_address": address,
                "signing_algo": "ecdsa",
                "tls_cert_fingerprint": tls,
            }
        }
        out = validate_report(report, nonce)
        self.assertTrue(out["valid"])
        self.assertTrue(out["checks"]["report_data"])

        # Negative case: mismatched TLS fingerprint must fail report_data check
        bad_report = {
            "gateway_attestation": {
                "request_nonce": nonce,
                "report_data": (prefix + bytes.fromhex(nonce)).hex(),
                "signing_address": address,
                "signing_algo": "ecdsa",
                "tls_cert_fingerprint": "33" * 32,
            }
        }
        self.assertFalse(validate_report(bad_report, nonce)["valid"])

        result = {"request_metadata": {"model": "requested/model", "provider": "vllm"},
                  "report": {"model": "returned/model", "provider": "external"}}
        summary = build_verification_summary(result)
        self.assertEqual(summary["correlation"]["model"], "mismatch")
        self.assertEqual(summary["correlation"]["provider"], "mismatch")

        value = {"Authorization": "Bearer sk-live-secret", "nested": "token-abcdef1234567890"}
        out = redact_secrets(value)
        self.assertEqual(out["Authorization"], "[REDACTED]")
        self.assertNotIn("sk-live-secret", str(out))
        self.assertNotIn("abcdef1234567890", str(out))

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, __import__('io').BytesIO(b'{"message":"invalid"}'))
        out = collect_evidence("https://example.test", opener=opener)
        err = out["errors"][0]
        self.assertEqual(err["http_status"], 400)
        self.assertEqual(err["response_body"], {"message": "invalid"})
        self.assertEqual(err["error_type"], "HTTPError")

    def test_http_text_error_preserves_status_and_body(self):
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, __import__('io').BytesIO(b"invalid request"))
        err = collect_evidence("https://example.test", opener=opener)["errors"][0]
        self.assertEqual(err["http_status"], 400)
        self.assertEqual(err["response_body"], "invalid request")


    def test_non_hex_or_short_nonce_fails_report_data(self):
        nonce = "abc"
        report = {"request_nonce": nonce, "report_data": nonce, "tls_cert_fingerprint": "22" * 32}
        self.assertFalse(validate_report(report, nonce)["valid"])

    def test_ecdsa_unpadded_20_byte_identity_without_tls(self):
        nonce = "c" * 64
        address = "0x" + "12" * 20
        prefix = bytes.fromhex("12" * 20).ljust(32, b"\x00")
        report = {
            "request_nonce": nonce,
            "report_data": (prefix + bytes.fromhex(nonce)).hex(),
            "signing_address": address,
            "signing_algo": "ecdsa",
        }
        out = validate_report(report, nonce, include_tls_fingerprint=False)
        self.assertTrue(out["valid"])
        self.assertTrue(out["checks"]["report_data"])

    def test_ita_nested_gateway_token_is_unverified(self):
        out = verify_ita_tokens({"gateway": {"ita_token": "jwt"}, "jwks_url": "https://example.test/jwks"})
        self.assertEqual(out["status"], "present_unverified")
        self.assertIn("gateway.ita_token", out["token_fields"])


        out = verify_ita_tokens({"gateway_token": "header.payload.sig", "jwks_url": "https://example.test/keys"})
        self.assertEqual(out["status"], "present_unverified")
        self.assertFalse(out["verified"])


        result = {
            "report": {"gateway_attestation": {"intel_quote": "quote"}, "model_attestations": [{}]},
            "validation": {"checks": {"request_nonce": True, "report_data": True, "tls_fingerprint": True}},
            "ohttp_verification": {"status": "verified"},
        }
        summary = build_verification_summary(result)
        self.assertEqual(summary["request_binding"], "verified")
        self.assertEqual(summary["ohttp_signature"], "verified")
        self.assertEqual(summary["gateway_quote"], "present_unverified")
        self.assertEqual(summary["model_execution"], "unverified")
        self.assertEqual(summary["overall"], "unverified")


        out = verify_ohttp_signature({})
        self.assertFalse(out["verified"])
        self.assertEqual(out["status"], "unavailable")


        nonce = "b" * 64
        report = {
            "gateway_attestation": {
                "request_nonce": nonce,
                "report_data": "a" * 64 + nonce,
                "signing_address": "a" * 64,
            }
        }
        out = validate_report(report, nonce, include_tls_fingerprint=False)
        self.assertTrue(out["valid"])
        self.assertTrue(out["checks"]["report_data"])

        self.assertFalse(validate_report({"report_data": "ab" * 64, "tls_cert_fingerprint": "22" * 32}, "cd" * 32)["valid"])

    def test_mismatch_nonce(self):
        report = {"request_nonce": "00" * 32, "report_data": "ab" * 64, "tls_cert_fingerprint": "22" * 32}
        self.assertFalse(validate_report(report, "11" * 32)["valid"])

    def test_missing_report_data_tls(self):
        out = validate_report({"request_nonce": "11" * 32}, "11" * 32)
        self.assertFalse(out["valid"])
        self.assertFalse(out["checks"]["tls_fingerprint"])

    def test_tls_not_required_when_disabled(self):
        nonce = "11" * 32
        address = "0x" + "12" * 20
        prefix = bytes.fromhex("12" * 20).ljust(32, b"\x00")
        report = {
            "request_nonce": nonce,
            "report_data": (prefix + bytes.fromhex(nonce)).hex(),
            "signing_address": address,
            "signing_algo": "ecdsa",
        }
        out = validate_report(report, nonce, include_tls_fingerprint=False)
        self.assertTrue(out["valid"])
        self.assertIsNone(out["checks"]["tls_fingerprint"])
        self.assertFalse(out["tls_fingerprint_required"])
        summary = build_verification_summary({"report": report, "validation": out})
        self.assertEqual(summary["tls_binding"], "disabled")

    def test_fail_closed_missing_identity_conflicting_nonce_and_spki_pinning(self):
        nonce = "11" * 32
        # Missing signing_address must fail closed even when TLS binding is disabled
        no_id_report = {"request_nonce": nonce, "report_data": ("00" * 32) + nonce}
        out = validate_report(no_id_report, nonce, include_tls_fingerprint=False)
        self.assertFalse(out["valid"])
        self.assertFalse(out["checks"]["report_data"])

        # Conflicting top-level vs gateway_attestation request_nonce must fail
        address = "0x" + "12" * 20
        prefix = bytes.fromhex("12" * 20).ljust(32, b"\x00")
        conflict = {
            "request_nonce": nonce,
            "gateway_attestation": {
                "request_nonce": "22" * 32,
                "report_data": (prefix + bytes.fromhex(nonce)).hex(),
                "signing_address": address,
                "signing_algo": "ecdsa",
            },
        }
        out_conflict = validate_report(conflict, nonce, include_tls_fingerprint=False)
        self.assertFalse(out_conflict["valid"])
        self.assertFalse(out_conflict["checks"]["request_nonce"])

        # Explicit peer SPKI pinning matches or fails closed
        tls = "22" * 32
        tls_prefix = hashlib.sha256(bytes.fromhex("12" * 20) + bytes.fromhex(tls)).digest()
        tls_report = {
            "gateway_attestation": {
                "request_nonce": nonce,
                "report_data": (tls_prefix + bytes.fromhex(nonce)).hex(),
                "signing_address": address,
                "signing_algo": "ecdsa",
                "tls_cert_fingerprint": tls,
            }
        }
        pinned_ok = validate_report(tls_report, nonce, include_tls_fingerprint=True, expected_tls_spki_sha256=tls)
        self.assertTrue(pinned_ok["valid"])
        self.assertTrue(pinned_ok["tls_peer_pinned"])
        summary_pinned = build_verification_summary({"report": tls_report, "validation": pinned_ok})
        self.assertEqual(summary_pinned["tls_binding"], "verified")

        unpinned = validate_report(tls_report, nonce, include_tls_fingerprint=True, expected_tls_spki_sha256=None)
        self.assertTrue(unpinned["valid"])
        self.assertFalse(unpinned["tls_peer_pinned"])
        summary_unpinned = build_verification_summary({"report": tls_report, "validation": unpinned})
        self.assertEqual(summary_unpinned["tls_binding"], "present_unpinned")

        pinned_bad = validate_report(tls_report, nonce, include_tls_fingerprint=True, expected_tls_spki_sha256="33" * 32)
        self.assertFalse(pinned_bad["valid"])
        self.assertFalse(pinned_bad["checks"]["tls_fingerprint"])
        self.assertFalse(pinned_bad["tls_peer_pinned"])

    def test_freshness_future_clock_absent_correlation_and_expanded_redaction(self):
        import time
        from attestation_evidence import evidence_freshness

        future = evidence_freshness({"collected_at": time.time() + 60.0})
        self.assertFalse(future["fresh"])
        self.assertEqual(future["status"], "invalid_clock")

        summary = build_verification_summary({"request_metadata": {"model": "m1"}, "report": {}})
        self.assertEqual(summary["correlation"]["model"], "absent")
        self.assertEqual(summary["correlation"]["provider"], "absent")

        redacted = redact_secrets({
            "X-API-Key": "raw-secret",
            "Cookie": "sid=abc",
            "url": "https://example.test/v1?api_key=mysecret123&token=tok999&ok=1",
        })
        self.assertEqual(redacted["X-API-Key"], "[REDACTED]")
        self.assertEqual(redacted["Cookie"], "[REDACTED]")
        self.assertNotIn("mysecret123", redacted["url"])
        self.assertNotIn("tok999", redacted["url"])
        self.assertIn("ok=1", redacted["url"])

    def test_collect_evidence_validates_gateway_and_caps_response_size(self):
        with self.assertRaises(ValueError):
            collect_evidence("http://remote-cleartext.example.com/v1")

        class _HugeResp:
            def read(self, n=-1):
                return b"x" * (5 * 1024 * 1024)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        out = collect_evidence("https://example.test/v1", opener=lambda req, timeout: _HugeResp())
        self.assertTrue(out["errors"])
        self.assertIn("limit", out["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
