"""Evidence-only nonce/TLS-bound attestation collector.

This module collects and validates server-provided evidence; it deliberately
performs no TDX/GPU cryptographic verification. Results are always labelled
``evidence`` and ``unverified``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

MAX_EVIDENCE_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MiB
MAX_FUTURE_CLOCK_SKEW_SECONDS = 5.0

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_REDACTED_KEYS = {
    "authorization",
    "proxy-authorization",
    "api_key",
    "x-api-key",
    "token",
    "access_token",
    "refresh_token",
    "cookie",
    "set-cookie",
    "secret",
    "near_ai_key",
    "password",
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_DEFAULT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if h in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _validate_evidence_gateway(gateway: str) -> str:
    """Enforce https:// (or loopback http://) and reject embedded userinfo/control chars."""
    u = (gateway or "").strip()
    if not u or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in u):
        raise ValueError("gateway must be a non-empty URL without whitespace or control characters")
    p = urllib.parse.urlsplit(u)
    if p.scheme not in ("https", "http"):
        raise ValueError("gateway must use https:// (or loopback http://)")
    if not p.hostname:
        raise ValueError("gateway must include a valid hostname")
    if p.username is not None or p.password is not None:
        raise ValueError("gateway must not contain embedded userinfo")
    if p.scheme == "http" and not _is_loopback_host(p.hostname):
        raise ValueError("http:// gateway is only permitted for loopback hosts")
    return u.rstrip("/")


def _read_bounded_response(resp: Any, max_bytes: int = MAX_EVIDENCE_RESPONSE_BYTES) -> bytes:
    """Read an HTTP response up to max_bytes while remaining compatible with 0-arg test mocks."""
    try:
        data = resp.read(max_bytes + 1)
    except TypeError:
        data = resp.read()
    if len(data) > max_bytes:
        raise ValueError(f"HTTP response exceeded {max_bytes} byte limit")
    return data


def _extract_peer_spki_sha256(gateway: str, timeout: float = 10.0) -> Optional[str]:
    """Extract live TLS peer certificate SPKI SHA-256 hex digest for an https:// gateway."""
    p = urllib.parse.urlsplit(gateway)
    if p.scheme != "https" or not p.hostname:
        return None
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    except ImportError:
        return None
    port = p.port or 443
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((p.hostname, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=p.hostname) as ssock:
                der = ssock.getpeercert(binary_form=True)
        if not der:
            return None
        cert = x509.load_der_x509_certificate(der)
        spki_der = cert.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        return hashlib.sha256(spki_der).hexdigest()
    except Exception:
        return None


def _decode(value: Any) -> Optional[bytes]:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        return None
    # API fields are commonly hexadecimal strings; prefer exact hex decoding
    # before attempting strict base64 decoding.
    try:
        if len(value) % 2 == 0 and value and all(c in "0123456789abcdefABCDEF" for c in value):
            return bytes.fromhex(value)
    except ValueError:
        pass
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return None


def verify_ohttp_signature(
    attestation: Dict[str, Any],
    gateway_attestation: Optional[Dict[str, Any]] = None,
    ohttp_key_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Verify an OHTTP Ed25519 signature when an optional verifier is available.

    Optionally cross-checks ``ohttp_key_config`` equality and that ``signing_key``
    matches ``gateway_attestation.signing_address`` (fail-closed when
    ``gateway_attestation`` is provided). Returns ``unavailable`` when the
    payload is incomplete or ``cryptography`` is absent.
    """
    key_config = _decode(attestation.get("key_config"))
    signature = _decode(attestation.get("signature"))
    signing_key = _decode(attestation.get("signing_key"))
    if not (key_config and signature and signing_key):
        return {"status": "unavailable", "verified": False, "reason": "missing OHTTP fields"}
    if ohttp_key_config is not None:
        expected_cfg = _decode(ohttp_key_config)
        if expected_cfg is None or not hmac.compare_digest(expected_cfg, key_config):
            return {"status": "failed", "verified": False, "reason": "ohttp_key_config mismatch"}
    if isinstance(gateway_attestation, dict):
        gw_algo = str(gateway_attestation.get("signing_algo") or "ed25519").lower()
        if gw_algo != "ed25519":
            return {"status": "failed", "verified": False, "reason": f"unsupported gateway_attestation signing_algo for OHTTP: {gw_algo}"}
        gw_addr = _parse_signing_identity(gateway_attestation.get("signing_address"), gw_algo)
        if gw_addr is None:
            return {"status": "failed", "verified": False, "reason": "gateway_attestation missing valid ed25519 signing_address"}
        if not hmac.compare_digest(gw_addr, signing_key):
            return {"status": "failed", "verified": False, "reason": "signing_key does not match gateway_attestation"}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(signing_key).verify(signature, key_config)
    except ImportError:
        return {"status": "unavailable", "verified": False, "reason": "cryptography package unavailable"}
    except Exception as exc:
        return {"status": "failed", "verified": False, "reason": type(exc).__name__}
    return {"status": "verified", "verified": True}


def redact_secrets(value: Any) -> Any:
    """Recursively redact bearer/API-key/cookie/query credential material from diagnostics."""
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if str(k).lower() in _REDACTED_KEYS else redact_secrets(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)(?:sk|key|token|secret)[-_][A-Za-z0-9._~-]{8,}", "[REDACTED]", value)
        value = re.sub(
            r"(?i)([?&](?:api[_-]?key|token|access[_-]?token|key|secret)=)[^&\s]+",
            r"\1[REDACTED]",
            value,
        )
        return value
    return value


def _safe_error(exc: Exception, endpoint: str) -> Dict[str, Any]:
    """Return structured, credential-safe transport error diagnostics."""
    item: Dict[str, Any] = {"endpoint": endpoint, "error": str(exc), "error_type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        item["http_status"] = exc.code
        try:
            raw = exc.read(64 * 1024)
            text = raw.decode("utf-8", errors="replace")
            try:
                item["response_body"] = json.loads(text)
            except (TypeError, ValueError):
                item["response_body"] = text
        except Exception:
            item["response_body"] = None
    elif isinstance(exc, urllib.error.URLError):
        item["reason"] = str(exc.reason)
    return redact_secrets(item)


def gpu_challenge(nonce: str, e2e_pubkey: str) -> str:
    """Return the documented textual GPU challenge digest.

    The Gateway contract hashes the UTF-8 text of the returned nonce followed by
    the returned base64 public-key text. Neither value is decoded first.
    """
    if not isinstance(nonce, str) or not isinstance(e2e_pubkey, str) or not nonce or not e2e_pubkey:
        raise ValueError("nonce and e2e_pubkey must be non-empty strings")
    return hashlib.sha256((nonce + e2e_pubkey).encode("utf-8")).hexdigest()


def _report_data_bytes(value: Any) -> Optional[bytes]:
    """Decode report_data strictly as hex or base64 (never raw UTF-8 fallback)."""
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        return None
    parsed_hex = _parse_hex_bytes(value)
    if parsed_hex is not None:
        return parsed_hex
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return None


def _parse_hex_bytes(value: Any, expected_len: Optional[int] = None) -> Optional[bytes]:
    """Decode a hexadecimal string (with optional 0x prefix) into raw bytes."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned.lower().startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned or len(cleaned) % 2 != 0:
        return None
    try:
        raw = bytes.fromhex(cleaned)
    except ValueError:
        return None
    if expected_len is not None and len(raw) != expected_len:
        return None
    return raw


def _parse_signing_identity(signing_address: Any, signing_algo: Any = None) -> Optional[bytes]:
    """Decode signing_address to 20-byte (ecdsa) or 32-byte (ed25519) raw identity."""
    raw = _parse_hex_bytes(signing_address)
    if raw is None:
        return None
    algo = signing_algo.lower() if isinstance(signing_algo, str) else None
    if algo == "ecdsa":
        return raw if len(raw) == 20 else None
    if algo == "ed25519":
        return raw if len(raw) == 32 else None
    if algo is None and len(raw) in (20, 32):
        return raw
    return None


def validate_report(report: Dict[str, Any], nonce: str,
                    include_tls_fingerprint: bool = True,
                    expected_tls_spki_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Validate documented bindings only; does not verify signatures or quotes.

    Per NEAR AI docs (quote-nonce-signer.md), report_data is 64 bytes:
    - report_data[32:64] == bytes.fromhex(nonce) (32-byte client nonce)
    - report_data[0:32] == SHA256(identity_bytes || spki_sha256_bytes) when
      include_tls_fingerprint=True, or identity_bytes.ljust(32, b"\\x00") otherwise.
    Fail-closed: requires a valid 32-byte client nonce and a valid signing_address
    in all modes. When ``expected_tls_spki_sha256`` is supplied alongside
    ``include_tls_fingerprint=True``, also verifies peer TLS SPKI equality.
    """
    checks: Dict[str, Optional[bool]] = {
        "request_nonce": False,
        "report_data": False,
        "tls_fingerprint": False if include_tls_fingerprint else None,
    }
    gateway = report.get("gateway_attestation")
    gw = gateway if isinstance(gateway, dict) else {}

    nonce_bytes = _parse_hex_bytes(nonce, expected_len=32)

    top_nonce = report.get("request_nonce")
    gw_nonce = gw.get("request_nonce")
    top_nonce_bytes = _parse_hex_bytes(top_nonce, expected_len=32) if top_nonce is not None else None
    gw_nonce_bytes = _parse_hex_bytes(gw_nonce, expected_len=32) if gw_nonce is not None else None

    if top_nonce is not None and gw_nonce is not None:
        nonce_consistent = (
            top_nonce_bytes is not None
            and gw_nonce_bytes is not None
            and hmac.compare_digest(top_nonce_bytes, gw_nonce_bytes)
        )
        echoed_nonce_bytes = top_nonce_bytes if nonce_consistent else None
    elif top_nonce is not None:
        echoed_nonce_bytes = top_nonce_bytes
    else:
        echoed_nonce_bytes = gw_nonce_bytes

    checks["request_nonce"] = (
        nonce_bytes is not None
        and echoed_nonce_bytes is not None
        and hmac.compare_digest(echoed_nonce_bytes, nonce_bytes)
    )

    report_data = report.get("report_data")
    if report_data is None:
        report_data = gw.get("report_data")
    raw = _parse_hex_bytes(report_data, expected_len=64) if isinstance(report_data, str) else (
        report_data if isinstance(report_data, bytes) and len(report_data) == 64 else None
    )

    tls = report.get("tls_cert_fingerprint")
    if tls is None:
        tls = report.get("tls_fingerprint")
    if tls is None:
        tls = gw.get("tls_cert_fingerprint")
    if tls is None:
        tls = gw.get("tls_fingerprint")
    tls_bytes = _parse_hex_bytes(tls, expected_len=32)
    peer_spki_pinned = False
    if include_tls_fingerprint:
        if expected_tls_spki_sha256 is not None:
            expected_spki_bytes = _parse_hex_bytes(expected_tls_spki_sha256, expected_len=32)
            tls_ok = (
                tls_bytes is not None
                and expected_spki_bytes is not None
                and hmac.compare_digest(tls_bytes, expected_spki_bytes)
            )
            peer_spki_pinned = tls_ok
        else:
            tls_ok = tls_bytes is not None
        checks["tls_fingerprint"] = tls_ok

    signing_address = report.get("signing_address")
    if signing_address is None:
        signing_address = gw.get("signing_address")
    signing_algo = report.get("signing_algo")
    if signing_algo is None:
        signing_algo = gw.get("signing_algo")
    identity_bytes = _parse_signing_identity(signing_address, signing_algo)

    if raw is not None and nonce_bytes is not None:
        nonce_match = hmac.compare_digest(raw[32:64], nonce_bytes)
        if include_tls_fingerprint:
            if identity_bytes is not None and tls_bytes is not None:
                expected_prefix = hashlib.sha256(identity_bytes + tls_bytes).digest()
                prefix_match = hmac.compare_digest(raw[:32], expected_prefix)
            else:
                prefix_match = False
        else:
            # Fail closed: require valid signing identity even when TLS binding is disabled.
            if identity_bytes is not None:
                prefix_match = hmac.compare_digest(raw[:32], identity_bytes.ljust(32, b"\x00"))
            else:
                prefix_match = False
        checks["report_data"] = nonce_match and prefix_match

    valid = bool(
        checks["request_nonce"]
        and checks["report_data"]
        and (checks["tls_fingerprint"] is True if include_tls_fingerprint else True)
    )
    return {
        "checks": checks,
        "valid": valid,
        "status": "unverified",
        "tls_fingerprint_required": include_tls_fingerprint,
        "tls_peer_pinned": peer_spki_pinned,
    }


def verify_ita_tokens(ita_payload: Any) -> Dict[str, Any]:
    """Perform non-cryptographic ITA JWT shape checks.

    Signature/JWKS verification remains unavailable until the official ITA
    verifier contract and key-fetch semantics are established.
    """
    if not isinstance(ita_payload, dict):
        return {"status": "unavailable", "verified": False, "reason": "missing ITA payload"}
    token_fields = [key for key in ita_payload if "token" in key.lower() or key.endswith("_jwt")]
    gateway = ita_payload.get("gateway")
    if isinstance(gateway, dict):
        token_fields.extend("gateway." + key for key in gateway if "token" in key.lower() or key.endswith("_jwt"))
    models = ita_payload.get("models")
    if isinstance(models, list):
        for index, model in enumerate(models):
            if isinstance(model, dict):
                token_fields.extend(f"models[{index}].{key}" for key in model if "token" in key.lower() or key.endswith("_jwt"))
    return {
        "status": "present_unverified" if token_fields else "unavailable",
        "verified": False,
        "token_fields": token_fields,
        "jwks_url_present": bool(ita_payload.get("jwks_url")),
    }
def parse_model_attestations(report: Any) -> Dict[str, Any]:
    """Extract model-attestation presence signals without verifying quotes."""
    items = report if isinstance(report, list) else []
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed.append({
            "model": item.get("model"),
            "provider": item.get("provider"),
            "quote": "present" if item.get("quote_b64") or item.get("intel_quote") else "absent",
            "certificate": "present" if item.get("certificate_b64") or item.get("certificate") else "absent",
            "e2e_pubkey": "present" if item.get("e2e_pubkey") else "absent",
            "gpu_evidence": "present" if item.get("gpu_evidence") else "absent",
            "gpu_verdict": item.get("gpu_verdict", "unavailable"),
            "nonce": item.get("nonce"),
            "status": "present_unverified",
        })
    return {"status": "present_unverified" if parsed else "unavailable", "items": parsed}


def correlate_gpu_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    """Check documented GPU challenge/report-data correlation only."""
    nonce = item.get("nonce")
    pubkey = item.get("e2e_pubkey")
    report_data = _report_data_bytes(item.get("report_data"))
    if not (isinstance(nonce, str) and isinstance(pubkey, str) and nonce and pubkey and report_data):
        return {"status": "unavailable", "matched": False}
    challenge = gpu_challenge(nonce, pubkey)
    expected = bytes.fromhex(challenge)
    matched = len(report_data) == 64 and hmac.compare_digest(report_data[:32], expected)
    return {"status": "matched" if matched else "mismatch", "matched": matched, "challenge_sha256": challenge}


def validate_ita_metadata(payload: Any) -> Dict[str, Any]:
    """Validate ITA response shape and model-token correlation, not JWTs."""
    if not isinstance(payload, dict):
        return {"status": "unavailable", "valid": False, "reason": "missing ITA payload"}
    models = payload.get("models")
    valid = isinstance(payload.get("jwks_url"), str) and bool(payload.get("jwks_url"))
    valid = valid and isinstance(payload.get("nonce"), str) and isinstance(payload.get("policy_ids"), list)
    valid = valid and isinstance(payload.get("policy_must_match"), bool) and isinstance(models, list)
    model_items = [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []
    valid = valid and all(isinstance(m.get("model"), str) and isinstance(m.get("token"), str) for m in model_items)
    return {"status": "present_unverified" if valid else "invalid_shape", "valid": valid, "model_count": len(model_items), "jwks_url_present": bool(payload.get("jwks_url"))}


def parse_signature_response(payload: Any, requested_model: str = "") -> Dict[str, Any]:
    """Parse documented response-signature payload shape without verifying it."""
    if not isinstance(payload, dict):
        return {"status": "unavailable", "valid": False}
    kind = payload.get("signature_kind")
    text = payload.get("text")
    valid = all(isinstance(payload.get(k), str) and payload.get(k) for k in ("text", "signature", "signing_address", "signing_algo"))
    parts = text.split(":") if isinstance(text, str) else []
    if kind == "provider_tee":
        valid = valid and len(parts) == 3 and (not requested_model or parts[0] == requested_model)
    elif kind == "gateway":
        valid = valid and len(parts) == 2
    else:
        valid = False
    return {"status": "present_unverified" if valid else "invalid_shape", "valid": valid, "signature_kind": kind}



def evidence_freshness(result: Dict[str, Any], max_age_seconds: float = 300.0) -> Dict[str, Any]:
    """Report collection freshness without asserting quote validity."""
    collected = result.get("collected_at")
    if not isinstance(collected, (int, float)):
        return {"status": "unavailable", "fresh": False, "reason": "missing collection timestamp"}
    delta = time.time() - collected
    if delta < -MAX_FUTURE_CLOCK_SKEW_SECONDS:
        return {
            "status": "invalid_clock",
            "fresh": False,
            "reason": "collection timestamp is in the future",
            "age_seconds": round(delta, 3),
            "max_age_seconds": max_age_seconds,
        }
    age = max(0.0, delta)
    return {"status": "fresh" if age <= max_age_seconds else "stale", "fresh": age <= max_age_seconds, "age_seconds": round(age, 3), "max_age_seconds": max_age_seconds}


def _correlate_field(requested_val: Any, reported_val: Any) -> str:
    if not requested_val or not reported_val:
        return "absent"
    return "matched" if requested_val == reported_val else "mismatch"


def build_verification_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return conservative, separately reported verification signals."""
    report = result.get("report") or {}
    gateway = report.get("gateway_attestation") or {}
    model_atts = report.get("model_attestations") or []
    model_signal = parse_model_attestations(model_atts)
    validation = result.get("validation") or {}
    checks = validation.get("checks") or {}
    ohttp = result.get("ohttp_verification") or {}
    ita = result.get("ita_verification") or {}
    report_model = report.get("model") or gateway.get("model")
    report_provider = report.get("provider") or gateway.get("provider")
    requested = result.get("request_metadata") or {}
    correlation = {
        "model": _correlate_field(requested.get("model"), report_model),
        "provider": _correlate_field(requested.get("provider"), report_provider),
    }
    gw_quote_ver = result.get("gateway_quote_verification") or {}
    if gw_quote_ver.get("status"):
        gateway_quote_status = gw_quote_ver["status"]
    else:
        gateway_quote_status = "present_unverified" if gateway.get("intel_quote") else "absent"
    model_quotes_ver = result.get("model_quote_verifications")
    gpu_ver = result.get("gpu_verifications")
    resp_sig = result.get("response_signature_verification") or {}

    tls_check = checks.get("tls_fingerprint")
    if validation.get("tls_fingerprint_required") is False or (
        "tls_fingerprint" in checks and tls_check is None
    ):
        tls_binding_status = "disabled"
    elif tls_check is True:
        if validation.get("tls_peer_pinned") is False:
            tls_binding_status = "present_unpinned"
        else:
            tls_binding_status = "verified"
    else:
        tls_binding_status = "unverified"

    summary = {
        "request_binding": "verified" if checks.get("request_nonce") and checks.get("report_data") else "unverified",
        "tls_binding": tls_binding_status,
        "ohttp_signature": ohttp.get("status", "unavailable"),
        "ita_tokens": ita.get("status", "unavailable"),
        "gateway_quote": gateway_quote_status,
        "model_execution": "unverified" if model_atts else "not_collected",
        "model_attestations": model_signal,
        "correlation": correlation,
        "overall": "unverified",
    }
    if model_quotes_ver is not None:
        summary["model_quotes"] = model_quotes_ver
    if gpu_ver is not None:
        summary["gpu_evidence"] = gpu_ver
    if resp_sig:
        summary["response_signature"] = resp_sig.get("status", "unavailable")
    return summary


def _verify_collected_crypto(
    report: Dict[str, Any],
    nonce: str,
    include_tls_fingerprint: bool,
    accepted_tcb_statuses: Optional[Any] = None,
    pccs_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run opt-in cryptographic verifiers for TDX quotes and NVIDIA GPU evidence."""
    from verifiers.nvidia import verify_chutes_gpu_evidence, verify_nvidia_gpu_evidence
    from verifiers.tdx import verify_tdx_quote

    out: Dict[str, Any] = {}
    gw = report.get("gateway_attestation")
    if isinstance(gw, dict) and gw.get("intel_quote"):
        out["gateway_quote_verification"] = verify_tdx_quote(
            gw.get("intel_quote"),
            nonce=nonce,
            signing_address=gw.get("signing_address"),
            signing_algo=gw.get("signing_algo"),
            tls_cert_fingerprint=gw.get("tls_cert_fingerprint") or gw.get("tls_fingerprint"),
            include_tls_fingerprint=include_tls_fingerprint,
            app_compose=gw.get("info", {}).get("app_compose") if isinstance(gw.get("info"), dict) else None,
            event_log=gw.get("event_log"),
            pccs_url=pccs_url,
            accepted_tcb_statuses=accepted_tcb_statuses,
        )

    model_atts = report.get("model_attestations")
    if isinstance(model_atts, list) and model_atts:
        quote_results = []
        gpu_results = []
        for item in model_atts:
            if not isinstance(item, dict):
                continue
            quote_val = item.get("intel_quote") or item.get("quote_b64")
            item_model = item.get("model") or item.get("model_name")
            item_tls = item.get("tls_cert_fingerprint") or item.get("tls_fingerprint")
            if quote_val:
                quote_results.append({
                    "model": item_model,
                    "result": verify_tdx_quote(
                        quote_val,
                        nonce=nonce,
                        signing_address=item.get("signing_address"),
                        signing_algo=item.get("signing_algo"),
                        tls_cert_fingerprint=item_tls,
                        include_tls_fingerprint=include_tls_fingerprint,
                        app_compose=item.get("info", {}).get("app_compose") if isinstance(item.get("info"), dict) else None,
                        event_log=item.get("event_log"),
                        pccs_url=pccs_url,
                        accepted_tcb_statuses=accepted_tcb_statuses,
                    ),
                })
            if item.get("nvidia_payload"):
                gpu_results.append({
                    "model": item_model,
                    "provider": item.get("provider") or "near",
                    "result": verify_nvidia_gpu_evidence(
                        item.get("nvidia_payload"),
                        nonce,
                    ),
                })
            elif item.get("gpu_evidence"):
                gpu_results.append({
                    "model": item_model,
                    "provider": item.get("provider") or "chutes",
                    "result": verify_chutes_gpu_evidence(item),
                })
        if quote_results:
            out["model_quote_verifications"] = quote_results
        if gpu_results:
            out["gpu_verifications"] = gpu_results
    return out


def collect_evidence(gateway: str, model: str = "", provider: str = "",
                     include_tls_fingerprint: bool = True, timeout: float = 20,
                     opener: Optional[Callable[..., Any]] = None,
                     verify_crypto: bool = False,
                     accepted_tcb_statuses: Optional[Any] = None,
                     pccs_url: Optional[str] = None,
                     expected_tls_spki_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Collect raw attestation evidence with an isolated cryptographic nonce."""
    gateway = _validate_evidence_gateway(gateway)
    nonce = secrets.token_hex(32)
    params = {"nonce": nonce, "include_tls_fingerprint": "true" if include_tls_fingerprint else "false"}
    if model:
        params["model"] = model
    if provider:
        params["provider"] = provider
    url = gateway + "/attestation/report?" + urllib.parse.urlencode(params)
    open_fn = opener or _DEFAULT_OPENER.open
    observed_spki = expected_tls_spki_sha256
    if observed_spki is None and opener is None and include_tls_fingerprint:
        observed_spki = _extract_peer_spki_sha256(gateway, timeout=min(timeout, 10.0))
    request_metadata = {"model": model or None, "provider": provider or None, "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(), "nonce_length": len(nonce)}
    result: Dict[str, Any] = {"label": "evidence", "status": "unverified", "nonce": nonce, "collected_at": time.time(),
                               "request_metadata": request_metadata,
                               "observed_tls_spki_sha256": observed_spki,
                              "report": None, "pins": None, "ita_token": None,
                              "ohttp_verification": None,
                              "validation": None, "errors": []}
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with open_fn(req, timeout=timeout) as response:
            result["report"] = json.loads(_read_bounded_response(response).decode("utf-8"))
        result["ohttp_verification"] = verify_ohttp_signature(
            result["report"].get("ohttp_attestation") or {},
            gateway_attestation=result["report"].get("gateway_attestation"),
            ohttp_key_config=result["report"].get("ohttp_key_config"),
        )
        result["validation"] = validate_report(
            result["report"],
            nonce,
            include_tls_fingerprint,
            expected_tls_spki_sha256=observed_spki,
        )
        if verify_crypto:
            result.update(_verify_collected_crypto(
                result["report"],
                nonce=nonce,
                include_tls_fingerprint=include_tls_fingerprint,
                accepted_tcb_statuses=accepted_tcb_statuses,
                pccs_url=pccs_url,
            ))
        result["ita_verification"] = verify_ita_tokens(result.get("ita_token"))
        result["verification_summary"] = build_verification_summary(result)
    except Exception as exc:
        result["errors"].append(_safe_error(exc, "/attestation/report"))
        return result
    for endpoint, key in (("/attestation/pins", "pins"), ("/attestation/ita-token", "ita_token")):
        try:
            req = urllib.request.Request(gateway + endpoint, headers={"Accept": "application/json"})
            with open_fn(req, timeout=timeout) as response:
                result[key] = json.loads(_read_bounded_response(response).decode("utf-8"))
        except Exception as exc:
            result["errors"].append(_safe_error(exc, endpoint))
    result["ita_verification"] = verify_ita_tokens(result.get("ita_token"))
    result["freshness"] = evidence_freshness(result)
    result["verification_summary"] = build_verification_summary(result)
    return result


def verify_chat_completion(
    gateway: str,
    api_key: str,
    model: str,
    *,
    prompt: str = "Reply with OK.",
    signing_algo: str = "ecdsa",
    include_tls_fingerprint: bool = True,
    verify_crypto: bool = True,
    accepted_tcb_statuses: Optional[Any] = None,
    pccs_url: Optional[str] = None,
    max_sig_retries: int = 3,
    retry_delay: float = 0.5,
    timeout: float = 30.0,
    opener: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run a preflight-attested /chat/completions call and verify /signature/{chat_id}."""
    from verifiers.signatures import hash_exact_bytes, verify_response_signature

    if not api_key:
        return {"label": "evidence", "status": "unverified", "errors": [{"error": "missing api_key"}]}
    if not model:
        return {"label": "evidence", "status": "unverified", "errors": [{"error": "missing model"}]}

    gateway = _validate_evidence_gateway(gateway)
    open_fn = opener or _DEFAULT_OPENER.open

    evidence = collect_evidence(
        gateway=gateway,
        model=model,
        include_tls_fingerprint=include_tls_fingerprint,
        timeout=timeout,
        opener=opener,
        verify_crypto=verify_crypto,
        accepted_tcb_statuses=accepted_tcb_statuses,
        pccs_url=pccs_url,
    )
    errors = list(evidence.get("errors") or [])

    request_body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 16,
        }
    ).encode("utf-8")

    chat_id: Optional[str] = None
    response_body = b""
    try:
        chat_req = urllib.request.Request(
            gateway + "/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "x-no-aliasing": "true",
            },
            method="POST",
        )
        with open_fn(chat_req, timeout=timeout) as resp:
            response_body = _read_bounded_response(resp)
        parsed_resp = json.loads(response_body.decode("utf-8"))
        chat_id = parsed_resp.get("id") if isinstance(parsed_resp, dict) else None
    except Exception as exc:
        errors.append(_safe_error(exc, "/chat/completions"))

    sig_payload: Any = None
    sig_status = 0
    if chat_id:
        sig_qs = urllib.parse.urlencode({"model": model, "signing_algo": signing_algo})
        sig_url = f"{gateway}/signature/{urllib.parse.quote(str(chat_id), safe='')}" f"?{sig_qs}"
        for attempt in range(max(1, max_sig_retries)):
            try:
                sig_req = urllib.request.Request(
                    sig_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                    },
                )
                with open_fn(sig_req, timeout=timeout) as resp:
                    sig_status = getattr(resp, "status", 200) or 200
                    sig_payload = json.loads(_read_bounded_response(resp).decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                sig_status = exc.code
                if exc.code == 404 and attempt + 1 < max_sig_retries:
                    time.sleep(retry_delay)
                    continue
                errors.append(_safe_error(exc, f"/signature/{chat_id}"))
                break
            except Exception as exc:
                errors.append(_safe_error(exc, f"/signature/{chat_id}"))
                break

    report = evidence.get("report") or {}
    if sig_payload is not None or sig_status:
        sig_ver = verify_response_signature(
            sig_payload,
            request_body,
            response_body,
            requested_model=model,
            gateway_attestation=report.get("gateway_attestation"),
            model_attestations=report.get("model_attestations"),
            http_status=sig_status or 200,
        )
    else:
        sig_ver = {"status": "unavailable", "verified": False, "reason": "chat_id unavailable"}

    evidence["response_signature_verification"] = sig_ver
    summary = build_verification_summary(evidence)
    return {
        "label": "evidence",
        "status": "unverified",
        "model": model,
        "chat_id": chat_id,
        "signing_algo": signing_algo,
        "request_hash": hash_exact_bytes(request_body),
        "response_hash": hash_exact_bytes(response_body) if response_body else None,
        "signature_verification": sig_ver,
        "verification_summary": summary,
        "errors": errors,
    }
