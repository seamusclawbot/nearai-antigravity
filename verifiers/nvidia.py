"""NVIDIA GPU attestation verification for NEAR AI (NRAS) and Chutes."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
DEFAULT_NRAS_JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
DEFAULT_NRAS_ISSUER = "https://nras.attestation.nvidia.com"
ALLOWED_NRAS_ALGS: tuple[str, ...] = ("ES384",)
MAX_NVIDIA_HTTP_BYTES = 4 * 1024 * 1024  # 4 MiB


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _validate_verifier_url(url: str) -> str:
    u = (url or "").strip()
    if not u or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in u):
        raise ValueError("verifier URL must be non-empty without whitespace or control characters")
    p = urllib.parse.urlsplit(u)
    if p.scheme not in ("https", "http") or not p.hostname:
        raise ValueError("verifier URL must use https:// (or loopback http://)")
    if p.username is not None or p.password is not None or p.fragment:
        raise ValueError("verifier URL must not contain userinfo or fragment")
    if p.scheme == "http":
        host = p.hostname.strip().lower()
        is_loopback = host in ("localhost", "127.0.0.1", "::1")
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("http:// verifier URL is only permitted for loopback hosts")
    return u


def _read_bounded(resp: Any, max_bytes: int = MAX_NVIDIA_HTTP_BYTES) -> bytes:
    data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"verifier HTTP response exceeded {max_bytes} byte limit")
    return data


def _default_post_json(url: str, body_bytes: bytes, timeout: float = 15.0) -> Any:
    safe_url = _validate_verifier_url(url)
    req = urllib.request.Request(
        safe_url,
        data=body_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(_read_bounded(resp).decode("utf-8"))


def _default_get_json(url: str, timeout: float = 15.0) -> Any:
    safe_url = _validate_verifier_url(url)
    req = urllib.request.Request(safe_url, headers={"Accept": "application/json"})
    with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(_read_bounded(resp).decode("utf-8"))


def verify_nvidia_gpu_evidence(
    nvidia_payload: Any,
    request_nonce: str,
    *,
    nras_url: str = DEFAULT_NRAS_URL,
    jwks_url: str = DEFAULT_NRAS_JWKS_URL,
    expected_issuer: str = DEFAULT_NRAS_ISSUER,
    allowed_algorithms: Sequence[str] = ALLOWED_NRAS_ALGS,
    post_fn: Optional[Callable[[str, bytes], Any]] = None,
    jwks_fetch_fn: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Verify NEAR-operated NVIDIA GPU evidence via NRAS and NVIDIA JWKS."""
    if not nvidia_payload:
        return {"status": "unavailable", "verified": False, "reason": "missing nvidia_payload"}
    if not isinstance(request_nonce, str) or not request_nonce:
        return {"status": "failed", "verified": False, "reason": "missing request_nonce"}

    if isinstance(nvidia_payload, str):
        raw_bytes = nvidia_payload.encode("utf-8")
        try:
            payload_obj = json.loads(nvidia_payload)
        except ValueError:
            return {"status": "failed", "verified": False, "reason": "invalid nvidia_payload JSON"}
    elif isinstance(nvidia_payload, dict):
        payload_obj = nvidia_payload
        raw_bytes = json.dumps(nvidia_payload).encode("utf-8")
    else:
        return {"status": "failed", "verified": False, "reason": "unsupported nvidia_payload type"}

    payload_nonce = payload_obj.get("nonce") if isinstance(payload_obj, dict) else None
    if not isinstance(payload_nonce, str) or not hmac.compare_digest(payload_nonce.lower(), request_nonce.lower()):
        return {"status": "failed", "verified": False, "reason": "gpu.payload_invalid: nonce mismatch"}

    algs = [a for a in allowed_algorithms if isinstance(a, str) and a and a.lower() != "none" and not a.upper().startswith("HS")]
    if not algs:
        return {"status": "failed", "verified": False, "reason": "invalid allowed_algorithms policy"}

    try:
        import jwt
    except ImportError:
        return {"status": "unavailable", "verified": False, "reason": "PyJWT package unavailable"}

    post = post_fn or (lambda u, b: _default_post_json(u, b))
    fetch_jwks = jwks_fetch_fn or (lambda u: _default_get_json(u))
    try:
        nras_resp = post(nras_url, raw_bytes)
        jwks_data = fetch_jwks(jwks_url)
    except Exception as exc:
        return {"status": "unavailable", "verified": False, "reason": f"NRAS/JWKS fetch error: {type(exc).__name__}"}

    if not isinstance(nras_resp, list) or not nras_resp:
        return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: empty NRAS response"}

    try:
        jwk_set = jwt.PyJWKSet.from_dict(jwks_data)
    except Exception as exc:
        return {"status": "failed", "verified": False, "reason": f"invalid JWKS: {type(exc).__name__}"}
    if not getattr(jwk_set, "keys", None):
        return {"status": "failed", "verified": False, "reason": "invalid JWKS: empty key set"}

    token_entries: List[tuple[str, str, bool]] = []
    for item in nras_resp:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "JWT" and isinstance(item[1], str):
            token_entries.append(("overall", item[1], True))
        elif isinstance(item, dict):
            for gpu_id, tok in item.items():
                if not isinstance(tok, str):
                    return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: unexpected NRAS dict entry"}
                token_entries.append((str(gpu_id), tok, False))
        else:
            return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: unexpected NRAS entry"}

    if not any(is_overall for _, _, is_overall in token_entries):
        return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: missing overall NRAS JWT"}

    verified_claims: List[Dict[str, Any]] = []
    for label, token, is_overall in token_entries:
        try:
            header = jwt.get_unverified_header(token)
            alg = header.get("alg")
            if alg not in algs:
                return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: disallowed JWT alg"}
            kid = header.get("kid")
            signing_key = jwk_set[kid] if kid else jwk_set.keys[0]
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=algs,
                issuer=expected_issuer,
                options={"require": ["exp", "nbf", "iat", "iss", "eat_nonce"]},
            )
        except Exception as exc:
            return {"status": "failed", "verified": False, "reason": f"gpu.jwt_verification_failed: {type(exc).__name__}"}

        eat_nonce = claims.get("eat_nonce")
        if not isinstance(eat_nonce, str) or not hmac.compare_digest(eat_nonce.lower(), request_nonce.lower()):
            return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: eat_nonce mismatch"}
        if is_overall:
            if claims.get("x-nvidia-overall-att-result") is not True:
                return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: att_result not true"}
        else:
            measres = claims.get("measres")
            if measres != "success":
                return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: per-GPU measres not success"}
            if "secboot" in claims and claims.get("secboot") is not True:
                return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: per-GPU secboot disabled"}
            if "dbgstat" in claims and claims.get("dbgstat") != "disabled":
                return {"status": "failed", "verified": False, "reason": "gpu.jwt_verification_failed: per-GPU debug enabled"}
        verified_claims.append({
            "label": label,
            "sub": claims.get("sub"),
            "eat_nonce": eat_nonce,
            "overall": claims.get("x-nvidia-overall-att-result") if is_overall else claims.get("measres"),
        })

    return {"status": "verified", "verified": True, "gpu_count": len(verified_claims), "claims": verified_claims}


def verify_chutes_gpu_evidence(
    item: Dict[str, Any],
    *,
    spdm_verifier: Optional[Callable[[List[Dict[str, Any]], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Verify Chutes GPU challenge binding and optional per-GPU SPDM evidence."""
    nonce = item.get("nonce")
    e2e_pubkey = item.get("e2e_pubkey")
    report_data = item.get("report_data")
    if not (isinstance(nonce, str) and nonce and isinstance(e2e_pubkey, str) and e2e_pubkey):
        return {"status": "unavailable", "verified": False, "reason": "missing nonce or e2e_pubkey"}

    challenge = hashlib.sha256((nonce + e2e_pubkey).encode("utf-8")).hexdigest()
    rd_bytes: Optional[bytes] = None
    if isinstance(report_data, bytes):
        rd_bytes = report_data
    elif isinstance(report_data, str):
        cleaned = report_data[2:] if report_data.lower().startswith("0x") else report_data
        try:
            rd_bytes = bytes.fromhex(cleaned)
        except ValueError:
            rd_bytes = None
    if not rd_bytes or len(rd_bytes) != 64 or not hmac.compare_digest(rd_bytes[:32], bytes.fromhex(challenge)):
        return {"status": "failed", "verified": False, "reason": "chutes GPU challenge report_data mismatch"}

    gpu_ev = item.get("gpu_evidence")
    if not isinstance(gpu_ev, list) or not gpu_ev:
        return {"status": "unavailable", "verified": False, "reason": "missing gpu_evidence", "challenge_sha256": challenge}
    if spdm_verifier is None:
        return {
            "status": "unavailable",
            "verified": False,
            "reason": "NVIDIA Attestation SDK SPDM verifier unavailable",
            "challenge_sha256": challenge,
        }
    res = spdm_verifier(gpu_ev, challenge)
    res.setdefault("challenge_sha256", challenge)
    return res
