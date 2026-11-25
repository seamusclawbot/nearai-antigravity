"""Exact-byte response signature verification for NEAR AI Gateway."""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, List, Optional


def hash_exact_bytes(data: bytes) -> str:
    """Return lowercase hexadecimal SHA-256 digest of exact raw bytes."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def _parse_hex_bytes(value: Any, expected_len: Optional[int] = None) -> Optional[bytes]:
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


def _verify_ecdsa_eip191(text: str, signature_hex: str, expected_address: str) -> Dict[str, Any]:
    sig_bytes = _parse_hex_bytes(signature_hex, expected_len=65)
    addr_bytes = _parse_hex_bytes(expected_address, expected_len=20)
    if sig_bytes is None or addr_bytes is None:
        return {"status": "failed", "verified": False, "reason": "invalid ecdsa signature or address format"}
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        return {"status": "unavailable", "verified": False, "reason": "eth-account package unavailable"}
    try:
        recovered = Account.recover_message(encode_defunct(text=text), signature=sig_bytes)
    except Exception as exc:
        return {"status": "failed", "verified": False, "reason": f"ecdsa recovery error: {type(exc).__name__}"}
    if not hmac.compare_digest(recovered.lower(), "0x" + addr_bytes.hex()):
        return {"status": "failed", "verified": False, "reason": "ecdsa recovered address mismatch"}
    return {"status": "verified", "verified": True}


def _verify_ed25519(text: str, signature_hex: str, expected_pubkey: str) -> Dict[str, Any]:
    sig_bytes = _parse_hex_bytes(signature_hex, expected_len=64)
    key_bytes = _parse_hex_bytes(expected_pubkey, expected_len=32)
    if sig_bytes is None or key_bytes is None:
        return {"status": "failed", "verified": False, "reason": "invalid ed25519 signature or pubkey format"}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return {"status": "unavailable", "verified": False, "reason": "cryptography package unavailable"}
    try:
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(sig_bytes, text.encode("utf-8"))
    except Exception as exc:
        return {"status": "failed", "verified": False, "reason": f"ed25519 verification error: {type(exc).__name__}"}
    return {"status": "verified", "verified": True}


def _matches_signer(candidate: Dict[str, Any], signing_address: str, signing_algo: str) -> bool:
    cand_algo = candidate.get("signing_algo")
    cand_addr = candidate.get("signing_address")
    if not isinstance(cand_algo, str) or not isinstance(cand_addr, str):
        return False
    if cand_algo.lower() != signing_algo.lower():
        return False
    norm_cand = _parse_hex_bytes(cand_addr)
    norm_sig = _parse_hex_bytes(signing_address)
    return norm_cand is not None and norm_sig is not None and hmac.compare_digest(norm_cand, norm_sig)


def verify_response_signature(
    payload: Any,
    request_body: bytes,
    response_body: bytes,
    *,
    requested_model: str = "",
    gateway_attestation: Optional[Dict[str, Any]] = None,
    model_attestations: Optional[List[Dict[str, Any]]] = None,
    http_status: int = 200,
) -> Dict[str, Any]:
    """Verify a /v1/signature/{chat_id} response against exact wire bytes.

    Fail-closed rules per docs.near.ai:
    - 404 or non-2xx, or 2xx containing error_code + message -> unavailable.
    - Missing/unfamiliar signature_kind -> unavailable (never infer from text).
    - No JSON canonicalization: hashes exact request_body and uncompressed response_body.
    - provider_tee requires unique match among verified model_attestations when provided.
    """
    if http_status == 404 or not (200 <= http_status < 300):
        return {"status": "unavailable", "verified": False, "reason": f"http_status_{http_status}"}
    if not isinstance(payload, dict):
        return {"status": "unavailable", "verified": False, "reason": "missing signature payload"}
    if "error_code" in payload and "message" in payload:
        return {"status": "unavailable", "verified": False, "reason": f"signature error: {payload.get('error_code')}"}

    kind = payload.get("signature_kind")
    if kind not in ("gateway", "provider_tee"):
        return {"status": "unavailable", "verified": False, "reason": "absent or unfamiliar signature_kind"}

    required = ("text", "signature", "signing_address", "signing_algo")
    if not all(isinstance(payload.get(k), str) and payload.get(k) for k in required):
        return {"status": "failed", "verified": False, "reason": "incomplete signature fields", "signature_kind": kind}

    text = payload["text"]
    signature = payload["signature"]
    signing_address = payload["signing_address"]
    signing_algo = payload["signing_algo"].lower()
    if signing_algo not in ("ecdsa", "ed25519"):
        return {"status": "unavailable", "verified": False, "reason": f"unsupported signing_algo: {signing_algo}", "signature_kind": kind}

    req_hash = hash_exact_bytes(request_body)
    resp_hash = hash_exact_bytes(response_body)

    if kind == "gateway":
        expected_text = f"{req_hash}:{resp_hash}"
        if not hmac.compare_digest(text, expected_text):
            return {"status": "failed", "verified": False, "reason": "gateway text hash mismatch", "signature_kind": kind}
        if gateway_attestation is not None and not _matches_signer(gateway_attestation, signing_address, signing_algo):
            return {"status": "failed", "verified": False, "reason": "signer does not match gateway_attestation", "signature_kind": kind}
        provenance = "gateway_only"
    else:
        parts = text.rsplit(":", 2)
        if len(parts) != 3 or not parts[0]:
            return {"status": "failed", "verified": False, "reason": "invalid provider_tee text shape", "signature_kind": kind}
        if requested_model and parts[0] != requested_model:
            return {"status": "failed", "verified": False, "reason": "provider_tee model mismatch", "signature_kind": kind}
        if not (hmac.compare_digest(parts[1], req_hash) and hmac.compare_digest(parts[2], resp_hash)):
            return {"status": "failed", "verified": False, "reason": "provider_tee text hash mismatch", "signature_kind": kind}
        if model_attestations is not None:
            matches = [m for m in model_attestations if isinstance(m, dict) and _matches_signer(m, signing_address, signing_algo)]
            if len(matches) != 1:
                return {
                    "status": "failed",
                    "verified": False,
                    "reason": f"expected 1 matching model_attestation candidate, found {len(matches)}",
                    "signature_kind": kind,
                }
        provenance = "provider_tee"

    crypto_res = (
        _verify_ecdsa_eip191(text, signature, signing_address)
        if signing_algo == "ecdsa"
        else _verify_ed25519(text, signature, signing_address)
    )
    crypto_res["signature_kind"] = kind
    crypto_res["provenance"] = provenance
    crypto_res["request_hash"] = req_hash
    crypto_res["response_hash"] = resp_hash
    return crypto_res
