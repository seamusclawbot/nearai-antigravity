"""Intel TDX quote verification using dcap-qvl and NEAR AI policy checks."""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import hmac
import ipaddress
import json
import struct
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set


def _validate_pccs_url(url: str) -> str:
    u = (url or "").strip()
    if not u or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in u):
        raise ValueError("invalid pccs_url")
    p = urllib.parse.urlsplit(u)
    if p.scheme not in ("https", "http") or not p.hostname:
        raise ValueError("pccs_url must use https:// (or loopback http://)")
    if p.username is not None or p.password is not None or p.fragment:
        raise ValueError("pccs_url must not contain userinfo or fragment")
    if p.scheme == "http":
        host = p.hostname.strip().lower()
        is_loopback = host in ("localhost", "127.0.0.1", "::1")
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("http:// pccs_url is only permitted for loopback hosts")
    return u


def _decode_quote_bytes(value: Any) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.lower().startswith("0x"):
        text = text[2:]
    if len(text) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in text):
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    try:
        return base64.b64decode(value.strip(), validate=True)
    except Exception:
        return None


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


def compute_mr_config_id(app_compose: str) -> bytes:
    """Compute expected 48-byte MRCONFIGID from raw app_compose UTF-8."""
    if not isinstance(app_compose, str):
        raise TypeError("app_compose must be a str")
    return b"\x01" + hashlib.sha256(app_compose.encode("utf-8")).digest() + (b"\x00" * 15)


def replay_rtmr3(event_log: Any) -> bytes:
    """Replay RTMR3 over IMR=3 event_log entries using SHA-384."""
    if isinstance(event_log, str):
        event_log = json.loads(event_log)
    if not isinstance(event_log, list):
        raise TypeError("event_log must be a list")
    rtmr3 = b"\x00" * 48
    for entry in event_log:
        if not isinstance(entry, dict) or entry.get("imr") != 3:
            continue
        event_type = int(entry["event_type"])
        event_name = str(entry.get("event") or "").encode("utf-8")
        payload_hex = entry.get("event_payload") or ""
        payload_bytes = bytes.fromhex(payload_hex)
        digest = hashlib.sha384(
            struct.pack("<I", event_type) + b":" + event_name + b":" + payload_bytes
        ).digest()
        expected_digest = entry.get("digest")
        if expected_digest:
            if not hmac.compare_digest(bytes.fromhex(str(expected_digest)), digest):
                raise ValueError("event_log entry digest mismatch")
        rtmr3 = hashlib.sha384(rtmr3 + digest).digest()
    return rtmr3


def _run_coro_factory(coro_factory: Any) -> Any:
    async def _runner() -> Any:
        return await coro_factory()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_runner())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _runner()).result()


def _check_report_data(
    report_data: bytes,
    nonce: str,
    signing_address: Optional[str],
    signing_algo: Optional[str],
    tls_cert_fingerprint: Optional[str],
    include_tls_fingerprint: bool,
) -> Optional[str]:
    if len(report_data) != 64:
        return "invalid report_data length"
    nonce_bytes = _parse_hex_bytes(nonce, expected_len=32)
    if nonce_bytes is None:
        return "invalid 32-byte hex nonce"
    if not hmac.compare_digest(report_data[32:64], nonce_bytes):
        return "report_data nonce mismatch"
    if signing_address is None:
        return "missing signing_address for report_data check"
    ident = _parse_hex_bytes(signing_address)
    algo = signing_algo.lower() if isinstance(signing_algo, str) else None
    if ident is None or (algo == "ecdsa" and len(ident) != 20) or (algo == "ed25519" and len(ident) != 32) or (algo is None and len(ident) not in (20, 32)):
        return "invalid signing_address"
    if include_tls_fingerprint:
        tls_bytes = _parse_hex_bytes(tls_cert_fingerprint, expected_len=32)
        if tls_bytes is None:
            return "missing or invalid tls_cert_fingerprint"
        expected_prefix = hashlib.sha256(ident + tls_bytes).digest()
    else:
        expected_prefix = ident.ljust(32, b"\x00")
    if not hmac.compare_digest(report_data[:32], expected_prefix):
        return "report_data prefix does not bind signer/TLS fingerprint"
    return None


def verify_tdx_quote(
    quote_input: Any,
    *,
    nonce: Optional[str] = None,
    signing_address: Optional[str] = None,
    signing_algo: Optional[str] = None,
    tls_cert_fingerprint: Optional[str] = None,
    include_tls_fingerprint: bool = True,
    app_compose: Optional[str] = None,
    event_log: Optional[List[Dict[str, Any]]] = None,
    collateral: Any = None,
    pccs_url: Optional[str] = None,
    now_secs: Optional[int] = None,
    accepted_tcb_statuses: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Verify an Intel TDX quote via dcap-qvl plus NEAR AI policy checks."""
    raw_quote = _decode_quote_bytes(quote_input)
    if not raw_quote:
        return {"status": "unavailable", "verified": False, "reason": "missing or invalid quote bytes"}
    try:
        import dcap_qvl
    except ImportError:
        return {"status": "unavailable", "verified": False, "reason": "dcap-qvl package unavailable"}

    try:
        parsed = dcap_qvl.parse_quote(raw_quote)
        td_report = parsed.report
    except Exception as exc:
        return {"status": "failed", "verified": False, "reason": f"quote parse error: {type(exc).__name__}"}

    td_attrs = bytes(getattr(td_report, "td_attributes", b""))
    if not td_attrs or (td_attrs[0] & 1) != 0:
        return {"status": "failed", "verified": False, "reason": "TDX debug mode is enabled"}

    if collateral is not None:
        if isinstance(collateral, dcap_qvl.QuoteCollateralV3):
            coll_obj = collateral
        elif isinstance(collateral, dict):
            coll_obj = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(collateral))
        elif isinstance(collateral, str):
            coll_obj = dcap_qvl.QuoteCollateralV3.from_json(collateral)
        else:
            return {"status": "failed", "verified": False, "reason": "invalid collateral type"}
    else:
        raw_url = (pccs_url or "").strip() or dcap_qvl.PHALA_PCCS_URL
        try:
            url = _validate_pccs_url(raw_url)
        except ValueError as exc:
            return {"status": "failed", "verified": False, "reason": str(exc)}
        try:
            coll_obj = _run_coro_factory(lambda: dcap_qvl.get_collateral(url, raw_quote))
        except Exception as exc:
            return {"status": "unavailable", "verified": False, "reason": f"collateral fetch error: {type(exc).__name__}"}

    ts = int(time.time()) if now_secs is None else int(now_secs)
    try:
        vr = dcap_qvl.verify(raw_quote, coll_obj, ts)
    except Exception as exc:
        return {"status": "failed", "verified": False, "reason": f"dcap-qvl verify error: {type(exc).__name__}: {exc}"}

    allowed_tcb = accepted_tcb_statuses if accepted_tcb_statuses is not None else {"UpToDate"}
    if vr.status not in allowed_tcb:
        return {
            "status": "failed",
            "verified": False,
            "reason": f"unacceptable TCB status: {vr.status}",
            "tcb_status": vr.status,
            "advisory_ids": list(vr.advisory_ids),
        }

    if app_compose is not None:
        expected_mr_config = compute_mr_config_id(app_compose)
        if not hmac.compare_digest(bytes(td_report.mr_config_id), expected_mr_config):
            return {"status": "failed", "verified": False, "reason": "mr_config_id mismatch with app_compose"}

    if event_log is not None:
        try:
            expected_rtmr3 = replay_rtmr3(event_log)
        except Exception as exc:
            return {"status": "failed", "verified": False, "reason": f"event_log replay error: {exc}"}
        if not hmac.compare_digest(bytes(td_report.rt_mr3), expected_rtmr3):
            return {"status": "failed", "verified": False, "reason": "rt_mr3 mismatch with event_log"}

    if nonce is not None:
        rd_err = _check_report_data(
            bytes(td_report.report_data),
            nonce=nonce,
            signing_address=signing_address,
            signing_algo=signing_algo,
            tls_cert_fingerprint=tls_cert_fingerprint,
            include_tls_fingerprint=include_tls_fingerprint,
        )
        if rd_err is not None:
            return {"status": "failed", "verified": False, "reason": rd_err}

    return {
        "status": "verified",
        "verified": True,
        "tcb_status": vr.status,
        "advisory_ids": list(vr.advisory_ids),
    }
