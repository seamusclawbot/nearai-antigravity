"""OHTTP and Compose Manager verification helpers."""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict, Optional


def verify_compose_manager_actions(
    actions: Any,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonicalize compose_manager_attestation actions using serde_json-compatible JSON."""
    if actions is None:
        return {"status": "unavailable", "verified": False, "reason": "missing actions"}
    try:
        canonical = json.dumps(
            actions,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return {"status": "failed", "verified": False, "reason": f"non-serializable actions: {type(exc).__name__}"}

    digest = hashlib.sha256(canonical).hexdigest()
    if expected_hash is not None:
        cleaned = expected_hash.strip().lower()
        if cleaned.startswith("0x"):
            cleaned = cleaned[2:]
        if not hmac.compare_digest(digest, cleaned):
            return {
                "status": "failed",
                "verified": False,
                "reason": "compose_manager actions hash mismatch",
                "sha256": digest,
            }
        return {"status": "verified", "verified": True, "sha256": digest}
    return {"status": "computed", "verified": False, "sha256": digest}
