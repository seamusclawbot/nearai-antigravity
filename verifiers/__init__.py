"""Cryptographic verifiers for resolved NEAR AI Gateway contracts."""
from __future__ import annotations

from verifiers.nvidia import verify_chutes_gpu_evidence, verify_nvidia_gpu_evidence
from verifiers.ohttp import verify_compose_manager_actions
from verifiers.signatures import hash_exact_bytes, verify_response_signature
from verifiers.tdx import compute_mr_config_id, replay_rtmr3, verify_tdx_quote

__all__ = [
    "compute_mr_config_id",
    "hash_exact_bytes",
    "replay_rtmr3",
    "verify_chutes_gpu_evidence",
    "verify_compose_manager_actions",
    "verify_nvidia_gpu_evidence",
    "verify_response_signature",
    "verify_tdx_quote",
]
