# NEAR AI Gateway Capability Matrix

**Sources checked:** Gateway OpenAPI at `https://cloud-api.near.ai/api-docs/openapi.json`, gateway base `https://cloud-api.near.ai/v1`, and local implementation. Checked 2026-11-13/14. This document distinguishes documented behavior from local observations; presence of an attestation field is not verification.

| Capability | Status | Notes |
|---|---|---|
| Model discovery | Documented/observed | `GET /v1/models`; local catalog handling also merges provider/privacy metadata where available. |
| Chat completions | Documented | `POST /v1/chat/completions`; OpenAI-shaped request/response. |
| Text completions | Documented | `POST /v1/completions`. |
| Gateway attestation report | Documented | `GET /v1/attestation/report`; report-data layout documents nonce binding and optional TLS binding. |
| Attestation pins | Documented | `GET /v1/attestation/pins`; pin semantics and lifecycle require authoritative clarification. |
| ITA token | Documented shape only | `GET /v1/attestation/ita-token`; response exposes nonce, policy IDs, JWKS URL, token types, and signing algorithm. Issuer/claim/policy semantics remain unresolved. |
| Response signatures | Documented shape only | `GET /v1/signature/{chat_id}`; documented payloads cover gateway or provider response hashes, but canonical bytes and key registry remain unresolved. |
| OHTTP signing | Locally checked evidence | The collector can perform narrowly scoped encoding/signature checks when exact inputs are supplied. This does not establish TEE authenticity. |
| Model attestation metadata | Advisory | May include quote, certificate, GPU evidence, nonce, and E2E key fields. Shape validation and correlation are implemented; quote/certificate trust validation is not. |
| Provider trust classes | Local policy | `vllm`, `chutes`, and `tinfoil` are recognized TEE classes. Unknown/legacy/external providers are not admitted by TEE-only selection. |
| Private profile routing | Fail-closed | `profiles/private.json` requires attestation. Fresh evidence, correlation, TLS/OHTTP/report binding, gateway quote verification, and model execution verification are required; unavailable authoritative verifiers cause rejection. |

## Documented report-data and challenge semantics

The OpenAPI contract states that:

- The final 32 bytes of `report_data` contain the request nonce.
- When TLS binding is used, the first 32 bytes bind `SHA256(signing_address || tls_cert_fingerprint)`.
- The GPU challenge is `SHA256(UTF8(nonce + e2e_pubkey))`, using returned textual values without decoding.
- OHTTP Ed25519 signatures are checked over decoded `key_config` bytes against `signing_key`.

These are input-binding rules, not a complete verifier. They do not by themselves establish quote authenticity, measurement policy, model execution, or provider identity.

## API Reference findings (reviewed 2026-11-13)

The official API Reference adds useful contract detail beyond the machine-readable OpenAPI summary:

- The report endpoint is public/non-billable; API keys are optional for listed models.
- `provider` accepts `near`, `chutes`, and `tinfoil`; omission selects the first responding provider.
- Chutes evidence can include per-GPU certificate chains and SPDM evidence. The GPU challenge uses the returned nonce hex text and E2E public-key base64 text without decoding.
- ITA responses are described as carrying only ITA-signed TEE evidence and expose `jwks_url`, policy IDs, `policy_must_match`, nonce, model entries, and alias resolution.
- Response signatures distinguish `provider_tee` from `gateway`; gateway signatures cover exact client-facing bytes and explicitly do not prove model execution inside a TEE. Historical signatures may omit `signature_kind` and have unknown provenance.

These findings resolve endpoint shape and binding questions, but not quote verifier libraries, trust roots, collateral, revocation, ITA issuer/claims/expiry rules, signing-key registries, or canonical request/response hashing.


1. TDX/GPU verifier libraries, trust roots, collateral retrieval and expiry, TCB/security-version policy, event logs, SPDM policy, and revocation.
2. ITA issuer/audience/claim requirements, clock skew, expiry, JWKS rotation, and `policy_must_match` semantics.
3. Signing-key registry, root of trust, rotation/revocation, and mapping from gateway/provider/model.
4. Canonical request and response serialization, hash algorithms, streaming/tool-call behavior, and Unicode/JSON normalization.

Until those answers are supplied by an authoritative source, consume this matrix for capability planning and operational gating—not as a cryptographic assurance.
