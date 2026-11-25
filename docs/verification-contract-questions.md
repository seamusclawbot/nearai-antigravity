# Verification Contract Questions for NEAR AI

This document is the question inventory for a future independent-research effort and, if needed, a NEAR AI support request. It deliberately distinguishes facts already documented by NEAR AI from verifier-contract gaps. A field's presence, a successful HTTP response, or a locally computed digest is insufficient to establish TEE authenticity.

## 1. Questions already answered by the official API Reference

The rendered API Reference documents the following contract elements:

- Attestation reports accept `near`, `chutes`, and `tinfoil`; Chutes evidence can include per-GPU certificate/SPDM material.
- The GPU challenge is derived from the textual nonce and textual base64 E2E public key as documented by the endpoint.
- `include_tls_fingerprint=true` binds report data to `SHA256(signing_address || tls_cert_fingerprint)`, where the fingerprint is for the TLS SPKI.
- ITA responses expose `jwks_url`, policy IDs, `policy_must_match`, nonce, model entries, and model-alias resolution.
- Response signatures distinguish `provider_tee` from `gateway`; the documented gateway signature does **not** prove model execution inside a TEE.
- Historical signatures may omit `signature_kind`, so their provenance is unknown.

These findings are authoritative documentation findings, not a complete independent-verification specification. The source URL and retrieval date should be recorded in any future research notes.

## 2. Questions requiring independent research first — RESOLVED

All four areas in Section 2 have been researched against `https://docs.near.ai/cloud/verification/`, `nearai/cloud-api`, `nearai/nearai-cloud-verifier`, `nearai/inference-sdk`, Chutes' TEE docs, Tinfoil's verifier, and Intel/NVIDIA/IETF specifications. See [`docs/verification-contract-answers.md`](verification-contract-answers.md) for the full evidence-formatted answers.

- **2.1 Intel TDX quote verification:** Resolved (`dcap-qvl` `TD10`, Phala PCCS / Intel PCS, `td_attributes[0] & 1` debug rejection, `MRCONFIGID` = `0x01 || SHA-256(app_compose) || 0x00*15`, `RTMR3` SHA-384 event-log replay, 64-byte `report_data` layout).
- **2.2 NVIDIA GPU evidence:** Resolved (NEAR uses NRAS `/v3/attest/gpu` + JWKS JWT validation with `x-nvidia-overall-att-result == true`; Chutes uses NVIDIA Attestation SDK + SPDM + `SHA256(UTF8(nonce + e2e_pubkey))`).
- **2.3 ITA/JWT conventions & `cloud-api` config:** Resolved (`PS384` / `RS256`, `iss` = `https://portal.trustauthority.intel.com`, `/certs` JWKS with `kid` rotation, `policy_must_match`, `ITA_POLICY_IDS` up to 10 UUIDs in `nearai/cloud-api`). Production `ITA_POLICY_IDS` & pinning rules remain in Section 3.1.
- **2.4 Cryptographic and serialization standards:** Resolved (RFC 9458/9292/9180 for OHTTP/BHTTP/HPKE, `ohttp_gateway.rs` KMS key derivation + Ed25519 signature over decoded `key_config`; `serde_json`-compatible canonicalization for `compose_manager_attestation`; exact-byte SHA-256 for response signatures).

## 3. Questions that require NEAR AI confirmation

Sections 3.3 and most of 3.2/3.4 are now documented in [`docs/verification-contract-answers.md`](verification-contract-answers.md). Only the remaining operational and roadmap gaps below require NEAR AI confirmation.

### 3.1 ITA token and policy semantics

- Which production `ITA_POLICY_IDS` UUIDs (and policy definitions) should clients expect and enforce?
- What clock-skew tolerance and expiration/replay rules apply to `/v1/attestation/ita-token` JWTs?
- How should `jwks_url` returned by `/v1/attestation/ita-token` be pinned/allowlisted (e.g., restricted to `portal.trustauthority.intel.com` / `portal.eu.trustauthority.intel.com`)?

### 3.2 Signing keys and provider mapping (remaining gaps)

- How are shared dstack KMS signing keys (used for `ohttp_gateway` and shared model instances) rotated or revoked?
- What is the `/v1/signature/{chat_id}` contract (if any) for non-NEAR providers (`chutes`, `tinfoil`)—e.g., `gateway`-only vs `provider_tee`?

### 3.3 Canonical request/response hashing — RESOLVED

Resolved in [`docs/verification-contract-answers.md`](verification-contract-answers.md): no JSON canonicalization is used; `request_hash` and `response_hash` are SHA-256 over exact request-body and uncompressed response-body bytes (including SSE framing), signed via EIP-191 (`ecdsa`) or UTF-8 detached (`ed25519`).

### 3.4 End-to-end binding and failure behavior (remaining gaps)

- Is a chained model-to-Gateway-to-client proof planned (to close the documented gap where `gateway` signatures do not prove model TEE execution and `provider_tee` signatures do not bind to Gateway TLS)?
- Which endpoint/API version is the long-term stability contract for verification clients, and how are breaking changes announced?

## 4. Required evidence format for every answer

For each resolved question, record:

1. **Authoritative source** — URL, repository path, document version, or signed specification.
2. **Verifier implementation** — library/service name, exact version or commit, supported platforms, and configuration.
3. **Trust roots and collateral** — root certificates/keys, retrieval endpoint, pinning/rotation rules, and revocation source.
4. **Test vector or fixture** — sanitized quote/token/signature/response fixture, expected digest/signature, and expected verification result.
5. **Negative cases** — at least one invalid, expired, mismatched, revoked, or tampered fixture and the expected failure code.
6. **Freshness and lifecycle** — validity interval, replay protection, cache policy, and key/policy rotation behavior.

Do not attach live credentials, private keys, raw production JWTs, raw quotes, certificates, or sensitive response bodies. Use sanitized or vendor-published fixtures.

## 5. Support-ticket-ready request outline

If public research leaves material gaps, open a support request with this concise context:

> We are implementing a conservative client for the NEAR AI Gateway. We currently collect and correlate attestation evidence, enforce freshness and documented report-data bindings, preserve `overall: unverified`, and fail closed for the private/attestation-required profile. We do not claim independent TEE verification from metadata, quotes, certificates, ITA tokens, or gateway signatures alone.
>
> Please provide the authoritative verifier contract for the questions in Sections 3.1–3.4, including source URLs or repositories, verifier versions, trust-root/collateral configuration, canonicalization rules, and sanitized positive/negative test vectors. In particular, please clarify the distinction between gateway signatures and provider/model-execution evidence, and the exact binding from quote/policy/model/replica to the client response.

Attach only sanitized diagnostics and repository references. Do not send credentials or raw production evidence.

## Acceptance criterion

Do not change `overall: unverified` or claim independent TEE verification until each applicable answer is authoritative, testable, and covered by a maintained verifier implementation and regression fixtures that contain no live secrets or raw production evidence.
