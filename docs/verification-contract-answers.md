# NEAR AI Verification Contract — Resolved Answers

**Retrieved / Verified:** 2026-11-13  
**Primary Sources:**
- NEAR AI Verification Documentation: `https://docs.near.ai/cloud/verification/`
- NEAR AI API Reference: `https://docs.near.ai/api-reference/`
- NEAR AI Gateway Backend (Rust): `https://github.com/nearai/cloud-api`
- Official Reference Verifier: `https://github.com/nearai/nearai-cloud-verifier`
- Official Inference SDK: `https://github.com/nearai/inference-sdk`
- Chutes TEE Verification Docs: `https://github.com/chutesai/chutes-api/blob/main/docs/tee-verification.md`
- Tinfoil Verifier: `https://github.com/tinfoilsh/tinfoil-go` (`@tinfoilsh/verifier`)
- NEAR AI Blog & Papers (high-level threat model only): `https://near.ai/blog`, `https://near.ai/papers` ("Proof of Response")

Every resolved item below follows the evidence requirements in [`docs/verification-contract-questions.md`](verification-contract-questions.md) Section 4. No live secrets, private keys, raw production JWTs, or raw production quotes are included.

---

## 2.1 Intel TDX Quote Verification

- **Authoritative source:**
  - `https://docs.near.ai/cloud/verification/reference/quote-nonce-signer.md`
  - `https://docs.near.ai/cloud/verification/cloud-api/gateway-attestation.md`
  - `https://docs.near.ai/cloud/verification/cloud-api/model-attestations.md`
  - `https://github.com/nearai/nearai-cloud-verifier/blob/main/py/model_verifier.py`
- **Verifier implementation:**
  - `dcap-qvl` (`https://github.com/Phala-Network/dcap-qvl`), pure-Rust DCAP Quote Verification Library with language bindings:
    - Python: `dcap-qvl>=0.3.2` (`requirements.txt` in `nearai-cloud-verifier`)
    - Node.js/TypeScript: `@phala/dcap-qvl-node@^0.3.3` (`package.json` in `nearai-cloud-verifier`)
    - Go: FFI/WASM bindings in `nearai/inference-sdk/go/internal/dcap/`
  - Supported quote format: Intel TDX `TD10` (DCAP v4).
- **Trust roots and collateral:**
  - `dcap-qvl.get_collateral_and_verify()` fetches Intel DCAP collateral (Intel Root CA, PCK cert chain, CRLs, TCB Info, QE Identity) from Phala's PCCS by default, or Intel PCS when `INTEL_PCS_URL` is passed as the second argument.
  - **TCB & Debug Policy:**
    - Reject any quote with TDX debug mode enabled (`td_attributes[0] & 1 != 0`).
    - Policy-configurable accepted TCB statuses (docs example: `{"UpToDate"}`; `nearai-cloud-verifier` accepts `{"UpToDate", "OutOfDate"}` because `OutOfDate` indicates TCB advisory level rather than a cryptographic failure).
  - **Configuration & Event-Log Binding:**
    - `MRCONFIGID` (48 bytes) must equal `0x01 || SHA-256(raw app_compose UTF-8) || 15 zero bytes`. `app_compose` must be hashed as raw UTF-8 without JSON parsing or re-serialization.
    - `RTMR3` (48 bytes) is replayed over `event_log` entries with `imr == 3` in order, starting from `0x00 * 48`:
      - `digest = SHA-384(uint32_le(event_type) || b":" || event.encode("utf-8") || b":" || bytes.fromhex(event_payload or ""))`
      - If `event.digest` is non-empty, require `bytes.fromhex(event.digest) == digest`.
      - `rtmr3 = SHA-384(rtmr3 || digest)`
      - Require final `rtmr3 == quote.rt_mr3`.
  - **Report-Data Binding (64 bytes):**
    - `report_data[32:64]` = `bytes.fromhex(nonce)` (32-byte client-generated nonce).
    - Without TLS binding: `report_data[0:32]` = `identity_bytes.ljust(32, b"\x00")` (`ecdsa`: 20-byte ETH address; `ed25519`: 32-byte pubkey).
    - With TLS binding (`include_tls_fingerprint=true`): `report_data[0:32]` = `SHA-256(identity_bytes || bytes.fromhex(tls_cert_fingerprint))`, where `tls_cert_fingerprint` is the 32-byte SHA-256 of the peer certificate's DER-encoded `SubjectPublicKeyInfo` (SPKI) observed on the **same TLS connection**.
- **Test vector / fixture:**
  - Public sanitized fixtures in `nearai/inference-sdk`: `go/internal/dcap/testdata/tdx_quote`, `tdx_quote_collateral.json`, and `native_verified_report.json`.
- **Negative cases:**
  - `td_attributes[0] & 1 != 0` → reject (`TDX debug mode is enabled`).
  - `report_data` mismatch with client nonce or SPKI hash → reject (`report data does not bind the signer and nonce`).
  - `mr_config_id` mismatch with `app_compose` → reject.
  - `rt_mr3` mismatch with replayed `event_log` → reject.
- **Freshness and lifecycle:**
  - Freshness is established per request by a client-generated 32-byte random nonce (64 hex chars). Reports are not valid across TLS reconnects when TLS binding is required.

---

## 2.2 NVIDIA GPU Evidence

- **Authoritative source:**
  - NEAR-operated models: `https://docs.near.ai/cloud/verification/reference/quote-nonce-signer.md#verify-gpu-evidence-when-present` and `nearai/inference-sdk/js/src/boundaries/nvidia.ts`
  - Chutes provider: `https://docs.near.ai/api-reference/attestation/get-attestation-report.md` and `https://github.com/chutesai/chutes-api/blob/main/docs/tee-verification.md`
- **Verifier implementation:**
  - **NEAR-operated (`provider=near`):** `nvidia_payload` (JSON string) is decoded, checked for `payload.nonce.lower() == request_nonce.lower()`, and POSTed unchanged to NVIDIA Remote Attestation Service (NRAS): `https://nras.attestation.nvidia.com/v3/attest/gpu`.
  - **Chutes (`provider=chutes`):** Per-GPU `gpu_evidence[]` (`arch`, `certificate`, `evidence` SPDM payload) is verified via NVIDIA Attestation SDK (`nvtrust` / `nvattest`).
- **Trust roots and collateral:**
  - NRAS returns a JSON array `[["JWT", "<token>"], ...]` signed by NVIDIA's JWKS at `https://nras.attestation.nvidia.com/.well-known/jwks.json` (issuer `https://nras.attestation.nvidia.com`).
  - `nearai/inference-sdk` validates the NRAS JWT signature against NVIDIA's JWKS and registered claims (`exp`, `nbf`, `iat`, `eat_nonce`), then requires `x-nvidia-overall-att-result == true`.
  - Under NVIDIA's architecture, NRAS / Attestation SDK validates GPU device certificate chains against the NVIDIA Root CA, checks revocation via NVIDIA OCSP, and compares SPDM 1.1 runtime measurements against NVIDIA Reference Integrity Manifests (RIM) for VBIOS and driver versions.
- **Multi-GPU correlation:**
  - **NEAR:** Each candidate in `model_attestations[]` carries its own `nvidia_payload` bound to `request_nonce`; every candidate must pass before being eligible.
  - **Chutes:** `gpu_evidence` is collected from the same evidence request as the CPU TDX quote and binds the GPU challenge `SHA-256(UTF-8(nonce_hex + e2e_pubkey_b64))` (without decoding either string), which also appears in CPU `report_data[0:32]`.
- **Negative cases:**
  - `nvidia_payload.nonce != request_nonce` → reject (`gpu.payload_invalid` / nonce mismatch).
  - `x-nvidia-overall-att-result != true` or invalid NRAS JWT signature/claims → reject (`gpu.jwt_verification_failed`).
  - Missing `nvidia_payload` when GPU policy requires it → treat as unverified.

---

## 2.3 ITA / JWT Conventions & `cloud-api` Configuration

- **Authoritative source:**
  - Intel Trust Authority documentation (`https://docs.trustauthority.intel.com/`)
  - NEAR AI API Reference: `https://docs.near.ai/api-reference/attestation/get-intel-trust-authority-attestation-token.md`
  - `https://github.com/nearai/cloud-api/blob/main/crates/config/src/ita/mod.rs`
  - `https://github.com/nearai/cloud-api/blob/main/crates/config/src/ita/policy.rs`
  - `https://github.com/nearai/cloud-api/blob/main/crates/api/src/routes/attestation/ita_token.rs`
- **Verifier implementation:**
  - Standard JWT/JWS verification (e.g., `PyJWT` / `jose` / `golang-jwt`) against ITA's JWKS endpoint.
- **Trust roots, defaults, and collateral (`nearai/cloud-api`):**
  - Default `ITA_API_BASE_URL`: `https://api.trustauthority.intel.com`; default `ITA_PORTAL_BASE_URL`: `https://portal.trustauthority.intel.com` (JWKS at `/certs`, OIDC discovery at `/.well-known/openid-configuration`; EU region uses `https://portal.eu.trustauthority.intel.com/certs`).
  - Issuer (`iss`): `https://portal.trustauthority.intel.com` (or `.eu.`).
  - Signing algorithm (`token_signing_alg`): `PS384` default (`ItaTokenSigningAlg::Ps384`), `RS256` also accepted (`ItaTokenSigningAlg::Rs256`).
  - `policy_ids`: Up to `MAX_ITA_POLICY_IDS = 10` Intel Trust Authority policy UUIDs (`ITA_POLICY_IDS`, overridable per request via query parameter).
  - `policy_must_match`: Boolean (`ITA_POLICY_MUST_MATCH` default `false`, overridable per request). When `true`, ITA refuses token issuance if any policy in `policy_ids` fails.
  - No custom NEAR-specific `aud` override is configured in `ItaAttestationConfig`.
  - JWKS rotation: Cache JWKS respecting HTTP cache headers/TTL; if a JWT arrives with an unknown `kid`, trigger a JWKS refresh before failing.
  - Standard EAT/JWT claims: `exp`, `nbf`, `iat`, `jti`, `eat_profile`, `dbgstat`, and TDX measurements (`tdx_mrtd`, `tdx_rtmr*`, `attester_tcb_status`).
- **Open NEAR-specific operational gaps:** Production `ITA_POLICY_IDS` UUIDs, recommended clock-skew tolerance, and `jwks_url` allowlist/pinning rules remain in Section 3.1 of `verification-contract-questions.md`.

---

## 2.4 Cryptographic & Serialization Standards

- **Authoritative source:**
  - IETF RFC 9458 (OHTTP), RFC 9292 (Binary HTTP), RFC 9180 (HPKE)
  - `https://github.com/nearai/cloud-api/blob/main/crates/api/src/ohttp_gateway.rs`
  - `https://github.com/nearai/nearai-cloud-verifier/blob/main/py/ohttp_client.py`
  - `https://github.com/nearai/nearai-cloud-verifier/blob/main/py/model_verifier.py` (`check_compose_manager_actions`)
  - `https://docs.near.ai/cloud/verification/reference/response-signature-verification.md`
- **Standards & bindings:**
  - **OHTTP:** Uses RFC 9458 (`application/ohttp-keys`), RFC 9292 known-length BHTTP (`0x00` framing), and RFC 9180 HPKE Base Mode (`DHKEM(X25519, HKDF-SHA256)` + `HKDF-SHA256` + `AES-128-GCM` or `ChaCha20-Poly1305`).
    - Per `ohttp_gateway.rs`, `OhttpGateway::new` deterministically derives the X25519 HPKE keypair via `KeyConfig::derive` (`OHTTP_KEY_ID = 1`) from a 32-byte Ed25519 secret loaded from dstack KMS (shared across `cloud-api` instances).
    - `GET /.well-known/ohttp-gateway` serves the raw RFC 9458 `key_config` bytes.
    - `GET /v1/attestation/report` returns `ohttp_key_config` (hex) and `ohttp_attestation` (`signing_algo`, `signing_key`, `key_config`, `signature`). `signature` is an Ed25519 signature over the decoded `key_config` bytes, verified against `signing_key` (the gateway's Ed25519 identity bound in `gateway_attestation`).
  - **Compose Manager JSON canonicalization:** `check_compose_manager_actions` canonicalizes `actions` with `json.dumps(actions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` (to match Rust `serde_json` UTF-8 output) before `SHA-256`.
  - **Response Signature Hashing:** **No JSON canonicalization** (see Section 3.3 below).

---

## 3.2 Signing Keys and Provider Mapping (Partially Resolved)

- **Authoritative source:**
  - `https://docs.near.ai/cloud/verification/reference/quote-nonce-signer.md`
  - `https://docs.near.ai/cloud/verification/cloud-api/response-signatures.md`
  - `https://docs.near.ai/cloud/verification/cloud-api/model-attestations.md`
- **Resolved behavior:**
  - There is no static CA-style key registry for NEAR-operated keys: ephemeral signing keys (`ecdsa` 20-byte ETH address or `ed25519` 32-byte pubkey) are generated inside the TEE and bound directly into the Intel TDX quote `report_data[0:32]`.
  - `gateway` signatures match `gateway_attestation.signing_address` + `signing_algo`.
  - `provider_tee` signatures match a uniquely matching verified candidate in `model_attestations[]` (`signing_address` + `signing_algo`).
  - **Important documented caveat:** Instances of the same model share one signing key, even across different measured configurations (`app_compose`), so a matching `provider_tee` signer does not prove the completion ran on the exact same instance as a single preflight report.
- **Remaining gaps:** Key rotation/revocation lifecycle for shared model keys, `ohttp_attestation` signature verification spec in official docs, and `/v1/signature` key mapping for non-NEAR providers (`chutes`, `tinfoil`).

---

## 3.3 Canonical Request / Response Hashing (Resolved)

- **Authoritative source:**
  - `https://docs.near.ai/cloud/verification/cloud-api/response-signatures.md`
  - `https://docs.near.ai/cloud/verification/reference/response-signature-verification.md`
- **Exact bytes hashed:**
  - `request_hash = SHA-256(exact request-body bytes)` (lowercase hex).
  - `response_hash = SHA-256(exact uncompressed response-body bytes)` (lowercase hex; request `Accept-Encoding: identity` or hash after HTTP decompression).
  - **No JSON canonicalization** is performed. Clients must not parse/re-serialize JSON, normalize whitespace or line endings, or reconstruct SSE streams from parsed events.
- **Streaming SSE, Responses API, and aliases:**
  - For streamed completions, `response_hash` covers the exact uncompressed SSE wire bytes including framing and terminal events.
  - The Gateway signs as `gateway` (`<request_hash>:<response_hash>`) whenever it modifies relayed bytes: by default on streamed completions (removes per-chunk usage data), when resolving a model alias, and on the Responses API (`resp_*`).
  - `provider_tee` (`<MODEL_ID>:<request_hash>:<response_hash>`) requires sending the canonical model ID with `x-no-aliasing: true`.
- **Signature verification:**
  - `ecdsa`: Ethereum EIP-191 personal-sign (`encode_defunct(text=text)` / `ethers.verifyMessage`), 65-byte signature, recover 20-byte address and compare with `signing_address`.
  - `ed25519`: Detached Ed25519 signature (64 bytes) over `text.encode("utf-8")`, verified against 32-byte `signing_address`.
  - Note: Default `signing_algo` differs by endpoint (`/v1/signature` defaults to `ecdsa`; `/v1/attestation/report` defaults to `ed25519` for gateway and `ecdsa` for model reports). Always pass `signing_algo` explicitly.
- **Negative cases:**
  - `signature_kind` absent or unfamiliar → do not infer from `text` shape; treat classification as unavailable.
  - `2xx` body containing `error_code` + `message` → unavailable result, not a signature.
  - `404` → signature not yet recorded / unavailable; bounded retry only, then fail closed if required.

---

## 3.4 End-to-End Binding and Failure Behavior (Substantially Resolved)

- **Authoritative source:**
  - `https://docs.near.ai/cloud/verification/reference/verification-policy.md`
  - `https://docs.near.ai/cloud/verification/cloud-api/response-signatures.md`
- **Fail-closed rules:**
  - Missing, unavailable, malformed, or mismatched quote/nonce/SPKI/signature/provenance (`404`) → treat as unverified.
  - Never substitute a report obtained *after* completion for preflight evidence.
  - For `provider_tee`, if zero or more than one verified `model_attestations[]` candidate matches `signing_address` + `signing_algo`, treat as unverified.
- **Documented architectural limit (remaining gap):**
  - `docs.near.ai` explicitly states: *"Preflight Gateway and model evidence plus one response signature do not yet form a complete model-to-Gateway-to-final-response chain."* A `gateway` signature proves the Gateway signed client-visible bytes (not model TEE execution), and a `provider_tee` signature does not bind its model-signed bytes to Gateway TLS.
