# Security, Privacy, and Incident Response

## Scope and trust boundary

This repository configures Antigravity (`agy`) to use the NEAR AI Gateway. It does **not** independently prove that a remote model is executing inside a genuine TEE. Gateway responses may contain quotes, certificates, tokens, signatures, or metadata; those are evidence signals unless validated against authoritative verifier libraries, trust roots, collateral, policy, and canonical request/response semantics.

The implementation therefore uses conservative provider classification and evidence correlation. `overall: unverified` remains the safe default. The `private` profile sets `"attestation_required": true` and `nearai_agy.py run` enforces a fail-closed attestation preflight at launch — refusing to start unless the user explicitly passes `--allow-unverified` (which still preserves `tee_only` provider filtering).

## Operational rules

- Do not put API keys, JWTs, certificates, quotes, private keys, or raw live responses in Git, issue trackers, shell history, or logs.
- Use `NEAR_AI_KEY` or the tool's mode-0600 local configuration. Prefer environment injection from an approved secret manager for automation.
- Treat `vllm`, `chutes`, and `tinfoil` as recognized provider classes, not cryptographic proof. Unknown, legacy, and external providers are excluded from TEE-only selection.
- Use ordinary profiles for ordinary routing. Use `profiles/private.json` only when the fail-closed attestation workflow is understood and acceptable.
- Review generated launchers and settings changes before running an agent over sensitive repositories.

## Incident response

If a credential or sensitive prompt may have reached an unintended provider:

1. Stop active agent sessions and disable private-profile launches.
2. Revoke and rotate `NEAR_AI_KEY` immediately; do not paste the old value into diagnostics.
3. Review shell history, process listings, CI logs, local caches, generated launchers, and Antigravity settings for exposure. Remove only after preserving the minimum non-sensitive incident timeline required by policy.
4. Record the approximate request time, selected model, provider class, request/model correlation identifiers, and sanitized error/evidence summaries. Do not retain raw prompt, response, JWT, quote, certificate, or key material.
5. Check gateway/account audit information through authorized NEAR AI support channels.
6. Keep private routing disabled until the exposure scope is understood and the verification contract is independently reviewable.

Re-enable private routing only after: the credential is rotated; affected caches/logs are reviewed; the selected provider/model is confirmed; fresh evidence is collected; and an authorized reviewer accepts the residual risk. `attest-route-check` should be treated as a gate, not as proof that the remote TEE is authentic.

## Evidence retention

Retain sanitized summaries only: endpoint, timestamp, model/provider identifiers, correlation identifiers, boolean check results, status codes, and short redacted diagnostics. Apply the project's retention policy and securely delete temporary evidence. Never commit raw live gateway output.

## Verification contract blockers

Independent verification still requires authoritative answers for:

- TDX and NVIDIA quote verifier implementations, trust roots, collateral, revocation, and policy.
- ITA issuer, audience, claims, clock-skew/expiry, JWKS rotation, and policy semantics.
- Gateway/provider signing-key registry, roots, rotation, revocation, and provider mapping.
- Canonical request/response bytes, hash algorithms, streaming/tool-call handling, and Unicode/JSON normalization.

See [`docs/gateway-capability-matrix.md`](docs/gateway-capability-matrix.md) and [`docs/verification-contract-questions.md`](docs/verification-contract-questions.md).

## Reporting a vulnerability

If you discover a security vulnerability in this repository (for example, credential leakage, loopback proxy bypass, shell injection, or cryptographic verification bypass), please report it privately rather than opening a public issue:

- Use **GitHub Private Vulnerability Reporting** under this repository's **Security → Advisories → Report a vulnerability** tab.
- Include reproduction steps, affected files/versions, and sanitized diagnostics only. Do **not** include live API keys, JWTs, private keys, or confidential prompts.

## Useful commands

```bash
./bin/nearai-private-check
./nearai_agy.py attest-evidence --verify-crypto
./nearai_agy.py attest-route-check Qwen/Qwen3.8-27B
./nearai_agy.py list --tee-only
pytest -q
```

The first command collects evidence only. The second intentionally fails closed when authoritative verification is unavailable.
