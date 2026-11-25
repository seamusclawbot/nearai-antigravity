# Contributing to `nearai-antigravity`

Thank you for your interest in contributing! Before opening a pull request, please review [`ArchitectureGuide.md`](ArchitectureGuide.md) (especially the engineering invariants) and [`SECURITY.md`](SECURITY.md).

## Core Principles

1. **Zero third-party runtime dependencies for the core CLI:** `nearai_agy.py` and `attestation_evidence.py` must run on Python 3.9+ standard library only. Cryptographic verifier dependencies belong strictly under `[project.optional-dependencies].verifiers` and `requirements-verifier.txt`.
2. **Fail-closed security posture:** Never weaken attestation checks, URL/redirect validation, header/payload bounds, or secret redaction. Never claim independent TEE verification without authoritative verifier semantics.
3. **Offline-first unit tests:** All unit tests in `tests/` must run completely offline without making live network calls or requiring a `NEAR_AI_KEY`.

## Local Development Setup

```bash
git clone https://github.com/seamusclawbot/nearai-antigravity.git
cd nearai-antigravity

# Core stdlib-only tests (no optional verifier deps required)
python3 -m venv .venv
.venv/bin/pip install pytest
.venv/bin/pytest -v --ignore=tests/test_cryptographic_verifiers.py

# Full test suite including cryptographic TDX / NRAS / OHTTP / signature verifiers
.venv/bin/pip install -e ".[verifiers,dev]"
.venv/bin/pytest -v
```

## Pre-PR Verification Checklist

Run the same checks enforced in CI before submitting:

```bash
python3 -m py_compile nearai_agy.py attestation_evidence.py verifiers/*.py
for f in bin/nearai-* examples/*.sh; do bash -n "$f"; done
.venv/bin/pytest -v
./nearai_agy.py doctor
./nearai_agy.py pro setup --dry-run
```

## Secret & Privacy Hygiene

- **Never** commit API keys, JWTs, private keys, raw live attestation responses, or personal `/Users/<name>` / `/home/<name>` paths.
- `tests/test_secret_hygiene.py` runs automatically in CI to catch accidental credential or personal-path leaks.
- For security vulnerabilities, **do not open a public issue** — follow the private reporting instructions in [`SECURITY.md`](SECURITY.md).
