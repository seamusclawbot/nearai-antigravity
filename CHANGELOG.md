# Changelog

All notable changes to `nearai-antigravity` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-25

Initial public release.

### Added
- **Unified Antigravity Engine (`nearai_agy.py`):** Zero-dependency Python 3.9+ stdlib CLI supporting **Antigravity CLI (`agy`)**, **Antigravity IDE**, and **Antigravity Desktop App** (`--target cli|ide|app`, `--all-targets`, and `--isolated` side-by-side Electron windows).
- **Curated Profiles & 32-Slot Accounting:** Five starter profiles (`private`, `frontier`, `anthropic`, `qwen`, `lite`) with automatic non-chat filtering, live request/tool-calling probes, and exact 32-slot tier planning (`plan_slots`).
- **Automatic Context Budget Sync:** Atomic `settings.json` and `jetski_state.pbtxt` `customModels` synchronization with `.bak` backups to prevent `ErrNoMessagesFit`.
- **Hardened Loopback `GatewayProxy`:** Ephemeral `127.0.0.1` proxy with constant-time session-token authentication, bounded concurrency (`BoundedSemaphore`), bounded request/upstream/SSE reads, redirect blocking, path-traversal rejection, and strict recursive `reasoning_effort` stripping for `anthropic/*`.
- **Method 2 "Agy Pro" Hybrid Plugin (`pro setup`):** Installs a native Antigravity plugin with 5 specialist NEAR AI subagents (`nearai-private`, `nearai-uncensored`, `nearai-claude`, `nearai-deepseek`, `nearai-general`), companion skills, usage ledger (`pro ledger`), empirical benchmark suite (`pro benchmark`), and task recommender (`pro recommend`).
- **Fail-Closed Attestation & Optional Cryptographic Verifiers:** Evidence collector (`attestation_evidence.py`) plus opt-in Intel TDX (`dcap-qvl`), NVIDIA NRAS GPU JWT, OHTTP Ed25519, and exact-byte `/v1/signature/{chat_id}` verifiers (`verifiers/`). Enforces fail-closed preflight on `attestation_required` profiles (`--allow-unverified` explicit override).
- **Packaging & CI:** PEP 621 `pyproject.toml` (`nearai-agy` console entry point and `[verifiers]` extra), GitHub Actions matrix across Python `3.10–3.13`, and automated secret/path hygiene tests.
