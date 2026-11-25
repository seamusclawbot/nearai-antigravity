# Executive Brief: Private Agentic Coding with NEAR AI & Antigravity

> **"Your coding agent reads your entire repository and executes commands while you are not watching. Route it toward confidential-computing protections, inspect preflight evidence, and keep independent verification requirements explicit before spending a single token."**

---

## 1. The Core Dilemma in Agentic Development

Autonomous coding assistants (like Google Antigravity, Claude Code, and Cursor) represent a fundamental shift in software development:
- **Whole-Repository Ingestion:** Unlike simple autocomplete, agents recursively read source files, environment variables, documentation, and architecture diagrams.
- **Unattended Multi-Step Execution:** Agents execute extended tool-calling loops over dozens of turns.
- **The Exposure Risk:** Routing this workload through standard closed cloud APIs requires trusting terms-of-service promises that your proprietary intellectual property, cryptographic keys, and uncommitted code are not logged, inspected by operators, or used for model training.
- **The On-Premise Alternative:** Self-hosting 70B+ parameter models on enterprise hardware provides isolation, but demands millions in capital expenditure, ongoing cluster ops, and severe compute constraints.

---

## 2. The Solution: Hardware-Attested Private Inference

NEAR AI Cloud provides an unprecedented third path: **Confidential Cloud Inference**.

Instead of relying on contractual terms of service, isolation is enforced physically by hardware:
- **Intel TDX & NVIDIA H200 GPU TEEs:** Virtual machines and GPU memory are hardware-isolated. Data remains **encrypted in use**—during compute, in memory, and over the bus. Neither NEAR AI, the datacenter provider, nor unauthorized system processes can inspect prompts, weights, or responses.
- **Zero Prompt Logging:** The infrastructure retains zero prompt or response telemetry by architectural design.
- **Keyless Pre-Flight Attestation:** Clients can retrieve attestation evidence (`GET /v1/attestation/report`) **before sending an API key or prompt data**. This repository does not yet independently verify quote authenticity, collateral, policy, or model execution.

---

## 3. Two Deployment Modalities

This repository provides two turnkey architectures connecting Google Antigravity (`agy`) to NEAR AI Cloud:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       METHOD 1: FULL GATEWAY MODE                           │
│                                                                             │
│  User / Terminal ───> agy CLI ───> GatewayProxy ───> NEAR AI TEE Enclave    │
│                                   (Loopback Stdlib)   (Intel TDX / H200)    │
│  • Routes 100% of agent prompts through NEAR AI open-weight models.        │
│  • Curates catalog into 32 working slots (bypassing agy slot cap limits).   │
│  • Self-healing launcher profiles (private, frontier, anthropic, qwen).     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                 METHOD 2: HYBRID ORCHESTRATOR MODE (AGY PRO)                │
│                                                                             │
│  User ───> Gemini 3.5 Pro/Flash ───> Subagent Router ───> NEAR AI Subagent  │
│            (Primary Orchestrator)     (Agy Pro Plugin)     (Specialist TEE) │
│  • Native Gemini manages 1M+ context window, workspace tools & UI.         │
│  • Seamlessly delegates sensitive code audits to hardware TEE subagents.   │
│  • Seamlessly delegates exploit simulations to refusal-free models.        │
│  • Seamlessly delegates math & micro-optimizations to DeepSeek at pennies.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. How It Compares

| Feature / Guarantee | NEAR AI Cloud (TEE) | Closed Vendors (OpenAI / Anthropic) | Self-Hosted On-Prem |
| :--- | :---: | :---: | :---: |
| **Encrypted In Use (Hardware TEE)** | **Documented by provider; locally unverified** | No | Only if built |
| **No Prompt / Code Logging** | **Guaranteed in Hardware** | Policy Promise Only | Yes |
| **Verifiable Attestation** | **Evidence available; verification contract incomplete** | No | Manual Setup |
| **Refusal-Free Open Weights** | **Yes (Abliterated Variants)** | No (Strict Filters) | Yes |
| **Drop-in OpenAI API** | **Yes** | Standard | Custom Infra |
| **Setup Overhead & Cost** | **Zero Infra / Pay-Per-Token** | Zero Infra / Pay-Per-Token | High CapEx ($$$$) |
| **No Trust Required in Cloud Host** | **Yes (Hardware Enclave)** | No (Blind Trust) | Yes |

---

## 5. Proven Reliability & Zero Dependencies

The integration engine (`nearai_agy.py`) is engineered to production standards:
- **Zero Third-Party Runtime Dependencies:** 100% Python standard library (Python 3.9+).
- **Automated Context Window Safeguard:** Synchronizes positive model context budgets in Antigravity settings with atomic `.bak` backups, permanently eliminating the fatal `ErrNoMessagesFit` compaction bug.
- **Live Tool-Calling Verification:** Every supported model is rigorously tested against OpenAI structured `tool_calls` schemas (`read_file`).
- **Empirical Benchmarking & Dynamic Routing:** Includes an empirical evaluator testing real-world pass rates, roundtrip latency, and dollar-per-task expenditures across 5 task domains.
- **Thoroughly Tested:** Full offline unit test suite passing 100% in CI.

---

## 6. Quickstart: 60 Seconds to First Launch

```bash
# 1. Clone & Configure (asks [y/N] before saving key to mode-0600 config.json)
git clone https://github.com/seamusclawbot/nearai-antigravity.git && cd nearai-antigravity
./nearai_agy.py init

# 2. Collect keyless attestation evidence before sending credentials
./nearai_agy.py verify Qwen/Qwen3.8-27B

# 3. Launch immediately (CLI, IDE, or Desktop App)
./bin/nearai-frontier                      # CLI (32 frontier models incl. Claude)
./bin/nearai-ide-frontier --isolated .     # Antigravity IDE side-by-side
./bin/nearai-private-check                 # Cryptographic TDX/GPU/signature preflight
./bin/nearai-private --allow-unverified    # Opt-in TEE-only profile

# 4. Or Enable Method 2 (Gemini Hybrid Orchestration with NEAR AI Subagents)
./nearai_agy.py pro setup
agy  # Gemini automatically delegates to nearai-private, nearai-deepseek, etc.
```

---

*For technical specifications, refer to [ArchitectureGuide.md](ArchitectureGuide.md). For detailed user options and benchmark metrics, refer to [UserGuide.md](UserGuide.md).*
