# Video Storyboard & Voiceover Script: Private Agentic Coding with NEAR AI & Antigravity

**Target Length:** ~4 minutes 30 seconds  
**Resolution:** 1080p60 or 4K, 16:9  
**Theme:** Crisp dark mode terminal (Catppuccin Mocha or Tokyo Night), clean monospace font (JetBrains Mono / Berkeley Mono), 16pt font size.

---

## Segment 1: The Hook (0:00 – 0:25)

**On Screen:**
- *0:00 – 0:10:* Close-up screencast of Antigravity CLI (`agy`) executing an autonomous 10-step agent loop across a local git repo. Terminal text scrolls rapidly as the agent reads source files, scans credentials, and writes tests.
- *0:10 – 0:25:* Sudden cut to a clean, illuminated architectural slide: An encrypted hardware enclave boundary showing an Intel TDX VM and NVIDIA H200 GPU chip with incoming prompts and weights locked in hardware memory.

**Voiceover:**
> *"Your coding agent reads your entire repository, environment configs, and uncommitted code — and it keeps running while you aren't watching.*  
> *When you route that agent to a closed cloud API, you have to blindly trust that your code isn't being logged, inspected by engineers, or fed into training sets.*  
> *What if you could route your agent somewhere the cloud operator physically cannot read the prompts — and where you can inspect preflight evidence before sending a single token — while keeping unresolved verification contracts visible?"*

---

## Segment 2: Why Hardware Privacy Matters (0:25 – 1:10)

**On Screen:**
- Split screen: On the left, typical cloud inference showing memory dumps and operator logging access.
- On the right, NEAR AI's confidential computing architecture: Intel TDX confidential VM and 8× NVIDIA H200 TEE GPUs.
- Highlight graphic: **"Encrypted In Use (CPU & GPU Memory)"**, **"Zero Prompt Logging"**, **"Zero Platform Fees"**.

**Voiceover:**
> *"Most 'private AI' vendors ask you to trust a privacy policy clause. NEAR AI replaces that policy promise with hardware enforcement.*  
> *In NEAR AI's Private Inference nodes, models run inside hardware Trusted Execution Environments (TEEs) powered by Intel TDX and NVIDIA H200 GPUs.*  
> *This means your code is encrypted in use — not just on disk or over HTTPS, but in active GPU memory while the model executes.*  
> *Neither NEAR AI, the datacenter provider, nor any unauthorized process on the physical host can extract weights, prompts, or generated tokens.*  
> *And attestation isn't an expensive enterprise add-on: it's included by default in the pay-as-you-go per-token rate."*

---

## Segment 3: The Proof: Keyless Attestation (1:10 – 1:50)

**On Screen:**
- Clean terminal. Type:
  ```bash
  ./nearai_agy.py verify Qwen/Qwen3.8-27B
  ```
- Output prints:
  ```
  NEAR AI Gateway Attestation
    endpoint        : https://cloud-api.near.ai/v1
    status          : present_unverified (quote present)
    signing address : 40599fb2...
    ohttp key match : yes

  Model Attestation (Qwen/Qwen3.8-27B)
    model           : Qwen/Qwen3.8-27B
    hardware quotes : Intel TDX, NVIDIA GPU TEE
  ```
- Fast cut to browser tab opening `cloud.near.ai/verify` showing the Intel quote verified with a green badge.

**Voiceover:**
> *"Don't take our word for it — and don't confuse evidence with verification. The attestation report endpoint requires zero authentication.*
> *You can query the gateway before spending an API key or transmitting your first prompt.*
> *This tool records nonce, TLS, OHTTP, and model-correlation signals, while clearly reporting `unverified` until authoritative quote, policy, key, and model verifiers are available."*

---

## Segment 4: Setup & Health Check (1:50 – 2:30)

**On Screen:**
- Terminal:
  ```bash
  git clone https://github.com/seamusclawbot/nearai-antigravity.git && cd nearai-antigravity
  ./nearai_agy.py init
  ./nearai_agy.py doctor
  ```
- Terminal outputs health summary showing Python 3.13, agy CLI detected, API key present, settings synchronized, and 0 external dependencies.

**Voiceover:**
> *"Getting started takes under two minutes. Clone the repo and run `init`. It prompts for your NEAR AI API key and stores it securely with 0600 file permissions.*  
> *The entire tool is written in zero-dependency Python standard library — no pip packages, no external daemons.*  
> *Running `doctor` verifies your local Antigravity binary, checks settings budgets, and inspects pre-built launchers."*

---

## Segment 5: Launching & The Reality Check (2:30 – 3:30)

**On Screen:**
- Terminal:
  ```bash
  ./bin/nearai-private-check
  ./bin/nearai-private --allow-unverified
  ```
- Camera pauses and zooms in on terminal output showing:
  `10 of 32 slots` and `models are re-validated against the catalog at launch`.
- Next, run:
  ```bash
  ./nearai_agy.py probe --tools-only
  ```
- Display the grouped failure analysis (`incompatible`, `reasoning_effort`, `route`).

**Voiceover:**
> *"Now for the practical reality that makes a tool necessary.*  
> *Antigravity hard-caps custom gateway models at 32 entries, silently discarding anything past the 32nd model. Yet NEAR AI serves over 60 models.*  
> *Furthermore, catalog membership does not guarantee a model is usable by an agent. Many models reject function-calling or require parameter shims.*  
> *`nearai_agy.py` probes every candidate with realistic OpenAI function-calling schemas, shims parameters like `reasoning_effort` for Anthropic models via a transparent local loopback, and packages working models into clean, self-healing launchers."*

---

## Segment 6: Hybrid Orchestration with Gemini (Method 2) (3:30 – 4:10)

**On Screen:**
- Terminal:
  ```bash
  ./nearai_agy.py pro setup
  ./nearai_agy.py pro list
  ```
- Display installed subagents: `nearai-private`, `nearai-uncensored`, `nearai-claude`, `nearai-deepseek`.
- Terminal demonstration of dynamic recommender:
  ```bash
  ./nearai_agy.py pro recommend "audit cryptographic timing leaks in token comparator"
  ```
- Show empirical latency and cost output.
- Next, launch Antigravity:
  ```bash
  agy
  ```
- Show Google Gemini operating as primary orchestrator, then delegating a sensitive cryptographic check to subagent `nearai-private`.

**Voiceover:**
> *"If you don't want to replace Gemini, you don't have to. With Method 2 — Agy Pro Hybrid Mode — Google Gemini remains your primary orchestrator in Antigravity, retaining its 1-million-token context and native workspace tools.*  
> *Gemini then delegates specialized tasks to NEAR AI subagents:*  
> *- Confidential security audits go to `nearai-private` inside the hardware TEE.*  
> *- Vulnerability fuzzing goes to `nearai-uncensored` without refusal interruptions.*  
> *- Deep AST refactoring delegates to `nearai-claude`.*  
> *- And math proofs run on `nearai-deepseek` at fractions of a cent.*  
> *Our dynamic recommender benchmarks models empirically, tracking exact token costs and latency for every task domain."*

---

## Segment 7: The Close (4:10 – 4:35)

**On Screen:**
- Full screen terminal displaying the benchmark matrix and the two uncensored TEE models:
  - `Qwen/Qwen3.8-27B-abliterated`
  - `deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated`
- End card showing:
  - GitHub Repository: `https://github.com/seamusclawbot/nearai-antigravity`
  - Documentation: `ExecutiveBrief.md` & `UserGuide.md`
  - NEAR AI Docs: `https://docs.near.ai/cloud/private-inference`

**Voiceover:**
> *"If you remember only one thing from this walkthrough, look at `Qwen 3.8 27B abliterated`: it is open-weight, refusal-free, agentic tool-capable, and routed through an attestation-aware, fail-closed workflow. This repository does not claim independent hardware-enclave verification until the verifier contract is authoritative and implemented.*
> *Run your agent with the privacy of on-prem, and the power of the cloud.*  
> *Clone the repo, verify the enclave, and start coding."*

---

## Production Checklist

- [ ] Record terminal at 60fps with cursor smoothing enabled.
- [ ] Never fake or mock the probe output — showing real dropped models builds credibility.
- [ ] When making the privacy claim, always show a TEE model (`Qwen/Qwen3.8-27B`) on screen, not an external frontier model.
- [ ] Voice track: measured, technical, authoritative tone (no hype adjectives).
