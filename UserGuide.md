# User Guide: NEAR AI Models in Antigravity (CLI, IDE & Desktop App)

A practical guide to running **Antigravity CLI (`agy`)**, **Antigravity IDE**,
and **Antigravity Desktop App** against **NEAR AI Cloud** instead of, or
alongside, the models they ship with.

- **Audience:** anyone who wants to use Antigravity with private TEE or open-weight models.
- **Time to first launch:** about two minutes.
- **Companion docs:** [`ArchitectureGuide.md`](ArchitectureGuide.md) explains how
  the tool works internally, testing strategy, and architecture.

> [!TIP]
> **60-Second Out-of-the-Box Start (Zero Learning Curve):**
> 1. **Set your key:** `./nearai_agy.py init` *(prompts `[y/N]` before saving to mode-0600 `config.json`, or just `export NEAR_AI_KEY="sk-..."`)*
> 2. **Launch immediately (pick one):**
>    - **CLI (Frontier / Qwen):** `./bin/nearai-frontier` or `./bin/nearai-qwen`
>    - **IDE / Desktop App (side-by-side):** `./bin/nearai-ide-frontier --isolated .` or `./bin/nearai-app-qwen --isolated` *(keep the terminal open until you quit so the local proxy tears down cleanly)*
>    - **TEE-only (`private` profile):** `./bin/nearai-private-check` then `./bin/nearai-private --allow-unverified`
>    - **Hybrid Mode (keep Gemini + add NEAR AI specialist subagents):** `./nearai_agy.py pro setup` then open a fresh chat in CLI, IDE, or Desktop App.

> [!NOTE]
> This guide doubles as a storyboard. Section 9 maps each part to a short video
> segment — see [§10](#10-turning-this-into-a-short-video).

---

## 1. Why NEAR AI with `agy`?

Short version: **NEAR AI is an OpenAI-compatible gateway where a subset of
models runs inside hardware enclaves that the operator cannot read — so you get
a coding agent's model choice without handing a vendor your source code.**

The rest of this section is the long version, with the evidence, because one
part of that sentence needs a caveat and the caveat matters.

### 1.0 The 30-second version

> Most "private" AI APIs ask you to trust a terms-of-service clause. NEAR AI
> runs open-weight models inside Intel TDX and NVIDIA confidential-computing
> enclaves, returns a hardware-signed attestation, and publishes the verifier so
> you can check the proof yourself — before you even send your API key. For a
> coding agent that reads your repository and runs unattended, "the infrastructure
> operator cannot read my prompts" is a different risk posture, not a marketing
> adjective. And because the API is OpenAI-shaped, it drops into an existing
> toolchain unchanged.

That block is deliberately self-contained: it is the seed for a standalone
one-pager. See also the comparison table in [§1.7](#17-how-it-compares).

### 1.1 Private AI inference — the actual differentiator

Everything else in this section is a convenience. This is the reason to care.

NEAR AI's [private inference
doc](https://docs.near.ai/cloud/private-inference) describes three guarantees,
and each maps onto a concrete mechanism:

| Guarantee | Mechanism |
| --- | --- |
| **Complete privacy** | Prompts, weights and outputs stay encrypted and isolated in hardware. The doc's claim is that infrastructure providers, model providers and NEAR itself cannot access your data "at any point in the process". |
| **Cryptographic verification** | Attestation reports provide evidence for a specific runtime and request path. |
| **Production performance** | Hardware-accelerated TEEs deliver "high-throughput inference with minimal latency overhead" — the enclave is not a toy. |

**The hardware.** Intel TDX creates confidential VMs that isolate the workload
from the host; NVIDIA TEE provides GPU-level isolation so weights and
computations remain private *during* processing. This is the distinction that
matters: **encrypted in use, not just at rest and in transit.** The Private LLM
Nodes run **8× NVIDIA H200 GPUs** each.

**The threat model is stated explicitly**, which is rarer than it should be. The
doc claims protection against malicious infrastructure providers (including
physical access), network-based attacks and MITM, model-extraction attempts, and
result tampering. It also notes that **no logs of prompts or responses** are
kept.

**Attestation is consumable, and it costs nothing to check.** The report
endpoint needs no API key — documented, and verified live with `./nearai_agy.py verify`:

```
$ ./nearai_agy.py verify Qwen/Qwen3.8-27B
NEAR AI Gateway Attestation
  endpoint        : https://cloud-api.near.ai/v1
  status          : present_unverified (Intel TDX quote present; not independently verified)
  signing algo    : ed25519
  signing address : 40599fb284f43953bb356b09080fd803c960fc10285ad90eca462ca711ad4acb
  ohttp key match : yes

Model Attestation (Qwen/Qwen3.8-27B)
  model           : Qwen/Qwen3.8-27B
  hardware quotes : Intel TDX, NVIDIA GPU TEE
  signing algo    : ecdsa
```

The API reference says the key "is optional, so a client can verify the gateway
before it sends any credential", and that "report retrieval is non-billable".
That ordering is the whole point: you verify the enclave, *then* spend your key.

There is also a browser option — the [Verification
Center](https://docs.near.ai/cloud/verification/verification-center) at
`cloud.near.ai/verify` runs the same checks with no code and no API key, and the
verifier itself is open source so you can reproduce the build.

Why this lands harder for *this* audience: `agy` is an agent. It reads your
repository, not a single prompt — and it keeps going while you are not watching.
"Encrypted in use, operator-blind, no prompt logs" is a materially different
position from a closed vendor's promise, and it is the single strongest argument
for routing an agent through this gateway.

> [!WARNING]
> **Read §1.8 before you rely on this.** Not every model in the catalog is
> private. The `frontier` profile sorts by price and therefore fills up with
> externally-hosted frontier models that are **not** in a TEE. Use the bundled
> `private` profile (`./nearai_agy.py build private`) or `--tee-only` when you
> want hardware-attested TEE models exclusively.

### 1.2 Uncensored, open-weight models — actually available

This is the motivation that usually gets buried under the privacy story, so it
gets its own section immediately after it. The two are complementary: NEAR AI is
the unusual place where you can have both at once.

NEAR AI serves **abliterated** (refusal-removed) variants of open-weight models,
and describes its stance as *"Why Our Uncensored Models Are Open to Everyone"*.
In the live catalog this tool reads, that means real, usable entries:

| Model | Tools | `agy` usable? |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated` | yes | **yes** (probe: ok) |
| `Qwen/Qwen3.8-27B-abliterated` | yes | **yes** (probe: ok) |

Both were verified with a real request in `agy`'s own shape, and both advertise
tool support — which is what an agentic coding CLI actually needs. This is not a
side catalogue entry: it is a working, tool-capable model you can select in
`agy` today.

Worth being precise about *why* this matters for an agentic coding tool. A
coding agent runs long, unattended loops over your own repository. Refusals in
that setting are not a safety feature so much as an interruption: a model that
declines a legitimate task mid-run costs you the whole loop. Running that loop
against a model whose behaviour is not gated by someone else's policy is a
genuine workflow benefit. What you accept in exchange is that the model's
judgement is yours to manage.

### 1.3 OpenAI-compatible — it drops into what you already run

NEAR AI's own FAQ answers "Do I need to change my code?" with:

> No. The API is OpenAI-compatible. Point your existing OpenAI SDK at our base
> URL, add your key, and you keep attestation as a bonus you can optionally
> verify.

That is the property this tool exploits. `agy` has a generic "bring your own
gateway" mode; because NEAR AI speaks the OpenAI wire format, `agy` can drive
NEAR AI models with no adapter, no proxy, and no patched build.

One practical caveat, learned the hard way: set
`AGY_LLM_GATEWAY_WIRE_PROTOCOL=openai`. The `anthropic` and `gemini` protocol
values do not work against this gateway. Compatibility is concretely
**OpenAI-shaped**, not "anything goes".

### 1.4 Pay per token — and attestation is not an add-on

The [pricing page](https://near.ai/pricing) is unusually specific about what is
*not* charged:

- **Per token**, input and output priced separately per 1M tokens.
- **No platform fee** on top of the model rate.
- **No minimums, no subscriptions, no seat licences** on pay-as-you-go.
- **Private execution and the hardware-signed attestation are "included in the
  rate, not billed as an add-on."**

That last line is the quiet advantage. Privacy is normally an enterprise upsell;
here it is the default transport. Live catalog examples: `Qwen/Qwen3.8-27B` at
**$0.04 / $1.35** per 1M in/out, `z-ai/glm-5.3-flash` at **$0.08 / $0.50**,
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at **$0.30 / $1.20**.

Because pricing comes from the same catalog this tool already reads, `build
--sort cheap` and `--sort expensive` use live numbers rather than a hardcoded
table that would rot.

### 1.5 One catalog, many families

The catalog is currently **64 models** across 11 providers — `openai` (20),
`anthropic` (13), `nearai` (12), `google` (6), `deepseek` (2), `x-ai` (2), plus
`qwen`, `deepseek-ai`, `moonshotai`, `z-ai`, `black-forest-labs`, `levanto` and
`typesafe`. Chat models range from about **$0.04/M** input tokens for small Qwen
models up to several dollars per million for the largest frontier models. You can
compare and switch families without changing tools, billing relationships or
credentials.

Two smaller conveniences worth knowing about:

- **Prompt caching is automatic** on TEE-hosted models — repeated prefixes (your
  system prompt, conversation history, a large file) are billed at a discounted
  cache-read rate, and `usage.prompt_tokens_details.cached_tokens` reports the
  hit. Nothing to enable. This matters for an agent, which re-sends a growing
  transcript every turn.
- **Open weights, no lock-in** — the models are portable, so the cost of leaving
  is low by construction.

### 1.6 Why you need a tool at all

Three quirks make hand-maintenance painful, and they are why this repo exists:

1. **`agy` keeps only the first 32 model ids** you give it, silently discarding
   the rest. NEAR AI lists 64.
2. **Being in the catalog ≠ working in `agy`.** Roughly a third of the catalog
   fails on request shape or endpoint, and the catalog's own `is_ready` flag is
   *wrong in both directions* (28 models flagged not-ready actually work; 3
   flagged ready fail).
3. **Being in the catalog ≠ being private.** Only 17 of the 64 models run inside
   a hardware TEE (`GET /v1/model/list`), while 47 are passthrough to external
   providers — see §1.8.

`nearai_agy.py` solves all three: it merges the privacy metadata, filters or
shims incompatible models, keeps the 32 slots filled with models that actually
work, and re-checks itself at every launch so it does not rot.

### 1.7 How it compares

NEAR AI's own comparison, restated. The right-hand column is the honest part:
self-hosting gives you the privacy and none of the convenience.

| | NEAR AI Cloud | Closed vendor (e.g. OpenAI) | Self-hosted on-prem |
| --- | --- | --- | --- |
| No logs of prompts/responses | yes | — | yes (you own them) |
| Confidential computing (TEE) | **yes, for TEE models** | — | only if you buy it |
| Hardware-signed attestation | **yes, for TEE models** | — | doable, you build it |
| Drop-in OpenAI API | **yes** | n/a | no |
| Setup cost | low | low | high |
| Complexity | low | low | high |
| Scalability | good | good | poor |
| No need to trust the operator | **yes, for TEE models** | no | yes |

The framing NEAR AI uses — *"Privacy of on-prem. Power of cloud."* — is accurate
for the TEE-hosted subset of the catalog. Which brings us to the caveat.

> [!NOTE]
> **One wording discrepancy worth knowing.** The marketing site says a
> hardware-signed proof comes back "with every response." The [private inference
> doc](https://docs.near.ai/cloud/private-inference) is more careful: attestation
> "provides cryptographic evidence for a specific runtime and request path," and
> a response signature is available in "supported flows" — adding that "a
> response without an available signature is not response-verified."
>
> Both are true; the docs version is the one to quote. Treat the attestation
> report (which is unconditional and keyless) as the reliable primitive, and
> per-response signatures as an optional extra whose availability you should
> check rather than assume.

### 1.8 The caveat: privacy is per-model — use `--tee-only` or the `private` profile

This is the most important thing to understand before you build a launcher, and
it is why §1.1 carries a warning.

**The catalog is a mix of two very different things.** The public `/v1/models`
response does not expose a privacy field at all, so `nearai_agy.py` also reads
the sibling endpoint `GET /v1/model/list` and merges its `metadata`:

| `providerType` | Models | In a TEE? |
| --- | --- | --- |
| `vllm` | 12 | **yes** — NEAR AI's own enclaves |
| `chutes` | 4 | **yes** — third-party TEE, verified end-to-end by NEAR AI |
| `tinfoil` | 1 | **yes** — third-party TEE |
| `external` | 47 | **no** — plain passthrough to the upstream vendor |

So of 64 catalog entries, **17 are attestable and 47 are not**. The gateway says
so itself, plainly, when you ask for attestation on the wrong model:

```
$ ./nearai_agy.py verify openai/gpt-5.6-sol
attestation rejected for openai/gpt-5.6-sol
  Provider error: Failed to fetch attestation report: ... TEE attestation is not
  supported for external providers. External providers run outside of our
  Trusted Execution Environment.
```

**Cross-referencing against what `agy` can actually use** narrows it further:

| Bucket | Count |
| --- | --- |
| Catalog entries | 64 |
| Attestable (TEE) | 17 |
| Usable by `agy` **and** attestable | **12** |
| — of those, tool-capable | **11** |
| Usable by `agy` (with `--strip-effort` shim) but **not** attestable | **42** |

The 11 that are both private and usable with tools are the genuinely
differentiated set: `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-27B-abliterated`,
`Qwen/Qwen3.6-35B-A3B-FP8`, `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
`deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated`, `deepseek/deepseek-v3.2`,
`moonshotai/kimi-k2.6`, `moonshotai/kimi-k3`, `qwen/qwen3.5-397b-a17b`,
`z-ai/glm-5.3`, `z-ai/glm-5.3-flash`.

**Two things to keep in mind:**

1. **The bundled `frontier` profile is not a privacy profile.** It sorts by
   price descending and takes 32, which by construction fills up with the
   expensive externally-hosted frontier models (OpenAI, Anthropic, Google, xAI).
   Its default is `openai/gpt-5.6-sol` — which is *not* in a TEE. If your reason
   for using NEAR AI is privacy, use the bundled **`private`** profile instead
   (`./nearai_agy.py build private`).
2. **Every table shows the `tee` tier.** `list`, `probe`, and `pick` display
   `vllm`, `chutes`, `tinfoil` (TEE) or `ext` (external non-TEE), and any
   command accepts `--tee-only` (`--private-only`).

> [!NOTE]
> The distinction between **"Private TEE"** (NEAR AI's own infrastructure) and
> **"Private 3P TEE"** (a third-party provider, verified by NEAR AI) is shown on
> the pricing page but not in either API list. Both are attestable; they are not
> the same supply chain. If that difference matters to you, verify per model
> rather than trusting this table.

---

## 2. Install

Requirements: **`bash`**, **`python3` ≥ 3.9** (stock macOS is fine), and `agy`
on your `PATH`. No third-party Python packages — standard library only, by
design.

```bash
cd /path/to/Near_AI
./nearai_agy.py init          # stores your API key, seeds starter profiles
./nearai_agy.py doctor        # health check: python, agy, key, catalog, cap
```

`init` prompts for an API key and asks `[y/N]` before writing it to
`~/.config/nearai-agy/config.json` with mode `0600` (environment-provided
`NEAR_AI_KEY` values stay ephemeral unless you explicitly pass `--save-key`).
Keys are never written into generated launchers.

`doctor` should look roughly like this:

```
nearai_agy 1.4.0
  python      : 3.13.9
  agy         : 1.5.5 (/Users/you/.local/bin/agy)
  api key     : present
  gateway     : https://cloud-api.near.ai/v1
  config      : /Users/you/.config/nearai-agy/config.json
  cache       : /Users/you/.cache/nearai-agy/models.json
  settings    : /Users/you/.gemini/antigravity-cli/settings.json (46 custom models, 46 with token limits)
  catalog     : 63 models, 0 min old
  model cap   : 32
```

---

## 3. Your first launcher in 60 seconds

The fastest path is to run directly using a bundled launcher in `bin/` (or via `./nearai_agy.py run <profile>`):

```bash
./bin/nearai-qwen                  # CLI with verified Qwen models
./bin/nearai-frontier              # CLI with 32 top models (includes Claude via local proxy)
./bin/nearai-ide-frontier --isolated .   # Antigravity IDE side-by-side with an existing window
```

For the **opt-in `private` profile** (TEE-only models, `"attestation_required": true`), `cmd_run` enforces a fail-closed attestation preflight by default:

```bash
./bin/nearai-private-check               # inspect live TDX/GPU/signature evidence
./bin/nearai-private --allow-unverified  # launch TEE-only routing explicitly
```

Or rebuild persistent forwarders whenever the catalog updates:

```bash
./nearai_agy.py build frontier           # refresh bin/nearai-frontier
./nearai_agy.py build --all-profiles --all-targets  # refresh CLI, IDE, and App launchers
```

> [!IMPORTANT]
> **How Anthropic Claude works out of the box (`--strip-effort`).** `agy`
> injects `reasoning_effort: "medium"` for the `anthropic/*` family, which the
> upstream validator rejects with HTTP 400 (`Extra inputs are not permitted`).
> Both `frontier` and `anthropic` profiles set `"strip_effort": true` (or CLI
> `--strip-effort`), which embeds a tiny zero-dependency local loopback shim in
> the launcher that strips `reasoning_effort` for `anthropic/*` requests before
> forwarding to NEAR AI.
>
> Meanwhile, the default `./nearai_agy.py probe` remains strictly faithful to
> unmodified `agy` (reporting `anthropic/*` as failing without the shim), while
> `./nearai_agy.py probe --what-if` tests with the field stripped.

### The five bundled profiles

```bash
./nearai_agy.py list-profiles
```

| Profile | What it gives you |
| --- | --- |
| `private` | All 11 tool-capable, hardware-attested TEE models (`vllm` / `chutes` / `tinfoil`). Best default for verifiable privacy. |
| `frontier` | The 32 most capable verified models (includes Claude via local shim). **Not a privacy profile** — see §1.8. |
| `anthropic` | All 12 tool-capable Anthropic Claude models with the local `reasoning_effort` stripper shim enabled. |
| `lite` | The 8 cheapest tool-capable verified models. Cheap daily driving. |
| `qwen` | Just the Qwen family, with three tiers (`pro` / `flash` / `flash_lite`). |

If privacy is the reason you are here, start with `private` or pass `--tee-only`:

```bash
./nearai_agy.py build private
./nearai_agy.py build --tee-only --family qwen --tools-only --name private-qwen
```

---

## 4. Choosing your own models

Interactive, with live probe verdicts:

```bash
./nearai_agy.py pick --tools-only
```

You get a checkbox list; space toggles, enter builds. It warns you if you
select more than the 32-slot cap.

Or by filter:

```bash
./nearai_agy.py build --tools-only --sort cheap --first 12 --name budget
./nearai_agy.py build --last 32 --name tail          # if you don't care which
./nearai_agy.py build --family qwen --tools-only --name qwen2
```

`--sort` accepts `catalog`, `cheap`, `expensive`, `context`, `new` and `name`.

---

## 5. Seeing what actually works

```bash
./nearai_agy.py list                       # the live catalog
./nearai_agy.py list --tools-verified      # only models verified to execute function calls
./nearai_agy.py list --probe               # + a real request per model
./nearai_agy.py list --probe --agy-only    # hide models agy cannot use
./nearai_agy.py probe --tools-only         # test basic ping
./nearai_agy.py probe --validate-tools     # verify structured OpenAI tool_calls (read_file)
./nearai_agy.py systems-test               # full live end-to-end systems test across models
```

`probe` sends one tiny request per model, shaped **exactly** the way `agy` shapes
it, and groups failures by cause:

```
summary
  ok   41 usable by agy
  fail 20 rejected agy's request shape

  13 model(s): agy sends reasoning_effort; the gateway rejects that field
      Not fixable from the launcher: agy adds the field for this model family
      and NEAR AI answers 400.
      anthropic/claude-fable-5                 reasoning_effort: Extra inputs are not permitted
      ...  (all 13 are anthropic/*)

  5 model(s): the gateway has no upstream route for it (404)
      Catalog metadata says it emits text, but nothing serves it at
      /chat/completions -- typical of image, audio, embedding and rerank models.
      black-forest-labs/FLUX.2-klein-4B        Upstream request failed with status 404 Not Found
      ...

  2 model(s): it lives on a different endpoint (/v1/systemone)
      The gateway answers 'Decision models require /v1/systemone', so this model
      is not reachable through the chat path agy uses.
      levanto/sage-2                           Decision models require /v1/systemone
      ...

  3 model(s) could not be reached (retry later; these are not verdicts)
      openai/o3-mini                           The model is currently unavailable. Please try again later.
      ...
```

Counts as of 2026-11-09: **64 models, 41 usable, 20 rejected, 3 transient.**

With `--tools-only` the failures are *all* Anthropic, because the non-chat models
in the catalog mostly do not advertise tools in the first place — so the
`404` and `/v1/systemone` groups disappear and you are left with 34 usable,
12 rejected, 3 unreachable.

The `13 x reasoning_effort` group is the interesting one: those models work fine
if you talk to the gateway yourself. See the callout in §3.

Verdicts are cached for 24 hours. Add `--refresh` to force a fresh sweep.

> [!TIP]
> A model added to the catalog five minutes ago may still show as failing from
> the cache. `--refresh` is the fix.

> [!NOTE]
> Only *decisive* verdicts are cached. Models that could not be reached at all
> (transient 5xx) are re-tested on every run, so a temporarily unavailable model
> fixes itself without you doing anything.

> [!TIP]
> **Adaptive Cache TTL:** Probe results use an adaptive expiration policy:
> working models are cached for 24 hours (`PROBE_TTL_OK = 86400s`), while failing
> or incompatible models expire after 2 hours (`PROBE_TTL_FAIL = 7200s`). This ensures
> that upstream model or gateway fixes are detected automatically without requiring
> a manual `--refresh`. Transient network errors (5xx/timeouts) are never cached.

---

## 6. Hybrid Orchestrator Mode: "Agy Pro" Plugin & Subagents

While **Method 1 (Full Gateway Mode)** routes your entire `agy` session through NEAR AI, **Method 2 (Hybrid Orchestrator Mode)** lets Google's native Gemini models (e.g. Gemini 3.5 Flash or Pro) remain your primary orchestrator in `agy`, delegating specialized multi-step tasks to NEAR AI models as subagents.

This pairs Gemini's expansive context window (1M+ tokens), fast reasoning, and native Antigravity workspace tooling with NEAR AI’s unique hardware-attested enclaves, refusal-free variants, and specialized open weights.

### Capability & Cost Routing Matrix

Rather than viewing models as a flat list, the Agy Pro integration operates as a **Task-Capability-Cost Router**:

| Specialized Need | Best-Fit Model | Why Gemini Delegates Here | Typical Skill / Task Match |
| :--- | :--- | :--- | :--- |
| **Confidential IP & Secrets** | `Qwen/Qwen3.8-27B` (TEE) | Zero-trust hardware enclave (AMD SEV / Intel SGX); neither NEAR AI nor the cloud host sees data or weights. | Security audits of private keys, proprietary algorithms, HIPAA/GDPR data analysis. |
| **Refusal-Free Security Research** | `Qwen/Qwen3.8-27B-abliterated` | No safety alignment refusals on offensive security or pen-testing tasks. | Vulnerability exploit reproduction, red-team fuzzing, malware reverse-engineering. |
| **Deep Architectural Refactoring** | `anthropic/claude-sonnet-4-5` | Premier code-editing precision and subtle semantic diffs across many files. | Large-scale AST rewrites, multi-module API design, framework migrations. |
| **High-Speed Math & Algorithms** | `deepseek/deepseek-v4.1-flash-1017` | Ultra-fast token generation, exceptional math & algorithm proofs at pennies per 1M tokens. | LeetCode/math proofs, micro-optimizations, regex generation, batch linting. |

### Installing the Plugin

To register the NEAR AI subagents and companion skills with Antigravity:

```bash
# Global install (available across all your projects in ~/.gemini/config/plugins/nearai)
./nearai_agy.py pro setup

# Or workspace-scoped install (.agents/plugins/nearai)
./nearai_agy.py pro setup --workspace
```

`pro setup` performs three tasks automatically:
1. Generates the Antigravity `plugin.json` manifest and 5 registered subagent markdown specs (`nearai-private.md`, `nearai-uncensored.md`, `nearai-claude.md`, `nearai-deepseek.md`, `nearai-general.md`).
2. Configures orchestration directives (`rules/AGENTS.md`) and 5 specialized companion skills under `skills/`:
   - `nearai-delegation`: Master delegation and dispatch directives.
   - `nearai-audit`: Confidential TEE security and cryptographic audit guidance.
   - `nearai-redteam`: Refusal-free exploit simulation and vulnerability analysis.
   - `nearai-refactor`: Multi-file AST refactoring and semantic diffing.
   - `nearai-optimize`: High-speed algorithmic complexity proofs and micro-optimizations.
3. Synchronizes positive context token limits in `~/.gemini/antigravity-cli/settings.json` under `customModels` with automatic `.bak` safety copies, preventing negative budget context errors (`ErrNoMessagesFit`).

### Inspecting & Testing Installed Subagents

```bash
# List all active NEAR AI subagents, target models, token budgets, and TEE attestation status
./nearai_agy.py pro list

# Test execution bridge directly from terminal (or within subagents via run_command)
./nearai_agy.py pro exec --model Qwen/Qwen3.8-27B --prompt "Audit this function for timing leaks"

# Or pipe code directly via stdin:
cat sensitive_auth.py | ./nearai_agy.py pro exec --model Qwen/Qwen3.8-27B --prompt "Check for constant-time comparison"
```

### Custom Subagent Management & Token Accounting

You can register custom subagents pinned to any catalog model and audit accumulated subagent spend:

```bash
# Register a custom subagent targeting Moonshot Kimi for long-context research
./nearai_agy.py pro add --name nearai-kimi --model moonshotai/kimi-k3 --role "Long-Context Code Researcher"

# Remove a subagent
./nearai_agy.py pro remove nearai-kimi

# View cumulative token usage and dollar spend across all subagent calls
./nearai_agy.py pro ledger

# Dump ledger transactions as structured JSON
./nearai_agy.py pro ledger --json

# Clear the usage ledger
./nearai_agy.py pro ledger --clear
```

### Empirical Model-to-Skill Benchmarking & Dynamic Routing

Rather than relying purely on static heuristics, `nearai_agy.py` features an empirical benchmarking suite and dynamic skill recommender that ground subagent selection in measured pass rates, latency, and exact token costs:

```bash
# Run the empirical benchmark suite across default specialist models
./nearai_agy.py pro benchmark

# Benchmark specific models on targeted domains
./nearai_agy.py pro benchmark --models deepseek/deepseek-v4.1-flash-1017,anthropic/claude-sonnet-4-5 --tasks algo_math,ast_refactor

# Save structured benchmark results to a custom report path
./nearai_agy.py pro benchmark --json reports/my_benchmark.json
```

Benchmark domains test realistic agent tasks:
- `sec_audit`: Constant-time comparison, cryptographic timing leaks, HMAC validation.
- `red_team`: Format string memory overwrite mechanics and fuzz testing (refusal-free test).
- `ast_refactor`: Async refactoring with concurrency (`TaskGroup` / `gather`) and typing.
- `algo_math`: Algorithm optimization from $O(2^N)$ to $O(N)$ time and $O(1)$ space with recurrence proofs.
- `json_schema`: Strict structured JSON schema compliance.

#### Query-Based Dynamic Recommendation

Query the recommendation engine to find the most cost-effective and capable specialist for any coding task:

```bash
# Recommend best subagent and model for a security audit
./nearai_agy.py pro recommend "audit cryptographic timing leaks in token comparator"

# Recommend specialist for high-performance algorithm refactoring
./nearai_agy.py pro recommend "optimize recursive dynamic programming to iterative O(1) space"

# Show JSON format for agent programmatic consumption
./nearai_agy.py pro recommend "refactor sync requests to asyncio TaskGroup" --json
```

The recommender outputs:
- Target subagent name (`nearai-private`, `nearai-deepseek`, etc.)
- Model identifier and TEE hardware enclave status
- Domain semantic match score and empirical pass rate
- Cost per 1M prompt/completion tokens
- Specific invocation prompt for `invoke_subagent` or `pro exec`

---

## 7. Everyday maintenance

Launchers are **self-healing**: at startup each one re-reads the catalog and
drops models that have disappeared, instead of dying with an HTTP 400. So a
launcher built last month still runs.

| Situation | Do this |
| --- | --- |
| Models disappeared from the catalog | Nothing — the launcher handles it |
| A brand-new model appeared | `./nearai_agy.py build frontier` (rebuild) |
| Something feels wrong | `./nearai_agy.py doctor` |
| Launcher reports missing token limits | `./nearai_agy.py sync-settings` |
| Model list looks stale | `./nearai_agy.py list --refresh` |
| Want to know why a model is skipped | `./nearai_agy.py probe --search <id>` |

Cleanup is trivial: everything generated lives in `bin/` next to the tool, and
profiles live in `profiles/`. Delete either freely.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `NEAR_AI_KEY is not set` | Launcher cannot find a key | `./nearai_agy.py init` or `export NEAR_AI_KEY=...` |
| `agy` launches with fewer models than expected | Some were dropped as incompatible | Read the `skipped` line; run `probe` |
| `no messages fit within the context token budget` | Gateway models lacked token limits in `settings.json`, causing negative context budget | Handled automatically in 1.4.0; run `./nearai_agy.py sync-settings` if `settings.json` was reset |
| A model 400s with `reasoning_effort` | `agy` injects the field for `anthropic/*`; the gateway rejects it | Build with `--strip-effort` (or use `frontier` / `anthropic` profiles, which enable it automatically) |
| `502 The model is currently unavailable` | Transient upstream outage | Retry with `probe --retry-transient`; not a permanent verdict |
| Launch refuses with an empty model list | `AGY_LLM_GATEWAY_MODELS` was unset | Rebuild the launcher |
| Models work in `curl` but not in `agy` | Wrong wire protocol | Ensure `AGY_LLM_GATEWAY_WIRE_PROTOCOL=openai` |
| `verify <model>` says `not supported for external providers` | That model is not in a TEE | Expected. Use `list --tee-only` or §1.8 for the attestable set. |
| Privacy was the goal, but `frontier` picked GPT/Claude/Grok | `frontier` sorts by price, not privacy | Use `./nearai_agy.py build private` or `--tee-only` — §1.8 |

---

## 9. Model selection cheat-sheet

| I want… | Use |
| --- | --- |
| Best all-round agentic coding (including Claude) | `frontier`, default `openai/gpt-5.6-sol` |
| Anthropic Claude models specifically | `anthropic`, default `anthropic/claude-sonnet-4-5` |
| Cheapest that still does tool calls | `lite` |
| **Private / verifiable TEE only** | `private` (`./nearai_agy.py build private`) — all 11 tool-capable TEE models |
| Unfiltered / uncensored **and** private | `Qwen/Qwen3.8-27B-abliterated` or `deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated` (both TEE, both tool-capable, included in `private`) |
| A specific family | `build --family <name> --tools-only` |
| Maximum context | `build --sort context --first 8` |
| To see everything before choosing | `pick --include-all-kinds` |

The two `*-abliterated` models are the intersection of every property this guide
argues for: open weights, refusal-free, tool-capable, and inside a hardware
enclave. If you only remember one row of this table, remember that one.

---

## 10. Turning this into a short video

A four-to-five minute walkthrough maps cleanly onto the sections above. Lead with
privacy, not with the catalog — "private by hardware" is the hook that survives
scrutiny.

| # | Segment | On screen | Keep it short by… |
| --- | --- | --- | --- |
| 1 | **Hook** (20s) | `agy` mid-run, then cut to the enclave diagram | Not explaining the 32-cap yet |
| 2 | **Why private** (45s) | §1.1: encrypted *in use*, operator-blind, no prompt logs | One threat (malicious host), not all four |
| 3 | **Prove it** (40s) | `curl` the attestation endpoint, no key; open the Verification Center | Not decoding the Intel quote on camera |
| 4 | **Setup** (40s) | `init` → `doctor` | Skipping the key prompt (cut away) |
| 5 | **First launch** (40s) | `build frontier` → `./bin/nearai-frontier` | Pausing on the `skipped:` line |
| 6 | **The catch** (60s) | §1.6: 32-cap + catalog ≠ usable; §1.8: catalog ≠ private | One probe example, not all four classes |
| 7 | **Pick your own** (60s) | `pick --tools-only`, choosing a TEE model | Toggling 3 boxes, then enter |
| 8 | **Close** (20s) | The two `*-abliterated` rows in §9 | Point at the docs |

Suggested framing for the hook, since it is the whole thesis in one sentence:

> "Your coding agent reads your whole repository and runs while you're not
> watching. Route it somewhere the operator can't read the prompts — and where
> you can check that claim instead of trusting it."

Two production notes. **Show the `skipped` line on camera** — it is the most
credible moment in the demo, because it proves the tool is not overselling. And
**do not fake the probe**; a real grouped-by-reason summary is more persuasive
than a clean animation.

One more honesty note, and it is worth saying on camera rather than hiding:
**not every NEAR AI model is private.** If the video claims "private" over a shot
of `frontier` running GPT-5.6, a knowledgeable viewer will catch it and the whole
segment loses credibility. Show a TEE model when you make the privacy claim. The
caveat makes the rest more believable, not less.

---

## 11. Where to go next

- [`ExecutiveBrief.md`](ExecutiveBrief.md) — standalone one-page executive overview: why private hardware-attested coding matters, comparative guarantees, and architecture
- [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md) — complete voiceover script and screen recording checklist for the video storyboard in §10
- [`examples/`](examples/) — runnable sample invocation scripts (`sample_nearai_agy.sh` for Method 1, `sample_nearai_agyPro_agents.sh` for Method 2)
- [`ArchitectureGuide.md`](ArchitectureGuide.md) — how it works, for humans and agents
- [`SECURITY.md`](SECURITY.md) — threat model, incident response, and vulnerability reporting
- [`README.md`](README.md) — quick start and command reference
- NEAR AI: <https://near.ai> · API docs: <https://cloud-api.near.ai/docs>
- Private inference (the source for §1.1): <https://docs.near.ai/cloud/private-inference>
- Verification Center: <https://cloud.near.ai/verify>