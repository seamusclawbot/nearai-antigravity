# NEAR AI for Google Antigravity (CLI, IDE & Desktop App)

Run **NEAR AI Cloud** models — including hardware-isolated TEE models (`Intel TDX` + `NVIDIA H200`), refusal-free open weights, DeepSeek, Qwen, and Anthropic Claude — directly inside **Antigravity CLI (`agy`)**, **Antigravity IDE**, and **Antigravity Desktop App**.

Zero third-party Python dependencies (`python3` 3.9+ stdlib only). Works out of the box in under 60 seconds.

---

## 60-Second Quick Start

### 1. Clone (or install via pip) & set your NEAR AI API key

```bash
git clone https://github.com/seamusclawbot/nearai-antigravity.git && cd nearai-antigravity

# Optional: install `nearai-agy` on PATH (core uses Python stdlib only; add [verifiers] for TDX/GPU crypto checks)
# pip install -e .                 # stdlib-only core
# pip install -e ".[verifiers]"    # include optional TDX/NRAS/OHTTP/signature verifiers

# Option A: Interactive setup (asks [y/N] before saving to mode-0600 config.json)
./nearai_agy.py init

# Option B: Keep your key purely in the shell environment (never written to disk)
export NEAR_AI_KEY="sk-..."
```

> Verify everything is ready at any time with `./nearai_agy.py doctor`.

### 2. Pick how you want to use NEAR AI in Antigravity

| Goal | Command |
| :--- | :--- |
| **Launch Antigravity CLI** with top frontier models (OpenAI, Claude, Gemini, etc.) | `./bin/nearai-frontier` |
| **Launch Antigravity CLI** with Qwen open-weight models | `./bin/nearai-qwen` |
| **Launch Antigravity IDE** side-by-side with your existing IDE window | `./bin/nearai-ide-frontier --isolated .` |
| **Launch Antigravity Desktop App** side-by-side | `./bin/nearai-app-qwen --isolated` |
| **Use TEE-only (`private`) models** *(checks attestation first; fails closed by default)* | `./bin/nearai-private-check` then `./bin/nearai-private --allow-unverified` |
| **Keep Gemini as main agent + add NEAR AI specialist subagents** *(Method 2)* | `./nearai_agy.py pro setup` then open a new `agy` / IDE chat |

---

## Why this tool exists

NEAR AI currently lists ~65 models and updates its catalog frequently, while Antigravity:
1. **Hard-caps `AGY_LLM_GATEWAY_MODELS` at 32 entries** (silently discarding extras, with tier slots consuming from the same pool).
2. **Requires positive token limits in `settings.json` / `jetski_state.pbtxt`** (without them, `agy` computes a negative context budget and crashes with `ErrNoMessagesFit`).
3. **Injects `reasoning_effort` for `anthropic/*`**, which NEAR AI rejects with HTTP 400 unless stripped by a local loopback proxy.

`nearai_agy.py` solves all three automatically — separating **intent** (`profiles/*.json`) from **snapshot**, syncing token budgets atomically, running a hardened loopback proxy (`127.0.0.1`), and shipping ready-to-run forwarders in `bin/`.

| Piece | Lives in | Changes |
|---|---|---|
| Intent ("all Qwen models with tools, capped at 12") | `profiles/*.json` (committed) | rarely |
| Live catalog | NEAR AI API, cached 1h | constantly |
| Generated launcher | `bin/nearai-<name>` | 19-line clean forwarder |

---

## Documentation

| Document | Read it when you want… |
|---|---|
| [`UserGuide.md`](UserGuide.md) | step-by-step onboarding, CLI/IDE/App workflows, Method 2 subagents, and troubleshooting |
| [`ExecutiveBrief.md`](ExecutiveBrief.md) | a one-page overview of private TEE coding, guarantees, and deployment modes |
| [`SECURITY.md`](SECURITY.md) | security boundary, vulnerability reporting, incident response, and attestation policy |
| [`ArchitectureGuide.md`](ArchitectureGuide.md) | internals, engineering invariants, proxy hardening, and testing strategy |
| [`docs/gateway-capability-matrix.md`](docs/gateway-capability-matrix.md) | confirmed Gateway endpoints, evidence semantics, and verification limits |
| [`docs/verification-contract-answers.md`](docs/verification-contract-answers.md) | researched cryptographic verification specifications (TDX, NRAS, OHTTP, signatures) |
| [`docs/verification-contract-questions.md`](docs/verification-contract-questions.md) | remaining upstream verification contract questions for NEAR AI |
| [`examples/`](examples/) | runnable walkthrough scripts (`sample_nearai_agy.sh` for Method 1, `sample_nearai_agyPro_agents.sh` for Method 2) |
| [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md) | voiceover script and screen-recording guide for a 4.5-minute walkthrough |

---

## Key Concepts & Privacy Guardrails

> [!WARNING]
> **Not every NEAR AI model is private.** Roughly **17 models run inside a TEE** (`providerType` of `vllm`, `chutes`, or `tinfoil`) while the rest are plain passthrough to external vendors. The bundled `frontier` profile sorts by price and selects mostly non-TEE models.
>
> If privacy is your goal, use the bundled `private` profile (`./bin/nearai-private`) or pass `--tee-only` (`--private-only`). You can inspect keyless hardware attestation evidence anytime with `./nearai_agy.py verify Qwen/Qwen3.8-27B` or run full cryptographic checks with `./bin/nearai-private-check`.

Profiles may set `"attestation_required": true` (enabled in [`profiles/private.json`](profiles/private.json)). When launching an `attestation_required` profile, `nearai_agy.py run` enforces a **fail-closed attestation preflight** before starting Antigravity. Because end-to-end upstream verification contracts remain partially open (`overall: unverified`), `./bin/nearai-private` will refuse to launch by default and direct you to run `./bin/nearai-private-check` (to inspect live TDX/GPU/signature evidence) or pass `--allow-unverified` to launch explicitly with TEE-only provider filtering.

---

## Command Reference



### Run (CLI, IDE, and Desktop App)

```bash
# Antigravity CLI (default)
./nearai_agy.py run qwen                  # launch agy directly using profiles/qwen.json
./nearai_agy.py run frontier models       # pass subcommands through to agy
./nearai_agy.py run anthropic --model ... # override default model

# Antigravity IDE (Antigravity IDE.app)
./nearai_agy.py run qwen --target ide               # launch Antigravity IDE with NEAR AI models
./nearai_agy.py run qwen --target ide --isolated .  # run side-by-side with an already-open IDE window

# Antigravity Desktop App (Antigravity.app)
./nearai_agy.py run qwen --target app
./nearai_agy.py run private --target app --isolated
```

> [!NOTE]
> **IDE / Desktop App Lifecycle & `--isolated`:**
> Both Antigravity IDE and Antigravity Desktop App are single-instance Electron apps.
> - Launchers stay attached in your terminal while the GUI app is open so the local
>   `GatewayProxy` (which strips `reasoning_effort` for `anthropic/*` models) stays alive
>   and tears down cleanly when you quit the app.
> - If Antigravity IDE or Desktop App is already running, launching without `--isolated`
>   exits with a reminder to either quit the existing instance or pass `--isolated`
>   (which uses an isolated `--user-data-dir` under `~/.config/nearai-agy/<target>-profiles/<profile>`
>   so you can run a NEAR AI session side-by-side with a standard Google session).
> - Custom install paths can be configured with `AGY_IDE_APP` / `AGY_IDE_BIN` and
>   `AGY_DESKTOP_APP` / `AGY_DESKTOP_BIN`.

### Inspect

```bash
./nearai_agy.py list                      # all chat models, catalog order
./nearai_agy.py list --tools-only         # only models that advertise tools
./nearai_agy.py list --tools-verified     # only models verified to execute function calls
./nearai_agy.py list --tee-only           # only hardware-attested TEE models (vllm/chutes/tinfoil)
./nearai_agy.py list --ready-only         # only is_ready = true (advisory)
./nearai_agy.py list --family qwen        # by id prefix (case-insensitive)
./nearai_agy.py list --provider anthropic # by owner
./nearai_agy.py list --search sonnet
./nearai_agy.py list --tools-only --sort cheap   # cheapest first
./nearai_agy.py list --refresh            # bypass the 1h cache
./nearai_agy.py list --include-all-kinds  # show image/embedding/rerank/systemone models too
./nearai_agy.py list --probe --agy-only   # only models agy can actually use
```

Sort keys: `catalog` (default, = truncation order), `cheap`, `expensive`,
`price_prompt`, `price_completion`, `context`, `name`.

The `tee` column shows the conservative provider trust class: `vllm`, `chutes`, and `tinfoil` are recognized TEE provider classes; `ext` is an external provider; `?` is unknown or legacy infrastructure. Unknown and external providers are never admitted by `--tee-only`, even when advisory attestation metadata is present. This incident-aware classification is not itself cryptographic verification.

```bash
./nearai_agy.py probe                     # live-test every model through agy's own request shape
./nearai_agy.py probe --validate-tools    # verify structured tool_calls (read_file schema)
./nearai_agy.py probe --tools-only        # the models that matter for a coding agent
./nearai_agy.py probe --what-if           # test with reasoning_effort stripped for anthropic/*
./nearai_agy.py probe --retry-transient   # re-probe only transient/unknown models
./nearai_agy.py systems-test              # full live systems test against gateway
./nearai_agy.py systems-test --profile qwen
./nearai_agy.py systems-test --all-catalog --workers 4 --update-cache --json reports/systems_test_report.json
./nearai_agy.py verify                    # verify gateway TEE attestation & measurement pins
./nearai_agy.py verify Qwen/Qwen3.8-27B   # verify gateway + model TEE attestation
./nearai_agy.py attest-evidence             # collect evidence-only nonce/TLS/OHTTP signals
./nearai_agy.py attest-evidence --verify-crypto  # also run TDX/GPU cryptographic verifiers
./nearai_agy.py attest-route-check Qwen/Qwen3.8-27B  # fail-closed opt-in preflight
```

`verify` and `attest-evidence` report evidence and narrowly scoped checks; `--verify-crypto` enables TDX (`dcap-qvl`), NVIDIA NRAS GPU, and OHTTP cryptographic verification when optional dependencies in [`requirements-verifier.txt`](requirements-verifier.txt) are installed (`--allow-out-of-date-tcb` optionally permits `OutOfDate` TCB advisories). Because end-to-end chained proof and ITA policy pinning remain open, `overall` stays `unverified`. See [`SECURITY.md`](SECURITY.md) for incident response, and run `python3 tools/check_gateway_openapi.py` only when an explicit live contract check is desired. Statuses such as `present_unverified` mean an artifact was returned but its trust contract is unresolved. `verified` is reserved for a specific local check whose inputs and semantics are fully defined. The opt-in `private` profile requires a fail-closed attestation preflight and may therefore refuse to launch.

### Select & Build

```bash
./nearai_agy.py pick                      # interactive multi-select → build
./nearai_agy.py build --first 32          # "just take the first 32"
./nearai_agy.py build --last 32           # "just take the last 32"
./nearai_agy.py build --first 12 --tools-only --name small
./nearai_agy.py build private             # from profiles/private.json (TEE models only)
./nearai_agy.py build qwen                # from profiles/qwen.json
./nearai_agy.py build --family anthropic --tools-only --strip-effort --name claude
./nearai_agy.py build --all-profiles                 # refresh every CLI launcher at once
./nearai_agy.py build qwen --target ide              # write bin/nearai-ide-qwen
./nearai_agy.py build qwen --target app              # write bin/nearai-app-qwen
./nearai_agy.py build --all-profiles --all-targets   # refresh CLI, IDE, and App launchers
./bin/nearai-ide-qwen --isolated .                   # launch IDE side-by-side on current repo
```

`pick` uses a curses UI on a TTY and a numbered prompt otherwise:

| Key | Action |
|---|---|
| `↑` / `↓` (or `k` / `j`) | move |
| `space` | toggle |
| `enter` | build with the current selection |
| `a` / `n` | select-visible / select-none |
| `/` | search mode (`enter` to accept, `esc` to clear) |
| `q` / `esc` | quit |

The status bar tracks `selected / usable slots`, reserving room for tiers, so you
see the 32-cap before you commit. In the numbered prompt, `f` = first N,
`l` = last N, `a` = all.

### Maintain & Health

```bash
./nearai_agy.py doctor            # versions, key, cache age, launcher staleness, settings sync, pro plugin
./nearai_agy.py doctor --save-cap # persist the detected model cap
./nearai_agy.py sync-settings     # sync model token limits into agy settings.json
./nearai_agy.py sync-settings --all-catalog # sync all catalog chat models
./nearai_agy.py sync-settings --check       # read-only audit: verify all models configured
./nearai_agy.py sync-settings --dry-run     # preview model limits without writing
./nearai_agy.py list-profiles
./nearai_agy.py build qwen --verify   # diff intended vs. agy-registered
./nearai_agy.py build qwen --dry-run  # print the script, write nothing
./nearai_agy.py build --strict ...    # exit 2 instead of warning on overflow
./nearai_agy.py build lite --probe    # warn about models agy cannot use
./nearai_agy.py build lite --drop-probe-failures  # swap them for working ones
```

### Agy Pro Plugin & External Subagents (Method 2)

Enable Hybrid Orchestrator Mode where Google Gemini acts as the main orchestrator in `agy`, delegating specialized tasks to NEAR AI subagents without relying on MCP servers:

```bash
./nearai_agy.py pro setup         # install plugin to ~/.gemini/config/plugins/nearai
./nearai_agy.py pro setup --workspace # or install to workspace (.agents/plugins/nearai)
./nearai_agy.py pro setup --dry-run   # preview generated files and model sync
./nearai_agy.py pro list          # list installed subagents, models, limits & TEE status
./nearai_agy.py pro add --name nearai-kimi --model moonshotai/kimi-k3 --role "Long Context Research"
./nearai_agy.py pro remove nearai-kimi # delete a subagent
./nearai_agy.py pro ledger        # view accumulated token usage and dollar spend across subagents
./nearai_agy.py pro exec --model Qwen/Qwen3.8-27B --prompt "Audit this token auth flow"
cat code.py | ./nearai_agy.py pro exec --model deepseek/deepseek-v4.1-flash-1017 # pipe code via stdin
./nearai_agy.py pro benchmark     # run empirical skill benchmarks across tasks & measure token cost
./nearai_agy.py pro recommend "audit cryptographic timing leaks" # dynamic model-to-skill routing
./nearai_agy.py pro uninstall     # cleanly remove plugin and bridge
```

> [!NOTE]
> **Context Window & Token Budget Self-Healing.** `agy` calculates its internal
> pre-invocation context compaction budget from `MaxTokens - MaxOutputTokens`.
> Gateway models passed only via environment variables default to `MaxTokens = 0`,
> creating a negative budget that crashes conversations with `no messages fit within the context token budget`.
> `nearai_agy.py` automatically registers model context lengths and output limits in
> `~/.gemini/antigravity-cli/settings.json` and `jetski_state.pbtxt` under `customModels`.
> Runtime execution performs read-only validation at startup, only updating atomically
> with `.bak` safety copies if an active limit is missing.

## Profiles

`profiles/<name>.json` is hand-editable and only describes *what you want*:

```json
{
  "description": "Qwen family, mirrors the original setupQwen.sh",
  "select": { "family": ["qwen"], "chat_only": true },
  "include": ["Qwen/Qwen3.8-27B", "Qwen/Qwen3-VL-30B-A3B-Instruct"],
  "tiers": {
    "pro": "qwen/qwen3.7-max",
    "flash": "Qwen/Qwen3.8-27B",
    "flash_lite": "Qwen/Qwen3.6-35B-A3B-FP8"
  },
  "launch": { "model": "Qwen/Qwen3.8-27B" }
}
```

| Key | Meaning |
|---|---|
| `select.family` / `select.provider` | filters (lists, case-insensitive) |
| `select.tools_only` / `tee_only` / `ready_only` / `chat_only` | booleans |
| `select.exclude` | exact ids to drop |
| `include` | exact ids forced to the **front**, in order |
| `count` | take the first N after filtering |
| `take.first` / `take.last` | slice after filtering |
| `tiers.pro` / `flash` / `flash_lite` | mapped onto the three tier variables |
| `launch.model` | default `--model` (skipped for `agy` subcommands and explicit `--model`) |
`launch.strip_effort` | embed local `reasoning_effort` stripper shim to unlock `anthropic/*` |
| `attestation_required` | explicit opt-in guard; requires `tee_only`/`private_only` and fail-closed attestation preflight |
| `sort` | order before slicing: `cheap`, `expensive`, `context`, `name`, ... |
| `probe.drop_failures` | live-test each candidate and swap out the ones agy cannot use |

Command-line flags override profile values.

Ships with `private`, `frontier`, `anthropic`, `qwen`, and `lite`. All five enable
`probe.drop_failures`, because catalog membership does **not** imply that `agy`
can talk to a model (see below). Both `frontier` and `anthropic` also enable
`launch.strip_effort`, so all 12 tool-capable Anthropic Claude models work out of
the box.


The active implementation uses the live OpenAPI contract for documented report-data, GPU-challenge, OHTTP, ITA-shape, and response-signature correlations. Complete TDX/GPU quote, ITA JWT, signing-key, and canonical-hash verification remains blocked on official trust roots and verifier policy.


The attestation collector is intentionally evidence-only. It validates documented request/TLS bindings and can report OHTTP signature status, but it does **not** claim TDX/GPU quote validity, certificate-chain validity, ITA JWT validity, or model execution authenticity. `overall` therefore remains `unverified` until NEAR AI publishes authoritative verifier semantics.

- **Rotate credentials:** revoke and replace `NEAR_AI_KEY` after any suspected exposure; prefer environment/config injection over committed files.
- **Review local state:** inspect `~/.config/nearai-agy`, generated launchers, model/probe caches, and shell history after an incident. Do not persist raw JWTs, quotes, certificates, or keys.
- **Trust classes:** only recognized `vllm`, `chutes`, and `tinfoil` providers are eligible for `--tee-only`; `unknown`/legacy and external providers remain excluded.
- **Emergency disable:** remove or rename private launchers, unset gateway credentials, and disable profiles marked `attestation_required` until evidence review is complete.
- **Opt-in enforcement:** profiles with `"attestation_required": true` must select `tee_only`/`private_only`; run `./nearai_agy.py attest-route-check MODEL` immediately before private execution. The check fails closed on unavailable, stale, mismatched, or unverified evidence. Ordinary profiles retain existing routing behavior.

> [!WARNING]
> Metadata fields, quote presence, token presence, and successful HTTP responses are not proof of confidential execution. Do not weaken the fail-closed policy without an official NEAR AI verification contract.


This is the most important thing this tool knows.

`agy` decides per model whether to send a `reasoning_effort` field. NEAR AI
rejects that field on some provider paths, so those models fail at request time
with:

```
400 Provider failed for model 'anthropic/claude-fable-5':
    reasoning_effort: Extra inputs are not permitted
```

The rule was derived by pointing `agy` at a local logging gateway and recording
the actual request bodies, then confirmed by end-to-end launches:

| Model family | `reasoning_effort` agy sends | Works on NEAR AI? |
|---|---|---|
| `anthropic/*` | `medium` | **no through `agy`** — all 13 rejected |
| `google/*` | `medium` | yes |
| `openai/o3*`, `openai/o4*` | `low` | yes (`o3-mini`/`o4-mini` were returning 502) |
| everything else | *not sent* | yes |

Measured on the 64-model catalog: **41 usable, 20 unusable, 3 transient.** Of the
49 tool-capable models, 34 work, 12 do not and 3 were unreachable.

> [!IMPORTANT]
> **The 13 Anthropic models are not broken — `agy` is.** They work fine if the
> field is absent. Verified on all 13 `anthropic/*` models (`probe --what-if --family anthropic`):
> without `reasoning_effort`, all 13 return HTTP 200, and end-to-end `agy`
> launches succeed through a local stripper shim (`build --strip-effort`).
>
> It is *still* correct for the default probe to report them unusable, because
> the default probe reproduces `agy`'s exact unmodified body. Pass
> `probe --what-if` to test with stripping enabled, or `build --strip-effort` to
> embed the local loopback shim in a generated launcher. See
> [`ArchitectureGuide.md`](ArchitectureGuide.md).

Nothing in the catalog metadata predicts this — `is_ready`, `supported_features`,
and `supported_sampling_parameters` all fail to distinguish working models from
broken ones. Only a real request tells you. Because `agy` does not expose a CLI
flag or environment variable to suppress per-family `reasoning_effort` injection,
`nearai_agy.py` handles it cleanly via its local loopback proxy (`GatewayProxy`).

Hence `probe`. It replays the exact body `agy` would send — same
`reasoning_effort` decision, same `max_completion_tokens` — so its verdict
matches a real launch. A full 64-model sweep takes about 30 seconds and a few
hundred tokens:

```
$ ./nearai_agy.py probe --search anthropic
summary
  ok   0 usable by agy
  fail 13 rejected agy's request shape

  13 model(s): agy sends reasoning_effort; the gateway rejects that field
      Not fixable from the launcher: agy adds the field for this model family
      and NEAR AI answers 400.
      anthropic/claude-fable-5                 reasoning_effort: Extra inputs are not permitted
      ...
```

Casualties are grouped by **why** they failed, not dumped as a flat list of
names — the four classes currently seen are `reasoning_effort` rejection,
no-upstream-route (404), wrong-endpoint (`/v1/systemone`), and transient 502.
`build` prints the same breakdown for whatever it drops, and repeats the count
on the `skipped` line of its summary so it is not scrolled past.

For the full inventory, the error taxonomy, and resolution details, see
[`ArchitectureGuide.md`](ArchitectureGuide.md).

`--drop-probe-failures` (or `"probe": {"drop_failures": true}` in a profile)
removes those models **before** the 32-slot slice, so the freed slots go to
models that work instead of being wasted. That is why `frontier` fills all 32
slots with 32 usable models rather than 32 entries of which 12 are dead.

## The 32-slot budget

Verified against `agy` 1.5.3:

- `AGY_LLM_GATEWAY_MODELS` keeps the **first 32** entries; extras are dropped
  silently.
- Tier models that are not already in the list are **appended**, consuming slots,
  in the order `flash`, `pro`, `flash_lite`.
- A tier already present in the list costs **no** extra slot.
- An **empty or unset** `AGY_LLM_GATEWAY_MODELS` makes `agy` auto-discover the
  whole catalog and truncate at 32 — the launcher treats that as a bug and
  refuses to launch rather than silently substituting different models.

The builder replicates this accounting and warns with the exact list of models
that will not be registered:

```
warning: 35 models requested but agy only keeps 32; these will NOT be registered:
         - openai/gpt-5
         - openai/gpt-5.1
         - openai/gpt-5.2
```

Use `--strict` to make that a hard error (exit 2) in scripts.

## Files

```
Near_AI/
├── nearai_agy.py                main CLI & execution engine (stdlib only)
├── attestation_evidence.py      fail-closed attestation evidence collector
├── verifiers/                   optional cryptographic verifiers (TDX, NRAS, OHTTP, signatures)
├── profiles/*.json              intent profiles (committed)
├── bin/nearai-*                 lightweight bash forwarders + attestation helpers
├── tests/                       offline unit & security-hygiene test suite (66 tests, pytest -v)
├── reports/                     empirical benchmark & systems-test snapshots
├── examples/                    runnable Method 1 & Method 2 sample scripts
└── tools/                       OpenAPI drift checker & local SSE mock gateway
```

Generated launchers are written next to the tool so they are easy to find — and
easy to delete.

## State

| Path | Purpose |
|---|---|
| `~/.config/nearai-agy/config.json` | API key (mode 600), gateway, cap, cache TTL |
| `~/.cache/nearai-agy/models.json` | full catalog for `list` (TTL, default 1h) |
| `~/.cache/nearai-agy/alive.json` | id-only list used by launchers |
| `~/.cache/nearai-agy/probe.json` | per-model agy usability verdicts (TTL 24h) |

The key is resolved as `$NEAR_AI_KEY` → config file → interactive prompt. It is
never written into a generated launcher.

## Notes

- The catalog endpoint is **public** — `GET /v1/models` returns HTTP 200 even with
  no `Authorization` header, so `list` works without a key. A key is only needed
  for actual inference.
- `is_ready` is **advisory**. `is_ready:false` models still served requests when
  tested (17 of them), and 3 models marked `is_ready:true` were broken, so the
  builder never filters on it automatically (`--ready-only` opts in). Use
  `probe` instead — it is the only reliable health check.
- Model ids are **case-sensitive** (`Qwen/Qwen3.8-27B` ≠ `qwen/qwen3.8-27b`) and
  the catalog contains both, so ids are passed through verbatim.
- Only Python's standard library is used, and the code targets Python 3.9 so it
  runs under macOS's `/usr/bin/python3`.
- The generated bash avoids `"${arr[@]}"` on empty arrays because bash 3.2 (the
  macOS default) treats that as an unbound-variable error under `set -u`.

## Disclaimer

This is an independent, community-maintained open-source project and is not affiliated with, endorsed by, or sponsored by Google LLC or NEAR AI. It interacts with Antigravity exclusively through its standard gateway environment variables, configuration files, and plugin interfaces using your own NEAR AI API key. Users are responsible for complying with the Google Antigravity Terms of Service, NEAR AI Terms of Service, and the terms of any upstream model providers they choose to use.
