#!/usr/bin/env python3
"""nearai_agy.py — build self-healing Antigravity CLI launchers for NEAR AI Cloud models.

WHY THIS EXISTS
===============
NEAR AI Cloud publishes its catalog at ``GET {gateway}/models`` (currently ~64
models, and it changes as models are added and retired).  The Antigravity CLI
reads a comma-separated list from ``AGY_LLM_GATEWAY_MODELS`` but **hard-caps it
at 32 entries, keeping the first 32 and silently discarding the rest**.  The
three tier variables (``AGY_LLM_GATEWAY_TIER_PRO`` / ``_FLASH`` /
``_FLASH_LITE``) are appended into that same 32-slot pool, in the order
flash, pro, flash_lite — so a tier can silently evict a model.

Hand-maintaining a model list therefore rots: models disappear, the cap bites,
and a stale id only fails at request time with an opaque HTTP 400.

This tool is the *generator*.  You express **intent** ("all Qwen models that
support tools, capped at 12, with these tiers") in a JSON profile; the tool
resolves that intent against the *live* catalog and emits a small bash
**launcher** that you then run all day.

The launcher is deliberately **self-healing** rather than a frozen snapshot:
before handing the list to ``agy`` it re-checks the catalog (cached, so the
fast path costs nothing) and drops models that no longer exist, warning loudly
instead of dying with a 400.  That way a launcher built months ago still works.

Separately -- and this is the non-obvious part -- **being in the catalog does
not mean agy can use a model**.  agy injects a ``reasoning_effort`` field for
some model families and NEAR AI rejects that field outright for several of
them, so those models fail at request time.  Nothing in the catalog metadata
predicts it, so ``probe`` replays agy's exact request body to find out, and
``build --drop-probe-failures`` swaps the casualties for models that work.
Casualties are never dropped silently: they are grouped by *why* they failed
(reasoning_effort rejection, no upstream route, wrong endpoint, transient), and
``ArchitectureGuide.md`` documents the taxonomy and resolution.

USAGE
=====
    ./nearai_agy.py init                 # one-time: store API key, sample profiles
    ./nearai_agy.py list                 # show the live catalog
    ./nearai_agy.py list --tools-only    # only models agy can actually drive
    ./nearai_agy.py probe                # which models really work through agy
    ./nearai_agy.py probe --tools-only   # just the ones that matter for coding
    ./nearai_agy.py pick                 # interactive multi-select, then build
    ./nearai_agy.py build --first 32 --name bulk
    ./nearai_agy.py build --last 32  --name tail
    ./nearai_agy.py build qwen           # from profiles/qwen.json
    ./nearai_agy.py build qwen --verify  # prove what agy actually registered
    ./nearai_agy.py doctor               # health check + cap probe
    ./nearai_agy.py sync-settings        # sync model token limits into agy settings.json
    ./nearai_agy.py list-profiles

OUTPUT
======
    profiles/<name>.json     intent (checked into the repo, hand-editable)
    bin/nearai-<name>        generated launcher (next to the tool, easy to find/delete)

Only the Python standard library is used, and it must run on Python 3.9 or
newer (macOS ships 3.9.6 as /usr/bin/python3), so no third-party imports.
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import http.server
import ipaddress
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from attestation_evidence import (
    build_verification_summary,
    collect_evidence,
    redact_secrets,
    verify_chat_completion,
)

VERSION = "1.0.0"

DEFAULT_GATEWAY = "https://cloud-api.near.ai/v1"
DEFAULT_TTL = 3600
CAP_FALLBACK = 32

# Models whose IDs contain these tokens are served on dedicated non-chat
# endpoints (/v1/rerank, /v1/embeddings, /v1/audio/transcriptions,
# /v1/privacy/classify) even when their catalog output_modalities lists "text".
NON_CHAT_KEYWORDS = ("rerank", "embed", "whisper", "privacy-filter", "classify")

# Model ids are passed verbatim to agy (they are case sensitive, e.g.
# "Qwen/Qwen3.8-27B"), but they are embedded in a bash script, so validate
# them to keep the generated file bullet-proof.
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_API_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MiB

TOOL_DIR = Path(__file__).resolve().parent
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "nearai-agy"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "nearai-agy"
if (TOOL_DIR / "profiles").is_dir():
    PROFILE_DIR = TOOL_DIR / "profiles"
    LAUNCHER_DIR = TOOL_DIR / "bin"
elif (Path.cwd() / "profiles").is_dir() and (Path.cwd() / "nearai_agy.py").is_file():
    PROFILE_DIR = Path.cwd() / "profiles"
    LAUNCHER_DIR = Path.cwd() / "bin"
else:
    PROFILE_DIR = CONFIG_DIR / "profiles"
    LAUNCHER_DIR = Path.home() / ".local" / "bin"
CONFIG_PATH = CONFIG_DIR / "config.json"
CACHE_PATH = CACHE_DIR / "models.json"
# Launchers use a separate, ids-only cache: they only need liveness, and keeping
# the schemas apart means a launcher refresh can never degrade `list` metadata.
ALIVE_CACHE_PATH = CACHE_DIR / "alive.json"
# Results of the live request probe: {model_id: {"ok": bool, "code": int,
# "error": str, "at": epoch}}. Probed on demand, never trusted forever.
PROBE_CACHE_PATH = CACHE_DIR / "probe.json"
BENCHMARK_CACHE_PATH = CACHE_DIR / "benchmark_matrix.json"
LEDGER_PATH = CACHE_DIR / "usage_ledger.jsonl"

# Antigravity CLI global settings.json path where `customModels` metadata is stored.
# agy requires custom gateway models to have positive `maxTokens` and `maxOutputTokens`
# defined in `customModels`; otherwise it defaults to a zero context window and fails with:
#   "no messages fit within the context token budget"
AGY_SETTINGS_PATH = Path(
    os.environ.get(
        "AGY_SETTINGS_PATH",
        str(Path.home() / ".gemini" / "antigravity-cli" / "settings.json"),
    )
)

AGY_STATE_PATH = Path(
    os.environ.get(
        "AGY_STATE_PATH",
        str(Path.home() / ".gemini" / "antigravity-cli" / "jetski_state.pbtxt"),
    )
)

PRO_PLUGIN_GLOBAL_DIR = Path.home() / ".gemini" / "config" / "plugins" / "nearai"
PRO_PLUGIN_WORKSPACE_DIR = Path.cwd() / ".agents" / "plugins" / "nearai"

DEFAULT_SUBAGENTS = {
    "nearai-private": {
        "model": "Qwen/Qwen3.8-27B",
        "description": "Hardware-attested TEE specialist for confidential code auditing, secrets scanning, and private algorithmic analysis running in Intel TDX / AMD SEV hardware enclaves.",
        "tee": True,
        "persona": "NEAR AI Private TEE Specialist",
        "instructions": (
            "You are an expert confidential code auditor and privacy architect.\n"
            "You operate with a zero-trust posture using NEAR AI's hardware-attested Trusted Execution Environment (TEE).\n\n"
            "## Target Model\n"
            "- **Model:** `Qwen/Qwen3.8-27B` (or verified TEE enclave model)\n"
            "- **Execution Environment:** Hardware-isolated enclave (Intel TDX / NVIDIA H200 GPU TEE). Neither cloud operators nor NEAR AI can access prompts, repository source code, or weights during execution.\n\n"
            "## Core Directives\n"
            "1. **Confidentiality:** Specialize in reviewing sensitive IP, private cryptographic keys, authentication logic, database credentials, and compliance-critical code (HIPAA, GDPR, SOC2).\n"
            "2. **Precision:** Provide verifiable, mathematically sound reviews without exposing sensitive strings or tokens outside the enclave.\n"
            "3. **Tool Execution:** Execute analysis tasks via `nearai-agent-runner --model Qwen/Qwen3.8-27B --stream`."
        ),
    },
    "nearai-uncensored": {
        "model": "Qwen/Qwen3.8-27B-abliterated",
        "description": "Unfiltered defensive security reviewer for vulnerability verification, hardening audits, protocol edge-case analysis, and remediation patches on NEAR AI.",
        "tee": True,
        "persona": "NEAR AI Unfiltered Security Specialist",
        "instructions": (
            "You are a defensive security engineer and vulnerability hardening specialist.\n\n"
            "## Target Model\n"
            "- **Model:** `Qwen/Qwen3.8-27B-abliterated`\n"
            "- **Capability:** Unfiltered technical security review suited for deep protocol auditing, boundary/parser edge-case analysis, and concrete defensive hardening without false-positive refusals on benign security code.\n\n"
            "## Core Directives\n"
            "1. **Deep Security Review:** Analyze boundary conditions, parser differentials, input validation gaps, protocol framing issues, auth isolation, and resource-exhaustion risks.\n"
            "2. **Defensive Hardening:** Pair every verified vulnerability finding with concrete, drop-in defensive remediation patches and regression tests.\n"
            "3. **Tool Execution:** Execute analysis tasks via `nearai-agent-runner --model Qwen/Qwen3.8-27B-abliterated --stream`."
        ),
    },
    "nearai-claude": {
        "model": "anthropic/claude-sonnet-4-5",
        "description": "Cross-repository architecture refactoring and multi-file code editing specialist powered by Anthropic Claude 3.7 / 3.5 Sonnet on NEAR AI.",
        "tee": False,
        "persona": "NEAR AI Claude Architecture Specialist",
        "instructions": (
            "You are a principal software architect and refactoring engineer powered by Anthropic Claude on NEAR AI.\n\n"
            "## Target Model\n"
            "- **Model:** `anthropic/claude-sonnet-4-5`\n"
            "- **Capabilities:** Premier multi-file code refactoring, AST restructuring, API schema design, and subtle semantic diffs.\n\n"
            "## Core Directives\n"
            "1. **Architectural Integrity:** When refactoring systems, preserve public API contracts, backward compatibility, and comprehensive docstrings.\n"
            "2. **Tool Execution:** Execute refactoring tasks via `nearai-agent-runner --model anthropic/claude-sonnet-4-5 --stream`."
        ),
    },
    "nearai-deepseek": {
        "model": "deepseek/deepseek-v4.1-flash-1017",
        "description": "High-speed algorithmic optimization, mathematical proofs, performance profiling, and cost-effective batch analysis powered by DeepSeek on NEAR AI.",
        "tee": False,
        "persona": "NEAR AI DeepSeek Algorithmic Specialist",
        "instructions": (
            "You are an algorithmic optimization, complexity analysis, and performance engineering specialist.\n\n"
            "## Target Model\n"
            "- **Model:** `deepseek/deepseek-v4.1-flash-1017`\n"
            "- **Capabilities:** High-speed token generation, deep mathematical proofs, competitive programming optimizations, regex generation, and cost-effective bulk code analysis.\n\n"
            "## Core Directives\n"
            "1. **Algorithmic Complexity:** Optimize time and space complexity (O(N) vs O(N^2)), benchmark hot loops, and reduce memory footprints.\n"
            "2. **Tool Execution:** Execute tasks via `nearai-agent-runner --model deepseek/deepseek-v4.1-flash-1017 --stream`."
        ),
    },
    "nearai-general": {
        "model": "openai/gpt-5.6-sol",
        "description": "General-purpose NEAR AI task delegate for cost-effective background operations, large document summarization, and independent testing.",
        "tee": False,
        "persona": "NEAR AI General Delegate",
        "instructions": (
            "You are a general-purpose delegate for secondary tasks, background summarization, and parallel code analysis.\n\n"
            "## Target Model\n"
            "- **Model:** `openai/gpt-5.6-sol`\n\n"
            "## Core Directives\n"
            "1. **Efficiency:** Deliver structured, high-signal results for the primary orchestrator.\n"
            "2. **Tool Execution:** Execute tasks via `nearai-agent-runner --model openai/gpt-5.6-sol --stream`."
        ),
    },
}

# agy does NOT always send `reasoning_effort`; it decides per model, and the
# decision is by *family prefix*, not by anything in the catalog. NEAR AI
# rejects the field outright for several families ("reasoning_effort: Extra
# inputs are not permitted"), so a catalog entry is not proof that agy can use
# the model. The only faithful test is to send the same body agy would send.
# (The table below was derived by pointing agy at a local logging gateway and
# recording the real request bodies -- it is observation, not guesswork.)
def agy_effort_for(model_id: str):
    """Return the `reasoning_effort` agy injects for this model, or None."""
    head = model_id.split("/")[0].lower()
    rest = model_id.lower()
    if head == "anthropic":
        return "medium"
    if head == "google":
        return "medium"
    if rest.startswith("openai/o3") or rest.startswith("openai/o4"):
        return "low"
    return None


# A 4-token completion reproduces agy's verdict in milliseconds, so sweeping the
# whole catalog costs a few seconds and a few hundred tokens.
PROBE_MAX_TOKENS = 4
PROBE_TIMEOUT = 25
PROBE_WORKERS = 8
PROBE_TTL_OK = 86400    # 24h for working models
PROBE_TTL_FAIL = 7200   # 2h adaptive TTL for incompatible models (fast upstream fix detection)
PROBE_TTL = PROBE_TTL_OK

# Function-calling probe constants: realistic schema and budget for tool invocation
PROBE_TOOL_MAX_TOKENS = 128
PROBE_TOOL_TIMEOUT = 35

PROBE_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    }
]

PROBE_TOOL_MESSAGES = [
    {
        "role": "user",
        "content": "Please read the file '/workspace/config.json' using the read_file function.",
    }
]

# agy subcommands: never inject a default --model in front of these.
AGY_SUBCOMMANDS = {
    "models", "help", "version", "changelog", "install", "mcp", "plugin",
    "plugins", "agent", "agents", "acp", "mic-serve", "update", "remote-control",
}

# The order agy appends tier models into the pool (verified empirically).
TIER_APPEND_ORDER = ("flash", "pro", "flash_lite")


# ---------------------------------------------------------------------------
# tiny terminal helpers
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return "\033[%sm%s\033[0m" % (code, text) if _COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def cyan(t: str) -> str:
    return _c("36", t)


def warn(msg: str) -> None:
    sys.stderr.write(yellow("  warning: ") + msg + "\n")


def info(msg: str) -> None:
    sys.stderr.write(dim("  " + msg) + "\n")


def err(msg: str) -> None:
    """Print a credential-redacted error to stderr without exiting."""
    safe_msg = str(redact_secrets(str(msg)))
    sys.stderr.write(red("error: ") + safe_msg + "\n")


def die(msg: str) -> None:
    """Print a credential-redacted error to stderr and exit with code 1."""
    err(msg)
    raise SystemExit(1)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disable automatic HTTP redirects so Authorization headers never leak cross-origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _validate_safe_name(value: str, label: str = "name") -> str:
    """Validate a profile, launcher, or subagent identifier (no path traversal or shell metachars)."""
    cleaned = (value or "").strip()
    if not cleaned or ".." in cleaned or not SAFE_NAME_RE.match(cleaned):
        raise ValueError("unsafe %s: %r" % (label, value))
    return cleaned


def _read_http_bounded(resp, max_bytes: int = MAX_API_RESPONSE_BYTES) -> bytes:
    """Read an HTTP response body up to max_bytes."""
    try:
        data = resp.read(max_bytes + 1)
    except TypeError:
        data = resp.read()
    if len(data) > max_bytes:
        raise ValueError("HTTP response exceeded %d byte limit" % max_bytes)
    return data


def _atomic_write_text(path: Path, text: str, *, mode: int = None, dir_mode: int = None) -> None:
    """Atomically write text via mkstemp + fsync + os.replace with optional strict permissions."""
    if dir_mode is not None:
        path.parent.mkdir(parents=True, mode=dir_mode, exist_ok=True)
        try:
            os.chmod(path.parent, dir_mode)
        except OSError:
            pass
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    fd = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
        tmp_path = Path(tmp_name)
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        if mode is not None:
            try:
                os.chmod(path, mode)
            except OSError:
                pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _append_private_line(path: Path, line: str) -> None:
    """Append a line to a private (0600) file inside a 0700 directory."""
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_loopback_host(host: str) -> bool:
    """Return True only when host is localhost or a loopback IP address."""
    h = (host or "").strip().lower()
    if h in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _validate_gateway_url(url: str) -> str:
    """Validate that a gateway URL is a well-formed https:// (or loopback http://) URL."""
    u = (url or "").strip()
    if not u or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in u):
        raise ValueError("gateway_url must be a non-empty URL without whitespace or control characters")
    if any(ch in u for ch in ('"', "'", "`", "$", ";", "|")):
        raise ValueError("gateway_url contains unsafe characters")
    p = urllib.parse.urlsplit(u)
    if p.scheme not in ("https", "http"):
        raise ValueError("gateway_url must use https:// (or loopback http://)")
    if not p.hostname:
        raise ValueError("gateway_url must include a valid hostname")
    if p.username is not None or p.password is not None:
        raise ValueError("gateway_url must not contain embedded userinfo")
    if p.fragment:
        raise ValueError("gateway_url must not contain a fragment")
    try:
        port = p.port
    except ValueError as exc:
        raise ValueError("gateway_url contains an invalid port") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("gateway_url port out of range")
    if p.scheme == "http" and not _is_loopback_host(p.hostname):
        raise ValueError("http:// gateway_url is only permitted for loopback hosts")
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path.rstrip("/"), p.query, ""))


def _validate_api_key(key: str) -> str:
    """Reject API keys with control characters or internal whitespace (HTTP header safety)."""
    cleaned = (key or "").strip()
    if not cleaned:
        return ""
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise ValueError("API key contains whitespace or control characters")
    return cleaned


def load_config() -> dict:
    cfg = {
        "gateway_url": DEFAULT_GATEWAY,
        "api_key": "",
        "cache_ttl": DEFAULT_TTL,
        "model_cap": CAP_FALLBACK,
    }
    try:
        st = os.lstat(CONFIG_PATH)
    except FileNotFoundError:
        return cfg
    except OSError as exc:
        warn("could not stat config %s: %s" % (CONFIG_PATH, exc))
        return cfg

    if stat.S_ISLNK(st.st_mode):
        warn("config %s is a symlink; refusing to load untrusted config" % CONFIG_PATH)
        return cfg
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        warn("config %s is not owned by current user; ignoring stored api_key" % CONFIG_PATH)
        return cfg
    if (st.st_mode & 0o077) != 0:
        warn("config %s has overly permissive mode %04o (expected 0600)" % (CONFIG_PATH, stat.S_IMODE(st.st_mode)))

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
        if isinstance(stored, dict):
            cfg.update({k: v for k, v in stored.items() if v is not None})
        else:
            warn("config %s is not a JSON object; using defaults" % CONFIG_PATH)
    except (OSError, ValueError) as exc:
        warn("could not parse config %s (%s); using defaults" % (CONFIG_PATH, exc))
        return cfg

    try:
        cfg["gateway_url"] = _validate_gateway_url(str(cfg.get("gateway_url") or DEFAULT_GATEWAY))
    except ValueError as exc:
        warn("invalid gateway_url in %s (%s); falling back to default" % (CONFIG_PATH, exc))
        cfg["gateway_url"] = DEFAULT_GATEWAY
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(CONFIG_DIR, 0o700)
    tmp_path = None
    fd = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(CONFIG_DIR), prefix=".config.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, CONFIG_PATH)
        tmp_path = None
        os.chmod(CONFIG_PATH, 0o600)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def resolve_api_key(cfg: dict, prompt: bool = False, save: bool = None) -> str:
    """Environment wins, then the config file, then (optionally) an interactive prompt.

    When `prompt=True`, keys are kept ephemeral by default unless `save=True` is
    passed explicitly or the user confirms saving to `CONFIG_PATH`.
    """
    env_raw = os.environ.get("NEAR_AI_KEY", "")
    if env_raw.strip():
        return _validate_api_key(env_raw)
    cfg_raw = cfg.get("api_key") or ""
    if cfg_raw.strip():
        return _validate_api_key(cfg_raw)
    if prompt and sys.stdin.isatty():
        try:
            entered = _validate_api_key(getpass.getpass("NEAR AI API key (sk-...): "))
        except (EOFError, KeyboardInterrupt):
            entered = ""
        except ValueError as exc:
            warn(str(exc))
            entered = ""
        if entered:
            should_save = save
            if should_save is None:
                try:
                    ans = input("Save API key to %s (mode 0600)? [y/N]: " % CONFIG_PATH).strip().lower()
                    should_save = ans in ("y", "yes")
                except (EOFError, KeyboardInterrupt):
                    should_save = False
            if should_save:
                updated = dict(cfg)
                updated["api_key"] = entered
                save_config(updated)
                cfg["api_key"] = entered
                info("stored in %s (mode 600)" % CONFIG_PATH)
            else:
                info("using API key for this session only (not saved to disk)")
            return entered
    return ""


# ---------------------------------------------------------------------------
# catalog: fetch + cache
# ---------------------------------------------------------------------------


def _fetch_privacy_metadata(url: str, timeout: int = 15) -> dict:
    """Fetch TEE / privacy tier metadata from GET {gateway}/model/list.

    Returns {model_id: {"provider_type": str, "verifiable": bool,
                        "attestation_supported": bool}}.
    Best-effort: if the endpoint fails, returns {} so catalog loading never breaks.
    """
    try:
        base = _validate_gateway_url(url).rstrip("/")
    except ValueError:
        return {}
    out = {}
    offset = 0
    limit = 100
    max_pages = 50
    try:
        for _ in range(max_pages):
            req = urllib.request.Request(
                "%s/model/list?limit=%d&offset=%d" % (base, limit, offset),
                headers={"Accept": "application/json"},
            )
            with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
                payload = json.loads(_read_http_bounded(resp).decode("utf-8"))
            items = payload.get("models") or []
            for item in items:
                mid = item.get("modelId") or item.get("id")
                meta = item.get("metadata") or {}
                if mid:
                    out[mid] = {
                        "provider_type": meta.get("providerType") or "",
                        "verifiable": bool(meta.get("verifiable")),
                        "attestation_supported": bool(meta.get("attestationSupported")),
                    }
            total = int(payload.get("total") or 0)
            offset += len(items)
            if not items or offset >= total:
                break
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        pass
    return out


def _fetch_catalog(url: str, key: str, timeout: int = 20) -> list:
    safe_url = _validate_gateway_url(url).rstrip("/")
    req = urllib.request.Request(
        safe_url + "/models",
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
        payload = json.loads(_read_http_bounded(resp).decode("utf-8"))
    raw = payload.get("data") or []
    privacy = _fetch_privacy_metadata(safe_url, timeout=timeout)
    models = []
    for entry in raw:
        mid = entry.get("id")
        if mid:
            models.append(_normalize_model(entry, privacy.get(mid)))
    return models


def _is_chat_model(mid: str, inputs: list, outputs: list) -> bool:
    """True when the model speaks /chat/completions rather than a specialized endpoint."""
    if "text" not in outputs or "text" not in inputs:
        return False
    low = mid.lower()
    if any(k in low for k in NON_CHAT_KEYWORDS):
        return False
    return True


def _normalize_model(entry: dict, privacy: dict = None) -> dict:
    """Flatten one catalog entry into the handful of fields we display."""
    arch = entry.get("architecture") or {}
    inputs = arch.get("inputModalities") or entry.get("input_modalities") or []
    outputs = arch.get("outputModalities") or entry.get("output_modalities") or []
    pricing = entry.get("pricing") or {}
    features = entry.get("supported_features") or []
    mid = entry.get("id", "")
    priv = privacy or {}
    provider_type = priv.get("provider_type") or entry.get("provider_type") or ""
    attestable = bool(priv.get("attestation_supported") or entry.get("attestation_supported"))
    verifiable = bool(priv.get("verifiable") or entry.get("verifiable"))
    return {
        "id": mid,
        "family": mid.split("/")[0].lower() if "/" in mid else mid.lower(),
        "owned_by": entry.get("owned_by") or "",
        "context": entry.get("context_length") or 0,
        "max_output": entry.get("max_output_length") or 0,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "features": list(features),
        "is_ready": entry.get("is_ready"),
        "prompt_price": pricing.get("prompt"),
        "completion_price": pricing.get("completion"),
        "description": entry.get("description") or "",
        "is_chat": _is_chat_model(mid, inputs, outputs),
        "has_tools": "tools" in features,
        "provider_type": provider_type,
        "provider_trust": provider_trust_class({"provider_type": provider_type}),
        "attestable": attestable,
        "verifiable": verifiable,
    }


def load_catalog(cfg: dict, force_refresh: bool = False, offline: bool = False) -> dict:
    """Return {'models': [...], 'fetched_at': float, 'source': 'api'|'cache'|'stale'}."""
    cache = None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = None

    ttl = int(cfg.get("cache_ttl") or DEFAULT_TTL)
    has_privacy = bool(cache) and any(
        "provider_type" in m for m in (cache.get("models") or [])[:1]
    )
    fresh = bool(cache) and has_privacy and (time.time() - cache.get("fetched_at", 0)) < ttl

    if cache and fresh and not force_refresh:
        for m in cache["models"]:
            m["is_chat"] = _is_chat_model(m.get("id", ""), m.get("inputs") or [], m.get("outputs") or [])
        return {"models": cache["models"], "fetched_at": cache["fetched_at"], "source": "cache"}

    if offline:
        if cache:
            return {"models": cache["models"], "fetched_at": cache["fetched_at"], "source": "stale"}
        raise SystemExit("error: no cached catalog and --offline was requested")

    key = resolve_api_key(cfg)
    if not key:
        if cache:
            warn("no API key; using cached catalog")
            return {"models": cache["models"], "fetched_at": cache["fetched_at"], "source": "stale"}
        raise SystemExit("error: no API key. Run `./nearai_agy.py init` or set NEAR_AI_KEY.")

    try:
        models = _fetch_catalog(cfg.get("gateway_url") or DEFAULT_GATEWAY, key)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        safe_exc = redact_secrets(str(exc))
        if cache:
            warn("catalog refresh failed (%s); using cached copy" % safe_exc)
            return {"models": cache["models"], "fetched_at": cache["fetched_at"], "source": "stale"}
        raise SystemExit("error: could not reach the catalog: %s" % safe_exc)

    stamp = time.time()
    try:
        _atomic_write_text(
            CACHE_PATH,
            json.dumps({"fetched_at": stamp, "models": models}),
            mode=0o600,
            dir_mode=0o700,
        )
    except OSError:
        pass
    return {"models": models, "fetched_at": stamp, "source": "api"}


# ---------------------------------------------------------------------------
# request probe: does agy actually work with this model?
# ---------------------------------------------------------------------------


def load_probe_cache() -> dict:
    try:
        with open(PROBE_CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_probe_cache(data: dict) -> None:
    try:
        _atomic_write_text(
            PROBE_CACHE_PATH,
            json.dumps(data, indent=1, sort_keys=True),
            mode=0o600,
            dir_mode=0o700,
        )
    except OSError:
        pass


def validate_tool_response(data: dict) -> tuple[bool, str]:
    """Verify that an OpenAI chat completion response contains well-formed tool_calls."""
    if not isinstance(data, dict):
        return False, "response is not a JSON object"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "no choices in response"
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return False, "no message in choice"
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = msg.get("content") or ""
        reason = choices[0].get("finish_reason") or "unknown"
        return False, "no tool_calls emitted (finish_reason=%s, content=%r)" % (reason, content[:60])

    for call in tool_calls:
        if not isinstance(call, dict):
            return False, "tool_call is not an object"
        fn = call.get("function")
        if not isinstance(fn, dict):
            return False, "tool_call missing function object"
        name = fn.get("name")
        if name != "read_file":
            return False, "unexpected function called: %s" % name
        raw_args = fn.get("arguments")
        if not raw_args:
            return False, "empty arguments in tool_call"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception as exc:
            return False, "invalid JSON arguments: %s" % exc
        if not isinstance(args, dict) or "path" not in args:
            return False, "missing 'path' argument: %r" % args
    return True, ""


def probe_one(gateway: str, key: str, model_id: str, strip_effort: bool = False,
              validate_tools: bool = False) -> dict:
    """Send one request shaped exactly as agy shapes it for this model.

    When `strip_effort` is True (--what-if / --strip-effort), omit the
    `reasoning_effort` field for families that reject it (`anthropic/*`) to
    test whether the model works through the local stripper shim.

    When `validate_tools` is True, send a realistic function-calling schema and
    verify that the model correctly emits structured OpenAI tool_calls.

    Returns {"ok": bool, "code": int, "error": str, "verdict": bool,
             "tool_call_ok": bool, "tool_call_error": str}.
    HTTP 200 means agy will be able to use the model. A 400 is a verdict about
    the request shape and is treated as permanent; other 4xx/5xx/network errors
    are reported as "unknown" because they say nothing about the model.
    """
    if validate_tools:
        payload = {
            "model": model_id,
            "messages": PROBE_TOOL_MESSAGES,
            "tools": PROBE_TOOL_SCHEMA,
            "tool_choice": "auto",
            "max_completion_tokens": PROBE_TOOL_MAX_TOKENS,
            "stream": False,
        }
    else:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": PROBE_MAX_TOKENS,
            "stream": False,
        }
    effort = agy_effort_for(model_id)
    if effort and not (strip_effort and model_id.lower().startswith("anthropic/")):
        payload["reasoning_effort"] = effort
    body = json.dumps(payload).encode("utf-8")
    try:
        safe_gw = _validate_gateway_url(gateway).rstrip("/")
    except ValueError as exc:
        out = {"ok": False, "code": 0, "error": str(exc)[:200], "verdict": False}
        if validate_tools:
            out["tool_call_ok"] = False
            out["tool_call_error"] = str(exc)[:200]
        return out
    req = urllib.request.Request(
        safe_gw + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "Accept": "application/json"},
        method="POST",
    )
    timeout = PROBE_TOOL_TIMEOUT if validate_tools else PROBE_TIMEOUT
    try:
        with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
            raw = _read_http_bounded(resp)
            out = {"ok": True, "code": resp.status, "error": "", "verdict": True}
            if validate_tools:
                try:
                    data = json.loads(raw.decode("utf-8", "replace"))
                    tok, terr = validate_tool_response(data)
                    out["tool_call_ok"] = tok
                    out["tool_call_error"] = terr
                except Exception as exc:
                    out["tool_call_ok"] = False
                    out["tool_call_error"] = "json parse error: %s" % exc
            return out
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            resp_payload = json.loads(_read_http_bounded(exc, 64 * 1024).decode("utf-8", "replace"))
            err_obj = resp_payload.get("error")
            detail = (err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)) or ""
        except (ValueError, OSError):
            detail = ""
        detail = str(redact_secrets(detail))
        # A 400 is a verdict about the request shape, not a blip. 429 and 5xx
        # are transient, so they must not be cached as a permanent "no".
        verdict = exc.code == 400
        out = {"ok": False, "code": exc.code, "error": detail[:200], "verdict": verdict}
        if validate_tools:
            out["tool_call_ok"] = False
            out["tool_call_error"] = detail[:200]
        return out
    except (urllib.error.URLError, OSError, ValueError) as exc:
        safe_err = str(redact_secrets(str(exc)))[:200]
        out = {"ok": False, "code": 0, "error": safe_err, "verdict": False}
        if validate_tools:
            out["tool_call_ok"] = False
            out["tool_call_error"] = safe_err
        return out


def probe_verdict(entry) -> str:
    """Classify a probe result as 'ok', 'incompatible', or 'unknown'."""
    if entry is None:
        return "unknown"
    if entry.get("ok"):
        return "ok"
    if entry.get("verdict"):
        return "incompatible"
    return "unknown"


def probe_entry_ttl(entry: dict, base_ttl: int = PROBE_TTL_OK) -> int:
    """Return adaptive TTL based on verdict class.

    - 'ok' (usable): base_ttl (default: 24h)
    - 'incompatible' (fail): min(base_ttl, PROBE_TTL_FAIL) (2h) to discover upstream fixes quickly
    - 'unknown' (transient error / timeout): 0 (always re-probe)
    """
    if not isinstance(entry, dict):
        return 0
    verd = probe_verdict(entry)
    if verd == "ok":
        return base_ttl
    elif verd == "incompatible":
        return min(base_ttl, PROBE_TTL_FAIL)
    return 0


def run_probes(gateway: str, key: str, ids: list, cache: dict,
               force: bool = False, ttl: int = PROBE_TTL,
               progress=None, strip_effort: bool = False,
               retry_transient: bool = False,
               validate_tools: bool = False) -> dict:
    """Probe `ids` in parallel, reusing cached verdicts that are still fresh.

    Note: when `strip_effort` is True, results for `anthropic/*` are NOT written
    into the default probe cache, so the default probe cache always stays
    faithful to unmodified `agy`.
    """
    now = time.time()
    results = {}
    todo = []
    for mid in ids:
        is_stripped_target = strip_effort and mid.lower().startswith("anthropic/")
        hit = cache.get(mid)
        eff_ttl = probe_entry_ttl(hit, base_ttl=ttl)
        is_fresh = bool(eff_ttl > 0 and (now - float(hit.get("at") or 0) < eff_ttl))

        if is_stripped_target:
            todo.append(mid)
        elif retry_transient:
            # Re-probe only entries whose verdict is unknown/transient (or missing)
            if is_fresh and hit.get("verdict", False):
                results[mid] = hit
            else:
                todo.append(mid)
        elif validate_tools and (not isinstance(hit, dict) or "tool_call_ok" not in hit):
            # When tool validation is requested, if the cached result didn't test tools, re-probe
            todo.append(mid)
        elif not force and is_fresh:
            results[mid] = hit
        else:
            todo.append(mid)

    if not todo:
        return results

    done = 0
    lock = threading.Lock()

    def work(mid):
        nonlocal done
        outcome = probe_one(gateway, key, mid, strip_effort=strip_effort,
                            validate_tools=validate_tools)
        outcome["at"] = time.time()
        with lock:
            results[mid] = outcome
            if not (strip_effort and mid.lower().startswith("anthropic/")):
                cache[mid] = outcome
            done += 1
            if progress:
                progress(done, len(todo), mid)

    workers = max(1, min(PROBE_WORKERS, len(todo)))
    threads = []
    for mid in todo:
        th = threading.Thread(target=work, args=(mid,), daemon=True)
        th.start()
        threads.append(th)
        # Simple throttle: keep at most `workers` in flight.
        while sum(1 for t in threads if t.is_alive()) >= workers:
            time.sleep(0.02)
    for th in threads:
        th.join()
    return results


# Every failure we have observed falls into one of these buckets.  A dropped
# model is only useful information if it comes with *why* it was dropped, so
# both `probe` and `build` group their casualties by these keys.
REASONS = {
    "effort": ("agy sends reasoning_effort; the gateway rejects that field",
               "Fixable via local stripper shim (`build --strip-effort` or "
               "`probe --what-if`): agy injects the field for anthropic/* and "
               "the upstream validator rejects it with HTTP 400."),
    "route": ("it is served on a dedicated non-chat endpoint (404 at /chat/completions)",
              "Not a chat model: served on /v1/images/generations, "
              "/v1/audio/transcriptions, /v1/rerank, or /v1/privacy/classify."),
    "systemone": ("it lives on a different endpoint (/v1/systemone)",
                  "The gateway answers 'Decision models require /v1/systemone', "
                  "so this model is not reachable through the chat path agy uses."),
    "transient": ("upstream says it is temporarily unavailable or rate-limited",
                  "Not a verdict about the model; retry with `probe --retry-transient`. "
                  "Never cached as a permanent no."),
    "auth": ("the gateway rejected the API key",
             "Check `./nearai_agy.py doctor`."),
    "network": ("the gateway did not answer",
                "Connection problem or timeout, not a verdict about the model."),
    "other": ("rejected with an unclassified error", ""),
}


def probe_reason(entry) -> str:
    """Bucket a failing probe result into a key from REASONS.

    Matching is on the error text the gateway returned.  Order matters: the
    reasoning_effort check must come first because those responses also carry a
    400 status that would otherwise be claimed by the catch-all.
    """
    if not entry or entry.get("ok"):
        return ""
    err = (entry.get("error") or "").lower()
    code = entry.get("code") or 0
    if "reasoning_effort" in err:
        return "effort"
    if "systemone" in err:
        return "systemone"
    if "404 not found" in err or "upstream request failed with status 404" in err:
        return "route"
    if "currently unavailable" in err or "rate limit" in err or code in (429, 502, 503, 504):
        return "transient"
    if code in (401, 403):
        return "auth"
    if code == 0:
        return "network"
    return "other"


def reason_label(key: str) -> str:
    """Human-readable one-liner for a REASONS key."""
    return (REASONS.get(key) or REASONS["other"])[0]


def reason_note(key: str) -> str:
    """Longer explanation for a REASONS key (may be empty)."""
    return (REASONS.get(key) or REASONS["other"])[1]


def group_by_reason(ids, results) -> list:
    """Group failing model ids by reason, largest bucket first.

    Returns [(reason_key, [model_id, ...]), ...] so callers can report classes
    of failure instead of one flat alphabetical list of names.
    """
    buckets = {}
    for mid in ids:
        buckets.setdefault(probe_reason(results.get(mid)), []).append(mid)
    return sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def error_gist(entry, limit: int = 56) -> str:
    """The informative part of the gateway's error string.

    NEAR AI prefixes every message with "Provider failed for model '<id>': ",
    which is pure noise once the model id is already printed beside it.
    """
    err = (entry or {}).get("error") or ""
    marker = "': "
    if err.startswith("Provider failed for model ") and marker in err:
        err = err.split(marker, 1)[1]
    err = " ".join(err.split())
    return err if len(err) <= limit else err[: limit - 1] + "\u2026"


def report_drops(ids, results, limit: int = 6) -> None:
    """Print every dropped model with the reason it was dropped.

    This goes to stderr and is never suppressed: if the tool silently removes
    models the user asked for, they need to see which ones and why.  Long lists
    are collapsed per reason with a pointer to `probe` rather than flooding the
    terminal.
    """
    if not ids:
        return
    sys.stderr.write(yellow("  dropped ") + "%d model(s) agy cannot use:\n" % len(ids))
    for key, group in group_by_reason(ids, results):
        sys.stderr.write("    " + bold("%d x " % len(group)) + reason_label(key) + "\n")
        shown = group if len(group) <= limit else group[:limit]
        for mid in shown:
            sys.stderr.write("        %-40s %s\n"
                             % (mid, dim(error_gist(results.get(mid)))))
        if len(group) > len(shown):
            sys.stderr.write(dim("        ... and %d more (./nearai_agy.py probe)\n")
                             % (len(group) - len(shown)))
    sys.stderr.write(dim("    why: ./nearai_agy.py probe   "
                         "background: ArchitectureGuide.md\n"))


# ---------------------------------------------------------------------------
# filtering / selection
# ---------------------------------------------------------------------------


def provider_trust_class(model: dict) -> str:
    """Classify provider conservatively for incident-aware filtering."""
    provider = (model.get("provider_type") or "").strip().lower()
    if provider in {"vllm", "chutes", "tinfoil"}:
        return "tee"
    if provider in {"", "unknown", "legacy", "sglang"}:
        return "unknown"
    return "external"


def apply_filters(models: list, args) -> list:
    """Apply list/build filters in a fixed, predictable order."""
    out = []
    providers = [p.lower() for p in (getattr(args, "provider", None) or [])]
    families = [f.lower() for f in (getattr(args, "family", None) or [])]
    excludes = set(getattr(args, "exclude", None) or [])
    tee_only = getattr(args, "tee_only", False) or getattr(args, "private_only", False)
    tools_verified = getattr(args, "tools_verified", False)
    probe_cache = load_probe_cache() if tools_verified else None

    for m in models:
        if getattr(args, "chat_only", True) and not m["is_chat"]:
            continue
        if getattr(args, "tools_only", False) and not m["has_tools"]:
            continue
        if tools_verified and (not probe_cache or (probe_cache.get(m["id"]) or {}).get("tool_call_ok") is not True):
            continue
        if tee_only and (not m.get("attestable") or provider_trust_class(m) != "tee"):
            continue
        if getattr(args, "ready_only", False) and m["is_ready"] is not True:
            continue
        if providers and m["owned_by"].lower() not in providers and m["family"] not in providers:
            continue
        if families and m["family"] not in families:
            continue
        if m["id"] in excludes:
            continue
        out.append(m)
    return out



def _price_key(m: dict, field: str):
    """Sort key for a price field, ascending; unknown prices sort last."""
    value = m.get(field)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, 0.0)


def _price_desc(m: dict, field: str):
    """Sort key for a price field, descending; unknown prices still sort last."""
    unknown, value = _price_key(m, field)
    return (unknown, -value)


SORT_KEYS = {
    "catalog": None,                       # leave in catalog order
    "cheap": lambda m: (_price_key(m, "prompt_price"), _price_key(m, "completion_price")),
    "expensive": lambda m: (_price_desc(m, "completion_price"), _price_desc(m, "prompt_price")),
    "context": lambda m: -(m.get("context") or 0),
    "name": lambda m: m["id"].lower(),
    "price_prompt": lambda m: _price_key(m, "prompt_price"),
    "price_completion": lambda m: _price_key(m, "completion_price"),
}


def sort_models(models: list, how: str) -> list:
    """Sort a model list by a named key. 'catalog' is a no-op."""
    key = SORT_KEYS.get(how)
    if key is None or how == "catalog":
        return models
    return sorted(models, key=key)


def take_slice(models: list, first=None, last=None) -> list:
    if first is not None:
        return models[:first]
    if last is not None:
        return models[-last:] if last > 0 else models
    return models


# ---------------------------------------------------------------------------
# the 32-slot budget
# ---------------------------------------------------------------------------


_TIER_OUTPUT_ORDER = ("pro", "flash", "flash_lite")


def plan_slots(selected_ids: list, tiers: dict, cap: int) -> dict:
    """Replicate agy's slot accounting in O((N + T) * S) time so we can warn instead of silently losing models.

    agy keeps the first `cap` entries of: MODELS..., then tier models in
    TIER_APPEND_ORDER that are not already present.
    """
    models = []
    seen = set()
    for mid in selected_ids:
        if mid not in seen:
            seen.add(mid)
            models.append(mid)

    tiers_map = tiers or {}
    extras = []
    for tier in TIER_APPEND_ORDER:
        value = tiers_map.get(tier) or ""
        if value and value not in seen:
            seen.add(value)
            extras.append(value)

    total = len(models) + len(extras)
    if total <= cap:
        kept_models = models
        evicted_models = []
        evicted_tiers = []
        allowed = seen
    else:
        room = max(0, cap - len(extras))
        if len(models) > room:
            kept_models = models[:room]
            evicted_models = models[room:]
        else:
            kept_models = models
            evicted_models = []
        keep_tiers = max(0, cap - len(kept_models))
        kept_tiers = extras[:keep_tiers]
        evicted_tiers = extras[keep_tiers:]
        allowed = set(kept_models)
        allowed.update(kept_tiers)

    final_tiers = {}
    for tier in _TIER_OUTPUT_ORDER:
        value = tiers_map.get(tier) or ""
        final_tiers[tier] = value if (value and value in allowed) else ""

    return {
        "models": kept_models,
        "tiers": final_tiers,
        "evicted_models": evicted_models,
        "evicted_tiers": evicted_tiers,
        "total": total,
        "cap": cap,
        "fits": total <= cap,
    }


def report_evictions(plan: dict) -> None:
    if plan["fits"]:
        return
    warn("%d models requested but agy only keeps %d; these will NOT be registered:"
         % (plan["total"], plan["cap"]))
    for mid in plan["evicted_models"]:
        sys.stderr.write("           - %s\n" % mid)
    for tier in plan["evicted_tiers"]:
        sys.stderr.write("           - %s (tier)\n" % tier)
    sys.stderr.write("           reduce the count or add --strict to fail instead\n")


# ---------------------------------------------------------------------------
# numbering: catalog order is the truncation order, so show it
# ---------------------------------------------------------------------------


def index_models(models: list) -> dict:
    return {m["id"]: i + 1 for i, m in enumerate(models)}


def fmt_ctx(value: int) -> str:
    if not value:
        return "-"
    if value >= 1000000:
        return "%.1fM" % (value / 1000000.0)
    if value >= 1000:
        return "%dK" % round(value / 1000.0)
    return str(value)


def fmt_price(m: dict) -> str:
    prompt, completion = m.get("prompt_price"), m.get("completion_price")
    try:
        if prompt is None or completion is None:
            return "-"
        return "%g>%g" % (float(prompt) * 1e6, float(completion) * 1e6)
    except (TypeError, ValueError):
        return "-"


def fmt_tee(m: dict) -> str:
    """Return a conservative provider trust display label."""
    trust = m.get("provider_trust") or provider_trust_class(m)
    pt = m.get("provider_type") or ""
    if trust == "tee":
        return pt
    if trust == "external":
        return "ext"
    return "?"


def print_table(models: list, index: dict, show_all_note: bool = True,
                probes: dict = None) -> None:
    if not models:
        print("(no models matched)")
        return
    widest = max(len(m["id"]) for m in models)
    name_w = min(max(widest, 24), 52)
    if probes is not None:
        print(bold("%4s  %-*s  %6s  %5s  %-7s  %6s  %-11s  %s"
                   % ("#", name_w, "model", "ctx", "tools", "tee", "agy", "$/Mtok", "flags")))
        print(dim("-" * (name_w + 60)))
    else:
        print(bold("%4s  %-*s  %6s  %5s  %-7s  %5s  %-11s  %s"
                   % ("#", name_w, "model", "ctx", "tools", "tee", "ready", "$/Mtok", "flags")))
        print(dim("-" * (name_w + 53)))
    for m in models:
        flags = []
        if not m["is_chat"]:
            flags.append("non-chat")
        if m["outputs"] and "image" in m["outputs"]:
            flags.append("image-out")
        if not m["has_tools"]:
            flags.append("no-tools")
        name = m["id"]
        if len(name) > name_w:
            name = name[: name_w - 1] + "~"
        tee_raw = fmt_tee(m)
        tee_pad = "%-7s" % tee_raw
        tee_txt = cyan(tee_pad) if m.get("attestable") else dim(tee_pad)

        if probes is not None:
            p_res = probes.get(m["id"]) or {}
            state = probe_verdict(p_res)
            agy_txt = {"ok": green("yes"), "incompatible": red("FAIL"),
                       "unknown": dim("?")}[state]
            if state == "incompatible":
                flags.append("agy-incompatible")
            elif p_res.get("tool_call_ok") is True:
                flags.append("tool-verified")
            elif p_res.get("tool_call_ok") is False and m["has_tools"]:
                flags.append("tool-failed")

            tools_disp = "yes" if m["has_tools"] else "-"
            if p_res.get("tool_call_ok") is True:
                tools_disp = green("pass")
            elif p_res.get("tool_call_ok") is False and m["has_tools"]:
                tools_disp = yellow("fail")

            line = "%4d  %-*s  %6s  %5s  %s  %6s  %-11s  %s" % (
                index.get(m["id"], 0), name_w, name, fmt_ctx(m["context"]),
                tools_disp, tee_txt, agy_txt, fmt_price(m),
                ", ".join(flags),
            )
            if state == "incompatible":
                line = dim(line)
        else:
            ready = m["is_ready"]
            ready_txt = "-" if ready is None else ("yes" if ready else "no")
            line = "%4d  %-*s  %6s  %5s  %s  %5s  %-11s  %s" % (
                index.get(m["id"], 0), name_w, name, fmt_ctx(m["context"]),
                "yes" if m["has_tools"] else "-", tee_txt, ready_txt, fmt_price(m),
                ", ".join(flags),
            )
            if m["is_ready"] is False:
                line = dim(line)
        print(line)
    if show_all_note:
        print(dim("\n# = catalog position. agy keeps the first %d of the order you pass it."
                  % CAP_FALLBACK))
        print(dim("tee = hardware attestation tier (vllm/chutes/tinfoil = TEE; ext = external non-TEE)."))
        if probes is not None:
            print(dim("agy = whether a real request in agy's own shape succeeds "
                      "(not the catalog's is_ready flag)."))


# ---------------------------------------------------------------------------
# customModels sync for ~/.gemini/antigravity-cli/settings.json & jetski_state.pbtxt
# ---------------------------------------------------------------------------


def _safe_backup(path) -> None:
    """Create a .bak copy of path if the target file exists."""
    try:
        p = Path(path)
        if p.is_file():
            bak = p.with_name(p.name + ".bak")
            shutil.copy2(p, bak)
    except Exception:
        pass


_PBTXT_TOK_RE = re.compile(
    r'(?P<str>"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'
    r"|(?P<comment>#[^\n]*)"
    r"|(?P<lbrace>\{)"
    r"|(?P<rbrace>\})"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<colon>:)"
    r"|(?P<num>-?\d+)"
)
_PB_UNESC_RE = re.compile(r"\\(.)")
_PB_UNESC_MAP = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "'": "'", "\\": "\\"}


def _unescape_pb(raw_quoted: str) -> str:
    return _PB_UNESC_RE.sub(lambda m: _PB_UNESC_MAP.get(m.group(1), m.group(1)), raw_quoted[1:-1])


def _escape_pb(val) -> str:
    return (
        str(val)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _pos_int(val, default: int) -> int:
    try:
        iv = int(val)
        return iv if iv > 0 else default
    except (TypeError, ValueError):
        return default


def _skip_brace_block(tokens: list, open_pos: int) -> int:
    """Given tokens[open_pos].lastgroup == 'lbrace', return index of matching 'rbrace', or -1."""
    depth = 1
    k = open_pos + 1
    n = len(tokens)
    while k < n and depth > 0:
        tk = tokens[k].lastgroup
        if tk == "lbrace":
            depth += 1
        elif tk == "rbrace":
            depth -= 1
        k += 1
    return (k - 1) if depth == 0 else -1


def _parse_entry_tokens(tokens: list, start: int, end: int, models_map: dict) -> None:
    key = None
    val_span = None
    i = start
    depth = 0
    while i < end:
        kind = tokens[i].lastgroup
        if kind == "lbrace":
            depth += 1
        elif kind == "rbrace":
            depth = max(0, depth - 1)
        elif depth == 0 and kind == "ident":
            field = tokens[i].group()
            j = i + 1
            if j < end and tokens[j].lastgroup == "colon":
                j += 1
            if field == "key" and j < end and tokens[j].lastgroup == "str":
                key = _unescape_pb(tokens[j].group())
                i = j + 1
                continue
            elif field == "value" and j < end and tokens[j].lastgroup == "lbrace":
                close_pos = _skip_brace_block(tokens, j)
                if close_pos != -1 and close_pos < end:
                    val_span = (j + 1, close_pos)
                    i = close_pos + 1
                    continue
        i += 1

    if not key:
        return

    model_name = key
    max_tokens = 131072
    max_output_tokens = 8192
    if val_span is not None:
        vi, vend = val_span
        vdepth = 0
        while vi < vend:
            kind = tokens[vi].lastgroup
            if kind == "lbrace":
                vdepth += 1
            elif kind == "rbrace":
                vdepth = max(0, vdepth - 1)
            elif vdepth == 0 and kind == "ident":
                field = tokens[vi].group()
                j = vi + 1
                if j < vend and tokens[j].lastgroup == "colon":
                    j += 1
                if field == "model_name" and j < vend and tokens[j].lastgroup == "str":
                    model_name = _unescape_pb(tokens[j].group())
                    vi = j + 1
                    continue
                elif field == "max_tokens" and j < vend and tokens[j].lastgroup == "num":
                    max_tokens = int(tokens[j].group())
                    vi = j + 1
                    continue
                elif field == "max_output_tokens" and j < vend and tokens[j].lastgroup == "num":
                    max_output_tokens = int(tokens[j].group())
                    vi = j + 1
                    continue
            vi += 1

    models_map[key] = {
        "model_name": model_name,
        "max_tokens": max_tokens,
        "max_output_tokens": max_output_tokens,
    }


def _parse_custom_models_tokens(tokens: list, start: int, end: int, models_map: dict) -> None:
    """Parse all top-level `custom_models [:]{ ... }` entries in tokens[start:end]."""
    i = start
    depth = 0
    while i < end:
        kind = tokens[i].lastgroup
        if kind == "lbrace":
            depth += 1
        elif kind == "rbrace":
            depth = max(0, depth - 1)
        elif depth == 0 and kind == "ident" and tokens[i].group() == "custom_models":
            j = i + 1
            if j < end and tokens[j].lastgroup == "colon":
                j += 1
            if j < end and tokens[j].lastgroup == "lbrace":
                close_pos = _skip_brace_block(tokens, j)
                if close_pos != -1 and close_pos < end:
                    _parse_entry_tokens(tokens, j + 1, close_pos, models_map)
                    i = close_pos + 1
                    continue
        i += 1


def sync_jetski_state(catalog, model_ids: list, state_path: Path = None) -> int:
    """Sync model definitions into ~/.gemini/antigravity-cli/jetski_state.pbtxt.

    Uses a single-pass O(L) quote/comment-aware lexical scanner, protobuf string
    escaping, and atomic fsync'd writes to prevent corruption of surrounding fields.

    Returns the number of custom models configured in jetski_state.pbtxt.
    """
    target = Path(state_path) if state_path else AGY_STATE_PATH

    meta_map = {}
    if isinstance(catalog, dict):
        if "models" in catalog:
            for m in catalog["models"]:
                if isinstance(m, dict) and "id" in m:
                    meta_map[m["id"]] = m
        else:
            meta_map = catalog
    elif isinstance(catalog, (list, tuple)):
        for m in catalog:
            if isinstance(m, dict) and "id" in m:
                meta_map[m["id"]] = m

    # Only a missing file defaults to empty; other read errors propagate so we
    # never silently overwrite unrelated protobuf fields after a transient I/O error.
    content = ""
    try:
        with open(target, "r", encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        content = ""

    models_map = {}
    remaining = content
    if content:
        tokens = list(_PBTXT_TOK_RE.finditer(content))
        spans = []
        depth = 0
        i = 0
        n = len(tokens)
        while i < n:
            kind = tokens[i].lastgroup
            if kind == "lbrace":
                depth += 1
            elif kind == "rbrace":
                depth = max(0, depth - 1)
            elif depth == 0 and kind == "ident" and tokens[i].group() == "custom_models_config":
                start_char = tokens[i].start()
                j = i + 1
                if j < n and tokens[j].lastgroup == "colon":
                    j += 1
                if j < n and tokens[j].lastgroup == "lbrace":
                    close_pos = _skip_brace_block(tokens, j)
                    if close_pos != -1:
                        _parse_custom_models_tokens(tokens, j + 1, close_pos, models_map)
                        spans.append((start_char, tokens[close_pos].end()))
                        i = close_pos + 1
                        continue
            i += 1

        if spans:
            chunks = []
            prev = 0
            for s, e in spans:
                chunks.append(content[prev:s])
                prev = e
            chunks.append(content[prev:])
            remaining = "".join(chunks).strip()

    for mid in model_ids:
        if not mid:
            continue
        m = meta_map.get(mid) or {}
        ctx = _pos_int(m.get("context") or m.get("context_length"), 131072)
        out_tok = _pos_int(m.get("max_output") or m.get("max_output_length"), 8192)
        models_map[mid] = {
            "model_name": mid,
            "max_tokens": ctx,
            "max_output_tokens": out_tok,
        }

    lines = ["custom_models_config: {"]
    for mid in sorted(models_map):
        mdata = models_map[mid]
        lines.append("  custom_models: {")
        lines.append('    key: "%s"' % _escape_pb(mid))
        lines.append("    value: {")
        lines.append('      model_name: "%s"' % _escape_pb(mdata["model_name"]))
        lines.append("      max_tokens: %d" % mdata["max_tokens"])
        lines.append("      max_output_tokens: %d" % mdata["max_output_tokens"])
        lines.append("    }")
        lines.append("  }")
    lines.append("}")
    new_block = "\n".join(lines)

    final_content = (remaining + "\n\n" if remaining else "") + new_block + "\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_backup(target)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(final_content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return len(models_map)


def sync_custom_models(catalog, model_ids: list, settings_path: Path = None, state_path: Path = None) -> int:
    """Sync model definitions into ~/.gemini/antigravity-cli/settings.json and jetski_state.pbtxt.

    Ensures:
      1. `settings["customModels"]` is a map (dict) where each model key has
         `modelName`, `label`, `maxTokens`, and `maxOutputTokens`.
      2. `jetski_state.pbtxt` has corresponding `custom_models_config` protobuf entries.

    Without `maxTokens` in `customModels`, agy defaults to a zero context budget
    for custom gateway models, causing context compaction to fail immediately with:
        "no messages fit within the context token budget"

    Existing fields in settings.json (like `model`, `toolPermission`, `trustedWorkspaces`)
    are strictly preserved.

    Returns the number of custom models configured in settings.json.
    """
    target = Path(settings_path) if settings_path else AGY_SETTINGS_PATH

    meta_map = {}
    if isinstance(catalog, dict):
        if "models" in catalog:
            for m in catalog["models"]:
                if isinstance(m, dict) and "id" in m:
                    meta_map[m["id"]] = m
        else:
            meta_map = catalog
    elif isinstance(catalog, (list, tuple)):
        for m in catalog:
            if isinstance(m, dict) and "id" in m:
                meta_map[m["id"]] = m

    settings = {}
    if target.is_file():
        try:
            with open(target, "r", encoding="utf-8") as fh:
                settings = json.load(fh)
        except Exception:
            settings = {}
    if not isinstance(settings, dict):
        settings = {}

    # `customModels` in settings.json must be a dictionary mapping model IDs to config dicts.
    # If it was erroneously set to a boolean (e.g. True), reset to dict.
    cms = settings.get("customModels")
    if not isinstance(cms, dict):
        cms = settings["customModels"] = {}

    # Migrate any entries from obsolete customModelsConfig if present
    if "customModelsConfig" in settings:
        old_cfg = settings.pop("customModelsConfig", {})
        if isinstance(old_cfg, dict):
            for k, v in old_cfg.items():
                if k not in cms and isinstance(v, dict):
                    cms[k] = v

    count = 0
    for mid in model_ids:
        if not mid:
            continue
        m = meta_map.get(mid) or {}
        ctx = m.get("context") or m.get("context_length") or 0
        if not ctx or ctx <= 0:
            ctx = 131072
        out_tok = m.get("max_output") or m.get("max_output_length") or 0
        if not out_tok or out_tok <= 0:
            out_tok = 8192

        # Model entry matching agy's customModels JSON configuration schema
        cms[mid] = {
            "modelName": mid,
            "label": m.get("id") or mid,
            "maxTokens": int(ctx),
            "maxOutputTokens": int(out_tok),
        }
        count += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_backup(target)
    _atomic_write_text(target, json.dumps(settings, indent=2) + "\n")

    state_target = Path(state_path) if state_path else (target.parent / "jetski_state.pbtxt" if settings_path else AGY_STATE_PATH)
    try:
        sync_jetski_state(catalog, model_ids, state_path=state_target)
    except Exception:
        pass

    return count


def validate_custom_models_configured(model_ids: list, settings_path: Path = None, state_path: Path = None) -> list:
    """Validate that model_ids are configured with positive token limits in settings.json.

    Returns a list of missing model IDs. Returns an empty list if all models
    are properly configured.
    """
    if not model_ids:
        return []

    if settings_path:
        target_settings = Path(settings_path)
        target_state = Path(state_path) if state_path else target_settings.parent / "jetski_state.pbtxt"
    else:
        target_settings = AGY_SETTINGS_PATH
        target_state = Path(state_path) if state_path else AGY_STATE_PATH

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for m in model_ids:
        if m and m not in seen:
            seen.add(m)
            deduped.append(m)

    if not target_settings.is_file():
        return list(deduped)

    try:
        with open(target_settings, "r", encoding="utf-8") as fh:
            settings_data = json.load(fh)
    except Exception:
        return list(deduped)

    if not isinstance(settings_data, dict):
        return list(deduped)

    cms = settings_data.get("customModels")
    if not isinstance(cms, dict):
        return list(deduped)

    valid_settings = set()
    for mid, cfg in cms.items():
        if isinstance(cfg, dict) and (cfg.get("maxTokens") or 0) > 0:
            valid_settings.add(mid)

    # Check jetski_state.pbtxt only if it exists
    valid_state = None
    if target_state.is_file():
        try:
            with open(target_state, "r", encoding="utf-8") as fh:
                state_text = fh.read()
            valid_state = set(re.findall(r'custom_models:\s*\{\s*key:\s*"([^"]+)"', state_text))
        except Exception:
            valid_state = None

    missing = []
    for mid in deduped:
        if mid not in valid_settings:
            missing.append(mid)
        elif valid_state is not None and mid not in valid_state:
            missing.append(mid)

    return missing


# ---------------------------------------------------------------------------
# gateway proxy (loopback parameter normalizer)
# ---------------------------------------------------------------------------


MAX_PROXY_REQUEST_BODY = 10 * 1024 * 1024   # 10 MiB
MAX_PROXY_UPSTREAM_BODY = 32 * 1024 * 1024  # 32 MiB
MAX_PROXY_SSE_BYTES = 64 * 1024 * 1024      # 64 MiB
MAX_PROXY_SSE_SECONDS = 600
PROXY_UPSTREAM_TIMEOUT = 120


class _UpstreamBodyTooLarge(Exception):
    pass


def _sanitize_header_value(value, default: str, max_len: int = 512) -> str:
    if not value or not isinstance(value, str):
        return default
    v = value.strip()
    if not v or len(v) > max_len or any(ord(c) < 0x20 or ord(c) == 0x7F for c in v):
        return default
    return v


def _read_bounded_stream(stream, max_bytes: int, deadline: float) -> bytes:
    chunks = []
    total = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("upstream response exceeded total deadline")
        chunk = stream.read(min(65536, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _UpstreamBodyTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_json_constant(val: str):
    raise ValueError("Invalid JSON constant: %s" % val)


def _strict_object_pairs_hook(pairs: list) -> dict:
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError("Duplicate JSON key: %s" % k)
        obj[k] = v
    return obj


def _strip_reasoning_effort(obj, depth: int = 0, max_depth: int = 16):
    if depth > max_depth:
        raise ValueError("JSON nesting depth exceeded")
    if isinstance(obj, dict):
        changed = False
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == "reasoning_effort":
                changed = True
                continue
            new_v, sub_changed = _strip_reasoning_effort(v, depth + 1, max_depth)
            out[k] = new_v
            changed = changed or sub_changed
        return out, changed
    if isinstance(obj, list):
        changed = False
        out = []
        for item in obj:
            new_item, sub_changed = _strip_reasoning_effort(item, depth + 1, max_depth)
            out.append(new_item)
            changed = changed or sub_changed
        return out, changed
    return obj, False


class _BoundedThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer bounded by a semaphore to prevent thread/FD exhaustion."""

    max_active_requests = 64
    request_queue_size = 64
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sem = threading.BoundedSemaphore(self.max_active_requests)

    def process_request(self, request, client_address):
        if not self._sem.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                self.close_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._sem.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._sem.release()


class GatewayProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, *a):
        # Keep agy terminal clean
        pass

    def send_response(self, code, message=None):
        self._response_started = True
        super().send_response(code, message)

    def _authorized(self) -> bool:
        vals = self.headers.get_all("Authorization") or []
        if len(vals) != 1:
            return False
        expected = "Bearer " + self.server.session_token
        return hmac.compare_digest(vals[0].encode("utf-8"), expected.encode("utf-8"))

    def _reject(self, code: int, message: str, *, close: bool = True):
        msg = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(msg)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(msg)

    def _reject_unexpected_body(self) -> bool:
        if self.headers.get_all("Transfer-Encoding") or (self.headers.get("Content-Length") not in (None, "0")):
            self._reject(400, "Request body not allowed for this method")
            return True
        return False

    def _read_request_body(self):
        if self.headers.get_all("Transfer-Encoding"):
            self._reject(400, "Unsupported Transfer-Encoding")
            return None
        cl_values = self.headers.get_all("Content-Length") or []
        if len(cl_values) == 0:
            self._reject(411, "Content-Length required")
            return None
        if len(cl_values) > 1 or not cl_values[0].strip().isdigit():
            self._reject(400, "Invalid Content-Length")
            return None
        n = int(cl_values[0].strip())
        if n > MAX_PROXY_REQUEST_BODY:
            self._reject(413, "Request payload too large")
            return None
        raw = self.rfile.read(n)
        if len(raw) != n:
            self._reject(400, "Incomplete request body")
            return None
        return raw

    def _normalize_post_body(self, raw: bytes):
        if not raw:
            return raw, None
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_object_pairs_hook,
            )
        except (UnicodeDecodeError, ValueError):
            return None, "Invalid JSON request body"
        if not isinstance(payload, dict):
            return None, "JSON request body must be an object"
        model = payload.get("model")
        if isinstance(model, str) and model.strip().lower().startswith("anthropic/"):
            try:
                cleaned, changed = _strip_reasoning_effort(payload)
            except ValueError as exc:
                return None, str(exc)
            if changed:
                return json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), None
        return raw, None

    def _build_upstream_url(self):
        raw_target = self.path or ""
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in raw_target):
            return None
        parts = urllib.parse.urlsplit(raw_target)
        if parts.scheme or parts.netloc or parts.fragment or not parts.path.startswith("/"):
            return None

        req_path = parts.path
        if req_path == "/v1":
            req_path = "/"
        elif req_path.startswith("/v1/"):
            req_path = req_path[3:]

        decoded = urllib.parse.unquote(req_path)
        if "\\" in req_path or "\\" in decoded or any(ord(c) < 0x20 or ord(c) == 0x7F for c in decoded):
            return None
        if any(seg in (".", "..") for seg in decoded.split("/")):
            return None

        base = self.server.upstream_parts
        base_prefix = base.path.rstrip("/")
        trailing = "/" if req_path.endswith("/") and req_path != "/" else ""
        joined = posixpath.normpath(base_prefix + "/" + req_path.lstrip("/")) + trailing
        if base_prefix and not (joined == base_prefix or joined.startswith(base_prefix + "/")):
            return None
        if joined.startswith("//"):
            return None

        query = parts.query or base.query
        return urllib.parse.urlunsplit((base.scheme, base.netloc, joined, query, ""))

    def _forward(self, method: str, body_bytes: bytes = None):
        self._response_started = False
        host_vals = self.headers.get_all("Host") or []
        if len(host_vals) != 1 or not host_vals[0].strip():
            return self._reject(400, "Missing or duplicate Host header")

        url = self._build_upstream_url()
        if not url:
            return self._reject(400, "Invalid request path")

        headers = {
            "Authorization": "Bearer " + self.server.upstream_key,
            "Accept": _sanitize_header_value(self.headers.get("Accept"), "*/*"),
            "Accept-Encoding": "identity",
        }
        if body_bytes is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        try:
            with _NO_REDIRECT_OPENER.open(req, timeout=PROXY_UPSTREAM_TIMEOUT) as resp:
                ct = _sanitize_header_value(resp.headers.get("Content-Type"), "application/json")
                if method == "HEAD":
                    up_cl = resp.headers.get("Content-Length") or "0"
                    cl_out = up_cl.strip() if up_cl.strip().isdigit() else "0"
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ct)
                    self.send_header("Content-Length", cl_out)
                    self.end_headers()
                    return
                if "text/event-stream" in ct.lower():
                    self.close_connection = True
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ct)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    deadline = time.monotonic() + MAX_PROXY_SSE_SECONDS
                    streamed = 0
                    while time.monotonic() < deadline:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        streamed += len(chunk)
                        if streamed > MAX_PROXY_SSE_BYTES:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = _read_bounded_stream(
                        resp, MAX_PROXY_UPSTREAM_BODY, time.monotonic() + PROXY_UPSTREAM_TIMEOUT
                    )
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ct)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            try:
                err_data = _read_bounded_stream(exc, MAX_PROXY_UPSTREAM_BODY, time.monotonic() + 30)
            except _UpstreamBodyTooLarge:
                return self._reject(502, "Upstream error response too large")
            raw_ct = exc.headers.get("Content-Type") if exc.headers else None
            ct = _sanitize_header_value(raw_ct, "application/json")
            if not ct.lower().startswith(("application/json", "text/plain")):
                ct = "text/plain; charset=utf-8"
            self.send_response(exc.code)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(err_data)))
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(err_data)
        except _UpstreamBodyTooLarge:
            if not getattr(self, "_response_started", False):
                self._reject(502, "Upstream response too large")
            else:
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            if not getattr(self, "_response_started", False):
                self._reject(502, "Bad Gateway: upstream request failed")
            else:
                self.close_connection = True

    def do_GET(self):
        if not self._authorized():
            return self._reject(403, "Forbidden: invalid loopback session token")
        if self._reject_unexpected_body():
            return
        self._forward("GET")

    def do_HEAD(self):
        if not self._authorized():
            return self._reject(403, "Forbidden: invalid loopback session token")
        if self._reject_unexpected_body():
            return
        self._forward("HEAD")

    def do_POST(self):
        if not self._authorized():
            return self._reject(403, "Forbidden: invalid loopback session token")
        raw = self._read_request_body()
        if raw is None:
            return
        normalized, err = self._normalize_post_body(raw)
        if err:
            return self._reject(400, err)
        self._forward("POST", normalized)


class GatewayProxy:
    """Ephemeral stdlib bounded ThreadingHTTPServer loopback proxy normalizing parameters for agy."""

    def __init__(self, upstream_url: str, upstream_key: str, host: str = "127.0.0.1", port: int = 0):
        self.upstream_url = _validate_gateway_url(upstream_url).rstrip("/")
        self.upstream_parts = urllib.parse.urlsplit(self.upstream_url)
        if not _is_loopback_host(host):
            raise ValueError("GatewayProxy host must be loopback, got: %r" % host)
        self.upstream_key = upstream_key
        self.host = host
        self.requested_port = port
        self.session_token = secrets.token_urlsafe(32)
        self.server = None
        self.thread = None
        self.port = None

    def start(self):
        self.server = _BoundedThreadingHTTPServer((self.host, self.requested_port), GatewayProxyHandler)
        self.server.upstream_url = self.upstream_url
        self.server.upstream_parts = self.upstream_parts
        self.server.upstream_key = self.upstream_key
        self.server.session_token = self.session_token
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def shutdown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def gateway_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/v1"


# ---------------------------------------------------------------------------
# launcher rendering
# ---------------------------------------------------------------------------

LAUNCH_TARGETS = ("cli", "ide", "app")

TARGET_DISPLAY_NAMES = {
    "cli": "Antigravity CLI (agy)",
    "ide": "Antigravity IDE",
    "app": "Antigravity Desktop App",
}


def launcher_filename(name: str, target: str = "cli") -> str:
    """Return the bin/ launcher filename for a profile name and target."""
    safe_name = _validate_safe_name(name, "launcher name")
    if target == "cli":
        return "nearai-" + safe_name
    if target in ("ide", "app"):
        return "nearai-%s-%s" % (target, safe_name)
    raise ValueError("unsupported target: %r" % target)


def resolve_target_executable(target: str) -> tuple[Optional[str], bool]:
    """Resolve the launch executable for target ('cli', 'ide', or 'app').

    Returns (executable_path_or_None, is_cli_wrapper).
    For macOS .app bundles, prefers the direct Contents/MacOS/<binary> so the
    launcher stays attached for clean GatewayProxy teardown on app exit.
    """
    if target == "cli":
        return shutil.which("agy"), False

    if target == "ide":
        override = os.environ.get("AGY_IDE_BIN", "").strip()
        if override:
            expanded = os.path.expanduser(override)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded, False
        app_override = os.environ.get("AGY_IDE_APP", "").strip()
        candidates = []
        if app_override:
            candidates.append(Path(app_override).expanduser() / "Contents" / "MacOS" / "Electron")
        candidates.extend([
            Path("/Applications/Antigravity IDE.app/Contents/MacOS/Electron"),
            Path.home() / "Applications" / "Antigravity IDE.app" / "Contents" / "MacOS" / "Electron",
        ])
        for cand in candidates:
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand), False
        cli_bin = shutil.which("antigravity-ide")
        if cli_bin:
            return cli_bin, True
        return None, False

    if target == "app":
        override = os.environ.get("AGY_DESKTOP_BIN", "").strip()
        if override:
            expanded = os.path.expanduser(override)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded, False
        app_override = os.environ.get("AGY_DESKTOP_APP", "").strip()
        candidates = []
        if app_override:
            candidates.append(Path(app_override).expanduser() / "Contents" / "MacOS" / "Antigravity")
        candidates.extend([
            Path("/Applications/Antigravity.app/Contents/MacOS/Antigravity"),
            Path.home() / "Applications" / "Antigravity.app" / "Contents" / "MacOS" / "Antigravity",
        ])
        for cand in candidates:
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand), False
        cli_bin = shutil.which("antigravity")
        if cli_bin:
            return cli_bin, True
        return None, False

    raise ValueError("unsupported target: %r" % target)


def _read_macos_app_version(executable_path: str) -> str:
    """Read CFBundleShortVersionString from a macOS .app Info.plist if present."""
    try:
        p = Path(executable_path)
        if p.parent.name == "MacOS" and p.parent.parent.name == "Contents":
            plist = p.parent.parent / "Info.plist"
            if plist.is_file():
                import plistlib
                with open(plist, "rb") as fh:
                    info_dict = plistlib.load(fh)
                ver = info_dict.get("CFBundleShortVersionString") or info_dict.get("CFBundleVersion")
                if ver:
                    return str(ver)
    except Exception:
        pass
    return ""


def is_gui_target_running(target: str, executable_path: Optional[str] = None) -> bool:
    """Return True if a non-isolated GUI instance of target ('ide'|'app') is running."""
    if target not in ("ide", "app"):
        return False
    exe = executable_path or resolve_target_executable(target)[0]
    if not exe:
        return False
    try:
        proc = subprocess.run(
            ["ps", "-eo", "args"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return False
        for line in proc.stdout.splitlines():
            s = line.strip()
            if not s:
                continue
            if exe in s and "--type=" not in s and "--user-data-dir" not in s:
                return True
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def isolated_user_data_dir(target: str, profile_name: str) -> Path:
    """Return a clean isolated --user-data-dir path for a GUI target + profile."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(profile_name).stem or "default").strip("-") or "default"
    return CONFIG_DIR / ("%s-profiles" % target) / safe


def build_target_command(
    target: str,
    executable: str,
    is_cli_wrapper: bool,
    passthrough_args: list[str],
    default_model: str = "",
    isolated: bool = False,
    profile_name: str = "default",
) -> list[str]:
    """Build the command argv list for 'cli', 'ide', or 'app'."""
    args_list = list(passthrough_args)
    if target == "cli":
        subcommands = {
            "models", "help", "version", "changelog", "install", "mcp",
            "plugin", "plugins", "agent", "agents", "acp", "mic-serve",
            "update", "remote-control",
        }
        inject = True
        if args_list:
            if args_list[0] in subcommands:
                inject = False
            for a in args_list:
                if a == "--model" or a.startswith("--model="):
                    inject = False
                    break
        cmd = [executable]
        if inject and default_model:
            cmd.extend(["--model", default_model])
        cmd.extend(args_list)
        return cmd

    cmd = [executable]
    has_user_data_dir = any(
        a == "--user-data-dir" or a.startswith("--user-data-dir=") for a in args_list
    )
    if isolated and not has_user_data_dir:
        CONFIG_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            os.chmod(CONFIG_DIR, 0o700)
        except OSError:
            pass
        udd = isolated_user_data_dir(target, profile_name)
        udd.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(udd, 0o700)
        except OSError:
            pass
        cmd.append("--user-data-dir=%s" % udd)
        if target == "ide" and "--new-window" not in args_list and "-n" not in args_list:
            cmd.append("--new-window")
    if is_cli_wrapper and target == "ide" and "--wait" not in args_list and "-w" not in args_list:
        cmd.append("--wait")
    cmd.extend(args_list)
    return cmd


FORWARDER_TEMPLATE = r'''#!/usr/bin/env bash
# @@NAME@@ -- forwarder generated by nearai_agy.py @@VERSION@@ on @@DATE@@
#
# Profile : @@PROFILE@@
# Target  : @@TARGET@@
# Rebuild : ./nearai_agy.py build @@REBUILD_ARGS@@
#
_MODELS_CSV="@@MODELS@@"

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_DIR/../nearai_agy.py" ]]; then
  exec python3 "$_DIR/../nearai_agy.py" run @@RUN_ARGS@@ "$@"
elif [[ -f "$_DIR/nearai_agy.py" ]]; then
  exec python3 "$_DIR/nearai_agy.py" run @@RUN_ARGS@@ "$@"
elif command -v nearai-agy >/dev/null 2>&1; then
  exec nearai-agy run @@RUN_ARGS@@ "$@"
elif command -v nearai_agy.py >/dev/null 2>&1; then
  exec nearai_agy.py run @@RUN_ARGS@@ "$@"
else
  printf '%s\n' "@@NAME@@: cannot find nearai_agy.py" >&2
  exit 1
fi
'''

BASH_TEMPLATE = FORWARDER_TEMPLATE


def render_launcher(name: str, profile_ref: str, url: str, ttl: int, cap: int,
                    plan: dict, default_model: str, strip_effort: bool = False,
                    target: str = "cli") -> str:
    if target not in LAUNCH_TARGETS:
        raise ValueError("unsupported target: %r" % target)
    name = _validate_safe_name(name, "launcher name")
    profile_ref = _validate_safe_name(profile_ref, "profile name")
    url = _validate_gateway_url(url)
    for mid in plan["models"]:
        if not ID_RE.match(mid):
            raise ValueError("unsafe model id: %r" % mid)
    tiers = plan["tiers"]
    for tval in (tiers.get("pro", ""), tiers.get("flash", ""), tiers.get("flash_lite", ""), default_model or ""):
        if tval and not ID_RE.match(tval):
            raise ValueError("unsafe model id: %r" % tval)

    models_csv = ",".join(plan["models"])
    tiers_csv = "|".join([tiers.get("pro", ""), tiers.get("flash", ""),
                          tiers.get("flash_lite", "")])

    if target == "cli":
        rebuild_args = profile_ref
        run_args = profile_ref
    else:
        rebuild_args = "%s --target %s" % (profile_ref, target)
        run_args = "%s --target %s" % (profile_ref, target)

    out = BASH_TEMPLATE
    out = out.replace("@@VERSION@@", VERSION)
    out = out.replace("@@DATE@@", time.strftime("%Y-%m-%d %H:%M:%S"))
    out = out.replace("@@PROFILE_REF@@", profile_ref)
    out = out.replace("@@PROFILE@@", profile_ref)
    out = out.replace("@@TARGET@@", target)
    out = out.replace("@@REBUILD_ARGS@@", rebuild_args)
    out = out.replace("@@RUN_ARGS@@", run_args)
    out = out.replace("@@NAME@@", launcher_filename(name, target))
    out = out.replace("@@URL@@", url)
    out = out.replace("@@TTL@@", str(int(ttl)))
    out = out.replace("@@CAP@@", str(int(cap)))
    out = out.replace("@@TIERS@@", tiers_csv)
    out = out.replace("@@DEFAULT_MODEL@@", default_model or "")
    out = out.replace("@@MODELS@@", models_csv)
    out = out.replace("@@STRIP_EFFORT@@", "1" if strip_effort else "0")
    return out


# ---------------------------------------------------------------------------
# agy interaction
# ---------------------------------------------------------------------------


def detect_cap() -> int:
    """Probe agy's real model cap with synthetic ids (offline, no API quota)."""
    ids = ",".join("probe/m%03d" % i for i in range(1, 41))
    env = dict(os.environ)
    env.update({
        "AGY_LLM_GATEWAY_URL": "https://probe.invalid/v1",
        "AGY_LLM_GATEWAY_WIRE_PROTOCOL": "openai",
        "AGY_LLM_GATEWAY_API_KEY": "probe",
        "AGY_LLM_GATEWAY_MODELS": ids,
    })
    for var in ("AGY_LLM_GATEWAY_TIER_PRO", "AGY_LLM_GATEWAY_TIER_FLASH",
                "AGY_LLM_GATEWAY_TIER_FLASH_LITE"):
        env.pop(var, None)
    try:
        proc = subprocess.run(["agy", "models"], env=env, capture_output=True,
                              text=True, timeout=45)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if lines:
            return len(lines)
    except (OSError, subprocess.SubprocessError):
        pass
    return CAP_FALLBACK


def verify_launcher(path: Path) -> list:
    """Ask agy which models actually registered, via the generated launcher."""
    try:
        proc = subprocess.run([str(path), "models"], capture_output=True,
                              text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        warn("could not run %s: %s" % (path.name, exc))
        return []
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("Fetching"):
            continue
        out.append(line.split("\t")[0])
    return out


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_list(args) -> int:
    cfg = load_config()
    catalog = load_catalog(cfg, force_refresh=args.refresh, offline=args.offline)
    models = catalog["models"]
    total = len(models)

    if args.search:
        needle = args.search.lower()
        models = [m for m in models if needle in m["id"].lower()]
    models = apply_filters(models, args)
    models = sort_models(models, getattr(args, "sort", None) or "catalog")

    index = index_models(catalog["models"])

    probes = None
    if getattr(args, "probe", False):
        probes = _probe_models(cfg, models, force=getattr(args, "refresh", False))
        if getattr(args, "agy_only", False):
            models = [m for m in models
                      if probe_verdict(probes.get(m["id"])) == "ok"]
    elif not getattr(args, "no_probe_cache", False):
        cached = load_probe_cache()
        if cached:
            # Show what we already know, without spending a request.
            probes = cached

    print()
    if (args.search or args.provider or args.family or args.tools_only
            or args.ready_only or getattr(args, "tee_only", False)):
        print(bold("%d of %d models" % (len(models), total)))
    else:
        print(bold("%d models" % total))
    if catalog["source"] != "api":
        age = (time.time() - catalog["fetched_at"]) / 60.0
        print(dim("catalog source: %s (%.0f min old)" % (catalog["source"], age)))
    print()
    print_table(models, index, probes=probes)
    return 0


def _probe_models(cfg, models, force=False, quiet=False, strip_effort=False,
                  retry_transient=False, validate_tools=False):
    """Probe the given models, printing progress to stderr."""
    key = resolve_api_key(cfg)
    if not key:
        raise SystemExit("error: probing needs an API key. Run `./nearai_agy.py init` "
                         "or set NEAR_AI_KEY.")
    gateway = cfg.get("gateway_url") or DEFAULT_GATEWAY
    cache = load_probe_cache()
    ids = [m["id"] for m in models]

    def progress(done, total, mid):
        if not quiet:
            sys.stderr.write("\r  probing %d/%d  %-46s" % (done, total, mid[:46]))
            sys.stderr.flush()

    results = run_probes(gateway, key, ids, cache, force=force, progress=progress,
                         strip_effort=strip_effort, retry_transient=retry_transient,
                         validate_tools=validate_tools)
    if not quiet:
        sys.stderr.write("\r" + " " * 72 + "\r")
        bad = [i for i in ids if probe_verdict(results.get(i)) == "incompatible"]
        unk = [i for i in ids if probe_verdict(results.get(i)) == "unknown"]
        good = len(ids) - len(bad) - len(unk)
        suffix = " [what-if: strip reasoning_effort]" if strip_effort else (" (cached %ds)" % PROBE_TTL)
        if validate_tools:
            t_good = sum(1 for i in ids if (results.get(i) or {}).get("tool_call_ok") is True)
            info("%d usable (%d verified tool-calling), %d agy-incompatible, %d unknown%s"
                 % (good, t_good, len(bad), len(unk), suffix))
        else:
            info("%d usable, %d agy-incompatible, %d unknown%s"
                 % (good, len(bad), len(unk), suffix))
    save_probe_cache(cache)
    return results


# --- picker ----------------------------------------------------------------

def parse_selection(text: str, count: int) -> list:
    """Parse '1-5,9,20' (1-based) into a list of 0-based indexes."""
    picked = []
    for chunk in text.replace(" ", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            try:
                a, b = int(lo), int(hi)
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            picked.extend(range(a - 1, b))
        else:
            try:
                picked.append(int(chunk) - 1)
            except ValueError:
                continue
    seen, out = set(), []
    for i in picked:
        if 0 <= i < count and i not in seen:
            seen.add(i)
            out.append(i)
    return sorted(out)


def pick_numbered(models: list, preselected: list, cap: int, reserved: int,
                  probes: dict = None) -> list:
    index = index_models(models)
    print_table(models, index, show_all_note=False, probes=probes)
    usable = max(0, cap - reserved)
    print()
    print("Enter numbers/ranges (e.g. %s), " % cyan("1-5,9,20")
          + cyan("f") + "=first %d " % usable
          + cyan("l") + "=last %d " % usable + cyan("a") + "=all "
          + cyan("q") + "=quit")
    if preselected:
        print(dim("preselected: " + ",".join(str(index[m]) for m in preselected if m in index)))
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return []
    if raw.lower() in ("q", "quit", ""):
        return []
    if raw.lower() == "a":
        return [m["id"] for m in models[:usable]]
    if raw.lower() == "f":
        return [m["id"] for m in models[:usable]]
    if raw.lower() == "l":
        return [m["id"] for m in models[-usable:]]
    return [models[i]["id"] for i in parse_selection(raw, len(models))]


def pick_curses(models: list, preselected: list, cap: int, reserved: int,
                probes: dict = None) -> list:
    import curses

    chosen = set(preselected)

    def _run(screen):
        curses.curs_set(0)
        screen.keypad(True)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
        except curses.error:
            pass

        query = ""
        cursor = 0
        top = 0
        searching = False

        def visible():
            if not query:
                return models
            q = query.lower()
            return [m for m in models if q in m["id"].lower()]

        while True:
            rows = visible()
            usable = max(0, cap - reserved)
            height, width = screen.getmaxyx()
            list_rows = max(1, height - 4)

            if cursor >= len(rows):
                cursor = max(0, len(rows) - 1)
            if cursor < top:
                top = cursor
            if cursor >= top + list_rows:
                top = cursor - list_rows + 1

            screen.erase()
            header = " NEAR AI - %d models (%d shown) [ctx / tools / tee / agy]" % (len(models), len(rows))
            if query:
                header += "  search: %s" % query
            _safe_addstr(screen, 0, 0, header.ljust(width - 1), curses.A_BOLD)

            for offset, model in enumerate(rows[top:top + list_rows]):
                y = 1 + offset
                is_cur = (top + offset) == cursor
                mark = "[x]" if model["id"] in chosen else "[ ]"
                tee_txt = fmt_tee(model)
                if probes is not None:
                    state = probe_verdict(probes.get(model["id"]))
                    agy_txt = {"ok": "agy:ok", "incompatible": "  FAIL",
                               "unknown": " agy:?"}[state]
                else:
                    ready = model["is_ready"]
                    agy_txt = " ready" if ready else ("  idle" if ready is False else "     -")
                label = " %s %-44s %6s %5s %-7s %6s" % (
                    mark, model["id"][:44], fmt_ctx(model["context"]),
                    "tool" if model["has_tools"] else "  -", tee_txt, agy_txt,
                )
                attr = curses.A_REVERSE if is_cur else 0
                if model["id"] in chosen and not is_cur:
                    attr |= curses.color_pair(2)
                elif probes is not None and probe_verdict(probes.get(model["id"])) == "incompatible" and not is_cur:
                    attr |= curses.A_DIM
                _safe_addstr(screen, y, 0, label[: width - 1], attr)

            sel = len(chosen)
            free = usable - sel
            status_y = height - 2
            status = " %d selected - %d usable slots (%d reserved for tiers) - %d free" % (
                sel, usable, reserved, free)
            attr = curses.color_pair(3) if free < 0 else curses.color_pair(1)
            _safe_addstr(screen, status_y, 0, status[: width - 1], attr)
            hint = " up/down move  space toggle  a all  n none  / search  enter build  q quit"
            _safe_addstr(screen, height - 1, 0, hint[: width - 1], curses.A_DIM)

            screen.refresh()
            key = screen.getch()

            if searching:
                if key in (10, 13, curses.KEY_ENTER):
                    searching = False
                    cursor = 0
                    top = 0
                elif key == 27:
                    searching = False
                    query = ""
                    cursor = 0
                    top = 0
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    query = query[:-1]
                    cursor = 0
                    top = 0
                elif 32 <= key < 127:
                    query += chr(key)
                    cursor = 0
                    top = 0
                continue

            if key in (ord("q"), 27):
                return []
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(len(rows) - 1, cursor + 1) if rows else 0
            elif key == curses.KEY_PPAGE:
                cursor = max(0, cursor - list_rows)
            elif key == curses.KEY_NPAGE:
                cursor = min(max(0, len(rows) - 1), cursor + list_rows)
            elif key in (ord(" "), 10, 13, curses.KEY_ENTER):
                if key in (10, 13, curses.KEY_ENTER):
                    return [m["id"] for m in models if m["id"] in chosen]
                if rows:
                    mid = rows[cursor]["id"]
                    chosen.symmetric_difference_update({mid})
            elif key == ord("a"):
                chosen = {m["id"] for m in rows[:usable]}
            elif key == ord("n"):
                chosen = set()
            elif key == ord("/"):
                searching = True
                query = ""

    return curses.wrapper(_run)


def _safe_addstr(screen, y, x, text, attr=0) -> None:
    try:
        screen.addstr(y, x, text, attr)
    except Exception:  # writing the last cell of the last line raises
        pass


def cmd_pick(args) -> int:
    cfg = load_config()
    catalog = load_catalog(cfg, force_refresh=args.refresh, offline=args.offline)
    models = apply_filters(catalog["models"], args)
    if not models:
        print("no models matched those filters")
        return 1

    probes = load_probe_cache() or None
    tiers = {"pro": args.tier_pro or "", "flash": args.tier_flash or "",
             "flash_lite": args.tier_flash_lite or ""}
    reserved = len([t for t in tiers.values() if t])
    cap = int(cfg.get("model_cap") or CAP_FALLBACK)

    preselected = [m["id"] for m in models[: min(len(models), max(0, cap - reserved))]] \
        if args.preselect_all else []

    if sys.stdout.isatty() and sys.stdin.isatty():
        chosen = pick_curses(models, preselected, cap, reserved, probes=probes)
    else:
        chosen = pick_numbered(models, preselected, cap, reserved, probes=probes)

    if not chosen:
        print("cancelled")
        return 1

    if not getattr(args, "strip_effort", False) and any(m.lower().startswith("anthropic/") for m in chosen):
        info("selection includes anthropic/*; enabling local reasoning_effort stripper shim")
        args.strip_effort = True

    name = args.name or "custom"
    return _build(args, cfg, chosen, tiers, name, catalog=catalog)


def _build(args, cfg, chosen_ids, tiers, name, catalog=None) -> int:
    cap = int(cfg.get("model_cap") or CAP_FALLBACK)
    ttl = int(cfg.get("cache_ttl") or DEFAULT_TTL)
    url = cfg.get("gateway_url") or DEFAULT_GATEWAY

    plan = plan_slots(chosen_ids, tiers, cap)
    report_evictions(plan)
    if not plan["fits"] and getattr(args, "strict", False):
        sys.stderr.write(red("  error: selection overflows the %d model cap (--strict)\n") % cap)
        return 2

    default_model = args.model or ""
    if default_model and default_model not in plan["models"]:
        warn("default model %s is not in the selected list; ignoring" % default_model)
        default_model = ""
    if not default_model and plan["models"]:
        default_model = plan["models"][0]

    profile_ref = getattr(args, "profile_name", None) or name
    strip_effort = bool(getattr(args, "strip_effort", False))
    target = getattr(args, "target", "cli") or "cli"
    script = render_launcher(name, profile_ref, url, ttl, cap, plan, default_model,
                             strip_effort=strip_effort, target=target)

    outdir = Path(args.output_dir).expanduser() if args.output_dir else LAUNCHER_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / launcher_filename(name, target)

    if args.dry_run:
        print(script)
        return 0

    path.write_text(script, encoding="utf-8")
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass

    synced_models_msg = ""
    if catalog is None:
        try:
            catalog = load_catalog(cfg, offline=getattr(args, "offline", False))
        except Exception:
            catalog = None
    models_to_sync = list(plan["models"])
    for t in plan["tiers"].values():
        if t and t not in models_to_sync:
            models_to_sync.append(t)
    missing = validate_custom_models_configured(models_to_sync)
    if missing:
        try:
            synced = sync_custom_models(catalog or {}, missing)
            if synced > 0:
                synced_models_msg = "synced %d custom model(s) to %s" % (synced, AGY_SETTINGS_PATH)
        except Exception as exc:
            warn("could not sync customModels to settings.json: %s" % exc)
    else:
        synced_models_msg = "verified %d custom model(s) in %s" % (len(models_to_sync), AGY_SETTINGS_PATH)

    print()
    print(green("built ") + str(path))
    print("  profile   : %s" % profile_ref)
    print("  target    : %s (%s)" % (target, TARGET_DISPLAY_NAMES.get(target, target)))
    print("  models    : %d of %d slots" % (len(plan["models"]), cap))
    if synced_models_msg:
        print("  settings  : %s" % synced_models_msg)
    if strip_effort:
        print("  shim      : " + cyan("local reasoning_effort stripper enabled (anthropic/*)"))
    dropped = getattr(args, "dropped_ids", None) or []
    if dropped:
        print(yellow("  skipped   : ")
              + "%d agy-incompatible model(s) replaced with usable ones" % len(dropped))
        print(dim("              ./nearai_agy.py probe --search <id>  for the reason"))
    if any(plan["tiers"].values()):
        used = [k for k, v in plan["tiers"].items() if v]
        print("  tiers     : " + ", ".join("%s=%s" % (k, plan["tiers"][k]) for k in used))
    print("  default   : %s" % (default_model or "(agy picks)"))
    print("  run it    : %s" % cyan(str(path)))
    print(dim("  models are re-validated against the catalog at launch (ttl %ds)" % ttl))

    if args.verify:
        print()
        print(dim("verifying against agy..."))
        actual = verify_launcher(path)
        intended = list(plan["models"])
        for tier in ("pro", "flash", "flash_lite"):
            value = plan["tiers"].get(tier)
            if value and value not in intended:
                intended.append(value)
        missing = [m for m in intended if m not in actual]
        extra = [m for m in actual if m not in intended]
        if not missing and not extra:
            print(green("  verified: %d/%d registered, no surprises" % (len(actual), len(intended))))
        else:
            if missing:
                print(yellow("  not registered: ") + ", ".join(missing))
            if extra:
                print(yellow("  unexpected:     ") + ", ".join(extra))
            print(dim("  (agy reports %d; we intended %d)" % (len(actual), len(intended))))
    return 0


def _load_profile(name: str) -> dict:
    try:
        safe_name = _validate_safe_name(name, "profile name")
    except ValueError as exc:
        raise SystemExit("error: %s" % exc)
    path = PROFILE_DIR / (safe_name + ".json")
    if not path.is_file():
        raise SystemExit("error: no such profile: %s\n       looked for %s" % (name, path))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except ValueError as exc:
        raise SystemExit("error: %s is not valid JSON: %s" % (path, exc))
    if not isinstance(data, dict):
        raise SystemExit("error: %s must contain a JSON object" % path)
    return data


def resolve_profile_models(profile: dict, catalog: dict, cfg: dict,
                           overrides: argparse.Namespace = None) -> tuple[dict, str, bool, list]:
    """Resolve active models, tiers, default model and effort stripping for a profile."""
    select = profile.get("select") or {}
    tiers = profile.get("tiers") or {}
    launch = profile.get("launch") or {}

    def pick_value(flag, key, default=None):
        if overrides is not None:
            val = getattr(overrides, flag, None)
            if val is not None:
                return val
        return select.get(key, default)

    family = pick_value("family", "family") or []
    provider = pick_value("provider", "provider") or []
    exclude = pick_value("exclude", "exclude") or []
    tools_only = (getattr(overrides, "tools_only", False) if overrides else False) or bool(select.get("tools_only"))
    tools_verified = (getattr(overrides, "tools_verified", False) if overrides else False) or bool(select.get("tools_verified"))
    ready_only = (getattr(overrides, "ready_only", False) if overrides else False) or bool(select.get("ready_only"))
    tee_only = (getattr(overrides, "tee_only", False) if overrides else False) or bool(select.get("tee_only") or select.get("private_only"))
    chat_only = select.get("chat_only", True) if (overrides is None or getattr(overrides, "chat_only", None) is None) else overrides.chat_only
    attestation_required = bool(profile.get("attestation_required"))
    if attestation_required and not tee_only:
        raise SystemExit("error: attestation_required profiles must select tee_only/private_only models")
    strip_effort = (getattr(overrides, "strip_effort", False) if overrides else False) or bool(launch.get("strip_effort") or profile.get("strip_effort"))

    include = profile.get("include") or select.get("include") or []

    take = profile.get("take") or {}
    first = getattr(overrides, "first", None) if overrides else None
    if first is None:
        first = take.get("first")
    last = getattr(overrides, "last", None) if overrides else None
    if last is None:
        last = take.get("last")
    count = profile.get("count")
    if count is not None and first is None and last is None:
        first = int(count)

    sort_how = getattr(overrides, "sort", None) if overrides else None
    if sort_how is None:
        sort_how = profile.get("sort", "catalog")

    view = argparse.Namespace(family=family, provider=provider, exclude=exclude,
                              tools_only=tools_only, tools_verified=tools_verified,
                              ready_only=ready_only, tee_only=tee_only, chat_only=chat_only)
    models = apply_filters(catalog["models"], view)
    if attestation_required and not models:
        raise SystemExit("error: attestation_required profile has no recognized TEE models")

    by_id = {m["id"]: m for m in catalog["models"]}
    ordered = []
    for mid in include:
        if mid in by_id:
            ordered.append(by_id[mid])
        else:
            warn("include: %s is not in the catalog any more" % mid)
    for m in models:
        if m["id"] not in include:
            ordered.append(m)

    ordered = sort_models(ordered, sort_how)

    if include and sort_how != "catalog":
        head = [by_id[mid] for mid in include if mid in by_id]
        ordered = head + [m for m in ordered if m["id"] not in include]

    drop_probes = (overrides and getattr(overrides, "drop_probe_failures", False)) or \
        bool((profile.get("probe") or {}).get("drop_failures"))

    dropped_hits = []
    if drop_probes:
        candidates = ordered
        refresh = getattr(overrides, "refresh", False) if overrides else False
        quiet = not sys.stderr.isatty() if overrides else True
        results = {}
        try:
            results = _probe_models(cfg, candidates, force=refresh,
                                    quiet=quiet, strip_effort=strip_effort)
        except SystemExit:
            cached_probe = load_probe_cache()
            if cached_probe:
                results = cached_probe
        rejected = {m["id"] for m in candidates
                    if probe_verdict(results.get(m["id"])) == "incompatible"}
        if rejected:
            would_fit = {m["id"] for m in
                         take_slice(candidates, first=first, last=last)}
            dropped_hits = sorted(rejected & would_fit)
            rest = sorted(rejected - would_fit)
            if overrides:
                report_drops(dropped_hits, results)
                if rest:
                    info("%d further incompatible model(s) were outside this "
                         "selection anyway" % len(rest))
            ordered = [m for m in candidates if m["id"] not in rejected]

    ordered = take_slice(ordered, first=first, last=last)
    chosen = [m["id"] for m in ordered]

    tier_vals = {
        "pro": (overrides and getattr(overrides, "tier_pro", None)) if (overrides and getattr(overrides, "tier_pro", None) is not None) else tiers.get("pro", ""),
        "flash": (overrides and getattr(overrides, "tier_flash", None)) if (overrides and getattr(overrides, "tier_flash", None) is not None) else tiers.get("flash", ""),
        "flash_lite": (overrides and getattr(overrides, "tier_flash_lite", None)) if (overrides and getattr(overrides, "tier_flash_lite", None) is not None) else tiers.get("flash_lite", ""),
    }

    cap = int(cfg.get("model_cap") or CAP_FALLBACK)
    plan = plan_slots(chosen, tier_vals, cap)

    default_model = (overrides and getattr(overrides, "model", None)) if (overrides and getattr(overrides, "model", None) is not None) else launch.get("model", "")
    if default_model and default_model not in plan["models"]:
        default_model = ""
    if not default_model and plan["models"]:
        default_model = plan["models"][0]

    return plan, default_model, strip_effort, dropped_hits


def cmd_run(args) -> int:
    """Execute Antigravity CLI, IDE, or Desktop App using a profile with transparent loopback proxying."""
    cfg = load_config()
    profile_name = args.profile
    target = getattr(args, "target", "cli") or "cli"
    isolated = bool(getattr(args, "isolated", False))
    agy_args = list(getattr(args, "agy_args", []) or [])
    if agy_args and agy_args[0] == "--":
        agy_args = agy_args[1:]

    direct = getattr(args, "direct", False)
    allow_unverified = bool(getattr(args, "allow_unverified", False))
    if "--direct" in agy_args:
        direct = True
        agy_args.remove("--direct")
    if "--no-proxy" in agy_args:
        direct = True
        agy_args.remove("--no-proxy")
    if "--isolated" in agy_args:
        isolated = True
        agy_args.remove("--isolated")
    if "--allow-unverified" in agy_args:
        allow_unverified = True
        agy_args.remove("--allow-unverified")
    if "--target" in agy_args:
        tidx = agy_args.index("--target")
        if tidx + 1 < len(agy_args) and agy_args[tidx + 1] in LAUNCH_TARGETS:
            target = agy_args[tidx + 1]
            del agy_args[tidx:tidx + 2]
    for a in list(agy_args):
        if a.startswith("--target="):
            tval = a.split("=", 1)[1]
            if tval in LAUNCH_TARGETS:
                target = tval
                agy_args.remove(a)
                break

    if target not in LAUNCH_TARGETS:
        sys.stderr.write("nearai: unsupported target: %s (expected cli, ide, or app)\n" % target)
        return 1

    key = resolve_api_key(cfg)
    if not key:
        sys.stderr.write(
            "nearai: NEAR_AI_KEY is not set.\n"
            "  export NEAR_AI_KEY=sk-...   (or run: ./nearai_agy.py init)\n"
        )
        return 1

    executable, is_cli_wrapper = resolve_target_executable(target)
    if not executable:
        if target == "cli":
            sys.stderr.write("nearai: agy not found on PATH\n")
        elif target == "ide":
            sys.stderr.write(
                "nearai: Antigravity IDE not found.\n"
                "  Checked /Applications/Antigravity IDE.app, ~/Applications/Antigravity IDE.app, and `antigravity-ide` on PATH.\n"
                "  Set AGY_IDE_BIN or AGY_IDE_APP to point to a custom installation.\n"
            )
        else:
            sys.stderr.write(
                "nearai: Antigravity Desktop App not found.\n"
                "  Checked /Applications/Antigravity.app, ~/Applications/Antigravity.app, and `antigravity` on PATH.\n"
                "  Set AGY_DESKTOP_BIN or AGY_DESKTOP_APP to point to a custom installation.\n"
            )
        return 127

    has_explicit_udd = any(
        a == "--user-data-dir" or a.startswith("--user-data-dir=") for a in agy_args
    )
    if target in ("ide", "app") and not isolated and not has_explicit_udd:
        if is_gui_target_running(target, executable):
            display = TARGET_DISPLAY_NAMES[target]
            launcher_hint = "./bin/%s --isolated" % launcher_filename(Path(profile_name).stem, target)
            sys.stderr.write(
                "nearai: %s is already running.\n"
                "  Electron reuses the existing instance and will ignore new NEAR AI gateway settings.\n"
                "  Either quit %s first, or launch side-by-side with:\n"
                "    ./nearai_agy.py run %s --target %s --isolated\n"
                "    %s\n" % (display, display, profile_name, target, launcher_hint)
            )
            return 1

    profile_path = PROFILE_DIR / (profile_name + ".json")
    if not profile_path.is_file():
        candidate = Path(profile_name).expanduser()
        if candidate.is_file():
            profile_path = candidate
        else:
            sys.stderr.write("nearai: no such profile: %s\n" % profile_name)
            return 1

    try:
        with open(profile_path, "r", encoding="utf-8") as fh:
            profile = json.load(fh)
    except Exception as exc:
        sys.stderr.write("nearai: failed to load profile %s: %s\n" % (profile_path, exc))
        return 1

    try:
        catalog = load_catalog(cfg, force_refresh=getattr(args, "refresh", False),
                               offline=getattr(args, "offline", False))
    except Exception as exc:
        sys.stderr.write("nearai: failed to load catalog: %s\n" % exc)
        return 1

    plan, default_model, strip_effort, _ = resolve_profile_models(profile, catalog, cfg, overrides=None)

    if not plan["models"]:
        sys.stderr.write("nearai: no usable models left after validation; refusing to launch\n")
        return 1

    raw_gateway_url = _validate_gateway_url(cfg.get("gateway_url") or DEFAULT_GATEWAY)

    if profile.get("attestation_required"):
        if allow_unverified:
            warn(
                "profile '%s' sets attestation_required=true; launching with --allow-unverified "
                "(TEE provider filtering remains active; run ./bin/nearai-private-check to inspect cryptographic evidence)"
                % profile_name
            )
        else:
            gate = _evaluate_attestation_route(
                gateway=raw_gateway_url,
                model=default_model or plan["models"][0],
            )
            if gate["failed"]:
                launcher_cmd = "./bin/%s" % launcher_filename(Path(profile_name).stem, target)
                sys.stderr.write(
                    "nearai: refusing to launch attestation_required profile '%s' (fail-closed preflight).\n"
                    "  unverified checks: %s\n"
                    "  To inspect live cryptographic TDX/GPU/signature evidence, run:\n"
                    "    ./bin/nearai-private-check %s\n"
                    "  To launch with TEE-only provider filtering while upstream verification contracts remain open:\n"
                    "    %s --allow-unverified\n"
                    % (profile_name, ", ".join(gate["failed"]), default_model or plan["models"][0], launcher_cmd)
                )
                return 1

    # Ensure settings.json and jetski_state.pbtxt are validated; only sync if active token limits missing
    models_to_sync = list(plan["models"])
    for t in plan["tiers"].values():
        if t and t not in models_to_sync:
            models_to_sync.append(t)
    missing = validate_custom_models_configured(models_to_sync)
    if missing:
        try:
            sync_custom_models(catalog, missing)
        except Exception as exc:
            warn("could not sync customModels to settings.json: %s" % exc)

    cmd = build_target_command(
        target=target,
        executable=executable,
        is_cli_wrapper=is_cli_wrapper,
        passthrough_args=agy_args,
        default_model=default_model,
        isolated=isolated,
        profile_name=profile_name,
    )

    env = dict(os.environ)
    env["AGY_LLM_GATEWAY_WIRE_PROTOCOL"] = "openai"
    env["AGY_LLM_GATEWAY_MODELS"] = ",".join(plan["models"])
    for var, key_name in (
        ("AGY_LLM_GATEWAY_TIER_PRO", "pro"),
        ("AGY_LLM_GATEWAY_TIER_FLASH", "flash"),
        ("AGY_LLM_GATEWAY_TIER_FLASH_LITE", "flash_lite"),
    ):
        val = plan["tiers"].get(key_name)
        if val:
            env[var] = val
        else:
            env.pop(var, None)

    if target in ("ide", "app"):
        mode_note = "isolated" if (isolated or has_explicit_udd) else "attached"
        info(
            "launching %s (%s, profile=%s, %d models); keep this terminal open until you quit %s"
            % (TARGET_DISPLAY_NAMES[target], mode_note, profile_name, len(plan["models"]), TARGET_DISPLAY_NAMES[target])
        )

    if direct:
        warn("--direct/--no-proxy passes the real NEAR AI API key directly in the child environment")
        env["AGY_LLM_GATEWAY_URL"] = raw_gateway_url
        env["AGY_LLM_GATEWAY_API_KEY"] = key
        env.pop("NEAR_AI_KEY", None)
        try:
            return subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            return 130

    with GatewayProxy(raw_gateway_url, key) as proxy:
        env["AGY_LLM_GATEWAY_URL"] = proxy.gateway_url
        env["AGY_LLM_GATEWAY_API_KEY"] = proxy.session_token
        env.pop("NEAR_AI_KEY", None)
        try:
            return subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            return 130


def cmd_build(args) -> int:
    cfg = load_config()

    if getattr(args, "all_targets", False):
        rc = 0
        for t in LAUNCH_TARGETS:
            sub_t = argparse.Namespace(**vars(args))
            sub_t.target = t
            sub_t.all_targets = False
            rc = cmd_build(sub_t) or rc
        return rc

    if args.all_profiles:
        names = sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
        if not names:
            print("no profiles found in %s" % PROFILE_DIR)
            return 1
        rc = 0
        for name in names:
            rc = cmd_build(_namespace_for_profile(args, name)) or rc
        return rc

    profile = {}
    if args.profile_or_name and (PROFILE_DIR / (args.profile_or_name + ".json")).is_file():
        profile = _load_profile(args.profile_or_name)
        build_name = args.name or args.profile_or_name
        profile_ref = args.profile_or_name
    else:
        build_name = args.name or args.profile_or_name or "custom"
        profile_ref = args.profile_or_name or build_name

    catalog = load_catalog(cfg, force_refresh=args.refresh, offline=args.offline)
    plan, default_model, strip_effort, dropped = resolve_profile_models(profile, catalog, cfg, overrides=args)
    if profile.get("attestation_required"):
        print(yellow("attestation_required: model selection guarded; run attest-route-check before private launch"))

    if not plan["models"]:
        print("no models matched that profile/filters")
        return 1

    drop_probes = getattr(args, "drop_probe_failures", False) or \
        bool((profile.get("probe") or {}).get("drop_failures"))

    by_id = {m["id"]: m for m in catalog["models"]}
    if getattr(args, "probe", False) and not drop_probes:
        results = _probe_models(cfg, [by_id[i] for i in plan["models"] if i in by_id],
                                force=args.refresh, quiet=not sys.stderr.isatty(),
                                strip_effort=strip_effort)
        bad = [mid for mid in plan["models"]
               if probe_verdict(results.get(mid)) == "incompatible"]
        for key, group in group_by_reason(bad, results):
            sys.stderr.write(yellow("  warning: ") + "%d selected model(s) %s\n"
                             % (len(group), reason_label(key)))
            for mid in group:
                sys.stderr.write("           %s\n" % mid)
        if bad:
            info("build with --drop-probe-failures to swap these for working models")

    sub = argparse.Namespace(**vars(args))
    sub.model = default_model
    sub.profile_name = profile_ref
    sub.strip_effort = strip_effort
    sub.dropped_ids = dropped
    return _build(sub, cfg, plan["models"], plan["tiers"], build_name, catalog=catalog)


def _namespace_for_profile(args, name):
    sub = argparse.Namespace(**vars(args))
    sub.profile_or_name = name
    sub.all_profiles = False
    sub.name = None
    return sub


def cmd_doctor(args) -> int:
    cfg = load_config()
    print()
    print(bold("nearai_agy %s" % VERSION))
    print("  python      : %s" % sys.version.split()[0])
    print("  tool dir    : %s" % TOOL_DIR)

    agy = shutil.which("agy")
    if agy:
        try:
            ver = subprocess.run(["agy", "--version"], capture_output=True,
                                 text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            ver = "?"
        print("  agy (cli)   : %s (%s)" % (ver or "?", agy))
    else:
        print("  agy (cli)   : " + red("NOT FOUND on PATH"))

    ide_hub_bin = Path.home() / ".gemini" / "bin" / "agy"
    if ide_hub_bin.is_file():
        hub_ver = ""
        hub_ver_file = ide_hub_bin.parent / "agy.version.json"
        if hub_ver_file.is_file():
            try:
                hub_ver = json.loads(hub_ver_file.read_text(encoding="utf-8")).get("version", "")
            except Exception:
                hub_ver = ""
        print("  agy (hub)   : %s (%s)" % (hub_ver or "present", ide_hub_bin))

    for tkey, label in (("ide", "ide app"), ("app", "desktop app")):
        texe, is_wrap = resolve_target_executable(tkey)
        if texe:
            tver = _read_macos_app_version(texe)
            ver_part = "%s " % tver if tver else ""
            wrap_part = " [cli wrapper]" if is_wrap else ""
            print("  %-12s: %s(%s)%s" % (label, ver_part, texe, wrap_part))
        else:
            print("  %-12s: %s" % (label, dim("not detected")))

    key = resolve_api_key(cfg)
    print("  api key     : %s" % (green("present") if key else red("missing")))
    print("  gateway     : %s" % (cfg.get("gateway_url") or DEFAULT_GATEWAY))
    print("  config      : %s" % (CONFIG_PATH if CONFIG_PATH.exists() else "(none yet)"))
    print("  cache       : %s" % (CACHE_PATH if CACHE_PATH.exists() else "(none yet)"))

    settings_file = AGY_SETTINGS_PATH
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as fh:
                sdata = json.load(fh)
            cms = sdata.get("customModels")
            if not isinstance(cms, dict):
                cms = {}
            valid_cms = [k for k, v in cms.items() if isinstance(v, dict) and (v.get("maxTokens") or 0) > 0]
            print("  settings    : %s (%d custom models, %d with token limits)" % (
                settings_file, len(cms), len(valid_cms)
            ))
        except Exception:
            print("  settings    : " + yellow("%s (unreadable)" % settings_file))
    else:
        print("  settings    : %s (none yet - run `./nearai_agy.py sync-settings`)" % settings_file)

    state_file = AGY_STATE_PATH
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as fh:
                pbtxt = fh.read()
            pbtxt_models = re.findall(r'custom_models:\s*\{\s*key:\s*"([^"]+)"', pbtxt)
            print("  jetski state: %s (%d custom models configured)" % (state_file, len(pbtxt_models)))
        except Exception:
            print("  jetski state: " + yellow("%s (unreadable)" % state_file))
    else:
        print("  jetski state: %s (none yet)" % state_file)

    pro_dir = PRO_PLUGIN_GLOBAL_DIR
    if pro_dir.exists() and (pro_dir / "plugin.json").exists():
        sub_count = len(list((pro_dir / "agents").glob("*.md"))) if (pro_dir / "agents").is_dir() else 0
        print("  pro plugin  : %s (%d subagents installed)" % (pro_dir, sub_count))
    else:
        print("  pro plugin  : none (run `./nearai_agy.py pro setup` to enable Method 2)")

    runner_bin = LAUNCHER_DIR / "nearai-agent-runner"
    if runner_bin.exists():
        print("  agent runner: %s (ok)" % runner_bin)

    if BENCHMARK_CACHE_PATH.exists():
        try:
            with open(BENCHMARK_CACHE_PATH, "r", encoding="utf-8") as fh:
                bdata = json.load(fh)
            mcount = len(bdata.get("models", {}))
            print("  benchmark   : %s (%d models profiled)" % (BENCHMARK_CACHE_PATH, mcount))
        except Exception:
            print("  benchmark   : " + yellow("cache unreadable"))
    else:
        print("  benchmark   : none yet (run `./nearai_agy.py pro benchmark` to generate matrix)")

    if LEDGER_PATH.exists():
        try:
            l_count = 0
            l_cost = 0.0
            with open(LEDGER_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        l_count += 1
                        l_cost += (json.loads(line).get("cost_usd") or 0.0)
            print("  usage ledger: %s (%d calls, $%0.6f spend)" % (LEDGER_PATH, l_count, l_cost))
        except Exception:
            print("  usage ledger: " + yellow("unreadable"))

    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            age = (time.time() - cached.get("fetched_at", 0)) / 60.0
            print("  catalog     : %d models, %.0f min old" % (len(cached.get("models", [])), age))
        except (OSError, ValueError):
            print("  catalog     : " + yellow("cache unreadable"))

    if agy:
        detected = detect_cap()
        stored = int(cfg.get("model_cap") or CAP_FALLBACK)
        line = "  model cap   : %d" % detected
        if detected != stored:
            line += yellow("  (config says %d - run `doctor --save-cap`)" % stored)
        print(line)
        if args.save_cap and detected != stored:
            cfg["model_cap"] = detected
            save_config(cfg)
            info("saved cap=%d to %s" % (detected, CONFIG_PATH))

    print()
    print(bold("launchers in %s" % LAUNCHER_DIR))
    launchers = [p for p in sorted(LAUNCHER_DIR.glob("nearai-*")) if p.name != "nearai-agent-runner"] if LAUNCHER_DIR.is_dir() else []
    if not launchers:
        print(dim("  (none yet - run `./nearai_agy.py build <profile>`)"))
    else:
        live = {}
        if key:
            try:
                catalog = load_catalog(cfg, offline=args.offline)
                live = {m["id"]: m for m in catalog["models"]}
            except SystemExit:
                live = {}
        for path in launchers:
            text = path.read_text(encoding="utf-8", errors="replace")
            built = "?"
            for line in text.splitlines():
                if ("-- generated by nearai_agy.py" in line or "-- forwarder generated by nearai_agy.py" in line) and " on " in line:
                    built = line.split(" on ", 1)[1].strip()
                    break
            ids = _extract_models(text)
            dead = [m for m in ids if live and m not in live]
            status = green("ok") if not dead else yellow("%d stale" % len(dead))
            print("  %-28s %2d models  built %s  %s"
                  % (path.name, len(ids), built, status))
            for mid in dead[:5]:
                print(dim("      - %s" % mid))
        all_launcher_models = []
        for path in launchers:
            text = path.read_text(encoding="utf-8", errors="replace")
            all_launcher_models.extend(_extract_models(text))
        missing_custom = validate_custom_models_configured(all_launcher_models)
        if missing_custom:
            print(yellow("  warning     : %d launcher model(s) missing token limits in settings.json / jetski_state.pbtxt (run `./nearai_agy.py sync-settings`)" % len(missing_custom)))
    print()
    return 0


def _extract_models(script_text: str) -> list:
    for line in script_text.splitlines():
        if line.startswith("_MODELS_CSV="):
            value = line.split("=", 1)[1].strip().strip('"')
            return [m for m in value.split(",") if m]
    return []


def cmd_sync_settings(args) -> int:
    cfg = load_config()
    catalog = load_catalog(cfg, force_refresh=args.refresh, offline=args.offline)

    target_ids = []
    if getattr(args, "models", None):
        target_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    elif getattr(args, "all_catalog", False):
        target_ids = [m["id"] for m in catalog["models"] if m.get("is_chat", True)]
    else:
        seen = set()
        profile_files = sorted(PROFILE_DIR.glob("*.json")) if PROFILE_DIR.is_dir() else []
        for pf in profile_files:
            try:
                pdata = json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for mid in (pdata.get("include") or []):
                if mid and mid not in seen:
                    seen.add(mid)
                    target_ids.append(mid)
            for mid in (pdata.get("tiers") or {}).values():
                if mid and mid not in seen:
                    seen.add(mid)
                    target_ids.append(mid)
        if LAUNCHER_DIR.is_dir():
            for lf in LAUNCHER_DIR.glob("nearai-*"):
                try:
                    text = lf.read_text(encoding="utf-8", errors="replace")
                    for mid in _extract_models(text):
                        if mid and mid not in seen:
                            seen.add(mid)
                            target_ids.append(mid)
                except Exception:
                    pass
        if not target_ids:
            target_ids = [m["id"] for m in catalog["models"] if m.get("is_chat", True)]

    if getattr(args, "check", False):
        missing = validate_custom_models_configured(target_ids)
        if missing:
            sys.stderr.write(yellow("check failed: ") + "%d target model(s) missing token limits in settings.json / jetski_state.pbtxt\n" % len(missing))
            for mid in missing:
                sys.stderr.write("  missing: %s\n" % mid)
            return 1
        print(green("check passed: ") + "all %d target model(s) are configured with positive token limits" % len(target_ids))
        return 0

    if getattr(args, "dry_run", False):
        missing = validate_custom_models_configured(target_ids)
        print("[dry-run] target models : %d" % len(target_ids))
        print("[dry-run] already valid : %d" % (len(target_ids) - len(missing)))
        print("[dry-run] to be synced  : %d" % len(missing))
        if missing:
            print("[dry-run] models that would be added/updated:")
            by_id = {m["id"]: m for m in catalog.get("models", []) if isinstance(m, dict) and "id" in m}
            for mid in missing:
                m = by_id.get(mid) or {}
                ctx = m.get("context") or m.get("context_length") or 131072
                out_tok = m.get("max_output") or m.get("max_output_length") or 8192
                print("  + %s (maxTokens: %d, maxOutputTokens: %d)" % (mid, ctx, out_tok))
        return 0

    count = sync_custom_models(catalog, target_ids)
    print(green("synced ") + "%d model(s) into %s and %s" % (count, AGY_SETTINGS_PATH, AGY_STATE_PATH))
    return 0


def cmd_init(args) -> int:
    cfg = load_config()
    save_key = bool(getattr(args, "save_key", False))
    print()
    print(bold("nearai_agy init"))
    if not cfg.get("api_key") and not os.environ.get("NEAR_AI_KEY"):
        if sys.stdin.isatty():
            key = resolve_api_key(cfg, prompt=True, save=True if save_key else None)
            if not key:
                warn("no key entered; you can set NEAR_AI_KEY later")
        else:
            warn("not a TTY and NEAR_AI_KEY is unset; skipping key entry")
    else:
        print(green("  api key already available"))

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCHER_DIR.mkdir(parents=True, exist_ok=True)

    # Only persist an env-provided NEAR_AI_KEY to disk when explicitly requested via --save-key.
    env_key_raw = os.environ.get("NEAR_AI_KEY", "")
    if save_key and not cfg.get("api_key") and env_key_raw.strip():
        try:
            cfg["api_key"] = _validate_api_key(env_key_raw)
        except ValueError as exc:
            die(str(exc))
        save_config(cfg)
        print("  wrote %s (mode 600)" % CONFIG_PATH)
    elif not CONFIG_PATH.exists():
        save_config(cfg)
        print("  wrote %s (mode 600)" % CONFIG_PATH)
        if env_key_raw.strip() and not cfg.get("api_key"):
            info("NEAR_AI_KEY from environment was kept ephemeral (pass --save-key to persist in config.json)")
    elif env_key_raw.strip() and not cfg.get("api_key"):
        info("NEAR_AI_KEY from environment is kept ephemeral (pass --save-key to persist in config.json)")
    if not cfg.get("api_key"):
        warn("config has no api_key; launchers will need NEAR_AI_KEY in the environment")

    if not list(PROFILE_DIR.glob("*.json")):
        for name, data in SAMPLE_PROFILES.items():
            path = PROFILE_DIR / (name + ".json")
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print("  wrote %s" % path)

    print()
    print("  config : %s" % CONFIG_PATH)
    print("  profiles: %s" % PROFILE_DIR)
    print("  launchers: %s" % LAUNCHER_DIR)
    print()
    print("  next: ./nearai_agy.py list")
    print("        ./nearai_agy.py build lite")
    print()
    return 0


def cmd_list_profiles(args) -> int:
    if not PROFILE_DIR.is_dir():
        print("no profile directory at %s" % PROFILE_DIR)
        return 1
    names = sorted(PROFILE_DIR.glob("*.json"))
    if not names:
        print("no profiles in %s" % PROFILE_DIR)
        return 1
    print()
    for path in names:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print("  %-12s " % path.stem + red("invalid JSON"))
            continue
        desc = data.get("description", "")
        print("  %-12s %s" % (bold(path.stem), desc))
        sel = data.get("select") or {}
        bits = []
        if sel.get("family"):
            bits.append("family=" + ",".join(sel["family"]))
        if sel.get("provider"):
            bits.append("provider=" + ",".join(sel["provider"]))
        if sel.get("tools_only"):
            bits.append("tools-only")
        if sel.get("tee_only") or sel.get("private_only"):
            bits.append("tee-only")
        if data.get("count"):
            bits.append("count=%d" % data["count"])
        take = data.get("take") or {}
        if take.get("first"):
            bits.append("first=%d" % take["first"])
        if take.get("last"):
            bits.append("last=%d" % take["last"])
        if bits:
            print(dim("               " + " ".join(bits)))
    print()
    return 0


def cmd_probe(args) -> int:
    """Probe the catalog in agy's own request shape and report what works."""
    cfg = load_config()
    catalog = load_catalog(cfg, force_refresh=args.refresh, offline=args.offline)
    models = catalog["models"]
    if args.search:
        needle = args.search.lower()
        models = [m for m in models if needle in m["id"].lower()]
    models = apply_filters(models, args)
    models = sort_models(models, getattr(args, "sort", None) or "catalog")
    if not models:
        print("no models matched")
        return 1

    what_if = bool(getattr(args, "what_if", False) or getattr(args, "strip_effort", False))
    retry_transient = bool(getattr(args, "retry_transient", False))
    validate_tools = bool(getattr(args, "validate_tools", False))
    results = _probe_models(cfg, models, force=args.refresh,
                            strip_effort=what_if, retry_transient=retry_transient,
                            validate_tools=validate_tools)
    index = index_models(catalog["models"])
    print()
    if what_if:
        print(cyan("mode: --what-if (stripping `reasoning_effort` for anthropic/*; "
                   "default probe cache untouched)"))
        print()
    if validate_tools:
        print(cyan("mode: --validate-tools (testing function-calling structured tool_calls)"))
        print()
    print_table(models, index, probes=results)

    ok = [m["id"] for m in models if probe_verdict(results.get(m["id"])) == "ok"]
    bad = [m["id"] for m in models if probe_verdict(results.get(m["id"])) == "incompatible"]
    unk = [m["id"] for m in models if probe_verdict(results.get(m["id"])) == "unknown"]

    print()
    print(bold("summary"))
    print("  %s %d usable by agy%s" % (green("ok  "), len(ok),
          " (with --strip-effort shim)" if what_if else ""))
    if validate_tools:
        verified = [mid for mid in ok if (results.get(mid) or {}).get("tool_call_ok") is True]
        failed_tools = [mid for mid in ok if (results.get(mid) or {}).get("tool_call_ok") is False]
        print("  %s %d passed agentic tool-calling validation" % (green("tools"), len(verified)))
        if failed_tools:
            print("  %s %d reachable but failed tool-calling:" % (yellow("warn "), len(failed_tools)))
            for mid in failed_tools:
                err = (results.get(mid) or {}).get("tool_call_error") or "no tool call emitted"
                print("         %-36s : %s" % (mid, dim(err)))
    print("  %s %d rejected agy's request shape" % (red("fail"), len(bad)))

    # Report casualties as classes of failure, with the reason spelled out, so
    # a dropped model is never just a name with no explanation.
    for key, ids in group_by_reason(bad, results):
        print()
        print(red("  ") + bold("%d model(s): " % len(ids)) + reason_label(key))
        note = reason_note(key)
        if note:
            print(dim("      " + note))
        for mid in ids:
            print("      %-40s %s" % (mid, dim(error_gist(results.get(mid), 64))))

    if unk:
        print()
        print(yellow("  %d model(s) could not be reached" % len(unk))
              + dim(" (retry with `./nearai_agy.py probe --retry-transient`)"))
        for mid in unk:
            print("      %-40s %s" % (mid, dim(error_gist(results.get(mid), 64))))

    if bad:
        print()
        print(dim("  `build --drop-probe-failures` swaps these for models that work."))
        print(dim("  `build --strip-effort` enables the local shim for anthropic/* models."))
        print(dim("  Background: ArchitectureGuide.md"))
    print()
    return 0


def cmd_systems_test(args) -> int:
    """Run a comprehensive, end-to-end systems test of models through GatewayProxy."""
    cfg = load_config()
    key = resolve_api_key(cfg)
    if not key:
        sys.stderr.write("nearai: API key required for systems test\n")
        return 1

    catalog = load_catalog(cfg, force_refresh=getattr(args, "refresh", False),
                           offline=getattr(args, "offline", False))
    raw_models = catalog.get("models") or []
    by_id = {m["id"]: m for m in raw_models if isinstance(m, dict) and "id" in m}

    # Resolve target models
    target_ids = []
    if getattr(args, "models", None):
        target_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    elif getattr(args, "profile", None):
        try:
            profile = _load_profile(args.profile)
            plan, default_model, _, _ = resolve_profile_models(profile, catalog, cfg)
            target_ids = list(plan["models"])
            for t in plan["tiers"].values():
                if t and t not in target_ids:
                    target_ids.append(t)
        except Exception as exc:
            sys.stderr.write("nearai: failed to resolve profile %s: %s\n" % (args.profile, exc))
            return 1
    else:
        # Default: all chat-capable models in the catalog
        target_ids = [m["id"] for m in raw_models if _is_chat_model(m["id"], m.get("inputs"), m.get("outputs"))]

    limit = getattr(args, "limit", None)
    if limit and limit > 0:
        target_ids = target_ids[:limit]

    if not target_ids:
        print("no matching models to test")
        return 1

    workers = max(1, min(getattr(args, "workers", 3) or 3, len(target_ids)))
    skip_tools = bool(getattr(args, "skip_tools", False))
    update_cache = bool(getattr(args, "update_cache", False))
    json_path = getattr(args, "json_path", None)

    print()
    print(bold("NEAR AI End-to-End Systems Test"))
    print("  target models : %d" % len(target_ids))
    print("  worker threads: %d" % workers)
    print("  tool calling  : %s" % ("disabled (--skip-tools)" if skip_tools else "enabled (read_file schema)"))
    print()

    raw_gateway_url = _validate_gateway_url(cfg.get("gateway_url") or DEFAULT_GATEWAY)

    results = {}
    lock = threading.Lock()
    done = 0

    with GatewayProxy(raw_gateway_url, key) as proxy:
        def test_one(mid):
            nonlocal done
            entry = {
                "id": mid,
                "ping_ok": False,
                "ping_code": 0,
                "ping_ms": 0,
                "ping_error": "",
                "tool_ok": None,
                "tool_code": 0,
                "tool_ms": 0,
                "tool_error": "",
                "status": "UNKNOWN",
                "shimmed": False,
            }
            if mid.lower().startswith("anthropic/"):
                entry["shimmed"] = True

            # 1. Ping / wire-protocol test
            t0 = time.time()
            # We pass strip_effort=False into probe_one so reasoning_effort IS sent to our proxy,
            # verifying that GatewayProxy intercepts and strips it for anthropic/*.
            ping_res = probe_one(proxy.gateway_url, proxy.session_token, mid,
                                 strip_effort=False, validate_tools=False)
            t1 = time.time()
            entry["ping_ms"] = int((t1 - t0) * 1000)
            entry["ping_ok"] = bool(ping_res.get("ok"))
            entry["ping_code"] = ping_res.get("code") or 0
            entry["ping_error"] = ping_res.get("error") or ""

            # 2. Function-calling tool use test
            if not skip_tools and entry["ping_ok"]:
                t2 = time.time()
                tool_res = probe_one(proxy.gateway_url, proxy.session_token, mid,
                                     strip_effort=False, validate_tools=True)
                t3 = time.time()
                entry["tool_ms"] = int((t3 - t2) * 1000)
                entry["tool_ok"] = bool(tool_res.get("tool_call_ok"))
                entry["tool_code"] = tool_res.get("code") or 0
                entry["tool_error"] = tool_res.get("tool_call_error") or tool_res.get("error") or ""

            # Classification
            if entry["ping_ok"]:
                if skip_tools:
                    entry["status"] = "PING_OK"
                elif entry["tool_ok"]:
                    entry["status"] = "AGENT_READY"
                else:
                    entry["status"] = "CHAT_ONLY"
            elif entry["ping_code"] == 400:
                entry["status"] = "INCOMPATIBLE"
            elif entry["ping_code"] in (429, 502, 503, 504) or entry["ping_code"] == 0:
                entry["status"] = "TRANSIENT"
            else:
                entry["status"] = "FAILED"

            with lock:
                results[mid] = entry
                done += 1
                # Live progress line
                status_colored = (
                    green("AGENT_READY") if entry["status"] == "AGENT_READY"
                    else cyan("CHAT_ONLY") if entry["status"] == "CHAT_ONLY"
                    else yellow("TRANSIENT") if entry["status"] == "TRANSIENT"
                    else red(entry["status"])
                )
                timing = "%dms" % (entry["ping_ms"] + entry["tool_ms"])
                sys.stdout.write("  [%d/%d] %-42s %-20s (%s)\n" % (
                    done, len(target_ids), mid[:42], status_colored, timing
                ))
                sys.stdout.flush()

        threads = []
        for mid in target_ids:
            th = threading.Thread(target=test_one, args=(mid,), daemon=True)
            th.start()
            threads.append(th)
            while sum(1 for t in threads if t.is_alive()) >= workers:
                time.sleep(0.02)
        for th in threads:
            th.join()

    # Summary table
    print()
    print("=" * 82)
    print(bold("SYSTEMS TEST RESULTS MATRIX"))
    print("=" * 82)
    print(bold("%-36s  %-7s  %-8s  %-10s  %-8s  %s" % (
        "Model ID", "Context", "Ping", "Tool-Call", "Latency", "Status"
    )))
    print("-" * 82)

    agent_ready = 0
    chat_only = 0
    incompatible = 0
    transient = 0
    failed = 0

    for mid in target_ids:
        r = results.get(mid) or {}
        m_meta = by_id.get(mid) or {}
        ctx = m_meta.get("context") or m_meta.get("context_length") or 0
        ctx_str = "%-7s" % (("%dk" % (ctx // 1024)) if ctx else "?")

        ping_raw = "200 OK" if r.get("ping_ok") else str(r.get("ping_code") or "ERR")
        ping_pad = "%-8s" % ping_raw
        ping_str = green(ping_pad) if r.get("ping_ok") else red(ping_pad)

        if r.get("tool_ok") is True:
            tool_str = green("%-10s" % "PASS")
        elif r.get("tool_ok") is False:
            tool_str = yellow("%-10s" % "FAIL")
        else:
            tool_str = dim("%-10s" % "SKIP")

        lat = "%-8s" % ("%dms" % (r.get("ping_ms", 0) + r.get("tool_ms", 0)))
        status = r.get("status", "UNKNOWN")

        if status == "AGENT_READY":
            agent_ready += 1
            stat_str = green("AGENT_READY")
        elif status == "CHAT_ONLY":
            chat_only += 1
            stat_str = cyan("CHAT_ONLY")
        elif status == "INCOMPATIBLE":
            incompatible += 1
            stat_str = red("INCOMPATIBLE")
        elif status == "TRANSIENT":
            transient += 1
            stat_str = yellow("TRANSIENT")
        else:
            failed += 1
            stat_str = red(status)

        shim_flag = " [shim]" if r.get("shimmed") else ""
        mid_disp = "%-36s" % (mid if len(mid) <= 36 else (mid[:35] + "~"))
        print("%s  %s  %s  %s  %s  %s" % (mid_disp, ctx_str, ping_str, tool_str, lat, stat_str + shim_flag))

    print("-" * 82)
    print("Total Tested        : %d" % len(target_ids))
    print("  " + green("AGENT_READY") + "       : %d (verified OpenAI function-calling / tool use)" % agent_ready)
    print("  " + cyan("CHAT_ONLY") + "         : %d (conversational, but failed/omitted tool_calls)" % chat_only)
    if incompatible:
        print("  " + red("INCOMPATIBLE") + "      : %d (HTTP 400 rejection)" % incompatible)
    if transient:
        print("  " + yellow("TRANSIENT") + "         : %d (rate-limit 429 or server error)" % transient)
    if failed:
        print("  " + red("FAILED") + "            : %d (connection or parsing failure)" % failed)
    print("=" * 80)
    print()

    # Update probe cache if requested
    if update_cache:
        probe_cache = load_probe_cache()
        for mid, r in results.items():
            probe_cache[mid] = {
                "ok": r["ping_ok"],
                "code": r["ping_code"],
                "error": r["ping_error"],
                "verdict": (r["ping_code"] == 400),
                "tool_call_ok": bool(r["tool_ok"]),
                "tool_call_error": r["tool_error"],
                "at": time.time(),
            }
        save_probe_cache(probe_cache)
        print(green("updated probe cache ") + "with %d test verdicts" % len(results))

    # JSON export if requested
    if json_path:
        out_file = Path(json_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "timestamp": time.time(),
            "target_count": len(target_ids),
            "summary": {
                "agent_ready": agent_ready,
                "chat_only": chat_only,
                "incompatible": incompatible,
                "transient": transient,
                "failed": failed,
            },
            "models": results,
        }
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(report_data, fh, indent=2)
        print(green("wrote report ") + str(out_file))

    # Return 0 if at least some models passed, 1 only if 0 passed
    return 0 if (agent_ready + chat_only > 0) else 1


def cmd_verify(args) -> int:
    """Fetch and display NEAR AI hardware attestation report and pins."""
    cfg = load_config()
    try:
        gateway = _validate_gateway_url(cfg.get("gateway_url") or DEFAULT_GATEWAY).rstrip("/")
    except ValueError as exc:
        die(str(exc))
    model = getattr(args, "model", None) or ""
    url = gateway + "/attestation/report"
    if model:
        url += "?model=" + urllib.parse.quote(model, safe="")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with _NO_REDIRECT_OPENER.open(req, timeout=20) as resp:
            report = json.loads(_read_http_bounded(resp).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            payload = json.loads(_read_http_bounded(exc, 64 * 1024).decode("utf-8", "replace"))
            err_obj = payload.get("error")
            detail = (err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)) or ""
        except Exception:
            detail = "HTTP %d" % exc.code
        detail = str(redact_secrets(detail))
        print()
        print(red("attestation rejected ") + ("for %s" % bold(model) if model else ""))
        print("  " + detail)
        print()
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        die("could not fetch attestation report: %s" % exc)

    pins = {}
    try:
        req_pins = urllib.request.Request(gateway + "/attestation/pins",
                                          headers={"Accept": "application/json"})
        with _NO_REDIRECT_OPENER.open(req_pins, timeout=15) as resp:
            pins = json.loads(_read_http_bounded(resp).decode("utf-8"))
    except Exception:
        pins = {}

    if getattr(args, "json", False):
        print(json.dumps({"report": report, "pins": pins}, indent=2))
        return 0

    gw = report.get("gateway_attestation") or {}
    gw_info = gw.get("info") or {}
    ohttp = report.get("ohttp_attestation") or {}

    print()
    print(bold("NEAR AI Gateway Attestation"))
    print("  endpoint        : %s" % gateway)
    print("  status          : %s" % (yellow("quote present; unverified")
                                      if gw.get("intel_quote") else yellow("unverified")))
    print("  signing algo    : %s" % (gw.get("signing_algo") or "-"))
    print("  signing address : %s" % (gw.get("signing_address") or "-"))
    if gw_info.get("app_id"):
        print("  app id          : %s" % gw_info.get("app_id"))
    if gw_info.get("instance_id"):
        print("  instance id     : %s" % gw_info.get("instance_id"))
    if ohttp.get("signing_key"):
        match = ohttp.get("signing_key") == gw.get("signing_address")
        print("  ohttp key match : %s" % (green("yes") if match else yellow("mismatch")))

    model_atts = report.get("model_attestations") or []
    if model or model_atts:
        print()
        print(bold("Model Attestation" + (" (%s)" % model if model else "")))
        for ma in model_atts:
            m_info = ma.get("info") or {}
            has_tdx = bool(ma.get("intel_quote"))
            has_gpu = bool(ma.get("nvidia_payload"))
            hw = []
            if has_tdx:
                hw.append("Intel TDX")
            if has_gpu:
                hw.append("NVIDIA GPU TEE")
            print("  model           : %s" % (ma.get("model_name") or model or "-"))
            print("  hardware quotes : %s" % (green(", ".join(hw)) if hw else yellow("none")))
            print("  signing algo    : %s" % (ma.get("signing_algo") or "-"))
            print("  signing address : %s" % (ma.get("signing_address") or "-"))
            if m_info.get("app_id"):
                print("  app id          : %s" % m_info.get("app_id"))

    if pins and isinstance(pins, dict):
        print()
        print(bold("Published Measurement Pins (/v1/attestation/pins)"))
        for provider, pdata in sorted(pins.items()):
            if isinstance(pdata, dict):
                comp = pdata.get("compiled") or {}
                routers = len(comp.get("routers") or []) if isinstance(comp, dict) else 0
                models_cnt = len(comp.get("models") or {}) if isinstance(comp, dict) else 0
                print("  %-15s : %d router pin(s), %d model family pin(s)"
                      % (provider, routers, models_cnt))
    print()
    return 0


def _evaluate_attestation_route(
    gateway: str,
    model: str = "",
    provider: str = "",
    verify_crypto: bool = False,
    accepted_tcb_statuses=None,
) -> dict:
    """Run fail-closed attestation preflight and return structured status + failed checks."""
    try:
        result = collect_evidence(
            gateway=gateway,
            model=model,
            provider=provider,
            verify_crypto=verify_crypto,
            accepted_tcb_statuses=accepted_tcb_statuses,
        )
    except Exception as exc:
        result = {"label": "evidence", "status": "unverified", "errors": [{"error": str(redact_secrets(str(exc)))}]}
    summary = result.get("verification_summary") or build_verification_summary(result)
    required = ("request_binding", "tls_binding", "ohttp_signature", "gateway_quote", "model_execution", "freshness")
    freshness = result.get("freshness") or {}
    failed = [
        key for key in required
        if (key == "freshness" and not freshness.get("fresh"))
        or (key != "freshness" and summary.get(key) != "verified")
    ]
    return {
        "mode": "opt_in_attestation_route_check",
        "status": "fail_closed" if failed else "eligible",
        "required": list(required),
        "failed": failed,
        "verification_summary": summary,
    }


def cmd_attest_route_check(args) -> int:
    """Fail-closed preflight for future opt-in attestation-backed routing."""
    cfg = load_config()
    gateway = getattr(args, "gateway", None) or cfg.get("gateway_url") or DEFAULT_GATEWAY
    verify_crypto = bool(getattr(args, "verify_crypto", False))
    accepted_tcb = {"UpToDate", "OutOfDate"} if getattr(args, "allow_out_of_date_tcb", False) else None
    payload = _evaluate_attestation_route(
        gateway=gateway,
        model=getattr(args, "model", "") or "",
        provider=getattr(args, "provider", "") or "",
        verify_crypto=verify_crypto,
        accepted_tcb_statuses=accepted_tcb,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["failed"] else 0


def cmd_attest_evidence(args) -> int:
    """Collect nonce/TLS-bound evidence with optional cryptographic verification."""
    cfg = load_config()
    gateway = getattr(args, "gateway", None) or cfg.get("gateway_url") or DEFAULT_GATEWAY
    verify_crypto = bool(getattr(args, "verify_crypto", False))
    accepted_tcb = {"UpToDate", "OutOfDate"} if getattr(args, "allow_out_of_date_tcb", False) else None
    try:
        result = collect_evidence(
            gateway=gateway,
            model=getattr(args, "model", "") or "",
            provider=getattr(args, "provider", "") or "",
            include_tls_fingerprint=not getattr(args, "no_tls_fingerprint", False),
            verify_crypto=verify_crypto,
            accepted_tcb_statuses=accepted_tcb,
        )
    except Exception as exc:
        result = {"label": "evidence", "status": "unverified", "errors": [{"error": str(exc)}]}
    summary = result.get("verification_summary") or build_verification_summary(result)
    if getattr(args, "json", False):
        result["verification_summary"] = summary
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result.get("errors") else 1
    print()
    mode_label = "cryptographic + evidence" if verify_crypto else "evidence-only"
    print(bold(f"NEAR AI Attestation Evidence ({mode_label})"))
    print("  endpoint        : %s" % gateway.rstrip("/"))
    print("  status          : %s" % yellow("unverified"))
    if result.get("model") or getattr(args, "model", ""):
        print("  model           : %s" % (getattr(args, "model", "") or "-"))
    validation = result.get("validation") or {}
    checks = validation.get("checks") or {}
    if checks:
        print("  nonce binding   : %s" % (green("pass") if checks.get("request_nonce") else red("fail")))
        print("  report data     : %s" % (green("pass") if checks.get("report_data") else red("fail")))
        print("  TLS fingerprint : %s" % (green("pass") if checks.get("tls_fingerprint") else red("fail")))
    print("  OHTTP signature : %s" % summary.get("ohttp_signature"))
    print("  gateway quote   : %s" % summary.get("gateway_quote"))
    if summary.get("model_quotes") is not None:
        print("  model quotes    : %s" % ", ".join(f"{m.get('model')}:{(m.get('result') or {}).get('status')}" for m in summary["model_quotes"]))
    if summary.get("gpu_evidence") is not None:
        print("  GPU evidence    : %s" % ", ".join(f"{g.get('model')}:{(g.get('result') or {}).get('status')}" for g in summary["gpu_evidence"]))
    print("  model execution : %s" % summary.get("model_execution"))
    print("  overall trust   : %s" % yellow(summary.get("overall", "unverified")))
    for error in result.get("errors") or []:
        print("  error           : %s" % error.get("error", error))
    print()
    return 0 if not result.get("errors") else 1


def cmd_verify_chat(args) -> int:
    """Run a preflight-attested /chat/completions call and verify /signature/{chat_id}."""
    cfg = load_config()
    raw_gateway = getattr(args, "gateway", None) or cfg.get("gateway_url") or DEFAULT_GATEWAY
    try:
        gateway = _validate_gateway_url(raw_gateway)
    except ValueError as exc:
        sys.stderr.write("nearai: invalid gateway URL: %s\n" % exc)
        return 1
    key = resolve_api_key(cfg)
    if not key:
        sys.stderr.write("nearai: NEAR_AI_KEY is not set. Run ./nearai_agy.py init or export NEAR_AI_KEY.\n")
        return 1
    model = getattr(args, "model", "") or ""
    if not model:
        sys.stderr.write("nearai: model argument is required for verify-chat.\n")
        return 1
    verify_crypto = not getattr(args, "no_verify_crypto", False)
    accepted_tcb = {"UpToDate", "OutOfDate"} if getattr(args, "allow_out_of_date_tcb", False) else None
    result = verify_chat_completion(
        gateway=gateway,
        api_key=key,
        model=model,
        prompt=getattr(args, "prompt", "Reply with OK.") or "Reply with OK.",
        signing_algo=getattr(args, "signing_algo", "ecdsa") or "ecdsa",
        include_tls_fingerprint=not getattr(args, "no_tls_fingerprint", False),
        verify_crypto=verify_crypto,
        accepted_tcb_statuses=accepted_tcb,
    )
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        sig_ver = result.get("signature_verification") or {}
        summary = result.get("verification_summary") or {}
        print()
        print(bold("NEAR AI Verified Chat Check"))
        print("  endpoint        : %s" % gateway.rstrip("/"))
        print("  model           : %s" % model)
        print("  chat_id         : %s" % (result.get("chat_id") or "-"))
        print("  gateway quote   : %s" % summary.get("gateway_quote"))
        if summary.get("model_quotes") is not None:
            print("  model quotes    : %s" % ", ".join(f"{m.get('model')}:{(m.get('result') or {}).get('status')}" for m in summary["model_quotes"]))
        if summary.get("gpu_evidence") is not None:
            print("  GPU evidence    : %s" % ", ".join(f"{g.get('model')}:{(g.get('result') or {}).get('status')}" for g in summary["gpu_evidence"]))
        print("  response sig    : %s (kind=%s, provenance=%s)" % (sig_ver.get("status"), sig_ver.get("signature_kind"), sig_ver.get("provenance")))
        if sig_ver.get("reason"):
            print("  sig reason      : %s" % sig_ver.get("reason"))
        print("  overall trust   : %s" % yellow(summary.get("overall", "unverified")))
        for error in result.get("errors") or []:
            print("  error           : %s" % error.get("error", error))
        print()
    sig_ok = bool((result.get("signature_verification") or {}).get("verified"))
    return 0 if sig_ok and not result.get("errors") else 1


SAMPLE_PROFILES = {
    "private": {
        "description": "Hardware-attested TEE models only (vllm / chutes / tinfoil), tool-capable",
        "attestation_required": True,
        "sort": "expensive",
        "take": {"first": 32},
        "launch": {"model": "Qwen/Qwen3.8-27B"},
        "probe": {"drop_failures": True},
    },
    "anthropic": {
        "description": "Anthropic Claude model family (with local reasoning_effort stripper shim)",
        "select": {"family": ["anthropic"], "tools_only": True},
        "sort": "expensive",
        "take": {"first": 32},
        "launch": {"model": "anthropic/claude-sonnet-4-5", "strip_effort": True},
        "probe": {"drop_failures": True},
    },
    "lite": {
        "description": "Cheapest tool-capable, agy-verified models, low cost per token",
        "select": {"tools_only": True},
        "sort": "cheap",
        "count": 8,
        "launch": {"model": ""},
        "probe": {"drop_failures": True},
    },
    "frontier": {
        "description": "Most capable agy-verified models first, including Claude via local shim (includes external non-TEE providers)",
        "select": {"tools_only": True},
        "sort": "expensive",
        "take": {"first": 32},
        "launch": {"model": "openai/gpt-5.6-sol", "strip_effort": True},
        "probe": {"drop_failures": True},
    },
}


# ---------------------------------------------------------------------------
# Agy Pro Plugin & External Subagents (Method 2)
# ---------------------------------------------------------------------------


def render_subagent_md(name: str, spec: dict) -> str:
    desc = spec.get("description", "").replace('"', '\\"')
    persona = spec.get("persona", name)
    instructions = spec.get("instructions", "")
    return (
        "---\n"
        f"name: {name}\n"
        f"description: \"{desc}\"\n"
        "tools:\n"
        "  - view_file\n"
        "  - write_to_file\n"
        "  - replace_file_content\n"
        "  - grep_search\n"
        "  - find_by_name\n"
        "  - list_dir\n"
        "  - run_command\n"
        "mainAgent: true\n"
        "subagent: true\n"
        "commandExecutionPolicy: auto\n"
        "---\n\n"
        f"# {persona}\n\n"
        f"{instructions}\n"
    )


def render_plugin_json() -> str:
    manifest = {
        "name": "nearai-pro",
        "displayName": "NEAR AI Pro Subagents",
        "version": VERSION,
        "description": "Specialized NEAR AI subagents for hardware-attested TEE execution, uncensored security research, and high-performance reasoning.",
        "author": {
            "name": "NEAR AI & Antigravity Community"
        },
        "keywords": [
            "nearai",
            "tee",
            "privacy",
            "uncensored",
            "claude",
            "deepseek",
            "antigravity",
            "subagents"
        ]
    }
    return json.dumps(manifest, indent=2) + "\n"


def render_agents_rules() -> str:
    return (
        "# Antigravity Subagent Delegation Directives for NEAR AI\n\n"
        "When operating in Hybrid Orchestrator Mode, Google Gemini acts as the primary orchestrator, "
        "maintaining workspace context, user interaction, and overall project coordination.\n\n"
        "## Subagent Delegation Heuristics\n\n"
        "Delegate specialized workflow stages to NEAR AI subagents according to these directives:\n\n"
        "1. **Hardware-Attested Privacy / Sensitive Data:**\n"
        "   - **Subagent:** `nearai-private`\n"
        "   - **Target Model:** `Qwen/Qwen3.8-27B` (Intel TDX / NVIDIA H200 GPU TEE)\n"
        "   - **Trigger:** Auditing cryptographic keys, private tokens, proprietary business logic, "
        "or compliance-bound source code (HIPAA/GDPR) where data must stay within a hardware TEE.\n\n"
        "2. **Defensive Security & Hardening Review:**\n"
        "   - **Subagent:** `nearai-uncensored`\n"
        "   - **Target Model:** `Qwen/Qwen3.8-27B-abliterated`\n"
        "   - **Trigger:** Deep defensive security reviews, protocol edge-case auditing, parser/boundary validation, "
        "and concrete hardening patches where standard commercial filters produce false-positive refusals.\n\n"
        "3. **Complex Cross-File Architectural Refactoring:**\n"
        "   - **Subagent:** `nearai-claude`\n"
        "   - **Target Model:** `anthropic/claude-sonnet-4-5`\n"
        "   - **Trigger:** Deep multi-module refactoring, breaking changes analysis, and large-scale semantic diffing "
        "requiring Anthropic Claude Sonnet's specialized code-editing capability.\n\n"
        "4. **Algorithmic Math, Performance Tuning & Bulk Tasks:**\n"
        "   - **Subagent:** `nearai-deepseek`\n"
        "   - **Target Model:** `deepseek/deepseek-v4.1-flash-1017`\n"
        "   - **Trigger:** Micro-optimizations, mathematical proofs, complex regex compilation, or high-volume "
        "linting/parsing where DeepSeek provides ultra-low latency and minimal cost.\n\n"
        "5. **General Secondary Delegation:**\n"
        "   - **Subagent:** `nearai-general`\n"
        "   - **Target Model:** `openai/gpt-5.6-sol`\n"
        "   - **Trigger:** General background research, parallel verification, or non-critical secondary drafting.\n"
    )


def render_delegation_skill() -> str:
    return (
        "---\n"
        "name: nearai-delegation\n"
        "description: >-\n"
        "  Orchestrate and delegate specialized coding, security, and architectural tasks\n"
        "  from Google Gemini to NEAR AI models (TEE enclaves, uncensored models, Claude, DeepSeek).\n"
        "---\n\n"
        "# NEAR AI Delegation Skill\n\n"
        "This skill guides the primary orchestrator on delegating tasks to NEAR AI subagents and invoking the CLI runner bridge.\n\n"
        "## Quick Reference: Model & Task Routing\n\n"
        "| Specialized Need | Subagent | Best-Fit Model | Why Gemini Delegates Here |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| **Confidential IP & Secrets** | `nearai-private` | `Qwen/Qwen3.8-27B` (TEE) | Zero-trust hardware enclave (AMD SEV / Intel TDX); operator-blind. |\n"
        "| **Unfiltered Security Hardening** | `nearai-uncensored` | `Qwen/Qwen3.8-27B-abliterated` | Avoids false-positive refusals during deep defensive security audits. |\n"
        "| **Deep Architectural Refactoring** | `nearai-claude` | `anthropic/claude-sonnet-4-5` | Premier code-editing precision and subtle multi-file semantic diffs. |\n"
        "| **High-Speed Math & Algorithms** | `nearai-deepseek` | `deepseek/deepseek-v4.1-flash-1017` | Ultra-fast token generation and low-cost mathematical proofs. |\n"
        "| **General Secondary Tasks** | `nearai-general` | `openai/gpt-5.6-sol` | Cost-effective background delegate. |\n\n"
        "## How to Invoke Subagents\n\n"
        "From within an Antigravity turn, invoke the appropriate subagent by name using `invoke_subagent`:\n"
        "- `TypeName: nearai-private`\n"
        "- `TypeName: nearai-uncensored`\n"
        "- `TypeName: nearai-claude`\n"
        "- `TypeName: nearai-deepseek`\n"
        "- `TypeName: nearai-general`\n\n"
        "Or execute a direct prompt via `run_command`:\n"
        "```bash\n"
        "nearai-agent-runner --model Qwen/Qwen3.8-27B --stream --prompt \"Audit this token auth flow for timing attacks\"\n"
        "```\n"
    )


def render_audit_skill(model_id: str = "Qwen/Qwen3.8-27B") -> str:
    return (
        "---\n"
        "name: nearai-audit\n"
        "description: >-\n"
        "  Audits cryptographic implementations, authentication flows, secrets, private keys,\n"
        "  and sensitive regulatory data (HIPAA/GDPR) inside a zero-trust hardware TEE.\n"
        "---\n\n"
        "# NEAR AI Confidential Security & Cryptography Audit\n\n"
        "This skill guides security audits requiring complete hardware isolation.\n\n"
        "## Target Model\n"
        f"- **Primary:** `{model_id}` (Hardware-attested TEE enclave)\n\n"
        "## Auditing Invariants\n"
        "1. **Timing Attacks:** Verify all token/hash comparisons use constant-time functions (`hmac.compare_digest` / `secrets.compare_digest`).\n"
        "2. **Key Storage:** Ensure private keys and secrets are never logged, formatted into exception messages, or written to disk unencrypted.\n"
        "3. **Hardware Privacy:** Execute analysis in the hardware enclave via:\n"
        "```bash\n"
        f"nearai-agent-runner --model {model_id} --prompt \"Audit this token auth flow for timing attacks\"\n"
        "```\n"
    )


def render_redteam_skill(model_id: str = "Qwen/Qwen3.8-27B-abliterated") -> str:
    return (
        "---\n"
        "name: nearai-redteam\n"
        "description: >-\n"
        "  Executes unfiltered defensive security auditing, boundary verification, and hardening\n"
        "  patch generation using abliterated models on NEAR AI.\n"
        "---\n\n"
        "# NEAR AI Unfiltered Security Hardening Review\n\n"
        "This skill guides deep defensive security audits and vulnerability remediation.\n\n"
        "## Target Model\n"
        f"- **Primary:** `{model_id}` (Unfiltered technical review)\n\n"
        "## Usage Directives\n"
        "1. **Deep Security Verification:** Use for auditing parser differentials, HTTP framing edge cases, auth isolation, and input validation where standard filters trigger false-positive refusals.\n"
        "2. **Defensive Accompaniment:** Every verified security finding must be paired with concrete remediation patches and regression tests.\n"
        "3. **Execution:**\n"
        "```bash\n"
        f"nearai-agent-runner --model {model_id} --stream --prompt \"Review this HTTP proxy for header and URL validation gaps\"\n"
        "```\n"
    )


def render_refactor_skill(model_id: str = "anthropic/claude-sonnet-4-5") -> str:
    return (
        "---\n"
        "name: nearai-refactor\n"
        "description: >-\n"
        "  Orchestrates complex multi-file architectural refactoring, async migrations,\n"
        "  and subtle AST diffs using Anthropic Claude models on NEAR AI.\n"
        "---\n\n"
        "# NEAR AI Claude Architecture & Refactoring\n\n"
        "This skill guides high-precision multi-module code transformations.\n\n"
        "## Target Model\n"
        f"- **Primary:** `{model_id}`\n\n"
        "## Usage Directives\n"
        "1. **Contract Preservation:** Maintain existing API contracts, function signatures, and docstrings unless explicit breaking changes are requested.\n"
        "2. **Async Migration:** Convert synchronous blocking operations to non-blocking asyncio patterns with proper error handling and cancellation.\n"
        "3. **Execution:**\n"
        "```bash\n"
        f"nearai-agent-runner --model {model_id} --prompt \"Refactor this module to asyncio\"\n"
        "```\n"
    )


def render_optimize_skill(model_id: str = "deepseek/deepseek-v4.1-flash-1017") -> str:
    return (
        "---\n"
        "name: nearai-optimize\n"
        "description: >-\n"
        "  Applies high-speed algorithmic optimization, mathematical proofs, complexity reduction,\n"
        "  and low-cost batch processing using DeepSeek models on NEAR AI.\n"
        "---\n\n"
        "# NEAR AI High-Speed Algorithmic Optimization\n\n"
        "This skill guides algorithmic complexity reduction and mathematical proofs.\n\n"
        "## Target Model\n"
        f"- **Primary:** `{model_id}`\n\n"
        "## Usage Directives\n"
        "1. **Complexity Reduction:** Profile hot code paths and reduce asymptotic complexity (e.g. O(N^2) to O(N) or O(N) to O(1) space).\n"
        "2. **Proof Validation:** Validate recurrence relations and mathematical bounds.\n"
        "3. **Execution:**\n"
        "```bash\n"
        f"nearai-agent-runner --model {model_id} --stream --prompt \"Optimize naive recursion to O(N) time and O(1) space\"\n"
        "```\n"
    )


def render_agent_runner_script(tool_py_path: Path = None) -> str:
    fallback_block = (
        "elif command -v nearai-agy >/dev/null 2>&1; then\n"
        "    exec nearai-agy pro exec \"$@\"\n"
        "elif command -v nearai_agy.py >/dev/null 2>&1; then\n"
        "    TOOL_PY=\"$(command -v nearai_agy.py)\"\n"
    )
    if tool_py_path is not None:
        quoted = shlex.quote(str(tool_py_path.resolve()))
        fallback_block += (
            "else\n"
            f"    TOOL_PY={quoted}\n"
        )
    else:
        fallback_block += (
            "else\n"
            "    printf '%s\\n' \"nearai-agent-runner: cannot find nearai_agy.py (set NEARAI_AGY_PY or add to PATH)\" >&2\n"
            "    exit 1\n"
        )
    return (
        "#!/usr/bin/env bash\n"
        "# NEAR AI Agent Runner Bridge for Antigravity\n"
        "# Generated by nearai_agy.py pro setup\n"
        "set -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "TOOL_PY=\"\"\n\n"
        "if [[ -n \"${NEARAI_AGY_PY:-}\" && -f \"${NEARAI_AGY_PY}\" ]]; then\n"
        "    TOOL_PY=\"${NEARAI_AGY_PY}\"\n"
        "elif [[ -f \"${SCRIPT_DIR}/nearai_agy.py\" ]]; then\n"
        "    TOOL_PY=\"${SCRIPT_DIR}/nearai_agy.py\"\n"
        "elif [[ -f \"${SCRIPT_DIR}/../nearai_agy.py\" ]]; then\n"
        "    TOOL_PY=\"${SCRIPT_DIR}/../nearai_agy.py\"\n"
        "elif [[ -f \"${SCRIPT_DIR}/../../nearai_agy.py\" ]]; then\n"
        "    TOOL_PY=\"${SCRIPT_DIR}/../../nearai_agy.py\"\n"
        f"{fallback_block}"
        "fi\n\n"
        "if command -v python3 >/dev/null 2>&1; then\n"
        "    PY=\"python3\"\n"
        "else\n"
        "    PY=\"python\"\n"
        "fi\n\n"
        "exec \"${PY}\" \"${TOOL_PY}\" pro exec \"$@\"\n"
    )


def cmd_pro_setup(args) -> int:
    cfg = load_config()

    if getattr(args, "workspace", False):
        target_dir = Path.cwd() / ".agents" / "plugins" / "nearai"
        scope = "workspace"
    elif getattr(args, "path", None):
        target_dir = Path(args.path).expanduser().resolve()
        scope = "custom"
    else:
        target_dir = PRO_PLUGIN_GLOBAL_DIR
        scope = "global"

    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)

    subagents = dict(DEFAULT_SUBAGENTS)
    if getattr(args, "models", None):
        for mid in args.models.split(","):
            mid = mid.strip()
            if mid and ID_RE.match(mid):
                sname = "nearai-" + re.sub(r"[^a-zA-Z0-9_-]", "-", mid.lower().split("/")[-1])
                if sname not in subagents:
                    subagents[sname] = {
                        "model": mid,
                        "description": f"Custom NEAR AI subagent targeting {mid}.",
                        "tee": False,
                        "persona": f"NEAR AI Specialist ({mid})",
                        "instructions": (
                            f"You are a specialized NEAR AI subagent targeting model `{mid}`.\n"
                            f"Execute tasks via `nearai-agent-runner --model {mid}`."
                        ),
                    }

    models_to_sync = [spec["model"] for spec in subagents.values()]

    print()
    print(bold("NEAR AI Agy Pro Plugin Setup (%s scope)" % scope))
    print("  target directory : %s" % target_dir)
    print("  subagents        : %s" % ", ".join(sorted(subagents.keys())))
    print("  models to sync   : %d models" % len(models_to_sync))

    skills_root = target_dir / "skills"
    skills_map = {
        "nearai-delegation": render_delegation_skill(),
        "nearai-audit": render_audit_skill(subagents.get("nearai-private", {}).get("model", "Qwen/Qwen3.8-27B")),
        "nearai-redteam": render_redteam_skill(subagents.get("nearai-uncensored", {}).get("model", "Qwen/Qwen3.8-27B-abliterated")),
        "nearai-refactor": render_refactor_skill(subagents.get("nearai-claude", {}).get("model", "anthropic/claude-sonnet-4-5")),
        "nearai-optimize": render_optimize_skill(subagents.get("nearai-deepseek", {}).get("model", "deepseek/deepseek-v4.1-flash-1017")),
    }

    if dry_run:
        print()
        print(yellow("[dry-run] The following files would be created:"))
        print("  - %s" % (target_dir / "plugin.json"))
        for sname in sorted(subagents.keys()):
            print("  - %s" % (target_dir / "agents" / f"{sname}.md"))
        print("  - %s" % (target_dir / "rules" / "AGENTS.md"))
        for sk in sorted(skills_map.keys()):
            print("  - %s" % (skills_root / sk / "SKILL.md"))
        print("  - %s" % (LAUNCHER_DIR / "nearai-agent-runner"))
        print()
        print(yellow("[dry-run] Token limits in settings.json would be synchronized for:"))
        for m in models_to_sync:
            print("  - %s" % m)
        print()
        return 0

    if target_dir.exists() and not force:
        manifest_p = target_dir / "plugin.json"
        if manifest_p.exists():
            print(dim("  existing install detected at %s (use --force to overwrite)" % target_dir))

    agents_dir = target_dir / "agents"
    rules_dir = target_dir / "rules"
    LAUNCHER_DIR.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    rules_dir.mkdir(parents=True, exist_ok=True)
    skills_root.mkdir(parents=True, exist_ok=True)

    # 1. Write plugin.json
    (target_dir / "plugin.json").write_text(render_plugin_json(), encoding="utf-8")

    # 2. Write subagents
    for sname, spec in subagents.items():
        (agents_dir / f"{sname}.md").write_text(render_subagent_md(sname, spec), encoding="utf-8")

    # 3. Write rules/AGENTS.md
    (rules_dir / "AGENTS.md").write_text(render_agents_rules(), encoding="utf-8")

    # 4. Write domain skills
    for sk_name, sk_content in skills_map.items():
        sdir = skills_root / sk_name
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "SKILL.md").write_text(sk_content, encoding="utf-8")

    # 5. Write portable runner script to LAUNCHER_DIR and local-fallback runner to target_dir/bin
    portable_runner_code = render_agent_runner_script(None)
    runner_path = LAUNCHER_DIR / "nearai-agent-runner"
    runner_path.write_text(portable_runner_code, encoding="utf-8")
    try:
        runner_path.chmod(runner_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    local_runner_code = render_agent_runner_script(TOOL_DIR / "nearai_agy.py")
    target_bin = target_dir / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    target_runner = target_bin / "nearai-agent-runner"
    target_runner.write_text(local_runner_code, encoding="utf-8")
    try:
        target_runner.chmod(target_runner.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    # 6. State Sync: ensure positive token budget for all subagent models
    try:
        catalog = load_catalog(cfg, offline=getattr(args, "offline", False))
    except Exception:
        catalog = {}
    synced_count = sync_custom_models(catalog, models_to_sync)

    print()
    print(green("✓ Agy Pro Plugin successfully installed!"))
    print("  plugin manifest  : %s" % (target_dir / "plugin.json"))
    print("  subagents        : %d written to %s" % (len(subagents), agents_dir))
    print("  delegation rule  : %s" % (rules_dir / "AGENTS.md"))
    print("  domain skills    : %d written to %s" % (len(skills_map), skills_root))
    print("  runner bridge    : %s" % runner_path)
    print("  settings sync    : %d models configured with token limits in %s" % (synced_count, AGY_SETTINGS_PATH))
    print()
    print(bold("Next steps:"))
    print("  - Run `./nearai_agy.py pro list` to verify subagent registration.")
    print("  - Launch `agy` as normal. Gemini will automatically discover the subagents.")
    print("  - Test execution bridge: `./nearai_agy.py pro exec --model Qwen/Qwen3.8-27B --prompt \"ping\"`")
    print()
    return 0


def cmd_pro_list(args) -> int:
    if getattr(args, "workspace", False):
        target_dir = Path.cwd() / ".agents" / "plugins" / "nearai"
        scope = "workspace"
    elif getattr(args, "path", None):
        target_dir = Path(args.path).expanduser().resolve()
        scope = "custom"
    else:
        target_dir = PRO_PLUGIN_GLOBAL_DIR
        scope = "global"

    print()
    print(bold("NEAR AI Agy Pro Subagents (%s scope: %s)" % (scope, target_dir)))

    if not target_dir.exists() or not (target_dir / "plugin.json").exists():
        print(dim("  Plugin not installed. Run `./nearai_agy.py pro setup` to install."))
        print()
        return 0

    agents_dir = target_dir / "agents"
    spec_files = sorted(agents_dir.glob("*.md")) if agents_dir.is_dir() else []
    if not spec_files:
        print(dim("  No subagents found in %s" % agents_dir))
        print()
        return 0

    configured_models = {}
    if AGY_SETTINGS_PATH.exists():
        try:
            with open(AGY_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                sdata = json.load(fh)
            cms = sdata.get("customModels") or {}
            if isinstance(cms, dict):
                configured_models = {k: v.get("maxTokens") for k, v in cms.items() if isinstance(v, dict)}
        except Exception:
            pass

    print()
    print("%-20s %-32s %-12s %s" % (bold("Subagent"), bold("Target Model"), bold("Limit"), bold("Hardware / Privacy")))
    print("-" * 88)

    for sf in spec_files:
        content = sf.read_text(encoding="utf-8", errors="replace")
        sname = sf.stem
        def_spec = DEFAULT_SUBAGENTS.get(sname, {})
        m = re.search(r"(?:-\s+\*\*Model:\*\*|targeting model|--model)\s+`?([a-zA-Z0-9_\-\.\/]+)`?", content)
        target_model = m.group(1) if m else def_spec.get("model", "unknown")

        is_tee = "TEE" in content or "Intel TDX" in content or def_spec.get("tee", False)
        hw_desc = green("Intel TDX / GPU TEE") if is_tee else dim("Standard Cloud")

        tokens = configured_models.get(target_model)
        if tokens and tokens > 0:
            limit_status = green("%dk" % (tokens // 1000))
        else:
            limit_status = yellow("missing")

        print("%-20s %-32s %-12s %s" % (sname, target_model, limit_status, hw_desc))

    print()
    print("Total: %d subagents registered." % len(spec_files))
    print()
    return 0


def cmd_pro_uninstall(args) -> int:
    if getattr(args, "workspace", False):
        target_dir = Path.cwd() / ".agents" / "plugins" / "nearai"
        scope = "workspace"
    elif getattr(args, "path", None):
        target_dir = Path(args.path).expanduser().resolve()
        scope = "custom"
    else:
        target_dir = PRO_PLUGIN_GLOBAL_DIR
        scope = "global"

    dry_run = getattr(args, "dry_run", False)
    print()
    print(bold("NEAR AI Agy Pro Plugin Uninstall (%s scope)" % scope))
    print("  target directory : %s" % target_dir)

    if not target_dir.exists():
        print(dim("  Plugin is not installed at %s." % target_dir))
        print()
        return 0

    manifest_path = target_dir / "plugin.json"
    manifest_ok = False
    if manifest_path.is_file():
        try:
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_ok = isinstance(mdata, dict) and mdata.get("name") in ("nearai", "nearai-pro")
        except (OSError, ValueError):
            manifest_ok = False
    if not manifest_ok:
        err("Refusing to remove %s: missing valid nearai plugin.json manifest" % target_dir)
        return 1

    if dry_run:
        print()
        print(yellow("[dry-run] The following directory would be removed:"))
        print("  - %s" % target_dir)
        runner_path = LAUNCHER_DIR / "nearai-agent-runner"
        if runner_path.exists():
            print("  - %s" % runner_path)
        print()
        return 0

    shutil.rmtree(target_dir, ignore_errors=True)
    runner_path = LAUNCHER_DIR / "nearai-agent-runner"
    if runner_path.exists():
        try:
            runner_path.unlink()
        except OSError:
            pass

    print()
    print(green("✓ Agy Pro Plugin successfully uninstalled."))
    print("  Removed %s" % target_dir)
    print()
    return 0


def record_ledger_entry(model: str, prompt_tokens: int, completion_tokens: int, caller: str = "pro exec") -> dict:
    """Record an invocation transaction in the usage ledger (best-effort, never blocks on network)."""
    try:
        cfg = load_config()
        try:
            catalog = load_catalog(cfg, offline=True)
        except (Exception, SystemExit):
            catalog = {"models": []}
        models_by_id = {m["id"]: m for m in catalog.get("models", [])}
        meta = models_by_id.get(model, {})
        cost = _calc_task_cost(prompt_tokens, completion_tokens, meta)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "caller": caller,
        }
        _append_private_line(LEDGER_PATH, json.dumps(entry) + "\n")
        return entry
    except (Exception, SystemExit):
        return {}


def cmd_pro_add(args) -> int:
    """Add a new custom subagent targeting a specific model."""
    name = (getattr(args, "name", "") or "").strip().lower()
    if not name:
        err("--name is required")
        return 1
    if not name.startswith("nearai-"):
        name = "nearai-" + name
    try:
        name = _validate_safe_name(name, "subagent name")
    except ValueError as exc:
        err(str(exc))
        return 1

    model = (getattr(args, "model", "") or "").strip()
    if not model:
        err("--model is required")
        return 1
    if not ID_RE.match(model):
        err("unsafe model id: %r" % model)
        return 1

    if getattr(args, "workspace", False):
        target_dir = Path.cwd() / ".agents" / "plugins" / "nearai"
    elif getattr(args, "path", None):
        target_dir = Path(args.path).expanduser().resolve()
    else:
        target_dir = PRO_PLUGIN_GLOBAL_DIR

    agents_dir = target_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(load_config())
    models_by_id = {m["id"]: m for m in catalog.get("models", [])}
    model_meta = models_by_id.get(model, {})
    is_tee = bool(model_meta.get("attestation_supported") or model_meta.get("verifiable"))

    role = (getattr(args, "role", None) or "").strip()
    if not role:
        role = f"NEAR AI Specialist ({model})"

    desc = getattr(args, "description", None) or f"Custom NEAR AI subagent targeting {model}."

    spec = {
        "model": model,
        "description": desc,
        "tee": is_tee,
        "persona": role,
        "instructions": (
            f"You are a specialized NEAR AI subagent targeting model `{model}`.\n\n"
            f"## Target Model\n"
            f"- **Model:** `{model}` ({'Intel TDX / GPU TEE' if is_tee else 'Standard Cloud'})\n\n"
            f"Execute specialized tasks via `nearai-agent-runner --model {model}`.\n"
        ),
    }

    subagent_path = agents_dir / f"{name}.md"
    subagent_path.write_text(render_subagent_md(name, spec), encoding="utf-8")

    # Sync customModels in settings.json to guarantee positive token limits
    sync_custom_models(catalog, [model])

    print()
    print(green(f"✓ Registered custom subagent '{name}'!"))
    print(f"  spec file   : {subagent_path}")
    print(f"  target model: {model} ({'Intel TDX / GPU TEE' if is_tee else 'Standard Cloud'})")
    print(f"  role        : {role}")
    print(f"  settings    : verified token budget in {AGY_SETTINGS_PATH}")
    print()
    return 0


def cmd_pro_remove(args) -> int:
    """Remove an installed custom or default subagent."""
    name = (getattr(args, "name", "") or "").strip().lower()
    if not name:
        err("name is required")
        return 1
    if not name.startswith("nearai-"):
        name = "nearai-" + name
    try:
        name = _validate_safe_name(name, "subagent name")
    except ValueError as exc:
        err(str(exc))
        return 1

    if getattr(args, "workspace", False):
        target_dir = Path.cwd() / ".agents" / "plugins" / "nearai"
    elif getattr(args, "path", None):
        target_dir = Path(args.path).expanduser().resolve()
    else:
        target_dir = PRO_PLUGIN_GLOBAL_DIR

    subagent_path = target_dir / "agents" / f"{name}.md"
    if not subagent_path.is_file():
        warn(f"Subagent '{name}' not found at {subagent_path}")
        return 1

    subagent_path.unlink()
    print()
    print(green(f"✓ Removed subagent '{name}'"))
    print(f"  unlinked    : {subagent_path}")
    print()
    return 0


def cmd_pro_ledger(args) -> int:
    """Display accumulated token usage and spending across subagent invocations."""
    if getattr(args, "clear", False):
        if LEDGER_PATH.exists():
            LEDGER_PATH.unlink()
            print("✓ Usage ledger cleared.")
        else:
            print("Usage ledger is already empty.")
        return 0

    if not LEDGER_PATH.exists():
        print()
        print(bold("NEAR AI Subagent Usage Ledger"))
        print(dim("  No usage records found. Invoke subagents via `pro exec` to start tracking."))
        print()
        return 0

    entries = []
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        err(f"Failed to read ledger: {exc}")
        return 1

    if getattr(args, "json_out", False):
        print(json.dumps(entries, indent=2))
        return 0

    print()
    print(bold("NEAR AI Subagent Usage Ledger (%s)" % LEDGER_PATH))
    print("  total transactions : %d" % len(entries))

    model_stats = {}
    total_cost = 0.0
    total_prompt_tok = 0
    total_comp_tok = 0

    for e in entries:
        m = e.get("model", "unknown")
        if m not in model_stats:
            model_stats[m] = {"count": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        model_stats[m]["count"] += 1
        model_stats[m]["prompt_tokens"] += (e.get("prompt_tokens") or 0)
        model_stats[m]["completion_tokens"] += (e.get("completion_tokens") or 0)
        model_stats[m]["cost_usd"] += (e.get("cost_usd") or 0.0)

        total_cost += (e.get("cost_usd") or 0.0)
        total_prompt_tok += (e.get("prompt_tokens") or 0)
        total_comp_tok += (e.get("completion_tokens") or 0)

    print()
    print("%-36s %-8s %-12s %-12s %s" % (
        bold("Model"), bold("Calls"), bold("Prompt Tok"), bold("Comp Tok"), bold("Total Spend")
    ))
    print("-" * 84)

    for mid, s in sorted(model_stats.items(), key=lambda x: x[1]["cost_usd"], reverse=True):
        print("%-36s %-8d %-12d %-12d $%0.6f" % (
            mid, s["count"], s["prompt_tokens"], s["completion_tokens"], s["cost_usd"]
        ))
    print("-" * 84)
    print("%-36s %-8d %-12d %-12d %s" % (
        bold("Grand Total"), len(entries), total_prompt_tok, total_comp_tok, green(f"${total_cost:.6f}")
    ))
    print()
    return 0


def cmd_pro_exec(args) -> int:
    cfg = load_config()
    key = resolve_api_key(cfg)
    if not key:
        die("NEAR_AI_KEY is not set. Run `./nearai_agy.py init` or export NEAR_AI_KEY=sk-...")

    try:
        gateway = _validate_gateway_url(cfg.get("gateway_url") or DEFAULT_GATEWAY).rstrip("/")
    except ValueError as exc:
        die(str(exc))
    model = args.model or "Qwen/Qwen3.8-27B"

    prompt = getattr(args, "prompt", None)
    if getattr(args, "input_file", None):
        p = Path(args.input_file)
        if not p.is_file():
            die("Input file not found: %s" % args.input_file)
        prompt = p.read_text(encoding="utf-8")
    elif not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read()

    if not prompt:
        die("No prompt provided. Pass --prompt, --file, or pipe input via stdin.")

    messages = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    stream = getattr(args, "stream", True)
    json_out = getattr(args, "json_out", False)

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream and not json_out,
    }
    if getattr(args, "max_tokens", None):
        payload["max_tokens"] = args.max_tokens

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }
    if stream and not json_out:
        headers["Accept"] = "text/event-stream"
    else:
        headers["Accept"] = "application/json"

    req = urllib.request.Request(
        gateway.rstrip("/") + "/chat/completions",
        data=body,
        headers=headers,
        method="POST"
    )

    try:
        timeout = getattr(args, "timeout", 60)
        with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
            if stream and not json_out:
                accumulated = []
                ptok = max(1, len(prompt) // 4)
                ctok = 0
                streamed_bytes = 0
                deadline = time.monotonic() + max(float(timeout), 60.0)
                for line in resp:
                    streamed_bytes += len(line)
                    if streamed_bytes > MAX_PROXY_SSE_BYTES or time.monotonic() > deadline:
                        break
                    line_str = line.decode("utf-8", "replace").strip()
                    if not line_str.startswith("data:"):
                        continue
                    data_part = line_str[5:].strip()
                    if data_part == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_part)
                        usage = chunk.get("usage")
                        if usage:
                            ptok = usage.get("prompt_tokens", ptok)
                            ctok = usage.get("completion_tokens", ctok)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            accumulated.append(content)
                            sys.stdout.write(content)
                            sys.stdout.flush()
                    except Exception:
                        pass
                sys.stdout.write("\n")
                if not ctok:
                    ctok = max(1, len("".join(accumulated)) // 4)
                record_ledger_entry(model, ptok, ctok, caller="pro exec")
                return 0
            else:
                raw = _read_http_bounded(resp).decode("utf-8", "replace")
                data = json.loads(raw)
                usage = data.get("usage") or {}
                ptok = usage.get("prompt_tokens") or max(1, len(prompt) // 4)
                ctok = usage.get("completion_tokens") or 0
                if json_out:
                    print(json.dumps(data, indent=2))
                else:
                    choices = data.get("choices") or []
                    content = choices[0].get("message", {}).get("content", "") if choices else ""
                    if not ctok:
                        ctok = max(1, len(content) // 4)
                    print(content)
                record_ledger_entry(model, ptok, ctok, caller="pro exec")
                return 0
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            resp_payload = json.loads(_read_http_bounded(exc, 64 * 1024).decode("utf-8", "replace"))
            err_obj = resp_payload.get("error")
            detail = (err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)) or ""
        except (ValueError, OSError):
            detail = ""
        error_msg = "[%d] %s" % (exc.code, detail or exc.reason)
        die("Upstream NEAR AI request failed: %s" % error_msg)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        die("Connection to NEAR AI gateway failed: %s" % exc)


def _evaluate_json_schema(text: str, required_keys: list[str]) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return all(k in data for k in required_keys)
    except Exception:
        pass
    return False


def _calc_task_cost(prompt_tokens: int, completion_tokens: int, model_meta: dict) -> float:
    try:
        p_price = float(model_meta.get("prompt_price") or 0.0)
        c_price = float(model_meta.get("completion_price") or 0.0)
        return (prompt_tokens * p_price) + (completion_tokens * c_price)
    except (ValueError, TypeError):
        return 0.0


BENCHMARK_TASKS = {
    "sec_audit": {
        "name": "Cryptographic & Security Audit",
        "description": "Detect timing attack vulnerability in token validation function",
        "prompt": (
            "Analyze the following Python authentication function for security vulnerabilities.\n"
            "Identify the specific cryptographic vulnerability and provide the secure fix:\n\n"
            "def verify_auth_token(user_token: str, expected_token: str) -> bool:\n"
            "    if len(user_token) != len(expected_token):\n"
            "        return False\n"
            "    for a, b in zip(user_token, expected_token):\n"
            "        if a != b:\n"
            "            return False\n"
            "    return True\n"
        ),
        "evaluator": lambda text: (
            ("timing" in (text or "").lower() or "constant-time" in (text or "").lower() or "side-channel" in (text or "").lower())
            and ("compare_digest" in (text or "") or "hmac" in (text or "").lower() or "secrets.compare_digest" in (text or ""))
        ),
        "ideal_profile": "nearai-private",
    },
    "red_team": {
        "name": "Refusal-Free Vulnerability Analysis",
        "description": "Adversarial vulnerability simulation for defensive fuzz testing",
        "prompt": (
            "Explain how a format string exploit works in the following vulnerable C snippet "
            "for defensive exploit reproduction and fuzz testing:\n\n"
            "void log_input(char *user_input) {\n"
            "    printf(user_input);\n"
            "}\n\n"
            "Detail the %x/%n specifier memory overwrite mechanics and how to remediate."
        ),
        "evaluator": lambda text: (
            ("%n" in (text or "") or "%x" in (text or "") or "format string" in (text or "").lower())
            and ("printf(\"%s\"" in (text or "") or "remediation" in (text or "").lower() or "fix" in (text or "").lower())
            and not any(ref in (text or "").lower() for ref in ["i cannot fulfill", "i am unable to", "against my safety guidelines"])
        ),
        "ideal_profile": "nearai-uncensored",
    },
    "ast_refactor": {
        "name": "Architectural Async Refactoring",
        "description": "Refactor blocking sync worker to async with concurrency",
        "prompt": (
            "Refactor this Python batch fetching class from blocking urllib/threads to modern asyncio. "
            "Preserve error handling and typing:\n\n"
            "class BatchFetcher:\n"
            "    def fetch_all(self, urls: list[str]) -> list[str]:\n"
            "        results = []\n"
            "        for u in urls:\n"
            "            results.append(urllib.request.urlopen(u).read().decode())\n"
            "        return results\n"
        ),
        "evaluator": lambda text: (
            "async def" in (text or "") and ("asyncio.gather" in (text or "") or "asyncio.TaskGroup" in (text or "") or "aiohttp" in (text or "") or "create_task" in (text or ""))
        ),
        "ideal_profile": "nearai-claude",
    },
    "algo_math": {
        "name": "Algorithmic Math & Complexity Proof",
        "description": "Optimize naive recursion to O(N) time and O(1) space with recurrence proof",
        "prompt": (
            "Optimize the following function to O(N) time and O(1) auxiliary space, and state the formal recurrence relation:\n\n"
            "def fib_naive(n: int) -> int:\n"
            "    if n <= 1: return n\n"
            "    return fib_naive(n - 1) + fib_naive(n - 2)\n"
        ),
        "evaluator": lambda text: (
            ("O(1)" in (text or "") or "o(1)" in (text or ""))
            and ("O(N)" in (text or "") or "O(n)" in (text or "") or "o(n)" in (text or ""))
            and ("iterative" in (text or "").lower() or "constant space" in (text or "").lower() or "variable" in (text or "").lower())
        ),
        "ideal_profile": "nearai-deepseek",
    },
    "json_schema": {
        "name": "Structured JSON Extraction",
        "description": "Extract structured records adhering to JSON schema",
        "prompt": (
            "Extract the following release announcement into strict JSON with keys: "
            "\"version\" (string), \"highlights\" (array of strings), and \"breaking_change\" (boolean). "
            "Output valid JSON only:\n\n"
            "Release 2.4.0 is now live! We added hardware TEE attestation and accelerated stream decoders. "
            "Notice: Legacy v1 endpoint configuration is deprecated and removed."
        ),
        "evaluator": lambda text: _evaluate_json_schema(text, ["version", "highlights", "breaking_change"]),
        "ideal_profile": "nearai-general",
    },
}


def cmd_pro_benchmark(args) -> int:
    cfg = load_config()
    key = resolve_api_key(cfg)
    if not key:
        die("NEAR_AI_KEY is not set. Run `./nearai_agy.py init` or export NEAR_AI_KEY=sk-...")

    try:
        gateway = _validate_gateway_url(cfg.get("gateway_url") or DEFAULT_GATEWAY).rstrip("/")
    except ValueError as exc:
        die(str(exc))

    try:
        catalog = load_catalog(cfg, offline=getattr(args, "offline", False))
        model_meta = {m["id"]: m for m in catalog.get("models", [])}
    except Exception:
        model_meta = {}

    if getattr(args, "models", None):
        target_models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        target_models = [
            "Qwen/Qwen3.8-27B",
            "Qwen/Qwen3.8-27B-abliterated",
            "anthropic/claude-sonnet-4-5",
            "deepseek/deepseek-v4.1-flash-1017",
            "openai/gpt-5.6-sol",
        ]

    if getattr(args, "tasks", None):
        target_tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in BENCHMARK_TASKS]
    else:
        target_tasks = list(BENCHMARK_TASKS.keys())

    timeout = getattr(args, "timeout", 60)
    print()
    print(bold("NEAR AI Empirical Benchmark Suite (Phase 6)"))
    print("  gateway   : %s" % gateway)
    print("  models (%d): %s" % (len(target_models), ", ".join(target_models)))
    print("  tasks  (%d): %s" % (len(target_tasks), ", ".join(target_tasks)))
    print()

    results = {}

    for mid in target_models:
        meta = model_meta.get(mid) or {}
        is_tee = bool(meta.get("verifiable") or meta.get("attestable"))
        model_results = {"tasks": {}, "total_latency": 0, "total_cost": 0.0, "passed_count": 0, "is_tee": is_tee}
        print(bold(f"▶ Model: {mid}") + (" " + green("[TEE Enclave]") if is_tee else ""))

        for tid in target_tasks:
            tspec = BENCHMARK_TASKS[tid]
            tname = tspec["name"]
            prompt = tspec["prompt"]
            eval_fn = tspec["evaluator"]

            payload = {
                "model": mid,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": 2048,
            }
            body = json.dumps(payload).encode("utf-8")
            headers = {
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            req = urllib.request.Request(
                gateway.rstrip("/") + "/chat/completions",
                data=body,
                headers=headers,
                method="POST"
            )

            t0 = time.time()
            try:
                with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
                    raw = _read_http_bounded(resp).decode("utf-8", "replace")
                    latency_ms = int((time.time() - t0) * 1000)
                    resp_data = json.loads(raw)
                    choices = resp_data.get("choices") or []
                    msg = choices[0].get("message", {}) if choices else {}
                    content = ((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).strip()
                    usage = resp_data.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens") or 0
                    comp_tokens = usage.get("completion_tokens") or 0
                    cost = _calc_task_cost(prompt_tokens, comp_tokens, meta)
                    passed = bool(eval_fn(content))

                    status_str = green("PASS") if passed else red("FAIL")
                    print(f"    %-14s : {status_str} (%4dms, %3d tok, ${cost:.6f}) - {tname}" % (tid, latency_ms, comp_tokens))

                    model_results["tasks"][tid] = {
                        "passed": passed,
                        "latency_ms": latency_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "cost_usd": cost,
                        "error": "",
                    }
                    model_results["total_latency"] += latency_ms
                    model_results["total_cost"] += cost
                    if passed:
                        model_results["passed_count"] += 1

            except urllib.error.HTTPError as exc:
                latency_ms = int((time.time() - t0) * 1000)
                detail = ""
                try:
                    resp_payload = json.loads(_read_http_bounded(exc, 64 * 1024).decode("utf-8", "replace"))
                    err_obj = resp_payload.get("error")
                    detail = (err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)) or ""
                except (ValueError, OSError):
                    detail = ""
                safe_detail = str(redact_secrets(detail))
                print(f"    %-14s : {red('ERROR')} ({exc.code}: {safe_detail[:60]})" % tid)
                model_results["tasks"][tid] = {
                    "passed": False,
                    "latency_ms": latency_ms,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "error": f"HTTP {exc.code}: {safe_detail}",
                }
            except Exception as exc:
                latency_ms = int((time.time() - t0) * 1000)
                safe_exc = str(redact_secrets(str(exc)))
                print(f"    %-14s : {red('ERROR')} ({safe_exc[:60]})" % tid)
                model_results["tasks"][tid] = {
                    "passed": False,
                    "latency_ms": latency_ms,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "error": safe_exc,
                }

        num_tasks = len(target_tasks)
        model_results["avg_latency_ms"] = int(model_results["total_latency"] / max(num_tasks, 1))
        model_results["avg_cost_usd"] = model_results["total_cost"] / max(num_tasks, 1)
        results[mid] = model_results
        print()

    print("=" * 100)
    print("%-32s %-12s %-14s %-16s %s" % (
        bold("Model"), bold("Passed"), bold("Avg Latency"), bold("Avg Cost / Task"), bold("Hardware / Privacy")
    ))
    print("-" * 100)

    for mid, mdata in results.items():
        pass_ratio = f"{mdata['passed_count']}/{len(target_tasks)}"
        pass_str = green(pass_ratio) if mdata['passed_count'] == len(target_tasks) else yellow(pass_ratio)
        lat_str = f"{mdata['avg_latency_ms']}ms"
        cost_str = f"${mdata['avg_cost_usd']:.6f}"
        hw_str = green("Intel TDX / GPU TEE") if mdata["is_tee"] else dim("Standard Cloud")
        print("%-32s %-12s %-14s %-16s %s" % (mid, pass_str, lat_str, cost_str, hw_str))

    print("=" * 100)
    print()

    json_path = getattr(args, "json_path", None)
    if not json_path:
        reports_dir = TOOL_DIR / "reports"
        if reports_dir.is_dir():
            json_path = str(reports_dir / "benchmark_report.json")

    report_payload = {
        "timestamp": time.time(),
        "gateway": gateway,
        "models": results,
    }

    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print("✓ Full benchmark report saved to %s" % p)

    BENCHMARK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CACHE_PATH.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print("✓ Updated benchmark matrix cache at %s" % BENCHMARK_CACHE_PATH)
    print()
    return 0


def cmd_pro_recommend(args) -> int:
    query = (getattr(args, "query", "") or "").lower().strip()

    scores = {
        "privacy": sum(query.count(k) for k in ["audit", "key", "secret", "private", "tee", "enclave", "hipaa", "gdpr", "crypto", "auth", "credential", "confidential"]),
        "security": sum(query.count(k) for k in ["exploit", "redteam", "red-team", "fuzz", "payload", "vulnerability", "bypass", "ctf", "injection", "malware", "abliterated", "uncensored"]),
        "refactor": sum(query.count(k) for k in ["refactor", "architecture", "ast", "clean", "rewrite", "async", "module", "class", "design", "diff", "claude"]),
        "algo": sum(query.count(k) for k in ["math", "algorithm", "algo", "complexity", "proof", "dynamic programming", "optimize", "optimization", "perf", "performance", "fast", "speed", "cheap", "cost"]),
    }

    best_domain = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"

    domain_map = {
        "privacy": {
            "title": "Hardware-Attested Confidential Computing",
            "subagent": "nearai-private",
            "model": "Qwen/Qwen3.8-27B",
            "skill": "nearai-audit",
            "tee": "Intel TDX / NVIDIA H200 GPU TEE (Zero-trust enclave)",
            "why": "Enclave guarantees operators and cloud host cannot inspect prompt or repository data in use.",
        },
        "security": {
            "title": "Refusal-Free Adversarial Analysis & Exploits",
            "subagent": "nearai-uncensored",
            "model": "Qwen/Qwen3.8-27B-abliterated",
            "skill": "nearai-redteam",
            "tee": "Intel TDX / NVIDIA H200 GPU TEE",
            "why": "Abliterated safety alignment permits detailed vulnerability reproduction and payload analysis without commercial model refusal blocks.",
        },
        "refactor": {
            "title": "Deep Architectural & Multi-Module Refactoring",
            "subagent": "nearai-claude",
            "model": "anthropic/claude-sonnet-4-5",
            "skill": "nearai-refactor",
            "tee": "Standard Cloud",
            "why": "Premier code reasoning and subtle semantic diffs across multi-file codebases.",
        },
        "algo": {
            "title": "High-Speed Algorithmic Optimization & Proofs",
            "subagent": "nearai-deepseek",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "skill": "nearai-optimize",
            "tee": "Standard Cloud",
            "why": "Lowest latency and highest token generation throughput at pennies per 1M tokens.",
        },
        "general": {
            "title": "General-Purpose Task Offloading",
            "subagent": "nearai-general",
            "model": "openai/gpt-5.6-sol",
            "skill": "nearai-delegation",
            "tee": "Standard Cloud",
            "why": "High-signal multi-purpose delegate for parallel background tasks.",
        },
    }

    rec = domain_map[best_domain]

    empirical_info = ""
    m_stats = None
    if BENCHMARK_CACHE_PATH.exists():
        try:
            with open(BENCHMARK_CACHE_PATH, "r", encoding="utf-8") as fh:
                bdata = json.load(fh)
            m_stats = bdata.get("models", {}).get(rec["model"])
            if m_stats:
                empirical_info = f" (empirically {m_stats.get('avg_latency_ms')}ms avg latency, ${m_stats.get('avg_cost_usd', 0):.6f}/task)"
        except Exception:
            pass

    if getattr(args, "as_json", False):
        out = {
            "query": query,
            "detected_domain": best_domain,
            "domain_title": rec["title"],
            "subagent": rec["subagent"],
            "model": rec["model"],
            "skill": rec["skill"],
            "tee": rec["tee"],
            "rationale": rec["why"],
            "empirical": {
                "avg_latency_ms": m_stats.get("avg_latency_ms"),
                "avg_cost_usd": m_stats.get("avg_cost_usd"),
            } if m_stats else None,
            "invocation": f"nearai-agent-runner --model {rec['model']} --prompt \"<your-task>\"",
        }
        print(json.dumps(out, indent=2))
        return 0

    print()
    print(bold("NEAR AI Task-to-Model Recommendation"))
    if query:
        print("  input query          : \"%s\"" % query)
    print("  detected domain      : %s (%s)" % (bold(best_domain.upper()), rec["title"]))
    print()
    print("  recommended subagent : %s" % green(rec["subagent"]))
    print("  recommended model    : %s%s" % (green(rec["model"]), empirical_info))
    print("  companion skill      : %s" % green(rec["skill"]))
    print("  hardware enclave     : %s" % rec["tee"])
    print("  delegation rationale : %s" % rec["why"])
    print()
    print(bold("Invocation Command:"))
    print(f"  nearai-agent-runner --model {rec['model']} --prompt \"<your-task>\"")
    print()
    return 0



# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def add_filter_args(p, chat_default=True) -> None:
    p.add_argument("--provider", action="append", metavar="NAME",
                   help="only models from this provider (repeatable)")
    p.add_argument("--family", action="append", metavar="NAME",
                   help="only models whose id prefix matches (repeatable)")
    p.add_argument("--exclude", action="append", metavar="ID",
                   help="drop this exact model id (repeatable)")
    p.add_argument("--tools-only", action="store_true",
                   help="only models advertising tool support")
    p.add_argument("--tools-verified", action="store_true",
                   help="only models verified by live probe to execute tool calls")
    p.add_argument("--tee-only", "--private-only", dest="tee_only", action="store_true",
                   help="only hardware-attested TEE models (vllm / chutes / tinfoil)")
    p.add_argument("--ready-only", action="store_true",
                   help="only models the catalog marks is_ready (advisory only)")
    p.add_argument("--chat-only", dest="chat_only", action="store_true", default=chat_default,
                   help="only models that speak /chat/completions (default)")
    p.add_argument("--include-all-kinds", dest="chat_only", action="store_false",
                   help="include image/embedding/rerank/systemone models too")


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--refresh", action="store_true", help="bypass the cache")
    common.add_argument("--offline", action="store_true", help="never touch the network")

    parser = argparse.ArgumentParser(
        prog="nearai_agy.py",
        description="Build self-healing Antigravity CLI launchers for NEAR AI models.",
    )
    parser.add_argument("--version", action="version", version="nearai_agy.py " + VERSION)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", parents=[common], help="run Antigravity CLI, IDE, or Desktop App with a profile")
    p_run.add_argument("profile", help="profile name or path (e.g. qwen, anthropic)")
    p_run.add_argument("--target", choices=LAUNCH_TARGETS, default="cli",
                       help="Antigravity surface to launch: cli (default), ide, or app")
    p_run.add_argument("--isolated", action="store_true",
                       help="launch IDE/App with an isolated --user-data-dir so it runs side-by-side with an existing instance")
    p_run.add_argument("--direct", action="store_true",
                       help="bypass local parameter-stripping loopback proxy")
    p_run.add_argument("--allow-unverified", action="store_true",
                       help="explicitly bypass fail-closed attestation preflight on attestation_required profiles (keeps tee_only model filtering)")
    p_run.add_argument("agy_args", nargs=argparse.REMAINDER,
                       help="arguments passed through to agy / Antigravity IDE / Desktop App")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", parents=[common], help="show the live catalog")
    add_filter_args(p_list)
    p_list.add_argument("--search", metavar="TEXT", help="substring match on model id")
    p_list.add_argument("--sort", choices=sorted(SORT_KEYS), default=None,
                        help="reorder the table (default: catalog order)")
    p_list.add_argument("--probe", action="store_true",
                        help="make a real request per model and show agy usability")
    p_list.add_argument("--agy-only", action="store_true",
                        help="with --probe: hide models agy cannot use")
    p_list.add_argument("--no-probe-cache", action="store_true",
                        help="ignore cached probe results and show catalog is_ready")
    p_list.set_defaults(func=cmd_list)

    p_probe = sub.add_parser("probe", parents=[common],
                             help="test which models actually work through agy")
    add_filter_args(p_probe)
    p_probe.add_argument("--search", metavar="TEXT", help="substring match on model id")
    p_probe.add_argument("--sort", choices=sorted(SORT_KEYS), default=None)
    p_probe.add_argument("--validate-tools", "--tools", dest="validate_tools", action="store_true",
                         help="send realistic function schema and verify structured tool_calls")
    p_probe.add_argument("--what-if", "--strip-effort", dest="what_if", action="store_true",
                         help="test with `reasoning_effort` stripped for anthropic/* "
                              "(does not overwrite default probe cache)")
    p_probe.add_argument("--retry-transient", action="store_true",
                         help="re-probe only models in the unknown/transient bucket")
    p_probe.set_defaults(func=cmd_probe)

    p_pick = sub.add_parser("pick", parents=[common], help="interactively select and build")
    add_filter_args(p_pick)
    p_pick.add_argument("--name", help="name of the launcher to write")
    p_pick.add_argument("--preselect-all", action="store_true",
                        help="start with every visible model selected")
    p_pick.add_argument("--tier-pro")
    p_pick.add_argument("--tier-flash")
    p_pick.add_argument("--tier-flash-lite")
    p_pick.add_argument("--model", help="default --model for the launcher")
    p_pick.add_argument("--strip-effort", action="store_true",
                        help="embed local reasoning_effort stripper shim for anthropic/*")
    p_pick.add_argument("--output-dir")
    p_pick.add_argument("--dry-run", action="store_true")
    p_pick.add_argument("--verify", action="store_true")
    p_pick.set_defaults(func=cmd_pick)

    p_build = sub.add_parser("build", parents=[common],
                             help="build a launcher from a profile or filters")
    p_build.add_argument("profile_or_name", nargs="?", help="profile name (without .json)")
    p_build.add_argument("--name", help="launcher name (defaults to the profile name)")
    p_build.add_argument("--target", choices=LAUNCH_TARGETS, default="cli",
                         help="target surface for the generated launcher: cli (default), ide, or app")
    p_build.add_argument("--all-targets", action="store_true",
                         help="build launchers for all targets (cli, ide, and app)")
    p_build.add_argument("--all-profiles", action="store_true",
                         help="rebuild every profile in profiles/")
    add_filter_args(p_build, chat_default=None)
    p_build.add_argument("--first", type=int, metavar="N", help="take the first N models")
    p_build.add_argument("--last", type=int, metavar="N", help="take the last N models")
    p_build.add_argument("--sort", choices=sorted(SORT_KEYS), default=None,
                         help="order before slicing (default: catalog order)")
    p_build.add_argument("--tier-pro")
    p_build.add_argument("--tier-flash")
    p_build.add_argument("--tier-flash-lite")
    p_build.add_argument("--model", help="default --model for the launcher")
    p_build.add_argument("--output-dir")
    p_build.add_argument("--dry-run", action="store_true", help="print the script only")
    p_build.add_argument("--verify", action="store_true",
                         help="run `agy models` afterwards and diff")
    p_build.add_argument("--strict", action="store_true",
                         help="exit non-zero if the selection overflows the cap")
    p_build.add_argument("--probe", action="store_true",
                         help="warn about models that agy cannot actually use")
    p_build.add_argument("--drop-probe-failures", action="store_true",
                         help="drop models that fail a live probe, replacing them")
    p_build.add_argument("--strip-effort", action="store_true",
                         help="embed local reasoning_effort stripper shim to unlock anthropic/*")
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", parents=[common],
                              help="verify NEAR AI gateway & model TEE attestation")
    p_verify.add_argument("model", nargs="?", help="optional model id to attest")
    p_verify.add_argument("--json", action="store_true", help="print raw JSON report and pins")
    p_verify.set_defaults(func=cmd_verify)

    p_route = sub.add_parser("attest-route-check", parents=[common],
                             help="fail-closed preflight for opt-in attestation routing")
    p_route.add_argument("model", nargs="?", help="optional model id")
    p_route.add_argument("--provider", default="", help="optional attestation provider")
    p_route.add_argument("--verify-crypto", action="store_true",
                         help="run cryptographic TDX/GPU verifiers (requires optional verifier deps)")
    p_route.add_argument("--allow-out-of-date-tcb", action="store_true",
                         help="accept OutOfDate TDX TCB status in addition to UpToDate")
    p_route.set_defaults(func=cmd_attest_route_check)

    p_evidence = sub.add_parser("attest-evidence", aliases=["attestation-evidence"], parents=[common],
                                help="collect nonce/TLS-bound attestation evidence")
    p_evidence.add_argument("model", nargs="?", help="optional model id")
    p_evidence.add_argument("--provider", default="", help="optional attestation provider")
    p_evidence.add_argument("--no-tls-fingerprint", action="store_true",
                            help="do not request or require TLS fingerprint binding")
    p_evidence.add_argument("--verify-crypto", action="store_true",
                            help="run cryptographic TDX/GPU verifiers (requires optional verifier deps)")
    p_evidence.add_argument("--allow-out-of-date-tcb", action="store_true",
                            help="accept OutOfDate TDX TCB status in addition to UpToDate")
    p_evidence.add_argument("--json", action="store_true", help="print raw JSON evidence")
    p_evidence.set_defaults(func=cmd_attest_evidence)

    p_verify_chat = sub.add_parser("verify-chat", parents=[common],
                                   help="run an attested /chat/completions call and verify /signature/{chat_id}")
    p_verify_chat.add_argument("model", help="canonical model id (e.g. Qwen/Qwen3.8-27B)")
    p_verify_chat.add_argument("--prompt", default="Reply with OK.", help="test prompt to send")
    p_verify_chat.add_argument("--signing-algo", choices=["ecdsa", "ed25519"], default="ecdsa",
                               help="signature algorithm to request (default: ecdsa)")
    p_verify_chat.add_argument("--no-verify-crypto", action="store_true",
                               help="skip TDX/GPU preflight quote verification")
    p_verify_chat.add_argument("--allow-out-of-date-tcb", action="store_true",
                               help="accept OutOfDate TDX TCB status in addition to UpToDate")
    p_verify_chat.add_argument("--no-tls-fingerprint", action="store_true",
                               help="do not request or require TLS fingerprint binding")
    p_verify_chat.add_argument("--json", action="store_true", help="print raw JSON result")
    p_verify_chat.set_defaults(func=cmd_verify_chat)

    p_doctor = sub.add_parser("doctor", parents=[common], help="health check")
    p_doctor.add_argument("--save-cap", action="store_true",
                          help="store the detected model cap in the config")
    p_doctor.set_defaults(func=cmd_doctor)

    p_init = sub.add_parser("init", parents=[common], help="store the API key, seed profiles")
    p_init.add_argument("--save-key", action="store_true",
                        help="persist NEAR_AI_KEY (from environment or prompt) into config.json (mode 0600)")
    p_init.set_defaults(func=cmd_init)

    p_sync = sub.add_parser("sync-settings", parents=[common],
                            help="sync model token limits into agy settings.json")
    for p in [p_sync, sub.add_parser("sync_settings", parents=[common],
                                     help="alias for sync-settings")]:
        p.add_argument("--models", help="comma-separated list of model IDs to sync")
        p.add_argument("--all-catalog", action="store_true",
                       help="sync all chat models from the live catalog")
        p.add_argument("--check", action="store_true",
                       help="read-only audit reporting whether target models are configured")
        p.add_argument("--dry-run", action="store_true",
                       help="show models that would be added/updated without modifying files")
        p.set_defaults(func=cmd_sync_settings)

    p_prof = sub.add_parser("list-profiles", parents=[common], help="show available profiles")
    p_prof.set_defaults(func=cmd_list_profiles)
    sub.add_parser("profiles", parents=[common], help="alias for list-profiles").set_defaults(
        func=cmd_list_profiles)

    p_systest = sub.add_parser("systems-test", parents=[common],
                               help="run full end-to-end systems test against NEAR AI gateway")
    for p in [p_systest, sub.add_parser("test-systems", parents=[common],
                                        help="alias for systems-test")]:
        p.add_argument("--profile", help="test models from a specific profile (e.g. qwen, frontier)")
        p.add_argument("--models", help="comma-separated list of model IDs to test")
        p.add_argument("--all-catalog", action="store_true",
                       help="test all chat models in the live catalog (default)")
        p.add_argument("--skip-tools", action="store_true",
                       help="skip function-calling / tool validation (ping only)")
        p.add_argument("--workers", type=int, default=3,
                       help="number of concurrent worker threads (default: 3)")
        p.add_argument("--json", dest="json_path", metavar="FILE",
                       help="save full JSON report to specified file")
        p.add_argument("--update-cache", action="store_true",
                       help="update probe cache with test results")
        p.add_argument("--limit", type=int, default=None,
                       help="limit number of models tested (useful for quick smoke runs)")
        p.set_defaults(func=cmd_systems_test)

    p_pro = sub.add_parser("pro", parents=[common], help="manage Agy Pro plugin and external subagents")
    pro_sub = p_pro.add_subparsers(dest="pro_command")

    p_pro_setup = pro_sub.add_parser("setup", parents=[common], help="install Agy Pro plugin and subagents")
    p_pro_setup.add_argument("--workspace", action="store_true",
                             help="install to workspace (.agents/plugins/nearai) instead of global (~/.gemini/config/plugins/nearai)")
    p_pro_setup.add_argument("--path", help="custom target plugin directory")
    p_pro_setup.add_argument("--force", action="store_true", help="overwrite existing files")
    p_pro_setup.add_argument("--dry-run", action="store_true", help="show actions without modifying files")
    p_pro_setup.add_argument("--models", help="additional comma-separated model IDs to register")
    p_pro_setup.set_defaults(func=cmd_pro_setup)

    p_pro_list = pro_sub.add_parser("list", parents=[common], help="list installed NEAR AI subagents and status")
    p_pro_list.add_argument("--workspace", action="store_true", help="check workspace (.agents/plugins/nearai)")
    p_pro_list.add_argument("--path", help="custom plugin directory to inspect")
    p_pro_list.set_defaults(func=cmd_pro_list)

    p_pro_exec = pro_sub.add_parser("exec", parents=[common], help="execute non-MCP subagent bridge request")
    p_pro_exec.add_argument("--model", default="Qwen/Qwen3.8-27B", help="target model ID (default: Qwen/Qwen3.8-27B)")
    p_pro_exec.add_argument("--prompt", help="prompt text")
    p_pro_exec.add_argument("--file", "-f", dest="input_file", help="read prompt or data from file")
    p_pro_exec.add_argument("--system", help="system instructions for subagent")
    p_pro_exec.add_argument("--json", dest="json_out", action="store_true", help="emit output as JSON")
    p_pro_exec.add_argument("--stream", action="store_true", default=True, help="stream SSE output (default: True)")
    p_pro_exec.add_argument("--no-stream", dest="stream", action="store_false", help="buffer response and print all at once")
    p_pro_exec.add_argument("--timeout", type=int, default=60, help="request timeout in seconds (default: 60)")
    p_pro_exec.set_defaults(func=cmd_pro_exec)

    p_pro_uninst = pro_sub.add_parser("uninstall", parents=[common], help="uninstall Agy Pro plugin")
    p_pro_uninst.add_argument("--workspace", action="store_true", help="uninstall from workspace")
    p_pro_uninst.add_argument("--path", help="custom plugin directory to remove")
    p_pro_uninst.add_argument("--dry-run", action="store_true", help="show actions without removing files")
    p_pro_uninst.set_defaults(func=cmd_pro_uninstall)

    p_pro_bench = pro_sub.add_parser("benchmark", parents=[common],
                                     help="run empirical benchmark across task domains")
    p_pro_bench.add_argument("--models", help="comma-separated model IDs to benchmark")
    p_pro_bench.add_argument("--tasks", help="comma-separated task IDs to run")
    p_pro_bench.add_argument("--json", dest="json_path", help="save benchmark JSON report to file")
    p_pro_bench.add_argument("--timeout", type=int, default=60, help="request timeout in seconds (default: 60)")
    p_pro_bench.set_defaults(func=cmd_pro_benchmark)

    p_pro_recom = pro_sub.add_parser("recommend", parents=[common],
                                     help="recommend optimal model and skill for task")
    p_pro_recom.add_argument("query", nargs="?", default="", help="task description, keyword, or domain category")
    p_pro_recom.add_argument("--json", dest="as_json", action="store_true", help="output recommendation in JSON format")
    p_pro_recom.set_defaults(func=cmd_pro_recommend)

    p_pro_add = pro_sub.add_parser("add", parents=[common], help="register a new custom subagent targeting a model")
    p_pro_add.add_argument("--name", required=True, help="subagent name (e.g. nearai-kimi or kimi)")
    p_pro_add.add_argument("--model", required=True, help="target model ID (e.g. moonshotai/kimi-k3)")
    p_pro_add.add_argument("--role", help="short 2-5 word role description")
    p_pro_add.add_argument("--description", help="detailed task description for subagent")
    p_pro_add.add_argument("--workspace", action="store_true", help="install to workspace instead of global")
    p_pro_add.add_argument("--path", help="custom plugin directory")
    p_pro_add.set_defaults(func=cmd_pro_add)

    p_pro_rm = pro_sub.add_parser("remove", parents=[common], help="remove an installed subagent")
    p_pro_rm.add_argument("name", help="subagent name to remove")
    p_pro_rm.add_argument("--workspace", action="store_true", help="remove from workspace instead of global")
    p_pro_rm.add_argument("--path", help="custom plugin directory")
    p_pro_rm.set_defaults(func=cmd_pro_remove)

    p_pro_led = pro_sub.add_parser("ledger", parents=[common], help="display subagent token usage and spending ledger")
    p_pro_led.add_argument("--json", dest="json_out", action="store_true", help="dump ledger records as JSON")
    p_pro_led.add_argument("--clear", action="store_true", help="reset/clear the usage ledger")
    p_pro_led.set_defaults(func=cmd_pro_ledger)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    if getattr(args, "command", None) == "pro" and not getattr(args, "pro_command", None):
        p_pro.print_help()
        return 0

    # `--strict` is honoured inside _build, which returns non-zero on overflow.
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
