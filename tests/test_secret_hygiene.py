"""Offline repository secret- and path-hygiene checks.

The test scans tracked text files for accidental credentials, private keys, and
developer-specific absolute home paths. Synthetic test placeholders are filtered
on a per-match basis rather than skipping entire files.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    r"(?:^|[^A-Za-z0-9])Bearer\s+[A-Za-z0-9._-]{24,}",
    r"(?:^|[^A-Za-z0-9])eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    r"(?:sk|api[_-]?key|token)[_-]?[A-Za-z0-9]*\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{24,}",
    r"\bsk-[A-Za-z0-9_-]{20,}\b",
    r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bAIza[0-9A-Za-z_-]{35}\b",
)

# Explicit synthetic placeholder values permitted only in tests/
ALLOWED_TEST_PLACEHOLDERS = {
    "sk-test-real-key-12345",
    "sk-ephemeral-env-key",
    "sk-fake-example.invalid",
}

LOCAL_HOME_PATH_RE = re.compile(r"(?:/Users/(?!you\b)[A-Za-z0-9._-]+|/home/(?!you\b|user\b)[A-Za-z0-9._-]+)")
PERSONAL_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?:gmail\.com|googlemail\.com|yahoo\.com|outlook\.com|hotmail\.com|icloud\.com|me\.com|proton\.me|protonmail\.com)\b",
    re.IGNORECASE,
)


def _iter_tracked_text_files():
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    for raw in files:
        if not raw:
            continue
        path = ROOT / os.fsdecode(raw)
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        yield str(path.relative_to(ROOT)), text


def test_tracked_files_have_no_obvious_live_credentials():
    offenders = []
    for rel, text in _iter_tracked_text_files():
        is_test_file = rel.startswith("tests/")
        for pattern in PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                token = match.group(0).strip()
                if is_test_file and token in ALLOWED_TEST_PLACEHOLDERS:
                    continue
                offenders.append(f"{rel}: {token[:16]}...")
                break
    assert not offenders, "possible credential material in tracked files: " + ", ".join(offenders)


def test_tracked_files_have_no_local_home_paths():
    offenders = []
    for rel, text in _iter_tracked_text_files():
        if LOCAL_HOME_PATH_RE.search(text):
            offenders.append(rel)
    assert not offenders, "developer-specific home paths in tracked files: " + ", ".join(offenders)


def test_tracked_files_have_no_personal_emails():
    offenders = []
    for rel, text in _iter_tracked_text_files():
        if PERSONAL_EMAIL_RE.search(text):
            offenders.append(rel)
    assert not offenders, "personal email addresses in tracked files: " + ", ".join(offenders)
