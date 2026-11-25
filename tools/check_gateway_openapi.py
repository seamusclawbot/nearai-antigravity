#!/usr/bin/env python3
"""Explicitly requested live OpenAPI drift check; offline tests do not call network."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "docs/gateway-openapi-contract.json").read_text())
URL = "https://cloud-api.near.ai/api-docs/openapi.json"


def main() -> int:
    with urllib.request.urlopen(URL, timeout=20) as response:
        document = json.load(response)
    paths = document.get("paths", {})
    missing = [path for path in CONTRACT["paths"] if path not in paths]
    haystack = json.dumps(document, sort_keys=True)
    missing_terms = [term for term in CONTRACT["required_schema_terms"] if term not in haystack]
    if missing or missing_terms:
        print(json.dumps({"missing_paths": missing, "missing_terms": missing_terms}, indent=2))
        return 1
    print(f"Gateway OpenAPI contract OK: {len(CONTRACT['paths'])} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
