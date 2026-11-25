#!/usr/bin/env python3
"""Scratch: mock OpenAI-compatible gateway that logs what agy actually sends.

Purpose: agy injects `reasoning_effort` for some models and not others, and
NearAI rejects that field for some of them. Rather than infer the rule from
failures, this server records the real request body per model so the rule can
be read directly. No real API calls, no cost.

Logs NDJSON to $MOCKLOG (default ~/.cache/nearai-agy/mock-gateway.jsonl).
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "nearai-agy"
LOG = os.environ.get("MOCKLOG", str(_DEFAULT_CACHE / "mock-gateway.jsonl"))
lock = threading.Lock()


def log(entry):
    with lock:
        log_path = Path(LOG)
        log_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            # Serve a catalog for whatever ids agy asks about: agy only needs
            # the listed ids to exist so it will route to them.
            ids = [x for x in (os.environ.get("MOCK_MODELS", "")).split(",") if x]
            self._json({"object": "list", "data": [
                {"id": i, "object": "model", "owned_by": "mock",
                 "context_length": 128000, "output_modalities": ["text"],
                 "input_modalities": ["text"]} for i in ids]})
            return
        log({"method": "GET", "path": self.path})
        self._json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        cl_raw = (self.headers.get("Content-Length") or "0").strip()
        if not cl_raw.isdigit() or int(cl_raw) > 10 * 1024 * 1024:
            self._json({"error": {"message": "invalid or oversized Content-Length"}}, 400)
            return
        n = int(cl_raw)
        raw = self.rfile.read(n).decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"_raw": raw}
        model = body.get("model", "?")
        log({"method": "POST", "path": self.path, "model": model,
             "keys": sorted(body.keys()),
             "reasoning_effort": body.get("reasoning_effort", "<ABSENT>"),
             "stream": body.get("stream"),
             "body": body})
        if "chat/completions" not in self.path:
            self._json({"error": {"message": "not found"}}, 404)
            return
        if body.get("stream"):
            # agy asks for SSE; a plain JSON reply makes it retry ~5x per call.
            chunks = [
                {"id": "mock-1", "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
                {"id": "mock-1", "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            payload = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            payload += "data: [DONE]\n\n"
            data = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json({
            "id": "mock-1", "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        })


if __name__ == "__main__":
    port = int(os.environ.get("MOCKPORT", "8899"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write("mock gateway on http://127.0.0.1:%d/v1  ->  %s\n" % (port, LOG))
    srv.serve_forever()
