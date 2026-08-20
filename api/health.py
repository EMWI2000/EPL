"""Lightweight liveness endpoint exposed as ``GET /api/health`` on Vercel."""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
import json
from typing import Any


API_VERSION = "1"


def health_payload() -> dict[str, Any]:
    """Return a dependency-free liveness response.

    This endpoint deliberately does not call FPL or Solio.  It reports whether
    the serverless function itself can start; upstream health is represented by
    the recommendation endpoint's explicit error responses.
    """

    return {
        "status": "ok",
        "service": "epl-fpl-api",
        "version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class handler(BaseHTTPRequestHandler):
    """Vercel Python function handler."""

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send_json(200, health_payload())

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self.send_response(204)
        self.send_header("Allow", "GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send_json(
            405,
            {"error": {"code": "method_not_allowed", "message": "Use GET for this endpoint."}},
        )
