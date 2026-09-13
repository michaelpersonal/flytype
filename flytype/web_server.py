"""Loopback-only HTTP/SSE presentation server for continuous FlyBreak."""

import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class GameRunner:
    """Single writer around a continuous play session."""

    def __init__(self, session, tick_seconds=0.18):
        self.session = session
        self.tick_seconds = max(0.0, float(tick_seconds))
        self.control_token = secrets.token_urlsafe(32)
        self.condition = threading.Condition()
        self.revision = 0
        self.sequence = 0
        self.status = "starting"
        self.snapshot = {
            "episode": session.episode,
            "observation": session.observation_count,
            "source": session.mode,
            "status": "starting",
        }
        self.error = None
        self._pause_requested = False
        self._stop_requested = False
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="flybreak")
        self._thread.start()

    def _run(self):
        while True:
            with self.condition:
                while self._pause_requested and not self._stop_requested:
                    if self.status != "paused":
                        self.status = "paused"
                        self.sequence += 1
                    self.condition.notify_all()
                    self.condition.wait()
                if self._stop_requested:
                    self.status = "stopped"
                    self.sequence += 1
                    self.condition.notify_all()
                    return
                self.status = "computing"
                self.sequence += 1
                self.condition.notify_all()
            started = time.monotonic()
            try:
                snapshot = self.session.advance()
            except Exception as exc:
                with self.condition:
                    self.error = {"type": type(exc).__name__, "reason": str(exc)}
                    self.status = "error"
                    self.sequence += 1
                    self.condition.notify_all()
                return
            elapsed = time.monotonic() - started
            with self.condition:
                self.snapshot = snapshot
                self.revision += 1
                self.sequence += 1
                self.status = "running"
                self.condition.notify_all()
            remaining = self.tick_seconds - elapsed
            if remaining > 0:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self._pause_requested or self._stop_requested,
                        timeout=remaining,
                    )

    def public_state(self):
        with self.condition:
            return {
                **self.snapshot,
                "revision": self.revision,
                "sequence": self.sequence,
                "runner_status": self.status,
                "runner_error": self.error,
            }

    def control(self, action):
        with self.condition:
            if action == "pause":
                self._pause_requested = True
                if self.status == "computing":
                    self.status = "pausing"
            elif action == "resume":
                if self._stop_requested or self.status == "error":
                    raise ValueError("Stopped or failed runner cannot resume")
                self._pause_requested = False
            elif action == "stop":
                self._stop_requested = True
            else:
                raise ValueError("Unknown control action")
            self.sequence += 1
            self.condition.notify_all()

    def stop(self):
        self.control("stop")
        if self._thread is not None:
            self._thread.join(timeout=10)


class FlyBreakHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def create_server(runner, port=8765, web_root=None):
    root = Path(web_root or Path(__file__).with_name("web")).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "FlyBreak/1"

        def log_message(self, _format, *_args):
            return

        def _headers(self, status, content_type, length=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'",
            )
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def _json(self, value, status=HTTPStatus.OK):
            payload = json.dumps(value, allow_nan=False).encode()
            self._headers(status, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                self._json(runner.public_state())
                return
            if parsed.path == "/api/events":
                token = parse_qs(parsed.query).get("token", [None])[0]
                if token != runner.control_token:
                    self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
                    return
                self._events()
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            asset = assets.get(parsed.path)
            if asset is None:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            path = root / asset[0]
            try:
                payload = path.read_bytes()
            except FileNotFoundError:
                self._json({"error": "asset missing"}, HTTPStatus.NOT_FOUND)
                return
            if asset[0] == "index.html":
                payload = payload.replace(
                    b"__CONTROL_TOKEN__", runner.control_token.encode()
                )
            self._headers(HTTPStatus.OK, asset[1], len(payload))
            self.wfile.write(payload)

        def _events(self):
            self._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8")
            sequence = -1
            try:
                while True:
                    with runner.condition:
                        runner.condition.wait_for(
                            lambda: runner.sequence != sequence
                            or runner.status in ("error", "stopped"),
                            timeout=15,
                        )
                    state = runner.public_state()
                    sequence = state["sequence"]
                    payload = json.dumps(state, allow_nan=False)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    if state["runner_status"] in ("error", "stopped"):
                        return
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if urlparse(self.path).path != "/api/control":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            if self.headers.get("X-FlyType-Token") != runner.control_token:
                self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 1024:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                runner.control(body.get("action"))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "state": runner.public_state()})

        def do_OPTIONS(self):
            self._json({"error": "cross-origin controls are not supported"}, HTTPStatus.FORBIDDEN)

    return FlyBreakHTTPServer(("127.0.0.1", int(port)), Handler)
