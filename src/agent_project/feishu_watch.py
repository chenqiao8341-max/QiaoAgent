from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv

from agent_project.tools.feishu import generate_feishu_report, handle_feishu_event
from agent_project.tools.progress import emit_progress


class FeishuHandler(BaseHTTPRequestHandler):
    report_interval_seconds = 300
    report_batch_size = 10
    last_report_at = time.monotonic()
    pending_messages = 0

    def log_message(self, format: str, *args: Any) -> None:
        emit_progress("feishu webhook " + (format % args))

    def do_GET(self) -> None:
        self._send_json(200, {"ok": True, "service": "agent-feishu-watch"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        status, response = handle_feishu_event(payload)
        if response.get("stored"):
            type(self).pending_messages += 1
            self._maybe_report()
        self._send_json(status, response)

    def _maybe_report(self) -> None:
        now = time.monotonic()
        enough_messages = self.pending_messages >= self.report_batch_size
        enough_time = now - self.last_report_at >= self.report_interval_seconds
        if not enough_messages and not enough_time:
            return

        report = generate_feishu_report.invoke({"limit": 200})
        if not report.startswith("No unsummarized"):
            print("\n=== Feishu Message Report ===")
            print(report)
            print("=== End Feishu Message Report ===\n")
        type(self).pending_messages = 0
        type(self).last_report_at = now

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int, report_interval_seconds: int, report_batch_size: int) -> None:
    FeishuHandler.report_interval_seconds = max(1, report_interval_seconds)
    FeishuHandler.report_batch_size = max(1, report_batch_size)
    FeishuHandler.last_report_at = time.monotonic()
    FeishuHandler.pending_messages = 0

    server = ThreadingHTTPServer((host, port), FeishuHandler)
    emit_progress(f"Feishu watcher listening on http://{host}:{port}")
    emit_progress("Configure this URL as the Feishu event callback endpoint.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFeishu watcher stopped.")
    finally:
        server.server_close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Feishu event watcher.")
    parser.add_argument("--host", default=os.getenv("FEISHU_WATCH_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FEISHU_WATCH_PORT", "8787")))
    parser.add_argument(
        "--report-interval-seconds",
        type=int,
        default=int(os.getenv("FEISHU_REPORT_INTERVAL_SECONDS", "300")),
    )
    parser.add_argument(
        "--report-batch-size",
        type=int,
        default=int(os.getenv("FEISHU_REPORT_BATCH_SIZE", "10")),
    )
    args = parser.parse_args()
    run_server(args.host, args.port, args.report_interval_seconds, args.report_batch_size)


if __name__ == "__main__":
    main()
