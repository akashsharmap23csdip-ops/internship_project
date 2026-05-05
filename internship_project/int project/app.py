from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PORT = 8000
START_PAGE = f"http://localhost:{PORT}/frontend/index.html"


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def open_browser_later(url: str) -> None:
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


def main() -> None:
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(ROOT_DIR))
    socketserver.TCPServer.allow_reuse_address = True

    open_browser_later(START_PAGE)
    print(f"Serving Sales Forecasting Dashboard at {START_PAGE}")
    print("Press Ctrl+C to stop the server.")

    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")


if __name__ == "__main__":
    main()