from __future__ import annotations

import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parent


class FrontendHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    host = os.getenv("FRONTEND_HOST", "127.0.0.1")
    port = int(os.getenv("FRONTEND_PORT", "8080"))
    handler = partial(FrontendHandler, directory=str(FRONTEND_DIR))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Frontend available at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
