"""Serve web/dist locally with the headers the live site sends.

  python3 web/tools/serve.py [port]     then open http://localhost:8060/

Camera access works on localhost without https.
"""
import http.server
import sys
from functools import partial
from pathlib import Path

DIST = Path(__file__).resolve().parents[1] / "dist"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8060
print(f"http://localhost:{port}/")
http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(DIST))).serve_forever()
