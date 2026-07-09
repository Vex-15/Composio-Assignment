"""
Composio App Research Pipeline — Report Generator
===================================================
Utility to serve the case study HTML page locally with proper CORS
for fetch() to work with file:// protocol.

The case-study.html page loads data via fetch() from ./data/*.json.
This requires either:
  1. A local HTTP server (this module provides one)
  2. Deployment to GitHub Pages / any static host

Usage:
    python -m agents.report          # Serves at http://localhost:8080
    python -m agents.report --port 3000  # Custom port
"""

import http.server
import os
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP handler with CORS headers for local development."""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, format, *args):
        # Cleaner logging
        print(f"  {args[0]}")


def serve(port: int = 8080):
    """Start a local HTTP server serving the project root."""
    os.chdir(PROJECT_ROOT)

    handler = partial(CORSRequestHandler, directory=str(PROJECT_ROOT))
    server = http.server.HTTPServer(('', port), handler)

    print("=" * 60)
    print("Composio App Research — Local Report Server")
    print("=" * 60)
    print(f"\n  [Page] Case Study:  http://localhost:{port}/reports/case-study.html")
    print(f"  [Data] Data files:  http://localhost:{port}/data/")
    print(f"\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        if sys.argv[1] == "--port" and len(sys.argv) > 2:
            port = int(sys.argv[2])
        else:
            try:
                port = int(sys.argv[1])
            except ValueError:
                print(f"Usage: python -m agents.report [--port PORT]")
                sys.exit(1)

    serve(port)
