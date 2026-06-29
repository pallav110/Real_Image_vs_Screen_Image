"""Tiny camera-demo web server (standard library only -- no Flask).

Run:
    python demo.py
Then open http://localhost:8000 in a browser, allow the camera, and either
capture a frame or upload a photo. The page POSTs the image bytes to /predict,
which runs the SAME predict.predict() the grader uses, and shows the 0-1 score.

Why a server (not pure JS): the model + native-resolution FFT features live in
Python; the browser only captures pixels and POSTs them, so there is a single,
trusted inference path (zero train/serve skew).
"""

import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
import predict as P

ROOT = Path(__file__).parent
HTML = (ROOT / "demo.html").read_bytes()
PORT = 8000


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            score = P.predict(tmp.name)
            self._send(200, json.dumps({"score": score}).encode())
        except Exception as e:  # surface errors to the page instead of 500-ing silently
            self._send(400, json.dumps({"error": str(e)}).encode())
        finally:
            os.unlink(tmp.name)

    def log_message(self, *args):
        pass  # quiet console


if __name__ == "__main__":
    P._load_bundle()  # fail fast if model.joblib is missing
    print(f"Recapture demo running at http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
