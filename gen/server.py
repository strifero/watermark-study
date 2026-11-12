"""Key-side scoring endpoint, 127.0.0.1:8950. POST /score {"text": ..., "model": "g31b"|"e4b"} -> {"score","z","n","flag"}.
flag uses the threshold in /home/sean/wm/key/threshold.json once calibrated. This is the only door to the key for
the attack code."""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from scorer import score
TH = "/home/sean/wm/key/threshold.json"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/score":
            self.send_response(404); self.end_headers(); return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        r = score(req.get("text", "")[:20000], req.get("model", "g31b"))
        if os.path.exists(TH):
            r["flag"] = r["z"] >= json.load(open(TH))["z"]
        b = json.dumps(r).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


ThreadingHTTPServer(("127.0.0.1", 8950), H).serve_forever()
