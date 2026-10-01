"""Serve the time-to-display benchmark, with DATA/decode mounted at /data/.

    python bench/tractography/decode/render/serve.py [port]   # default 8766, 127.0.0.1 only

The inputs stay in DATA/decode (outside the repo), with the npm packages and the Draco decoder
beside them; this directory holds the page only.
"""
import functools, http.server, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = (Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography")) / "decode").resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/data/"):
            target = (WEB / clean[len("/data/"):]).resolve()
            return str(target) if WEB in target.parents else str(HERE / "missing")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    if not (WEB / "render_12.json").exists():
        sys.exit(f"no export at {WEB}: run decode_prep.py and render_prep.py first")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(HERE)))
    print(f"render benchmark on http://127.0.0.1:{port}/  (data: {WEB})", flush=True)
    server.serve_forever()
