"""Serve the viewer locally, with the web export mounted at /data/.

    python bench/tractography/viewer/serve.py [port]        # default 8765, 127.0.0.1 only

The export stays in DATA/hcp/web (outside the repo and Dropbox, and HCP data: local only until
the data-use terms are checked), so the viewer directory holds code only and nothing is
copied or linked into it.
"""
import functools, http.server, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = (Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography")) / "hcp" / "web").resolve()


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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    if not (WEB / "manifest.json").exists():
        sys.exit(f"no export at {WEB}: run bench/tractography/export.py first")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(HERE)))
    print(f"viewer on http://127.0.0.1:{port}/  (data: {WEB})", flush=True)
    server.serve_forever()
