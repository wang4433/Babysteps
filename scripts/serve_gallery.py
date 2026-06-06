# scripts/serve_gallery.py
"""Tiny static file server with HTTP Range support, for the demo/execution gallery.

The stdlib ``http.server`` does not honour ``Range`` requests, so browsers
cannot seek (and some refuse to play) embedded ``<video>`` elements served by
it. This handler adds ``Range`` -> ``206 Partial Content`` so the gallery's MP4
players scrub smoothly. It also follows the symlinks under ``videos/`` (the
clips live in gitignored ``renders/`` + ``datasets/`` and are symlinked into the
gallery).

Bind is localhost-only by default — view it over an SSH tunnel:

    # on the cluster (in the gallery dir or pass --dir):
    python scripts/serve_gallery.py --dir reports/stage5/demo_execution_gallery --port 8899
    # on your laptop:
    ssh -N -L 8899:localhost:8899 <user>@gilbreth-fe00.rcac.purdue.edu
    # then open http://localhost:8899/index.html
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + minimal single-range ``bytes=start-end`` support."""

    def send_head(self):  # noqa: C901 - mirrors the stdlib method's shape
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            size = fs[6]
            start, end = self._parse_range(rng, size)
            if start is None:
                # Unsatisfiable / malformed -> fall back to a normal 200.
                f.seek(0)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return f
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            f.seek(start)
            self._remaining = length
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        # Stream only the requested byte window.
        chunk = 64 * 1024
        while remaining > 0:
            buf = source.read(min(chunk, remaining))
            if not buf:
                break
            outputfile.write(buf)
            remaining -= len(buf)
        self._remaining = None

    @staticmethod
    def _parse_range(header: str, size: int):
        try:
            unit, _, spec = header.partition("=")
            if unit.strip() != "bytes" or "," in spec:
                return None, None
            s, _, e = spec.partition("-")
            s, e = s.strip(), e.strip()
            if s == "":                       # suffix range: bytes=-N
                n = int(e)
                if n <= 0:
                    return None, None
                start = max(0, size - n)
                return start, size - 1
            start = int(s)
            end = int(e) if e else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                return None, None
            return start, end
        except (ValueError, TypeError):
            return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="reports/stage5/demo_execution_gallery",
                    help="Directory to serve (default: the gallery).")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="Bind address (default localhost — use with an SSH tunnel).")
    args = ap.parse_args()

    directory = os.path.abspath(args.dir)
    handler = partial(RangeHTTPRequestHandler, directory=directory)
    httpd = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"[serve_gallery] serving {directory}")
    print(f"[serve_gallery] http://{args.bind}:{args.port}/index.html  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve_gallery] stopped")
        httpd.shutdown()


if __name__ == "__main__":
    main()
