"""
python -m unittest discover -s tests   (needs requests; serves files from a local HTTP server)

Replays the ways the public R2 domain failed workers on Nov 17-18 2026: slow or failing answers,
and streams cut off mid-file.
"""

import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_download import AudioDownloadError, download_audio  # noqa: E402

BODY = bytes(range(256)) * 4096  # 1 MiB


class Script:
    """What the server does on each request, in order; the last step repeats."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    def next(self, headers):
        self.requests.append(dict(headers))
        return self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]


def make_handler(script):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            step = script.next(self.headers)
            start = 0
            rng = self.headers.get("Range")
            if rng and step in ("ok", "cut"):
                start = int(rng.split("=")[1].split("-")[0])
            if isinstance(step, int):
                self.send_response(step)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = BODY[start:]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if step == "cut":
                # promise the whole file, send a third, hang up (a stalled R2 stream)
                self.wfile.write(body[: len(body) // 3])
                self.wfile.flush()
                self.connection.shutdown(2)
                return
            self.wfile.write(body)

    return Handler


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def serve(self, script):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(script))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/AB12.m4a"

    def download(self, url, **kwargs):
        self.sleeps = []
        return download_audio(url, directory=self.dir, sleep=self.sleeps.append, **kwargs)

    def test_plain_download(self):
        path = self.download(self.serve(Script("ok")))
        self.assertTrue(path.endswith(".m4a"))
        with open(path, "rb") as file:
            self.assertEqual(file.read(), BODY)

    def test_server_errors_are_retried_with_backoff(self):
        script = Script(503, 502, "ok")
        path = self.download(self.serve(script))
        with open(path, "rb") as file:
            self.assertEqual(file.read(), BODY)
        self.assertEqual(self.sleeps, [1, 2])
        self.assertEqual(len(script.requests), 3)

    def test_a_cut_off_stream_resumes_where_it_stopped(self):
        script = Script("cut", "cut", "ok")
        path = self.download(self.serve(script))
        with open(path, "rb") as file:
            self.assertEqual(file.read(), BODY)
        # the second and third requests ask only for what's missing
        self.assertNotIn("Range", script.requests[0])
        self.assertTrue(script.requests[1]["Range"].startswith("bytes="))
        self.assertNotEqual(script.requests[1]["Range"], "bytes=0-")

    def test_missing_file_fails_at_once_with_a_clear_error(self):
        script = Script(404)
        with self.assertRaises(AudioDownloadError) as raised:
            self.download(self.serve(script))
        self.assertIn("Couldn't download the audio", str(raised.exception))
        self.assertIn("HTTP 404", str(raised.exception))
        self.assertEqual(len(script.requests), 1)
        self.assertEqual(os.listdir(self.dir), [])

    def test_gives_up_after_the_last_try_and_cleans_up(self):
        script = Script(503)
        with self.assertRaises(AudioDownloadError) as raised:
            self.download(self.serve(script), max_tries=3)
        self.assertIn("after 3 tries", str(raised.exception))
        self.assertEqual(len(script.requests), 3)
        self.assertEqual(os.listdir(self.dir), [])

    def test_nothing_listening(self):
        with self.assertRaises(AudioDownloadError) as raised:
            self.download("http://127.0.0.1:9/AB12.m4a", max_tries=2, connect_timeout=1)
        self.assertIn("after 2 tries", str(raised.exception))
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
