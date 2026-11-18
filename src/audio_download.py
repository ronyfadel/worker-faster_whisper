"""
Download a job's audio patiently.

RunPod's `download_files_from_urls` gives each try a 5-s timeout and makes 3 tries, then returns
None, which surfaced as "No such file or directory: 'None'". On Nov 18 2026 three good
recordings failed that way in five hours (Whisper and Parakeet; 3 to 56 min, 0.7 to 109 MB):
each file was in R2, valid, and streamed in seconds from elsewhere. The public R2 domain is
sometimes slow to answer from a worker's data center for a few seconds, or stalls mid-file.

This downloader:
- waits longer for a connection (15 s) and between bytes (60 s);
- retries connection errors, timeouts, 408, 429 and 5xx up to 6 times, backing off 1, 2, 4,
  8, 16 s;
- resumes a stalled or cut-off download where it stopped (HTTP Range) instead of starting over;
- checks the size against Content-Length so a truncated file never reaches the decoder;
- gives up with a clear AudioDownloadError ("Couldn't download the audio …") rather than None.
"""

import os
import tempfile
import time
from urllib.parse import urlparse

import requests

CONNECT_TIMEOUT_SEC = 15
READ_TIMEOUT_SEC = 60
MAX_TRIES = 6
BACKOFF_SEC = (1, 2, 4, 8, 16)
# A long recording is ~110 MB; give up after this long whatever happens.
DEADLINE_SEC = 10 * 60
CHUNK_BYTES = 64 * 1024  # small enough that a cut-off stream keeps most of what arrived
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
HEADERS = {"User-Agent": "faster-whisper-worker"}


class AudioDownloadError(Exception):
    """The audio couldn't be downloaded; the message says why."""


class _Retry(Exception):
    pass


def _extension(url):
    return os.path.splitext(urlparse(url).path)[1][:10]


def _content_length(response, already):
    """Total size of the file, from Content-Range on a resumed request, else Content-Length."""
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = response.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + already
    return None


def download_audio(url, directory=None, session=None, sleep=time.sleep,
                   connect_timeout=CONNECT_TIMEOUT_SEC, read_timeout=READ_TIMEOUT_SEC,
                   max_tries=MAX_TRIES, deadline_sec=DEADLINE_SEC):
    """
    Download `url` to a new file in `directory` (a temp dir by default) and return its path.
    Raises AudioDownloadError when it can't.
    """
    session = session or requests.Session()
    fd, path = tempfile.mkstemp(suffix=_extension(url), prefix="audio-", dir=directory)
    os.close(fd)
    started = time.monotonic()
    written = 0
    total = None
    last_error = None

    for attempt in range(1, max_tries + 1):
        headers = dict(HEADERS)
        if written:
            headers["Range"] = f"bytes={written}-"
        try:
            with session.get(url, stream=True, timeout=(connect_timeout, read_timeout),
                             headers=headers) as response:
                if response.status_code in RETRYABLE_STATUS:
                    raise _Retry(f"HTTP {response.status_code}")
                if response.status_code == 416 and total is not None and written >= total:
                    break  # asked for bytes past the end: we already have the whole file
                if response.status_code >= 400:
                    raise AudioDownloadError(
                        f"Couldn't download the audio from {url}: HTTP {response.status_code}")
                resumed = written and response.status_code == 206
                if written and not resumed:
                    written = 0  # the server ignored the Range header: start over
                total = _content_length(response, written) or total
                with open(path, "ab" if resumed else "wb") as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            file.write(chunk)
                            written += len(chunk)
                        if time.monotonic() - started > deadline_sec:
                            raise AudioDownloadError(
                                f"Couldn't download the audio from {url} within "
                                f"{deadline_sec} s ({written} bytes so far)")
            if total is not None and written < total:
                raise _Retry(f"connection closed at {written} of {total} bytes")
            if written == 0:
                raise _Retry("empty response")
            return path
        except AudioDownloadError:
            _remove(path)
            raise
        except (requests.RequestException, _Retry) as error:
            last_error = error
            print(f"audio download try {attempt}/{max_tries} failed ({written} bytes kept): {error}")
            if attempt == max_tries or time.monotonic() - started > deadline_sec:
                break
            sleep(BACKOFF_SEC[min(attempt - 1, len(BACKOFF_SEC) - 1)])

    if total is not None and written >= total and written > 0:
        return path
    _remove(path)
    raise AudioDownloadError(
        f"Couldn't download the audio from {url} after {max_tries} tries: {last_error}")


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
