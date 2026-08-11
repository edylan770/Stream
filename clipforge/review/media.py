"""HTTP range serving for the proxy.

Phase 1 reviews candidates by seeking a 700 MB proxy rather than playing
pre-rendered clips (§7.2's preview assets are Phase 3). That decision only works
if the browser can ask for byte 400,000,000 and get it — without range support
the `<video>` element re-downloads from zero on every seek, and C4's four
seconds per candidate becomes four seconds of buffering.

Written out rather than delegated to the framework's file response. Starlette
handles ranges in current versions, but this is the one behaviour the entire
review experience rests on, and thirty lines that are tested beat a dependency
behaviour that changes between releases.
"""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from starlette.responses import FileResponse, Response, StreamingResponse

#: Bytes per chunk when streaming a range. Large enough that the syscall
#: overhead is irrelevant, small enough not to hold a megabyte per request.
CHUNK = 256 * 1024

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int  # inclusive, as HTTP defines it

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def parse_range(header: str | None, size: int) -> ByteRange | None:
    """Parse a single `Range: bytes=...` header against a known file size.

    Returns None when there is no range, the syntax is unrecognised, or the
    request is unsatisfiable — the caller then serves the whole file, which is
    what RFC 9110 asks for on a malformed range.

    Handles the three forms browsers actually send:
      `bytes=0-`         open-ended, the initial probe
      `bytes=500-999`    explicit, what a seek produces
      `bytes=-500`       suffix, used to read a trailing index
    """
    if not header or size <= 0:
        return None

    match = _RANGE.match(header.strip())
    if not match:
        return None

    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:                      # suffix: the last N bytes
        length = int(raw_end)
        if length <= 0:
            return None
        start, end = max(size - length, 0), size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1

    if start >= size or start > end:
        return None                        # unsatisfiable
    return ByteRange(start=start, end=min(end, size - 1))


def content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def serve(path: Path, range_header: str | None) -> Response:
    """Serve a media file, honouring a range request when there is one."""
    if not path.is_file():
        return Response("not found", status_code=404)

    size = path.stat().st_size
    media_type = content_type_for(path)
    requested = parse_range(range_header, size)

    if requested is None:
        response = FileResponse(path, media_type=media_type)
        # Advertise range support even on the full-file response, so the
        # browser knows it may seek rather than buffering the whole file.
        response.headers["Accept-Ranges"] = "bytes"
        return response

    def chunks():
        remaining = requested.length
        with path.open("rb") as handle:
            handle.seek(requested.start)
            while remaining > 0:
                block = handle.read(min(CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(
        chunks(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {requested.start}-{requested.end}/{size}",
            "Content-Length": str(requested.length),
            "Accept-Ranges": "bytes",
            # The proxy is regenerated in place by the pipeline; a cached copy
            # of the previous encode would show the wrong frames.
            "Cache-Control": "no-store",
        },
    )
