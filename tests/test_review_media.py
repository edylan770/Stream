"""HTTP range serving.

Phase 1 reviews candidates by seeking a 700 MB proxy rather than playing
pre-rendered clips (§7.2 is Phase 3). Without working ranges the <video>
element re-downloads from byte zero on every seek and C4's four seconds per
candidate becomes four seconds of buffering, so this is the one behaviour the
whole review experience rests on.
"""

from __future__ import annotations

import pytest

from clipforge.review import media


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / "proxy.mp4"
    path.write_bytes(bytes(range(256)) * 40)   # 10240 bytes, byte i = i % 256
    return path


def body_of(response) -> bytes:
    if hasattr(response, "body_iterator"):
        import asyncio

        async def drain():
            return b"".join([chunk async for chunk in response.body_iterator])

        return asyncio.run(drain())
    return response.body


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_open_ended_range_is_the_initial_probe():
    """`bytes=0-` is what a browser sends first."""
    assert media.parse_range("bytes=0-", 1000) == media.ByteRange(0, 999)


def test_explicit_range_is_what_a_seek_produces():
    assert media.parse_range("bytes=500-999", 10000) == media.ByteRange(500, 999)


def test_suffix_range_reads_the_tail():
    """Used to fetch a trailing index; MP4 with +faststart does not need it,
    but a browser may still ask."""
    assert media.parse_range("bytes=-500", 10000) == media.ByteRange(9500, 9999)


def test_range_past_the_end_is_clamped():
    assert media.parse_range("bytes=9000-99999", 10000) == media.ByteRange(9000, 9999)


@pytest.mark.parametrize(
    "header",
    ["", None, "bytes=", "items=0-10", "bytes=abc-def", "nonsense", "bytes=-0"],
)
def test_unusable_headers_fall_back_to_the_whole_file(header):
    """RFC 9110: serve the whole representation rather than erroring."""
    assert media.parse_range(header, 1000) is None


def test_unsatisfiable_range_falls_back():
    assert media.parse_range("bytes=5000-", 1000) is None
    assert media.parse_range("bytes=900-100", 1000) is None


def test_zero_length_file_has_no_ranges():
    assert media.parse_range("bytes=0-", 0) is None


def test_range_length_is_inclusive():
    """HTTP ranges include both endpoints — an off-by-one here truncates every
    seek by a byte."""
    assert media.ByteRange(0, 0).length == 1
    assert media.ByteRange(10, 19).length == 10


# --------------------------------------------------------------------------
# serving
# --------------------------------------------------------------------------


def test_full_request_advertises_range_support(payload):
    """The browser only seeks if it is told it may."""
    response = media.serve(payload, None)
    assert response.status_code == 200
    assert response.headers["accept-ranges"] == "bytes"


def test_range_request_returns_206_with_the_right_bytes(payload):
    response = media.serve(payload, "bytes=1000-1099")
    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 1000-1099/10240"
    assert response.headers["content-length"] == "100"

    data = body_of(response)
    assert len(data) == 100
    assert data == bytes((1000 + i) % 256 for i in range(100))


def test_seeking_near_the_end_works(payload):
    """The case that matters: a candidate at t=670s of a 682s stream."""
    response = media.serve(payload, "bytes=10200-")
    assert response.status_code == 206
    assert len(body_of(response)) == 40


def test_a_range_spanning_many_chunks_is_complete(payload, monkeypatch):
    """Chunked streaming must not drop or duplicate a boundary."""
    monkeypatch.setattr(media, "CHUNK", 7)
    response = media.serve(payload, "bytes=0-10239")
    assert body_of(response) == payload.read_bytes()


def test_missing_file_is_404(tmp_path):
    assert media.serve(tmp_path / "nope.mp4", None).status_code == 404


def test_content_type_is_inferred(payload):
    assert media.content_type_for(payload) == "video/mp4"
    assert "matroska" in media.content_type_for(payload.with_suffix(".mkv"))


def test_range_responses_are_not_cached(payload):
    """The pipeline regenerates the proxy in place; a cached copy of the
    previous encode would show the wrong frames."""
    response = media.serve(payload, "bytes=0-99")
    assert response.headers["cache-control"] == "no-store"
