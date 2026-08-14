"""§7.3's transcript beside the window, and Render from the browser.

> Transcript text for the window displayed alongside.

The interesting assertions are made against the speech fixture's manifest —
authored text, authored speaker, authored placement — rather than against
anything written down here. Segments are inserted from that manifest the way
`tests/test_speakers.py` already does it, so none of this needs a Whisper model.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from clipforge import config, db, paths
from clipforge.review import app as review_app
from clipforge.review import jobs, queries
from clipforge.render.words import MIC, PARTY

TRACK_MAP = json.dumps({
    "roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3},
    "warnings": [], "source": "metadata tags", "titles": {},
})


@pytest.fixture
def spoken(tmp_path, speech_manifest):
    """A stream whose segments are the fixture's authored utterances."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    paths.StreamPaths(cfg.data_root, "sp").ensure()

    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, "
        "audio_track_map, duration_s, resolution) "
        "VALUES ('sp', '2026-08-14', 'm.mkv', 'vod', ?, 95.0, '640x360')",
        (TRACK_MAP,),
    )
    tracks = json.loads(TRACK_MAP)["roles"]
    for utterance in speech_manifest["utterances"]:
        conn.execute(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text, "
            "speaker, track, words) VALUES ('sp', ?, ?, ?, ?, ?, ?, '[]')",
            (utterance["index"], utterance["t_start"], utterance["t_end"],
             utterance["text"],
             "operator" if utterance["track"] == "mic" else "party",
             tracks[utterance["track"]]),
        )
    conn.commit()
    yield cfg, conn, speech_manifest
    conn.close()


def add_candidate(conn, t_start, t_end, score=1.0):
    """A minimal `candidates` row. Every NOT NULL column §3.2 declares, and
    nothing else — the transcript slice does not care about the scores."""
    conn.execute(
        "INSERT INTO candidates (stream_id, generation, is_current, profile, "
        "t_start, t_end, t_peak, score_entertainment, score_gameplay, "
        "score_combined, feature_vector, feature_schema_version, config_version) "
        "VALUES ('sp', 1, 1, 'naive', ?, ?, ?, 0.0, 0.0, ?, '{}', 1, 'v1')",
        (t_start, t_end, (t_start + t_end) / 2, score),
    )


def utterances_overlapping(manifest, t_start, t_end):
    return [u for u in manifest["utterances"]
            if u["t_end"] >= t_start and u["t_start"] <= t_end]


# --------------------------------------------------------------------------
# §7.3 — the transcript for the window
# --------------------------------------------------------------------------


def test_the_window_gets_exactly_the_lines_that_overlap_it(spoken):
    """Asserted against the manifest's own utterances, not a literal."""
    _, conn, manifest = spoken
    transcript = queries.load_transcript(conn, "sp")

    for window in [(0.0, 20.0), (20.0, 35.0), (50.0, 60.0), (70.0, 95.0)]:
        expected = [u["text"] for u in utterances_overlapping(manifest, *window)]
        assert [line["text"] for line in queries.lines_in(transcript, *window)] \
            == expected, window


def test_a_line_starting_before_the_window_is_included(spoken):
    """Overlap, not containment: that speech is inside the clip, so hiding it
    would misreport what the clip contains."""
    _, conn, manifest = spoken
    first = min(manifest["utterances"], key=lambda u: u["t_start"])
    # A window opening halfway through the first authored line.
    midpoint = (first["t_start"] + first["t_end"]) / 2

    lines = queries.lines_in(queries.load_transcript(conn, "sp"), midpoint,
                             midpoint + 1.0)

    assert first["text"] in [line["text"] for line in lines]


def test_a_line_entirely_outside_the_window_is_not_included(spoken):
    _, conn, manifest = spoken
    first = min(manifest["utterances"], key=lambda u: u["t_start"])

    lines = queries.lines_in(queries.load_transcript(conn, "sp"),
                             first["t_end"] + 0.5, first["t_end"] + 1.0)

    assert first["text"] not in [line["text"] for line in lines]


def test_every_line_carries_the_role_its_track_implies(spoken):
    """The provenance rule the captions use, so review and the export agree."""
    _, conn, manifest = spoken
    expected_role = {"mic": MIC, "party": PARTY}
    by_text = {u["text"]: u for u in manifest["utterances"]}

    for line in queries.load_transcript(conn, "sp"):
        assert line["role"] == expected_role[by_text[line["text"]]["track"]]


def test_the_fixtures_overlap_shows_both_speakers(spoken):
    """52.6-54.36 s: both people talking. Two tracks, so two roles, so two
    colours in the panel."""
    _, conn, manifest = spoken
    start, end = manifest["overlap_windows"][0]

    lines = queries.lines_in(queries.load_transcript(conn, "sp"), start, end)

    assert {line["role"] for line in lines} == {MIC, PARTY}
    assert len(lines) == 2


def test_lines_are_ordered_by_time_across_speakers(spoken):
    _, conn, _ = spoken

    times = [line["t"] for line in queries.load_transcript(conn, "sp")]

    assert times == sorted(times)


def test_the_transcript_survives_a_stream_with_no_track_map(tmp_path):
    """A single-track recording still has speech; it just has no provenance,
    so the role falls back to §5.8's verdict."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    conn.execute("INSERT INTO streams (id, date, master_path, marker_time_base) "
                 "VALUES ('x', '2026-08-14', 'm.mkv', 'vod')")
    conn.execute("INSERT INTO segments (stream_id, seq, t_start, t_end, text, "
                 "speaker, track, words) VALUES "
                 "('x', 0, 1.0, 2.0, 'hello', 'operator', NULL, '[]')")
    conn.commit()
    try:
        assert queries.load_transcript(conn, "x")[0]["role"] == MIC
    finally:
        conn.close()


# --------------------------------------------------------------------------
# it must not cost a query per candidate (C4)
# --------------------------------------------------------------------------


def test_the_transcript_costs_one_query_for_the_whole_stream(spoken):
    """`load_candidates` puts everything in one response because a round trip
    per `j` press would spend most of C4's four seconds on latency. Loading
    segments per window would reintroduce that cost server-side."""
    cfg, conn, _ = spoken
    for index in range(6):
        add_candidate(conn, index * 10.0, index * 10.0 + 8.0, score=1.0)
    conn.commit()

    # `Connection.execute` is read-only, so the count comes from sqlite's own
    # trace hook — which also catches a query issued anywhere down the call
    # tree rather than only the ones this module makes directly.
    counted: list[str] = []
    conn.set_trace_callback(lambda sql: counted.append(" ".join(sql.split())))
    try:
        found = queries.load_candidates(conn, "sp")
    finally:
        conn.set_trace_callback(None)

    assert len(found) == 6
    segment_queries = [sql for sql in counted if "FROM segments" in sql]
    assert len(segment_queries) == 1, segment_queries


def test_each_candidate_carries_its_own_lines(spoken):
    cfg, conn, manifest = spoken
    early, late = manifest["utterances"][0], manifest["utterances"][-1]
    for index, utterance in enumerate((early, late)):
        add_candidate(conn, utterance["t_start"], utterance["t_end"],
                      score=2.0 - index)
    conn.commit()

    found = queries.load_candidates(conn, "sp")

    texts = [[line["text"] for line in c.transcript] for c in found]
    assert early["text"] in texts[0]
    assert late["text"] in texts[1]
    assert early["text"] not in texts[1]


# --------------------------------------------------------------------------
# the column is conditional
# --------------------------------------------------------------------------


def test_a_stream_with_segments_reports_a_transcript(spoken):
    _, conn, _ = spoken
    assert queries.stream_detail(conn, "sp")["has_transcript"] is True


def test_a_stream_without_segments_does_not(spoken):
    """Phase 2 ships `extract.whisperx.enabled: false`, so this is the case
    every stream is in until the operator turns it on. The review screen uses
    this to decide whether to spend a third of its width on the panel."""
    _, conn, _ = spoken
    conn.execute("INSERT INTO streams (id, date, master_path, marker_time_base) "
                 "VALUES ('quiet', '2026-08-14', 'm.mkv', 'vod')")
    conn.commit()

    assert queries.stream_detail(conn, "quiet")["has_transcript"] is False
    assert queries.load_transcript(conn, "quiet") == []


# --------------------------------------------------------------------------
# the palette is shared with §8.3
# --------------------------------------------------------------------------


def test_role_colours_come_from_the_caption_config():
    """If review and the burned-in captions disagreed, the operator would rate
    a moment believing one person said it and watch the export attribute it to
    the other — and nothing would report the mismatch."""
    cfg = config.load()

    colours = queries.role_colours(cfg)

    assert colours[MIC] == cfg.get("render.captions.styles.mic.colour")
    assert colours[PARTY] == cfg.get("render.captions.styles.party.colour")
    assert colours[MIC] != colours[PARTY]


def test_the_candidates_route_sends_the_palette(spoken):
    cfg, conn, _ = spoken
    conn.commit()
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1",
                    headers={"X-ClipForge": "1"}) as client:
        payload = client.get("/api/streams/sp/candidates").json()

    assert payload["role_colours"][MIC] == cfg.get("render.captions.styles.mic.colour")
    assert payload["stream"]["has_transcript"] is True


# --------------------------------------------------------------------------
# Render from the browser
# --------------------------------------------------------------------------


@pytest.fixture
def client(spoken):
    cfg, conn, _ = spoken
    conn.commit()
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1",
                    headers={"X-ClipForge": "1"}) as test_client:
        yield test_client


def test_render_is_a_job_kind(spoken):
    assert jobs.KIND_RENDER in jobs._RUNNERS
    assert jobs.KIND_RUN in jobs._RUNNERS


def test_an_unknown_job_kind_fails_before_a_thread_starts(spoken):
    cfg, _, _ = spoken
    with pytest.raises(ValueError, match="unknown job kind"):
        jobs.JobRegistry().start(cfg, "sp", "nonsense")


def test_the_render_route_404s_on_an_unknown_stream(client):
    assert client.post("/api/streams/nope/render").status_code == 404


def test_the_render_route_needs_the_header(spoken):
    """It mutates — it writes files and `exports` rows — so `guard.py` requires
    the header that forces a CORS preflight."""
    cfg, _, _ = spoken
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1") as bare:
        assert bare.post("/api/streams/sp/render").status_code == 403


def test_a_render_job_reports_its_kind(client):
    snapshot = client.post("/api/streams/sp/render").json()

    assert snapshot["kind"] == jobs.KIND_RENDER
    assert snapshot["stream_id"] == "sp"


def test_one_job_per_stream_whatever_the_kind(client):
    """A stream is being worked on or it is not. Two live jobs would mean the
    run view tracking two logs for one screen."""
    first = client.post("/api/streams/sp/render").json()

    second = client.post("/api/streams/sp/run")

    # Either the first is still live (409) or it finished first — this render
    # has nothing rated, so it fails fast. Both are legitimate; what must not
    # happen is two live jobs.
    assert second.status_code in (200, 409)
    if second.status_code == 409:
        assert first["id"] in second.json()["detail"]


def test_a_render_with_nothing_approved_fails_with_the_reason(client):
    """Not an exception in a thread nobody is watching: the message lands in
    the job log, which is what the run view is showing."""
    job = client.post("/api/streams/sp/render").json()

    for _ in range(200):
        snapshot = client.get(f"/api/jobs/{job['id']}").json()
        if snapshot["state"] != jobs.RUNNING:
            break
    assert snapshot["state"] == jobs.FAILED
    assert "review it first" in " ".join(snapshot["lines"]).lower() \
        or "nothing rated" in " ".join(snapshot["lines"]).lower()
