"""The add-a-recording screen: drop-to-locate, recent folders, track preview.

Every path here is the fixture's own — nothing asserts a literal location, and
the sizes come from `stat()` rather than from the manifest, because the question
being tested is whether the server agrees with the filesystem.

**The property that matters most is a negative one**: dropping a file must not
upload it. §13.1 puts a master at 40-55 GB and it is already on the server's
disk. `locate` therefore takes a name and a size and nothing else, which is
enforced by its signature rather than by a comment.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from clipforge import config, db
from clipforge.ingest import register
from clipforge.review import app as review_app
from clipforge.review import browse


def make_client(cfg, **kwargs):
    headers = {"X-ClipForge": "1"}
    headers.update(kwargs.pop("headers", {}))
    return TestClient(
        review_app.create_app(cfg), base_url="http://127.0.0.1", headers=headers, **kwargs
    )


@pytest.fixture
def cfg(tmp_path):
    built = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(built.db_path).close()
    return built


@pytest.fixture
def client(cfg):
    with make_client(cfg) as test_client:
        yield test_client


@pytest.fixture
def master(fixture_dir, manifest):
    return fixture_dir / manifest["video"]


# --------------------------------------------------------------------------
# locate — the drop target that moves no bytes
# --------------------------------------------------------------------------


def test_locate_finds_a_file_by_name_and_size(master):
    found = browse.locate(
        master.name, master.stat().st_size, directories=[master.parent]
    )
    assert found.path == str(master)
    assert str(master.parent) in found.searched


def test_locate_refuses_a_name_match_of_the_wrong_size(master):
    """OBS names recordings by timestamp, so the same name on two drives is
    ordinary. Registering the wrong one would attach every signal in the stream
    to footage the operator was not looking at."""
    found = browse.locate(
        master.name, master.stat().st_size + 1, directories=[master.parent]
    )
    assert found.path is None
    assert found.size_mismatch == [str(master)]


def test_locate_accepts_a_missing_size_rather_than_failing(master):
    """A browser always supplies `File.size`, but a caller that cannot should
    still get the name match rather than nothing."""
    found = browse.locate(master.name, None, directories=[master.parent])
    assert found.path == str(master)


def test_locate_reports_where_it_looked(tmp_path, master):
    empty = tmp_path / "nowhere"
    empty.mkdir()
    found = browse.locate("no-such-file.mkv", 1, directories=[empty, master.parent])

    assert found.path is None
    assert found.searched == [str(empty), str(master.parent)]


def test_locate_searches_in_the_order_given(tmp_path, master):
    """The caller decides priority — registered folders, then the open one, then
    config — so `locate` must not reorder them."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / master.name).write_bytes(b"x" * 10)

    size = master.stat().st_size
    assert browse.locate(master.name, size,
                         directories=[decoy, master.parent]).path == str(master)
    assert browse.locate(master.name, 10,
                         directories=[decoy, master.parent]).path == str(decoy / master.name)


def test_locate_ignores_directories_that_are_not_there(tmp_path, master):
    """A recording drive that is unplugged is ordinary, not exceptional."""
    found = browse.locate(
        master.name, master.stat().st_size,
        directories=[tmp_path / "unplugged", None, master.parent],
    )
    assert found.path == str(master)


def test_locate_does_not_recurse(tmp_path, master):
    """A recursive scan of a recording drive is minutes of I/O, and the answer is
    almost always in the first directory."""
    nested = tmp_path / "outer" / "inner"
    nested.mkdir(parents=True)
    (nested / master.name).write_bytes(b"x" * 10)

    found = browse.locate(master.name, 10, directories=[tmp_path / "outer"])
    assert found.path is None


def test_locate_takes_a_bare_name_not_a_path(master):
    """A browser hands over `File.name`, but a caller that sends a whole path
    must not be able to reach outside the directories being searched."""
    found = browse.locate(
        f"../{master.name}", master.stat().st_size, directories=[master.parent]
    )
    assert found.path == str(master)


def test_the_locate_route_never_receives_the_file(client, master):
    """The route's contract, asserted: name and size are the entire payload."""
    response = client.post("/api/browse/locate", json={
        "name": master.name, "size": master.stat().st_size,
        "path": str(master.parent),
    })
    assert response.status_code == 200
    assert response.json()["path"] == str(master)


def test_a_first_drop_has_somewhere_to_look(client):
    """On a fresh database there is no history and nothing has been browsed, so
    without a fallback the very first drop would report an empty list of places
    looked — the worst moment for the feature to do the least."""
    body = client.post("/api/browse/locate", json={
        "name": "no-such-recording.mkv", "size": 1,
    }).json()

    assert body["path"] is None
    assert body["searched"], "a first drop must still say where it looked"
    assert set(body["searched"]) & set(browse.default_search_dirs())


def test_the_fallbacks_come_last(cfg, tmp_path, master):
    """They are a guess about somebody else's OBS configuration. A folder this
    install has actually registered from must win."""
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        register.register_stream(cfg, conn, register.RegisterSpec(master=master))
    finally:
        conn.close()

    with make_client(cfg) as client:
        body = client.post("/api/browse/locate", json={
            "name": master.name, "size": master.stat().st_size,
        }).json()

    assert body["path"] == str(master)
    assert body["searched"][0] == str(master.parent)


def test_the_locate_route_searches_the_open_folder(client, master):
    """Before any stream is registered there is no history to search, so the
    folder the operator is already looking at is what makes the first drop
    work."""
    response = client.post("/api/browse/locate", json={
        "name": master.name, "size": master.stat().st_size,
        "path": str(master.parent),
    })
    assert response.json()["path"] == str(master)


def test_the_locate_route_searches_configured_directories(tmp_path, master):
    """`review.browse.search_dirs` is what makes the very first drop work when
    the recording folder is not where the app happens to be pointed."""
    built = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        f"review.browse.search_dirs=['{master.parent.as_posix()}']",
    ])
    db.open_db(built.db_path).close()
    with make_client(built) as client:
        response = client.post("/api/browse/locate", json={
            "name": master.name, "size": master.stat().st_size,
        })
    assert response.json()["path"] == str(master)


def test_the_locate_route_needs_the_guard_header(cfg, master):
    """It is a POST, and `guard.py` requires the header on anything that is not
    a plain GET — a cross-origin simple POST needs no preflight."""
    with make_client(cfg, headers={"X-ClipForge": ""}) as client:
        client.headers.pop("X-ClipForge", None)
        response = client.post("/api/browse/locate", json={"name": master.name})
    assert response.status_code == 403


def test_the_locate_route_rejects_an_empty_name(client):
    assert client.post("/api/browse/locate", json={"name": "  "}).status_code == 400


# --------------------------------------------------------------------------
# recent locations, derived rather than stored
# --------------------------------------------------------------------------


def test_recent_locations_come_from_registered_masters(cfg, master):
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        assert browse.registered_directories(conn) == []
        register.register_stream(cfg, conn, register.RegisterSpec(master=master))
        assert browse.registered_directories(conn) == [str(master.parent)]
    finally:
        conn.close()


def test_recent_locations_are_deduplicated(cfg, master):
    """Ten streams from one folder is one entry, not ten."""
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        for index in range(3):
            register.register_stream(
                cfg, conn,
                register.RegisterSpec(master=master, stream_id=f"s{index}"),
            )
        assert browse.registered_directories(conn) == [str(master.parent)]
    finally:
        conn.close()


def test_the_browse_route_offers_recent_locations(client, cfg, master):
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        register.register_stream(cfg, conn, register.RegisterSpec(master=master))
    finally:
        conn.close()

    body = client.get("/api/browse", params={"path": str(master.parent)}).json()
    assert str(master.parent) in body["recent"]


def test_locate_prefers_a_folder_recordings_came_from(cfg, tmp_path, master):
    """The best guess by a distance, and it costs no new state to keep."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / master.name).write_bytes(b"x" * 10)

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        register.register_stream(cfg, conn, register.RegisterSpec(master=master))
    finally:
        conn.close()

    with make_client(cfg) as client:
        response = client.post("/api/browse/locate", json={
            "name": master.name, "size": master.stat().st_size, "path": str(decoy),
        })
    assert response.json()["path"] == str(master)


# --------------------------------------------------------------------------
# the listing gains a modification time, for newest-first
# --------------------------------------------------------------------------


def test_entries_carry_a_modification_time(client, master):
    body = client.get("/api/browse", params={"path": str(master.parent)}).json()
    entry = next(e for e in body["entries"] if e["name"] == master.name)

    assert entry["mtime"] == pytest.approx(master.stat().st_mtime, abs=1.0)
    assert entry["size"] == master.stat().st_size


def test_directories_carry_no_size_or_time(client, tmp_path):
    (tmp_path / "sub").mkdir()
    body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    entry = next(e for e in body["entries"] if e["name"] == "sub")

    assert entry["is_dir"] and entry["size"] is None and entry["mtime"] is None


# --------------------------------------------------------------------------
# the §4.2 track preview — the thing that answers "where does the mic go?"
# --------------------------------------------------------------------------


def test_preflight_reports_the_tracks_inside_the_file(client, master, manifest):
    """OBS records §4.2's four tracks INTO THE ONE FILE, which is why there is
    no second upload for mic or party audio. The preview says so, from the
    container itself rather than from a promise."""
    body = client.post("/api/register/preflight", json={"master": str(master)}).json()
    audio = body["audio"]

    assert audio["track_count"] == manifest["expected_audio_track_count"]
    assert set(audio["roles"]) == set(manifest["expected_track_map"])
    for role, index in manifest["expected_track_map"].items():
        assert audio["roles"][role] == index


def test_the_track_preview_happens_before_anything_is_written(client, master, cfg):
    client.post("/api/register/preflight", json={"master": str(master)})

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        assert conn.execute("SELECT COUNT(*) FROM streams").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_file_with_too_few_tracks_is_visible_as_such(client, tmp_path, needs_ffmpeg):
    """§4.2: "mic RMS analysis is garbage if game audio is mixed in [...] This is
    unrecoverable after the fact." `audio_split` refuses two or three tracks
    outright, and the operator should not have to reach that stage to find out.
    """
    from clipforge import ffmpeg

    single = tmp_path / "single.mkv"
    ffmpeg.run([
        ffmpeg.require("ffmpeg"), "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "flac", str(single),
    ])

    body = client.post("/api/register/preflight", json={"master": str(single)}).json()
    audio = body["audio"]
    assert audio["track_count"] == 1
    # One track maps to `mic` with a warning — §4.2's contamination case.
    assert "party" not in audio["roles"]
    assert audio["warnings"]


def test_an_unreadable_file_does_not_break_the_preflight(client, tmp_path):
    """Refusing to show a preflight because the extra detail is unavailable
    would be worse than showing it without.

    ffprobe exits non-zero on a non-container, which raises `FFmpegError` —
    neither a `ProbeError` nor an `OSError`, and so uncaught until a browser
    test produced a 500 from this route.
    """
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"not a container")

    response = client.post("/api/register/preflight", json={"master": str(broken)})
    assert response.status_code == 200
    assert response.json()["audio"]["error"]


def test_the_audio_error_is_a_sentence_not_an_ffprobe_invocation(client, tmp_path):
    """VERIFIED in a browser, which is where this was noticed: the raw text is
    an ffprobe command line complete with argv. That is the right thing to keep
    and the wrong thing to put in front of somebody who has just dragged in the
    wrong file."""
    broken = tmp_path / "not-a-video.mkv"
    broken.write_bytes(b"nope")

    audio = client.post(
        "/api/register/preflight", json={"master": str(broken)}
    ).json()["audio"]

    assert broken.name in audio["error"]
    assert "ffprobe" not in audio["error"].lower()
    assert "command:" not in audio["error"]
    # The detail is kept, just not shown as the headline.
    assert audio["detail"]


def test_the_cli_preflight_does_not_pay_for_the_probe(cfg, master):
    """`clipforge register` runs `probe` moments later, where the same map is
    resolved and stored. Only the shell needs it early."""
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        pre = register.preflight(cfg, conn, register.RegisterSpec(master=master))
        assert pre.audio is None
        assert pre.to_json()["audio"] is None
    finally:
        conn.close()


def test_the_summary_matches_what_probe_would_store(cfg, master):
    """Two readers of one fact. If they disagreed, the screen would promise a
    track layout the pipeline then refused."""
    from clipforge.ingest import probe

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        result = register.register_stream(
            cfg, conn, register.RegisterSpec(master=master))
        summary = register.audio_summary(master, cfg, cfg.ffprobe_override)
        analysed = probe.analyse(master, cfg, cfg.ffprobe_override)
    finally:
        conn.close()

    assert summary["roles"] == analysed.track_map.roles
    assert summary["source"] == analysed.track_map.source
    assert result.stream_id


# --------------------------------------------------------------------------
# the page itself
# --------------------------------------------------------------------------


def test_the_add_view_has_every_element_its_script_addresses(client):
    """The same check commit 25a added for the review screen: the views address
    the DOM as `$("some-id")`, and a missing element is `null` that fails at the
    moment a key is pressed rather than in any route test."""
    import re

    script = client.get("/static/add.js").text
    page = client.get("/static/index.html").text

    for element_id in sorted(set(re.findall(r'\$\("([a-z0-9-]+)"\)', script))):
        assert f'id="{element_id}"' in page, f"add.js addresses missing #{element_id}"


def strip_comments(source: str) -> str:
    """Code only. The comments in `add.js` NAME the APIs it must not use, in
    order to explain why — so a scan that included them would fail on its own
    documentation."""
    import re

    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?<!:)//[^\n]*", "", source)


def test_the_add_script_never_reads_the_dropped_file(client):
    """The property the whole design rests on. A `FileReader`, an `arrayBuffer`
    or a `FormData` in this module would mean a 40 GB copy to localhost."""
    code = strip_comments(client.get("/static/add.js").text)

    for forbidden in ("FileReader", "arrayBuffer", "FormData", ".text()", ".stream()"):
        assert forbidden not in code, f"add.js must not use {forbidden}"


def test_the_comment_stripper_does_not_eat_the_code(client):
    """Guarding the guard: a stripper that removed everything would make the
    test above vacuous."""
    code = strip_comments(client.get("/static/add.js").text)
    assert "async function dropped" in code
    assert "/api/browse/locate" in code


def test_the_drop_handler_sends_only_metadata(client):
    """`file.name` and `file.size` are read; `file` itself must not be."""
    code = strip_comments(client.get("/static/add.js").text)
    assert "file.name" in code and "file.size" in code
    assert "body: file" not in code
