"""The review server: payload shape, ratings, and the §7.5 instrumentation."""

from __future__ import annotations

import json
import re
import shutil

import pytest
from fastapi.testclient import TestClient

from clipforge import config, db, paths
from clipforge.pipeline import runner
from clipforge.review import app as review_app
from clipforge.review import queries
from tests.conftest import hand_rows
from tests.fixtures.make_fixture import FixtureSpec, generate, load_manifest


@pytest.fixture(scope="session")
def processed(tmp_path_factory):
    """A fully processed stream: proxy, signals, markers, candidates."""
    directory = generate(FixtureSpec(name="long", duration_s=600.0))
    manifest = load_manifest(directory)

    root = tmp_path_factory.mktemp("review")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    anchor = json.loads((directory / manifest["anchor"]).read_text())["record_start_epoch_ms"]
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, marker_time_base, "
        "record_start_epoch_ms) VALUES ('fx', '2026-08-14', 'Fixture', ?, 'epoch', ?)",
        (str(directory / manifest["video"]), anchor),
    )
    sp = paths.StreamPaths(cfg.data_root, "fx").ensure()
    shutil.copy2(directory / manifest["markers_file"], sp.markers_jsonl)

    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    engine.execute(engine.plan())
    conn.close()
    return cfg, manifest


@pytest.fixture
def client(processed):
    cfg, _ = processed
    # TestClient defaults to `Host: testserver`, which guard.py refuses — a
    # request arriving under a name this server does not answer to is how DNS
    # rebinding reaches a local server. The header goes on every request for the
    # same reason app.js sends it.
    with TestClient(
        review_app.create_app(cfg),
        base_url="http://127.0.0.1",
        headers={"X-ClipForge": "1"},
    ) as test_client:
        yield test_client


@pytest.fixture
def conn(processed):
    cfg, _ = processed
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    yield connection
    connection.close()


# --------------------------------------------------------------------------
# the rubric (the learning layer -- no § reference)
# --------------------------------------------------------------------------


@pytest.fixture
def rubric_client(tmp_path):
    """Its own database, deliberately not the session-scoped `processed` one.

    The rubric tests write rubrics, and `processed` is shared and mutable
    across this whole file — `test_metrics_include_the_approval_rate` already
    asserts a range rather than a value for exactly that reason. "Nothing has
    been written yet" is not a range, so it needs a database nothing else has
    touched.
    """
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1",
                    headers={"X-ClipForge": "1"}) as test_client:
        yield cfg, test_client


def test_an_unwritten_rubric_says_so_rather_than_returning_nothing(rubric_client):
    """Three states, not two. "Nothing written yet" is a different answer from
    "the request failed", and the editor prefills from this."""
    _, client = rubric_client
    body = client.get("/api/rubric").json()
    assert body["absent"] is True
    assert body["text"] == "" and body["version"] == 0


def test_writing_a_rubric_appends_a_version(rubric_client):
    _, client = rubric_client
    written = client.post("/api/rubric", json={"text": "her reaction over mine"}).json()
    assert (written["ok"], written["unchanged"], written["version"]) == (True, False, 1)

    current = client.get("/api/rubric").json()
    assert current["absent"] is False
    assert current["text"] == "her reaction over mine"
    assert current["version"] == 1


def test_saving_the_same_text_does_not_mint_a_version(rubric_client):
    """The editor prefills with the current rubric so the operator amends
    rather than retypes -- so opening the summary and clicking Save would
    otherwise write a version that says nothing new, permanently, since §9.1's
    reasoning makes every version permanent."""
    _, client = rubric_client
    client.post("/api/rubric", json={"text": "silences before a punchline"})
    again = client.post("/api/rubric", json={"text": "  silences before a punchline  "})

    assert again.json()["unchanged"] is True
    assert again.json()["version"] == 1
    assert client.get("/api/rubric").json()["version"] == 1


def test_an_empty_rubric_is_refused_with_a_reason(rubric_client):
    _, client = rubric_client
    response = client.post("/api/rubric", json={"text": "   "})
    assert response.status_code == 400
    assert "deletion" in response.json()["detail"]


def test_writing_a_rubric_needs_the_header(rubric_client):
    """The guard covers every mutating route; asserted here because this is a
    new one and "covered by middleware" is a claim worth checking once."""
    cfg, _ = rubric_client
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1") as bare:
        assert bare.post("/api/rubric", json={"text": "x"}).status_code == 403


def test_a_rating_body_without_a_rating_is_a_400_not_a_500(client, conn):
    """`api_rate` caught only ValueError while `api_nudge` beside it caught
    KeyError too, so `int(body["rating"])` on a body without one surfaced as an
    unhandled 500. Unreachable while the only client always sent one."""
    candidate_id = conn.execute(
        "SELECT id FROM candidates WHERE is_current = 1 LIMIT 1").fetchone()["id"]
    response = client.post(f"/api/candidates/{candidate_id}/rating", json={"note": "hi"})
    assert response.status_code == 400


def test_a_note_is_stored_with_the_rating_and_survives_a_bare_re_rate(client, conn):
    """§7.3's `n`. `save_rating` COALESCEs the note, so re-rating without one
    keeps it -- the same "absent means no opinion" rule the adjusted window
    follows."""
    candidate_id = conn.execute(
        "SELECT id FROM candidates WHERE is_current = 1 LIMIT 1").fetchone()["id"]

    client.post(f"/api/candidates/{candidate_id}/rating",
                json={"rating": 2, "note": "the pause before it lands"})
    client.post(f"/api/candidates/{candidate_id}/rating", json={"rating": 1})

    row = conn.execute(
        "SELECT rating, note FROM ratings WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    assert (row["rating"], row["note"]) == (1, "the pause before it lands")


# --------------------------------------------------------------------------
# the page and its assets are self-contained
# --------------------------------------------------------------------------


def test_index_serves(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ClipForge" in response.text


def test_no_external_resources(client):
    """§2.2 says avoid heavy frameworks; a dedicated streaming PC may have no
    internet at all, so nothing may be fetched from a CDN."""
    page = client.get("/").text
    assert "/static/app.css" in page
    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "https://" not in page


def test_every_module_is_served_and_local(client):
    """ES modules, no bundler: every import has to resolve from /static, or the
    page silently loads nothing. Walked from main.js rather than listed, so a
    new module cannot be added without this noticing."""
    seen: set[str] = set()
    queue = ["/static/main.js"]
    while queue:
        url = queue.pop()
        if url in seen:
            continue
        seen.add(url)
        response = client.get(url)
        assert response.status_code == 200, f"{url} is not served"
        for target in re.findall(r'^\s*(?:import|export)[^\n]*?from\s+"([^"]+)"',
                                 response.text, re.MULTILINE):
            assert target.startswith("./"), f"{url} imports {target!r} from outside /static"
            queue.append(f"/static/{target[2:]}")

    assert {"/static/main.js", "/static/api.js", "/static/router.js",
            "/static/review.js"} <= seen


# search.js was missing from this list and from the one in
# test_every_element_the_js_reaches_for_exists, so commit 40's module was
# covered by neither check. Both are hand-maintained; there is no way to
# discover them, since the point is to name what must be scanned.
MODULES = ("main.js", "api.js", "router.js", "shell.js", "library.js",
           "add.js", "run.js", "review.js", "search.js")


def _strip_literals(source: str) -> str:
    """Blank out comments and string bodies, keeping length and newlines.

    Enough to scan declarations without a JavaScript parser. Regex literals are
    left alone — one containing the word `const` would be a false positive, and
    none does.
    """
    out = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        pair = source[i:i + 2]
        if pair == "//":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
        elif pair == "/*":
            while i < n and source[i:i + 2] != "*/":
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        elif ch in "\"'`":
            quote = ch
            out.append(" ")
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    out.append(" ")
                    i += 1
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append(" ")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def test_no_module_declares_the_same_name_twice_in_one_scope(client):
    """MEASURED THE HARD WAY, in commit 30.

    A second `const span` inside `drawSpark` — the name was already the
    sparkline's dB range — is a SyntaxError, and an ES module that does not
    parse does not load *at all*. The entire review screen went blank while
    every test in this file still passed, because none of them execute the JS;
    the route tests only prove the server is right. C4 makes that screen the
    critical path, so the gap is worth closing.

    Written in Python rather than shelling out to node, because there is no node
    on the machine this project is built on and a check that always skips is not
    a check. Deliberately conservative: same brace depth only, so it reports the
    failure that happened and cannot fire on legitimate shadowing.

    A declaration inside PARENTHESES is skipped, because `for (const x of ...)`
    binds `x` to the loop's own scope — two such loops side by side are legal
    and were a false positive until commit 34 hit it.
    """
    for name in MODULES:
        source = _strip_literals(client.get(f"/static/{name}").text)

        depth = 0
        parens = 0
        seen: dict[int, set[str]] = {}
        for token in re.finditer(
            r"[{}()]|\b(?:const|let)\s+([A-Za-z_$][\w$]*)", source
        ):
            text = token.group(0)
            if text == "{":
                depth += 1
            elif text == "}":
                seen.pop(depth, None)
                depth -= 1
            elif text == "(":
                parens += 1
            elif text == ")":
                parens = max(parens - 1, 0)
            elif parens == 0:
                names = seen.setdefault(depth, set())
                assert token.group(1) not in names, (
                    f"{name} declares {token.group(1)!r} twice at the same brace "
                    f"depth — that is a SyntaxError and the module will not load"
                )
                names.add(token.group(1))


def test_the_scope_checker_allows_two_loops_over_the_same_variable():
    """The false positive that commit 34 hit. `for (const name of ...)` twice in
    one scope is legal JS — each binds to its own loop — and a checker that
    refuses it teaches people to rename around it rather than trust it."""
    legal = (
        'for (const name of ["a", "b"]) { use(name); }\n'
        'for (const name of ["c", "d"]) { use(name); }\n'
    )
    depth, parens, seen = 0, 0, {}
    for token in re.finditer(
        r"[{}()]|\b(?:const|let)\s+([A-Za-z_$][\w$]*)", _strip_literals(legal)
    ):
        text = token.group(0)
        if text == "{":
            depth += 1
        elif text == "}":
            seen.pop(depth, None)
            depth -= 1
        elif text == "(":
            parens += 1
        elif text == ")":
            parens = max(parens - 1, 0)
        elif parens == 0:
            names = seen.setdefault(depth, set())
            assert token.group(1) not in names, "false positive on a for-of loop"
            names.add(token.group(1))


def test_the_scope_checker_actually_catches_the_bug_it_was_written_for():
    """A guard on the guard: the checker above is only worth having if a
    re-declaration really does trip it."""
    broken = "function f() {\n  const span = 1;\n  const span = 2;\n}"
    depth, seen, caught = 0, {}, False
    for token in re.finditer(r"[{}]|\b(?:const|let)\s+([A-Za-z_$][\w$]*)",
                             _strip_literals(broken)):
        if token.group(0) == "{":
            depth += 1
        elif token.group(0) == "}":
            seen.pop(depth, None)
            depth -= 1
        elif token.group(1) in seen.setdefault(depth, set()):
            caught = True
        else:
            seen[depth].add(token.group(1))
    assert caught

    # ...and does not fire on the same name in two different functions.
    fine = "function a() {\n const x = 1;\n}\nfunction b() {\n const x = 2;\n}"
    depth, seen = 0, {}
    for token in re.finditer(r"[{}]|\b(?:const|let)\s+([A-Za-z_$][\w$]*)",
                             _strip_literals(fine)):
        if token.group(0) == "{":
            depth += 1
        elif token.group(0) == "}":
            seen.pop(depth, None)
            depth -= 1
        else:
            assert token.group(1) not in seen.setdefault(depth, set())
            seen[depth].add(token.group(1))


def test_every_element_the_js_reaches_for_exists(client):
    """The one coupling a redesign can break silently.

    The views address the DOM as `$("some-id")` and a missing element is
    `null`, so a renamed or dropped id fails at the moment the operator presses
    a key — not at load, and never in a way the JSON route tests would see.
    The ids are read out of the modules rather than listed here, so this covers
    ids added later without anyone remembering to update it.
    """
    page = client.get("/").text
    wanted: dict[str, set[str]] = {}
    for name in MODULES:
        source = client.get(f"/static/{name}").text
        for element_id in re.findall(r'\$\("([a-z0-9-]+)"\)', source):
            wanted.setdefault(element_id, set()).add(name)

    missing = {
        element_id: sorted(files)
        for element_id, files in wanted.items()
        if f'id="{element_id}"' not in page
    }
    assert not missing, f"the page has no element for: {missing}"
    # A guard on the guard: if the regex ever stops matching, this test would
    # pass by checking nothing.
    assert len(wanted) > 20


# --------------------------------------------------------------------------
# the candidate payload
# --------------------------------------------------------------------------


def test_candidates_arrive_in_one_payload(client):
    """C4's four seconds per candidate cannot afford a round trip per keypress,
    so everything the screen needs is loaded once."""
    data = client.get("/api/streams/fx/candidates").json()
    assert data["candidates"]
    first = data["candidates"][0]
    for key in ("t_start", "t_end", "t_peak", "score", "contributions",
                "sparklines", "marker_anchored", "rating",
                "preview_url", "thumbstrip_url"):
        assert key in first


def test_candidates_are_ranked(client):
    scores = [c["score"] for c in client.get("/api/streams/fx/candidates").json()["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_only_current_candidates_are_served(client, conn, processed):
    """A superseded generation must not appear in review."""
    cfg, _ = processed
    conn.execute("UPDATE candidates SET is_current = 0 WHERE stream_id='fx'")
    try:
        assert client.get("/api/streams/fx/candidates").json()["candidates"] == []
    finally:
        conn.execute("UPDATE candidates SET is_current = 1 WHERE stream_id='fx'")


def test_contributions_and_context_are_separated(client):
    """Signal contributions drive the bars; the dB context is prose beneath."""
    first = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    assert "mic_rms" in first["contributions"]
    assert all(not k.startswith("_") for k in first["contributions"])
    assert "total_raw" in first["context"]
    assert "total_smoothed" in first["context"]


def test_sparkline_is_drawn_from_stored_signals(client):
    """§7.2's waveform asset, as data rather than as a PNG per candidate.

    See `clipforge/review/previews.py` for why no file is written: this payload
    plus `review.js` already draws it, and a PNG would be a second copy of the
    same numbers orphaned by every re-score.
    """
    first = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    kinds = [track["kind"] for track in first["sparklines"]]
    assert "mic_rms" in kinds
    for track in first["sparklines"]:
        assert 2 < len(track["points"]) <= queries.SPARKLINE_POINTS
        # The role is the kind name -- extract/features.py writes one series per
        # §4.2 role -- so no track_roles lookup is involved and none can drift.
        assert track["kind"].startswith(track["role"] + "_")
    low, high = first["sparkline_range"]
    assert low < high


def test_every_sparkline_track_shares_one_range(client, conn):
    """§7.2 asks for "mic + party"; per-track scaling would make them a lie.

    Two tracks normalised independently draw a party mic sitting at its noise
    floor at the same height as an operator shouting, so the one comparison the
    picture exists to support is the one it cannot show. The range therefore
    spans EVERY track, which this asserts by arithmetic rather than by a number:
    no track may hold a sample outside the range sent with it.
    """
    candidates = client.get("/api/streams/fx/candidates").json()["candidates"]
    checked = 0
    for candidate in candidates:
        if len(candidate["sparklines"]) < 1:
            continue
        low, high = candidate["sparkline_range"]
        assert low < high
        for track in candidate["sparklines"]:
            assert min(track["points"]) >= low - 0.01
            assert max(track["points"]) <= high + 0.01

        # The TOP of the range is attained by some track, which is what makes it
        # the union rather than an arbitrary pair of bounds.
        every = [v for track in candidate["sparklines"] for v in track["points"]]
        assert max(every) == pytest.approx(high, abs=0.01)
        # The BOTTOM deliberately is not. `points` are bucket maxima -- an
        # envelope has to show peaks, and averaging would flatten the transients
        # being looked for -- while `low` is the true minimum of the raw samples.
        # So the drawn line never reaches the floor, and the floor is still the
        # honest dB range of the window rather than the range of the envelope.
        assert min(every) >= low
        checked += 1
    assert checked, "no candidate carried a sparkline"


def test_preview_urls_are_served_and_fetchable(client):
    """§7.2's assets reach the browser, and only their names travel.

    The stored path is `streams/<id>/previews/<name>`; the URL carries the name
    alone so the route can resolve it under the stream's own directory and check
    the result is still inside it.
    """
    candidates = client.get("/api/streams/fx/candidates").json()["candidates"]
    first = candidates[0]
    assert first["preview_url"] and first["thumbstrip_url"]
    assert "/" not in first["preview_url"].rsplit("/", 1)[-1]

    clip = client.get(first["preview_url"])
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/webm"
    assert client.get(first["thumbstrip_url"]).status_code == 200


def test_the_screen_still_works_with_no_preview_assets(client, conn):
    """`previews` is a dependency of nothing, so review must work without it.

    Phase 1's seek-and-hold is the fallback, and `review.js` decides on these
    two fields being null — so a stream processed before the stage existed, or
    one with `previews.enabled: false`, must produce null rather than a URL that
    404s on every candidate.
    """
    # Restored by hand rather than by rollback: `db.connect` uses
    # isolation_level=None, so this UPDATE is already committed and a rollback
    # would be a silent no-op that left the session fixture broken for whichever
    # test ran next. Same reason `test_only_current_candidates_are_served`
    # restores explicitly.
    saved = conn.execute(
        "SELECT id, preview_path, thumbstrip_path FROM candidates "
        "WHERE stream_id = 'fx'").fetchall()
    conn.execute("UPDATE candidates SET preview_path = NULL, thumbstrip_path = NULL "
                 "WHERE stream_id = 'fx'")
    try:
        for candidate in client.get("/api/streams/fx/candidates").json()["candidates"]:
            assert candidate["preview_url"] is None
            assert candidate["thumbstrip_url"] is None
    finally:
        conn.executemany(
            "UPDATE candidates SET preview_path = ?, thumbstrip_path = ? WHERE id = ?",
            [(r["preview_path"], r["thumbstrip_path"], r["id"]) for r in saved],
        )


def test_preview_route_refuses_to_escape_the_previews_directory(client):
    """Loopback is not a security boundary here — see review/guard.py.

    The payload only ever emits names this server wrote, but the route is
    reachable directly.
    """
    for attempt in ("..%2f..%2fclipforge.db", "..%5c..%5cclipforge.db"):
        assert client.get(f"/media/fx/preview/{attempt}").status_code == 404


def test_marker_anchored_candidates_are_flagged(client):
    """§7.4's fourth section: the operator marked these deliberately, so they
    are the safety net when the weights rank them low."""
    candidates = client.get("/api/streams/fx/candidates").json()["candidates"]
    assert any(c["marker_anchored"] for c in candidates)
    flagged = next(c for c in candidates if c["marker_anchored"])
    assert flagged["markers"] or flagged["contributions"].get("marker_definite")


def test_stream_warnings_reach_the_screen(processed, tmp_path):
    """The §4.2 contamination notice belongs in front of the operator."""
    cfg, _ = processed
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    connection.execute(
        "UPDATE streams SET audio_track_map = ? WHERE id='fx'",
        (json.dumps({"roles": {"mic": 0}, "warnings": ["single track, mic is contaminated"]}),),
    )
    detail = queries.stream_detail(connection, "fx")
    connection.close()
    assert detail["warnings"] == ["single track, mic is contaminated"]


def test_unknown_stream_is_404(client):
    assert client.get("/api/streams/nope/candidates").status_code == 404


# --------------------------------------------------------------------------
# rating (§7.3, §7.5)
# --------------------------------------------------------------------------


def test_rating_is_saved_with_review_ms(client, conn):
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    response = client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 2, "review_ms": 3400},
    )
    assert response.status_code == 200

    row = conn.execute(
        "SELECT rating, review_ms, rating_source FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert (row["rating"], row["review_ms"], row["rating_source"]) == (2, 3400, "operator")


def test_rating_twice_updates_rather_than_failing(client, conn):
    """Changing your mind is normal; ratings has candidate_id as its primary
    key, so an insert would otherwise conflict."""
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][1]
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 0})
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 2})

    rows = conn.execute(
        "SELECT rating FROM ratings WHERE candidate_id = ?", (candidate["id"],)
    ).fetchall()
    assert len(rows) == 1 and rows[0]["rating"] == 2


def test_an_operator_rating_overrides_an_inherited_one(client, conn):
    """§14's tuning input counts operator ratings only, so a human keypress has
    to clear the inherited flag."""
    listing = client.get("/api/streams/fx/candidates").json()["candidates"]
    candidate, source = listing[2], listing[3]
    conn.execute(
        "INSERT OR REPLACE INTO ratings (candidate_id, rating, rating_source, inherited_from) "
        "VALUES (?, 1, 'inherited', ?)",
        (candidate["id"], source["id"]),
    )
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 2})

    row = conn.execute(
        "SELECT rating, rating_source, inherited_from FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert (row["rating"], row["rating_source"], row["inherited_from"]) == (2, "operator", None)


@pytest.mark.parametrize("bad", [-1, 3, 99])
def test_invalid_ratings_are_rejected(client, bad):
    """§7.3 defines exactly three: 0 skip, 1 maybe, 2 clip it."""
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    assert client.post(
        f"/api/candidates/{candidate['id']}/rating", json={"rating": bad}
    ).status_code == 400


def test_ratings_show_up_on_the_next_load(client):
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 1})
    reloaded = client.get("/api/streams/fx/candidates").json()["candidates"]
    assert next(c for c in reloaded if c["id"] == candidate["id"])["rating"] == 1


# --------------------------------------------------------------------------
# §7.3's nudge keys — GUESSES gap 1
# --------------------------------------------------------------------------


def _unrated(client, conn, index: int = 0):
    """A candidate with no rating row, so nudge tests start from a clean slate.

    The index wraps: how many candidates the fixture produces is decided by
    `score.peak.target_candidates_per_hour` against the fixture's duration, so a
    literal index would be an assertion about a config value in disguise.
    Wrapping is safe because the ratings row is deleted, which clears any
    adjusted window a previous test left.
    """
    found = client.get("/api/streams/fx/candidates").json()["candidates"]
    assert found, "the fixture stream should have candidates"
    candidate = found[index % len(found)]
    conn.execute("DELETE FROM ratings WHERE candidate_id = ?", (candidate["id"],))
    return candidate


def test_the_payload_carries_the_nudge_step_from_config(client):
    """The browser used to hold its own copy of `target_ms` and drift from the
    config silently. The step size is told, not guessed."""
    cfg = config.load()
    data = client.get("/api/streams/fx/candidates").json()
    assert data["nudge_step_s"] == float(cfg.get("review.nudge_step_s"))


def test_a_rating_carries_the_nudged_window(client, conn):
    candidate = _unrated(client, conn)
    start, end = candidate["t_start"] + 1.5, candidate["t_end"] - 2.0

    assert client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 2, "adjusted_start": start, "adjusted_end": end},
    ).status_code == 200

    row = conn.execute(
        "SELECT adjusted_start, adjusted_end FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert (row["adjusted_start"], row["adjusted_end"]) == (start, end)


def test_the_stored_window_comes_back_in_the_payload(client, conn):
    """Otherwise reopening a stream shows the detector's window and silently
    hides the operator's own trim — and `space` would play the wrong bounds."""
    candidate = _unrated(client, conn, 1)
    start, end = candidate["t_start"] + 2.0, candidate["t_end"]
    client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 1, "adjusted_start": start, "adjusted_end": end},
    )

    reloaded = client.get("/api/streams/fx/candidates").json()["candidates"]
    found = next(c for c in reloaded if c["id"] == candidate["id"])
    assert (found["adjusted_start"], found["adjusted_end"]) == (start, end)
    # The detector's own window is still reported alongside, unchanged.
    assert found["t_start"] == candidate["t_start"]


def test_re_rating_without_nudging_preserves_an_existing_trim(client, conn):
    """Absent means "no opinion", not "restore the detector's window"."""
    candidate = _unrated(client, conn, 2)
    start, end = candidate["t_start"] + 1.0, candidate["t_end"] - 1.0
    client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 1, "adjusted_start": start, "adjusted_end": end},
    )
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 2})

    row = conn.execute(
        "SELECT rating, adjusted_start, adjusted_end FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert row["rating"] == 2
    assert (row["adjusted_start"], row["adjusted_end"]) == (start, end)


def test_a_window_shorter_than_min_window_s_is_accepted(client, conn):
    """THE assertion that the measurement is not circular.

    §17 tunes `min_window_s` against how the operator nudges boundaries, so
    refusing a nudge that leaves its range would mean the operator could never
    record "I wanted this shorter than 8 seconds" — which is precisely the
    observation §17 asks for. A window under the minimum is a finding.
    """
    cfg = config.load()
    minimum = float(cfg.get("score.window.min_window_s"))
    candidate = _unrated(client, conn, 3)
    start = candidate["t_start"]
    end = start + minimum / 4.0

    assert client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 2, "adjusted_start": start, "adjusted_end": end},
    ).status_code == 200

    row = conn.execute(
        "SELECT adjusted_start, adjusted_end FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert row["adjusted_end"] - row["adjusted_start"] < minimum


@pytest.mark.parametrize("start,end", [
    (50.0, 50.0),        # zero length
    (60.0, 40.0),        # inverted
    (-5.0, 20.0),        # before the recording
    (10.0, 10_000.0),    # past the end of the recording
])
def test_an_impossible_window_is_refused_and_writes_nothing(client, conn, start, end):
    candidate = _unrated(client, conn, 4)
    response = client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 2, "adjusted_start": start, "adjusted_end": end},
    )
    assert response.status_code == 400
    assert conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE candidate_id = ?", (candidate["id"],)
    ).fetchone()[0] == 0


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_window_is_refused(client, conn, literal):
    """Sent as a raw body, because that is the only way one can arrive.

    `json.dumps` refuses these outright, but `json.loads` — which is what
    Starlette's `request.json()` calls — accepts all three by default. So a
    non-finite boundary is reachable from the wire and unreachable from the
    test client's `json=` argument, and `math.isfinite` is what stops it
    becoming a NULL-ish REAL in `ratings`.
    """
    candidate = _unrated(client, conn, 4)
    body = (f'{{"rating": 2, "adjusted_start": {literal}, "adjusted_end": 20.0}}')
    response = client.post(
        f"/api/candidates/{candidate['id']}/rating",
        content=body, headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE candidate_id = ?", (candidate["id"],)
    ).fetchone()[0] == 0


def test_half_a_window_is_refused(client, conn):
    candidate = _unrated(client, conn, 5)
    assert client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 2, "adjusted_start": candidate["t_start"]},
    ).status_code == 400


def test_the_nudge_metric_records_direction_and_the_clamp_flags(client, conn):
    """§17 asks "how often", but a count cannot tune anything on its own.

    The clamp flags are the field that moves a bound: a window that came out at
    exactly `max_window_s` and then got extended IS `max_window_s` being too low.
    """
    cfg = config.load()
    maximum = float(cfg.get("score.window.max_window_s"))
    candidate = _unrated(client, conn, 6)

    # Make the detector's window exactly max_window_s, which is what the clamp
    # produces, so the flag has something true to report. Restored afterwards:
    # the processed stream is a session fixture and every other test in this
    # module reads it.
    original = conn.execute(
        "SELECT t_start, t_end, t_peak FROM candidates WHERE id = ?",
        (candidate["id"],),
    ).fetchone()
    conn.execute(
        "UPDATE candidates SET t_start = ?, t_end = ?, t_peak = ? WHERE id = ?",
        (100.0, 100.0 + maximum, 110.0, candidate["id"]),
    )
    try:
        response = client.post(
            f"/api/candidates/{candidate['id']}/nudge",
            json={"adjusted_start": 98.0, "adjusted_end": 100.0 + maximum + 3.0,
                  "presses": 10},
        )
        assert response.status_code == 200
    finally:
        conn.execute(
            "UPDATE candidates SET t_start = ?, t_end = ?, t_peak = ? WHERE id = ?",
            (original["t_start"], original["t_end"], original["t_peak"],
             candidate["id"]),
        )

    row = conn.execute(
        "SELECT stream_id, value, meta FROM tool_metrics WHERE metric = ? "
        "ORDER BY id DESC LIMIT 1", (queries.NUDGE_METRIC,),
    ).fetchone()
    meta = json.loads(row["meta"])
    assert row["stream_id"] == "fx"
    assert row["value"] == pytest.approx(5.0)          # 2 s earlier + 3 s later
    assert meta["start_delta"] == pytest.approx(-2.0)
    assert meta["end_delta"] == pytest.approx(3.0)
    assert meta["presses"] == 10
    assert meta["at_max"] is True
    assert meta["at_min"] is False
    assert meta["keeps_peak"] is True


def test_the_nudge_route_never_creates_a_rating(client, conn):
    """`ratings.rating` is NOT NULL, so storing an unrated nudge there would
    mean inventing a rating — and a fabricated rating corrupts §14's
    `signal_firing_rate_by_rating`, which §17 calls the primary tuning input."""
    candidate = _unrated(client, conn, 7)
    client.post(
        f"/api/candidates/{candidate['id']}/nudge",
        json={"adjusted_start": candidate["t_start"] + 1,
              "adjusted_end": candidate["t_end"], "presses": 2},
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE candidate_id = ?", (candidate["id"],)
    ).fetchone()[0] == 0


def test_a_nudge_that_drops_the_peak_is_allowed_and_flagged(client, conn):
    """`candidates` forbids this by CHECK; `ratings` does not, and trimming to
    the second half of a moment is a legitimate thing to want."""
    candidate = _unrated(client, conn, 8)
    peak = candidate["t_peak"]
    client.post(
        f"/api/candidates/{candidate['id']}/nudge",
        json={"adjusted_start": peak + 0.5, "adjusted_end": candidate["t_end"],
              "presses": 1},
    )
    meta = json.loads(conn.execute(
        "SELECT meta FROM tool_metrics WHERE metric = ? ORDER BY id DESC LIMIT 1",
        (queries.NUDGE_METRIC,),
    ).fetchone()["meta"])
    assert meta["keeps_peak"] is False


def test_an_impossible_nudge_is_refused_and_records_nothing(client, conn):
    candidate = _unrated(client, conn)
    before = conn.execute(
        "SELECT COUNT(*) FROM tool_metrics WHERE metric = ?", (queries.NUDGE_METRIC,)
    ).fetchone()[0]

    assert client.post(
        f"/api/candidates/{candidate['id']}/nudge",
        json={"adjusted_start": 90.0, "adjusted_end": 20.0, "presses": 1},
    ).status_code == 400
    assert conn.execute(
        "SELECT COUNT(*) FROM tool_metrics WHERE metric = ?", (queries.NUDGE_METRIC,)
    ).fetchone()[0] == before


def test_the_session_row_carries_the_denominator(client, conn):
    """§17 asks how OFTEN. The count of nudged candidates lives beside
    `reviewed` in the same row, so the two cannot be mismatched later."""
    client.post("/api/streams/fx/session",
                json={"duration_s": 300.0, "reviewed": 20, "nudged": 7})
    row = conn.execute(
        "SELECT meta FROM tool_metrics WHERE metric='review_session_duration_s' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    meta = json.loads(row["meta"])
    assert (meta["reviewed"], meta["nudged"]) == (20, 7)


# --------------------------------------------------------------------------
# instrumentation (§7.5, §14)
# --------------------------------------------------------------------------


def test_session_duration_is_recorded(client, conn):
    client.post("/api/streams/fx/session", json={"duration_s": 412.0, "reviewed": 6})
    row = conn.execute(
        "SELECT value, meta FROM tool_metrics WHERE metric='review_session_duration_s' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["value"] == 412.0
    assert json.loads(row["meta"])["reviewed"] == 6


def test_metrics_report_the_median_not_the_mean(conn):
    """Leave the tab open over lunch and one candidate reads forty minutes; a
    mean would be swamped and §7.1's target unmeasurable."""
    rows = conn.execute(
        "SELECT id FROM candidates WHERE stream_id='fx' AND is_current=1 LIMIT 4"
    ).fetchall()
    for row, ms in zip(rows, [2000, 2500, 3000, 2_400_000], strict=False):
        conn.execute(
            "INSERT OR REPLACE INTO ratings (candidate_id, rating, review_ms, rating_source) "
            "VALUES (?, 2, ?, 'operator')",
            (row["id"], ms),
        )

    metrics = queries.review_metrics(conn, "fx")
    assert metrics["median_review_ms"] < 5_000
    assert metrics["mean_review_ms"] > 500_000      # the outlier is still recorded


def test_an_inherited_rating_counts_the_way_the_screen_shows_it(tmp_path):
    """THE FIX. `load_candidates` LEFT JOINs ratings with no `rating_source`
    filter, so the review screen shows an inherited rating as rated. The summary
    beside it used to filter to 'operator', so after any re-score the screen
    said "rated" on every moment while the summary said `rated 0` and
    `approval rate 0.0%` — a summary disagreeing with the screen it summarises.

    Nothing can be double-counted at this scope: `is_current = 1` is one
    generation and a candidate has at most one `ratings` row. §14's
    count-one-judgment-once hazard is `clipforge/tuning.py`'s, because that
    spans generations.
    """
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    try:
        conn.execute(
            "INSERT INTO streams (id, date, master_path, marker_time_base) "
            "VALUES ('s', '2026-08-14', 'D:/m.mkv', 'vod')"
        )
        hand_rows(conn, "s", [
            (2, 1, 0.0, 10.0, 2, "2026-08-14 10:00:00", "inherited"),
            (2, 1, 100.0, 110.0, 0, "2026-08-14 10:01:00", "operator"),
        ])

        m = queries.review_metrics(conn, "s")
        assert m["rated"] == 2
        assert m["by_rating"] == {2: 1, 0: 1}
        assert m["approval_rate"] == 0.5
        # ...and the mix stays visible rather than being quietly merged.
        assert m["by_source"] == {"inherited": 1, "operator": 1}
    finally:
        conn.close()


def test_metrics_include_the_approval_rate(conn):
    """§14: 'approved / total candidates — is the threshold correct?'"""
    metrics = queries.review_metrics(conn, "fx")
    assert 0.0 <= metrics["approval_rate"] <= 1.0
    assert metrics["candidates"] > 0


def test_metrics_carry_the_target_so_the_browser_does_not_guess_it(conn):
    """The summary screen used to grade the session against a hardcoded 4000
    while `clipforge metrics` read `review.target_ms_per_candidate`. Changing
    the config moved one and not the other, silently."""
    assert queries.review_metrics(conn, "fx", target_ms=2500)["target_ms"] == 2500


def test_the_metrics_route_serves_the_configured_target(processed):
    cfg, _ = processed
    tweaked = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "review.target_ms_per_candidate=2500",
    ])
    with TestClient(review_app.create_app(tweaked), base_url="http://127.0.0.1",
                    headers={"X-ClipForge": "1"}) as tweaked_client:
        payload = tweaked_client.get("/api/streams/fx/metrics").json()

    assert payload["target_ms"] == 2500


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------


def test_proxy_is_served_with_ranges(client):
    response = client.get("/media/fx/proxy", headers={"Range": "bytes=0-1023"})
    assert response.status_code == 206
    assert len(response.content) == 1024
    assert response.headers["content-range"].startswith("bytes 0-1023/")


def test_proxy_seek_returns_different_bytes(client):
    """Proves the seek is real rather than always returning the file head."""
    head = client.get("/media/fx/proxy", headers={"Range": "bytes=0-255"}).content
    middle = client.get("/media/fx/proxy", headers={"Range": "bytes=500000-500255"}).content
    assert len(middle) == 256
    assert head != middle


def test_proxy_for_a_stream_without_one_is_404(client, conn):
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('noproxy', '2026-08-14', 'D:/m.mkv', 'vod')"
    )
    try:
        response = client.get("/media/noproxy/proxy")
        assert response.status_code == 404
        assert "clipforge run" in response.json()["detail"]
    finally:
        conn.execute("DELETE FROM streams WHERE id='noproxy'")


# --------------------------------------------------------------------------
# stream list
# --------------------------------------------------------------------------


def test_stream_list_reports_readiness(client):
    streams = client.get("/api/streams").json()["streams"]
    fx = next(s for s in streams if s["id"] == "fx")
    assert fx["has_proxy"] is True
    assert fx["candidates"] > 0
    assert fx["rated"] >= 0
    assert fx["stages_done"] > 0
