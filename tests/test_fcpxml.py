"""The FCPXML writer and `clipforge export` (§10.5).

The central assertion is made by RE-PARSING the written file and checking every
time attribute in the tree against the format's own declared frameDuration.
Trusting the writer to have used the writer correctly would prove nothing.

Both rates are exercised: the 30 fps fixture, and the 29.97 one — the rate where
a rounding bug does not divide out.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

from clipforge import cli, config, db
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.render import fcpxml, selection
from clipforge.render.timebase import DOWN, TimeBase, parse_value

#: Every attribute FCPXML expresses on the frame grid.
TIME_ATTRIBUTES = ("offset", "start", "duration", "tcStart", "frameDuration")


def build_stream(root, fixture_dir, manifest, ratings=(2, 2)):
    """Register, run, and rate — the state export expects to find."""
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)

    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=fixture_dir / manifest["video"])
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan())

    rows = conn.execute(
        "SELECT id FROM candidates WHERE stream_id = ? AND is_current = 1 ORDER BY t_start",
        (result.stream_id,),
    ).fetchall()
    with db.transaction(conn):
        for row, rating in zip(rows, ratings, strict=False):
            conn.execute(
                "INSERT INTO ratings (candidate_id, rating, rating_source) "
                "VALUES (?, ?, 'operator')",
                (row["id"], rating),
            )
    conn.close()
    return cfg, result.stream_id


@pytest.fixture(scope="module")
def exported(tmp_path_factory, fixture_dir, manifest):
    """30 fps — the integer case."""
    cfg, stream_id = build_stream(
        tmp_path_factory.mktemp("export30"), fixture_dir, manifest
    )
    out = cfg.data_root / "out30.fcpxml"
    code = cli.main(["export", stream_id, "--out", str(out),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])
    assert code == 0
    return cfg, stream_id, out, manifest


@pytest.fixture(scope="module")
def exported_ntsc(tmp_path_factory, ntsc_fixture_dir, ntsc_manifest):
    """29.97 — the rate that actually tests the rational path."""
    cfg, stream_id = build_stream(
        tmp_path_factory.mktemp("export2997"), ntsc_fixture_dir, ntsc_manifest
    )
    out = cfg.data_root / "out2997.fcpxml"
    code = cli.main(["export", stream_id, "--out", str(out),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])
    assert code == 0
    return cfg, stream_id, out, ntsc_manifest


def parsed(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def frame_duration_of(root: ET.Element) -> Fraction:
    return parse_value(root.find(".//format").get("frameDuration"))


def every_time_attribute(root: ET.Element):
    for element in root.iter():
        for name in TIME_ATTRIBUTES:
            value = element.get(name)
            if value is not None:
                yield element.tag, name, value


# --------------------------------------------------------------------------
# the requirement: exact integer multiples of a rational frameDuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["exported", "exported_ntsc"])
def test_every_time_is_an_exact_multiple_of_the_frame_duration(which, request):
    """HANDOFF: get this wrong and Resolve rejects the file or silently rounds.

    Read back off disk, not from the writer's own values."""
    _cfg, _stream_id, out, _manifest = request.getfixturevalue(which)
    root = parsed(out)
    frame = frame_duration_of(root)

    checked = 0
    for tag, name, value in every_time_attribute(root):
        seconds = parse_value(value)
        assert seconds % frame == 0, f"<{tag} {name}={value}> is not on the frame grid"
        checked += 1
    assert checked > 3


def test_the_ntsc_frame_duration_is_the_true_rational(exported_ntsc):
    """1001/30000, not 1/29.97 — and taken from the fixture's declared rate."""
    _cfg, _stream_id, out, manifest = exported_ntsc
    rate = Fraction(manifest["fps"])
    assert frame_duration_of(parsed(out)) == 1 / rate
    assert parsed(out).find(".//format").get("frameDuration") == f"{rate.denominator}/{rate.numerator}s"


def test_the_ntsc_denominator_is_the_timescale(exported_ntsc):
    """Unreduced, as Final Cut writes them. A reduced 1001/1000s is the same
    instant and exactly what a naive parser mishandles."""
    _cfg, _stream_id, out, manifest = exported_ntsc
    rate = Fraction(manifest["fps"])
    for _tag, _name, value in every_time_attribute(parsed(out)):
        if value == "0s":
            continue
        assert value.endswith("s") and "/" in value
        assert int(value[:-1].split("/")[1]) == rate.numerator


def test_the_declared_rate_matches_what_was_probed(exported_ntsc):
    cfg, stream_id, out, manifest = exported_ntsc
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    row = conn.execute("SELECT fps_num, fps_den FROM streams WHERE id = ?",
                       (stream_id,)).fetchone()
    conn.close()
    assert Fraction(row["fps_num"], row["fps_den"]) == Fraction(manifest["fps"])
    assert frame_duration_of(parsed(out)) == Fraction(row["fps_den"], row["fps_num"])


# --------------------------------------------------------------------------
# the timeline holds together
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["exported", "exported_ntsc"])
def test_clips_abut_with_no_overlap_or_gap(which, request):
    """A tight assembly: each clip starts exactly where the last one ended."""
    _cfg, _stream_id, out, _manifest = request.getfixturevalue(which)
    root = parsed(out)
    cursor = Fraction(0)
    clips = root.findall(".//spine/asset-clip")
    assert clips
    for clip in clips:
        assert parse_value(clip.get("offset")) == cursor
        cursor += parse_value(clip.get("duration"))
    assert parse_value(root.find(".//sequence").get("duration")) == cursor


@pytest.mark.parametrize("which", ["exported", "exported_ntsc"])
def test_no_clip_runs_past_the_end_of_the_source(which, request):
    """Referencing footage that does not exist reports in Resolve as a media
    error rather than as the rounding problem it actually is."""
    _cfg, _stream_id, out, _manifest = request.getfixturevalue(which)
    root = parsed(out)
    asset_end = parse_value(root.find(".//asset").get("duration"))
    for clip in root.findall(".//spine/asset-clip"):
        end = parse_value(clip.get("start")) + parse_value(clip.get("duration"))
        assert end <= asset_end


def test_the_asset_never_claims_more_than_the_file_holds(exported, manifest):
    _cfg, _stream_id, out, _m = exported
    duration = parse_value(parsed(out).find(".//asset").get("duration"))
    assert float(duration) <= manifest["duration_s"]


@pytest.mark.parametrize("which", ["exported", "exported_ntsc"])
def test_clip_windows_cover_the_approved_moments(which, request):
    """C2: boundaries expand onto the grid, so a clip is never shorter than the
    moment it came from.

    WITH ONE EXCEPTION, AND IT IS ARITHMETIC RATHER THAN A CHOICE: at the very
    end of a recording there is no next frame to expand onto. The NTSC fixture
    is 30.000 s at 30000/1001 fps, which is 899.1 frames — so the last WHOLE
    frame ends at 29.99663 s and a moment running to 30.000 s cannot be covered
    by footage that exists. `plan_clips` clamps to `source_frames` for exactly
    that reason: a clip referencing a frame past the end is a media error in
    Resolve rather than a rounding problem.

    Found by Phase 3's cross-profile merge, which unions windows and so reached
    the end of the fixture for the first time. The invariant is therefore stated
    against the footage rather than against the clock.
    """
    cfg, stream_id, out, _manifest = request.getfixturevalue(which)
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    moments = selection.approved_moments(conn, stream_id, min_rating=2)
    row = conn.execute("SELECT * FROM streams WHERE id = ?", (stream_id,)).fetchone()
    conn.close()

    # The last instant the source can express, derived exactly the way
    # `cmd_export` derives it rather than written down here.
    timebase = TimeBase.from_stream(row)
    source_frames = timebase.frame_at(float(row["duration_s"]), DOWN)
    last_frame_end = float(source_frames * timebase.frame_duration)

    clips = parsed(out).findall(".//spine/asset-clip")
    assert len(clips) == len(moments)
    for clip, moment in zip(clips, moments, strict=True):
        start = float(parse_value(clip.get("start")))
        end = start + float(parse_value(clip.get("duration")))
        assert start <= moment.t_start
        assert end >= min(moment.t_end, last_frame_end)


# --------------------------------------------------------------------------
# what Resolve needs to find the media
# --------------------------------------------------------------------------


def test_the_media_path_round_trips(exported, fixture_dir, manifest):
    _cfg, _stream_id, out, _m = exported
    src = parsed(out).find(".//media-rep").get("src")
    assert src.startswith("file:///")
    assert Path(fixture_dir / manifest["video"]).as_uri() == src


def test_a_path_with_spaces_is_escaped(tmp_path):
    """A hand-built file:// string gets this wrong; as_uri() does not."""
    spaced = tmp_path / "Marvel Rivals ranked.mkv"
    spaced.write_bytes(b"x")
    tree = fcpxml.build(
        source=spaced, timebase=TimeBase(30, 1), source_frames=300,
        clips=[fcpxml.Clip(0, 30, 0, "one")], resolution=(1920, 1080),
        project_name="p",
    )
    src = tree.getroot().find(".//media-rep").get("src")
    assert "%20" in src and " " not in src


def test_the_asset_uid_is_stable_across_exports(tmp_path):
    """Re-importing should relink, not duplicate."""
    source = tmp_path / "a.mkv"
    source.write_bytes(b"x")
    build = lambda: fcpxml.build(  # noqa: E731
        source=source, timebase=TimeBase(30, 1), source_frames=300,
        clips=[fcpxml.Clip(0, 30, 0, "one")], resolution=(1920, 1080), project_name="p",
    ).getroot().find(".//asset").get("uid")
    assert build() == build()


def test_the_document_declares_the_configured_version(exported):
    _cfg, _stream_id, out, _m = exported
    version = config.load().get("export.fcpxml_version")
    assert parsed(out).get("version") == str(version)


def test_the_file_carries_the_doctype(exported):
    _cfg, _stream_id, out, _m = exported
    head = out.read_bytes()[:80].decode()
    assert head.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<!DOCTYPE fcpxml>" in head


def test_the_structure_is_the_one_resolve_expects(exported):
    _cfg, _stream_id, out, _m = exported
    root = parsed(out)
    assert root.tag == "fcpxml"
    for path in (".//resources/format", ".//resources/asset/media-rep",
                 ".//library/event/project/sequence/spine"):
        assert root.find(path) is not None, f"missing {path}"


def test_resolution_comes_from_the_probe(exported, manifest):
    _cfg, _stream_id, out, _m = exported
    fmt = parsed(out).find(".//format")
    assert f"{fmt.get('width')}x{fmt.get('height')}" == manifest["resolution"]


def test_audio_sources_match_the_probed_track_map(exported, manifest):
    cfg, stream_id, out, _m = exported
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    import json
    roles = json.loads(
        conn.execute("SELECT audio_track_map FROM streams WHERE id = ?",
                     (stream_id,)).fetchone()["audio_track_map"]
    )["roles"]
    conn.close()
    assert int(parsed(out).find(".//asset").get("audioSources")) == len(roles)


# --------------------------------------------------------------------------
# what the command records
# --------------------------------------------------------------------------


def test_export_rows_describe_the_file(exported):
    cfg, stream_id, out, _m = exported
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    row = conn.execute(
        "SELECT * FROM exports WHERE stream_id = ? ORDER BY id DESC LIMIT 1", (stream_id,)
    ).fetchone()
    items = conn.execute(
        "SELECT * FROM export_items WHERE export_id = ? ORDER BY seq", (row["id"],)
    ).fetchall()
    conn.close()

    assert row["kind"] == "fcpxml"
    assert row["item_count"] == len(items)
    assert len(items) == len(parsed(out).findall(".//spine/asset-clip"))
    assert all(item["role"] == "clip" for item in items)


def test_export_items_hold_the_frame_snapped_times(exported):
    """The schema calls that column "frame-snapped, as written to the file", so
    it must describe the file rather than the intention behind it."""
    cfg, stream_id, out, manifest = exported
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    items = conn.execute(
        "SELECT ei.* FROM export_items ei JOIN exports e ON e.id = ei.export_id "
        "WHERE e.stream_id = ? ORDER BY ei.seq", (stream_id,)
    ).fetchall()
    conn.close()

    rate = Fraction(manifest["fps"])
    timebase = TimeBase(rate.numerator, rate.denominator)
    clips = parsed(out).findall(".//spine/asset-clip")
    for item, clip in zip(items, clips, strict=True):
        assert item["t_start"] == pytest.approx(float(parse_value(clip.get("start"))))
        assert timebase.frame_at(item["t_start"]) * timebase.frame_duration \
            == parse_value(clip.get("start"))


def test_contributing_candidates_are_marked_frame_snapped(exported):
    cfg, stream_id, _out, _m = exported
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    snapped = conn.execute(
        "SELECT COUNT(*) FROM candidates c JOIN ratings r ON r.candidate_id = c.id "
        "WHERE c.stream_id = ? AND r.rating >= 2 AND c.frame_snapped = 1", (stream_id,)
    ).fetchone()[0]
    conn.close()
    assert snapped > 0


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_vfr_master_is_refused(exported, capsys):
    """A variable-rate source has no frameDuration, so a constant grid over it
    imports cleanly and is quietly misaligned — the worst available outcome."""
    cfg, stream_id, _out, _m = exported
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    conn.execute("UPDATE streams SET is_vfr = 1 WHERE id = ?", (stream_id,))
    conn.close()
    try:
        code = cli.main(["export", stream_id,
                         "--set", f"paths.data_root={cfg.data_root.as_posix()}"])
        output = capsys.readouterr().out
        assert code == 1
        assert "variable frame rate" in output
        assert "export.source=proxy" in output
    finally:
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        conn.execute("UPDATE streams SET is_vfr = 0 WHERE id = ?", (stream_id,))
        conn.close()


def test_allow_vfr_overrides_the_refusal(exported, tmp_path):
    cfg, stream_id, _out, _m = exported
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    conn.execute("UPDATE streams SET is_vfr = 1 WHERE id = ?", (stream_id,))
    conn.close()
    try:
        assert cli.main(["export", stream_id, "--allow-vfr",
                         "--out", str(tmp_path / "forced.fcpxml"),
                         "--set", f"paths.data_root={cfg.data_root.as_posix()}"]) == 0
    finally:
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        conn.execute("UPDATE streams SET is_vfr = 0 WHERE id = ?", (stream_id,))
        conn.close()


def test_a_stream_with_nothing_approved_is_refused(exported, capsys, tmp_path):
    """Writing an empty timeline would look like success."""
    cfg, stream_id, _out, _m = exported
    code = cli.main(["export", stream_id, "--min-rating", "2",
                     "--out", str(tmp_path / "none.fcpxml"),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}",
                     "--profile", "naive"])
    assert code == 0            # this stream does have approvals

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    conn.execute("UPDATE ratings SET rating = 0")
    conn.close()
    try:
        assert cli.main(["export", stream_id,
                         "--set", f"paths.data_root={cfg.data_root.as_posix()}"]) == 1
        assert "nothing rated" in capsys.readouterr().out
    finally:
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        conn.execute("UPDATE ratings SET rating = 2")
        conn.close()


def test_an_unknown_stream_is_refused(capsys, tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    assert cli.main(["export", "nope",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"]) == 1
    assert "no stream" in capsys.readouterr().out


# --------------------------------------------------------------------------
# clip planning, without the XML
# --------------------------------------------------------------------------


def test_a_moment_beyond_the_source_is_clamped():
    timebase = TimeBase(30000, 1001)
    moments = [selection.Moment(t_start=0.0, t_end=1000.0, rating=2, rated_at="")]
    clips = fcpxml.plan_clips(moments, timebase, source_frames=300, label=lambda i, m: "x")
    assert clips[0].source_end == 300


def test_a_moment_entirely_past_the_source_is_dropped():
    timebase = TimeBase(30, 1)
    moments = [selection.Moment(t_start=500.0, t_end=520.0, rating=2, rated_at="")]
    assert fcpxml.plan_clips(moments, timebase, 300, lambda i, m: "x") == []
