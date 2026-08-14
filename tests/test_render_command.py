"""`clipforge render` — §8's auto-finish, end to end.

The command half is pure and tested without ffmpeg: A1's seek placement, which
audio stream is chosen, what the window is after handles, and every refusal in
`assert_sound`. The encode itself is ffmpeg-gated and asserts against the
fixture manifest and the config's own template, never a literal.
"""

from __future__ import annotations

import json

import pytest

from clipforge import cli, config, db, ffmpeg
from clipforge.render import clips, crop, timeline
from clipforge.render.clips import ClipCheck, Options
from clipforge.render.selection import Moment

# Preset and CRF are overridden in the encoding tests: they decide how long
# libx264 thinks about each frame, and nothing asserted here is about quality.
FAST = ["--set", "render.video.preset=ultrafast", "--set", "render.video.crf=30"]

TRACK_MAP = json.dumps({
    "roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3},
    "warnings": [], "source": "metadata tags", "titles": {},
})


class FakeStream(dict):
    """A `streams` row, as sqlite3.Row-alike, for the pure tests."""

    def __getitem__(self, key):
        return self.get(key)


def moment(t_start=10.0, t_end=20.0, rating=2):
    return Moment(t_start=t_start, t_end=t_end, rating=rating,
                  rated_at="2026-08-13", candidate_ids=[1], generation=1)


def plan(start=10.0, end=20.0):
    return timeline.build(start, end)


def template():
    return crop.get_template(config.load())


# --------------------------------------------------------------------------
# A1, and the shape of the command
# --------------------------------------------------------------------------


def test_seek_goes_before_the_input():
    """A1. On a 50 GB master this is milliseconds against minutes, and
    `ffmpeg.run` refuses the command if it is the wrong way round."""
    argv = clips.build_command(
        "ffmpeg", "master.mkv", "out.mp4", cfg=config.load(), plan=plan(),
        graph="[0:v]null[vout]", audio_index=0,
    )

    assert argv.index("-ss") < argv.index("-i")
    ffmpeg.validate_seek_placement(argv)      # raises if A1 is violated


def test_the_duration_comes_from_the_plan_not_the_window():
    """`-t` is what ffmpeg reads. With a cut in the plan the two differ, which
    is the whole reason `source_duration` is separate from `duration`."""
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=config.load(),
        plan=timeline.build(10.0, 25.0, [timeline.Cut(12.0, 13.0)]),
        graph="[0:v]null[vout]", audio_index=0,
    )

    assert argv[argv.index("-t") + 1] == f"{15.0:.6f}"


def test_the_video_comes_from_the_graphs_label():
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=config.load(), plan=plan(),
        graph="[0:v]null[vout]", audio_index=2,
    )

    assert argv[argv.index("-map") + 1] == f"[{clips.VIDEO_LABEL}]"
    assert "0:a:2" in argv


def test_faststart_is_set_so_the_first_play_does_not_stall():
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=config.load(), plan=plan(),
        graph="[0:v]null[vout]", audio_index=0,
    )

    assert argv[argv.index("-movflags") + 1] == "+faststart"


def test_no_audio_track_map_renders_silent_rather_than_failing():
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=config.load(), plan=plan(),
        graph="[0:v]null[vout]", audio_index=None,
    )

    assert "-an" in argv
    assert not any(a.startswith("0:a:") for a in argv)


def test_a_still_is_one_silent_frame():
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.png", cfg=config.load(), plan=plan(),
        graph="[0:v]null[vout]", audio_index=0, still=True,
    )

    assert argv[argv.index("-frames:v") + 1] == "1"
    assert "-an" in argv
    assert "-c:a" not in argv


# --------------------------------------------------------------------------
# which audio stream (finding 5)
# --------------------------------------------------------------------------


def test_auto_prefers_the_probed_mix():
    """§8.3's bare `-af` takes stream 0, which is the mix only by luck of the
    OBS layout. This reuses the rule §5.2 already needed."""
    stream = FakeStream(audio_track_map=json.dumps(
        {"roles": {"mic": 0, "game": 1, "party": 2, "mixed": 3}}))

    assert clips.audio_stream_index(config.load(), stream) == 3


def test_auto_falls_back_to_the_lowest_known_role():
    stream = FakeStream(audio_track_map=json.dumps({"roles": {"mic": 1, "party": 2}}))

    assert clips.audio_stream_index(config.load(), stream) == 1


def test_an_explicit_index_overrides_the_probe():
    cfg = config.load(overrides=["render.audio.source=2"])
    stream = FakeStream(audio_track_map=TRACK_MAP)

    assert clips.audio_stream_index(cfg, stream) == 2


def test_no_roles_means_no_audio():
    assert clips.audio_stream_index(config.load(), FakeStream()) is None
    assert clips.audio_stream_index(
        config.load(), FakeStream(audio_track_map=json.dumps({"roles": {}}))) is None


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


def test_handles_expand_the_approved_window():
    cfg = config.load(overrides=["render.handles_s=0.5"])

    window = clips.clip_window(cfg, moment(10.0, 20.0), 60.0)

    assert window.clip_start == pytest.approx(9.5)
    assert window.clip_end == pytest.approx(20.5)


def test_handles_clamp_at_the_ends_of_the_source():
    cfg = config.load(overrides=["render.handles_s=5.0"])

    window = clips.clip_window(cfg, moment(1.0, 59.0), 60.0)

    assert window.clip_start == 0.0
    assert window.clip_end == 60.0


def test_a_moment_outside_the_source_is_refused():
    with pytest.raises(clips.RenderError, match="outside the source"):
        clips.clip_window(config.load(), moment(70.0, 80.0), 60.0)


# --------------------------------------------------------------------------
# source selection
# --------------------------------------------------------------------------


def test_render_source_is_separate_from_export_source(tmp_path):
    """An FCPXML may be pointed at the proxy while the deliverable still wants
    the master — they are different decisions."""
    cfg = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'd').as_posix()}",
        "export.source=proxy", "render.source=master",
    ])
    from clipforge import paths

    stream = FakeStream(id="s", master_path=str(tmp_path / "m.mkv"), proxy_path="p.mp4")
    stream_paths = paths.StreamPaths(cfg.data_root, "s")

    assert clips.source_for(cfg, stream, stream_paths) == tmp_path / "m.mkv"


def test_rendering_from_a_missing_proxy_says_what_to_run(tmp_path):
    cfg = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'd').as_posix()}", "render.source=proxy"])
    from clipforge import paths

    with pytest.raises(clips.RenderError, match="clipforge run"):
        clips.source_for(cfg, FakeStream(id="s", master_path="m.mkv", proxy_path=None),
                         paths.StreamPaths(cfg.data_root, "s"))


def test_an_unknown_render_source_is_refused(tmp_path):
    cfg = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'd').as_posix()}", "render.source=elsewhere"])
    from clipforge import paths

    with pytest.raises(clips.RenderError, match="master' or 'proxy"):
        clips.source_for(cfg, FakeStream(id="s", master_path="m.mkv"),
                         paths.StreamPaths(cfg.data_root, "s"))


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def sound(**overrides):
    spec = template()
    base = ClipCheck(width=spec.output[0], height=spec.output[1],
                     duration_s=10.0, pix_fmt="yuv420p", has_audio=True)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_a_sound_clip_passes():
    clips.assert_sound(sound(), template=template(), plan=plan(),
                       want_audio=True, cfg=config.load())


def test_the_wrong_output_size_is_caught():
    with pytest.raises(clips.RenderError, match="expected"):
        clips.assert_sound(sound(width=1920, height=1080), template=template(),
                           plan=plan(), want_audio=True, cfg=config.load())


def test_a_short_clip_is_caught():
    with pytest.raises(clips.RenderError, match="off by"):
        clips.assert_sound(sound(duration_s=4.0), template=template(),
                           plan=plan(), want_audio=True, cfg=config.load())


def test_the_duration_tolerance_comes_from_config():
    cfg = config.load(overrides=["render.verify.max_duration_delta_s=10"])

    clips.assert_sound(sound(duration_s=4.0), template=template(), plan=plan(),
                       want_audio=True, cfg=cfg)


def test_a_pixel_format_no_platform_can_decode_is_caught():
    with pytest.raises(clips.RenderError, match="yuv420p"):
        clips.assert_sound(sound(pix_fmt="yuv422p10le"), template=template(),
                           plan=plan(), want_audio=True, cfg=config.load())


def test_a_silent_short_is_caught_here_not_on_the_upload_page():
    with pytest.raises(clips.RenderError, match="no audio"):
        clips.assert_sound(sound(has_audio=False), template=template(),
                           plan=plan(), want_audio=True, cfg=config.load())


def test_a_deliberately_silent_render_is_allowed():
    clips.assert_sound(sound(has_audio=False), template=template(), plan=plan(),
                       want_audio=False, cfg=config.load())


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rated(tmp_path_factory, fixture_dir, manifest):
    """A registered, processed, rated stream — what render expects to find."""
    from tests.test_fcpxml import build_stream

    return build_stream(tmp_path_factory.mktemp("render"), fixture_dir, manifest)


@pytest.fixture(scope="module")
def rendered(rated, needs_ffmpeg):
    cfg, stream_id = rated
    code = cli.main(["render", stream_id, "--limit", "1",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}", *FAST])
    assert code == 0
    return cfg, stream_id


def outputs(cfg, stream_id, suffix=".mp4"):
    from clipforge import paths

    exports = paths.StreamPaths(cfg.data_root, stream_id).exports_dir
    return sorted(exports.glob(f"*{suffix}"))


def test_a_clip_is_written(rendered):
    cfg, stream_id = rendered

    files = outputs(cfg, stream_id)

    assert len(files) == 1
    assert files[0].stat().st_size > 0


def test_the_clip_is_the_templates_output_size(rendered):
    """Read from the template, not written down here — the whole point of
    §8.4 is that the operator changes those numbers."""
    cfg, stream_id = rendered
    spec = crop.get_template(cfg)

    found = clips.inspect(outputs(cfg, stream_id)[0], ffmpeg.require("ffprobe"))

    assert (found.width, found.height) == spec.output


def test_the_clip_matches_the_approved_window(rendered):
    """Against the database's own candidate row plus the configured handles,
    not a number in this file."""
    cfg, stream_id = rendered
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        row = conn.execute(
            "SELECT t_start, t_end FROM export_items ei "
            "JOIN exports e ON e.id = ei.export_id WHERE e.stream_id = ? "
            "ORDER BY e.id LIMIT 1", (stream_id,)).fetchone()
    finally:
        conn.close()

    found = clips.inspect(outputs(cfg, stream_id)[0], ffmpeg.require("ffprobe"))

    expected = row["t_end"] - row["t_start"]
    tolerance = float(cfg.get("render.verify.max_duration_delta_s"))
    assert found.duration_s == pytest.approx(expected, abs=tolerance)


def test_the_clip_carries_audio_and_a_decodable_pixel_format(rendered):
    cfg, stream_id = rendered

    found = clips.inspect(outputs(cfg, stream_id)[0], ffmpeg.require("ffprobe"))

    assert found.has_audio
    assert found.pix_fmt == cfg.get("render.video.pix_fmt")


def test_the_export_is_recorded(rendered):
    cfg, stream_id = rendered
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        row = conn.execute(
            "SELECT kind, path, candidate_id, item_count FROM exports "
            "WHERE stream_id = ? AND kind = 'clip'", (stream_id,)).fetchone()
        items = conn.execute(
            "SELECT COUNT(*) FROM export_items ei JOIN exports e "
            "ON e.id = ei.export_id WHERE e.stream_id = ? AND e.kind = 'clip'",
            (stream_id,)).fetchone()[0]
    finally:
        conn.close()

    assert row["kind"] == "clip"
    assert row["candidate_id"] is not None
    assert row["path"].startswith("streams/")
    assert items == 1


def test_the_render_params_hash_is_logged(rendered):
    """`render:` sits outside VERSIONED_SUBTREES on purpose, so nothing else
    would ever detect that a file on disk is stale against its settings."""
    cfg, stream_id = rendered
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        row = conn.execute(
            "SELECT value, meta FROM tool_metrics WHERE stream_id = ? "
            "AND metric = 'render_clips'", (stream_id,)).fetchone()
    finally:
        conn.close()

    meta = json.loads(row["meta"])
    assert row["value"] == 1.0
    assert len(meta["params_hash"]) == 16
    assert meta["template"] == cfg.get("render.crop.template")


def test_changing_a_caption_colour_changes_the_params_hash(rated, needs_ffmpeg):
    """The point of logging it: two files that look alike were made
    differently."""
    cfg, stream_id = rated
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        hashes = set()
        for colour in ("#FFFFFF", "#FF0000"):
            tweaked = config.load(overrides=[
                f"paths.data_root={cfg.data_root.as_posix()}",
                f"render.captions.styles.mic.colour={colour}"])
            result = clips.render_stream(
                tweaked, conn, stream_id,
                Options(limit=1, dry_run=True), log=lambda *_: None)
            payload = {
                "template": result.template.name,
                "output": list(result.template.output),
                "crop": tweaked.get("render.crop"),
                "captions": tweaked.get("render.captions"),
                "video": tweaked.get("render.video"),
                "handles_s": tweaked.get("render.handles_s"),
            }
            import hashlib

            from clipforge.config import canonical_json
            hashes.add(hashlib.sha256(
                canonical_json(payload).encode()).hexdigest()[:16])
    finally:
        conn.close()

    assert len(hashes) == 2


def test_stills_are_written_and_are_the_output_size(rated, needs_ffmpeg, tmp_path):
    """The fast loop for placeholder crop coordinates."""
    cfg, stream_id = rated
    out = tmp_path / "stills"

    code = cli.main(["render", stream_id, "--stills", "--limit", "1",
                     "--out", str(out),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 0
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 1
    found = clips.inspect(pngs[0], ffmpeg.require("ffprobe"))
    assert (found.width, found.height) == crop.get_template(cfg).output


def test_dry_run_encodes_nothing(rated, tmp_path):
    cfg, stream_id = rated
    out = tmp_path / "nothing"

    code = cli.main(["render", stream_id, "--dry-run", "--out", str(out),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 0
    assert not out.exists() or not list(out.iterdir())


def test_an_unrated_stream_says_to_review_it_first(rated, capsys):
    cfg, stream_id = rated

    code = cli.main(["render", stream_id, "--min-rating", "2",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}",
                     "--set", "render.crop.template=no_such"])

    assert code == 1
    assert "no crop template" in capsys.readouterr().out


def test_an_unknown_stream_is_refused(rated, capsys):
    cfg, _ = rated

    code = cli.main(["render", "not-a-stream",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 1
    assert "no stream" in capsys.readouterr().out
