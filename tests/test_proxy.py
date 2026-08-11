"""proxy: command construction, verification, and the real encode.

Every number asserted here comes from the fixture manifest or from probing the
master, never from a constant typed into the test.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from clipforge import config, db
from clipforge.ingest import probe, proxy
from clipforge.pipeline import runner


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "configured", "expected"),
    [
        (1080, 720, 720),   # the normal case: downscale
        (720, 720, 720),    # already there: no-op
        (360, 720, 360),    # §5.2's bare -2:720 would upscale this
        (1440, 720, 720),
        (721, 720, 720),
        (361, 720, 360),    # odd source height must not produce an odd target
    ],
)
def test_target_height_never_upscales_and_stays_even(source, configured, expected):
    """libx264 with yuv420p rejects odd dimensions outright, and upscaling
    spends encode time adding no information."""
    assert proxy.target_height(source, configured) == expected


def test_scale_filter_is_omitted_when_no_scaling_is_needed():
    cfg = config.load()
    argv = proxy.build_command(
        "ffmpeg", __file__, "out.mp4", cfg=cfg, source_height=720,
        rate=Fraction(30, 1), audio_index=0,
    )
    assert "-vf" not in argv


def test_scale_filter_is_present_when_downscaling():
    cfg = config.load()
    argv = proxy.build_command(
        "ffmpeg", __file__, "out.mp4", cfg=cfg, source_height=1080,
        rate=Fraction(30, 1), audio_index=0,
    )
    assert argv[argv.index("-vf") + 1] == "scale=-2:720"


# --------------------------------------------------------------------------
# audio track selection
# --------------------------------------------------------------------------


def test_mixed_role_is_preferred():
    """§5.2 hardcodes 0:a:0, which is only right when stream 0 is the mix."""
    track_map = {"roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3}}
    assert proxy.audio_map_index(track_map) == 0


def test_mixed_is_found_even_when_it_is_not_first():
    """An OBS layout that orders the mic first would otherwise give every
    review session mic-only audio."""
    assert proxy.audio_map_index({"roles": {"mic": 0, "mixed": 2}}) == 2


def test_falls_back_to_the_lowest_indexed_role():
    assert proxy.audio_map_index({"roles": {"mic": 1, "game": 2}}) == 1


def test_empty_track_map_falls_back_to_stream_zero():
    assert proxy.audio_map_index({}) == 0
    assert proxy.audio_map_index({"roles": {}}) == 0


# --------------------------------------------------------------------------
# the §5.2 command
# --------------------------------------------------------------------------


@pytest.fixture
def command():
    cfg = config.load()
    return cfg, proxy.build_command(
        "ffmpeg", "master.mkv", "proxy.mp4", cfg=cfg, source_height=1080,
        rate=Fraction(30000, 1001), audio_index=1,
    )


def test_command_carries_the_a2_gop_flags(command):
    """A2: without these, x264 inserts keyframes on scene changes and seek
    granularity becomes unpredictable."""
    cfg, argv = command
    assert argv[argv.index("-g") + 1] == str(cfg.get("ingest.proxy.gop"))
    assert argv[argv.index("-keyint_min") + 1] == str(cfg.get("ingest.proxy.keyint_min"))
    assert argv[argv.index("-sc_threshold") + 1] == str(cfg.get("ingest.proxy.sc_threshold"))


def test_command_forces_cfr_with_an_exact_rational(command):
    """A float frame rate cannot express 29.97 (30000/1001)."""
    cfg, argv = command
    assert argv[argv.index("-fps_mode") + 1] == "cfr"
    assert argv[argv.index("-r") + 1] == "30000/1001"


def test_command_sets_pix_fmt(command):
    """§5.2 omits this; a 10-bit master would produce a proxy no browser can
    decode, and the review UI would show a black player with no error."""
    cfg, argv = command
    assert argv[argv.index("-pix_fmt") + 1] == cfg.get("ingest.proxy.pix_fmt")


def test_command_maps_the_selected_audio_stream(command):
    cfg, argv = command
    assert "0:a:1" in argv


def test_command_uses_faststart(command):
    cfg, argv = command
    assert argv[argv.index("-movflags") + 1] == "+faststart"


def test_command_respects_extra_output_args():
    """A last-resort escape hatch, independent of the per-encoder args block."""
    from clipforge.ingest import encoders

    cfg = config.load(overrides=['ingest.proxy.extra_output_args=["-maxrate", "3M"]'])
    argv = proxy.build_command(
        "ffmpeg", "m.mkv", "p.mp4", cfg=cfg, source_height=1080,
        rate=Fraction(60, 1), audio_index=0,
        encoder=encoders.EncoderChoice("h264_nvenc", "p4", ["-rc", "vbr"], "configured"),
    )
    assert argv[argv.index("-c:v") + 1] == "h264_nvenc"
    assert "-rc" in argv and "vbr" in argv
    assert argv[argv.index("-maxrate") + 1] == "3M"


def test_command_does_not_violate_a1():
    """No -ss at all here, but the guard must accept the command."""
    cfg = config.load()
    argv = proxy.build_command(
        "ffmpeg", "m.mkv", "p.mp4", cfg=cfg, source_height=720,
        rate=Fraction(30, 1), audio_index=0,
    )
    from clipforge import ffmpeg
    ffmpeg.validate_seek_placement(argv)


# --------------------------------------------------------------------------
# verification rules
# --------------------------------------------------------------------------


def make_check(duration, frames, expected, interval=30.0):
    return proxy.ProxyCheck(
        duration_s=duration, frames=frames, expected_frames=expected,
        keyframe_intervals=[], median_keyframe_interval=interval,
    )


def test_matching_proxy_passes():
    cfg = config.load()
    proxy.assert_sound(
        make_check(60.0, 1800, 1800), master_duration=60.0, gop=30,
        is_vfr=False, cfg=cfg, log=lambda *_: None,
    )


def test_duration_mismatch_is_always_fatal():
    """Every timestamp in the system assumes proxy and master are
    interchangeable (§5.2, §13.1)."""
    cfg = config.load()
    with pytest.raises(proxy.ProxyError, match="duration"):
        proxy.assert_sound(
            make_check(55.0, 1800, 1800), master_duration=60.0, gop=30,
            is_vfr=False, cfg=cfg, log=lambda *_: None,
        )


def test_frame_mismatch_is_fatal_for_a_cfr_master():
    cfg = config.load()
    with pytest.raises(proxy.ProxyError, match="frames"):
        proxy.assert_sound(
            make_check(60.0, 1700, 1800), master_duration=60.0, gop=30,
            is_vfr=False, cfg=cfg, log=lambda *_: None,
        )


def test_frame_mismatch_is_advisory_for_a_vfr_master():
    """Forcing CFR on variable-rate input legitimately changes the count;
    failing would wedge every such stream forever."""
    cfg = config.load()
    messages = []
    proxy.assert_sound(
        make_check(60.0, 1700, 1800), master_duration=60.0, gop=30,
        is_vfr=True, cfg=cfg, log=messages.append,
    )
    assert any("VFR" in m for m in messages)


def test_irregular_keyframes_are_fatal():
    """A2's fixed GOP is the entire basis for Phase 1 live-seeking the proxy
    instead of pre-rendering previews (§7.2)."""
    cfg = config.load()
    with pytest.raises(proxy.ProxyError, match="keyframes"):
        proxy.assert_sound(
            make_check(60.0, 1800, 1800, interval=250.0), master_duration=60.0,
            gop=30, is_vfr=False, cfg=cfg, log=lambda *_: None,
        )


def test_keyframe_check_tolerates_a_little_slack():
    cfg = config.load()
    gop = int(cfg.get("ingest.proxy.gop"))
    ratio = float(cfg.get("ingest.proxy.verify.max_keyframe_interval_ratio"))
    proxy.assert_sound(
        make_check(60.0, 1800, 1800, interval=gop * ratio), master_duration=60.0,
        gop=gop, is_vfr=False, cfg=cfg, log=lambda *_: None,
    )


def test_missing_frame_count_is_not_an_error():
    """A container that does not report nb_frames must not fail the stage."""
    cfg = config.load()
    proxy.assert_sound(
        make_check(60.0, None, 1800), master_duration=60.0, gop=30,
        is_vfr=False, cfg=cfg, log=lambda *_: None,
    )


# --------------------------------------------------------------------------
# the real encode, against the fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def encoded(tmp_path_factory, fixture_dir, manifest):
    """Register, probe and proxy the fixture once for the whole session."""
    root = tmp_path_factory.mktemp("proxy")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('fx', '2026-08-14', ?, 'epoch')",
        (str(fixture_dir / manifest["video"]),),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan(["register_stream", "probe", "proxy"]))
    yield cfg, conn, engine, results
    conn.close()


def test_encode_succeeds(encoded):
    cfg, conn, engine, results = encoded
    assert [r.status for r in results] == ["done", "done"], [r.error for r in results]


def test_proxy_duration_matches_the_master(encoded, manifest):
    cfg, conn, engine, _ = encoded
    info = probe.probe_json(cfg.data_root / "streams/fx/proxy/proxy.mp4", "-show_format")
    duration = float(info["format"]["duration"])
    tolerance = float(cfg.get("ingest.proxy.verify.max_duration_delta_s"))
    assert duration == pytest.approx(manifest["duration_s"], abs=tolerance)


def test_proxy_frame_count_matches_duration_times_rate(encoded, manifest):
    """Derived from the manifest, not typed in."""
    cfg, conn, engine, _ = encoded
    from clipforge import ffmpeg

    frames = proxy.count_frames(
        cfg.data_root / "streams/fx/proxy/proxy.mp4", ffmpeg.require("ffprobe")
    )
    expected = round(manifest["duration_s"] * float(Fraction(manifest["fps"])))
    assert abs(frames - expected) <= int(cfg.get("ingest.proxy.verify.max_frame_delta"))


def test_keyframes_land_on_the_configured_gop(encoded):
    """A2, measured on the real encode. This is what makes the review UI's
    live seeking viable and it is verified rather than trusted."""
    cfg, conn, engine, _ = encoded
    from clipforge import ffmpeg

    positions = proxy.keyframe_positions(
        cfg.data_root / "streams/fx/proxy/proxy.mp4", 30.0, ffmpeg.require("ffprobe")
    )
    assert len(positions) > 5

    row = conn.execute("SELECT fps_num, fps_den FROM streams WHERE id='fx'").fetchone()
    rate = float(Fraction(row["fps_num"], row["fps_den"]))
    intervals = [round((b - a) * rate) for a, b in zip(positions, positions[1:], strict=False)]
    assert set(intervals) == {int(cfg.get("ingest.proxy.gop"))}


def test_proxy_is_not_taller_than_the_source(encoded, manifest):
    cfg, conn, engine, _ = encoded
    info = probe.probe_json(
        cfg.data_root / "streams/fx/proxy/proxy.mp4", "-select_streams", "v:0", "-show_streams"
    )
    source_h = int(manifest["resolution"].split("x")[1])
    assert int(info["streams"][0]["height"]) == source_h


def test_proxy_carries_the_mixed_track(encoded, manifest):
    cfg, conn, engine, _ = encoded
    info = probe.probe_json(
        cfg.data_root / "streams/fx/proxy/proxy.mp4", "-select_streams", "a", "-show_streams"
    )
    assert len(info["streams"]) == 1  # §5.2: one audio track


def test_proxy_path_is_recorded_relative_to_data_root(encoded):
    """The database has to survive being carried to another drive (§13.2)."""
    cfg, conn, engine, _ = encoded
    stored = conn.execute("SELECT proxy_path FROM streams WHERE id='fx'").fetchone()[0]
    assert stored == "streams/fx/proxy/proxy.mp4"
    assert (cfg.data_root / stored).is_file()


def test_stage_is_idempotent(encoded):
    cfg, conn, engine, _ = encoded
    decision = next(d for d in engine.plan() if d.stage == "proxy")
    assert decision.action == "skip"


def test_deleting_the_proxy_reruns_only_proxy(encoded):
    """The resume case: probe stays done, proxy comes back."""
    cfg, conn, engine, _ = encoded
    (cfg.data_root / "streams/fx/proxy/proxy.mp4").unlink()

    decisions = {d.stage: d for d in engine.plan()}
    assert decisions["probe"].action == "skip"
    assert decisions["proxy"].action == "run"
    assert "missing output" in decisions["proxy"].reason

    engine.execute(engine.plan())
    assert (cfg.data_root / "streams/fx/proxy/proxy.mp4").is_file()


def test_reprobing_makes_the_proxy_stale(encoded):
    """§5.1's rule alone cannot see this: proxy stays done forever while
    serving an artifact built from a superseded track map."""
    cfg, conn, engine, _ = encoded
    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"probe"}, log=lambda *_: None)
    decisions = {d.stage: d for d in forced.plan()}
    assert decisions["probe"].action == "run"
    assert decisions["proxy"].action == "run"
    assert "probe" in decisions["proxy"].reason


def test_changing_proxy_settings_forces_a_reencode(encoded):
    cfg, conn, engine, _ = encoded
    retuned = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "ingest.proxy.video_bitrate=4M",
    ])
    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx", log=lambda *_: None)
    decision = next(d for d in engine2.plan() if d.stage == "proxy")
    assert decision.action == "run"
    assert decision.reason == "parameters changed"


def test_verify_is_cheap(encoded):
    """verify() runs for every done stage on every plan; if it shelled out to
    ffprobe, `clipforge status` would slow down with each stream added."""
    cfg, conn, engine, _ = encoded
    import clipforge.ffmpeg as ffmpeg_module

    calls = []
    original = ffmpeg_module.run
    ffmpeg_module.run = lambda *a, **k: calls.append(a) or original(*a, **k)
    try:
        from clipforge.pipeline import stages
        stages.get("proxy").verify(engine.ctx)
    finally:
        ffmpeg_module.run = original
    assert calls == []


def test_verify_notices_a_wiped_proxy_path(encoded):
    cfg, conn, engine, _ = encoded
    conn.execute("UPDATE streams SET proxy_path = NULL WHERE id='fx'")
    try:
        ok, why = __import__(
            "clipforge.pipeline.stages", fromlist=["x"]
        ).get("proxy").verify(engine.ctx)
        assert not ok
        assert "proxy_path" in why
    finally:
        conn.execute(
            "UPDATE streams SET proxy_path = 'streams/fx/proxy/proxy.mp4' WHERE id='fx'"
        )
