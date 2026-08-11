"""probe: frame-rate classification, track roles, and the streams row."""

from __future__ import annotations

from fractions import Fraction

import pytest

from clipforge import config, db
from clipforge.ingest import probe

# --------------------------------------------------------------------------
# frame-rate classification
# --------------------------------------------------------------------------


def cfr_pts(count, rate, time_base=None, start=0.0):
    """PTS for a constant-rate stream, quantized to the timebase like a real
    container does."""
    step = 1.0 / float(rate)
    values = [start + i * step for i in range(count)]
    if time_base:
        tick = float(time_base)
        values = [round(v / tick) * tick for v in values]
    return values


def test_millisecond_quantized_cfr_is_not_vfr():
    """The measured case that breaks the two obvious detectors.

    A CFR 30 fps MKV on a 1/1000 timebase cannot represent 33.333 ms, so it
    stores alternating 33/34 ms frames. Comparing PTS deltas calls that
    variable-rate — which would flag every OBS recording ever made.
    """
    pts = cfr_pts(120, Fraction(30, 1), Fraction(1, 1000))
    deltas = {round(b - a, 6) for a, b in zip(pts, pts[1:], strict=False)}
    assert len(deltas) > 1, "the fixture case requires genuinely varying deltas"

    verdict = probe.classify_frame_rate(
        Fraction(30, 1), Fraction(30, 1), Fraction(1, 1000), pts
    )
    assert not verdict.is_vfr
    assert verdict.rate == Fraction(30, 1)


def test_ntsc_rate_on_a_coarse_timebase_is_not_vfr():
    pts = cfr_pts(200, Fraction(30000, 1001), Fraction(1, 1000))
    verdict = probe.classify_frame_rate(
        Fraction(30000, 1001), Fraction(30000, 1001), Fraction(1, 1000), pts
    )
    assert not verdict.is_vfr


def test_fine_timebase_cfr_is_not_vfr():
    pts = cfr_pts(200, Fraction(60, 1), Fraction(1, 15360))
    verdict = probe.classify_frame_rate(
        Fraction(60, 1), Fraction(60, 1), Fraction(1, 15360), pts
    )
    assert not verdict.is_vfr


def test_drifting_timestamps_are_vfr():
    """Dropped frames accumulate; the residual grows without bound."""
    pts = cfr_pts(120, Fraction(30, 1), Fraction(1, 1000))
    drifting = [value + (index * 0.004) for index, value in enumerate(pts)]
    verdict = probe.classify_frame_rate(
        Fraction(30, 1), Fraction(30, 1), Fraction(1, 1000), drifting
    )
    assert verdict.is_vfr
    assert "drifts" in verdict.reason


def test_a_single_long_pause_is_vfr():
    """Screen capture that stops emitting frames when nothing changes."""
    pts = cfr_pts(60, Fraction(30, 1), Fraction(1, 1000))
    pts += [value + 2.0 for value in cfr_pts(60, Fraction(30, 1), Fraction(1, 1000), start=2.0)]
    verdict = probe.classify_frame_rate(
        Fraction(30, 1), Fraction(30, 1), Fraction(1, 1000), pts
    )
    assert verdict.is_vfr


def test_too_few_frames_falls_back_rather_than_guessing():
    verdict = probe.classify_frame_rate(Fraction(30, 1), Fraction(30, 1), Fraction(1, 1000), [0.0])
    assert not verdict.is_vfr
    assert "too few" in verdict.reason


def test_missing_rate_is_an_error():
    with pytest.raises(probe.ProbeError, match="frame rate"):
        probe.classify_frame_rate(None, None, Fraction(1, 1000), [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [("30/1", Fraction(30, 1)), ("30000/1001", Fraction(30000, 1001)),
     ("0/0", None), ("N/A", None), (None, None), ("", None)],
)
def test_parse_rate(text, expected):
    assert probe.parse_rate(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (Fraction(5994, 100), Fraction(60000, 1001)),   # ugly VFR average
        (Fraction(2997, 100), Fraction(30000, 1001)),
        (Fraction(2398, 100), Fraction(24000, 1001)),
        (Fraction(48, 1), Fraction(48, 1)),             # genuinely non-standard
        (Fraction(144, 1), Fraction(144, 1)),
    ],
)
def test_standard_rate_snapping(raw, expected):
    """FCPXML needs frameDuration as an exact rational; 5994/100 is not one."""
    assert probe.snap_to_standard(raw) == expected


@pytest.mark.parametrize("rate", probe.STANDARD_RATES)
def test_exact_standard_rates_are_never_rewritten(rate):
    """59.94 and 60 sit 0.1% apart, so any tolerance loose enough to catch a
    scruffy 59.94 also spans exact 60. Snapping to the first match in range
    would rewrite genuine 60 fps footage as 59.94 and put every exported
    timeline on the wrong timebase."""
    assert probe.snap_to_standard(rate) == rate


def test_snapping_picks_the_nearest_not_the_first():
    assert probe.snap_to_standard(Fraction(599, 10)) == Fraction(60000, 1001)   # 59.9
    assert probe.snap_to_standard(Fraction(6001, 100)) == Fraction(60, 1)       # 60.01


# --------------------------------------------------------------------------
# track roles
# --------------------------------------------------------------------------


def streams_with_titles(*titles):
    return [{"codec_type": "audio", "tags": {"title": t} if t else {}} for t in titles]


def test_obs_style_titles_are_recognised():
    """The fixture carries the titles OBS writes."""
    result = probe.resolve_track_roles(
        streams_with_titles("Mixed", "Mic/Aux", "Desktop Audio", "Discord"), {}
    )
    assert result.roles == {"mixed": 0, "mic": 1, "game": 2, "party": 3}
    assert result.source == "metadata tags"
    assert not result.warnings


def test_positional_fallback_matches_the_spec_layout():
    """§4.2: track 1 mixed, 2 mic, 3 game, 4 party — 0-based for ffmpeg."""
    result = probe.resolve_track_roles(streams_with_titles(None, None, None, None), {})
    assert result.roles == {"mixed": 0, "mic": 1, "game": 2, "party": 3}
    assert "positional" in result.source


def test_single_track_maps_to_mic_and_warns_loudly():
    result = probe.resolve_track_roles(streams_with_titles(None), {})
    assert result.roles == {"mic": 0}
    assert len(result.warnings) == 1
    assert "unrecoverable" in result.warnings[0]


def test_two_tracks_refuse_rather_than_guess():
    """Guessing which of two tracks is the mic is a coin flip, and losing it
    poisons every RMS reading in the stream (§4.2)."""
    with pytest.raises(probe.ProbeError, match="track_roles"):
        probe.resolve_track_roles(streams_with_titles(None, None), {})


def test_three_tracks_also_refuse():
    with pytest.raises(probe.ProbeError, match="guessing"):
        probe.resolve_track_roles(streams_with_titles(None, None, None), {})


def test_explicit_config_overrides_everything():
    result = probe.resolve_track_roles(
        streams_with_titles("Mixed", "Mic/Aux", "Desktop Audio", "Discord"),
        {"mic": 3, "game": 0},
    )
    assert result.roles == {"mic": 3, "game": 0}
    assert result.source == "config"


def test_config_pointing_outside_the_file_is_an_error():
    with pytest.raises(probe.ProbeError, match="audio stream 9"):
        probe.resolve_track_roles(streams_with_titles(None, None), {"mic": 9})


def test_null_config_entries_are_ignored():
    """`track_roles: {mic: null}` means infer, not 'map mic to nothing'."""
    result = probe.resolve_track_roles(
        streams_with_titles(None), {"mic": None, "game": None, "party": None}
    )
    assert result.roles == {"mic": 0}


def test_no_audio_is_an_error():
    with pytest.raises(probe.ProbeError, match="no audio"):
        probe.resolve_track_roles([], {})


def test_generic_mp4_handler_names_are_not_treated_as_titles():
    """MP4 writes handler_name='SoundHandler' on every track; treating that as
    a label would produce confident nonsense."""
    streams = [{"codec_type": "audio", "tags": {"handler_name": "SoundHandler"}}]
    result = probe.resolve_track_roles(streams, {})
    assert result.roles == {"mic": 0}
    assert result.warnings


@pytest.mark.parametrize(
    ("title", "role"),
    [("Mic/Aux", "mic"), ("Desktop Audio", "game"), ("Discord", "party"),
     ("Mixed", "mixed"), ("Voice Chat", "mic"), ("nonsense", None)],
)
def test_title_hints(title, role):
    assert probe.role_from_title(title) == role


# --------------------------------------------------------------------------
# against the real fixture
# --------------------------------------------------------------------------


def test_probe_reads_the_fixture_correctly(fixture_dir, manifest):
    cfg = config.load()
    result = probe.analyse(fixture_dir / manifest["video"], cfg)

    assert result.duration_s == pytest.approx(manifest["duration_s"], abs=0.2)
    assert f"{result.width}x{result.height}" == manifest["resolution"]
    assert result.rate == Fraction(manifest["fps"])
    assert result.is_vfr is manifest["expected_is_vfr"]
    assert result.track_map.roles == manifest["expected_track_map"]
    assert not result.track_map.warnings


def test_fixture_duration_comes_from_the_container(fixture_dir, manifest):
    """MKV reports neither nb_frames nor a per-stream duration, so probe must
    not depend on stream-level fields for either."""
    info = probe.probe_container(fixture_dir / manifest["video"])
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert "duration" not in video and "nb_frames" not in video
    assert float(info["format"]["duration"]) == pytest.approx(manifest["duration_s"], abs=0.2)


def test_fixture_classifies_as_cfr_from_real_timestamps(fixture_dir, manifest):
    """End to end on the real file, not synthesized PTS."""
    video = fixture_dir / manifest["video"]
    pts = probe.sample_pts(video, start_s=0, seconds=4.0)
    assert len(pts) > 100

    deltas = {round(b - a, 6) for a, b in zip(pts, pts[1:], strict=False)}
    assert len(deltas) > 1, "MKV millisecond quantization should vary the deltas"

    verdict = probe.classify_frame_rate(
        Fraction(30, 1), Fraction(30, 1), Fraction(1, 1000), pts
    )
    assert not verdict.is_vfr


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


@pytest.fixture
def probed(tmp_path, fixture_dir, manifest):
    """Register the fixture and run probe through the real runner."""
    from clipforge.pipeline import runner

    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) VALUES "
        "('fx', '2026-08-14', ?, 'epoch')",
        (str(fixture_dir / manifest["video"]),),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan(["register_stream", "probe"]))
    yield cfg, conn, engine, results
    conn.close()


def test_stage_writes_the_streams_row(probed, manifest):
    cfg, conn, engine, results = probed
    assert [r.status for r in results] == ["done"]

    row = conn.execute("SELECT * FROM streams WHERE id='fx'").fetchone()
    assert row["duration_s"] == pytest.approx(manifest["duration_s"], abs=0.2)
    assert row["resolution"] == manifest["resolution"]
    assert row["is_vfr"] == 0
    # Exact rational, not a float: FCPXML needs 1001-denominator rates intact.
    assert (row["fps_num"], row["fps_den"]) == (30, 1)


def test_stage_is_idempotent(probed):
    cfg, conn, engine, _ = probed
    decision = next(d for d in engine.plan(["register_stream", "probe"]) if d.stage == "probe")
    assert decision.action == "skip"


def test_verify_catches_a_wiped_row(probed):
    """probe's product is columns; file-existence checks cannot see this."""
    cfg, conn, engine, _ = probed
    conn.execute("UPDATE streams SET audio_track_map = NULL WHERE id='fx'")
    decision = next(d for d in engine.plan(["register_stream", "probe"]) if d.stage == "probe")
    assert decision.action == "run"
    assert "track map" in decision.reason


def test_changing_track_role_config_reruns_probe(probed, tmp_path):
    """The mapping is the whole point of the stage, so overriding it must not
    leave a stale map in place."""
    cfg, conn, engine, _ = probed
    from clipforge.pipeline import runner

    retuned = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        "ingest.audio.track_roles.mic=2",
    ])
    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx", log=lambda *_: None)
    decision = next(d for d in engine2.plan(["register_stream", "probe"]) if d.stage == "probe")
    assert decision.action == "run"
    assert decision.reason == "parameters changed"
