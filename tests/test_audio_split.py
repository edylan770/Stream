"""audio_split, and the resume behaviour the stage DAG exists to provide.

Levels are asserted against `manifest.json`'s measured values, which are
themselves measured from the authored signal rather than declared — so nothing
here depends on a number typed into a test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import soundfile as sf

from clipforge import config, db
from clipforge.extract import audio
from clipforge.pipeline import runner, stages

#: Lossless FLAC in, PCM out; the only transform is the 48k -> 16k resample of
#: band-limited content, which measures at hundredths of a dB.
DB_TOLERANCE = 0.25


# --------------------------------------------------------------------------
# role selection
# --------------------------------------------------------------------------


def test_extracts_the_spec_roles_in_order():
    """§5.3 maps a:1/2/3 -- mic, game, party."""
    track_map = {"roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3}}
    assert audio.roles_to_extract(track_map, ["mic", "game", "party"]) == {
        "mic": 1, "game": 2, "party": 3
    }


def test_mixed_is_never_extracted():
    """The proxy already carries the mix for review; extracting it again would
    add ~460 MB per stream for something nothing analyses."""
    track_map = {"roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3}}
    assert "mixed" not in audio.roles_to_extract(track_map, ["mic", "game", "party"])


def test_single_track_file_yields_only_mic():
    """Demanding all three would make a one-track recording unprocessable."""
    assert audio.roles_to_extract({"roles": {"mic": 0}}, ["mic", "game", "party"]) == {"mic": 0}


def test_missing_track_map_yields_nothing():
    assert audio.roles_to_extract({}, ["mic"]) == {}


def test_config_controls_which_roles_are_extracted():
    track_map = {"roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3}}
    assert audio.roles_to_extract(track_map, ["mic"]) == {"mic": 1}


# --------------------------------------------------------------------------
# the §5.3 command
# --------------------------------------------------------------------------


def test_command_extracts_every_track_in_one_pass(tmp_path):
    """One invocation, not one per track: ffmpeg demuxes the master once, which
    on a 50 GB file is the difference between one read pass and three."""
    cfg = config.load()
    targets = [
        ("mic", 1, tmp_path / "mic.wav"),
        ("game", 2, tmp_path / "game.wav"),
        ("party", 3, tmp_path / "party.wav"),
    ]
    argv = audio.build_command("ffmpeg", tmp_path / "m.mkv", targets, cfg=cfg)

    assert argv.count("-i") == 1
    for _role, index, path in targets:
        assert f"0:a:{index}" in argv
        assert str(path) in argv


def test_command_uses_the_spec_audio_format(tmp_path):
    """§5.3: 'Do not extract at higher rates.'"""
    cfg = config.load()
    argv = audio.build_command(
        "ffmpeg", tmp_path / "m.mkv", [("mic", 1, tmp_path / "mic.wav")], cfg=cfg
    )
    assert argv[argv.index("-ar") + 1] == str(cfg.get("ingest.audio.sample_rate_hz")) == "16000"
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-c:a") + 1] == "pcm_s16le"


def test_command_does_not_violate_a1(tmp_path):
    from clipforge import ffmpeg

    cfg = config.load()
    argv = audio.build_command(
        "ffmpeg", tmp_path / "m.mkv", [("mic", 1, tmp_path / "mic.wav")], cfg=cfg
    )
    ffmpeg.validate_seek_placement(argv)


# --------------------------------------------------------------------------
# verification rules
# --------------------------------------------------------------------------


def summary(role="mic", duration=60.0, rate=16000, channels=1, peak=-6.0, path=None):
    from pathlib import Path

    return audio.TrackSummary(
        role=role, path=path or Path(f"{role}.wav"), duration_s=duration,
        sample_rate=rate, channels=channels, peak_dbfs=peak,
    )


def test_matching_tracks_pass():
    audio.assert_sound(
        [summary()], master_duration=60.0, cfg=config.load(), log=lambda *_: None
    )


def test_truncated_track_is_fatal():
    """Silent corruption otherwise: candidates simply stop appearing partway
    through a stream, which is near-impossible to diagnose from the far end."""
    with pytest.raises(audio.AudioSplitError, match="truncated"):
        audio.assert_sound(
            [summary(duration=40.0)], master_duration=60.0,
            cfg=config.load(), log=lambda *_: None,
        )


def test_wrong_sample_rate_is_fatal():
    with pytest.raises(audio.AudioSplitError, match="§5.3"):
        audio.assert_sound(
            [summary(rate=44100)], master_duration=60.0,
            cfg=config.load(), log=lambda *_: None,
        )


def test_silent_mic_warns_loudly_but_does_not_fail():
    """A silent mic means the track mapping is wrong, and nothing else in the
    pipeline would say so -- scoring just quietly finds nothing."""
    messages = []
    audio.assert_sound(
        [summary(peak=float("-inf"))], master_duration=60.0,
        cfg=config.load(), log=messages.append,
    )
    assert any("WARNING" in m and "track mapping" in m for m in messages)


def test_silent_party_is_merely_noted():
    """Nobody in Discord is the ordinary state of a solo stream."""
    messages = []
    audio.assert_sound(
        [summary(role="party", peak=float("-inf"))], master_duration=60.0,
        cfg=config.load(), log=messages.append,
    )
    assert not any("WARNING" in m for m in messages)
    assert any("silent" in m for m in messages)


# --------------------------------------------------------------------------
# the real extraction
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def split(tmp_path_factory, fixture_dir, manifest):
    """Register, probe, proxy and split the fixture once for the session."""
    root = tmp_path_factory.mktemp("split")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('fx', '2026-08-14', ?, 'epoch')",
        (str(fixture_dir / manifest["video"]),),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan())
    yield cfg, conn, engine, results
    conn.close()


def test_split_succeeds(split):
    cfg, conn, engine, results = split
    assert all(r.status == "done" for r in results), [r.error for r in results]


def test_writes_one_wav_per_analysis_role(split, manifest):
    cfg, conn, engine, _ = split
    audio_dir = cfg.data_root / "streams/fx/audio"
    written = sorted(p.stem for p in audio_dir.glob("*.wav"))
    assert written == ["game", "mic", "party"]
    assert not (audio_dir / "mixed.wav").exists()


def test_wavs_are_16k_mono(split, manifest):
    cfg, conn, engine, _ = split
    for role in ("mic", "game", "party"):
        info = sf.info(cfg.data_root / f"streams/fx/audio/{role}.wav")
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.frames / info.samplerate == pytest.approx(manifest["duration_s"], abs=0.05)


def test_extracted_levels_match_the_manifest(split, manifest):
    """The point of the whole fixture: the pipeline's audio is the audio that
    was authored, to a fraction of a dB."""
    cfg, conn, engine, _ = split
    tracks = {
        role: sf.read(cfg.data_root / f"streams/fx/audio/{role}.wav", dtype="float32")[0]
        for role in ("mic", "game", "party")
    }

    for region in manifest["regions"]:
        role = region["source_track"]
        data = tracks[role]
        pad = int(0.02 * 16000)
        start = int(region["t_start"] * 16000) + pad
        end = int(region["t_end"] * 16000) - pad
        measured = 20 * math.log10(float(np.sqrt(np.mean(data[start:end] ** 2))))
        assert measured == pytest.approx(
            region["measured_dbfs"][role], abs=DB_TOLERANCE
        ), region["label"]


def test_contamination_survives_into_the_extracted_audio(split, manifest):
    """§4.2 made concrete on real files: the game explosion and the operator's
    loudest speech are separable on their own tracks and not on the mix."""
    cfg, conn, engine, _ = split
    by_label = {r["label"].split("#")[0]: r for r in manifest["regions"]}
    explosion, speech = by_label["game_explosion"], by_label["speech_peak"]

    mic = sf.read(cfg.data_root / "streams/fx/audio/mic.wav", dtype="float32")[0]
    game = sf.read(cfg.data_root / "streams/fx/audio/game.wav", dtype="float32")[0]

    def level(data, region):
        a, b = int(region["t_start"] * 16000) + 320, int(region["t_end"] * 16000) - 320
        return 20 * math.log10(float(np.sqrt(np.mean(data[a:b] ** 2))))

    # The mic is near its floor while the game is at its loudest...
    assert level(game, explosion) - level(mic, explosion) > 20
    # ...and the two loud events are the same level on their own tracks.
    assert level(game, explosion) == pytest.approx(level(mic, speech), abs=0.5)


def test_silence_summary_reports_real_levels(split, manifest):
    cfg, conn, engine, _ = split
    result = audio.summarize("mic", cfg.data_root / "streams/fx/audio/mic.wav", 12)
    assert not result.is_silent
    assert result.peak_dbfs <= manifest["peak_ceiling_dbfs"] + 0.5
    assert result.duration_s == pytest.approx(manifest["duration_s"], abs=0.05)


def test_stage_is_idempotent(split):
    cfg, conn, engine, _ = split
    decision = next(d for d in engine.plan() if d.stage == "audio_split")
    assert decision.action == "skip"


def test_partial_extraction_cannot_look_complete(split):
    """atomic_outputs: two of three tracks in place would satisfy an existence
    check while being wrong."""
    cfg, conn, engine, _ = split
    (cfg.data_root / "streams/fx/audio/game.wav").unlink()
    ok, why = stages.get("audio_split").verify(engine.ctx)
    assert not ok
    assert "game.wav" in why
    engine.execute(engine.plan())
    assert (cfg.data_root / "streams/fx/audio/game.wav").is_file()


# --------------------------------------------------------------------------
# THE FULL RESUME TEST
# --------------------------------------------------------------------------


def test_deleting_the_proxy_reruns_only_the_proxy(split):
    """What the DAG buys over §5.1's numbered list.

    proxy and audio_split are siblings — both depend on probe, neither on the
    other. Deleting the proxy must not cost a re-extraction of the audio, and a
    chain-shaped pipeline would do exactly that.
    """
    cfg, conn, engine, _ = split
    (cfg.data_root / "streams/fx/proxy/proxy.mp4").unlink()

    decisions = {d.stage: d for d in engine.plan()}
    assert decisions["probe"].action == "skip"
    assert decisions["proxy"].action == "run"
    assert decisions["audio_split"].action == "skip"

    ran = [r.stage for r in engine.execute(engine.plan())]
    assert ran == ["proxy"]


def test_deleting_a_wav_leaves_the_proxy_alone(split):
    """The mirror case: re-extracting audio must not re-encode the proxy.

    Stages downstream of audio_split legitimately re-run with it — that is the
    cascade working — so the invariant is about the *sibling*, not about the
    count. Written this way it does not need editing every time a stage is
    added below audio_split.
    """
    cfg, conn, engine, _ = split
    (cfg.data_root / "streams/fx/audio/mic.wav").unlink()

    decisions = {d.stage: d for d in engine.plan()}
    assert decisions["probe"].action == "skip"
    assert decisions["proxy"].action == "skip"
    assert decisions["audio_split"].action == "run"

    ran = [r.stage for r in engine.execute(engine.plan())]
    assert ran[0] == "audio_split"
    assert "proxy" not in ran
    assert set(ran) <= {"audio_split"} | stages.descendants("audio_split")


def test_reprobing_reruns_both_children(split):
    """probe is upstream of both, so both go stale — this is the case §5.1's
    'skip if done' rule cannot see at all."""
    cfg, conn, engine, _ = split
    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"probe"}, log=lambda *_: None)
    decisions = {d.stage: d for d in forced.plan()}
    assert decisions["proxy"].action == "run"
    assert decisions["audio_split"].action == "run"
    assert "probe" in decisions["audio_split"].reason


def test_changing_audio_config_does_not_touch_the_proxy(split):
    """params_hash is per stage, so retuning extraction must not invalidate an
    unrelated four-hour encode."""
    cfg, conn, engine, _ = split
    retuned = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        'ingest.audio.extract_roles=["mic"]',
    ])
    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx", log=lambda *_: None)
    decisions = {d.stage: d for d in engine2.plan()}
    assert decisions["audio_split"].action == "run"
    assert decisions["audio_split"].reason == "parameters changed"
    assert decisions["proxy"].action == "skip"
