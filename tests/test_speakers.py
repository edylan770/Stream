"""§5.8's speaker assignment.

The interesting tests read the speech fixture's manifest, which records the
measured level of every track across every authored line — so the expected label
is *derived from the fixture's own energies* rather than written down here. That
is the payoff of making utterances Regions in commit 18: a test that checks the
rule, not a test that checks a transcription.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge import config, db, signals
from clipforge.extract import speakers, transcript
from clipforge.extract.transcript import Segment, Word
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext


def linear(db_value: float) -> float:
    return 10 ** (db_value / 10.0)


@pytest.fixture
def labelled(tmp_path, speech_fixture_dir, speech_manifest):
    """The speech fixture, transcribed from its own manifest, then labelled."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)

    result = register.register_stream(
        cfg, conn,
        register.RegisterSpec(master=speech_fixture_dir / speech_manifest["video"]),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(
        ["register_stream", "probe", "audio_split", "audio_features"]
    ))

    ctx = StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id, log=lambda *_: None)

    class FromManifest:
        def transcribe(self, path, *, role, track):
            return [
                Segment(u["t_start"], u["t_end"], u["text"], track, role,
                        [Word(w, None, None, None) for w in u["text"].split()])
                for u in speech_manifest["utterances"] if u["track"] == role
            ]

    transcript.run(ctx, transcriber=FromManifest())
    speakers.run(ctx)

    yield cfg, conn, ctx
    conn.close()


def segments_of(conn, stream_id):
    return conn.execute(
        "SELECT * FROM segments WHERE stream_id = ? ORDER BY seq", (stream_id,)
    ).fetchall()


# --------------------------------------------------------------------------
# the arithmetic §5.8 gets wrong
# --------------------------------------------------------------------------


def test_db_converts_to_linear_power():
    """-10 dB is a tenth of the power, not a tenth of -10."""
    assert speakers.db_to_power(np.array([0.0]))[0] == pytest.approx(1.0)
    assert speakers.db_to_power(np.array([-10.0]))[0] == pytest.approx(0.1)
    assert speakers.db_to_power(np.array([-20.0]))[0] == pytest.approx(0.01)


def test_the_mean_is_taken_in_the_linear_domain():
    """Averaging dB gives the geometric mean of the powers, which understates
    any window containing a peak — exactly the windows that matter."""
    series = signals.Series(kind="x", values=np.array([-40.0, 0.0]), sample_rate_hz=10.0)
    got = speakers.mean_power(series, 0.0, 0.1)
    assert got == pytest.approx((linear(-40.0) + linear(0.0)) / 2)
    # The naive answer, for contrast: mean(-40, 0) = -20 dB = 0.01.
    assert got > linear(-20.0)


def test_the_literal_spec_rule_mislabels_the_fixtures_overlap(speech_manifest):
    """MEASURED. §5.8 compares `mean_energy` values with a 1.5 ratio, and ours
    are dBFS — logarithms. On the fixture's overlapped party line the party
    track is the louder, and the literal rule returns 'operator'."""
    overlaps = speech_manifest["overlap_windows"]
    entry = next(
        u for u in speech_manifest["utterances"]
        if u["speaker"] == "B"
        and any(u["t_start"] < end and u["t_end"] > start for start, end in overlaps)
    )
    mic, party = entry["measured_dbfs"]["mic"], entry["measured_dbfs"]["party"]

    assert party > mic, "the party track should be the louder one here"
    assert mic > party * 1.5, "the naive dB comparison should (wrongly) fire"
    assert speakers.classify(linear(mic), linear(party), 1.5) == speakers.BOTH


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mic,party,expected", [
    (1.0, 0.01, speakers.OPERATOR),
    (0.01, 1.0, speakers.PARTY),
    (1.0, 1.0, speakers.BOTH),
    (1.0, 0.8, speakers.BOTH),      # within the ratio, so neither dominates
    (None, None, speakers.UNKNOWN),
])
def test_classify(mic, party, expected):
    assert speakers.classify(mic, party, 1.5) == expected


def test_a_single_track_recording_is_all_operator():
    """§4.2's contaminated case: there is no party track to lose to."""
    assert speakers.classify(0.5, None, 1.5) == speakers.OPERATOR


def test_the_ratio_moves_the_boundary():
    """Higher ratio means more `both`, which is the tuning knob §17 forgot."""
    assert speakers.classify(1.0, 0.5, 1.5) == speakers.OPERATOR
    assert speakers.classify(1.0, 0.5, 3.0) == speakers.BOTH


def test_the_ratio_comes_from_config():
    assert config.load().get("extract.speaker.dominance_ratio") == 1.5


# --------------------------------------------------------------------------
# against the fixture's own measured energies
# --------------------------------------------------------------------------


def test_every_segment_is_labelled(labelled, speech_manifest):
    _cfg, conn, ctx = labelled
    rows = segments_of(conn, ctx.stream_id)
    assert len(rows) == len(speech_manifest["utterances"])
    assert all(r["speaker"] for r in rows)


def test_labels_match_what_the_fixtures_energies_imply(labelled, speech_manifest):
    """Derived from the manifest's measured per-track levels, not written down:
    the test checks §5.8's rule rather than a memorised answer."""
    _cfg, conn, ctx = labelled
    rows = {round(r["t_start"], 2): r["speaker"] for r in segments_of(conn, ctx.stream_id)}

    for entry in speech_manifest["utterances"]:
        expected = speakers.classify(
            linear(entry["measured_dbfs"]["mic"]),
            linear(entry["measured_dbfs"]["party"]),
            1.5,
        )
        assert rows[round(entry["t_start"], 2)] == expected, entry["label"]


def test_solo_lines_get_their_authored_speaker(labelled, speech_manifest):
    """Outside the overlap the separation is ~35 dB, so there is one right
    answer and it is the one the script authored."""
    _cfg, conn, ctx = labelled
    rows = {round(r["t_start"], 2): r["speaker"] for r in segments_of(conn, ctx.stream_id)}
    overlaps = speech_manifest["overlap_windows"]

    checked = 0
    for entry in speech_manifest["utterances"]:
        if any(entry["t_start"] < end and entry["t_end"] > start for start, end in overlaps):
            continue
        expected = speakers.OPERATOR if entry["speaker"] == "A" else speakers.PARTY
        assert rows[round(entry["t_start"], 2)] == expected, entry["label"]
        checked += 1
    assert checked > 4


def test_the_overlap_is_labelled_both(labelled, speech_manifest):
    """§5.8's third branch, and the only case with no single right answer."""
    _cfg, conn, ctx = labelled
    rows = segments_of(conn, ctx.stream_id)
    overlaps = speech_manifest["overlap_windows"]

    inside = [
        r for r in rows
        if any(r["t_start"] < end and r["t_end"] > start for start, end in overlaps)
    ]
    assert inside
    assert any(r["speaker"] == speakers.BOTH for r in inside)


def test_the_game_track_does_not_steal_a_line(labelled, speech_manifest):
    """§4.2's contamination case: the game is 8 dB over the mic for three lines,
    and §5.8 must still call them the operator because the game is not a
    speaker."""
    _cfg, conn, ctx = labelled
    rows = {round(r["t_start"], 2): r["speaker"] for r in segments_of(conn, ctx.stream_id)}
    # Authored as A, specifically: every one of B's lines also has the game
    # above the mic, because the mic is at its floor while B is talking.
    loud = [
        u for u in speech_manifest["utterances"]
        if u["speaker"] == "A" and u["measured_dbfs"]["game"] > u["measured_dbfs"]["mic"]
    ]
    assert loud
    for entry in loud:
        assert rows[round(entry["t_start"], 2)] == speakers.OPERATOR


# --------------------------------------------------------------------------
# stage plumbing
# --------------------------------------------------------------------------


def test_running_twice_is_stable(labelled):
    _cfg, conn, ctx = labelled
    before = [r["speaker"] for r in segments_of(conn, ctx.stream_id)]
    speakers.run(ctx)
    assert [r["speaker"] for r in segments_of(conn, ctx.stream_id)] == before


def test_verify_needs_every_segment_labelled(labelled):
    _cfg, conn, ctx = labelled
    assert speakers.verify(ctx)[0]

    conn.execute(
        "UPDATE segments SET speaker = NULL WHERE stream_id = ? AND seq = 0",
        (ctx.stream_id,),
    )
    ok, why = speakers.verify(ctx)
    assert not ok and "speaker" in why


def test_it_refuses_when_there_is_no_transcript(labelled):
    _cfg, conn, ctx = labelled
    conn.execute("DELETE FROM segments WHERE stream_id = ?", (ctx.stream_id,))
    with pytest.raises(speakers.SpeakerError, match="no segments"):
        speakers.run(ctx)


def test_it_refuses_when_there_is_no_audio_signal(labelled):
    _cfg, conn, ctx = labelled
    conn.execute("DELETE FROM signal_series WHERE stream_id = ?", (ctx.stream_id,))
    with pytest.raises(speakers.SpeakerError, match="audio_features"):
        speakers.run(ctx)


def test_it_follows_the_transcript_stages_availability(labelled):
    """Labelling nothing is not worth a stage run, and a Phase 1 machine with no
    ASR extra must not see this fail."""
    _cfg, _conn, ctx = labelled
    usable, why = speakers.available(ctx)
    assert not usable and "enabled" in why


def test_the_stage_is_registered():
    from clipforge.pipeline import stages

    spec = stages.get("speaker_assign")
    assert spec.module == "clipforge.extract.speakers"
    assert set(spec.requires) == {"whisperx", "audio_features"}
