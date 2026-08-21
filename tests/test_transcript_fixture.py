"""The authored three-chapter stream, and the scripted LLM source.

Both are instruments rather than features: §9.4's digest is a map-reduce over
chapters, and nothing here had more than one chapter to map over or any way to
exercise §12 without a key. The measuring instrument comes before the thing
measured.

Every number below is read from `transcript_fixture.json` or from the config
object. `tests/test_gates.py` states the rule ("no number here is written
down") and this file keeps it.
"""

from __future__ import annotations

import pytest

from clipforge import config, db
from clipforge.digest import chapters
from clipforge.extract import phrases
from clipforge.llm import parse_reply, validate_selections
from clipforge.pipeline.context import StageContext
from tests import fakes
from tests.fixtures import transcript


@pytest.fixture(scope="module")
def authored(tmp_path_factory):
    """Seeded once: no ffmpeg, no Ollama, no network. Well under a second."""
    root = tmp_path_factory.mktemp("authored")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    data = transcript.seed(cfg, conn)
    yield cfg, conn, data
    conn.close()


def _segments(conn, stream_id):
    return {int(r["seq"]): r["text"] for r in conn.execute(
        "SELECT seq, text FROM segments WHERE stream_id = ?", (stream_id,))}


# --------------------------------------------------------------------------
# the fixture's own precondition
# --------------------------------------------------------------------------


def test_the_gate_reads_the_authored_speech_and_silence(authored):
    """Everything else here rests on §6.4's gate separating the two levels.

    Checked in one place so a library change breaks this with a sentence about
    the gate, rather than leaving a chapter assertion failing for a reason that
    looks like a chapter bug.
    """
    cfg, conn, data = authored
    transcript.assert_gate_separates(cfg, conn, data)


def test_a_flat_level_would_not_work_which_is_why_it_is_not_flat(authored):
    """The constraint that shaped the fixture, asserted so it is not rediscovered.

    `speech_gate` is `rms > rolling_mean + margin_db`. A constant level cannot
    exceed its own rolling mean, so a flat "speech" stretch reads as silence
    everywhere and the whole stream would be one chapter.
    """
    cfg, _, data = authored
    assert float(data["burst_s"]) > 0 and float(data["pause_s"]) > 0
    span = float(data["levels_dbfs"]["speech"]) - float(data["levels_dbfs"]["pause"])
    assert span > float(cfg.get("score.derived.vad.margin_db")), (
        "the authored bursts and pauses must differ by more than the VAD margin")


# --------------------------------------------------------------------------
# §9.3, on a stream that finally has chapters
# --------------------------------------------------------------------------


def test_the_authored_chapters_are_found(authored):
    cfg, conn, data = authored
    result = chapters.segment(conn, str(data["stream_id"]), cfg,
                              float(data["duration_s"]))

    assert len(result.chapters) == len(data["chapters"])
    tolerance = 1.0 / float(cfg.get("score.score_grid_hz"))
    found = [c.t_start for c in result.chapters[1:]]
    for expected, actual in zip(data["expected_boundaries_s"], found, strict=True):
        assert abs(actual - float(expected)) <= tolerance, (expected, actual)


def test_a_boundary_lands_at_the_END_of_its_gap(authored):
    """§9.3's rule, and the first stream with room to show it: the dead air
    belongs to the chapter that just finished, not the one about to start."""
    cfg, conn, data = authored
    result = chapters.segment(conn, str(data["stream_id"]), cfg,
                              float(data["duration_s"]))
    tolerance = 1.0 / float(cfg.get("score.score_grid_hz"))

    for chapter, gap in zip(result.chapters[1:], data["silence_gaps"], strict=True):
        assert abs(chapter.t_start - float(gap["t_end"])) <= tolerance
        assert chapter.t_start - float(gap["t_start"]) > tolerance


def test_silence_is_the_only_input_that_contributed(authored):
    """Said rather than implied. §9.3 names four inputs and three of them
    produce nothing on a stream processed with shipped defaults."""
    cfg, conn, data = authored
    result = chapters.segment(conn, str(data["stream_id"]), cfg,
                              float(data["duration_s"]))

    assert result.contributing == ["silence"]
    for name in ("embedding", "scene", "game"):
        assert name in result.inputs and result.inputs[name]


def test_every_chapter_falls_inside_9_3s_target_range(authored):
    """The first fixture where `target_min_s`/`target_max_s` mean anything: a
    95-second stream is honestly one chapter and cannot exercise them."""
    cfg, conn, data = authored
    result = chapters.segment(conn, str(data["stream_id"]), cfg,
                              float(data["duration_s"]))

    low = float(cfg.get("digest.chapters.target_min_s"))
    high = float(cfg.get("digest.chapters.target_max_s"))
    for chapter in result.chapters:
        assert low <= chapter.duration_s <= high, (chapter.index, chapter.duration_s)
    assert not result.unmet_targets


def test_every_utterance_lands_in_the_chapter_it_was_authored_for(authored):
    """The property §9.4 depends on: a map over chapters must not hand a model
    a line from the wrong one, and a gap in the tiling would silently drop a
    stretch of transcript with no error at all."""
    cfg, conn, data = authored
    result = chapters.segment(conn, str(data["stream_id"]), cfg,
                              float(data["duration_s"]))

    for u in data["utterances"]:
        t = float(u["t"])
        authored_index = transcript.chapter_of(data, t)
        found = [c.index for c in result.chapters if c.t_start <= t < c.t_end]
        assert found == [authored_index], (u["seq"], t, authored_index, found)


# --------------------------------------------------------------------------
# the planted phrases, through the real detector
# --------------------------------------------------------------------------


def test_the_planted_repeat_fires_on_the_third_occurrence(authored):
    """Not planted as `events` rows: run the real detector over the authored
    segments, so the fixture proves the code path rather than restating it."""
    cfg, conn, data = authored
    ctx = StageContext(cfg=cfg, conn=conn, stream_id=str(data["stream_id"]),
                       log=lambda _m: None)
    phrases.run(ctx)

    planted = data["planted"]["repeat_phrase"]
    fired = conn.execute(
        "SELECT t, meta FROM events WHERE stream_id = ? AND kind = 'phrase_repeat' "
        "ORDER BY t", (str(data["stream_id"]),)).fetchall()

    assert fired, "the planted repeat did not fire at all"
    assert any(planted["phrase"] in (row["meta"] or "") for row in fired)

    # It fires on the THIRD, not the first: reaching backwards to score a
    # moment that had not happened yet is what firing earlier would mean.
    third = float(next(u for u in data["utterances"]
                       if u["seq"] == planted["seqs"][2])["t"])
    first = float(next(u for u in data["utterances"]
                       if u["seq"] == planted["seqs"][0])["t"])
    assert min(float(row["t"]) for row in fired) >= first
    assert any(abs(float(row["t"]) - third) < 30.0 for row in fired)


def test_the_planted_excitement_phrases_fire(authored):
    cfg, conn, data = authored
    ctx = StageContext(cfg=cfg, conn=conn, stream_id=str(data["stream_id"]),
                       log=lambda _m: None)
    phrases.run(ctx)

    count = conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = ? AND kind = 'phrase_excitement'",
        (str(data["stream_id"]),)).fetchone()[0]
    assert count >= len(data["planted"]["excitement_seqs"]) - 1


# --------------------------------------------------------------------------
# §12, exercised without a key
# --------------------------------------------------------------------------


def test_the_scripted_source_answers_in_order_and_records_the_prompts():
    source = fakes.ScriptedSource(replies=["first", "second"])
    assert source.available(None) == (True, "")
    assert source.complete(None, "prompt one") == "first"
    assert source.complete(None, "prompt two", schema={"type": "object"}) == "second"

    assert source.prompts == ["prompt one", "prompt two"]
    assert source.schemas == [None, {"type": "object"}]
    assert source.calls == 2


def test_calling_more_often_than_scripted_is_an_error_not_a_default():
    """"The code called the model more times than the test expected" is a
    finding — §12.4 budgets a whole stream's reasoning, and a map that ran
    twice per chapter would otherwise be invisible."""
    source = fakes.ScriptedSource(replies=["only one"])
    source.complete(None, "a")
    with pytest.raises(fakes.ExhaustedError):
        source.complete(None, "b")


def test_a_good_reply_survives_every_check(authored):
    _, conn, data = authored
    known = _segments(conn, str(data["stream_id"]))
    seq = int(data["utterances"][0]["seq"])

    result = validate_selections(
        parse_reply(fakes.reply([fakes.good(seq, known[seq])]))["selections"],
        known, id_field="seq", text_for=lambda text: text,
        noun="a segment of this stream")

    assert not result.dropped
    assert [s.key for s in result.selections] == [seq]
    assert result.invalid_id_rate == 0.0


def test_a_hallucinated_id_is_dropped_and_counted(authored):
    """§12.2. The rate is what §14 monitors, so the drop has to reach it."""
    _, conn, data = authored
    known = _segments(conn, str(data["stream_id"]))
    seq = int(data["utterances"][0]["seq"])

    result = validate_selections(
        parse_reply(fakes.reply([
            fakes.good(seq, known[seq]),
            fakes.hallucinated_id(known[seq]),
        ]))["selections"],
        known, id_field="seq", text_for=lambda text: text,
        noun="a segment of this stream")

    assert len(result.selections) == 1
    assert [reason for reason, _ in result.dropped] == ["unknown-id"]
    assert result.returned == 2
    assert result.invalid_id_rate == 0.5


def test_a_real_quote_from_the_wrong_segment_is_dropped(authored):
    """THE adversarial case. The words are genuinely in the transcript, so a
    check against the corpus accepts it — and §12.3 exists to catch a model
    answering about material it did not read."""
    _, conn, data = authored
    known = _segments(conn, str(data["stream_id"]))
    first, second = (int(u["seq"]) for u in data["utterances"][:2])

    result = validate_selections(
        parse_reply(fakes.reply([fakes.misquoted(first, known[second])]))["selections"],
        known, id_field="seq", text_for=lambda text: text,
        noun="a segment of this stream")

    assert not result.selections
    assert [reason for reason, _ in result.dropped] == ["bad-quote"]
    # Not an id problem, so §14's hallucination rate must stay clean.
    assert result.invalid_id_rate == 0.0


@pytest.mark.parametrize("body", [fakes.malformed(), fakes.truncated()])
def test_an_unparseable_reply_returns_none_rather_than_raising(body):
    """A validator that dies on bad input has stopped validating."""
    assert parse_reply(body) is None


def test_prose_around_the_json_is_tolerated(authored):
    """What comes out of a paste round trip has text wrapped around it, and
    asking the operator to trim that by hand is the friction that makes a tool
    go unused."""
    _, conn, data = authored
    known = _segments(conn, str(data["stream_id"]))
    seq = int(data["utterances"][0]["seq"])

    wrapped = fakes.reply([fakes.good(seq, known[seq])])
    assert not wrapped.lstrip().startswith("{")
    assert parse_reply(wrapped)["selections"][0]["seq"] == seq
