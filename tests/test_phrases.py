"""§5.6 phrase detection and §5.4.1's transcript-derived signals.

The speech fixture's script was written for this: it carries four of §5.6's own
phrases, one profanity, and one line repeated three times inside 90 s. Every
expectation below is read from `manifest.json` or `phrases.yaml`, never typed in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from clipforge import config, db, signals
from clipforge.extract import phrases, speakers, transcript
from clipforge.extract.transcript import Segment, Word
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext


@pytest.fixture(scope="module")
def phrase_list():
    return phrases.load_phrases(config.load())


@pytest.fixture
def detected(tmp_path, speech_fixture_dir, speech_manifest):
    """The speech fixture, transcribed from its manifest, labelled, searched.

    Word timestamps are synthesised evenly across each authored utterance —
    the fixture knows exactly when each line starts and ends, so the words fall
    inside their own line, which is all speech_rate needs.
    """
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
            out = []
            for u in speech_manifest["utterances"]:
                if u["track"] != role:
                    continue
                words = u["text"].split()
                span = (u["t_end"] - u["t_start"]) / max(len(words), 1)
                out.append(Segment(
                    u["t_start"], u["t_end"], u["text"], track, role,
                    [Word(w, round(u["t_start"] + i * span, 3),
                          round(u["t_start"] + (i + 1) * span, 3), 0.9)
                     for i, w in enumerate(words)],
                ))
            return out

    transcript.run(ctx, transcriber=FromManifest())
    speakers.run(ctx)
    phrases.run(ctx)

    yield cfg, conn, ctx
    conn.close()


def events_of(conn, stream_id, kind=None):
    sql = "SELECT * FROM events WHERE stream_id = ? AND source = ?"
    args = [stream_id, phrases.SOURCE]
    if kind:
        sql += " AND kind = ?"
        args.append(kind)
    return conn.execute(sql + " ORDER BY t", args).fetchall()


# --------------------------------------------------------------------------
# normalisation — the rule §5.4.2 needs and never gives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Let's go!", "lets go"),
    ("lets go", "lets go"),
    ("LET'S   GO.", "lets go"),
    ("Oh my god,", "oh my god"),
])
def test_normalisation_makes_variants_one_phrase(text, expected):
    """"Let's go!", "lets go" and "Let's go." are one bit."""
    assert phrases.normalise(text) == expected


# --------------------------------------------------------------------------
# matching (§5.6)
# --------------------------------------------------------------------------


def test_matching_respects_word_boundaries():
    """§5.6 says only "match against the transcript". A substring search fires
    "no way" inside "I know ways around it", and every one of those is a
    candidate the operator rejects by hand."""
    patterns = phrases.compile_patterns(["no way"])
    assert patterns[0][1].search("No way, that was insane")
    assert not patterns[0][1].search("I know ways around it")


def test_matching_is_case_insensitive():
    patterns = phrases.compile_patterns(["oh my god"])
    assert patterns[0][1].search("OH MY GOD")


def test_apostrophes_are_optional():
    """So "let's" and "lets" are one config entry rather than two."""
    patterns = phrases.compile_patterns(["let's go"])
    assert patterns[0][1].search("lets go")
    assert patterns[0][1].search("let's go")


def test_the_spec_list_is_what_ships(phrase_list):
    """§5.6 gives the list verbatim; it should be recognisable in config."""
    normalised = {phrases.normalise(p) for p in phrase_list.excitement}
    assert {"oh my god", "no way", "did you see that", "lets go"} <= normalised


# --------------------------------------------------------------------------
# events, against the fixture's authored script
# --------------------------------------------------------------------------


def test_the_scripts_phrases_are_all_found(detected, speech_manifest, phrase_list):
    _cfg, conn, ctx = detected
    found = {json.loads(e["meta"])["phrase"]
             for e in events_of(conn, ctx.stream_id, phrases.KIND_EXCITEMENT)}

    spoken = phrases.normalise(" ".join(u["text"] for u in speech_manifest["utterances"]))
    expected = {
        phrases.normalise(p) for p in phrase_list.excitement
        if phrases.compile_patterns([p])[0][1].search(spoken)
    }
    assert expected, "the fixture should contain some of §5.6's phrases"
    assert expected <= found


def test_the_manifests_passive_phrases_are_detected(detected, speech_manifest):
    """The fixture declares which of §5.6's phrases it planted."""
    _cfg, conn, ctx = detected
    found = {json.loads(e["meta"])["phrase"]
             for e in events_of(conn, ctx.stream_id, phrases.KIND_EXCITEMENT)}
    planted = {phrases.normalise(p) for p in speech_manifest["passive_phrases"]}
    assert len(planted & found) >= 3


def test_events_land_inside_their_utterance(detected, speech_manifest):
    _cfg, conn, ctx = detected
    windows = [(u["t_start"], u["t_end"]) for u in speech_manifest["utterances"]]
    for event in events_of(conn, ctx.stream_id):
        assert any(start <= event["t"] <= end for start, end in windows)


def test_events_carry_the_speaker(detected):
    """§5.4.2 weights laugh_party above laugh_operator because someone ELSE
    reacting is external validation. The same argument applies to phrases, so
    Phase 3 can split the weight without re-extracting."""
    _cfg, conn, ctx = detected
    speakers_seen = {
        json.loads(e["meta"])["speaker"] for e in events_of(conn, ctx.stream_id)
    }
    assert speakers_seen <= {speakers.OPERATOR, speakers.PARTY, speakers.BOTH,
                             speakers.UNKNOWN}
    assert speakers.PARTY in speakers_seen, "the party says one of §5.6's phrases"


def test_running_twice_replaces_rather_than_doubles(detected):
    _cfg, conn, ctx = detected
    before = len(events_of(conn, ctx.stream_id))
    phrases.run(ctx)
    assert len(events_of(conn, ctx.stream_id)) == before


def test_marker_events_are_untouched(detected):
    """The replace is scoped to source='phrase'."""
    _cfg, conn, ctx = detected
    markers = conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = ? AND source = 'marker'",
        (ctx.stream_id,),
    ).fetchone()[0]
    phrases.run(ctx)
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = ? AND source = 'marker'",
        (ctx.stream_id,),
    ).fetchone()[0] == markers


# --------------------------------------------------------------------------
# phrase_repeat (§5.4.2)
# --------------------------------------------------------------------------


def rows_for(texts, spacing=10.0):
    return [
        {"seq": i, "t_start": i * spacing, "t_end": i * spacing + 2.0,
         "text": text, "speaker": "operator", "words": "[]"}
        for i, text in enumerate(texts)
    ]


def test_a_repeat_fires_on_the_third_occurrence(phrase_list):
    """Firing on the first would mean reaching backwards to score a moment that
    had not happened yet."""
    events, _ = phrases.find_repeats(
        rows_for(["clutch hawkeye play"] * 3), phrase_list,
        window_s=90.0, minimum=3, low=2, high=6,
    )
    assert events
    assert min(e["t"] for e in events) == 20.0      # the third row


def test_two_occurrences_are_not_a_bit(phrase_list):
    events, _ = phrases.find_repeats(
        rows_for(["clutch hawkeye play"] * 2), phrase_list,
        window_s=90.0, minimum=3, low=2, high=6,
    )
    assert events == []


def test_occurrences_outside_the_window_do_not_count(phrase_list):
    """§5.4.2's 90 s is what separates a bit from saying it twice a stream."""
    events, _ = phrases.find_repeats(
        rows_for(["clutch hawkeye play"] * 3, spacing=60.0), phrase_list,
        window_s=90.0, minimum=3, low=2, high=6,
    )
    assert events == []


def test_stopword_ngrams_are_filtered(phrase_list):
    """§11.2's real answer is is_baseline_tic measured from ten streams. This
    list stands in until the corpus exists."""
    events, _ = phrases.find_repeats(
        rows_for(["and then it was"] * 4), phrase_list,
        window_s=90.0, minimum=3, low=2, high=6,
    )
    assert events == []


def test_configured_tics_are_filtered(phrase_list):
    events, _ = phrases.find_repeats(
        rows_for(["you know"] * 4), phrase_list,
        window_s=90.0, minimum=3, low=2, high=6,
    )
    assert events == []


def test_the_fixtures_repeated_line_is_detected(detected, speech_manifest):
    _cfg, conn, ctx = detected
    repeated = phrases.normalise(speech_manifest["repeated_phrase"])
    found = {json.loads(e["meta"])["phrase"]
             for e in events_of(conn, ctx.stream_id, phrases.KIND_REPEAT)}
    assert any(phrase in repeated for phrase in found), found


# --------------------------------------------------------------------------
# the continuous signals (§5.4.1, §5.6)
# --------------------------------------------------------------------------


def test_both_signals_are_stored(detected):
    _cfg, conn, ctx = detected
    present = set(signals.kinds(conn, ctx.stream_id))
    assert {phrases.SPEECH_RATE, phrases.SWEAR_DENSITY} <= present


def test_speech_rate_is_zero_across_the_silence(detected, speech_manifest):
    """A4's silence, seen from the other side: nobody is talking, so the rate
    §5.4.1 uses to detect dead air should say so."""
    _cfg, conn, ctx = detected
    series = signals.load(conn, ctx.stream_id, phrases.SPEECH_RATE)
    start, end = max(speech_manifest["silence_windows"], key=lambda w: w[1] - w[0])
    # Inside the window, clear of the sliding window's overhang at each edge.
    quiet = series.slice(start + 2.0, end - 2.0)
    assert quiet.size
    assert float(np.max(quiet)) == 0.0


def test_speech_rate_is_positive_while_someone_talks(detected, speech_manifest):
    _cfg, conn, ctx = detected
    series = signals.load(conn, ctx.stream_id, phrases.SPEECH_RATE)
    for entry in speech_manifest["utterances"][:4]:
        middle = (entry["t_start"] + entry["t_end"]) / 2
        assert series.value_at(middle) > 0.0, entry["label"]


def test_speech_rate_is_plausible(detected, speech_manifest):
    """Conversational speech is 2-4 words/sec; the fixture is authored speech,
    so the peak should land in that neighbourhood rather than an order out."""
    _cfg, conn, ctx = detected
    series = signals.load(conn, ctx.stream_id, phrases.SPEECH_RATE)
    assert 1.0 < float(np.max(series.values)) < 8.0


def test_swear_density_peaks_where_the_script_swears(detected, speech_manifest,
                                                    phrase_list):
    """§5.6's continuous profanity signal."""
    _cfg, conn, ctx = detected
    series = signals.load(conn, ctx.stream_id, phrases.SWEAR_DENSITY)

    patterns = phrases.compile_patterns(phrase_list.profanity)
    sworn = [
        u for u in speech_manifest["utterances"]
        if any(p.search(u["text"]) for _n, p in patterns)
    ]
    assert sworn, "the fixture should contain one profanity"
    for entry in sworn:
        middle = (entry["t_start"] + entry["t_end"]) / 2
        assert series.value_at(middle) > 0.0, entry["label"]


def test_swear_density_is_zero_where_nobody_swears(detected, speech_manifest):
    _cfg, conn, ctx = detected
    series = signals.load(conn, ctx.stream_id, phrases.SWEAR_DENSITY)
    start, end = max(speech_manifest["silence_windows"], key=lambda w: w[1] - w[0])
    assert float(np.max(series.slice(start + 2.0, end - 2.0))) == 0.0


def test_the_signals_are_at_the_configured_rate(detected):
    _cfg, conn, ctx = detected
    expected = float(config.load().get("extract.signal_hz"))
    for kind in (phrases.SPEECH_RATE, phrases.SWEAR_DENSITY):
        assert signals.load(conn, ctx.stream_id, kind).sample_rate_hz == expected


def test_unaligned_words_do_not_count_toward_speech_rate():
    """They have no timestamp, so they belong to no window — which biases the
    rate DOWN wherever alignment struggled, and §5.4.1 reads a low rate as dead
    air."""
    rows = [{"words": json.dumps([
        {"w": "a", "start": 1.0, "end": 1.2, "score": 0.9},
        {"w": "b", "start": None, "end": None, "score": None},
    ])}]
    centres, aligned, total = phrases.aligned_words(rows)
    assert (aligned, total) == (1, 2)
    assert centres == [pytest.approx(1.1)]


# --------------------------------------------------------------------------
# stage plumbing
# --------------------------------------------------------------------------


def test_verify_wants_both_signals(detected):
    _cfg, conn, ctx = detected
    assert phrases.verify(ctx)[0]
    conn.execute(
        "DELETE FROM signal_series WHERE stream_id = ? AND kind = ?",
        (ctx.stream_id, phrases.SPEECH_RATE),
    )
    ok, why = phrases.verify(ctx)
    assert not ok and phrases.SPEECH_RATE in why


def test_it_refuses_without_a_transcript(detected):
    _cfg, conn, ctx = detected
    conn.execute("DELETE FROM segments WHERE stream_id = ?", (ctx.stream_id,))
    with pytest.raises(phrases.PhraseError, match="no segments"):
        phrases.run(ctx)


def test_the_stage_is_registered():
    from clipforge.pipeline import stages

    spec = stages.get("phrase_detect")
    assert spec.module == "clipforge.extract.phrases"
    # §5.4.2 weights a party reaction above the operator's own, so the events
    # carry a speaker — which means the labels must exist first.
    assert "speaker_assign" in spec.requires


def test_only_the_longest_repeated_phrase_fires(phrase_list):
    """MEASURED. A repeated four-word line fires every n-gram inside it, so
    "Let's go, Hawkeye" produced `lets go`, `go hawkeye` AND `lets go hawkeye`
    at the same instant. Non-marker event kernels combine with `sum` (§6.2 step
    4), so a longer catchphrase would score higher purely for containing more
    n-grams — length as a signal, which it is not."""
    events, _ = phrases.find_repeats(
        rows_for(["lets go hawkeye"] * 3), phrase_list,
        window_s=90.0, minimum=3, low=2, high=6,
    )
    at_third = [e for e in events if e["t"] == 20.0]
    assert len(at_third) == 1
    assert json.loads(json.dumps(at_third[0]["meta"]))["phrase"] == "lets go hawkeye"


def test_containment_is_by_words_not_substring():
    """"go" is not inside "going"."""
    assert phrases._contains("lets go hawkeye", "lets go")
    assert not phrases._contains("going home", "go")


def test_unrelated_repeats_at_the_same_moment_both_fire():
    """Suppression is containment, not "one event per segment"."""
    kept = phrases.maximal({"lets go": 3, "nice shot": 3, "go": 3})
    assert set(kept) == {"lets go", "nice shot"}
