"""§8.3's inputs: which words are in a clip, when, and who said them.

The interesting cases are all about *provenance*. §8.3 wants a colour per
speaker, and the only field that cannot lie about who produced a word is
`segments.track` — `segments.speaker` is an energy comparison over a span and
says something different (see `render/words.py`).
"""

from __future__ import annotations

import json

import pytest

from clipforge import config, db
from clipforge.render import timeline, words
from clipforge.render.words import MIC, PARTY

TRACK_MAP = json.dumps({
    "roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3},
    "warnings": [], "source": "metadata tags", "titles": {},
})


def make_words(*specs):
    """`("hi", 1.0, 1.4)` -> the §3.2 word dict, `None` times allowed."""
    return [{"w": text, "start": start, "end": end, "score": 0.9}
            for text, start, end in specs]


@pytest.fixture
def conn(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    connection.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, "
        "audio_track_map) VALUES ('s', '2026-08-13', 'x.mkv', 'vod', ?)",
        (TRACK_MAP,),
    )
    connection.commit()
    yield connection
    connection.close()


def add_segment(conn, seq, t_start, t_end, text, track, speaker, word_list):
    conn.execute(
        "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, "
        "track, words) VALUES ('s', ?, ?, ?, ?, ?, ?, ?)",
        (seq, t_start, t_end, text, speaker, track, json.dumps(word_list)),
    )
    conn.commit()


# --------------------------------------------------------------------------
# track -> role: the finding that motivates this module
# --------------------------------------------------------------------------


def test_track_roles_inverts_the_probed_map():
    assert words.track_roles(TRACK_MAP) == {1: MIC, 3: PARTY}


def test_track_roles_ignores_tracks_that_cannot_identify_a_speaker():
    """`game` has no speech and `mixed` has both people in it."""
    roles = words.track_roles(TRACK_MAP)
    assert 0 not in roles and 2 not in roles


@pytest.mark.parametrize("raw", [None, "", "not json", "[]"])
def test_track_roles_survives_an_unusable_map(raw):
    assert words.track_roles(raw) == {}


def test_provenance_beats_the_energy_verdict():
    """The case §8.3 gets wrong.

    `speakers.classify` compares mic and party energy over a segment's span, so
    a segment TRANSCRIBED FROM THE PARTY TRACK can be labelled 'operator' when
    the operator happened to be louder across it. Colouring by `speaker` paints
    the party's words in the operator's colour; colouring by `track` cannot.
    """
    roles = words.track_roles(TRACK_MAP)
    assert words.role_for(3, "operator", roles) == PARTY
    assert words.role_for(1, "party", roles) == MIC


def test_speaker_is_the_fallback_when_there_is_no_track():
    roles = words.track_roles(TRACK_MAP)
    assert words.role_for(None, "operator", roles) == MIC
    assert words.role_for(None, "party", roles) == PARTY
    assert words.role_for(None, "both", roles) == ""
    assert words.role_for(None, None, roles) == ""


def test_a_both_segment_resolves_to_its_own_track():
    """§8.3's third rule, made implementable.

    'both' means both people had energy — it does not say whose words these
    are. Two overlapping segments both labelled 'both' come from two different
    tracks, so they alternate by source track exactly as §8.3 asks.
    """
    roles = words.track_roles(TRACK_MAP)
    assert words.role_for(1, "both", roles) == MIC
    assert words.role_for(3, "both", roles) == PARTY


# --------------------------------------------------------------------------
# unaligned words
# --------------------------------------------------------------------------


def test_unaligned_words_keep_their_text_and_get_times():
    """§5.7 stores null times rather than inventing them. Captions cannot show
    a hole, so the gap is divided — and the guess is flagged."""
    resolved = words.interpolate(
        make_words(("one", 0.0, 1.0), ("two", None, None), ("three", None, None),
                   ("four", 3.0, 4.0)),
        0.0, 4.0,
    )
    assert [w["w"] for w in resolved] == ["one", "two", "three", "four"]
    assert resolved[1]["start"] == pytest.approx(1.0)
    assert resolved[1]["end"] == pytest.approx(2.0)
    assert resolved[2]["start"] == pytest.approx(2.0)
    assert resolved[2]["end"] == pytest.approx(3.0)
    assert resolved[1]["interpolated"] and resolved[2]["interpolated"]
    assert not resolved[0].get("interpolated")


def test_interpolation_falls_back_to_the_segment_bounds_at_the_edges():
    resolved = words.interpolate(
        make_words(("lead", None, None), ("real", 2.0, 3.0), ("tail", None, None)),
        1.0, 5.0,
    )
    assert resolved[0]["start"] == pytest.approx(1.0)
    assert resolved[0]["end"] == pytest.approx(2.0)
    assert resolved[2]["start"] == pytest.approx(3.0)
    assert resolved[2]["end"] == pytest.approx(5.0)


def test_a_non_positive_gap_yields_zero_length_words_not_negative_ones():
    """Forced alignment does produce an unaligned word between two that touch."""
    resolved = words.interpolate(
        make_words(("a", 0.0, 1.0), ("b", None, None), ("c", 1.0, 2.0)), 0.0, 2.0
    )
    assert resolved[1]["start"] == pytest.approx(1.0)
    assert resolved[1]["end"] >= resolved[1]["start"]


def test_a_fully_aligned_segment_is_untouched():
    raw = make_words(("a", 0.0, 1.0), ("b", 1.0, 2.0))
    assert words.interpolate(raw, 0.0, 2.0) == raw


# --------------------------------------------------------------------------
# loading against a window
# --------------------------------------------------------------------------


def test_words_come_back_on_the_output_clock(conn):
    """A1 puts `-ss` before `-i`, so the output restarts at zero. A caption
    written in stream time lands hours away and ffmpeg still exits 0."""
    add_segment(conn, 0, 100.0, 102.0, "hello there", 1, "operator",
                make_words(("hello", 100.0, 100.5), ("there", 101.0, 101.4)))
    plan = timeline.build(99.0, 105.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert [w.text for w in found] == ["hello", "there"]
    assert found.words[0].start == pytest.approx(1.0)
    assert found.words[1].start == pytest.approx(2.0)
    assert max(w.end for w in found) < plan.duration


def test_words_outside_the_window_are_dropped_and_counted(conn):
    add_segment(conn, 0, 10.0, 20.0, "in out", 1, "operator",
                make_words(("in", 12.0, 12.4), ("out", 19.0, 19.4)))
    plan = timeline.build(11.0, 13.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert [w.text for w in found] == ["in"]
    assert found.dropped == 1


def test_two_tracks_speaking_at_once_come_back_interleaved_and_split(conn):
    """The overlap case. Both segments are kept (§5.8's deviation) and each
    keeps its own role, so the caption layer can stack them."""
    add_segment(conn, 0, 0.0, 2.0, "mine here", 1, "both",
                make_words(("mine", 0.0, 0.4), ("here", 1.0, 1.4)))
    add_segment(conn, 1, 0.2, 2.2, "yours too", 3, "both",
                make_words(("yours", 0.2, 0.6), ("too", 1.2, 1.6)))
    plan = timeline.build(0.0, 3.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert [w.text for w in found] == ["mine", "yours", "here", "too"]
    streams = words.by_role(found.words)
    assert [w.text for w in streams[MIC]] == ["mine", "here"]
    assert [w.text for w in streams[PARTY]] == ["yours", "too"]


def test_interpolated_words_are_counted_on_the_way_out(conn):
    add_segment(conn, 0, 0.0, 2.0, "a b c", 1, "operator",
                make_words(("a", 0.0, 0.5), ("b", None, None), ("c", 1.5, 2.0)))
    plan = timeline.build(0.0, 3.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert found.interpolated == 1
    assert [w.interpolated for w in found] == [False, True, False]


def test_a_segment_with_no_track_falls_back_and_is_counted(conn):
    add_segment(conn, 0, 0.0, 1.0, "solo", None, "operator",
                make_words(("solo", 0.0, 0.5)))
    plan = timeline.build(0.0, 2.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert found.words[0].role == MIC
    assert found.unroled == 0


def test_a_segment_with_neither_track_nor_speaker_is_unroled(conn):
    add_segment(conn, 0, 0.0, 1.0, "solo", None, None,
                make_words(("solo", 0.0, 0.5)))
    plan = timeline.build(0.0, 2.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert found.words[0].role == ""
    assert found.unroled == 1


def test_unreadable_words_json_names_the_segment(conn):
    conn.execute(
        "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, "
        "track, words) VALUES ('s', 7, 0.0, 1.0, 'x', 'operator', 1, '{oops')"
    )
    conn.commit()

    with pytest.raises(words.WordError, match="segment 7"):
        words.load(conn, "s", timeline.build(0.0, 2.0), TRACK_MAP)


def test_coverage_is_reported_not_enforced(conn):
    add_segment(conn, 0, 0.0, 10.0, "one", 1, "operator",
                make_words(("one", 0.0, 1.0)))
    plan = timeline.build(0.0, 10.0)

    found = words.load(conn, "s", plan, TRACK_MAP)

    assert words.coverage(found, plan) == pytest.approx(0.1)


# --------------------------------------------------------------------------
# against the fixture's authored ground truth
# --------------------------------------------------------------------------


def test_fixture_utterances_colour_by_their_authored_track(conn, speech_manifest):
    """Every authored line, labelled from the manifest's own speaker->track map.

    The manifest says speaker A is on `mic` and B on `party`. Nothing here
    writes down which colour that is — the assertion is that the role the
    loader resolves matches the track the fixture authored the line onto.
    """
    expected_role = speech_manifest["speaker_tracks"]
    roles = words.track_roles(TRACK_MAP)
    track_index = json.loads(TRACK_MAP)["roles"]

    for utterance in speech_manifest["utterances"]:
        track = track_index[utterance["track"]]
        # Deliberately pass the WRONG speaker for half of them: provenance has
        # to win, which is the whole point of reading `track`.
        misleading = "party" if utterance["speaker"] == "A" else "operator"
        assert words.role_for(track, misleading, roles) == \
            expected_role[utterance["speaker"]]


def test_the_fixtures_overlap_produces_two_roles(conn, speech_manifest):
    """52.6-54.36 s in the fixture: both people talking. Two source tracks, so
    two roles, so two stacked caption rows rather than one interleaved line."""
    overlap_start, overlap_end = speech_manifest["overlap_windows"][0]
    track_index = json.loads(TRACK_MAP)["roles"]

    speaking = [
        u for u in speech_manifest["utterances"]
        if u["t_end"] > overlap_start and u["t_start"] < overlap_end
    ]
    assert len(speaking) == 2, "the manifest's overlap should have two utterances"

    for seq, utterance in enumerate(speaking):
        add_segment(
            conn, seq, utterance["t_start"], utterance["t_end"], utterance["text"],
            track_index[utterance["track"]], "both",
            make_words(*[
                (word, utterance["t_start"] + index * 0.2,
                 utterance["t_start"] + index * 0.2 + 0.15)
                for index, word in enumerate(utterance["text"].split())
            ]),
        )

    plan = timeline.build(overlap_start, overlap_end)
    found = words.load(conn, "s", plan, TRACK_MAP)

    assert set(words.by_role(found.words)) == {MIC, PARTY}
