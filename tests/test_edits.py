"""§8.6's profanity muting and §8.2's filler removal.

Both are off by default, so the first thing asserted is that nothing changed.
After that, most of these test **refusals** — the planner's job is not removing
words, it is removing only the ones that can be removed without wrecking the
clip, and every constraint is a thing §8.2's one line does not mention.
"""

from __future__ import annotations

import json

import pytest

from clipforge import config, db, ffmpeg, paths
from clipforge.render import clips, crop, edits, timeline
from clipforge.render.edits import Span
from clipforge.render.timeline import Cut

TRACK_MAP = json.dumps({
    "roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3},
    "warnings": [], "source": "metadata tags", "titles": {},
})

ON = ["render.mute.enabled=true", "render.filler.enabled=true"]


def span(text, start, end, role="mic", track=1):
    return Span(start=start, end=end, text=text, role=role, track=track)


@pytest.fixture
def conn(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    db.open_db(cfg.db_path).close()
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    connection.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, "
        "audio_track_map) VALUES ('s', '2026-08-14', 'm.mkv', 'vod', ?)",
        (TRACK_MAP,),
    )
    connection.commit()
    yield connection
    connection.close()


def add_segment(conn, seq, t_start, t_end, words, track=1, speaker="operator"):
    conn.execute(
        "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, "
        "track, words) VALUES ('s', ?, ?, ?, ?, ?, ?, ?)",
        (seq, t_start, t_end, " ".join(w for w, _s, _e in words), speaker, track,
         json.dumps([{"w": w, "start": s, "end": e, "score": 0.9}
                     for w, s, e in words])),
    )
    conn.commit()


# --------------------------------------------------------------------------
# off by default — §16 lists profanity muting among the rejected features
# --------------------------------------------------------------------------


def test_both_toggles_are_off_by_default():
    cfg = config.load()

    assert cfg.get("render.mute.enabled") is False
    assert cfg.get("render.filler.enabled") is False


def test_nothing_happens_by_default(conn):
    add_segment(conn, 0, 1.0, 3.0, [("holy", 1.0, 1.4), ("shit", 1.5, 1.9),
                                    ("um", 2.0, 2.3)])
    cfg = config.load()
    plan = timeline.build(0.0, 5.0)

    assert edits.mute_ranges(cfg, conn, "s", plan, TRACK_MAP) == []
    assert edits.filler_cuts(cfg, conn, "s", plan, TRACK_MAP).cuts == []
    assert clips.audio_filters(cfg, plan) == ""


def test_a_clip_with_no_cuts_emits_no_trim_chain():
    """Load bearing: a default render must produce byte-identical commands to
    the one before this module existed. An empty string is what makes that
    true, rather than a trim spanning the whole window that would merely
    probably be equivalent."""
    plan = timeline.build(10.0, 20.0)

    assert edits.trim_chain(plan, audio=True) == ""
    assert edits.trim_chain(plan, audio=False, entry="[0:v]", label="vcut") == ""


def test_the_video_graph_is_unchanged_without_cuts():
    cfg = config.load()
    spec = crop.get_template(cfg)

    assert crop.filter_complex(spec, trim="") == crop.filter_complex(spec)


# --------------------------------------------------------------------------
# §8.6 — muting
# --------------------------------------------------------------------------


def test_profanity_is_muted_from_the_shipped_list(conn):
    """The list already exists for §5.6's swear_density. One list, one place."""
    add_segment(conn, 0, 1.0, 3.0, [("holy", 1.0, 1.4), ("shit", 1.5, 1.9)])
    cfg = config.load(overrides=ON)

    ranges = edits.mute_ranges(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert len(ranges) == 1
    start, end = ranges[0]
    assert start < 1.5 and end > 1.9      # the word, plus padding


def test_the_range_is_padded_outwards(conn):
    """C2, and MEASURED: the boundary leaks, so a mute that starts late leaves
    the consonant audible and defeats the whole point."""
    add_segment(conn, 0, 1.0, 3.0, [("shit", 2.0, 2.4)])
    pad = 0.15
    cfg = config.load(overrides=[*ON, f"render.mute.pad_s={pad}"])

    (start, end), = edits.mute_ranges(
        cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert start == pytest.approx(2.0 - pad)
    assert end == pytest.approx(2.4 + pad)


def test_padding_is_clamped_to_the_clip(conn):
    add_segment(conn, 0, 0.0, 1.0, [("shit", 0.0, 0.2)])
    cfg = config.load(overrides=[*ON, "render.mute.pad_s=5"])

    (start, end), = edits.mute_ranges(
        cfg, conn, "s", timeline.build(0.0, 2.0), TRACK_MAP)

    assert start == 0.0
    assert end <= 2.0


def test_only_whole_words_match(conn):
    """A substring rule would mute a syllable out of an innocent word."""
    add_segment(conn, 0, 1.0, 3.0, [("shitake", 1.0, 1.5), ("mushrooms", 1.6, 2.0)])
    cfg = config.load(overrides=ON)

    assert edits.mute_ranges(
        cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP) == []


def test_the_word_list_can_be_overridden(conn):
    add_segment(conn, 0, 1.0, 3.0, [("banana", 1.0, 1.4)])
    cfg = config.load(overrides=[*ON, "render.mute.words=[banana]"])

    assert len(edits.mute_ranges(
        cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)) == 1


def test_ranges_are_in_output_time_after_a_cut(conn):
    """The mute has to land where the word ENDED UP, not where it was."""
    add_segment(conn, 0, 0.0, 10.0, [("shit", 8.0, 8.4)])
    cfg = config.load(overrides=ON)
    plan = timeline.build(0.0, 10.0, [Cut(2.0, 5.0)])   # 3 s removed before it

    (start, _end), = edits.mute_ranges(cfg, conn, "s", plan, TRACK_MAP)

    assert start == pytest.approx(8.0 - 3.0 - 0.08)


def test_a_word_inside_a_cut_is_not_muted(conn):
    """It is gone; muting it would silence whatever moved into its place."""
    add_segment(conn, 0, 0.0, 10.0, [("shit", 3.0, 3.4)])
    cfg = config.load(overrides=ON)

    assert edits.mute_ranges(
        cfg, conn, "s", timeline.build(0.0, 10.0, [Cut(2.0, 5.0)]), TRACK_MAP) == []


def test_overlapping_ranges_merge():
    merged = edits.merge_ranges([(1.0, 2.0), (1.5, 3.0), (5.0, 6.0)])

    assert merged == [(1.0, 3.0), (5.0, 6.0)]


def test_one_filter_covers_every_range():
    """`+` is logical OR in ffmpeg's expression language, so several ranges do
    not need several filters."""
    text = edits.mute_filter([(1.0, 2.0), (5.0, 6.0)])

    # One filter, not one per range. (`volume` appears twice in the string --
    # once as the filter name, once as its option.)
    assert text.count("volume=enable") == 1
    assert "between(t,1.000,2.000)+between(t,5.000,6.000)" in text
    assert text.endswith(":volume=0")


def test_no_ranges_means_no_filter():
    assert edits.mute_filter([]) == ""


# --------------------------------------------------------------------------
# §8.2 — filler removal, which is mostly refusals
# --------------------------------------------------------------------------


def test_filler_on_the_configured_track_is_cut(conn):
    add_segment(conn, 0, 0.0, 5.0, [("so", 1.0, 1.2), ("um", 2.0, 2.4),
                                    ("yes", 3.0, 3.3)])
    cfg = config.load(overrides=ON)

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert len(plan.cuts) == 1
    assert plan.cuts[0].t_start == pytest.approx(2.0)
    assert plan.removed_s == pytest.approx(0.4)


def test_filler_on_another_track_is_refused(conn):
    """Cutting the other person's speech because the operator said 'um' is not
    a thing anyone wants."""
    add_segment(conn, 0, 0.0, 5.0, [("um", 2.0, 2.4)], track=3, speaker="party")
    cfg = config.load(overrides=ON)

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert plan.cuts == []
    assert any("party" in note for note in plan.refused)


def test_a_cut_overlapping_another_speaker_is_refused(conn):
    """It would take a syllable out of someone who did not say the filler."""
    add_segment(conn, 0, 0.0, 5.0, [("um", 2.0, 2.4)])
    add_segment(conn, 1, 0.0, 5.0, [("hello", 2.1, 2.9)], track=3, speaker="party")
    cfg = config.load(overrides=ON)

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert plan.cuts == []
    assert any("overlaps" in note for note in plan.refused)


def test_a_cut_shorter_than_the_minimum_is_refused(conn):
    add_segment(conn, 0, 0.0, 5.0, [("um", 2.0, 2.05)])
    cfg = config.load(overrides=[*ON, "render.filler.min_duration_s=0.12"])

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert plan.cuts == []
    assert any("glitch" in note for note in plan.refused)


def test_nearby_cuts_merge(conn):
    """Two removals 200 ms apart leave an island of audio that reads as a
    stutter rather than an edit."""
    add_segment(conn, 0, 0.0, 5.0, [("um", 2.0, 2.3), ("uh", 2.5, 2.8)])
    cfg = config.load(overrides=[*ON, "render.filler.merge_gap_s=0.35"])

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert len(plan.cuts) == 1
    assert plan.cuts[0].t_start == pytest.approx(2.0)
    assert plan.cuts[0].t_end == pytest.approx(2.8)


def test_distant_cuts_do_not_merge(conn):
    add_segment(conn, 0, 0.0, 8.0, [("um", 2.0, 2.3), ("uh", 6.0, 6.3)])
    cfg = config.load(overrides=ON)

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 8.0), TRACK_MAP)

    assert len(plan.cuts) == 2


def test_an_implausible_plan_is_refused_whole(conn):
    """Not 'this clip was full of filler' -- the word list is matching
    something it should not, and silently deleting a third of an approved
    moment is the failure worth stopping for."""
    words = [(f"um", 0.5 + i * 0.5, 0.9 + i * 0.5) for i in range(8)]
    add_segment(conn, 0, 0.0, 5.0, words)
    cfg = config.load(overrides=[*ON, "render.filler.max_share=0.25"])

    plan = edits.filler_cuts(cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP)

    assert plan.cuts == []
    assert any("word list problem" in note for note in plan.refused)


def test_unaligned_words_are_never_cut(conn):
    """Cutting on an invented timestamp would remove audio at a time nobody
    measured. The caption path interpolates; this one does not."""
    conn.execute(
        "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, "
        "track, words) VALUES ('s', 9, 0.0, 5.0, 'um', 'operator', 1, ?)",
        (json.dumps([{"w": "um", "start": None, "end": None, "score": None}]),),
    )
    conn.commit()
    cfg = config.load(overrides=ON)

    assert edits.filler_cuts(
        cfg, conn, "s", timeline.build(0.0, 5.0), TRACK_MAP).cuts == []


# --------------------------------------------------------------------------
# the filter graphs
# --------------------------------------------------------------------------


def test_the_trim_chain_is_clip_relative():
    """`-ss` before `-i` means the decoder already starts at clip_start, so a
    source-time trim would cut the wrong place."""
    plan = timeline.build(100.0, 110.0, [Cut(102.0, 103.0)])

    chain = edits.trim_chain(plan, audio=True)

    assert "atrim=0.000:2.000" in chain
    assert "atrim=3.000:10.000" in chain
    assert "100" not in chain


def test_the_audio_chain_cuts_before_it_mutes():
    """The mute ranges are in output time, which is the clock the cuts create.
    Muting first would place them against a timeline that no longer exists."""
    plan = timeline.build(0.0, 10.0, [Cut(2.0, 3.0)])

    chain = edits.audio_chain(plan, [(5.0, 5.5)])

    assert chain.index("concat=") < chain.index("volume=")


def test_the_video_and_audio_chains_keep_the_same_ranges():
    """They must, or the clip drifts out of sync."""
    plan = timeline.build(0.0, 10.0, [Cut(4.0, 5.0)])

    video = edits.trim_chain(plan, audio=False, entry="[0:v]", label="vcut")
    audio = edits.trim_chain(plan, audio=True)

    assert video.replace("[0:v]", "").replace("[vcut]", "").replace("setpts", "X") \
        == audio.replace("asplit", "split").replace("atrim", "trim") \
               .replace("asetpts", "X").replace("v=0:a=1", "v=1:a=0")


def test_the_video_chain_labels_do_not_collide_with_the_crop_graph():
    """Two graphs sharing a pad name is a silent mis-wire."""
    cfg = config.load()
    plan = timeline.build(0.0, 10.0, [Cut(4.0, 5.0)])

    graph = crop.filter_complex(
        crop.get_template(cfg),
        trim=edits.trim_chain(plan, audio=False, entry="[0:v]", label="vcut"),
    )

    pads = [p for p in graph.replace("[", " ").replace("]", " ").split()
            if p.startswith(("k", "r_s"))]
    assert len(pads) == len(set(pads)) or all(pads.count(p) == 2 for p in set(pads))
    assert "[vcut]" in graph


# --------------------------------------------------------------------------
# against the fixture's authored dialogue
# --------------------------------------------------------------------------


def test_the_fixtures_swear_word_is_found(conn, speech_manifest):
    """Utterance 4 is "Namor's turrets are melting me, holy shit." — testable
    as shipped, because the profanity list already covers it."""
    utterance = next(u for u in speech_manifest["utterances"]
                     if "shit" in u["text"].lower())
    spoken = utterance["text"].split()
    step = (utterance["t_end"] - utterance["t_start"]) / len(spoken)
    add_segment(conn, 0, utterance["t_start"], utterance["t_end"], [
        (word, utterance["t_start"] + i * step,
         utterance["t_start"] + i * step + step * 0.8)
        for i, word in enumerate(spoken)
    ], track=3, speaker="party")

    cfg = config.load(overrides=ON)
    plan = timeline.build(utterance["t_start"] - 1, utterance["t_end"] + 1)
    ranges = edits.mute_ranges(cfg, conn, "s", plan, TRACK_MAP)

    assert len(ranges) == 1
    # The last word of the line, so the range sits near its end.
    (start, end), = ranges
    assert start > (utterance["t_end"] - utterance["t_start"]) * 0.7


def test_a_word_the_fixture_actually_says_can_be_cut(conn, speech_manifest):
    """The fixture has no "um"/"uh", so filler removal is exercised against a
    word the authored dialogue does contain — and the expected duration is read
    from the same segment rows the planner reads."""
    utterance = next(u for u in speech_manifest["utterances"]
                     if "actually" in u["text"].lower())
    spoken = utterance["text"].split()
    step = (utterance["t_end"] - utterance["t_start"]) / len(spoken)
    rows = [(word, utterance["t_start"] + i * step,
             utterance["t_start"] + i * step + step * 0.8)
            for i, word in enumerate(spoken)]
    add_segment(conn, 0, utterance["t_start"], utterance["t_end"], rows)

    cfg = config.load(overrides=[
        "render.filler.enabled=true", "render.filler.words=[actually]",
        "render.filler.min_duration_s=0.01",
    ])
    plan = timeline.build(utterance["t_start"] - 1, utterance["t_end"] + 1)
    cut_plan = edits.filler_cuts(cfg, conn, "s", plan, TRACK_MAP)

    expected = sum(e - s for w, s, e in rows if "actually" in w.lower())
    assert cut_plan.removed_s == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_a_cut_clip_comes_out_shorter_and_in_sync(
    tmp_path, speech_fixture_dir, speech_manifest, needs_ffmpeg
):
    """Video and audio must agree, or the clip drifts."""
    master = speech_fixture_dir / speech_manifest["video"]
    plan = timeline.build(8.0, 18.0, [Cut(11.0, 12.0)])
    out = tmp_path / "cut.mp4"

    graph = crop.filter_complex(
        crop.resolve(crop.get_template(config.load()), 640, 360),
        trim=edits.trim_chain(plan, audio=False, entry="[0:v]", label="vcut"),
    )
    ffmpeg.run(clips.build_command(
        ffmpeg.require("ffmpeg"), master, out, cfg=config.load(overrides=[
            "render.video.preset=ultrafast", "render.video.crf=30"]),
        plan=plan, graph=graph, audio_index=0,
        audio_chain=edits.audio_chain(plan, []),
    ))

    found = clips.inspect(out, ffmpeg.require("ffprobe"))
    assert found.duration_s == pytest.approx(plan.duration, abs=0.15)
    assert found.duration_s < plan.source_duration
