"""§8.3's caption file.

Three of these guard things that are silently wrong when broken: A3's BGR order
(every colour still parses, they are just the wrong colour), the gap §8.3's
pseudocode leaves between words (the caption strobes), and `PlayRes*` being the
output resolution rather than the master's (every font is the wrong size).
"""

from __future__ import annotations

import json
import re

import pytest

from clipforge import config
from clipforge.render import ass
from clipforge.render.ass import DEFAULT_STYLE, Style
from clipforge.render.words import MIC, PARTY, RenderWord

DIALOGUE = re.compile(
    r"^Dialogue: 0,(?P<start>[\d:.]+),(?P<end>[\d:.]+),(?P<style>[^,]*),"
    r",0,0,0,,(?P<text>.*)$"
)


def word(text, start, end, role=MIC, track=1):
    return RenderWord(text=text, start=start, end=end, role=role, track=track,
                      speaker=None, segment_seq=0)


def run_of(count, *, role=MIC, track=1, step=0.5, length=0.3, first=0.0):
    return [
        word(f"w{index}", first + index * step, first + index * step + length,
             role=role, track=track)
        for index in range(count)
    ]


def styles(**overrides):
    table = {
        DEFAULT_STYLE: Style(name=DEFAULT_STYLE, colour="#FFFFFF", margin_v=220),
        MIC: Style(name=MIC, colour="#FFFFFF", margin_v=220),
        PARTY: Style(name=PARTY, colour="#7FE7FF", margin_v=330),
    }
    table.update(overrides)
    return table


def events_of(script):
    return [DIALOGUE.match(line).groupdict()
            for line in script.render().splitlines()
            if line.startswith("Dialogue:")]


def centiseconds(stamp: str) -> int:
    hours, minutes, rest = stamp.split(":")
    seconds, centis = rest.split(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 100 + int(centis)


# --------------------------------------------------------------------------
# A3 — colour is BGR
# --------------------------------------------------------------------------


def test_colour_is_bgr_not_rgb():
    """The asymmetric case. A swapped implementation still produces a valid
    ASS colour, so a symmetric test value (#FFFFFF, #808080) proves nothing."""
    assert ass.ass_colour("#FF8000") == "&H0080FF&"
    assert ass.ass_colour("#123456") == "&H563412&"


def test_colour_accepts_the_hash_or_not_and_normalises_case():
    assert ass.ass_colour("ff8000") == ass.ass_colour("#FF8000")


def test_style_colour_carries_alpha_and_alpha_zero_is_opaque():
    """ASS's alpha is transparency — the opposite of every other alpha here."""
    assert ass.style_colour("#FF8000") == "&H000080FF"
    assert ass.style_colour("#FF8000", 128) == "&H800080FF"


def test_a_bad_colour_says_what_the_format_is():
    with pytest.raises(ass.AssError, match="RRGGBB"):
        ass.ass_colour("red")
    with pytest.raises(ass.AssError):
        ass.ass_colour("#FFF")


def test_alpha_is_range_checked():
    with pytest.raises(ass.AssError, match="0-255"):
        ass.style_colour("#FFFFFF", 300)


# --------------------------------------------------------------------------
# text and time
# --------------------------------------------------------------------------


def test_braces_are_escaped_so_they_cannot_open_an_override_block():
    """An unbalanced `{` swallows the rest of the line."""
    assert ass.escape("a {b} c") == r"a \{b\} c"


def test_newlines_become_spaces_rather_than_ending_the_event():
    assert ass.escape("a\nb\r\nc") == "a b c"


def test_a_backslash_before_n_cannot_become_a_line_break():
    """`\\N` is ASS's hard line break. A zero-width space separates the pair, so
    the text still reads correctly and the escape cannot fire."""
    rendered = ass.escape("path\\name")

    assert rendered == "path\\​name"
    assert "\\n" not in rendered


def test_escaping_leaves_ordinary_text_alone():
    assert ass.escape("Hawkeye just hit that shot") == \
        "Hawkeye just hit that shot"


def test_timestamps_are_ass_centiseconds():
    assert ass.timestamp(0.0) == "0:00:00.00"
    assert ass.timestamp(1.23) == "0:00:01.23"
    assert ass.timestamp(61.5) == "0:01:01.50"
    assert ass.timestamp(3661.0) == "1:01:01.00"


def test_negative_time_clamps_rather_than_wrapping():
    assert ass.timestamp(-1.0) == "0:00:00.00"


# --------------------------------------------------------------------------
# grouping — §8.3's "3-5 words on screen"
# --------------------------------------------------------------------------


def test_groups_are_the_configured_size():
    groups = ass.group_words(run_of(8), size=4, minimum=3)

    assert [len(group) for group in groups] == [4, 4]


def test_a_one_word_tail_is_rebalanced_away():
    """§8.3 asks for 3-5 words. Greedy chunking of 9 at size 4 leaves 4/4/1,
    and a single word alone on screen for a beat reads as a bug."""
    groups = ass.group_words(run_of(9), size=4, minimum=3)

    assert [len(group) for group in groups] == [4, 5]
    assert all(3 <= len(group) <= 5 for group in groups)
    assert sum(len(group) for group in groups) == 9


def test_rebalancing_moves_words_back_when_there_are_enough():
    """7 words at size 3 greedily gives 3/3/1; the tail borrows instead of
    merging, because merging would put 4 on the last row and 3 on the others."""
    groups = ass.group_words(run_of(7), size=3, minimum=2)

    assert [len(group) for group in groups] == [3, 2, 2]


def test_too_few_words_for_two_groups_become_one():
    groups = ass.group_words(run_of(5), size=4, minimum=3)

    assert [len(group) for group in groups] == [5]


def test_a_long_silence_ends_a_group():
    """Otherwise the word before a four-second pause shares the screen with the
    word after it, and the highlight sits on a word nobody is saying."""
    words = [word("a", 0.0, 0.3), word("b", 0.4, 0.7), word("c", 9.0, 9.3)]

    groups = ass.group_words(words, size=4, minimum=1, max_gap_s=1.2)

    assert [[w.text for w in group] for group in groups] == [["a", "b"], ["c"]]


def test_a_sentence_boundary_ends_a_group():
    words = [word("done.", 0.0, 0.3), word("next", 0.4, 0.7)]

    groups = ass.group_words(words, size=4, minimum=1)

    assert [[w.text for w in group] for group in groups] == [["done."], ["next"]]


def test_trailing_punctuation_does_not_hide_the_sentence_end():
    words = [word('"insane!"', 0.0, 0.3), word("next", 0.4, 0.7)]

    groups = ass.group_words(words, size=4, minimum=1)

    assert len(groups) == 2


def test_sentence_breaking_can_be_turned_off():
    words = [word("done.", 0.0, 0.3), word("next", 0.4, 0.7)]

    groups = ass.group_words(words, size=4, minimum=1, break_on_sentence_end=False)

    assert len(groups) == 1


def test_group_size_zero_is_refused():
    with pytest.raises(ass.AssError, match="group_size"):
        ass.group_words(run_of(3), size=0)


# --------------------------------------------------------------------------
# the finding: §8.3's pseudocode leaves the screen blank between words
# --------------------------------------------------------------------------


def test_lines_within_a_group_are_contiguous():
    """§8.3 sets each line to the active word's own start/end, so between every
    pair of words nothing is on screen and the caption strobes at the word
    rate. Lines must meet exactly instead."""
    script = ass.build(run_of(4, step=1.0, length=0.3), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=10.0,
                       group_size=4, min_group_size=1, tail_hold_s=0.35)

    events = events_of(script)
    assert len(events) == 4
    for earlier, later in zip(events, events[1:], strict=False):
        assert centiseconds(earlier["end"]) == centiseconds(later["start"])


def test_the_group_holds_past_the_last_word():
    script = ass.build(run_of(3, step=1.0, length=0.3), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=10.0,
                       group_size=3, min_group_size=1, tail_hold_s=0.5)

    last = events_of(script)[-1]
    # last word starts at 2.0 and ends at 2.3; the hold takes it to 2.8.
    assert centiseconds(last["end"]) == 280


def test_a_group_never_runs_into_the_next_one():
    script = ass.build(run_of(6, step=0.4, length=0.2), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=10.0,
                       group_size=3, min_group_size=1, tail_hold_s=5.0)

    events = events_of(script)
    for earlier, later in zip(events, events[1:], strict=False):
        assert centiseconds(earlier["end"]) <= centiseconds(later["start"])


def test_captions_never_outlast_the_clip():
    script = ass.build(run_of(3, step=1.0, length=0.4), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=2.5,
                       group_size=3, min_group_size=1, tail_hold_s=5.0)

    for event in events_of(script):
        assert centiseconds(event["end"]) <= 250


def test_a_degenerate_highlight_state_is_skipped_not_flashed():
    """Two words sharing a start — what interpolation produces. The word is
    still on screen as part of its group; it just never gets the highlight."""
    words = [word("a", 0.0, 0.5), word("b", 0.5, 0.5), word("c", 0.5, 1.0)]

    script = ass.build(words, width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=5.0, group_size=3,
                       min_group_size=1, min_line_s=0.08)

    assert script.skipped == 1
    assert all("b" in event["text"] for event in events_of(script))


# --------------------------------------------------------------------------
# highlighting and speaker colour
# --------------------------------------------------------------------------


def test_exactly_one_word_is_highlighted_per_line():
    script = ass.build(run_of(3, step=1.0), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=10.0,
                       group_size=3, min_group_size=1)

    highlight = ass.override("#FFD400")
    for index, event in enumerate(events_of(script)):
        assert event["text"].count(highlight) == 1
        assert f"{highlight}w{index}" in event["text"]


def test_the_override_is_the_statement_form_not_the_bare_value():
    """A3 names the value `&HBBGGRR&`; §8.3 wraps it as `{\\c&HBBGGRR&}`.
    Emitting the bare value renders it as visible text in the caption."""
    assert ass.override("#FF8000") == "{\\c&H0080FF&}"


def test_the_highlight_is_closed_so_later_words_do_not_inherit_it():
    script = ass.build(run_of(3, step=1.0), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=10.0,
                       group_size=3, min_group_size=1)

    base = ass.override(styles()[MIC].colour)
    first = events_of(script)[0]["text"]
    assert first.index(ass.override("#FFD400")) < first.index(base)


def test_every_word_of_the_group_is_on_screen_in_every_line():
    script = ass.build(run_of(4, step=1.0), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=10.0,
                       group_size=4, min_group_size=1)

    for event in events_of(script):
        for index in range(4):
            assert f"w{index}" in event["text"]


def test_two_speakers_get_two_styles_at_different_heights():
    """§8.3's differentiator, and the thing that stops two simultaneous
    speakers being drawn on top of each other."""
    words = run_of(3, role=MIC, track=1, step=1.0) + \
        run_of(3, role=PARTY, track=3, step=1.0, first=0.2)

    script = ass.build(words, width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=10.0, group_size=3,
                       min_group_size=1)

    used = {event["style"] for event in events_of(script)}
    assert used == {MIC, PARTY}
    assert script.styles[MIC].margin_v != script.styles[PARTY].margin_v


def test_a_group_never_mixes_two_speakers_words():
    """Interleaving two people into one line reads as gibberish."""
    words = run_of(3, role=MIC, track=1, step=1.0) + \
        run_of(3, role=PARTY, track=3, step=1.0, first=0.2)
    for index, item in enumerate(words):
        words[index] = RenderWord(
            text=f"{item.role}{index}", start=item.start, end=item.end,
            role=item.role, track=item.track, speaker=None, segment_seq=0)

    script = ass.build(words, width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=10.0, group_size=3,
                       min_group_size=1)

    for event in events_of(script):
        other = PARTY if event["style"] == MIC else MIC
        assert other not in event["text"]


def test_an_unroled_word_takes_the_default_style():
    script = ass.build([word("solo", 0.0, 0.4, role="", track=None)],
                       width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=5.0, min_group_size=1)

    assert events_of(script)[0]["style"] == DEFAULT_STYLE


def test_a_missing_default_style_is_refused():
    table = styles()
    del table[DEFAULT_STYLE]

    with pytest.raises(ass.AssError, match=DEFAULT_STYLE):
        ass.build(run_of(2), width=1080, height=1920, styles=table,
                  highlight="#FFD400", duration=5.0)


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------


def test_playres_is_the_output_resolution_not_the_masters():
    """libass scales every font and margin by PlayRes*. §8.4's whole point is
    that the output is a different shape from the source."""
    script = ass.build(run_of(2), width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=5.0, min_group_size=1)

    rendered = script.render()
    assert "PlayResX: 1080" in rendered
    assert "PlayResY: 1920" in rendered


def test_the_style_table_declares_every_style_in_the_format_order():
    script = ass.build(run_of(2), width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=5.0, min_group_size=1)

    rendered = script.render()
    assert f"Format: {ass.STYLE_FORMAT}" in rendered
    for name in (DEFAULT_STYLE, MIC, PARTY):
        assert re.search(rf"^Style: {name},", rendered, re.M)


def test_style_field_count_matches_the_declared_format():
    """Positional fields: one extra comma and every colour after it is read as
    something else."""
    expected = len(ass.STYLE_FORMAT.split(","))

    line = styles()[MIC].to_line()

    assert len(line.removeprefix("Style: ").split(",")) == expected


def test_events_are_ordered_by_time_across_speakers():
    words = run_of(3, role=PARTY, track=3, step=1.0, first=0.5) + \
        run_of(3, role=MIC, track=1, step=1.0)

    script = ass.build(words, width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=10.0, group_size=3,
                       min_group_size=1)

    starts = [centiseconds(event["start"]) for event in events_of(script)]
    assert starts == sorted(starts)


def test_no_words_produces_a_valid_empty_script():
    script = ass.build([], width=1080, height=1920, styles=styles(),
                       highlight="#FFD400", duration=5.0)

    rendered = script.render()
    assert "[Events]" in rendered
    assert script.lines == 0


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_styles_come_from_config_and_share_the_common_settings():
    cfg = config.load()

    table = ass.styles_from_config(cfg, roles=(MIC, PARTY))

    assert set(table) >= {MIC, PARTY, DEFAULT_STYLE}
    assert table[MIC].font == cfg.get("render.captions.font")
    assert table[MIC].font_size == cfg.get("render.captions.font_size")


def test_config_stacks_the_two_speakers_by_default():
    """If these ever match, two people talking at once collide on one row."""
    table = ass.styles_from_config(config.load(), roles=(MIC, PARTY))

    assert table[MIC].margin_v != table[PARTY].margin_v


def test_config_gives_the_two_speakers_different_colours():
    table = ass.styles_from_config(config.load(), roles=(MIC, PARTY))

    assert table[MIC].colour != table[PARTY].colour


def test_build_from_config_reads_every_knob():
    cfg = config.load(overrides=["render.captions.group_size=2",
                                 "render.captions.uppercase=true"])

    script = ass.build_from_config(cfg, run_of(4, step=1.0), width=1080,
                                   height=1920, duration=10.0, roles=(MIC,))

    assert all("W0" in e["text"] or "W2" in e["text"] for e in events_of(script))


def test_the_configured_highlight_is_what_appears():
    cfg = config.load()

    script = ass.build_from_config(cfg, run_of(2, step=1.0), width=1080,
                                   height=1920, duration=10.0, roles=(MIC,))

    expected = ass.override(cfg.get("render.captions.highlight"))
    assert all(expected in event["text"] for event in events_of(script))


# --------------------------------------------------------------------------
# the fixture's authored dialogue
# --------------------------------------------------------------------------


def test_the_fixtures_overlap_stacks_rather_than_collides(speech_manifest):
    """52.6-54.36 s: both people talking. The manifest says who is on which
    track; nothing here writes down a colour."""
    overlap_start, overlap_end = speech_manifest["overlap_windows"][0]
    speaking = [u for u in speech_manifest["utterances"]
                if u["t_end"] > overlap_start and u["t_start"] < overlap_end]
    role_of = {"mic": MIC, "party": PARTY}

    words = []
    for utterance in speaking:
        role = role_of[utterance["track"]]
        spoken = utterance["text"].split()
        step = (utterance["t_end"] - utterance["t_start"]) / max(len(spoken), 1)
        for index, text in enumerate(spoken):
            start = utterance["t_start"] - overlap_start + index * step
            words.append(RenderWord(
                text=text, start=max(start, 0.0), end=max(start + step * 0.8, 0.0),
                role=role, track=1 if role == MIC else 3,
                speaker="both", segment_seq=utterance["index"]))

    cfg = config.load()
    script = ass.build_from_config(
        cfg, sorted(words, key=lambda w: w.start), width=1080, height=1920,
        duration=overlap_end - overlap_start, roles=(MIC, PARTY))

    events = events_of(script)
    assert {event["style"] for event in events} == {MIC, PARTY}
    assert script.styles[MIC].margin_v != script.styles[PARTY].margin_v


def test_every_fixture_line_survives_grouping(speech_manifest):
    """No authored word is lost between the transcript and the caption file."""
    role_of = {"mic": MIC, "party": PARTY}
    utterance = speech_manifest["utterances"][0]
    spoken = utterance["text"].split()
    step = (utterance["t_end"] - utterance["t_start"]) / len(spoken)

    words = [
        RenderWord(text=text, start=index * step, end=index * step + step * 0.8,
                   role=role_of[utterance["track"]], track=1, speaker="operator",
                   segment_seq=0)
        for index, text in enumerate(spoken)
    ]
    groups = ass.group_words(words, size=4, minimum=3)

    assert [w.text for group in groups for w in group] == spoken


def test_the_vocabulary_terms_survive_escaping(speech_manifest):
    """Hero names carry apostrophes ("Namor's"), which must not be mangled."""
    for term in speech_manifest["vocabulary_terms"]:
        assert ass.escape(term) == term
    assert ass.escape("Namor's") == "Namor's"


def test_json_of_the_manifest_is_not_needed_for_colour(speech_manifest):
    """Guard against the test above quietly depending on a colour constant."""
    assert json.dumps(speech_manifest["speaker_tracks"])


# --------------------------------------------------------------------------
# does libass actually accept it?
# --------------------------------------------------------------------------


def burn(path, needs_ffmpeg=None):
    """Render the ASS file over a black source. Returns ffmpeg's stderr."""
    from clipforge import ffmpeg

    value, working_dir = ffmpeg.filter_file(path)
    proc = ffmpeg.run(
        [
            ffmpeg.require("ffmpeg"), "-hide_banner", "-nostdin",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30:d=4",
            # `ass` after `scale` is the order §8.3 uses; there is no scale
            # here, but the filter still has to accept the file.
            "-vf", f"ass={value}",
            "-frames:v", "120", "-f", "null", "-",
        ],
        cwd=working_dir,
    )
    return proc.stderr or ""


def test_libass_parses_the_generated_file(tmp_path, needs_ffmpeg):
    """Everything above checks what we wrote. This checks what reads it.

    An unbalanced override, a style row with the wrong field count, or a
    malformed colour all produce a file that still looks like ASS. libass
    reports them on stderr and then renders the clip WITHOUT the broken part —
    ffmpeg exits 0 either way, which is precisely why this has to be asserted
    rather than eyeballed.
    """
    words = run_of(4, role=MIC, track=1, step=0.5) + \
        run_of(4, role=PARTY, track=3, step=0.5, first=0.25)
    cfg = config.load()
    script = ass.build_from_config(cfg, sorted(words, key=lambda w: w.start),
                                   width=1080, height=1920, duration=4.0,
                                   roles=(MIC, PARTY))
    # Text that would break a naive escaper, to prove the escaping is load
    # bearing rather than decorative.
    script.events.append((0, "Dialogue: 0,0:00:00.00,0:00:01.00,mic,,0,0,0,,"
                             + ass.escape("brace { and } and Namor's")))

    path = tmp_path / "captions.ass"
    path.write_text(script.render(), encoding="utf-8")

    stderr = burn(path)

    assert "Error" not in stderr, stderr
    assert "Unable to parse" not in stderr, stderr


def test_a_filter_path_survives_an_apostrophe_in_a_parent_directory(
    tmp_path, needs_ffmpeg
):
    """`C:\\Users\\O'Brien` is an ordinary Windows home directory.

    MEASURED: no escaping of an absolute path survives the filter parser — not
    quoting, not `\\'`, not shell-style `'\\''`. Running from the file's own
    directory and naming it bare is the only form that works, which is what
    `ffmpeg.filter_file` returns.
    """
    awkward = tmp_path / "o'brien" / "with,comma"
    awkward.mkdir(parents=True)
    path = awkward / "captions.ass"
    script = ass.build(run_of(3, step=0.5), width=1080, height=1920,
                       styles=styles(), highlight="#FFD400", duration=4.0,
                       min_group_size=1)
    path.write_text(script.render(), encoding="utf-8")

    stderr = burn(path)

    assert "Error" not in stderr, stderr


def test_filter_file_never_returns_a_drive_letter(tmp_path):
    """The colon is the whole problem: ffmpeg reads it as an option separator
    and splits `C:/x/y.ass` into two options."""
    from clipforge import ffmpeg

    value, working_dir = ffmpeg.filter_file(tmp_path / "captions.ass")

    assert value == "captions.ass"
    assert ":" not in value
    assert working_dir == str(tmp_path)
