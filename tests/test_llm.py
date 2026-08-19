"""§12's rules, and §9.4's round trip driven adversarially.

The reply is the one thing in this system nobody controls, so the tests that
matter are the ones that hand it a bad reply. Four kinds, all of which a real
model produces sooner or later: an invented id, a real id belonging to something
else, a quote that was never said, and JSON buried in prose.

**The second is the interesting one.** A segment id from another chapter EXISTS,
so a plain existence check passes it — and the result is a citation attached to a
chapter whose transcript does not contain it, which is exactly the fabrication
§12.2 is for. It has to be scoped, not merely checked.
"""

from __future__ import annotations

import json

import pytest

from clipforge import config, db, llm
from clipforge.digest import build, prompt
from clipforge.digest import stage as digest_stage


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "t.db")
    connection.execute(
        "INSERT INTO streams (id, date, master_path, duration_s) "
        "VALUES ('s', '2026-01-01', '/m.mkv', 1200)"
    )
    yield connection
    connection.close()


#: Two chapters, six lines each, with a clean id split between them.
CHAPTER_A = ["we are going in through the side", "that went badly",
             "I will try the other route next time"]
CHAPTER_B = ["okay this is the tier list", "Namor is obviously top",
             "why is nobody agreeing with me"]


@pytest.fixture
def views(conn, cfg):
    for seq, text in enumerate(CHAPTER_A + CHAPTER_B):
        t = seq * 100.0
        conn.execute(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text) "
            "VALUES ('s', ?, ?, ?, ?)", (seq, t, t + 20.0, text))
    conn.execute(
        "INSERT INTO candidates (stream_id, generation, is_current, profile, "
        "t_start, t_end, t_peak, score_entertainment, score_gameplay, "
        "score_combined, feature_vector, feature_schema_version, config_version) "
        "VALUES ('s', 1, 1, 'entertainment', 100, 150, 120, 0, 0, 1, '{}', 1, 'x@y')")

    content = {
        "stream_id": "s", "date": "2026-01-01", "games": [], "duration_s": 1200.0,
        "chapters": [
            {"index": 0, "t_start": 0.0, "t_end": 300.0, "title": None,
             "summary": None, "notable_segment_ids": []},
            {"index": 1, "t_start": 300.0, "t_end": 1200.0, "title": None,
             "summary": None, "notable_segment_ids": []},
        ],
        "themes_observed": [], "open_loops": [], "top_candidates": [],
        "recurring_phrases": [], "emotional_arc": [],
        "segmentation": {"sources": [], "absent": {}},
    }
    return content, prompt.chapter_views(conn, "s", content)


def _reply(chapters, version=1):
    return {"digest_version": version, "chapters": chapters}


def _good(index, view):
    seq, text = view.lines[0]
    return {"index": index, "title": "a title", "summary": "a summary.",
            "themes": ["a theme"], "notable_segment_ids": [seq],
            "quote": text, "open_loops": []}


# --------------------------------------------------------------------------
# §12's shared machinery
# --------------------------------------------------------------------------


def test_the_last_json_object_wins_over_prose_around_it():
    """A chat window wraps its answer in prose; trimming it by hand is friction."""
    text = ('Sure! Here is my analysis.\n\n{"chapters": []}\n\n'
            'Let me know if you would like me to adjust anything.')
    assert llm.parse_reply(text) == {"chapters": []}


def test_a_reply_with_no_json_at_all_parses_to_none():
    assert llm.parse_reply("I'm afraid I can't help with that.") is None


def test_the_quote_check_forgives_rewrapping_but_not_rewording():
    """"Verbatim" is the requirement, so punctuation is deliberately significant.

    A model that re-wrapped a line has still quoted it; one that changed a word
    has not, and that is the fabrication the check exists to catch.
    """
    original = "Namor's turrets are melting me, holy shit."
    assert llm.normalise("Namor's turrets   are melting\nme, holy shit.") \
        in llm.normalise(original)
    assert llm.normalise("Namors turrets are melting me holy shit") \
        not in llm.normalise(original)


def test_only_unknown_ids_count_towards_the_hallucination_rate():
    """§14 monitors hallucination, not reply hygiene.

    A missing quote is a badly-behaved reply; an id that does not exist is the
    model inventing a handle. Folding both into one number would make the metric
    move for reasons that are not hallucination.
    """
    result = llm.Validation(returned=4)
    result.drop("unknown-id", "invented")
    result.drop("no-quote", "sloppy")
    assert result.invalid_id_rate == pytest.approx(0.25)


def test_the_rate_is_zero_when_the_model_returned_nothing():
    assert llm.Validation().invalid_id_rate == 0.0


def test_hooks_still_reaches_the_shared_helpers_under_its_own_names():
    """The §12 machinery moved to `clipforge/llm.py`; §8.5's callers did not."""
    from clipforge.render import hooks
    assert hooks.normalise is llm.normalise
    assert hooks.parse_reply is llm.parse_reply
    assert hooks.INVALID_ID_METRIC == llm.INVALID_ID_METRIC
    assert issubclass(hooks.Validated, llm.Validation)


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def test_the_prompt_carries_real_segment_ids_scoped_per_chapter(cfg, views):
    """§12.1, and the reason ids must be real database handles.

    §12.2 wants a hallucinated id to be detectable. If a prompt numbered six
    lines 1 to 6, any number a model invented in that range would look valid.
    """
    content, chapter_views = views
    text = prompt.build_prompt(cfg, content, 1, chapter_views)

    for seq, line in chapter_views[0].lines:
        assert f"[{seq}] {line}" in text
    # Each chapter states its own id range, which is what scopes the citations.
    low, high = min(chapter_views[1].seqs), max(chapter_views[1].seqs)
    assert f"segment ids {low}-{high}" in text


def test_the_prompt_states_the_digest_version_it_was_built_from(cfg, views):
    content, chapter_views = views
    assert "digest_version: 7" in prompt.build_prompt(cfg, content, 7,
                                                      chapter_views)


def test_every_chapter_gets_its_own_section_in_one_prompt(cfg, views):
    """Map-reduce structurally, one round trip in practice.

    C8 budgets ~35 minutes hands-on per stream and there is no API key, so the
    transport is a person pasting. `render/hooks.py` made the same call for §8.5.
    """
    content, chapter_views = views
    text = prompt.build_prompt(cfg, content, 1, chapter_views)
    for view in chapter_views:
        assert f"### chapter {view.index}" in text


def test_a_chapter_with_no_transcript_says_so_instead_of_being_empty(
        cfg, conn, views):
    """The default case today: Phase 2 ships off, so nothing is transcribed."""
    content, _ = views
    conn.execute("DELETE FROM segments WHERE stream_id = 's'")
    chapter_views = prompt.chapter_views(conn, "s", content)

    text = prompt.build_prompt(cfg, content, 1, chapter_views)

    assert "no transcript for this chapter" in text


# --------------------------------------------------------------------------
# the reply, adversarially
# --------------------------------------------------------------------------


def test_a_valid_reply_is_accepted_whole(views):
    content, chapter_views = views
    reply = _reply([_good(v.index, v) for v in chapter_views])

    result = prompt.validate(reply, chapter_views, version=1)

    assert len(result.chapters) == 2
    assert result.dropped == []
    assert result.invalid_id_rate == 0.0


def test_an_invented_chapter_index_is_dropped_and_counted(views):
    """§12.2: non-existent ids are dropped, and the drop is logged.

    The rate is a fraction of IDS, not of entries — §12.2 says "every returned
    ID is checked for existence". Here that is chapter 0's index plus its one
    notable id, both valid, and chapter 99's index, which is not: one bad out of
    three. Counting entries instead would put sub-entity drops over an entry
    denominator, and a reply whose only invented handle was one chapter index
    reported 1.00 — "the model hallucinated everything" — when two of three
    chapters were usable.
    """
    content, chapter_views = views
    reply = _reply([_good(0, chapter_views[0]),
                    {**_good(1, chapter_views[1]), "index": 99}])

    result = prompt.validate(reply, chapter_views, version=1)

    assert [c.index for c in result.chapters] == [0]
    assert any(reason == "unknown-id" for reason, _ in result.dropped)
    assert result.returned == 3
    assert result.invalid_id_rate == pytest.approx(1 / 3, abs=1e-3)


def test_a_bad_id_inside_an_accepted_chapter_does_not_condemn_the_whole_reply(
        views):
    """The bug this unit fixed, pinned.

    One accepted chapter that mis-cites two segment ids is a partly-good reply,
    not a total hallucination — and the rate has to say so, or §14's monitoring
    cannot distinguish "the prompt has stopped being clear" from "one id slipped".
    """
    content, chapter_views = views
    stray = chapter_views[1].lines[0][0]
    good_seq = chapter_views[0].lines[0][0]
    reply = _reply([{**_good(0, chapter_views[0]),
                     "notable_segment_ids": [good_seq, stray]}])

    result = prompt.validate(reply, chapter_views, version=1)

    # The chapter survives; only the stray citation is dropped.
    assert [c.index for c in result.chapters] == [0]
    assert result.chapters[0].notable_segment_ids == [good_seq]
    # index + two notable ids = 3 ids, one bad.
    assert result.returned == 3
    assert result.invalid_id_rate == pytest.approx(1 / 3, abs=1e-3)


def test_a_real_segment_id_from_the_wrong_chapter_is_dropped(views):
    """The check a plain existence test would pass.

    The id exists, so `seq in all_segments` says yes — and the citation lands on
    a chapter whose transcript does not contain it. Scoped, not merely checked.
    """
    content, chapter_views = views
    other = chapter_views[1].lines[0][0]
    assert other not in chapter_views[0].seqs

    reply = _reply([{**_good(0, chapter_views[0]),
                     "notable_segment_ids": [other]}])

    result = prompt.validate(reply, chapter_views, version=1)

    assert result.chapters[0].notable_segment_ids == []
    assert any(reason == "unknown-id" and "not in this chapter" in detail
               for reason, detail in result.dropped)


def test_a_quote_that_is_not_in_the_chapter_discards_that_chapter(views):
    """§12.3 is the machine-checkable guard against answering about unread text."""
    content, chapter_views = views
    reply = _reply([{**_good(0, chapter_views[0]),
                     "quote": "a line nobody ever said"}])

    result = prompt.validate(reply, chapter_views, version=1)

    assert result.chapters == []
    assert any(reason == "bad-quote" for reason, _ in result.dropped)
    # A bad quote is not hallucinating an id, so it must not move §14's rate.
    assert result.invalid_id_rate == 0.0


def test_a_quote_from_the_wrong_chapter_is_rejected(views):
    """Verbatim in the STREAM is not verbatim in the chapter being described."""
    content, chapter_views = views
    reply = _reply([{**_good(0, chapter_views[0]),
                     "quote": chapter_views[1].lines[0][1]}])

    result = prompt.validate(reply, chapter_views, version=1)

    assert result.chapters == []
    assert any(reason == "bad-quote" for reason, _ in result.dropped)


def test_json_buried_in_prose_is_still_applied(views):
    """What a chat window actually returns."""
    content, chapter_views = views
    body = json.dumps(_reply([_good(0, chapter_views[0])]))
    text = f"Certainly! Here's the digest.\n\n```json\n{body}\n```\n\nHope that helps!"

    result = prompt.validate(llm.parse_reply(text), chapter_views, version=1)

    assert len(result.chapters) == 1


def test_a_reply_for_a_different_digest_version_is_refused_outright(views):
    """Not a drop — a refusal. There is no way to detect it from the reply.

    Segment ids are scoped per chapter, so if the segmenter re-ran between
    prompt and reply every citation now points into the wrong chapter, and each
    one would still pass every other check.
    """
    content, chapter_views = views
    reply = _reply([_good(0, chapter_views[0])], version=1)

    with pytest.raises(prompt.DigestPromptError, match="digest v1"):
        prompt.validate(reply, chapter_views, version=2)


def test_the_same_chapter_returned_twice_is_only_taken_once(views):
    content, chapter_views = views
    reply = _reply([_good(0, chapter_views[0]), _good(0, chapter_views[0])])

    result = prompt.validate(reply, chapter_views, version=1)

    assert [c.index for c in result.chapters] == [0]
    assert any(reason == "duplicate" for reason, _ in result.dropped)


def test_an_open_loop_with_an_unrecognised_kind_is_kept_without_one(views):
    """§3.2 documents three kinds. Losing a real finding to a bad label is the
    wrong trade — the text is the finding."""
    content, chapter_views = views
    seq = chapter_views[0].lines[0][0]
    reply = _reply([{**_good(0, chapter_views[0]), "open_loops": [
        {"text": "try the other route", "kind": "resolution", "segment_id": seq}]}])

    result = prompt.validate(reply, chapter_views, version=1)

    loop = result.chapters[0].open_loops[0]
    assert loop["text"] == "try the other route"
    assert loop["kind"] is None
    assert loop["segment_id"] == seq


def test_an_open_loop_citing_another_chapter_keeps_the_text_and_drops_the_id(
        views):
    content, chapter_views = views
    other = chapter_views[1].lines[0][0]
    reply = _reply([{**_good(0, chapter_views[0]), "open_loops": [
        {"text": "a real finding", "kind": "promise", "segment_id": other}]}])

    result = prompt.validate(reply, chapter_views, version=1)

    loop = result.chapters[0].open_loops[0]
    assert loop["text"] == "a real finding"
    assert loop["segment_id"] is None
    assert any(reason == "unknown-id" for reason, _ in result.dropped)


# --------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------


def test_reduce_fills_the_model_fields_and_resolves_times_from_the_database(
        conn, views):
    """§12.1: the model returns ids; every timestamp is resolved here."""
    content, chapter_views = views
    seq = chapter_views[0].lines[0][0]
    reply = _reply([{**_good(0, chapter_views[0]), "open_loops": [
        {"text": "try the other route", "kind": "promise", "segment_id": seq}]}])
    result = prompt.validate(reply, chapter_views, version=1)

    merged = prompt.reduce_into(content, result, conn, "s")

    assert merged["chapters"][0]["title"] == "a title"
    assert merged["themes_observed"] == ["a theme"]
    expected = conn.execute(
        "SELECT t_start FROM segments WHERE stream_id='s' AND seq=?", (seq,)
    ).fetchone()["t_start"]
    assert merged["open_loops"][0]["t"] == pytest.approx(expected)


def test_reduce_does_not_mutate_the_stored_digest(conn, views):
    """v1 must survive whatever a model said about it."""
    content, chapter_views = views
    before = json.dumps(content, sort_keys=True)
    result = prompt.validate(_reply([_good(0, chapter_views[0])]),
                             chapter_views, version=1)

    prompt.reduce_into(content, result, conn, "s")

    assert json.dumps(content, sort_keys=True) == before


def test_a_chapter_the_model_skipped_keeps_its_deterministic_fields(conn, views):
    content, chapter_views = views
    result = prompt.validate(_reply([_good(0, chapter_views[0])]),
                             chapter_views, version=1)

    merged = prompt.reduce_into(content, result, conn, "s")

    assert merged["chapters"][1]["title"] is None
    assert merged["chapters"][1]["t_end"] == content["chapters"][1]["t_end"]


def test_re_applying_a_reply_does_not_duplicate_open_loops(conn, views):
    """`open_loops` has no natural key, so a second attempt would accumulate.

    §11.3 tracks these across streams, where a triplicated loop reads as three
    independent findings.
    """
    loops = [{"text": "try the other route", "kind": "promise",
              "segment_id": 0, "t": 0.0}]
    with db.transaction(conn):
        prompt.store_open_loops(conn, "s", loops)
    with db.transaction(conn):
        prompt.store_open_loops(conn, "s", loops)

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM open_loops WHERE stream_id='s'").fetchone()
    assert rows["n"] == 1


def test_a_loop_the_operator_resolved_is_not_reopened_by_a_re_run(conn, views):
    """That status is theirs, not this pass's."""
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO open_loops (stream_id, text, kind, status) "
            "VALUES ('s', 'already dealt with', 'promise', 'resolved')")
    with db.transaction(conn):
        prompt.store_open_loops(conn, "s", [])

    row = conn.execute(
        "SELECT status FROM open_loops WHERE stream_id='s'").fetchone()
    assert row["status"] == "resolved"


# --------------------------------------------------------------------------
# versioning
# --------------------------------------------------------------------------


def test_applying_a_reply_adds_a_version_and_leaves_the_first_intact(
        conn, cfg, views):
    """§9.1: "first-class rows, never regenerable cache." """
    content, chapter_views = views
    deterministic = build.Digest(content=content,
                                 markdown=build.render_markdown(content))
    with db.transaction(conn):
        build.write(conn, "s", deterministic, model_used=None)

    result = prompt.validate(_reply([_good(0, chapter_views[0])]),
                             chapter_views, version=1)
    merged = prompt.reduce_into(content, result, conn, "s")
    with db.transaction(conn):
        build.write(conn, "s", build.Digest(
            content=merged, markdown=build.render_markdown(merged)),
            model_used="manual:pasted")

    rows = conn.execute(
        "SELECT version, content, model_used FROM digests WHERE stream_id='s' "
        "ORDER BY version").fetchall()
    assert [r["version"] for r in rows] == [1, 2]
    assert json.loads(rows[0]["content"])["chapters"][0]["title"] is None
    assert json.loads(rows[1]["content"])["chapters"][0]["title"] == "a title"


def test_the_prompt_file_is_config_not_a_string_literal(cfg):
    """§17 and the operator's instruction: prompts live in config files.

    They will be rewritten once there are digests of streams worth remembering,
    and that must not be a code change.
    """
    prompts = prompt.load_prompts(cfg)
    for key in ("intro", "rules", "fields", "chapters_header", "no_transcript"):
        assert prompts[key].strip()


def test_a_prompt_file_missing_a_section_fails_loudly(cfg, tmp_path):
    bad = tmp_path / "prompts.yaml"
    bad.write_text("intro: hello\n", encoding="utf-8")
    broken = config.load(overrides=[f"digest.prompts_file={bad.as_posix()}"])
    with pytest.raises(prompt.DigestPromptError, match="missing"):
        prompt.load_prompts(broken)
