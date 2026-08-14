"""§8.5's hook options, and §12's rules about how a model may answer.

Nothing here reaches a network. The interesting tests are the **refusals** —
§12 exists because a model that answers confidently about a clip it never read
is the failure mode, and the only defences against it are an id that has to
exist and a quote that has to be present.
"""

from __future__ import annotations

import json

import pytest

from clipforge import cli, config, db, paths
from clipforge.render import hooks

TRACK_MAP = json.dumps({
    "roles": {"mixed": 0, "mic": 1, "game": 2, "party": 3},
    "warnings": [], "source": "metadata tags", "titles": {},
})


@pytest.fixture
def rendered(tmp_path, speech_manifest):
    """A stream with two rendered clips and the transcript behind them."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    paths.StreamPaths(cfg.data_root, "s").ensure()

    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, "
        "audio_track_map, duration_s, resolution) VALUES "
        "('s', '2026-08-14', 'm.mkv', 'vod', ?, 95.0, '640x360')",
        (TRACK_MAP,),
    )
    tracks = json.loads(TRACK_MAP)["roles"]
    for utterance in speech_manifest["utterances"]:
        conn.execute(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text, "
            "speaker, track, words) VALUES ('s', ?, ?, ?, ?, ?, ?, '[]')",
            (utterance["index"], utterance["t_start"], utterance["t_end"],
             utterance["text"],
             "operator" if utterance["track"] == "mic" else "party",
             tracks[utterance["track"]]),
        )

    windows = [(8.0, 22.0), (49.0, 58.0)]
    for index, (start, end) in enumerate(windows):
        cursor = conn.execute(
            "INSERT INTO exports (stream_id, kind, path, item_count) "
            "VALUES ('s', 'clip', ?, 1)", (f"streams/s/exports/clip{index}.mp4",))
        conn.execute(
            "INSERT INTO export_items (export_id, seq, t_start, t_end, role) "
            "VALUES (?, 0, ?, ?, 'clip')", (cursor.lastrowid, start, end))
    conn.commit()
    yield cfg, conn, speech_manifest
    conn.close()


def reply_for(clips, *, quote=None, export_id=None, options=None):
    clip = clips[0]
    return json.dumps({"hooks": [{
        "export_id": clip.export_id if export_id is None else export_id,
        "quote": quote if quote is not None else clip.transcript.split("\n")[0],
        "options": options if options is not None else ["one", "two", "three"],
    }]})


# --------------------------------------------------------------------------
# what the model is asked about
# --------------------------------------------------------------------------


def test_only_rendered_clips_are_offered(rendered):
    """§8.5 stores the result in exports.hook_text, and that row exists only
    once something has been rendered."""
    _cfg, conn, _m = rendered

    clips = hooks.load_clips(conn, "s")

    assert len(clips) == 2
    assert all(clip.export_id > 0 for clip in clips)


def test_re_rendering_does_not_ask_about_the_same_clip_twice(rendered):
    """`exports` is append-only, so rendering a stream three times leaves three
    rows per moment. Asking a model about the same clip three times spends its
    attention and the operator's on nothing."""
    _cfg, conn, _m = rendered
    before = hooks.load_clips(conn, "s")
    first = conn.execute(
        "SELECT e.id, i.t_start, i.t_end FROM exports e JOIN export_items i "
        "ON i.export_id = e.id WHERE e.stream_id = 's' ORDER BY e.id LIMIT 1"
    ).fetchone()

    cursor = conn.execute(
        "INSERT INTO exports (stream_id, kind, path, item_count) "
        "VALUES ('s', 'clip', 'streams/s/exports/again.mp4', 1)")
    conn.execute(
        "INSERT INTO export_items (export_id, seq, t_start, t_end, role) "
        "VALUES (?, 0, ?, ?, 'clip')",
        (cursor.lastrowid, first["t_start"], first["t_end"]))
    conn.commit()

    after = hooks.load_clips(conn, "s")

    assert len(after) == len(before)
    # And it is the NEW one that is offered, not the stale path.
    assert cursor.lastrowid in {clip.export_id for clip in after}


def test_a_clip_carries_the_transcript_of_its_own_window(rendered):
    _cfg, conn, manifest = rendered
    clips = hooks.load_clips(conn, "s")

    inside = [u for u in manifest["utterances"]
              if u["t_end"] >= clips[0].t_start and u["t_start"] <= clips[0].t_end]

    for utterance in inside:
        assert utterance["text"] in clips[0].transcript


def test_the_transcript_says_who_spoke(rendered):
    """The same provenance rule the captions use, so the model is told what
    the operator would see."""
    _cfg, conn, _m = rendered

    transcript = hooks.load_clips(conn, "s")[1].transcript

    assert "mic:" in transcript or "party:" in transcript


def test_a_stream_with_nothing_rendered_says_what_to_do(rendered):
    _cfg, conn, _m = rendered
    conn.execute("DELETE FROM exports")
    conn.commit()

    with pytest.raises(hooks.HookError, match="clipforge render"):
        hooks.build_prompt(config.load(), hooks.load_clips(conn, "s"), "s")


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def test_one_prompt_covers_every_clip(rendered):
    """One, not one per clip: C8 budgets ~35 minutes of hands-on time per
    stream and five paste round trips would spend a chunk of it."""
    cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")

    prompt = hooks.build_prompt(cfg, clips, "s")

    for clip in clips:
        assert f"export_id: {clip.export_id}" in prompt


def test_the_prompt_asks_for_the_configured_number_of_options(rendered):
    cfg, conn, _m = rendered
    wanted = int(cfg.get("render.hooks.options"))

    prompt = hooks.build_prompt(cfg, hooks.load_clips(conn, "s"), "s")

    assert f"{wanted} hook variants" in prompt


def test_the_prompt_demands_a_verbatim_quote(rendered):
    """§12.3. It is the machine-checkable half of the contract, so the prompt
    has to say it is checked."""
    cfg, conn, _m = rendered

    prompt = hooks.build_prompt(cfg, hooks.load_clips(conn, "s"), "s")

    assert "VERBATIM" in prompt
    assert "discarded" in prompt


def test_the_prompt_carries_a_json_schema(rendered):
    """§12.3: "constrained JSON schema on every call"."""
    cfg, conn, _m = rendered

    prompt = hooks.build_prompt(cfg, hooks.load_clips(conn, "s"), "s")

    assert '"export_id"' in prompt and '"options"' in prompt and '"quote"' in prompt


def test_the_prompt_never_asks_for_a_timestamp(rendered):
    """§12.1: the model returns ids, and every time is resolved here."""
    cfg, conn, _m = rendered

    prompt = hooks.build_prompt(cfg, hooks.load_clips(conn, "s"), "s")

    assert "timestamp" not in prompt.lower()
    assert "t_start" not in prompt


# --------------------------------------------------------------------------
# parsing whatever came back
# --------------------------------------------------------------------------


def test_prose_around_the_json_is_tolerated():
    """What comes out of a chat window has an explanation wrapped round it.
    Asking the operator to trim that by hand is how a tool goes unused."""
    text = ('Sure! Here are some hooks:\n\n```json\n'
            '{"hooks": [{"export_id": 3, "quote": "x", "options": ["a"]}]}\n'
            '```\n\nLet me know if you want more.')

    parsed = hooks.parse_reply(text)

    assert parsed["hooks"][0]["export_id"] == 3


def test_the_last_object_wins():
    text = '{"hooks": []} and then {"hooks": [{"export_id": 9}]}'

    assert hooks.parse_reply(text)["hooks"][0]["export_id"] == 9


@pytest.mark.parametrize("junk", ["", "no json at all", "{not json}", "{{{"])
def test_an_unparseable_reply_is_none(junk):
    assert hooks.parse_reply(junk) is None


# --------------------------------------------------------------------------
# §12.2 — ids are checked, drops are counted
# --------------------------------------------------------------------------


def test_a_valid_reply_is_accepted(rendered):
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")

    result = hooks.validate(hooks.parse_reply(reply_for(clips)), clips)

    assert len(result.proposals) == 1
    assert result.dropped == []
    assert result.invalid_id_rate == 0.0


def test_an_id_that_does_not_exist_is_dropped_and_counted(rendered):
    """§12.2: "Non-existent IDs are silently dropped, and the drop is logged
    to tool_metrics for monitoring hallucination rate"."""
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, export_id=9999)), clips)

    assert result.proposals == []
    assert result.dropped[0][0] == "unknown-id"
    assert result.invalid_id_rate == 1.0


def test_the_rate_is_a_fraction_of_what_was_returned(rendered):
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    good = json.loads(reply_for(clips))["hooks"][0]
    bad = dict(good, export_id=9999)

    result = hooks.validate({"hooks": [good, bad]}, clips)

    assert result.returned == 2
    assert result.invalid_id_rate == 0.5


def test_an_id_from_another_stream_is_not_valid_here(rendered):
    """Real database ids rather than 1..5, so a fabrication has somewhere to
    fail — with five clips numbered 1 to 5, any invented number looks fine."""
    _cfg, conn, _m = rendered
    conn.execute("INSERT INTO streams (id, date, master_path, marker_time_base) "
                 "VALUES ('other', '2026-08-14', 'x.mkv', 'vod')")
    cursor = conn.execute(
        "INSERT INTO exports (stream_id, kind, path, item_count) "
        "VALUES ('other', 'clip', 'x.mp4', 1)")
    conn.execute("INSERT INTO export_items (export_id, seq, t_start, t_end, role) "
                 "VALUES (?, 0, 0.0, 5.0, 'clip')", (cursor.lastrowid,))
    conn.commit()
    clips = hooks.load_clips(conn, "s")

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, export_id=cursor.lastrowid)), clips)

    assert result.proposals == []
    assert result.dropped[0][0] == "unknown-id"


def test_the_rate_reaches_tool_metrics(rendered):
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, export_id=9999)), clips)

    hooks.record(conn, "s", result)

    row = conn.execute(
        "SELECT value, meta FROM tool_metrics WHERE metric = ?",
        (hooks.INVALID_ID_METRIC,)).fetchone()
    assert row["value"] == 1.0
    assert json.loads(row["meta"])["returned"] == 1


def test_the_metric_is_the_name_ss14_gives_it():
    """§14's table calls it `llm_invalid_id_rate`. A different name here means
    whatever reads it later finds nothing."""
    assert hooks.INVALID_ID_METRIC == "llm_invalid_id_rate"


# --------------------------------------------------------------------------
# §12.3 — the quote has to be in the clip
# --------------------------------------------------------------------------


def test_a_fabricated_quote_is_rejected(rendered):
    """The cheap machine-checkable guard against a model answering about a
    clip it never read."""
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, quote="he said the thing about bees")),
        clips)

    assert result.proposals == []
    assert result.dropped[0][0] == "bad-quote"


def test_a_quote_from_the_fixtures_own_dialogue_passes(rendered):
    """Against the manifest's authored text, not something written here."""
    _cfg, conn, manifest = rendered
    clips = hooks.load_clips(conn, "s")
    spoken = next(u["text"] for u in manifest["utterances"]
                  if u["t_start"] >= clips[0].t_start
                  and u["t_end"] <= clips[0].t_end)

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, quote=spoken)), clips)

    assert len(result.proposals) == 1


def test_rewrapping_a_quote_is_still_verbatim(rendered):
    """Whitespace and case only. A model that re-wrapped a line has quoted the
    clip; one that rewrote the words has not."""
    _cfg, conn, manifest = rendered
    clips = hooks.load_clips(conn, "s")
    spoken = next(u["text"] for u in manifest["utterances"]
                  if u["t_start"] >= clips[0].t_start
                  and u["t_end"] <= clips[0].t_end)
    rewrapped = "  " + spoken.upper().replace(" ", "\n  ") + " "

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, quote=rewrapped)), clips)

    assert len(result.proposals) == 1


def test_a_missing_quote_is_rejected(rendered):
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, quote="")), clips)

    assert result.dropped[0][0] == "no-quote"


def test_an_entry_with_no_options_is_rejected(rendered):
    _cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")

    result = hooks.validate(
        hooks.parse_reply(reply_for(clips, options=[])), clips)

    assert result.dropped[0][0] == "no-options"


def test_a_reply_with_no_hooks_array_is_reported(rendered):
    _cfg, conn, _m = rendered

    result = hooks.validate({"answer": "sure"}, hooks.load_clips(conn, "s"))

    assert result.dropped[0][0] == "malformed"


def test_nothing_at_all_is_not_a_crash(rendered):
    _cfg, conn, _m = rendered

    result = hooks.validate(None, hooks.load_clips(conn, "s"))

    assert result.proposals == [] and result.invalid_id_rate == 0.0


# --------------------------------------------------------------------------
# storing the operator's choice
# --------------------------------------------------------------------------


def test_picking_stores_the_text(rendered):
    _cfg, conn, _m = rendered
    clip = hooks.load_clips(conn, "s")[0]

    hooks.store(conn, clip.export_id, "he did WHAT with Hawkeye")

    assert conn.execute("SELECT hook_text FROM exports WHERE id = ?",
                        (clip.export_id,)).fetchone()[0] == "he did WHAT with Hawkeye"


def test_picking_an_export_that_does_not_exist_is_refused(rendered):
    _cfg, conn, _m = rendered

    with pytest.raises(hooks.HookError, match="no export"):
        hooks.store(conn, 9999, "x")


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def test_the_command_writes_a_prompt(rendered, tmp_path, capsys):
    cfg, conn, _m = rendered
    conn.commit()
    out = tmp_path / "prompt.md"

    code = cli.main(["hook", "s", "--out", str(out),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 0
    assert "export_id:" in out.read_text(encoding="utf-8")
    assert "--apply" in capsys.readouterr().out


def test_apply_reports_drops_and_stores_nothing(rendered, tmp_path, capsys):
    """§8.5: the operator picks and rewrites. `--apply` validates and reports;
    it does not choose."""
    cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    conn.commit()
    reply = tmp_path / "reply.json"
    reply.write_text(reply_for(clips), encoding="utf-8")

    code = cli.main(["hook", "s", "--apply", str(reply),
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 0
    assert "--pick" in capsys.readouterr().out
    assert conn.execute("SELECT hook_text FROM exports WHERE id = ?",
                        (clips[0].export_id,)).fetchone()[0] is None


def test_pick_by_index_reads_the_reply_it_came_from(rendered, tmp_path):
    cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    conn.commit()
    reply = tmp_path / "reply.json"
    reply.write_text(reply_for(clips, options=["first", "second"]), encoding="utf-8")

    code = cli.main(["hook", "s", "--apply", str(reply),
                     "--pick", str(clips[0].export_id), "2",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 0
    fresh = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        assert fresh.execute("SELECT hook_text FROM exports WHERE id = ?",
                             (clips[0].export_id,)).fetchone()[0] == "second"
    finally:
        fresh.close()


def test_pick_by_index_without_the_reply_says_why(rendered, capsys):
    """The options are never stored — storing four hooks nobody chose is how
    `hook_text` stops meaning "the hook"."""
    cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    conn.commit()

    code = cli.main(["hook", "s", "--pick", str(clips[0].export_id), "2",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 1
    assert "--apply" in capsys.readouterr().out


def test_pick_with_literal_text_needs_no_reply(rendered, tmp_path):
    cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    conn.commit()

    code = cli.main(["hook", "s", "--pick", str(clips[0].export_id),
                     "my own words", "--set",
                     f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 0
    fresh = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        assert fresh.execute("SELECT hook_text FROM exports WHERE id = ?",
                             (clips[0].export_id,)).fetchone()[0] == "my own words"
    finally:
        fresh.close()


def test_an_out_of_range_option_is_refused(rendered, tmp_path, capsys):
    cfg, conn, _m = rendered
    clips = hooks.load_clips(conn, "s")
    conn.commit()
    reply = tmp_path / "reply.json"
    reply.write_text(reply_for(clips, options=["only one"]), encoding="utf-8")

    code = cli.main(["hook", "s", "--apply", str(reply),
                     "--pick", str(clips[0].export_id), "5",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 1
    assert "out of range" in capsys.readouterr().out


def test_an_unknown_stream_is_refused(rendered, capsys):
    cfg, _conn, _m = rendered

    code = cli.main(["hook", "nope",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 1
    assert "no stream" in capsys.readouterr().out


def test_only_the_manual_source_exists_and_says_so(rendered, capsys):
    cfg, _conn, _m = rendered

    code = cli.main(["hook", "s", "--set", "render.hooks.source=anthropic",
                     "--set", f"paths.data_root={cfg.data_root.as_posix()}"])

    assert code == 1
    assert "needs a key" in capsys.readouterr().out


def test_the_manual_source_is_always_available():
    """It needs a person and a browser, not a credential — so unlike the Ollama
    and WhisperX checks, there is nothing to defer on."""
    usable, why = hooks.ManualHookSource().available(config.load())

    assert usable and why == ""
