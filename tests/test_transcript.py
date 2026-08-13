"""The `whisperx` stage (§5.7, §5.8).

Two layers. Almost everything the stage *does* — resolving settings, merging two
passes, assigning `seq`, writing rows — is ordinary code that must not need a
3 GB model to test, so it runs against a fake transcriber that replays the
speech fixture's own authored utterances.

The `asr`-marked tests load a real Whisper model and need `--asr`. They are the
only thing that proves the wiring against an API which has changed shape across
whisperx releases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge import config, db
from clipforge.extract import transcript
from clipforge.extract.transcript import Segment, Word
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext


# --------------------------------------------------------------------------
# a transcriber that needs no model
# --------------------------------------------------------------------------


class FakeTranscriber:
    """Replays known utterances, so the stage is testable end to end.

    Takes the fixture's manifest, which already holds the exact text and
    placement of every line — the whole reason the speech fixture authors them.
    """

    def __init__(self, utterances, words_per_line: int | None = None) -> None:
        self.utterances = utterances
        self.words_per_line = words_per_line
        self.calls: list[tuple[str, int]] = []

    def transcribe(self, path: Path, *, role: str, track: int) -> list[Segment]:
        self.calls.append((role, track))
        out = []
        for entry in self.utterances:
            if entry["track"] != role:
                continue
            words = entry["text"].split()
            if self.words_per_line:
                words = words[: self.words_per_line]
            span = (entry["t_end"] - entry["t_start"]) / max(len(words), 1)
            out.append(Segment(
                t_start=entry["t_start"], t_end=entry["t_end"],
                text=entry["text"], track=track, role=role,
                words=[
                    Word(w=w,
                         start=round(entry["t_start"] + i * span, 3),
                         end=round(entry["t_start"] + (i + 1) * span, 3),
                         score=0.9)
                    for i, w in enumerate(words)
                ],
            ))
        return out


@pytest.fixture
def transcribed(tmp_path, speech_fixture_dir, speech_manifest):
    """The speech fixture, processed through Phase 1 and then transcribed."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)

    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=speech_fixture_dir / speech_manifest["video"])
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe", "audio_split"]))

    ctx = StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id, log=lambda *_: None)
    fake = FakeTranscriber(speech_manifest["utterances"])
    transcript.run(ctx, transcriber=fake)

    yield cfg, conn, ctx, fake
    conn.close()


def rows(conn, stream_id):
    return conn.execute(
        "SELECT * FROM segments WHERE stream_id = ? ORDER BY seq", (stream_id,)
    ).fetchall()


# --------------------------------------------------------------------------
# settings §5.7 gets wrong for anything but a CUDA box
# --------------------------------------------------------------------------


def test_compute_type_auto_resolves_per_device():
    """§5.7 hardcodes float16, which CTranslate2 REFUSES on CPU. The spec is
    right for the streaming PC and fails everywhere else with an error naming
    neither the setting nor the cause."""
    assert transcript.resolve_compute_type("cpu", "auto") == transcript.CPU_COMPUTE_TYPE
    assert transcript.resolve_compute_type("cuda", "auto") == transcript.CUDA_COMPUTE_TYPE
    assert transcript.resolve_compute_type("cuda:1", None) == transcript.CUDA_COMPUTE_TYPE


def test_an_explicit_compute_type_is_respected():
    assert transcript.resolve_compute_type("cpu", "float32") == "float32"


def test_the_shipped_default_is_auto():
    cfg = config.load()
    assert str(cfg.get("extract.whisperx.compute_type")).lower() == "auto"


def test_the_shipped_defaults_follow_the_spec():
    """§5.7's model and device, so the streaming PC is right out of the box."""
    cfg = config.load()
    assert cfg.get("extract.whisperx.model") == "large-v3"
    assert cfg.get("extract.whisperx.device") == "cuda"
    assert cfg.get("extract.whisperx.align") is True


def test_only_speech_tracks_are_transcribed():
    """§5.8 runs on mic and party. The game track is game audio; transcribing it
    produces confident nonsense that would then be scored and shown."""
    roles = config.load().get("extract.whisperx.roles")
    assert "game" not in roles
    assert "mic" in roles


def test_the_language_is_pinned_by_default():
    """Whisper detects per chunk, and a chunk of near-silence is exactly where
    it decides the language is Welsh and hallucinates to match."""
    assert config.load().get("extract.whisperx.language") == "en"


# --------------------------------------------------------------------------
# vocabulary (§5.7)
# --------------------------------------------------------------------------


def test_the_vocabulary_loads_and_holds_hero_names():
    terms = transcript.load_vocabulary(config.load())
    assert "Hawkeye" in terms
    assert "Iron Fist" in terms


def test_the_vocabulary_is_order_stable():
    """It feeds params_hash. A list that reshuffled between runs would
    invalidate every completed transcription for no reason."""
    cfg = config.load()
    assert transcript.load_vocabulary(cfg) == transcript.load_vocabulary(cfg)


def test_a_local_vocabulary_is_merged(tmp_path, monkeypatch):
    """§5.7 wants real people's names in there, which should not need a commit."""
    monkeypatch.setattr(transcript, "CONFIG_DIR", tmp_path)
    (tmp_path / "vocabulary.txt").write_text("Hawkeye\n# a comment\nMantis\n")
    (tmp_path / "vocabulary.local.txt").write_text("Wobbles\n")

    terms = transcript.load_vocabulary(config.load())
    assert terms == ["Hawkeye", "Mantis", "Wobbles"]


def test_comments_and_duplicates_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "CONFIG_DIR", tmp_path)
    (tmp_path / "vocabulary.txt").write_text("Jeff\njeff  # dupe\n\n# nothing\nNamor\n")
    assert transcript.load_vocabulary(config.load()) == ["Jeff", "Namor"]


def test_the_term_count_is_capped(tmp_path, monkeypatch):
    """Hotwords and the prompt each get half the prompt budget and are truncated
    to fit, so an unbounded list crowds out the terms that matter."""
    monkeypatch.setattr(transcript, "CONFIG_DIR", tmp_path)
    (tmp_path / "vocabulary.txt").write_text("\n".join(f"term{i}" for i in range(500)))
    cfg = config.load(overrides=["extract.whisperx.max_vocabulary_terms=10"])
    assert len(transcript.load_vocabulary(cfg)) == 10


@pytest.mark.parametrize("mode,expected", [
    ("both", {"hotwords", "initial_prompt"}),
    ("hotwords", {"hotwords"}),
    ("prompt", {"initial_prompt"}),
    ("none", set()),
])
def test_vocabulary_mode_selects_the_mechanism(mode, expected):
    cfg = config.load(overrides=[f"extract.whisperx.vocabulary_mode={mode}"])
    assert set(transcript.vocabulary_options(cfg, ["Hawkeye", "Mantis"])) == expected


def test_an_unknown_vocabulary_mode_is_refused():
    cfg = config.load(overrides=["extract.whisperx.vocabulary_mode=telepathy"])
    with pytest.raises(transcript.TranscriptError, match="vocabulary_mode"):
        transcript.vocabulary_options(cfg, ["Hawkeye"])


def test_no_terms_means_no_options():
    assert transcript.vocabulary_options(config.load(), []) == {}


def test_the_vocabulary_is_trimmed_to_whispers_prompt_limit():
    """MEASURED, by running it. Whisper has positional encodings for 448 prompt
    tokens; faster-whisper truncates hotwords to 223 AND the initial prompt to
    223, and the framing pushes the total to 451. Seeding both at any real
    length raised `No position encodings are defined for positions >= 448` from
    inside CTranslate2, with nothing pointing at the vocabulary."""
    terms = [f"HeroNumber{i}" for i in range(500)]
    for mode in ("hotwords", "prompt", "both"):
        fitted = transcript.fit_terms(terms, mode)
        budget = transcript.VOCABULARY_TOKEN_BUDGET[mode]
        assert len(", ".join(fitted)) <= budget * transcript.CHARS_PER_TOKEN
        assert fitted, f"{mode} trimmed everything"

    # `both` spends the budget twice over, so it gets less of it.
    assert len(transcript.fit_terms(terms, "both")) < \
        len(transcript.fit_terms(terms, "hotwords"))


def test_both_modes_together_stay_under_the_position_limit():
    """The specific arithmetic that broke: the two mechanisms are concatenated
    into one prompt, so their combined cost is what has to fit."""
    terms = [f"HeroNumber{i}" for i in range(500)]
    fitted = transcript.fit_terms(terms, "both")
    combined_tokens = 2 * len(", ".join(fitted)) / transcript.CHARS_PER_TOKEN
    assert combined_tokens < transcript.MAX_PROMPT_TOKENS - 8


def test_the_default_mode_is_the_one_that_works():
    """§5.7 says both; both is the configuration that raised."""
    assert config.load().get("extract.whisperx.vocabulary_mode") == "hotwords"


def test_trimming_is_reported(capsys):
    said: list[str] = []
    cfg = config.load(overrides=["extract.whisperx.vocabulary_mode=both"])
    transcript.vocabulary_options(cfg, [f"Term{i}" for i in range(500)], log=said.append)
    assert any("trimmed" in line for line in said)


# --------------------------------------------------------------------------
# merging two passes (§5.8)
# --------------------------------------------------------------------------


def segment(start, end, track, text="x"):
    return Segment(t_start=start, t_end=end, text=text, track=track)


def test_passes_interleave_by_time():
    merged = transcript.merge_passes([
        [segment(10.0, 12.0, 1), segment(30.0, 32.0, 1)],
        [segment(20.0, 22.0, 3)],
    ])
    assert [s.t_start for s in merged] == [10.0, 20.0, 30.0]


def test_overlapping_segments_from_different_tracks_both_survive():
    """§5.8's "merge by timestamp" is underspecified the moment both tracks have
    speech at once. Two people talking over each other are two segments, not
    one — combining them would interleave two sentences into one unreadable
    text."""
    merged = transcript.merge_passes([
        [segment(52.0, 55.0, 1, "let's go")],
        [segment(52.6, 55.6, 3, "oh my god")],
    ])
    assert len(merged) == 2
    assert {s.text for s in merged} == {"let's go", "oh my god"}


def test_the_merge_order_is_total():
    """Same start and end on two tracks still has to sort deterministically, or
    `seq` moves between runs and §12.1's identifiers stop meaning anything."""
    first = transcript.merge_passes([[segment(5.0, 6.0, 3)], [segment(5.0, 6.0, 1)]])
    second = transcript.merge_passes([[segment(5.0, 6.0, 1)], [segment(5.0, 6.0, 3)]])
    assert [s.track for s in first] == [s.track for s in second] == [1, 3]


def test_merging_nothing_is_empty():
    assert transcript.merge_passes([[], []]) == []


# --------------------------------------------------------------------------
# what lands in the database
# --------------------------------------------------------------------------


def test_both_tracks_are_transcribed(transcribed):
    _cfg, _conn, _ctx, fake = transcribed
    assert {role for role, _track in fake.calls} == {"mic", "party"}


def test_every_utterance_becomes_a_segment(transcribed, speech_manifest):
    _cfg, conn, ctx, _fake = transcribed
    assert len(rows(conn, ctx.stream_id)) == len(speech_manifest["utterances"])


def test_seq_is_assigned_after_the_merge(transcribed):
    """§12.1 makes seq the id every LLM call returns instead of a timestamp, so
    it must be contiguous, unique per stream, and in time order."""
    _cfg, conn, ctx, _fake = transcribed
    found = rows(conn, ctx.stream_id)
    assert [r["seq"] for r in found] == list(range(len(found)))
    starts = [r["t_start"] for r in found]
    assert starts == sorted(starts)


def test_the_track_each_segment_came_from_is_recorded(transcribed, speech_manifest):
    _cfg, conn, ctx, _fake = transcribed
    expected = speech_manifest["expected_track_map"]
    tracks = {r["track"] for r in rows(conn, ctx.stream_id)}
    assert tracks == {expected["mic"], expected["party"]}


def test_speaker_is_left_for_the_next_stage(transcribed):
    """§5.8's assignment is speaker_assign's job, and it needs track energy this
    stage never looks at."""
    _cfg, conn, ctx, _fake = transcribed
    assert all(r["speaker"] is None for r in rows(conn, ctx.stream_id))


def test_words_are_stored_in_the_schema_s_shape(transcribed):
    """§3.2: `words TEXT -- JSON: [{w,start,end,score},...]`."""
    _cfg, conn, ctx, _fake = transcribed
    words = json.loads(rows(conn, ctx.stream_id)[0]["words"])
    assert words
    assert set(words[0]) == {"w", "start", "end", "score"}


def test_word_timestamps_fall_inside_their_segment(transcribed):
    _cfg, conn, ctx, _fake = transcribed
    for row in rows(conn, ctx.stream_id):
        for word in json.loads(row["words"]):
            assert row["t_start"] - 0.01 <= word["start"] <= row["t_end"] + 0.01


def test_running_twice_replaces_rather_than_doubles(transcribed, speech_manifest):
    """seq has a UNIQUE index behind it, so appending would both collide and
    silently double the transcript."""
    _cfg, conn, ctx, fake = transcribed
    transcript.run(ctx, transcriber=fake)
    found = rows(conn, ctx.stream_id)
    assert len(found) == len(speech_manifest["utterances"])
    assert [r["seq"] for r in found] == list(range(len(found)))


def test_verify_fails_before_the_stage_runs(tmp_path, speech_fixture_dir, speech_manifest):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=speech_fixture_dir / speech_manifest["video"])
    )
    ctx = StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id, log=lambda *_: None)
    ok, why = transcript.verify(ctx)
    conn.close()
    assert not ok and "segments" in why


def test_verify_passes_after_it(transcribed):
    _cfg, _conn, ctx, _fake = transcribed
    assert transcript.verify(ctx)[0]


# --------------------------------------------------------------------------
# unaligned words
# --------------------------------------------------------------------------


def test_an_unaligned_word_keeps_its_text(transcribed):
    """whisperx cannot place every word. Dropping it corrupts the text §5.6
    matches against; inventing a timestamp corrupts §6.3's snapping."""
    _cfg, conn, ctx, _fake = transcribed

    class Unaligned:
        def transcribe(self, path, *, role, track):
            if role != "mic":
                return []
            return [Segment(1.0, 2.0, "hello there", track, role,
                            [Word("hello", 1.0, 1.5, 0.9), Word("there", None, None, None)])]

    transcript.run(ctx, transcriber=Unaligned())
    words = json.loads(rows(conn, ctx.stream_id)[0]["words"])
    assert [w["w"] for w in words] == ["hello", "there"]
    assert words[1]["start"] is None and words[1]["end"] is None


def test_word_reports_whether_it_is_aligned():
    assert Word("a", 1.0, 2.0, 0.9).aligned
    assert not Word("a", None, None, None).aligned


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_stream_with_no_speech_tracks_is_refused(transcribed):
    _cfg, conn, ctx, _fake = transcribed
    conn.execute(
        "UPDATE streams SET audio_track_map = ? WHERE id = ?",
        (json.dumps({"roles": {"game": 2}}), ctx.stream_id),
    )
    with pytest.raises(transcript.TranscriptError, match="no speech tracks"):
        transcript.run(ctx, transcriber=FakeTranscriber([]))


def test_a_missing_wav_names_the_fix(transcribed):
    _cfg, _conn, ctx, fake = transcribed
    ctx.paths.audio("mic").unlink()
    with pytest.raises(transcript.TranscriptError, match="audio_split"):
        transcript.run(ctx, transcriber=fake)


# --------------------------------------------------------------------------
# staleness
# --------------------------------------------------------------------------


def test_params_change_with_the_model_but_not_the_device(transcribed):
    """Device and compute_type do nudge the output, but invalidating a
    forty-minute transcription because the library moved from the streaming PC
    to a laptop is the mistake master_identity already avoids."""
    _cfg, conn, ctx, _fake = transcribed
    base = transcript.params(ctx)

    moved = StageContext(
        cfg=config.load(overrides=[
            f"paths.data_root={ctx.cfg.data_root.as_posix()}",
            "extract.whisperx.device=cpu",
            "extract.whisperx.compute_type=int8",
        ]),
        conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
    )
    assert transcript.params(moved) == base

    retuned = StageContext(
        cfg=config.load(overrides=[
            f"paths.data_root={ctx.cfg.data_root.as_posix()}",
            "extract.whisperx.model=small",
        ]),
        conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
    )
    assert transcript.params(retuned) != base


def test_the_stage_is_registered():
    from clipforge.pipeline import stages

    spec = stages.get("whisperx")
    assert spec.module == "clipforge.extract.transcript"
    assert spec.implemented


# --------------------------------------------------------------------------
# availability — built is not the same as runnable
# --------------------------------------------------------------------------


def test_the_stage_is_off_by_default(transcribed):
    """MEASURED BUG. Marking the stage `implemented` put it in the default plan,
    so `clipforge run` began trying to pull a 3 GB CUDA model on every machine —
    and because `execute` stops at the first failure, `score` never ran and the
    stream produced no candidates at all. 48 tests went red.

    §15 also says to ship Phase 1 first: until speaker_assign, speech_rate and
    phrase_detect exist, a transcript feeds nothing."""
    _cfg, _conn, ctx, _fake = transcribed
    usable, why = transcript.available(ctx)
    assert not usable
    assert "enabled" in why


def test_enabling_it_makes_it_available(transcribed):
    _cfg, conn, ctx, _fake = transcribed
    enabled = StageContext(
        cfg=config.load(overrides=[
            f"paths.data_root={ctx.cfg.data_root.as_posix()}",
            "extract.whisperx.enabled=true",
        ]),
        conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
    )
    usable, why = transcript.available(enabled)
    # whisperx is installed in this venv; on a machine without it the reason
    # names the extra rather than failing the run.
    assert usable or "asr" in why


def test_an_unavailable_stage_is_deferred_not_failed(transcribed):
    """A Phase 1 user without the ASR extra must still get candidates."""
    from clipforge.pipeline import runner as runner_mod

    _cfg, conn, ctx, _fake = transcribed
    engine = runner_mod.Runner(cfg=ctx.cfg, conn=conn, stream_id=ctx.stream_id,
                               log=lambda *_: None)
    decisions = {d.stage: d for d in engine.plan()}

    assert decisions["whisperx"].action == runner_mod.DEFERRED
    assert "enabled" in decisions["whisperx"].reason
    # and the thing that matters: scoring is still reachable
    assert decisions["score"].action != runner_mod.BLOCKED


def test_availability_defaults_to_true_for_other_stages():
    """The hook is opt-in; nothing else grew a gate."""
    from clipforge.pipeline import stages

    assert stages.get("probe").available(None) == (True, "")
    assert stages.get("score").available(None) == (True, "")


# --------------------------------------------------------------------------
# the real thing (--asr)
# --------------------------------------------------------------------------


@pytest.fixture
def real_cfg(tmp_path):
    """tiny on CPU. large-v3 is the shipped default and needs a GPU."""
    return config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        "extract.whisperx.model=tiny",
        "extract.whisperx.device=cpu",
        "extract.whisperx.compute_type=int8",
        "extract.whisperx.batch_size=4",
    ])


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over words, normalised by reference length."""
    ref = reference.lower().replace(",", "").replace(".", "").replace("?", "").split()
    hyp = hypothesis.lower().replace(",", "").replace(".", "").replace("?", "").split()
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / max(len(ref), 1)


@pytest.mark.asr
def test_a_real_transcription_of_the_fixture(real_cfg, speech_fixture_dir, speech_manifest):
    """The one test that proves the wiring against the actual library."""
    db.open_db(real_cfg.db_path).close()
    conn = db.open_db(real_cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        real_cfg, conn,
        register.RegisterSpec(master=speech_fixture_dir / speech_manifest["video"]),
    )
    engine = runner.Runner(cfg=real_cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe", "audio_split"]))

    ctx = StageContext(cfg=real_cfg, conn=conn, stream_id=result.stream_id, log=print)
    transcript.run(ctx)

    found = rows(conn, result.stream_id)
    conn.close()

    assert found, "nothing was transcribed"
    assert {r["track"] for r in found} == {
        speech_manifest["expected_track_map"]["mic"],
        speech_manifest["expected_track_map"]["party"],
    }

    reference = " ".join(u["text"] for u in speech_manifest["utterances"])
    hypothesis = " ".join(r["text"] for r in found)
    rate = word_error_rate(reference, hypothesis)
    print(f"\nWER on tiny/CPU: {rate:.1%}")
    # Generous: `tiny` is the smallest model there is and this is measuring that
    # the pipeline works, not that Whisper is accurate.
    assert rate < 0.6, f"word error rate {rate:.1%} is too high to be wiring-correct"


def transcribe_fixture(cfg, speech_fixture_dir, speech_manifest) -> str:
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        cfg, conn,
        register.RegisterSpec(master=speech_fixture_dir / speech_manifest["video"]),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe", "audio_split"]))
    ctx = StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id, log=lambda *_: None)
    transcript.run(ctx)
    text = " ".join(
        r["text"] for r in conn.execute(
            "SELECT text FROM segments WHERE stream_id = ? ORDER BY seq",
            (result.stream_id,))
    )
    conn.close()
    return text


@pytest.mark.asr
def test_vocabulary_seeding_measurably_helps(tmp_path, speech_fixture_dir, speech_manifest):
    """§5.7: "Without this the transcript says 'Hawk I' and 'Iron First'
    permanently, and bad captions read as amateur immediately."

    Differential, because "the seeded run gets Hawkeye right" would be vacuous
    if the unseeded run did too. MEASURED on tiny/CPU: unseeded gives 16.3% WER
    and 7/11 hero-name occurrences, hotwords gives 2.9% and 11/11. Unseeded it
    hears "How could I" for Hawkeye and "Numbers'" for "Namor's"."""
    terms = speech_manifest["vocabulary_terms"]
    reference = " ".join(u["text"] for u in speech_manifest["utterances"])

    def score(mode: str) -> tuple[float, int]:
        cfg = config.load(overrides=[
            f"paths.data_root={(tmp_path / mode / 'data').as_posix()}",
            "extract.whisperx.model=tiny",
            "extract.whisperx.device=cpu",
            "extract.whisperx.compute_type=int8",
            "extract.whisperx.batch_size=4",
            f"extract.whisperx.vocabulary_mode={mode}",
        ])
        text = transcribe_fixture(cfg, speech_fixture_dir, speech_manifest).lower()
        found = sum(
            min(text.count(t.lower()), reference.lower().count(t.lower())) for t in terms
        )
        return word_error_rate(reference, text), found

    bare_wer, bare_terms = score("none")
    seeded_wer, seeded_terms = score("hotwords")
    print(f"\nunseeded: WER {bare_wer:.1%}, {bare_terms} vocabulary hits")
    print(f"seeded:   WER {seeded_wer:.1%}, {seeded_terms} vocabulary hits")

    assert seeded_terms > bare_terms, "seeding did not improve the vocabulary terms"
    assert seeded_wer <= bare_wer, "seeding made the transcript worse overall"


@pytest.mark.asr
def test_vad_keeps_whisper_out_of_the_silence(real_cfg, speech_fixture_dir, speech_manifest):
    """A4: "Without it, Whisper hallucinates phantom text over silence and
    music — a well-documented failure mode that would poison both the transcript
    and every downstream signal."

    The fixture's 19-second silence exists to make that assertable rather than
    trusted."""
    db.open_db(real_cfg.db_path).close()
    conn = db.open_db(real_cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        real_cfg, conn,
        register.RegisterSpec(master=speech_fixture_dir / speech_manifest["video"]),
    )
    engine = runner.Runner(cfg=real_cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe", "audio_split"]))

    ctx = StageContext(cfg=real_cfg, conn=conn, stream_id=result.stream_id, log=print)
    transcript.run(ctx)
    found = rows(conn, result.stream_id)
    conn.close()

    longest = max(speech_manifest["silence_windows"], key=lambda w: w[1] - w[0])
    start, end = longest
    inside = [
        r for r in found
        if r["t_start"] >= start + 1.0 and r["t_end"] <= end - 1.0
    ]
    print(f"\nsegments inside the {end - start:.1f}s silence: {len(inside)}")
    for row in inside:
        print(f"   PHANTOM {row['t_start']:.1f}-{row['t_end']:.1f}: {row['text']!r}")
    assert not inside
