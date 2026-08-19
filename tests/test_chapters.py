"""§9.3's chapter segmentation.

**These tests prove the mechanism and the invariants. They do not prove that
these are good chapters**, and nothing available here could: the speech fixture
is thirteen lines of one continuous conversation with no topic change in it, and
`fixture_long` is band-limited noise. What can be asserted is that a boundary
lands where the manifest says the silence is, that the embedding side responds
to a real topic seam, that the result tiles the stream, and that the output does
not depend on settings that must not affect it.

The silence threshold is overridden throughout: §9.3's 60 s is longer than any
gap any fixture authors. That is assertion against **config plus manifest**,
which is the rule — never against a literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clipforge import config, db, paths
from clipforge.digest import chapters
from clipforge.extract import embeddings
from clipforge.pipeline import runner, stages
from clipforge.pipeline.context import StageContext

SPEECH = Path(__file__).parent / "fixtures" / "_generated" / "speech"

#: Low enough that the speech fixture's longest authored gap (19.376 s) counts,
#: and the merge window small enough that two boundaries 22.6 s apart stay
#: separate. Both are config, so overriding them is how a 95 s fixture exercises
#: rules written for a three-hour stream.
SHORT = [
    "digest.chapters.min_silence_s=15",
    "digest.chapters.merge_within_s=10",
    "digest.chapters.target_min_s=20",
    "digest.chapters.target_max_s=60",
]


def _manifest() -> dict:
    return json.loads((SPEECH / "manifest.json").read_text(encoding="utf-8"))


def _ollama_ready(cfg) -> bool:
    models = embeddings.list_models(str(cfg.get("extract.embeddings.host")))
    if models is None:
        return False
    return embeddings.model_present(models, str(cfg.get("extract.embeddings.model")))


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """The speech fixture with segments seeded from the manifest and embedded.

    Seeded rather than transcribed for the reason `test_search.py` gives: ASR
    errors would decide whether segmentation passes, and the manifest's
    `utterances` are the authored ground truth.
    """
    if not SPEECH.is_dir():
        pytest.skip("speech fixture not generated")

    root = tmp_path_factory.mktemp("chapters")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    manifest = _manifest()

    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, duration_s) "
        "VALUES ('speech', '2026-08-17', 'Speech fixture', ?, ?)",
        (str(SPEECH / manifest["video"]), float(manifest["duration_s"])),
    )
    paths.StreamPaths(cfg.data_root, "speech").ensure()

    # §9.3's silence input reads mic_rms/party_rms through §6.4's gate, so the
    # audio stages have to have run. Only those three: `score` and `previews`
    # would add a minute of encoding to prove nothing this file asserts.
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="speech", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    wanted = {"audio_features"} | stages.ancestors("audio_features")
    engine.execute(engine.plan(sorted(wanted)))

    tracks = {"mic": 1, "party": 3}
    with db.transaction(conn):
        conn.executemany(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, "
            "track) VALUES (?,?,?,?,?,?,?)",
            [
                ("speech", seq, u["t_start"], u["t_start"] + u.get("duration_s", 3.0),
                 u["text"], "operator" if u["track"] == "mic" else "party",
                 tracks.get(u["track"]))
                for seq, u in enumerate(manifest["utterances"], start=1)
            ],
        )
    if _ollama_ready(cfg):
        embeddings.run(StageContext(cfg=cfg, conn=conn, stream_id="speech",
                                    log=lambda _m: None))
    yield cfg, conn, manifest
    conn.close()


# --------------------------------------------------------------------------
# the invariant that would be invisible downstream
# --------------------------------------------------------------------------


def _tiles(result, duration_s: float) -> None:
    assert result.chapters
    assert result.chapters[0].t_start == pytest.approx(0.0)
    assert result.chapters[-1].t_end == pytest.approx(duration_s)
    for before, after in zip(result.chapters, result.chapters[1:], strict=False):
        assert before.t_end == pytest.approx(after.t_start)
    assert all(c.duration_s > 0 for c in result.chapters)


def test_chapters_tile_the_stream(seeded):
    """§9.2's structure is a partition and §9.4 chunks over it, so a gap between
    two chapters silently drops that transcript from the digest — no error, just
    a stream the model was never shown part of."""
    cfg, conn, manifest = seeded
    tuned = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}", *SHORT])
    _tiles(chapters.segment(conn, "speech", tuned), float(manifest["duration_s"]))


def test_a_broken_tiling_is_refused_rather_than_returned():
    """The check is real, not decorative."""
    made = [chapters.Chapter(0, 0.0, 10.0), chapters.Chapter(1, 20.0, 30.0)]
    with pytest.raises(chapters.ChapterError, match="do not meet"):
        chapters._assert_tiles(made, 30.0)
    with pytest.raises(chapters.ChapterError, match="not 0"):
        chapters._assert_tiles([chapters.Chapter(0, 5.0, 30.0)], 30.0)


# --------------------------------------------------------------------------
# input 3: silence, asserted against the manifest
# --------------------------------------------------------------------------


def test_the_silence_boundary_lands_at_the_authored_gap(seeded):
    """The manifest authors the gaps; the threshold is config. Neither is a
    literal in this file.

    The boundary is the END of the gap, not its middle: the dead air belongs to
    the chapter that just finished.
    """
    cfg, conn, manifest = seeded
    tuned = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}", *SHORT])
    minimum = float(tuned.get("digest.chapters.min_silence_s"))

    long_enough = [(a, b) for a, b in manifest["silence_windows"]
                   if b - a >= minimum and b < float(manifest["duration_s"])]
    assert long_enough, "no authored gap clears the configured threshold"

    result = chapters.segment(conn, "speech", tuned)
    silences = [b for b in result.boundaries if b.source == "silence"]
    assert len(silences) == len(long_enough)
    for boundary, (_start, end) in zip(silences, long_enough, strict=True):
        assert boundary.t == pytest.approx(end, abs=0.5)


def test_a_threshold_above_every_gap_finds_no_silence_boundary(seeded):
    """§9.3's own 60 s is longer than anything any fixture authors, which is why
    every other test here overrides it. Worth asserting rather than assuming."""
    cfg, conn, manifest = seeded
    strict = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        f"digest.chapters.min_silence_s={float(manifest['longest_silence_s']) + 10}",
    ])
    found, reason = chapters.silence_boundaries(
        conn, "speech", strict, float(manifest["duration_s"]))
    assert found == []
    assert reason == ""     # the input ran; it simply found nothing


def test_silence_uses_the_same_gate_as_6_4(seeded):
    """§6.4's "ANY speech" and §5.4.1's "VAD on both tracks" are one question, and
    HANDOFF records that asking it twice was a real hazard. This asserts chapters
    asks it through `gates`, not through a third copy."""
    source = Path(chapters.__file__).read_text(encoding="utf-8")
    assert "gates.speech_activity" in source
    # ...and no local re-derivation of the union. Both names appear in prose
    # explaining the rule, so this looks for a CALL rather than a mention.
    assert "speech |=" not in source
    assert "derived.speech_gate(" not in source


# --------------------------------------------------------------------------
# input 1: embedding shift
# --------------------------------------------------------------------------


def test_the_embedding_side_finds_a_real_topic_seam(seeded):
    """The mechanism test §9.3's own formulation failed.

    Two obviously unrelated topics, embedded at test time; the strongest
    boundary must land at the seam. MEASURED during design: §9.3's "consecutive
    rolling-window" form misses this at window 2 where the before/after form
    hits it at 2, 3 and 4.
    """
    cfg, _conn, _ = seeded
    if not _ollama_ready(cfg):
        pytest.skip("ollama is not running with the configured embedding model")

    gaming = ["Hawkeye just hit that shot from across the map.",
              "Luna Snow ulted right as I got the pick.",
              "Iron Fist is diving our backline again.",
              "I'm on Mantis, I'll try to heal you through it.",
              "Namor's turrets are melting me."]
    baking = ["So you want to preheat the oven to about 200 degrees.",
              "Fold the butter into the flour until it looks like breadcrumbs.",
              "Let the dough rest in the fridge for half an hour.",
              "Roll it out thin and line the tin with it.",
              "Blind bake it with beans on top so it keeps its shape."]

    embedder = embeddings.OllamaEmbedder(
        model=str(cfg.get("extract.embeddings.model")),
        host=str(cfg.get("extract.embeddings.host")),
        prefix=str(cfg.get("extract.embeddings.document_prefix")))
    vectors = np.stack([embeddings.normalise(v)
                        for v in embedder.embed(gaming + baking)]).astype(np.float64)

    window = 3
    depths = []
    for i in range(window, len(vectors) - window + 1):
        before = chapters._centroid(vectors[i - window:i])
        after = chapters._centroid(vectors[i:i + window])
        depths.append(1.0 - float(before @ after))

    # Gap index i compares [i-window, i) against [i, i+window); the seam is
    # between segment 4 and 5, i.e. i == 5.
    seam = 5 - window
    assert int(np.argmax(depths)) == seam, (
        f"strongest split at gap {int(np.argmax(depths))}, seam at {seam}: {depths}")


def test_the_embedding_side_reports_why_when_there_is_no_transcript(tmp_path):
    """The common case on shipped defaults, and it must not look like "no
    topic changes"."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    try:
        conn.execute("INSERT INTO streams (id, date, master_path, duration_s) "
                     "VALUES ('s', '2026-08-18', 'x.mkv', 3600.0)")
        found, reason = chapters.embedding_boundaries(conn, "s", cfg)
        assert found == []
        assert "whisperx" in reason
    finally:
        conn.close()


def test_too_few_segments_to_compare_is_reported_not_crashed(seeded):
    cfg, conn, _ = seeded
    wide = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}",
                                  "digest.chapters.embedding.window=50"])
    found, reason = chapters.embedding_boundaries(conn, "speech", wide)
    assert found == []
    assert "window of 50" in reason


# --------------------------------------------------------------------------
# merging, and the priority the fixture forced
# --------------------------------------------------------------------------


def test_silence_outranks_embedding_inside_one_cluster():
    """FOUND BY THE FIXTURE, and the reason `merge` is not "earliest wins".

    The first cut took the earliest boundary in a cluster. On the speech fixture
    that let a 1.21-sd embedding bump at 29.4 s displace a NINETEEN-SECOND
    silence at 52.0 s — putting the chapter boundary 23 seconds early, in the
    middle of a sentence. Silence is the only one of §9.3's four inputs
    validated on real data and its timestamp means something exact.
    """
    weak = chapters.Boundary(t=29.4, source="embedding", strength=1.21)
    strong = chapters.Boundary(t=52.0, source="silence", strength=19.0)
    survivors = chapters.merge([weak, strong], within_s=30.0)
    assert len(survivors) == 1
    assert survivors[0] is strong
    assert survivors[0].corroborated_by == ["embedding"]


def test_boundaries_further_apart_than_the_window_both_survive():
    a = chapters.Boundary(t=100.0, source="silence", strength=90.0)
    b = chapters.Boundary(t=400.0, source="silence", strength=70.0)
    assert len(chapters.merge([a, b], within_s=120.0)) == 2


def test_scene_changes_never_propose_a_boundary_alone():
    """§9.3 calls them a "weak signal, tie-breaker only" and §16 rejects them as
    a scorer. A tie-breaker that proposes boundaries by itself is just a weak
    detector — and this one runs on a parser that has never seen a real log."""
    scene = chapters.Boundary(t=300.0, source="scene", strength=1.0)
    assert chapters.merge([scene], within_s=120.0) == []

    silence = chapters.Boundary(t=310.0, source="silence", strength=80.0)
    survivors = chapters.merge([scene, silence], within_s=120.0)
    assert [b.source for b in survivors] == ["silence"]
    assert survivors[0].corroborated_by == ["scene"]


def test_merging_is_independent_of_input_order():
    a = chapters.Boundary(t=100.0, source="embedding", strength=2.0)
    b = chapters.Boundary(t=150.0, source="embedding", strength=3.0)
    forward = [x.t for x in chapters.merge([a, b], within_s=120.0)]
    a2 = chapters.Boundary(t=100.0, source="embedding", strength=2.0)
    b2 = chapters.Boundary(t=150.0, source="embedding", strength=3.0)
    backward = [x.t for x in chapters.merge([b2, a2], within_s=120.0)]
    assert forward == backward


# --------------------------------------------------------------------------
# §9.3's target range, as guidance
# --------------------------------------------------------------------------


def test_a_short_stream_is_honestly_one_chapter(seeded):
    """§9.3 targets 10–30 minutes. A 95 s stream cannot have one, and
    fabricating boundaries to pretend otherwise would hand §9.4 invented topic
    changes."""
    cfg, conn, manifest = seeded
    default = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}"])
    result = chapters.segment(conn, "speech", default)
    assert len(result.chapters) == 1
    assert result.unmet_targets
    assert "one chapter is the honest answer" in " ".join(result.unmet_targets)
    _tiles(result, float(manifest["duration_s"]))


def test_an_over_long_chapter_with_no_interior_candidate_is_reported_not_split():
    """Splitting at an invented midpoint would be a fabricated topic change."""
    made = [chapters.Chapter(0, 0.0, 4000.0)]
    cfg = config.load()
    kept, notes = chapters.apply_targets(made, spare=[], cfg=cfg)
    assert len(kept) == 1
    assert any("no boundary was found inside it" in n for n in notes)


# --------------------------------------------------------------------------
# determinism — the lesson from commit 40
# --------------------------------------------------------------------------


def test_the_same_stream_segments_identically_every_run(seeded):
    cfg, conn, _ = seeded
    tuned = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}", *SHORT])
    runs = [[(c.t_start, c.t_end) for c in chapters.segment(conn, "speech", tuned).chapters]
            for _ in range(3)]
    assert all(run == runs[0] for run in runs)


def test_inputs_that_produced_nothing_all_say_why(seeded):
    """The claim this stage has to make honestly. On shipped defaults three of
    §9.3's four produce nothing, and silence alone is not "four sources agreed"."""
    cfg, conn, _ = seeded
    tuned = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}", *SHORT])
    result = chapters.segment(conn, "speech", tuned)

    for name in chapters.INPUTS:
        assert name in result.inputs or name in result.contributing
    for name, reason in result.inputs.items():
        assert reason and len(reason) > 20, f"{name} gave no usable reason"

    # `game` has no producer anywhere in the system and none is invented.
    assert "game" in result.inputs
    assert "Phase 7" in result.inputs["game"]


def test_config_lives_outside_the_versioned_subtrees():
    """A chapter rule must never invalidate a candidate or force a re-score."""
    from clipforge.config import VERSIONED_SUBTREES
    assert "digest" not in VERSIONED_SUBTREES


def test_every_threshold_comes_from_config():
    cfg = config.load()
    for key in ("min_silence_s", "merge_within_s", "target_min_s", "target_max_s",
                "embedding.window", "embedding.prominence_sd"):
        assert cfg.get(f"digest.chapters.{key}") is not None


def test_prominence_is_relative_not_absolute():
    """MEASURED: the distance scale moves 3x between window 1 and window 2, so
    an absolute threshold would mean different things at different settings —
    the trap that kept `min_similarity` out of `search:`."""
    source = Path(chapters.__file__).read_text(encoding="utf-8")
    assert "prominence_sd" in source
    # The name appears in prose explaining why it is absent, so this looks for
    # the thing that would make it real: a config read.
    assert "embedding.min_distance" not in source
    assert config.load().get("digest.chapters.embedding.min_distance", None) is None
