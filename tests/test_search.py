"""§11.6's pull search.

> "Keyword search cannot do this; the operator remembers the vibe, not the
> words."

That is the claim, so the retrieval tests use queries that share **no content
words** with their targets. A test that searched for "Hawkeye" and found the line
containing "Hawkeye" would pass against a `LIKE` query and prove nothing.

**Segments are seeded from the fixture manifest, not transcribed.** Running
WhisperX to test *search* would couple two unrelated things and let ASR errors
decide whether retrieval passes — and it would make this file depend on a
multi-GB model download. The manifest's `utterances` are the authored ground
truth; embedding those measures the search and nothing else.

Retrieval assertions are **recall@3**, never recall@1 and never a similarity
number. Both limits are measured rather than chosen — see the module docstring
of `clipforge/search.py` and the `search` section of `spec/GUESSES.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clipforge import config, db, search
from clipforge.extract import embeddings
from clipforge.pipeline.context import StageContext

FIXTURE = Path(__file__).parent / "fixtures" / "_generated" / "speech"

#: query -> the authored line it must retrieve. Every query is deliberately
#: written to share no content words with its target: that is what separates
#: §11.6's semantic search from the keyword search it says cannot do this.
PROBES = {
    "an enemy is attacking the people behind us":
        "Iron Fist is diving our backline again, watch out.",
    "I am amazed it succeeded":
        "I can't believe that actually worked.",
    "switching to a different character":
        "Okay, I'm going Jeff. This is not working.",
    "something is killing me":
        "Namor's turrets are melting me, holy shit.",
    "keeping my teammate alive":
        "I'm on Mantis, I'll try to heal you through it.",
}

#: MEASURED: 4 of the 5 probes above put their target at rank 1 and one puts it
#: at rank 3, out of 13 segments. Asserting rank 1 would mean either a flaky
#: test or a probe set trimmed until it passed, which is the failure house rule
#: 6 exists to prevent. Rank 3 of 13 is the top 23% and is a real claim.
RECALL_AT = 3


def _manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))


def _ollama_ready(cfg) -> bool:
    models = embeddings.list_models(str(cfg.get("extract.embeddings.host")))
    if models is None:
        return False
    return embeddings.model_present(models, str(cfg.get("extract.embeddings.model")))


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    """The manifest's authored lines, embedded and searchable.

    Skips rather than fails when Ollama is not running: §5.10 puts embeddings on
    a local model by design, and a machine without one should not report a
    broken search.
    """
    if not FIXTURE.is_dir():
        pytest.skip("speech fixture not generated")

    root = tmp_path_factory.mktemp("search")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    if not _ollama_ready(cfg):
        pytest.skip("ollama is not running with the configured embedding model")

    conn = db.open_db(cfg.db_path)
    manifest = _manifest()
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, duration_s) "
        "VALUES ('speech', '2026-08-17', 'Speech fixture', ?, ?)",
        (str(FIXTURE / manifest["video"]), float(manifest["duration_s"])),
    )
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

    ctx = StageContext(cfg=cfg, conn=conn, stream_id="speech", log=lambda _m: None)
    embeddings.run(ctx)
    yield cfg, conn, manifest
    conn.close()


# --------------------------------------------------------------------------
# what §11.6 actually claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", list(PROBES), ids=lambda q: q[:28])
def test_a_query_sharing_no_words_with_its_target_still_finds_it(indexed, query):
    """§11.6: "the operator remembers the vibe, not the words".

    Asserted at recall@3 of 13 segments — see RECALL_AT for why not rank 1.
    """
    cfg, conn, _ = indexed
    hits, corpus, model = search.run_query(conn, cfg, query, limit=RECALL_AT)
    assert corpus.searchable and model
    assert PROBES[query] in [h.text for h in hits], (
        f"{query!r} did not retrieve its target within the top {RECALL_AT}"
    )


def test_the_probes_share_no_content_words_with_their_targets(indexed):
    """Guards the tests above from becoming keyword tests by accident.

    If a probe is ever reworded to share a distinctive word with its target, the
    retrieval assertion would start passing for the wrong reason.
    """
    stop = {"a", "an", "the", "is", "are", "was", "i", "am", "my", "me", "we",
            "us", "our", "it", "its", "to", "of", "and", "in", "on", "at", "that",
            "this", "with", "for", "you", "your", "they", "them", "he", "she",
            "did", "do", "does", "not", "no", "so", "up", "out", "just", "still"}
    for query, target in PROBES.items():
        q = {w.strip(".,!?'\"").lower() for w in query.split()} - stop
        t = {w.strip(".,!?'\"").lower() for w in target.split()} - stop
        assert not (q & t), f"{query!r} shares {q & t} with its target"


# --------------------------------------------------------------------------
# the chunked implementation must not change the answer
# --------------------------------------------------------------------------


def test_chunking_returns_exactly_what_a_single_matmul_would(indexed):
    """The optimisation §5.10's phrasing does not suggest, held to the obvious
    implementation's answer.

    Streaming in chunks is faster and 49x lighter (see clipforge/search.py), and
    it would be worth nothing if it quietly reordered results.
    """
    cfg, conn, _ = indexed
    vector = search.embed_query(cfg, "an amazing shot")
    model = search.default_model(conn, cfg)

    # The one-big-matrix version, written out here rather than shipped.
    rows = conn.execute(
        "SELECT segment_id, vec FROM segment_embeddings WHERE model = ? "
        "ORDER BY segment_id", (model,)).fetchall()
    matrix = np.frombuffer(b"".join(r["vec"] for r in rows),
                           dtype=embeddings.DTYPE).reshape(len(rows), vector.size)
    scores = matrix @ vector
    ids = np.array([int(r["segment_id"]) for r in rows], dtype=np.int64)
    # Same tie-break the implementation uses: score descending, then id.
    expected = ids[np.lexsort((ids, -scores))][:5].tolist()

    for chunk in (1, 2, 3, 5, 4096):
        got = [h.segment_id for h in
               search.search(conn, vector, model=model, limit=5, chunk=chunk)]
        assert got == expected, f"chunk={chunk} reordered the results"


def test_tied_scores_break_deterministically(indexed):
    """FOUND BY THE TEST ABOVE, and it is a real property rather than a fixture
    quirk.

    Identical text produces a bit-identical embedding and therefore a
    bit-identical score. The speech fixture says "Let's go, Hawkeye." three
    times; a real stream repeats a catchphrase far more, which is the whole
    premise of §5.4.2's `phrase_repeat`. Without an explicit tie-break the
    ordering of those falls out of wherever the chunk boundaries landed, so the
    same query returns a different fifth result depending on
    `search.chunk_size` -- results shuffling for no reason the operator can see.
    """
    cfg, conn, _ = indexed
    model = search.default_model(conn, cfg)
    vector = search.embed_query(cfg, "an amazing shot")

    runs = [[h.segment_id for h in
             search.search(conn, vector, model=model, limit=8, chunk=c)]
            for c in (1, 2, 3, 5, 7, 4096)]
    assert all(run == runs[0] for run in runs), (
        f"chunk size changed the ordering: {runs}")

    # ...and there really are ties here, or this asserts nothing.
    hits = search.search(conn, vector, model=model, limit=13)
    scores = [round(h.score, 6) for h in hits]
    assert len(scores) != len(set(scores)), "no tied scores; test proves nothing"


def test_the_limit_is_honoured(indexed):
    cfg, conn, manifest = indexed
    model = search.default_model(conn, cfg)
    vector = search.embed_query(cfg, "anything at all")
    for limit in (1, 3, 5):
        assert len(search.search(conn, vector, model=model, limit=limit)) == limit
    # Asking for more than exists returns everything, not an error.
    everything = search.search(conn, vector, model=model, limit=1000)
    assert len(everything) == len(manifest["utterances"])


def test_results_are_ranked_by_descending_score(indexed):
    """Descending at the precision the RANKING uses, which is the actual contract.

    Not at full float precision, and that is deliberate rather than sloppy. Two
    segments here score 0.5009792447 and 0.5009793043 — apart at the seventh
    decimal, which is float32 matmul noise rather than a difference in meaning.
    Both quantise to one rank and `segment_id` orders them, so the raw scores
    come back very slightly out of order. Asserting strict descending on the raw
    value would be asserting that `search.chunk_size` changes results, which is
    the exact property `_RANK_DECIMALS` exists to remove.
    """
    cfg, conn, _ = indexed
    hits, _corpus, _model = search.run_query(conn, cfg, "a great play", limit=10)
    ranked = [round(h.score, search._RANK_DECIMALS) for h in hits]
    assert ranked == sorted(ranked, reverse=True)

    # Within one rank, ties order by segment id — so the sequence is fully
    # determined and two runs cannot disagree.
    for (a_score, a_id), (b_score, b_id) in zip(
            [(r, h.segment_id) for r, h in zip(ranked, hits)],
            [(r, h.segment_id) for r, h in zip(ranked, hits)][1:], strict=False):
        if a_score == b_score:
            assert a_id < b_id


# --------------------------------------------------------------------------
# the model filter — the silent-failure guard
# --------------------------------------------------------------------------


def test_a_search_never_crosses_two_embedding_models(indexed):
    """Two models' vectors occupy unrelated spaces; a cosine between them is
    finite, ordered and meaningless. `extract/embeddings.py` says so, and this
    is the enforcement."""
    cfg, conn, manifest = indexed
    model = search.default_model(conn, cfg)
    vector = search.embed_query(cfg, "a great play")
    real = len(search.search(conn, vector, model=model, limit=100))
    assert real == len(manifest["utterances"])
    # A model nobody indexed with returns nothing rather than everything.
    assert search.search(conn, vector, model="some-other-model", limit=100) == []


def test_mixed_dimensionality_under_one_model_name_is_refused(indexed):
    """The one corrupt-index case that cannot be silently skipped: the same
    model name holding two geometries."""
    cfg, conn, _ = indexed
    model = search.default_model(conn, cfg)
    vector = search.embed_query(cfg, "a great play")
    segment_id = conn.execute(
        "SELECT segment_id FROM segment_embeddings LIMIT 1").fetchone()[0]
    original = conn.execute(
        "SELECT vec, dim FROM segment_embeddings WHERE segment_id = ?",
        (segment_id,)).fetchone()
    conn.execute(
        "UPDATE segment_embeddings SET dim = ?, vec = ? WHERE segment_id = ?",
        (384, np.zeros(384, dtype=embeddings.DTYPE).tobytes(), segment_id))
    try:
        with pytest.raises(search.SearchError, match="dimensionality"):
            search.search(conn, vector, model=model, limit=5)
    finally:
        conn.execute(
            "UPDATE segment_embeddings SET dim = ?, vec = ? WHERE segment_id = ?",
            (original["dim"], original["vec"], segment_id))


def test_the_query_uses_the_paired_task_prefix():
    """Nomic models are trained with paired prefixes and omitting one degrades
    retrieval WITHOUT erroring — the worst way for it to be wrong.

    Structural: the query side must come from `extract/embeddings.py`, which is
    where the document side is configured, rather than being restated here.
    """
    source = Path(search.__file__).read_text(encoding="utf-8")
    assert "QUERY_PREFIX" in source
    assert "search_query: " not in source, (
        "the query prefix is written out in search.py; it must come from "
        "embeddings.QUERY_PREFIX so the pair cannot drift")


# --------------------------------------------------------------------------
# scoping and the candidate link
# --------------------------------------------------------------------------


def test_a_search_can_be_scoped_to_one_stream(indexed):
    cfg, conn, manifest = indexed
    model = search.default_model(conn, cfg)
    vector = search.embed_query(cfg, "a great play")
    assert len(search.search(conn, vector, model=model, limit=100,
                             stream_id="speech")) == len(manifest["utterances"])
    assert search.search(conn, vector, model=model, limit=100,
                         stream_id="nonexistent") == []


def test_a_segment_with_no_candidate_over_it_says_so(indexed):
    """§11.6 returns SEGMENTS, and they exist independently of scoring. A
    memorable line is frequently nowhere near a detected peak, so this is
    nullable by design rather than a gap to be filled with the nearest match."""
    cfg, conn, _ = indexed
    hits, _corpus, _model = search.run_query(conn, cfg, "a great play", limit=5)
    assert hits
    assert all(h.candidate_id is None for h in hits), (
        "nothing has been scored in this fixture, so nothing should claim a "
        "candidate covers it")


def test_a_candidate_covering_a_hit_is_reported(indexed):
    cfg, conn, _ = indexed
    hit = search.run_query(conn, cfg, "a great play", limit=1)[0][0]
    conn.execute(
        "INSERT INTO candidates (stream_id, generation, is_current, profile, "
        " t_start, t_end, t_peak, score_entertainment, score_gameplay, "
        " score_combined, feature_vector, feature_schema_version, config_version) "
        "VALUES ('speech', 1, 1, 'test', ?, ?, ?, 0, 0, 1.0, '{}', 1, 'test@0')",
        (max(hit.t_start - 1.0, 0.0), hit.t_start + 1.0, hit.t_start),
    )
    try:
        again = search.run_query(conn, cfg, "a great play", limit=1)[0][0]
        assert again.candidate_id is not None
    finally:
        conn.execute("DELETE FROM candidates WHERE stream_id = 'speech'")


# --------------------------------------------------------------------------
# empty states — three causes, three answers
# --------------------------------------------------------------------------


def test_an_unindexed_library_is_distinguished_from_no_results(tmp_path):
    """"No results" for an empty index is how a broken search looks healthy —
    and on shipped defaults an empty index is the COMMON case, because §5.7's
    transcription is off."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    try:
        hits, corpus, model = search.run_query(conn, cfg, "anything")
        assert hits == []
        assert not corpus.searchable
        assert model is None
        # ...and it never reached Ollama, so this works with nothing installed.
        assert corpus.segments == 0
    finally:
        conn.close()


def test_segments_without_embeddings_are_reported_separately(tmp_path):
    """A transcribed but un-embedded library needs a different sentence again:
    the fix is running one stage, not turning on transcription."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    try:
        conn.execute("INSERT INTO streams (id, date, master_path) "
                     "VALUES ('s', '2026-08-18', 'x.mkv')")
        conn.execute("INSERT INTO segments (stream_id, seq, t_start, t_end, text) "
                     "VALUES ('s', 1, 0.0, 1.0, 'hello')")
        corpus = search.survey(conn)
        assert corpus.segments == 1
        assert corpus.embedded == 0
        assert not corpus.searchable
    finally:
        conn.close()


def test_an_unreachable_embedder_raises_its_own_type(tmp_path):
    """Distinct from SearchError so the caller can say "the search did not run"
    rather than "nothing matched"."""
    cfg = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'd').as_posix()}",
        "extract.embeddings.host=http://127.0.0.1:1",
    ])
    with pytest.raises(search.Unavailable):
        search.embed_query(cfg, "anything")


def test_an_empty_query_is_refused(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    for text in ("", "   "):
        with pytest.raises(search.SearchError):
            search.embed_query(cfg, text)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_search_config_is_outside_the_versioned_subtrees():
    """A search parameter must never invalidate a candidate or force a re-score.
    Same reasoning that put `render:` and `previews:` outside."""
    from clipforge.config import VERSIONED_SUBTREES
    assert "search" not in VERSIONED_SUBTREES


def test_there_is_no_similarity_threshold():
    """MEASURED: cosine scores span 0.345-0.610 over unrelated pairs, so a
    threshold anywhere in that band keeps roughly half of everything while
    reading like a relevance filter. This asserts the knob stays absent."""
    cfg = config.load()
    assert cfg.get("search.min_similarity", None) is None
    # The name appears in prose in both files explaining why it is absent, so
    # this looks for the thing that would make it real: a config read.
    source = Path(search.__file__).read_text(encoding="utf-8")
    assert "search.min_similarity" not in source


def test_the_defaults_come_from_config_not_literals():
    cfg = config.load()
    assert int(cfg.get("search.limit")) > 0
    assert int(cfg.get("search.chunk_size")) > 0
    assert int(cfg.get("search.snippet_chars")) > 0


# --------------------------------------------------------------------------
# the HTTP surface
# --------------------------------------------------------------------------


def _client(cfg):
    from fastapi.testclient import TestClient

    from clipforge.review import app as review_app
    return TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1",
                      headers={"X-ClipForge": "1"})


def test_the_corpus_route_answers_before_anything_is_typed(indexed):
    """The screen has to be able to say "nothing is indexed" up front rather
    than looking ready and explaining after the first query."""
    cfg, _conn, manifest = indexed
    with _client(cfg) as client:
        body = client.get("/api/search/corpus").json()
    assert body["searchable"] is True
    assert body["embedded"] == len(manifest["utterances"])
    assert body["model"]


def test_the_search_route_returns_ranked_hits(indexed):
    cfg, _conn, _ = indexed
    with _client(cfg) as client:
        body = client.get("/api/search",
                          params={"q": "an enemy is attacking the people behind us",
                                  "limit": 3}).json()
    assert len(body["hits"]) == 3
    assert body["hits"][0]["text"] == PROBES[
        "an enemy is attacking the people behind us"]
    for hit in body["hits"]:
        assert {"segment_id", "stream_id", "t_start", "text", "candidate_id"} <= set(hit)


def test_an_unreachable_embedder_is_a_503_not_a_500(indexed, tmp_path):
    """The search did not run, and the operator can fix it by starting Ollama.
    A 500 sends them to the logs instead."""
    cfg, _conn, _ = indexed
    broken = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "extract.embeddings.host=http://127.0.0.1:1",
    ])
    with _client(broken) as client:
        response = client.get("/api/search", params={"q": "anything"})
    assert response.status_code == 503
    assert "ollama" in response.text.lower()


def test_an_empty_query_is_a_400(indexed):
    cfg, _conn, _ = indexed
    with _client(cfg) as client:
        assert client.get("/api/search", params={"q": "   "}).status_code == 400
