"""§5.10's per-segment embeddings.

Nothing reads these until Phase 6, so the tests are about the properties a
search will later depend on: unit length, one model per row, and the right
vector attached to the right sentence.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge import config, db
from clipforge.extract import embeddings, transcript
from clipforge.extract.transcript import Segment, Word
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext


class FakeEmbedder:
    """Deterministic vectors derived from the text, so similarity is meaningful
    without a model: the same sentence always gives the same direction."""

    model = "fake-embed"

    def __init__(self, dim: int = 8, batch_size_seen: list | None = None) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            # Deliberately NOT unit length, so the stage's normalisation is
            # doing observable work.
            out.append((rng.standard_normal(self.dim) * 5.0).tolist())
        return out


@pytest.fixture
def embedded(tmp_path, speech_fixture_dir, speech_manifest):
    """The speech fixture, transcribed from its manifest, then embedded."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
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

    class FromManifest:
        def transcribe(self, path, *, role, track):
            return [
                Segment(u["t_start"], u["t_end"], u["text"], track, role,
                        [Word(w, None, None, None) for w in u["text"].split()])
                for u in speech_manifest["utterances"] if u["track"] == role
            ]

    transcript.run(ctx, transcriber=FromManifest())
    fake = FakeEmbedder()
    embeddings.run(ctx, embedder=fake)

    yield cfg, conn, ctx, fake
    conn.close()


def vectors_of(conn, stream_id):
    return conn.execute(
        "SELECT e.*, s.seq, s.text FROM segment_embeddings e "
        "JOIN segments s ON s.id = e.segment_id WHERE s.stream_id = ? ORDER BY s.seq",
        (stream_id,),
    ).fetchall()


# --------------------------------------------------------------------------
# vector handling
# --------------------------------------------------------------------------


def test_vectors_are_stored_unit_length(embedded):
    """§5.10's "~50 ms over 200k vectors" is one matmul, which holds only over
    unit vectors. Normalising at query time instead allocates a second 600 MB
    array on every search."""
    _cfg, conn, ctx, _fake = embedded
    for row in vectors_of(conn, ctx.stream_id):
        vector = embeddings.from_blob(row["vec"])
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_normalising_preserves_direction():
    raw = [3.0, 4.0]
    unit = embeddings.normalise(raw)
    assert float(np.linalg.norm(unit)) == pytest.approx(1.0)
    assert unit[0] / unit[1] == pytest.approx(0.75)


def test_a_zero_vector_stays_zero():
    """An all-zero embedding is a model failure, not a direction. Leaving it
    zero means it matches nothing rather than matching everything."""
    assert not embeddings.normalise([0.0, 0.0]).any()


def test_blobs_round_trip(embedded):
    _cfg, conn, ctx, _fake = embedded
    row = vectors_of(conn, ctx.stream_id)[0]
    vector = embeddings.from_blob(row["vec"])
    assert np.allclose(embeddings.from_blob(embeddings.to_blob(vector)), vector)


def test_cosine_is_a_dot_product_over_unit_vectors(embedded):
    """The property §11.6's search will rest on."""
    _cfg, conn, ctx, _fake = embedded
    rows = vectors_of(conn, ctx.stream_id)
    first = embeddings.from_blob(rows[0]["vec"])
    second = embeddings.from_blob(rows[1]["vec"])

    assert embeddings.similarity(first, first) == pytest.approx(1.0, abs=1e-5)
    assert embeddings.similarity(first, second) < 1.0
    assert -1.0001 <= embeddings.similarity(first, second) <= 1.0001


# --------------------------------------------------------------------------
# what goes in the table
# --------------------------------------------------------------------------


def test_every_non_empty_segment_gets_a_vector(embedded, speech_manifest):
    _cfg, conn, ctx, _fake = embedded
    assert len(vectors_of(conn, ctx.stream_id)) == len(speech_manifest["utterances"])


def test_the_model_and_dim_are_recorded(embedded):
    """§5.10 never says why the schema has these. Two models' vectors occupy
    unrelated spaces, so a cosine across them is finite, ordered and
    meaningless — a search has to filter by model rather than compare."""
    _cfg, conn, ctx, fake = embedded
    for row in vectors_of(conn, ctx.stream_id):
        assert row["model"] == fake.model
        assert row["dim"] == fake.dim
        assert embeddings.from_blob(row["vec"]).size == row["dim"]


def test_each_vector_belongs_to_its_own_sentence(embedded):
    """An off-by-one in the batching attaches every embedding to the wrong
    sentence, and nothing downstream could tell."""
    _cfg, conn, ctx, fake = embedded
    for row in vectors_of(conn, ctx.stream_id):
        expected = embeddings.normalise(fake.embed([row["text"]])[0])
        assert np.allclose(embeddings.from_blob(row["vec"]), expected, atol=1e-6)


def test_whitespace_segments_are_skipped(embedded):
    """An embedding of "" is a valid vector pointing somewhere arbitrary, and it
    would match queries — a result with no text, ranked among real ones."""
    _cfg, conn, ctx, fake = embedded
    conn.execute(
        "INSERT INTO segments (stream_id, seq, t_start, t_end, text, track) "
        "VALUES (?, 999, 1.0, 2.0, '   ', 1)", (ctx.stream_id,)
    )
    embeddings.run(ctx, embedder=fake)

    blank = conn.execute(
        "SELECT COUNT(*) FROM segment_embeddings e JOIN segments s ON s.id = e.segment_id "
        "WHERE s.stream_id = ? AND TRIM(s.text) = ''", (ctx.stream_id,)
    ).fetchone()[0]
    assert blank == 0


def test_running_twice_replaces_rather_than_doubling(embedded, speech_manifest):
    _cfg, conn, ctx, fake = embedded
    embeddings.run(ctx, embedder=fake)
    assert len(vectors_of(conn, ctx.stream_id)) == len(speech_manifest["utterances"])


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_texts_are_sent_in_batches(embedded):
    """One HTTP round trip per segment is minutes of latency for milliseconds of
    compute on a 2000-segment stream."""
    cfg, conn, ctx, _fake = embedded
    small = StageContext(
        cfg=config.load(overrides=[
            f"paths.data_root={ctx.cfg.data_root.as_posix()}",
            "extract.embeddings.batch_size=5",
        ]),
        conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
    )
    fake = FakeEmbedder()
    embeddings.run(small, embedder=fake)

    total = sum(len(call) for call in fake.calls)
    assert len(fake.calls) == 3          # 13 segments at 5 per batch
    assert total == 13
    assert max(len(call) for call in fake.calls) <= 5


def test_batched_splits_evenly():
    assert list(embeddings.batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_a_short_reply_is_refused(embedded):
    """Silently dropping one would shift every later vector onto the wrong
    sentence."""
    _cfg, _conn, ctx, _fake = embedded

    class Short:
        model = "x"

        def embed(self, texts):
            return [[1.0, 0.0]] * (len(texts) - 1)

    with pytest.raises(embeddings.EmbeddingError, match="vectors for"):
        embeddings.run(ctx, embedder=Short())


def test_mixed_dimensions_are_refused(embedded):
    _cfg, _conn, ctx, _fake = embedded

    class Ragged:
        model = "x"

        def embed(self, texts):
            return [[1.0, 0.0] if i % 2 else [1.0, 0.0, 0.0]
                    for i in range(len(texts))]

    with pytest.raises(embeddings.EmbeddingError, match="mixed dimensions"):
        embeddings.run(ctx, embedder=Ragged())


# --------------------------------------------------------------------------
# availability and staleness
# --------------------------------------------------------------------------


def test_it_defers_when_the_transcript_is_off(embedded):
    _cfg, _conn, ctx, _fake = embedded
    usable, why = embeddings.available(ctx)
    assert not usable and "enabled" in why


def test_it_defers_when_ollama_is_not_running(embedded, monkeypatch):
    """§5.10's Ollama is a separate process the operator may not have running.
    Failing here would stop `score` and cost the whole run."""
    _cfg, conn, ctx, _fake = embedded
    monkeypatch.setattr(embeddings, "list_models", lambda *a, **k: None)
    on = StageContext(
        cfg=config.load(overrides=[
            f"paths.data_root={ctx.cfg.data_root.as_posix()}",
            "extract.whisperx.enabled=true",
        ]),
        conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
    )
    usable, why = embeddings.available(on)
    assert not usable and "not reachable" in why


def test_it_defers_when_the_model_is_not_pulled(embedded, monkeypatch):
    _cfg, conn, ctx, _fake = embedded
    monkeypatch.setattr(embeddings, "list_models", lambda *a, **k: ["llama3:latest"])
    on = StageContext(
        cfg=config.load(overrides=[
            f"paths.data_root={ctx.cfg.data_root.as_posix()}",
            "extract.whisperx.enabled=true",
        ]),
        conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
    )
    usable, why = embeddings.available(on)
    assert not usable and "ollama pull" in why


def test_a_tagged_model_name_counts_as_present():
    """Ollama reports `nomic-embed-text:latest` for a `nomic-embed-text` pull."""
    assert embeddings.model_present(["nomic-embed-text:latest"], "nomic-embed-text")
    assert not embeddings.model_present(["nomic-embed-text-v2"], "nomic-embed-text")


def test_params_track_the_model_and_prefix_but_not_the_host(embedded):
    """Host and batch size change how the work is fetched, never what comes
    back. The prefix changes the text the model actually sees."""
    _cfg, conn, ctx, _fake = embedded
    base = embeddings.params(ctx)

    def with_override(*overrides):
        return embeddings.params(StageContext(
            cfg=config.load(overrides=[
                f"paths.data_root={ctx.cfg.data_root.as_posix()}", *overrides
            ]),
            conn=conn, stream_id=ctx.stream_id, log=lambda *_: None,
        ))

    assert with_override("extract.embeddings.host=http://elsewhere:1234") == base
    assert with_override("extract.embeddings.batch_size=1") == base
    assert with_override("extract.embeddings.model=other") != base
    assert with_override("extract.embeddings.document_prefix=x") != base


def test_verify_wants_every_segment_covered(embedded):
    _cfg, conn, ctx, _fake = embedded
    assert embeddings.verify(ctx)[0]
    conn.execute(
        "DELETE FROM segment_embeddings WHERE segment_id IN "
        "(SELECT id FROM segments WHERE stream_id = ? AND seq = 0)", (ctx.stream_id,)
    )
    ok, why = embeddings.verify(ctx)
    assert not ok and "no embedding" in why


def test_it_refuses_without_a_transcript(embedded):
    _cfg, conn, ctx, fake = embedded
    conn.execute("DELETE FROM segments WHERE stream_id = ?", (ctx.stream_id,))
    with pytest.raises(embeddings.EmbeddingError, match="no segments"):
        embeddings.run(ctx, embedder=fake)


def test_the_stage_is_registered():
    from clipforge.pipeline import stages

    spec = stages.get("embeddings")
    assert spec.module == "clipforge.extract.embeddings"
    # §5.10 embeds what a sentence MEANS; who said it does not change that.
    assert spec.requires == ("whisperx",)


def test_the_query_prefix_is_recorded_for_the_future_search():
    """Nomic models are trained with paired prefixes. §11.6's search must use
    this one against documents written with `document_prefix`."""
    assert embeddings.QUERY_PREFIX == "search_query: "
    assert config.load().get("extract.embeddings.document_prefix") == "search_document: "


# --------------------------------------------------------------------------
# real Ollama
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def needs_ollama():
    cfg = config.load()
    host = str(cfg.get("extract.embeddings.host"))
    wanted = str(cfg.get("extract.embeddings.model"))
    models = embeddings.list_models(host)
    if models is None:
        pytest.skip(f"ollama not running at {host}")
    if not embeddings.model_present(models, wanted):
        pytest.skip(f"ollama has no {wanted!r} — run `ollama pull {wanted}`")
    return cfg


def test_real_ollama_returns_unit_vectors_of_the_expected_width(needs_ollama):
    cfg = needs_ollama
    engine = embeddings.OllamaEmbedder(
        model=str(cfg.get("extract.embeddings.model")),
        host=str(cfg.get("extract.embeddings.host")),
        prefix=str(cfg.get("extract.embeddings.document_prefix")),
    )
    vectors = engine.embed(["Hawkeye just hit that shot", "I am going Jeff"])
    assert len(vectors) == 2
    dims = {len(v) for v in vectors}
    assert len(dims) == 1
    assert next(iter(dims)) > 100


def test_real_embeddings_put_related_sentences_closer(needs_ollama):
    """The only test that shows the vectors mean anything rather than merely
    existing — and the property §11.6's "that time I did the sad voice" query
    depends on entirely."""
    cfg = needs_ollama
    engine = embeddings.OllamaEmbedder(
        model=str(cfg.get("extract.embeddings.model")),
        host=str(cfg.get("extract.embeddings.host")),
        prefix=str(cfg.get("extract.embeddings.document_prefix")),
    )
    hawkeye_a, hawkeye_b, jeff = (
        embeddings.normalise(v) for v in engine.embed([
            "Hawkeye just hit that shot from across the map.",
            "Hawkeye, one shot, right through the Mantis.",
            "Okay, I'm going Jeff. This is not working.",
        ])
    )
    related = embeddings.similarity(hawkeye_a, hawkeye_b)
    unrelated = embeddings.similarity(hawkeye_a, jeff)
    print(f"\nHawkeye/Hawkeye {related:.3f}  vs  Hawkeye/Jeff {unrelated:.3f}")
    assert related > unrelated
