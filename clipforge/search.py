"""§11.6's pull search — free text in, segments out.

> **Pull (manual):** free-text query — *"that time I did the sad voice"* —
> embedded and matched by cosine similarity across all segments, 5 months deep.
> Keyword search cannot do this; the operator remembers the vibe, not the words.

That last sentence is the requirement, and it is what the tests assert: a query
sharing no content words with its target still has to find it.

WHAT §5.10 GETS RIGHT, AND THE HALF IT LEAVES OUT

> "Brute-force cosine similarity over 200k vectors in numpy is ~50 ms. **No
> vector database is required.**"

MEASURED at 768 dims, 200k vectors: the matmul is **47 ms**, so §5.10's headline
is right and its conclusion holds. But a query is not a matmul — it is a read
followed by a matmul, and the read dominates it twenty to one:

    load every row, then one matmul     1004 ms      614 MB peak
    stream in 4096-row chunks           * 716 ms *   * 12.6 MB peak *

Streaming is **faster and 49x lighter** — not a trade-off, a win on both axes,
because it never materialises the 600 MB intermediate. So that is what this does,
even though §5.10's phrasing invites the one-big-matrix version. Chunks of 4096
beat 16384 and 65536 on both time and memory.

None of that matters at today's scale: zero real streams exist and the speech
fixture holds thirteen segments. It is built this way because the shape is
cheaper to get right now than to retrofit, and because the numbers say which way
is right rather than leaving it to taste.

THERE IS NO SIMILARITY THRESHOLD, AND THAT IS A MEASUREMENT

The obvious knob is `min_similarity`, and it would be actively harmful. MEASURED
over 65 query-document pairs on the speech fixture, cosine scores span
**0.345 to 0.610** (mean 0.477, sd 0.060) — a band so compressed that a threshold
anywhere inside it keeps roughly half of *everything*, related or not, while
reading like a meaningful relevance filter. Rank is the only thing these scores
support, so rank is all this returns.

TWO THINGS THAT WOULD BE SILENT IF GOT WRONG

**The query prefix.** Nomic models are trained with paired task prefixes:
documents were embedded with `extract.embeddings.document_prefix` and a query
MUST use `QUERY_PREFIX`. Omitting it puts the two in slightly different regions
and degrades retrieval **without erroring** — the worst way for it to be wrong,
which is why `extract/embeddings.py` holds the pair in one place and this imports
it rather than restating it.

**The model filter.** Two models' vectors occupy unrelated spaces, and a cosine
between them is finite, ordered and meaningless. Every query filters by `model`,
so a library embedded with two different models returns results from one of them
rather than a plausible-looking mixture.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from clipforge.extract import embeddings

#: Rows pulled from SQLite per chunk. MEASURED faster and lighter than both
#: larger chunks and a single full load — see the module docstring.
DEFAULT_CHUNK = 4096

#: Results returned when nothing else is asked for.
DEFAULT_LIMIT = 20

#: Decimal places the RANKING is computed at. Not a tunable and deliberately not
#: in config: it exists so that `search.chunk_size` cannot change what a query
#: returns. See the comment in `search()` for the measurement behind it.
_RANK_DECIMALS = 5


class SearchError(RuntimeError):
    pass


class Unavailable(SearchError):
    """The query could not be embedded — no Ollama, or no model.

    Its own type because it is a different message to the operator from "no
    results": one means the search did not run, the other means it ran and found
    nothing, and showing "no results" for both is how a broken search looks
    healthy.
    """


@dataclass
class Hit:
    """One matching segment, with everything a result row needs."""

    segment_id: int
    stream_id: str
    seq: int
    t_start: float
    t_end: float
    text: str
    speaker: str | None
    score: float
    #: Filled in by `attach_candidates`. None means no candidate covers this
    #: moment, which is ordinary — segments and candidates are independent.
    candidate_id: int | None = None
    stream_title: str | None = None
    stream_date: str | None = None

    def to_json(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "stream_id": self.stream_id,
            "stream_title": self.stream_title,
            "stream_date": self.stream_date,
            "seq": self.seq,
            "t_start": round(self.t_start, 2),
            "t_end": round(self.t_end, 2),
            "text": self.text,
            "speaker": self.speaker,
            # Rounded for display only. Never compare it to a threshold: the
            # measured band is 0.345-0.610 and means nothing absolute.
            "score": round(self.score, 4),
            "candidate_id": self.candidate_id,
        }


@dataclass
class Corpus:
    """What is actually searchable, so an empty result can say why."""

    segments: int = 0
    embedded: int = 0
    models: list[str] = field(default_factory=list)
    streams: int = 0

    @property
    def searchable(self) -> bool:
        return self.embedded > 0


def survey(conn: sqlite3.Connection) -> Corpus:
    """Describe the index before searching it.

    Read separately from the search because "no results" and "nothing is
    indexed" are different answers, and on a library processed with shipped
    defaults the second is the common one: §5.7's transcript ships
    `extract.whisperx.enabled: false`, so most streams have no segments at all.
    """
    row = conn.execute("SELECT COUNT(*) FROM segments").fetchone()
    corpus = Corpus(segments=int(row[0]))
    row = conn.execute("SELECT COUNT(*) FROM segment_embeddings").fetchone()
    corpus.embedded = int(row[0])
    corpus.models = [
        str(r[0]) for r in conn.execute(
            "SELECT DISTINCT model FROM segment_embeddings ORDER BY model")
    ]
    corpus.streams = int(conn.execute(
        "SELECT COUNT(DISTINCT s.stream_id) FROM segments s "
        "JOIN segment_embeddings e ON e.segment_id = s.id").fetchone()[0])
    return corpus


def default_model(conn: sqlite3.Connection, cfg) -> str | None:
    """Which embedding space to search.

    The configured model when it is present in the index, otherwise whichever
    one actually has vectors. Falling back matters because changing
    `extract.embeddings.model` does not re-embed the library — the note in
    `extract/embeddings.py` is explicit that it invalidates every stored
    vector — so the config can legitimately name a model nothing was indexed
    with, and searching that returns zero with no explanation.
    """
    corpus = survey(conn)
    wanted = str(cfg.get("extract.embeddings.model", "") or "")
    if wanted and wanted in corpus.models:
        return wanted
    return corpus.models[0] if corpus.models else None


def embed_query(cfg, text: str) -> np.ndarray:
    """The operator's words as a unit vector, with §5.10's query prefix.

    `QUERY_PREFIX` rather than a literal: it is one half of a pair whose other
    half is `extract.embeddings.document_prefix`, and a mismatch degrades
    retrieval without erroring.
    """
    query = str(text or "").strip()
    if not query:
        raise SearchError("empty query")

    embedder = embeddings.OllamaEmbedder(
        model=str(cfg.get("extract.embeddings.model")),
        host=str(cfg.get("extract.embeddings.host", "http://127.0.0.1:11434")),
        prefix=embeddings.QUERY_PREFIX,
        timeout_s=float(cfg.get("extract.embeddings.timeout_s", 120.0)),
    )
    try:
        vectors = embedder.embed([query])
    except (embeddings.EmbeddingError, OSError) as exc:
        raise Unavailable(
            f"could not embed the query: {exc}. §11.6's search needs Ollama "
            f"running with {embedder.model!r} pulled "
            f"(`ollama pull {embedder.model}`)."
        ) from exc
    if not vectors:
        raise Unavailable("the embedder returned nothing for this query")
    return embeddings.normalise(vectors[0])


def search(
    conn: sqlite3.Connection,
    vector: np.ndarray,
    *,
    model: str,
    limit: int = DEFAULT_LIMIT,
    chunk: int = DEFAULT_CHUNK,
    stream_id: str | None = None,
) -> list[Hit]:
    """Top `limit` segments by cosine similarity, streamed in chunks.

    A plain dot product, because both sides are unit length —
    `extract/embeddings.py` normalises on the way in precisely so that §5.10's
    "one matmul" claim holds without allocating a second copy of the index here.

    The running top-K is trimmed with `argpartition` rather than a sort: the
    order within the kept set does not matter until the end, and partitioning is
    linear where sorting is not.
    """
    if vector is None or not len(vector):
        raise SearchError("empty query vector")
    want = max(1, int(limit))

    sql = (
        "SELECT e.segment_id, e.vec, e.dim "
        "  FROM segment_embeddings e "
        "  JOIN segments s ON s.id = e.segment_id "
        # The model filter is not optional: two models' vectors occupy
        # unrelated spaces and a cosine between them is meaningless.
        " WHERE e.model = ?"
    )
    params: list = [model]
    if stream_id:
        sql += " AND s.stream_id = ?"
        params.append(stream_id)

    cursor = conn.execute(sql, params)
    dim = int(vector.size)
    best_scores = np.empty(0, dtype=np.float32)
    best_ids = np.empty(0, dtype=np.int64)
    mismatched = 0

    while True:
        rows = cursor.fetchmany(max(1, int(chunk)))
        if not rows:
            break

        usable = [r for r in rows if int(r[2]) == dim]
        mismatched += len(rows) - len(usable)
        if not usable:
            continue

        matrix = np.frombuffer(
            b"".join(r[1] for r in usable), dtype=embeddings.DTYPE
        ).reshape(len(usable), dim)
        scores = matrix @ vector
        ids = np.fromiter((int(r[0]) for r in usable), dtype=np.int64,
                          count=len(usable))

        best_scores = np.concatenate([best_scores, scores])
        best_ids = np.concatenate([best_ids, ids])
        if best_scores.size > want * 4:
            keep = np.argpartition(-best_scores, want)[:want]
            best_scores, best_ids = best_scores[keep], best_ids[keep]

    if mismatched:
        # Same model name, different dimensionality: impossible unless the model
        # itself changed under one name. Loud, because it is a corrupt index.
        raise SearchError(
            f"{mismatched} vector(s) stored under model {model!r} have a "
            f"different dimensionality from the query ({dim}). The index holds "
            f"two different models under one name; re-run the `embeddings` "
            f"stage for the affected streams."
        )
    if not best_scores.size:
        return []

    # MEASURED, and not what it first looks like. The speech fixture says
    # "Let's go, Hawkeye." three times, so those three segments hold identical
    # vectors -- and their scores came back DIFFERENT across chunk sizes, by
    # about 1e-6:
    #
    #     chunk=1   0.462986  0.462986  0.462986
    #     chunk=2   0.462986  0.462985  0.462985
    #
    # That is not a tie-break bug. `matrix @ vector` dispatches to a different
    # BLAS path depending on how many rows it is handed, and float32
    # accumulation over 768 terms is not associative, so the chunk size changes
    # the arithmetic. Sorting raw scores therefore made the fifth result depend
    # on `search.chunk_size` -- a config value with no business changing what a
    # query returns.
    #
    # So ordering uses scores QUANTISED to `_RANK_DECIMALS`, with `segment_id`
    # breaking what remains. 1e-5 sits comfortably above the observed ~1e-6 of
    # FP noise and far below anything meaningful: the whole measured score band
    # is 0.345-0.610, so this discards 0.004% of the range and buys an ordering
    # that is identical on every machine and every chunk size. The UNROUNDED
    # score is what gets returned and displayed.
    #
    # `lexsort` reads its LAST key as primary: score descending, then id.
    ranking = np.round(best_scores.astype(np.float64), _RANK_DECIMALS)
    order = np.lexsort((best_ids, -ranking))[:want]
    return _hydrate(conn, best_ids[order].tolist(), best_scores[order].tolist())


def _hydrate(conn: sqlite3.Connection, ids: list[int],
             scores: list[float]) -> list[Hit]:
    """One query for the text of every hit, in the ranked order.

    A query per hit would be twenty round trips for a twenty-result page; the
    same reasoning `review/queries.load_candidates` documents for its own
    single-pass load.
    """
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = {
        int(r["id"]): r
        for r in conn.execute(
            f"SELECT s.id, s.stream_id, s.seq, s.t_start, s.t_end, s.text, "
            f"       s.speaker, st.title, st.date "
            f"  FROM segments s "
            f"  JOIN streams st ON st.id = s.stream_id "
            f" WHERE s.id IN ({placeholders})",
            ids,
        )
    }
    hits = []
    for segment_id, score in zip(ids, scores, strict=True):
        row = rows.get(int(segment_id))
        if row is None:
            # An embedding whose segment was deleted. ON DELETE CASCADE should
            # prevent it; skipped rather than crashing a search over it.
            continue
        hits.append(Hit(
            segment_id=int(row["id"]), stream_id=str(row["stream_id"]),
            seq=int(row["seq"]), t_start=float(row["t_start"]),
            t_end=float(row["t_end"]), text=str(row["text"] or ""),
            speaker=row["speaker"], score=float(score),
            stream_title=row["title"], stream_date=row["date"],
        ))
    return hits


def attach_candidates(conn: sqlite3.Connection, hits: list[Hit]) -> list[Hit]:
    """Which current candidate, if any, covers each hit.

    §11.6 returns segments, and segments exist independently of scoring — a
    memorable line is frequently nowhere near a detected peak. So this is
    nullable by design and the UI says "no candidate covers this" rather than
    focusing an unrelated moment, which would misrepresent the result as
    something the detector had found.
    """
    for hit in hits:
        row = conn.execute(
            "SELECT id FROM candidates "
            " WHERE stream_id = ? AND is_current = 1 "
            "   AND t_start <= ? AND t_end >= ? "
            " ORDER BY score_combined DESC LIMIT 1",
            (hit.stream_id, hit.t_start, hit.t_start),
        ).fetchone()
        hit.candidate_id = int(row["id"]) if row else None
    return hits


def run_query(conn: sqlite3.Connection, cfg, text: str, *,
              limit: int | None = None,
              stream_id: str | None = None) -> tuple[list[Hit], Corpus, str | None]:
    """The whole search, as the CLI and the web route both want it.

    Returns the hits, what the index contains, and which model was searched —
    the second and third being what lets an empty result explain itself.
    """
    corpus = survey(conn)
    model = default_model(conn, cfg)
    if not corpus.searchable or model is None:
        return [], corpus, None

    vector = embed_query(cfg, text)
    hits = search(
        conn, vector, model=model,
        limit=int(limit or cfg.get("search.limit", DEFAULT_LIMIT)),
        chunk=int(cfg.get("search.chunk_size", DEFAULT_CHUNK)),
        stream_id=stream_id,
    )
    return attach_candidates(conn, hits), corpus, model
