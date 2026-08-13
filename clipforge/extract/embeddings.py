"""§5.1 stage 13 — per-segment embeddings (§5.10).

> Storage math at 384 dims: 100 streams × ~2,000 segments × 384 × 4 bytes ≈
> 300 MB. Brute-force cosine similarity over 200k vectors in numpy is ~50 ms.
> **No vector database is required.**

Nothing reads these yet: §11.6's pull search and §11.1's clustering are Phase 6.
They are written now because §11.1 is explicit — *"Build the logging now (it
cannot be reconstructed); build the clustering pipeline when the corpus
exists"* — and because the decisions below are what make that later search a
short function rather than a migration.

**Vectors are stored L2-normalised.** §5.10's ~50 ms figure is a single matmul,
which is only true over unit vectors; normalising 200k of them at query time
means allocating a second 600 MB array on every search and the claim quietly
stops holding. Nothing needs the magnitude — cosine is the only consumer §5.10,
§11.1 and §11.6 describe — so it is divided out once, here. MEASURED: Ollama's
`/api/embed` already returns unit vectors for this model, so in practice this is
idempotent — but the legacy endpoint and other models make no such promise, and
a search that assumes unit length has to be right about it.

**The model matters more than §5.10 implies.** It names `bge-small-en-v1.5`,
which Ollama's library does not ship under that name; the default here is
§5.10's own second choice, `nomic-embed-text` (768-dim). Two models' vectors
occupy unrelated spaces, so `segment_embeddings.model` and `.dim` are recorded
per row and **a search must filter by model rather than compare across them** —
a cosine between two geometries is finite, ordered, and meaningless.

**Nomic models are trained with task prefixes.** Indexed text gets
`search_document: ` and a query gets `search_query: `. Omitting them puts the
two in slightly different regions and degrades retrieval without erroring, which
is the worst way for it to be wrong. The document prefix is config and is
stamped into `params_hash`; §11.6's search must use the query prefix in
`QUERY_PREFIX` below.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from clipforge import db
from clipforge.pipeline.context import StageContext

#: What §11.6's pull search must prepend to the operator's query text, to match
#: the `document_prefix` these vectors were written with. Paired values: change
#: one in config and this has to change with it.
QUERY_PREFIX = "search_query: "

#: float32, matching `signals.store`'s convention. §5.10's storage math assumes
#: it; float64 would double a 600 MB index for no retrieval benefit.
DTYPE = np.float32


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def model(self) -> str: ...


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


def _post(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def list_models(host: str, timeout_s: float = 5.0) -> list[str] | None:
    """Models Ollama has pulled, or None when it is not reachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout_s) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None
    return [str(entry.get("name", "")) for entry in data.get("models") or []]


def model_present(available: list[str], wanted: str) -> bool:
    """Ollama reports `nomic-embed-text:latest` for a `nomic-embed-text` pull."""
    return any(name == wanted or name.startswith(f"{wanted}:") for name in available)


@dataclass
class OllamaEmbedder:
    """§5.10's local embedder. No dependency: Ollama is HTTP and urllib is stdlib."""

    model: str
    host: str = "http://127.0.0.1:11434"
    prefix: str = ""
    timeout_s: float = 120.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [f"{self.prefix}{text}" for text in texts]
        try:
            return self._batch(prefixed)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise EmbeddingError(f"ollama returned {exc.code}: {exc.reason}") from exc
            # An Ollama old enough not to have /api/embed. One request per text,
            # which is why it is the fallback rather than the default.
            return [self._single(text) for text in prefixed]
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise EmbeddingError(f"cannot reach ollama at {self.host}: {exc}") from exc

    def _batch(self, texts: list[str]) -> list[list[float]]:
        data = _post(
            f"{self.host}/api/embed",
            {"model": self.model, "input": texts},
            self.timeout_s,
        )
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise EmbeddingError(
                f"ollama returned {len(vectors or [])} vectors for {len(texts)} inputs"
            )
        return vectors

    def _single(self, text: str) -> list[float]:
        data = _post(
            f"{self.host}/api/embeddings",
            {"model": self.model, "prompt": text},
            self.timeout_s,
        )
        vector = data.get("embedding")
        if not vector:
            raise EmbeddingError(f"ollama returned no embedding for {text[:40]!r}")
        return vector


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------


def normalise(vector) -> np.ndarray:
    """Divide out the magnitude, so cosine similarity is a dot product."""
    array = np.asarray(vector, dtype=DTYPE)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        # An all-zero embedding is a model failure, not a direction. Leaving it
        # zero means it matches nothing rather than matching everything.
        return array
    return (array / norm).astype(DTYPE)


def to_blob(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype=DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — a plain dot product, because both are unit length."""
    return float(np.dot(a, b))


def batched(items: list, size: int):
    for start in range(0, len(items), max(size, 1)):
        yield items[start:start + size]


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _cfg(ctx: StageContext, key: str, default=None):
    return ctx.cfg.get(f"extract.embeddings.{key}", default)


def params(ctx: StageContext) -> dict:
    """What would change the vectors.

    `host`, `batch_size` and `timeout_s` are absent: they change how the work is
    fetched, never what comes back. The prefix is present because it changes the
    text the model actually sees.
    """
    return {
        "model": _cfg(ctx, "model"),
        "document_prefix": _cfg(ctx, "document_prefix"),
        "normalise": _cfg(ctx, "normalise"),
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # segment_embeddings rows


def available(ctx: StageContext) -> tuple[bool, str]:
    """Defer rather than fail — see `StageSpec.available`.

    §5.10's Ollama is a separate process the operator may simply not have
    running. Failing here would stop `score` and cost the whole run.
    """
    from clipforge.extract import transcript

    usable, why = transcript.available(ctx)
    if not usable:
        return False, why
    if not bool(_cfg(ctx, "enabled", True)):
        return False, "disabled (extract.embeddings.enabled)"

    host = str(_cfg(ctx, "host"))
    models = list_models(host)
    if models is None:
        return False, f"ollama is not reachable at {host}"
    wanted = str(_cfg(ctx, "model"))
    if not model_present(models, wanted):
        return False, f"ollama has no model {wanted!r} (ollama pull {wanted})"
    return True, ""


def verify(ctx: StageContext) -> tuple[bool, str]:
    missing = ctx.conn.execute(
        """
        SELECT COUNT(*) FROM segments s
         WHERE s.stream_id = ? AND TRIM(s.text) != ''
           AND NOT EXISTS (SELECT 1 FROM segment_embeddings e WHERE e.segment_id = s.id)
        """,
        (ctx.stream_id,),
    ).fetchone()[0]
    if missing:
        return False, f"{missing} segment(s) have no embedding"

    total = ctx.conn.execute(
        "SELECT COUNT(*) FROM segments WHERE stream_id = ?", (ctx.stream_id,)
    ).fetchone()[0]
    if not total:
        return False, "no segments to embed (has whisperx run?)"
    return True, ""


def build_embedder(ctx: StageContext) -> OllamaEmbedder:
    return OllamaEmbedder(
        model=str(_cfg(ctx, "model")),
        host=str(_cfg(ctx, "host")),
        prefix=str(_cfg(ctx, "document_prefix", "")),
        timeout_s=float(_cfg(ctx, "timeout_s", 120.0)),
    )


@dataclass
class Stored:
    count: int = 0
    dim: int = 0
    skipped: int = 0
    dims_seen: set = field(default_factory=set)


def run(ctx: StageContext, embedder: Embedder | None = None) -> None:
    rows = ctx.conn.execute(
        "SELECT id, seq, text FROM segments WHERE stream_id = ? ORDER BY seq",
        (ctx.stream_id,),
    ).fetchall()
    if not rows:
        raise EmbeddingError(
            "no segments to embed. whisperx has not run, or produced nothing."
        )

    # An embedding of "" is a valid vector pointing somewhere arbitrary, and it
    # would match queries — a result with no text, ranked among real ones.
    usable = [row for row in rows if (row["text"] or "").strip()]
    result = Stored(skipped=len(rows) - len(usable))

    engine = embedder or build_embedder(ctx)
    should_normalise = bool(_cfg(ctx, "normalise", True))
    batch_size = int(_cfg(ctx, "batch_size", 32))
    model = str(getattr(engine, "model", _cfg(ctx, "model")))

    pending: list[tuple[int, bytes, int]] = []
    for chunk in batched(usable, batch_size):
        ctx.heartbeat()
        vectors = engine.embed([row["text"] for row in chunk])
        if len(vectors) != len(chunk):
            # An off-by-one here attaches every embedding to the wrong sentence
            # and nothing downstream could tell.
            raise EmbeddingError(
                f"embedder returned {len(vectors)} vectors for {len(chunk)} segments"
            )
        for row, vector in zip(chunk, vectors, strict=True):
            array = normalise(vector) if should_normalise else np.asarray(vector, DTYPE)
            result.dims_seen.add(int(array.size))
            pending.append((row["id"], to_blob(array), int(array.size)))

    if len(result.dims_seen) > 1:
        raise EmbeddingError(
            f"the embedder returned mixed dimensions {sorted(result.dims_seen)}; "
            f"a cosine across two widths is not meaningful"
        )

    with db.transaction(ctx.conn):
        ctx.conn.execute(
            "DELETE FROM segment_embeddings WHERE segment_id IN "
            "(SELECT id FROM segments WHERE stream_id = ?)",
            (ctx.stream_id,),
        )
        ctx.conn.executemany(
            "INSERT INTO segment_embeddings (segment_id, model, dim, vec) "
            "VALUES (?, ?, ?, ?)",
            [(sid, model, dim, blob) for sid, blob, dim in pending],
        )

    result.count = len(pending)
    result.dim = next(iter(result.dims_seen), 0)
    _report(ctx, result, model)


def _report(ctx: StageContext, result: Stored, model: str) -> None:
    megabytes = result.count * result.dim * 4 / 1_000_000
    ctx.log(
        f"    {result.count} embedding(s), {result.dim}-dim, {model} "
        f"({megabytes:.2f} MB)"
    )
    ctx.metric("segment_embeddings", float(result.count))
    if result.skipped:
        ctx.log(f"    skipped {result.skipped} empty segment(s)")
