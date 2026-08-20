"""The written rubric — the learning layer's judgement half.

**This has no § reference. The spec never gave it a section**, and that is the
omission it exists to fix. §17's tuning procedure adjusts weights from
`signal_firing_rate_by_rating` (commit 44b), and §14 calls that the primary
weight-tuning input — but a firing rate cannot carry "the bit only works when
she doesn't see it coming", and no feature vector ever will.

So the primary learning mechanism here is a versioned document the operator
writes in plain language after a review batch, fed into the LLM ranking and
ideation prompts. It works at n=1, where fitted weights need dozens of streams;
it is interpretable six months later, where a weight vector is not; and it
survives a re-score untouched.

THE ONE HARD CONSTRAINT: IT MUST NEVER REACH SCORING

Scoring is deterministic (C1) and `config.VERSIONED_SUBTREES` is
`("extract", "score")`. A rubric feeding scoring would make every re-score
depend on prose and break §6.1's promise that re-scoring is free and infinitely
repeatable — and `config_version` could not describe it, because the text is not
in config at all. `rubric:` is therefore top-level and outside that tuple, and a
test asserts nothing under `clipforge/score/` imports this module.

APPEND-ONLY, AND WHY THE VERSION IS THE IDENTITY

Never UPDATE a row. §9.1 keeps digests forever and never regenerates them, so a
digest made under v3 is a different artifact from one made under v4 and v3 has
to stay readable — `digests.rubric_version` is what records which was in force.

There is deliberately no `rubric_of()` hash beside `prompts.digest_of()`. That
one exists because a prompt template lives in a config file that can be edited
in place, so its identity is its content. A rubric row is immutable by
construction, so the version integer IS its identity and a hash would be a
second name for the same thing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

#: What a prompt is given when nothing has been written yet. An explicit
#: sentence rather than an empty string: a prompt with a blank section under a
#: heading reads as though something went missing, and a model asked to weigh
#: guidance that is not there will invent some.
ABSENT = "No rubric has been written yet — there is no operator guidance to apply."


@dataclass(frozen=True)
class Rubric:
    version: int
    text: str
    created_at: str = ""
    n_streams_at_write: int | None = None
    n_ratings_at_write: int | None = None

    @property
    def evidence(self) -> str:
        """How much stood behind it, for a human reading an old version."""
        if self.n_streams_at_write is None:
            return "unknown corpus"
        return (f"{self.n_streams_at_write} stream(s), "
                f"{self.n_ratings_at_write or 0} rating(s)")


def _row(row: sqlite3.Row) -> Rubric:
    return Rubric(
        version=int(row["version"]),
        text=str(row["text"]),
        created_at=str(row["created_at"] or ""),
        n_streams_at_write=row["n_streams_at_write"],
        n_ratings_at_write=row["n_ratings_at_write"],
    )


def current(conn: sqlite3.Connection) -> Rubric | None:
    """The newest version, or None when nothing has been written."""
    row = conn.execute(
        "SELECT version, text, created_at, n_streams_at_write, n_ratings_at_write "
        "FROM rubrics ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return _row(row) if row else None


def get(conn: sqlite3.Connection, version: int) -> Rubric | None:
    row = conn.execute(
        "SELECT version, text, created_at, n_streams_at_write, n_ratings_at_write "
        "FROM rubrics WHERE version = ?", (int(version),)
    ).fetchone()
    return _row(row) if row else None


def versions(conn: sqlite3.Connection) -> list[Rubric]:
    """Every version, oldest first. The history is the point."""
    return [_row(r) for r in conn.execute(
        "SELECT version, text, created_at, n_streams_at_write, n_ratings_at_write "
        "FROM rubrics ORDER BY version"
    )]


def corpus(conn: sqlite3.Connection) -> tuple[int, int]:
    """`(streams, operator ratings)` — recorded with a new version.

    Ratings are counted across generations and filtered to `'operator'`, the
    same reading `clipforge/moments.py` uses: an inherited copy is not a second
    judgement, and after commit 43 the operator's own row routinely lives on a
    superseded generation.
    """
    streams = conn.execute("SELECT COUNT(*) FROM streams").fetchone()[0]
    ratings = conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE rating_source = 'operator'"
    ).fetchone()[0]
    return int(streams), int(ratings)


def write(conn: sqlite3.Connection, text: str) -> Rubric:
    """Append a new version. Never edits an existing one.

    The caller opens the transaction, the rule every other writer here follows.
    """
    body = text.strip()
    if not body:
        raise ValueError("a rubric with no text is not a version, it is a deletion")

    streams, ratings = corpus(conn)
    version = int(conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM rubrics").fetchone()[0])
    conn.execute(
        "INSERT INTO rubrics (version, text, n_streams_at_write, n_ratings_at_write) "
        "VALUES (?, ?, ?, ?)",
        (version, body, streams, ratings),
    )
    return Rubric(version=version, text=body,
                  n_streams_at_write=streams, n_ratings_at_write=ratings)


def for_prompt(conn: sqlite3.Connection) -> str:
    """What goes into a `$rubric` placeholder.

    Always a non-empty string, because `llm.prompts.render` uses
    `Template.substitute`, which raises on anything unsupplied — deliberately,
    per commit 42, so an unfilled `$name` can never survive into text a model
    then reads.
    """
    latest = current(conn)
    return latest.text if latest else ABSENT


def oversized(rubric: Rubric | None, cfg) -> int:
    """Characters over `rubric.warn_chars`, or 0.

    A WARNING, never a truncation. Every downstream prompt carries this text,
    so an enormous rubric is worth knowing about — but silently dropping the
    end of the operator's own judgement is precisely the failure this feature
    exists to prevent, and §12.4 prices a whole stream at cents, so the tokens
    are not the binding constraint. C2, one layer up from a clip.
    """
    if rubric is None:
        return 0
    limit = int(cfg.get("rubric.warn_chars"))
    return max(0, len(rubric.text) - limit)


def to_markdown(entries: list[Rubric]) -> str:
    """§13.2's tier 2 applied to the rubric.

    "Digests + ideas as plain markdown. Human-readable, survives any schema
    change or full application rewrite. Cheap insurance against the app itself
    becoming the failure point." The judgement calls deserve that at least as
    much as the digests do.
    """
    out = ["# ClipForge rubric", ""]
    for entry in entries:
        out.append(f"## v{entry.version}")
        stamp = entry.created_at or "unknown date"
        out.append(f"*{stamp} — written against {entry.evidence}*")
        out.append("")
        out.append(entry.text.strip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"
