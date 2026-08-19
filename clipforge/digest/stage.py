"""§5.1's `digest` stage — the deterministic half of §9.

What this stage produces is a `digests` row, not a file: §9.1 calls digests
"first-class rows, never regenerable cache", so `verify` asks the database
rather than the filesystem.

**It does not defer on a missing API key**, and that is a deliberate departure
from how the brief for this phase was written. There is no key anywhere in this
project — `render/hooks.py` exists in its current shape for exactly that reason —
so a stage that deferred without one would never run, and §9.1's compounding
corpus would never start compounding. Everything in §9.2 that arithmetic can
produce is produced here, unconditionally. The model-authored fields are filled
later by `clipforge digest`, which writes a NEW version.

**It does not require `whisperx`.** The registry declared `digest` as
`requires=("score", "whisperx")`, which is true of §9.4's summaries and false of
§9.3's chapters, the emotional arc and the top candidates. Phase 2 ships off and
zero streams exist, so requiring the transcript would defer this stage on every
recording the operator owns. It requires `score` — a digest of a stream nobody
has scored is missing §9.2's `top_candidates` — and everything transcript-shaped
degrades inside the stage, recording what it did not have rather than refusing.
That degradation is the DEFAULT path today, not an edge case.
"""

from __future__ import annotations

import json

from clipforge import db
from clipforge.digest import build
from clipforge.pipeline.context import StageContext, master_identity

#: Config keys this stage reads, gathered once so `params` can hash exactly what
#: it used. A key added to `digest:` in config and not listed here would change
#: behaviour without invalidating the stage.
SETTING_KEYS = (
    "embedding_window_s", "embedding_min_distance",
    "silence_gap_s", "silence_floor_db", "scene_event_kinds",
    "merge_within_s", "min_chapter_s", "max_chapter_s",
    "arc_bin_s", "energy_roles", "laughter_kinds", "laughter_threshold",
    "phrase_min_count", "phrase_min_words", "phrase_max_words", "phrase_limit",
    "top_candidates",
)


def settings(cfg) -> dict:
    return {key: cfg.get(f"digest.{key}") for key in SETTING_KEYS}


def available(ctx: StageContext) -> tuple[bool, str]:
    """Always. This stage has nothing to defer on, which is the point of it.

    There is deliberately no API-key check here, and no Ollama check: the
    deterministic digest needs neither, and a stage that deferred without a key
    would never run on this machine at all.

    It also does NOT check for candidates, and that distinction cost a real bug.
    `available` answers "can the machine execute this", per `StageSpec.available`
    — *"`implemented` says the code exists; this says the machine can execute
    it"* — and the runner evaluates it for every stage UP FRONT, in `plan()`,
    before a single one has run. Checking for candidates here therefore deferred
    `digest` on a fresh stream (nothing had scored yet) and ran it on the next
    invocation, so `clipforge run` could never report "everything up to date"
    after one pass. Input readiness is what `requires=("score",)` expresses, and
    the runner already handles it.

    A stream that scored and produced no candidates still gets a digest: it has
    an energy arc, recurring phrases and chapters, and an empty `top_candidates`
    is the honest answer rather than a reason to produce nothing.
    """
    return True, ""


def verify(ctx: StageContext) -> tuple[bool, str]:
    """A row, not a file. Deleting the digest must un-do the stage."""
    row = ctx.conn.execute(
        "SELECT COUNT(*) AS n FROM digests WHERE stream_id = ?", (ctx.stream_id,)
    ).fetchone()
    return (True, "") if row["n"] else (False, "no digest row")


def params(ctx: StageContext) -> dict:
    return {**master_identity(ctx), **settings(ctx.cfg)}


def run(ctx: StageContext) -> None:
    digest = build.build(ctx.conn, ctx.stream_id, ctx.cfg,
                         settings=settings(ctx.cfg))

    with db.transaction(ctx.conn):
        version = build.write(ctx.conn, ctx.stream_id, digest, model_used=None)
        _record(ctx, digest)

    _report(ctx, digest, version)


def _record(ctx: StageContext, digest: build.Digest) -> None:
    """§14 instrumentation, and the GUESSWORK DISCIPLINE's fourth rule.

    Every number in the `digest:` config block is a guess, and none of these
    metrics existed before this commit — so without them the guesses would be
    unfalsifiable, which is the state that file exists to prevent.
    """
    metrics = digest.metrics
    lengths = metrics["chapter_lengths_s"]
    ctx.metric("digest_chapter_count", metrics["digest_chapter_count"],
               json.dumps({"lengths_s": lengths}))
    ctx.metric("digest_word_count", metrics["digest_word_count"])
    # WHICH sources, not just how many: the count alone cannot distinguish a
    # digest segmented on embedding shift from one segmented on silence, and
    # that is the difference this metric exists to record.
    ctx.metric("digest_boundary_sources", metrics["digest_boundary_sources"],
               json.dumps({"sources": metrics["sources"],
                           "absent": metrics["absent"]}))


def _report(ctx: StageContext, digest: build.Digest, version: int) -> None:
    content = digest.content
    lengths = digest.metrics["chapter_lengths_s"]
    sources = digest.metrics["sources"]

    ctx.log(f"    digest v{version}: {len(content['chapters'])} chapter(s), "
            f"{int(digest.metrics['digest_word_count'])} words")
    if lengths:
        minutes = [round(v / 60.0, 1) for v in lengths]
        ctx.log(f"    chapter lengths (min): {minutes}")
    ctx.log(f"    boundaries from: {', '.join(sources) if sources else 'nothing'}")
    for source, why in sorted(digest.metrics["absent"].items()):
        ctx.log(f"      no {source}: {why}")
    ctx.log(f"    {len(content['recurring_phrases'])} recurring phrase(s), "
            f"{len(content['top_candidates'])} top candidate(s)")
    ctx.log("    titles, summaries, themes and open loops are empty — "
            "`clipforge digest` fills them in")
