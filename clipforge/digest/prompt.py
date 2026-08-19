"""§9.4's map-reduce, and §12's rules applied to whatever comes back.

> **Map:** for each chapter independently, send the chapter transcript (with
> segment IDs) to the LLM. Request a 3–4 sentence summary, notable segment IDs,
> observed themes, and open loops.
>
> **Reduce:** combine the chapter outputs plus deterministically-computed
> statistics into the final digest JSON.

**One prompt, every chapter as its own section.** Map-reduce structurally — each
chapter is described from its own transcript, its ids scoped to itself, with no
claim allowed to cross a boundary — but ONE round trip, because there is no API
key and the transport is a person pasting into a browser. C8 budgets ~35 minutes
of hands-on time per stream and `render/hooks.py` already made this exact call
for §8.5: *"five paste round trips would spend a chunk of it on clerical work."*
Eighteen would be worse. When an API-backed source lands it can fan these same
per-chapter sections out as genuinely parallel calls; the sections are built
separately here precisely so that it can.

**The prompt is pinned to a digest version**, and `--apply` refuses a reply built
against a different one. Segment ids are scoped per chapter, so if the segmenter
re-runs between prompt and reply — a config change, a late `whisperx` — every id
in the reply silently points into the wrong chapter. There is no way to detect
that from the reply itself, so the version rides in the prompt and is checked.

**The reduce step writes a NEW digest version.** §9.1: *"first-class rows, never
regenerable cache. Keep every version forever."* v1 is the deterministic digest;
applying a reply produces v2 carrying v1's chapters with the model's fields
filled in. Nothing is ever updated in place.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from clipforge import db, llm
from clipforge.config import CONFIG_DIR

#: §9.4 asks for "a 3-4 sentence summary".
SENTENCES = "3-4"

#: `kind` values §3.2's `open_loops` column accepts.
LOOP_KINDS = frozenset({"promise", "question", "unsolved"})


class DigestPromptError(RuntimeError):
    pass


@dataclass
class ChapterView:
    """One chapter as the model is shown it."""

    index: int
    t_start: float
    t_end: float
    #: `(seq, text)` in order. `seq` is `segments.seq` — §12.1's LLM-facing id.
    lines: list[tuple[int, str]] = field(default_factory=list)

    @property
    def seqs(self) -> set[int]:
        return {seq for seq, _text in self.lines}

    @property
    def transcript(self) -> str:
        return "\n".join(f"[{seq}] {text}" for seq, text in self.lines)


@dataclass
class ChapterResult:
    """One chapter's accepted output."""

    index: int
    title: str | None
    summary: str | None
    themes: list[str] = field(default_factory=list)
    notable_segment_ids: list[int] = field(default_factory=list)
    quote: str | None = None
    open_loops: list[dict] = field(default_factory=list)


@dataclass
class Validated(llm.Validation):
    """What survived §12's checks."""

    chapters: list[ChapterResult] = field(default_factory=list)


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def load_prompts(cfg) -> dict:
    path = Path(cfg.get("digest.prompts_file", "digest_prompts.yaml"))
    if not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.is_file():
        raise DigestPromptError(f"digest prompt file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in
               ("intro", "rules", "fields", "chapters_header", "no_transcript")
               if not str(data.get(key) or "").strip()]
    if missing:
        raise DigestPromptError(
            f"{path} is missing: {', '.join(missing)}")
    return data


def chapter_views(conn: sqlite3.Connection, stream_id: str,
                  content: dict) -> list[ChapterView]:
    """The chapters of a stored digest, with their transcript lines.

    Read from the DIGEST's chapters rather than re-segmenting: the reply will be
    validated against these ids, and a boundary that moved in between would
    re-point every one of them.
    """
    views: list[ChapterView] = []
    for chapter in content.get("chapters") or []:
        rows = conn.execute(
            "SELECT seq, text FROM segments WHERE stream_id = ? "
            "AND t_end >= ? AND t_start < ? AND TRIM(text) != '' ORDER BY seq",
            (stream_id, chapter["t_start"], chapter["t_end"]),
        ).fetchall()
        views.append(ChapterView(
            index=int(chapter["index"]),
            t_start=float(chapter["t_start"]),
            t_end=float(chapter["t_end"]),
            lines=[(int(r["seq"]), str(r["text"]).strip()) for r in rows],
        ))
    return views


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_prompt(cfg, content: dict, version: int,
                 views: list[ChapterView]) -> str:
    """One prompt for the whole stream, one section per chapter."""
    if not views:
        raise DigestPromptError("this digest has no chapters")

    prompts = load_prompts(cfg)
    games = content.get("games") or []
    example_seq = next((seq for view in views for seq, _t in view.lines), 0)

    lines = [
        prompts["intro"].format(
            stream_id=content["stream_id"],
            date=content.get("date") or "undated",
            games=f", {', '.join(games)}" if games else "",
            duration=_clock(content.get("duration_s") or 0),
            chapter_count=len(views),
            sentences=SENTENCES,
        ).rstrip(),
        "",
        # NOT decoration: --apply refuses a reply whose version does not match,
        # because segment ids are scoped per chapter and a re-segmentation
        # between prompt and reply would silently re-point every one of them.
        f"digest_version: {version}",
        "",
        prompts["rules"].rstrip(),
        "",
        prompts["fields"].format(sentences=SENTENCES).rstrip(),
        "",
        "```json",
        json.dumps({
            "digest_version": version,
            "chapters": [{
                "index": views[0].index,
                "title": "what happened here",
                "summary": f"{SENTENCES} sentences.",
                "themes": ["a theme"],
                "notable_segment_ids": [example_seq],
                "quote": "a verbatim line from this chapter",
                "open_loops": [
                    {"text": "what he said he would try",
                     "kind": "promise", "segment_id": example_seq},
                ],
            }],
        }, indent=2),
        "```",
        "",
        prompts["chapters_header"].rstrip(),
    ]

    for view in views:
        lines += [
            "",
            f"### chapter {view.index} "
            f"({_clock(view.t_start)}-{_clock(view.t_end)})",
            "",
        ]
        if view.lines:
            # The id range is stated as well as shown: it is what scopes the
            # model's citations to this chapter, and §12.2 rejects anything
            # outside it.
            low, high = min(view.seqs), max(view.seqs)
            lines += [f"segment ids {low}-{high}", "", view.transcript]
        else:
            lines.append(prompts["no_transcript"].rstrip())

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the reply
# --------------------------------------------------------------------------


def validate(reply: dict | None, views: list[ChapterView], *,
             version: int) -> Validated:
    """§12.2 and §12.3, applied to whatever the model said.

    Every drop carries a reason. §12.2 makes drops silent to the *model*, not to
    the person watching the tool — an operator who cannot see that four of six
    chapters were discarded has no way to tell a bad reply from a stream with
    nothing in it.
    """
    result = Validated()
    if not reply:
        result.drop("malformed", "no JSON object found in the reply")
        return result

    claimed = reply.get("digest_version")
    if claimed is not None and int(claimed) != version:
        raise DigestPromptError(
            f"this reply was written for digest v{claimed}, but the newest "
            f"digest is v{version}. Segment ids are scoped per chapter, so "
            f"applying it would point every citation at the wrong chapter. "
            f"Regenerate the prompt."
        )

    entries = reply.get("chapters")
    if not isinstance(entries, list):
        result.drop("malformed", "no `chapters` array in the reply")
        return result

    by_index = {view.index: view for view in views}
    seen: set[int] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            result.drop("malformed", f"not an object: {entry!r:.60}")
            continue

        # §12.2 counts IDS, not entries: one chapter carries its index plus a
        # notable id per citation plus an id per open loop, and putting those
        # drops over an entry denominator reported 1.00 for a reply whose only
        # invented handle was a single chapter index.
        result.saw_id()
        raw = entry.get("index")
        try:
            index = int(raw)
        except (TypeError, ValueError):
            result.drop("unknown-id", f"index {raw!r} is not an index")
            continue

        view = by_index.get(index)
        if view is None:
            # §12.2: dropped, and counted into the hallucination rate.
            result.drop("unknown-id", f"chapter {index} is not in this digest")
            continue
        if index in seen:
            result.drop("duplicate", f"chapter {index} appeared twice")
            continue
        seen.add(index)

        # A chapter with no transcript has nothing to quote and nothing to
        # summarise; the prompt asks for a null title there. Accepting it
        # unchecked would be accepting a summary of nothing.
        if not view.lines:
            result.chapters.append(ChapterResult(
                index=index, title=None, summary=None))
            continue

        quote = str(entry.get("quote") or "")
        if not quote.strip():
            result.drop("no-quote", f"chapter {index} has no quote")
            continue
        # §12.3: the quote must actually be in THIS chapter.
        if llm.normalise(quote) not in llm.normalise(view.transcript):
            result.drop(
                "bad-quote",
                f"chapter {index}: {quote[:60]!r} is not in its transcript")
            continue

        notable, dropped_ids, seen_ids = _scope_ids(
            entry.get("notable_segment_ids"), view)
        result.saw_id(seen_ids)
        for detail in dropped_ids:
            result.drop("unknown-id", f"chapter {index}: {detail}")

        loops, dropped_loops, seen_loop_ids = _open_loops(
            entry.get("open_loops"), view, index)
        result.saw_id(seen_loop_ids)
        for reason, detail in dropped_loops:
            result.drop(reason, detail)

        result.chapters.append(ChapterResult(
            index=index,
            title=_text(entry.get("title")),
            summary=_text(entry.get("summary")),
            themes=[t for t in (_text(x) for x in (entry.get("themes") or [])) if t],
            notable_segment_ids=notable,
            quote=quote.strip(),
            open_loops=loops,
        ))

    return result


def _text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _scope_ids(raw, view: ChapterView) -> tuple[list[int], list[str], int]:
    """§12.2, with the twist that matters here.

    An id from ANOTHER chapter is as wrong as an invented one. It exists, so a
    plain existence check passes it — and the result is a citation attached to a
    chapter whose transcript does not contain it, which is exactly the
    fabrication §12.2 exists to catch. Scoped to the chapter's own range.
    """
    out: list[int] = []
    dropped: list[str] = []
    seen = 0
    for item in (raw or []):
        seen += 1
        try:
            seq = int(item)
        except (TypeError, ValueError):
            dropped.append(f"segment id {item!r} is not an id")
            continue
        if seq not in view.seqs:
            dropped.append(f"segment {seq} is not in this chapter")
            continue
        if seq not in out:
            out.append(seq)
    return out, dropped, seen


def _open_loops(raw, view: ChapterView, index: int
                ) -> tuple[list[dict], list[tuple[str, str]], int]:
    """§9.5's open loops — a field in the chapter prompt, not a separate pass."""
    out: list[dict] = []
    dropped: list[tuple[str, str]] = []
    seen = 0
    for item in (raw or []):
        if not isinstance(item, dict):
            dropped.append(("malformed", f"chapter {index}: loop {item!r:.40}"))
            continue
        text = _text(item.get("text"))
        if not text:
            dropped.append(("no-text", f"chapter {index}: a loop has no text"))
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in LOOP_KINDS:
            # §3.2 documents the three values. An unrecognised one is stored as
            # NULL rather than dropped: the loop itself is still a real finding,
            # and losing it to a bad label would be the wrong trade.
            kind = None
        seq = item.get("segment_id")
        if seq is not None:
            seen += 1
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            seq = None
        if seq is not None and seq not in view.seqs:
            dropped.append(("unknown-id",
                            f"chapter {index}: loop cites segment {seq}, "
                            f"which is not in this chapter"))
            seq = None
        out.append({"text": text, "kind": kind, "segment_id": seq})
    return out, dropped, seen


# --------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------


def reduce_into(content: dict, result: Validated,
                conn: sqlite3.Connection, stream_id: str) -> dict:
    """§9.4's reduce: the chapter outputs onto the deterministic digest.

    A deep copy, because the source is a stored digest that must not change: v1
    stays exactly as it was written.
    """
    merged = json.loads(json.dumps(content))
    by_index = {r.index: r for r in result.chapters}
    themes: list[str] = []
    loops: list[dict] = []

    for chapter in merged.get("chapters") or []:
        accepted = by_index.get(chapter["index"])
        if accepted is None:
            continue
        chapter["title"] = accepted.title
        chapter["summary"] = accepted.summary
        chapter["notable_segment_ids"] = accepted.notable_segment_ids
        for theme in accepted.themes:
            if theme not in themes:
                themes.append(theme)
        for loop in accepted.open_loops:
            loops.append({**loop, "t": _time_of(conn, stream_id,
                                                loop.get("segment_id"))})

    merged["themes_observed"] = themes
    merged["open_loops"] = loops
    return merged


def _time_of(conn: sqlite3.Connection, stream_id: str, seq) -> float | None:
    """§12.1: the model returns ids; every time is resolved here."""
    if seq is None:
        return None
    row = conn.execute(
        "SELECT t_start FROM segments WHERE stream_id = ? AND seq = ?",
        (stream_id, int(seq)),
    ).fetchone()
    return None if row is None else round(float(row["t_start"]), 3)


def store_open_loops(conn: sqlite3.Connection, stream_id: str,
                     loops: list[dict]) -> int:
    """§9.5: "Write results to the `open_loops` table."

    Replaces this stream's rows rather than appending. `open_loops` has no
    natural key, so re-applying a reply would otherwise accumulate a duplicate
    set on every attempt — and §11.3 tracks these across streams, where a
    triplicated loop is three findings that look independent.

    Rows already marked `resolved` by a later stream are left alone: that status
    is the operator's, not this pass's, and re-running a digest must not
    reopen something they closed.
    """
    conn.execute(
        "DELETE FROM open_loops WHERE stream_id = ? AND status = 'open'",
        (stream_id,),
    )
    for loop in loops:
        conn.execute(
            "INSERT INTO open_loops (stream_id, t, text, kind) VALUES (?, ?, ?, ?)",
            (stream_id, loop.get("t"), loop["text"], loop.get("kind")),
        )
    return len(loops)
