"""§8.5 — hook text, and §12's rules about how a model is allowed to answer.

> The hook -- first 1-2 seconds plus on-screen text -- is the single
> highest-leverage decision in short-form. It stays manual, **but generating
> candidate options is a genuine time save.** Have the LLM propose 5 hook
> variants from the clip's transcript. The operator picks and rewrites.

**No API key, so the round trip is the product.** §2.2 puts reasoning through
an external frontier model and §12.4 prices it at cents per stream, but the
operator does not want a paid key yet -- and nothing here can drive a chat
website on their behalf. So the deliverable is a prompt that goes out and a
reply that comes back, with every §12 rule enforced on the way in.

That turns out to satisfy §12 more strictly than an API call would, because
every check below runs on a reply nobody controlled:

**§12.1 -- the model never emits timestamps.** The prompt carries
`{export_id, transcript}` pairs and asks for ids back; every time is resolved
from the database here.

**Real database ids, not 1..5.** §12.2 wants a hallucinated id to be
*detectable*, and if the prompt numbers five clips 1 to 5 then any number a
model invents in that range looks valid. Export ids are the real handles, so a
fabrication has somewhere to fail.

**§12.2 -- validation is mandatory and drops are counted.** Unknown ids are
dropped silently and the rate is written to `tool_metrics` as
`llm_invalid_id_rate`, which is §14's own name for the metric.

**§12.3 -- every selection carries a verbatim quote**, checked by normalised
substring against the transcript of that clip's window. It is the cheap,
machine-checkable guard against a model answering about a clip it never read:
fabrication becomes visible rather than plausible.

**The reply is parsed tolerantly.** What comes back from a chat window has
prose wrapped around the JSON, and asking the operator to trim it by hand is
the kind of friction that makes a tool go unused. The last JSON object wins --
the same tactic `loudness.parse` already needed against ffmpeg's diagnostics.

**Nothing is chosen automatically.** `--apply` validates and reports; `--pick`
writes. §8.5 calls the hook the highest-leverage decision in short-form and
says it stays manual, so a mode that quietly stored the model's first option
would be the model deciding.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Protocol

from clipforge import db
from clipforge.render import RenderError
from clipforge.render.words import role_for, track_roles

#: §14's name for it. Written to `tool_metrics` on every `--apply`.
INVALID_ID_METRIC = "llm_invalid_id_rate"

#: §8.5: "propose 5 hook variants".
WANTED_OPTIONS = 5

_SPACE = re.compile(r"\s+")


class HookError(RenderError):
    pass


@dataclass
class Clip:
    """A rendered clip the model is being asked about."""

    export_id: int
    path: str
    t_start: float
    t_end: float
    transcript: str
    hook_text: str | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Proposal:
    """One clip's hooks, after validation."""

    export_id: int
    quote: str
    options: list[str]


@dataclass
class Validated:
    """What came back, and what was thrown away."""

    proposals: list[Proposal] = field(default_factory=list)
    #: (reason, detail) for everything dropped. §12.2 says drops are silent to
    #: the caller of the model, not to the operator watching the tool.
    dropped: list[tuple[str, str]] = field(default_factory=list)
    #: How many ids the model returned in total, valid or not.
    returned: int = 0

    @property
    def invalid_id_rate(self) -> float:
        """§14's `llm_invalid_id_rate`. 0.0 when the model returned nothing."""
        if not self.returned:
            return 0.0
        bad = sum(1 for reason, _ in self.dropped if reason == "unknown-id")
        return round(bad / self.returned, 4)


# --------------------------------------------------------------------------
# a source of hooks
# --------------------------------------------------------------------------


class HookSource(Protocol):
    """Where hook options come from.

    One implementation today. `AnthropicHookSource` drops in beside it when
    there is a key, and `available` is what lets the command say which is in
    play rather than failing -- the same shape as `StageSpec.available` and the
    Ollama check in `extract/embeddings.py`.
    """

    name: str

    def available(self, cfg) -> tuple[bool, str]: ...


@dataclass
class ManualHookSource:
    """The paste round trip: a prompt out, a reply back, validated here."""

    name: str = "manual"

    def available(self, cfg) -> tuple[bool, str]:
        # Always available: it needs a person and a browser, not a credential.
        return True, ""


def source_for(cfg) -> HookSource:
    """The configured source. Only one exists, and it says so honestly."""
    wanted = str(cfg.get("render.hooks.source", "manual"))
    if wanted != "manual":
        raise HookError(
            f"no hook source {wanted!r}. Only 'manual' is built -- an API-backed "
            f"source needs a key, and §12.4 prices it at roughly $0.10-0.30 per "
            f"stream if you want one."
        )
    return ManualHookSource()


# --------------------------------------------------------------------------
# what the model is asked about
# --------------------------------------------------------------------------


def load_clips(conn: sqlite3.Connection, stream_id: str) -> list[Clip]:
    """Every rendered clip for a stream, with the transcript of its window.

    Rendered clips rather than approved moments, because §8.5 stores the result
    in `exports.hook_text` and that row exists only once something has been
    written. It also matches §8.1's loop: render the batch, watch it, pick
    hooks for the ones worth posting.
    """
    rows = conn.execute(
        """
        SELECT e.id, e.path, e.hook_text, e.candidate_id, i.t_start, i.t_end
          FROM exports e
          JOIN export_items i ON i.export_id = e.id
         WHERE e.stream_id = ? AND e.kind = 'clip'
         ORDER BY i.t_start, e.id
        """,
        (stream_id,),
    ).fetchall()
    rows = _newest_per_moment(rows)

    track_row = conn.execute(
        "SELECT audio_track_map FROM streams WHERE id = ?", (stream_id,)
    ).fetchone()
    roles = track_roles(track_row["audio_track_map"] if track_row else None)

    clips: list[Clip] = []
    for row in rows:
        clips.append(Clip(
            export_id=int(row["id"]),
            path=str(row["path"]),
            t_start=float(row["t_start"]),
            t_end=float(row["t_end"]),
            transcript=_transcript(conn, stream_id, row["t_start"], row["t_end"],
                                   roles),
            hook_text=row["hook_text"],
        ))
    return clips


def _newest_per_moment(rows) -> list:
    """One clip per moment: the most recent export of it.

    `exports` is append-only, so re-rendering a stream three times leaves three
    rows for every moment. Asking a model about the same clip three times
    spends its attention and the operator's on nothing, and the duplicates are
    indistinguishable in the reply.

    Keyed on `candidate_id` where there is one, and on the window otherwise --
    an export written before that column was populated still has a window, and
    two clips of the same window are the same moment whatever produced them.
    """
    newest: dict = {}
    for row in rows:
        key = (row["candidate_id"] if row["candidate_id"] is not None
               else (round(float(row["t_start"]), 2), round(float(row["t_end"]), 2)))
        if key not in newest or row["id"] > newest[key]["id"]:
            newest[key] = row
    return sorted(newest.values(), key=lambda r: (float(r["t_start"]), r["id"]))


def _transcript(conn, stream_id: str, t_start: float, t_end: float,
                roles: dict[int, str]) -> str:
    """The clip's own words, attributed.

    §12.4 says to send transcript around candidate windows and nothing more.
    For a hook the clip's own words are the material -- the hook has to be
    about what is in the clip.
    """
    lines = []
    for row in conn.execute(
        "SELECT text, speaker, track FROM segments "
        "WHERE stream_id = ? AND t_end >= ? AND t_start <= ? ORDER BY t_start",
        (stream_id, t_start, t_end),
    ):
        who = role_for(row["track"], row["speaker"], roles) or "unknown"
        text = (row["text"] or "").strip()
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines)


def normalise(text: str) -> str:
    """For the §12.3 quote check: whitespace and case, nothing else.

    Deliberately not punctuation-insensitive. "Verbatim" is the requirement,
    and a model that rewrote the words has not quoted the clip -- but one that
    re-wrapped a line has, and that should not be a rejection.
    """
    return _SPACE.sub(" ", str(text or "")).strip().lower()


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def build_prompt(cfg, clips: list[Clip], stream_id: str) -> str:
    """One prompt covering every clip in the stream.

    One, not one per clip: C8 budgets ~35 minutes of hands-on time per stream,
    and five paste round trips would spend a chunk of it on clerical work.
    """
    if not clips:
        raise HookError(
            f"{stream_id} has no rendered clips. §8.5's hooks are written for "
            f"clips you have watched, so render first:\n"
            f"    clipforge render {stream_id}"
        )

    count = int(cfg.get("render.hooks.options", WANTED_OPTIONS))
    lines = [
        f"# Hook options for {stream_id}",
        "",
        f"You are helping choose the on-screen hook text for {len(clips)} "
        f"short-form clip(s) cut from a gaming stream.",
        "",
        f"For EACH clip below, propose {count} hook variants. A hook is the "
        "on-screen text over the first 1-2 seconds: short, specific, and "
        "written to make someone stop scrolling. Use what is actually said in "
        "the clip -- do not invent events.",
        "",
        "## Rules for your reply",
        "",
        "- Reply with ONE JSON object and nothing else outside it.",
        "- Use the `export_id` values exactly as given below. Do not invent ids.",
        "- For each clip include `quote`: a VERBATIM span copied from that "
        "clip's transcript. It is checked automatically against the "
        "transcript, and an entry whose quote does not appear is discarded.",
        "",
        "```json",
        json.dumps({"hooks": [{
            "export_id": clips[0].export_id,
            "quote": "a verbatim span from this clip's transcript",
            "options": [f"hook variant {i + 1}" for i in range(count)],
        }]}, indent=2),
        "```",
        "",
        "## The clips",
    ]

    for clip in clips:
        lines += [
            "",
            f"### export_id: {clip.export_id}",
            f"- file: `{clip.path}`",
            f"- length: {clip.duration:.1f}s",
            "- transcript:",
            "",
        ]
        lines.append(clip.transcript or "  (no transcript for this window)")
        if clip.hook_text:
            lines.append("")
            lines.append(f"- a hook is already stored: {clip.hook_text!r}")

    return "\n".join(lines) + "\n"


def to_clipboard(text: str) -> bool:
    """Best effort. Returns whether it worked, and never raises."""
    command = ["clip"] if sys.platform == "win32" else ["pbcopy"]
    try:
        subprocess.run(command, input=text.encode("utf-8"), check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------
# the reply
# --------------------------------------------------------------------------


def parse_reply(text: str) -> dict | None:
    """The last JSON object in whatever came back, or None.

    A chat window wraps its answer in prose. Asking the operator to trim that
    by hand is the friction that stops a tool being used, so the parser scans
    backwards for a balanced object instead.
    """
    depth = 0
    end = -1
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == "}":
            if depth == 0:
                end = index
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0 and end != -1:
                try:
                    return json.loads(text[index:end + 1])
                except json.JSONDecodeError:
                    depth, end = 0, -1
    return None


def validate(reply: dict | None, clips: list[Clip]) -> Validated:
    """§12.2 and §12.3, applied to whatever the model said.

    Every drop is recorded with a reason. §12.2 makes drops silent to the
    *model*, not to the person watching the tool -- an operator who cannot see
    that four of five entries were discarded has no way to know the reply was
    bad rather than the clips being unhookable.
    """
    result = Validated()
    if not reply:
        return result

    entries = reply.get("hooks")
    if not isinstance(entries, list):
        result.dropped.append(("malformed", "no `hooks` array in the reply"))
        return result

    by_id = {clip.export_id: clip for clip in clips}

    for entry in entries:
        if not isinstance(entry, dict):
            result.dropped.append(("malformed", f"not an object: {entry!r:.60}"))
            continue

        result.returned += 1
        raw_id = entry.get("export_id")
        try:
            export_id = int(raw_id)
        except (TypeError, ValueError):
            result.dropped.append(("unknown-id", f"export_id {raw_id!r} is not an id"))
            continue

        clip = by_id.get(export_id)
        if clip is None:
            # §12.2: dropped, and counted into the hallucination rate.
            result.dropped.append(
                ("unknown-id", f"export_id {export_id} is not a clip of this stream"))
            continue

        options = [str(o).strip() for o in (entry.get("options") or [])
                   if str(o).strip()]
        if not options:
            result.dropped.append(("no-options", f"export_id {export_id} has none"))
            continue

        # §12.3: the quote must actually be in the clip.
        quote = str(entry.get("quote") or "")
        if not quote.strip():
            result.dropped.append(("no-quote", f"export_id {export_id} has no quote"))
            continue
        if normalise(quote) not in normalise(clip.transcript):
            result.dropped.append((
                "bad-quote",
                f"export_id {export_id}: {quote[:60]!r} is not in its transcript",
            ))
            continue

        result.proposals.append(Proposal(export_id, quote, options))

    return result


def record(conn: sqlite3.Connection, stream_id: str, result: Validated) -> None:
    """§12.2's "logged to tool_metrics for monitoring hallucination rate"."""
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) "
            "VALUES (?, ?, ?, ?)",
            (stream_id, INVALID_ID_METRIC, result.invalid_id_rate, json.dumps({
                "returned": result.returned,
                "accepted": len(result.proposals),
                "dropped": [reason for reason, _detail in result.dropped],
            })),
        )


def store(conn: sqlite3.Connection, export_id: int, text: str) -> None:
    """Write the operator's choice. §8.5's `exports.hook_text`."""
    row = conn.execute("SELECT id FROM exports WHERE id = ?", (export_id,)).fetchone()
    if row is None:
        raise HookError(f"no export {export_id}")
    with db.transaction(conn):
        conn.execute("UPDATE exports SET hook_text = ? WHERE id = ?",
                     (text, export_id))
