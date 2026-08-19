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

**§12 now lives in `clipforge/llm/`, not here.** It was written in this file
first, against `exports.id`, and §9.4's digest map, §10.3's ground pass and
§11.1's cluster labels need the identical four rules against different ids. A
second copy would be a second place for a §12 rule to drift, so the generic
half moved out and this module became its first caller. Everything it used to
define is re-exported below, and `tests/test_hooks.py` passes unmodified --
which is the only evidence that the move changed nothing.

What is left here is what is actually about hooks: which clips to ask about,
what the prompt says, and where the operator's choice is stored.

That round trip turns out to satisfy §12 more strictly than an API call would,
because every check runs on a reply nobody controlled:

**§12.1 -- the model never emits timestamps.** The prompt carries
`{export_id, transcript}` pairs and asks for ids back; every time is resolved
from the database here.

**Real database ids, not 1..5.** §12.2 wants a hallucinated id to be
*detectable*, and if the prompt numbers five clips 1 to 5 then any number a
model invents in that range looks valid. Export ids are the real handles, so a
fabrication has somewhere to fail -- which is also why the JSON example in the
prompt is built here around a real one rather than living in `prompts.yaml`.

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
import sqlite3
from dataclasses import dataclass

from clipforge import db, llm
from clipforge.llm import prompts
from clipforge.render import RenderError
from clipforge.render.words import role_for, track_roles

# Re-exported so every caller and every test that knew these as `hooks.X`
# still finds them. §12's implementation moved; its names did not.
INVALID_ID_METRIC = llm.INVALID_ID_METRIC
normalise = llm.normalise
parse_reply = llm.parse_reply
record = llm.record
to_clipboard = llm.to_clipboard
#: The paste round trip, under the name this module gave it first.
ManualHookSource = llm.ManualSource

#: §8.5: "propose 5 hook variants".
WANTED_OPTIONS = 5

#: The prompt text lives in `config/prompts.yaml` so it can be rewritten
#: without touching code -- see `clipforge/llm/prompts.py` for why that matters
#: more here than anywhere else in the system.
PROMPT_NAME = "hook"


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


class Validated(llm.Validated):
    """§12's result, in the shape `clipforge hook` prints.

    The generic validator knows about ids and quotes, which §12 makes
    universal. `options` is what *this* prompt asked for, so the hook-shaped
    view of the survivors is assembled here rather than in a field the digest
    would have to carry and ignore.
    """

    @property
    def proposals(self) -> list[Proposal]:
        return [Proposal(int(s.key), s.quote, _options(s.entry))
                for s in self.selections]


# --------------------------------------------------------------------------
# a source of hooks
# --------------------------------------------------------------------------


def source_for(cfg) -> llm.Source:
    """The configured source, whether or not it can run.

    Reads `render.hooks.source` rather than `llm.source`: hooks predate the
    shared package and the key is already in local configs. Returning an
    unavailable source instead of raising is what lets `clipforge hook` report
    *which* of a missing package and a missing key is the problem.
    """
    try:
        return llm.source_for(cfg, "render.hooks.source")
    except llm.LLMError as exc:
        raise HookError(str(exc)) from None


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


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def build_prompt(cfg, clips: list[Clip], stream_id: str) -> str:
    """One prompt covering every clip in the stream.

    One, not one per clip: C8 budgets ~35 minutes of hands-on time per stream,
    and five paste round trips would spend a chunk of it on clerical work.

    The fixed prose comes from `prompts.yaml`. The per-clip blocks and the JSON
    example do not: the example has to carry a REAL `export_id`, which is
    §12.2's entire argument for real handles, and the clip blocks are database
    rows rather than text anybody would want to edit.
    """
    if not clips:
        raise HookError(
            f"{stream_id} has no rendered clips. §8.5's hooks are written for "
            f"clips you have watched, so render first:\n"
            f"    clipforge render {stream_id}"
        )

    count = int(cfg.get("render.hooks.options", WANTED_OPTIONS))
    schema = json.dumps({"hooks": [{
        "export_id": clips[0].export_id,
        "quote": "a verbatim span from this clip's transcript",
        "options": [f"hook variant {i + 1}" for i in range(count)],
    }]}, indent=2)

    lines = [prompts.render(
        cfg, PROMPT_NAME,
        stream_id=stream_id,
        clip_count=len(clips),
        option_count=count,
        schema=schema,
    ).rstrip("\n")]

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


# --------------------------------------------------------------------------
# the reply
# --------------------------------------------------------------------------


def _options(entry: dict) -> list[str]:
    return [str(option).strip() for option in (entry.get("options") or [])
            if str(option).strip()]


def _needs_options(entry: dict, key, _clip) -> tuple[str, str] | None:
    """The one requirement §12 has no opinion about: this prompt asked for a
    list of variants, and an entry with none is a clip the model skipped."""
    if not _options(entry):
        return ("no-options", f"export_id {key} has none")
    return None


def validate(reply: dict | None, clips: list[Clip]) -> Validated:
    """§12.2 and §12.3 against this stream's clips.

    The loop itself is `llm.validate_selections`; what is hook-specific is the
    handle (`export_id`), the text a quote has to appear in (the clip's own
    transcript), and the `options` requirement above.
    """
    result = Validated()
    if not reply:
        return result

    entries = reply.get("hooks")
    if not isinstance(entries, list):
        result.dropped.append(("malformed", "no `hooks` array in the reply"))
        return result

    return llm.validate_selections(
        entries,
        {clip.export_id: clip for clip in clips},
        id_field="export_id",
        text_for=lambda clip: clip.transcript,
        noun="a clip of this stream",
        check=_needs_options,
        into=result,
    )


def store(conn: sqlite3.Connection, export_id: int, text: str) -> None:
    """Write the operator's choice. §8.5's `exports.hook_text`."""
    row = conn.execute("SELECT id FROM exports WHERE id = ?", (export_id,)).fetchone()
    if row is None:
        raise HookError(f"no export {export_id}")
    with db.transaction(conn):
        conn.execute("UPDATE exports SET hook_text = ? WHERE id = ?",
                     (text, export_id))
