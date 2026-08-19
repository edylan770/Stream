"""§12's rules, in one place, for every call site that talks to a model.

These checks were written once inside `render/hooks.py` and are lifted here
unchanged, because §9.4's digest, §10.3's three passes and §11.1's cluster
labelling all need exactly the same four:

**§12.1 -- the model never emits a timestamp.** It is handed ids and returns
ids; every time is resolved from the database by the caller. Nothing in this
module converts anything to a time, which is what makes that structural rather
than a rule somebody has to remember.

**§12.2 -- unknown ids are dropped, and the drop is counted.** `Validated`
carries both the survivors and the casualties, and `invalid_id_rate` is §14's
own `llm_invalid_id_rate`. A model that answers about a moment that does not
exist has to *fail*, not blend in.

**§12.3 -- every selection carries a verbatim quote**, checked by normalised
substring against the text the caller says that id refers to. It is the cheap
machine-checkable guard against a model answering about material it never read.

**The reply is parsed tolerantly.** Chat windows wrap JSON in prose. That is a
property of the transport, not of any one prompt, so `parse_reply` lives here
and both the manual and API paths run through it -- the API path can constrain
the schema server-side and still gets the same checks applied to what comes
back, because a validator only one path uses is a validator that drifts.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from clipforge import db

#: §14's name for it, and §12.2's requirement that it be logged. A constant
#: because it is a column value: a different string here means whatever reads
#: it later finds nothing.
INVALID_ID_METRIC = "llm_invalid_id_rate"

_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """For the §12.3 quote check: whitespace and case, nothing else.

    Deliberately not punctuation-insensitive. "Verbatim" is the requirement,
    and a model that rewrote the words has not quoted the source -- but one
    that re-wrapped a line has, and that should not be a rejection.
    """
    return _SPACE.sub(" ", str(text or "")).strip().lower()


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


@dataclass
class Selection:
    """One entry that survived §12.2 and §12.3.

    `entry` is the model's own object, still unvalidated apart from the id and
    the quote. Whatever else a given prompt asked for is the caller's to read
    and to police -- this module knows about ids and quotes, which are the two
    things §12 makes universal.
    """

    key: Any
    quote: str
    entry: dict


@dataclass
class Validated:
    """What came back, and what was thrown away."""

    selections: list[Selection] = field(default_factory=list)
    #: (reason, detail) for everything dropped. §12.2 says drops are silent to
    #: the caller of the model, not to the operator watching the tool.
    dropped: list[tuple[str, str]] = field(default_factory=list)
    #: How many entries the model returned in total, valid or not.
    returned: int = 0

    @property
    def invalid_id_rate(self) -> float:
        """§14's `llm_invalid_id_rate`. 0.0 when the model returned nothing."""
        if not self.returned:
            return 0.0
        bad = sum(1 for reason, _ in self.dropped if reason == "unknown-id")
        return round(bad / self.returned, 4)


def validate_selections(
    entries: Iterable[Any],
    known: dict,
    *,
    id_field: str,
    text_for: Callable[[Any], str],
    quote_field: str = "quote",
    noun: str = "known",
    check: Callable[[dict, Any, Any], tuple[str, str] | None] | None = None,
    into: Validated | None = None,
) -> Validated:
    """§12.2 and §12.3, applied to whatever the model said.

    `known` maps the ids the prompt handed out to whatever they stand for;
    `text_for` returns the text a quote about one of them has to appear in.
    Those two arguments are the entire difference between checking a hook
    reply against `exports.id` and checking a digest reply against
    `segments.seq`, which is why this is one function rather than one per
    caller.

    Every drop is recorded with a reason. §12.2 makes drops silent to the
    *model*, not to the person watching the tool -- an operator who cannot see
    that four of five entries were discarded has no way to know the reply was
    bad rather than the material being unusable.

    `check` runs after the id resolves and before the quote is examined, for
    the per-prompt requirements this module has no opinion about. It returns
    None to accept, or a (reason, detail) pair to drop.
    """
    result = into if into is not None else Validated()

    for entry in entries:
        if not isinstance(entry, dict):
            result.dropped.append(("malformed", f"not an object: {entry!r:.60}"))
            continue

        result.returned += 1
        raw = entry.get(id_field)
        key = _coerce(raw, known)
        if key is None:
            # §12.2: dropped, and counted into the hallucination rate. Both
            # arms of this are `unknown-id` -- a model that returns "seg-4"
            # and one that returns 4000 have each named something that is not
            # a handle it was given.
            result.dropped.append(
                ("unknown-id", f"{id_field} {raw!r} is not {noun}"))
            continue

        subject = known[key]
        if check is not None:
            failed = check(entry, key, subject)
            if failed is not None:
                result.dropped.append(failed)
                continue

        quote = str(entry.get(quote_field) or "")
        if not quote.strip():
            result.dropped.append(("no-quote", f"{id_field} {key} has no quote"))
            continue
        # §12.3: the quote must actually be in the material.
        if normalise(quote) not in normalise(text_for(subject)):
            result.dropped.append((
                "bad-quote",
                f"{id_field} {key}: {quote[:60]!r} is not in its text",
            ))
            continue

        result.selections.append(Selection(key, quote, entry))

    return result


def _coerce(raw: Any, known: dict):
    """The id the model returned, as a key of `known`, or None.

    JSON has no integer/string distinction a model reliably honours, so `"7"`
    and `7` are both tried before anything is called a hallucination. Nothing
    else is tried: the point of §12.2 is that a fabricated id fails, and a
    coercion generous enough to rescue a typo is generous enough to rescue an
    invention.
    """
    try:
        if raw in known:
            return raw
    except TypeError:                      # unhashable, e.g. a list
        return None
    try:
        as_int = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return as_int if as_int in known else None


def record(conn: sqlite3.Connection, stream_id: str, result: Validated,
           metric: str = INVALID_ID_METRIC) -> None:
    """§12.2's "logged to tool_metrics for monitoring hallucination rate"."""
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) "
            "VALUES (?, ?, ?, ?)",
            (stream_id, metric, result.invalid_id_rate, json.dumps({
                "returned": result.returned,
                "accepted": len(result.selections),
                "dropped": [reason for reason, _detail in result.dropped],
            })),
        )
