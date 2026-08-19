"""§12's rules, in one place.

> **These rules apply to every LLM call in the system without exception.**

They were written for §8.5's hook options and lived in `render/hooks.py`, which
was the only caller. §9.4's digest is the second, and two implementations of
"is this quote verbatim" or "how do I find the JSON in a chat reply" would drift
— quietly, because both would keep passing their own tests. So they live here,
at package root beside `signals.py`, for the same reason that does: something
every layer needs is not a detail of one of them.

Nothing in this module talks to a network. There is no API key anywhere in this
project, so the transport is a person with a browser: a prompt goes out, a reply
comes back, and every rule below is enforced on the way in. That turns out to
satisfy §12 more strictly than an API call would, because each check runs on a
reply nobody controlled.

**§12.1 — the model never emits timestamps.** Callers send `{id, text}` pairs and
ask for ids back; every time is resolved from the database. The ids must be real
database handles, never `1..n`: §12.2 wants a hallucinated id to be *detectable*,
and if a prompt numbers five things 1 to 5 then any number a model invents in
that range looks valid.

**§12.2 — validation is mandatory, and drops are counted.** Unknown ids are
dropped and the rate is written to `tool_metrics` as `llm_invalid_id_rate`, which
is §14's own name for it. Silent to the *model*, never to the operator: someone
who cannot see that four of five entries were discarded has no way to tell a bad
reply from a stream with nothing in it.

**§12.3 — every selection carries a verbatim quote**, checked by normalised
substring against the text the model was shown. It is the cheap, machine-checkable
guard against a model answering about something it never read.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field

from clipforge import db

#: §14's name for it.
INVALID_ID_METRIC = "llm_invalid_id_rate"

#: Drop reasons that count as hallucination. A malformed entry or a missing
#: field is a badly-behaved reply; an id that does not exist is the model
#: inventing a handle, and only that belongs in the rate §14 asks for.
HALLUCINATION_REASONS = frozenset({"unknown-id"})

_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """For the §12.3 quote check: whitespace and case, nothing else.

    Deliberately not punctuation-insensitive. "Verbatim" is the requirement, and
    a model that rewrote the words has not quoted the source — but one that
    re-wrapped a line has, and that should not be a rejection.
    """
    return _SPACE.sub(" ", str(text or "")).strip().lower()


def parse_reply(text: str) -> dict | None:
    """The last JSON object in whatever came back, or None.

    A chat window wraps its answer in prose. Asking the operator to trim that by
    hand is the friction that stops a tool being used, so the parser scans
    backwards for a balanced object instead — the same tactic `loudness.parse`
    needed against ffmpeg's diagnostics.
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
class Validation:
    """What survived a reply, and what did not.

    Subclassed by each caller to carry whatever it accepted; the bookkeeping
    §12.2 requires is the same for all of them.
    """

    #: (reason, detail) for everything thrown away.
    dropped: list[tuple[str, str]] = field(default_factory=list)
    #: How many IDS the model returned in total, valid or not. **Ids, not
    #: entries**, because §12.2 says "every returned ID is checked for
    #: existence" and the rate below is a fraction of that same population.
    #:
    #: The distinction is invisible for §8.5, where one hook entry carries
    #: exactly one export_id — and wrong for §9.4, where one chapter carries an
    #: index plus its notable ids plus an id per open loop. Counting entries
    #: there put sub-entity drops over an entry denominator: a reply with one
    #: bad chapter index and two mis-scoped segment ids inside an otherwise
    #: ACCEPTED chapter reported a rate of 1.00, which reads as "the model
    #: hallucinated everything" when two of three chapters were usable.
    returned: int = 0

    @property
    def invalid_id_rate(self) -> float:
        """§14's `llm_invalid_id_rate`. 0.0 when the model returned nothing."""
        if not self.returned:
            return 0.0
        bad = sum(1 for reason, _ in self.dropped if reason in HALLUCINATION_REASONS)
        return round(bad / self.returned, 4)

    def saw_id(self, count: int = 1) -> None:
        """Count ids the model returned, whether or not they resolve."""
        self.returned += count

    def drop(self, reason: str, detail: str) -> None:
        self.dropped.append((reason, detail))


def record(conn: sqlite3.Connection, stream_id: str, result: Validation, *,
           accepted: int, metric: str = INVALID_ID_METRIC) -> None:
    """§12.2's "logged to tool_metrics for monitoring hallucination rate"."""
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) "
            "VALUES (?, ?, ?, ?)",
            (stream_id, metric, result.invalid_id_rate, json.dumps({
                "returned": result.returned,
                "accepted": accepted,
                "dropped": [reason for reason, _detail in result.dropped],
            })),
        )


def to_clipboard(text: str) -> bool:
    """Best effort. Returns whether it worked, and never raises."""
    command = ["clip"] if sys.platform == "win32" else ["pbcopy"]
    try:
        subprocess.run(command, input=text.encode("utf-8"), check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
