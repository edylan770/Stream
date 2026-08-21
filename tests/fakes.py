"""A scripted LLM source, so §12 can be exercised without a key.

Nothing in this project has ever called a model. `AnthropicSource` reports
itself unavailable twice over (no package, no key) and `ManualSource` needs a
person and a browser — so the only way to test §12's four rules against a
digest-shaped reply is to script one.

INJECTED, NOT REGISTERED. `sources.SOURCES` stays as it is and `source_for`
keeps refusing unknown names; a stage takes `source=None` and a test passes one
of these, the pattern `transcript.run(ctx, transcriber=None)` and
`embeddings.run(ctx, embedder=None)` already set. A fake reachable from config
is a fake that can be selected by accident.

THE PROMPTS ARE RECORDED, not just the replies. What a digest ASKED for is as
much a part of §12 as what it did with the answer — §12.1 says the model never
sees a timestamp, and the only way to check that is to read the prompt that was
sent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


class ExhaustedError(RuntimeError):
    """More calls than scripted replies.

    Its own type because "the code called the model more times than the test
    expected" is a finding — §12.4 budgets the whole of a stream's reasoning,
    and a map that quietly ran twice per chapter would be invisible otherwise.
    """


@dataclass
class ScriptedSource:
    """Satisfies `llm.Source`, answers from a queue, remembers what it was asked."""

    replies: list[str] = field(default_factory=list)
    name: str = "scripted"
    #: Every prompt handed to `complete`, in order.
    prompts: list[str] = field(default_factory=list)
    #: Every schema handed to `complete` — §12.3's constrained output, which
    #: `sources.build_request` puts in `output_config.format`.
    schemas: list[dict | None] = field(default_factory=list)
    reason: str = ""

    def available(self, cfg) -> tuple[bool, str]:
        return (not self.reason, self.reason)

    def complete(self, cfg, prompt: str, *, schema: dict | None = None) -> str:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if not self.replies:
            raise ExhaustedError(
                f"{self.name} was called {len(self.prompts)} time(s) with "
                f"{len(self.prompts) - 1} reply/replies scripted")
        return self.replies.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


# --------------------------------------------------------------------------
# the four reply shapes §12 exists to tell apart
# --------------------------------------------------------------------------


def reply(entries: list[dict], *, key: str = "selections") -> str:
    """A well-formed reply, wrapped in the prose a chat window adds.

    Wrapped deliberately: `parse_reply` scans backwards for the last balanced
    object precisely because what comes out of a paste round trip has text
    around it, and a test feeding bare JSON would not exercise that.
    """
    body = json.dumps({key: entries}, indent=2)
    return f"Happy to help. Here is the JSON you asked for:\n\n{body}\n\nLet me know."


def good(seq: int, quote: str, **extra) -> dict:
    """One selection that should survive every check."""
    return {"seq": int(seq), "quote": quote, **extra}


def hallucinated_id(quote: str, *, seq: int = 9999, **extra) -> dict:
    """§12.2: an id that does not exist. Dropped, and COUNTED.

    The id is deliberately far outside any real `segments.seq`, which is the
    whole argument for handing a model real database handles: an invented
    number has somewhere to fail.
    """
    return {"seq": int(seq), "quote": quote, **extra}


def misquoted(seq: int, quote: str, **extra) -> dict:
    """§12.3's adversarial case: a real quote, from the WRONG segment.

    HANDOFF names this as the one a lazy check misses. The words are genuinely
    in the transcript, so a check against the corpus accepts it — and §12.3's
    entire purpose is catching a model that answered about material it did not
    read.
    """
    return {"seq": int(seq), "quote": quote, **extra}


def malformed() -> str:
    """No JSON object at all. `parse_reply` returns None rather than raising."""
    return "I'd rather not produce JSON for this one, sorry."


def truncated() -> str:
    """An object that never closes — the shape a `max_tokens` cutoff produces.

    Distinct from `malformed`: there IS a brace, so a naive "find the first {"
    parser would take it and fail somewhere less obvious.
    """
    return '{"selections": [{"seq": 1, "quote": "Right, ranked. I am on'
