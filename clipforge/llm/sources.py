"""Where a model's answer comes from -- a person with a browser, or an API.

§2.2 puts reasoning through an external frontier model and §12.4 prices a whole
stream at cents. The operator does not want a paid key yet, so **the paste
round trip is the product and the API is the option**, not the other way round.

Both are the same shape as `StageSpec.available` and the Ollama check in
`extract/embeddings.py`: a source reports whether it can run *and why not*,
rather than failing at the moment it is used. That is what lets `clipforge
hook` and `clipforge doctor` say "no key" instead of raising.

**The two paths share one validator.** `AnthropicSource` can constrain the
reply schema server-side, which the paste path cannot; it still returns plain
text that goes through `validate.parse_reply` and `validate.validate_selections`
exactly as a pasted reply does. A §12 check that only ran on one transport
would drift the moment a key appeared.

NOTHING IN `AnthropicSource` HAS EVER RUN. There is no key on this machine and
the `anthropic` package is not installed, so what is asserted about it is that
it refuses cleanly and that it feeds the shared validator -- not that a single
request shape is correct. The request is built in one function and every knob
is config, so correcting it is an edit rather than a rewrite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Protocol

#: Config default for `llm.anthropic.model`. §12.4's budget is cents per
#: stream, so the capable model is the right default -- a digest is one call.
DEFAULT_MODEL = "claude-opus-5"


class LLMError(RuntimeError):
    """A source is misconfigured, unavailable, or answered unusably."""


class Source(Protocol):
    """Where model output comes from."""

    name: str

    def available(self, cfg) -> tuple[bool, str]:
        """(usable, why not). The reason is shown to the operator verbatim."""
        ...

    def complete(self, cfg, prompt: str, *, schema: dict | None = None) -> str:
        """The model's reply as text, for `validate.parse_reply` to read."""
        ...


@dataclass
class ManualSource:
    """The paste round trip: a prompt out, a reply back, validated here."""

    name: str = "manual"

    def available(self, cfg) -> tuple[bool, str]:
        # Always available: it needs a person and a browser, not a credential.
        return True, ""

    def complete(self, cfg, prompt: str, *, schema: dict | None = None) -> str:
        raise LLMError(
            "the manual source cannot call anything -- write the prompt, paste "
            "it into a frontier model, and hand the reply back with --apply."
        )


@dataclass
class AnthropicSource:
    """The API path. Written, unavailable, and honest about both.

    `available` reports the missing package and the missing key **separately**,
    because they need different fixes and a single "not configured" would hide
    which one is wrong.
    """

    name: str = "anthropic"

    def available(self, cfg) -> tuple[bool, str]:
        reasons = []
        if find_spec("anthropic") is None:
            reasons.append(
                'the anthropic package is not installed (pip install -e ".[llm]")')
        variable = str(cfg.get("llm.anthropic.api_key_env", "ANTHROPIC_API_KEY"))
        if not os.environ.get(variable):
            reasons.append(
                f"needs a key in ${variable}. §12.4 prices a stream at roughly "
                f"$0.10-0.30 if you want one")
        if reasons:
            return False, "; ".join(reasons)
        return True, ""

    def complete(self, cfg, prompt: str, *, schema: dict | None = None) -> str:
        usable, why = self.available(cfg)
        if not usable:
            raise LLMError(f"{self.name}: {why}")

        # Imported here, never at module scope: `anthropic` is an optional
        # extra and every existing command has to keep starting without it.
        # Same rule pyproject.toml already states for librosa.
        import anthropic

        request = build_request(cfg, prompt, schema=schema)
        client = anthropic.Anthropic()
        # Streaming, not a plain create: a digest map is long in and long out,
        # and the SDK refuses a non-streaming request it estimates will outlive
        # the HTTP timeout. `get_final_message` gives back the whole thing.
        # The beta endpoint only when a beta flag is actually in play.
        endpoint = client.beta.messages if "betas" in request else client.messages
        with endpoint.stream(**request) as stream:
            message = stream.get_final_message()

        return read_text(message)


def build_request(cfg, prompt: str, *, schema: dict | None = None) -> dict:
    """Every knob in one dict, so a wrong one is a config edit.

    Adaptive thinking rather than a token budget, and `effort` for depth.
    `output_config.format` constrains the reply to a JSON schema server-side
    when the caller has one -- §12.3's "constrained JSON schema on every call",
    enforced by the API rather than merely requested in the prompt. The reply
    still goes through the shared validator afterwards.
    """
    output_config: dict = {"effort": str(cfg.get("llm.anthropic.effort", "high"))}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    request: dict = {
        "model": str(cfg.get("llm.anthropic.model", DEFAULT_MODEL)),
        "max_tokens": int(cfg.get("llm.anthropic.max_tokens", 16000)),
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
        "messages": [{"role": "user", "content": prompt}],
    }

    # A safety classifier can decline a request outright, and a stopped digest
    # is worse than one answered by a slightly smaller model. Config so it can
    # be dropped in one line if the beta moves.
    fallbacks = str(cfg.get("llm.anthropic.fallbacks", "") or "")
    if fallbacks:
        request["fallbacks"] = fallbacks
        request["betas"] = [str(cfg.get("llm.anthropic.fallback_beta",
                                        "server-side-fallback-2026-07-01"))]
    return request


def read_text(message) -> str:
    """The reply's text, after checking it is a reply at all.

    `stop_reason` is inspected before `content` because a refusal returns a
    perfectly successful response with nothing in it, and indexing straight
    into `content[0]` turns that into an IndexError three layers from the
    cause.
    """
    if getattr(message, "stop_reason", None) == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) or "unstated"
        raise LLMError(
            f"the model declined this request ({category}). Nothing was "
            f"returned; the prompt is on disk if you want to read it."
        )
    parts = [block.text for block in getattr(message, "content", [])
             if getattr(block, "type", None) == "text"]
    if not parts:
        raise LLMError(
            f"no text in the reply (stop_reason "
            f"{getattr(message, 'stop_reason', None)!r})")
    return "\n".join(parts)


#: Every source that exists. `source_for` refuses anything else by name rather
#: than falling back to the default, because silently using a different model
#: than the one configured is the failure nobody notices.
SOURCES: dict[str, type] = {
    "manual": ManualSource,
    "anthropic": AnthropicSource,
}


def source_for(cfg, key: str = "llm.source") -> Source:
    """The configured source, whether or not it can run.

    Returning an unavailable source rather than raising is deliberate: the
    caller gets to decide whether an unusable source is fatal, and gets a
    reason to print either way.
    """
    wanted = str(cfg.get(key, "manual"))
    if wanted not in SOURCES:
        raise LLMError(
            f"no source {wanted!r}. Built: {', '.join(sorted(SOURCES))}.")
    return SOURCES[wanted]()


def to_clipboard(text: str) -> bool:
    """Best effort. Returns whether it worked, and never raises."""
    command = ["clip"] if sys.platform == "win32" else ["pbcopy"]
    try:
        subprocess.run(command, input=text.encode("utf-8"), check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
