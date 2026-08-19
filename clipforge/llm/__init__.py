"""§12's rules and the transports that carry them, for every LLM call site.

§12 is four rules -- ids not timestamps, constrained JSON, a verbatim quote per
selection, unknown ids dropped and counted -- and they apply identically to
§8.5's hooks, §9.4's digest map, §10.3's ground pass and §11.1's cluster labels.
They were written once, inside `render/hooks.py`, against `exports.id`. This
package is that code with the hook-shaped parts left behind, so the next three
call sites share one validator instead of growing three.

    validate.py   §12.2 and §12.3, plus the tolerant reply parser
    sources.py    where an answer comes from: a person, or the API
    prompts.py    prompt text, in config, with a hash to store beside output

`render/hooks.py` re-exports what it used to define, so nothing that imported
it has to change -- and `tests/test_hooks.py` passes unmodified, which is the
only evidence that this moved code rather than changing it.

`prompts` is deliberately NOT re-exported here: it holds a `render`, and this
package is imported from inside `clipforge/render/`, where a bare `render`
would be two different things one line apart. Reach it as
`from clipforge.llm import prompts`.
"""

from clipforge.llm.sources import (
    AnthropicSource,
    LLMError,
    ManualSource,
    Source,
    source_for,
    to_clipboard,
)
from clipforge.llm.validate import (
    INVALID_ID_METRIC,
    Selection,
    Validated,
    normalise,
    parse_reply,
    record,
    validate_selections,
)

__all__ = [
    "INVALID_ID_METRIC",
    "AnthropicSource",
    "LLMError",
    "ManualSource",
    "Selection",
    "Source",
    "Validated",
    "normalise",
    "parse_reply",
    "record",
    "source_for",
    "to_clipboard",
    "validate_selections",
]
