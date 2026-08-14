"""Export and finish.

Phase 1 shipped FCPXML only (§10.5) — `cmd_export`, `selection`, `fcpxml`,
`timebase`. Phase 4 adds §8's auto-finish renderer on top: `timeline`, `words`
and `ass` are the caption half, `crop` and `clips` the picture half.

Every module here raises something derived from `RenderError`, so a command can
catch one thing and print one line. Without a shared base the first failure a
new operator meets — a template name they typed wrong — arrives as a traceback,
which is exactly the moment the message matters most.
"""

from __future__ import annotations


class RenderError(RuntimeError):
    """Anything the render layer refuses to do, with a reason worth printing."""
