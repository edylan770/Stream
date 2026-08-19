"""Command-line entry point.

A thin dispatcher. Each subcommand lives in the layer module that owns it and
exposes `add_arguments(parser)` and `main(args) -> int`.

Only the module for the command actually being invoked is imported. `clipforge
--help` builds its command list from the table below, so listing the commands
costs nothing even once `score` and `review` are pulling in numpy and fastapi.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from clipforge import __version__

#: command name -> (module path, one-line help)
#: Later phases register here; the dispatcher itself never changes.
COMMANDS: dict[str, tuple[str, str]] = {
    "backup": ("clipforge.db.cmd_backup", "back up the database, and verify one (§13.2)"),
    "config": ("clipforge.config", "show the merged configuration and its version hash"),
    "db": ("clipforge.db.commands", "init, inspect, or relink the database"),
    "digest": ("clipforge.digest.cmd_digest", "chapter summaries for a digest (§9.4)"),
    "doctor": ("clipforge.doctor", "check external tools and the Python environment"),
    "export": ("clipforge.render.cmd_export", "approved moments as an FCPXML timeline (§10.5)"),
    "hook": ("clipforge.render.cmd_hook", "hook text options for rendered clips (§8.5)"),
    "metrics": ("clipforge.review.cmd_metrics", "is review fast enough? (§7.1's target)"),
    "register": ("clipforge.ingest.register", "register a recording as a stream"),
    "render": ("clipforge.render.cmd_render", "approved moments as postable clips (§8)"),
    "review": ("clipforge.review.cmd_review", "open the review UI"),
    "run": ("clipforge.pipeline.cmd_run", "run the extraction pipeline for a stream"),
    "scene-events": (
        "clipforge.extract.cmd_scene_events",
        "check an OBS log against the (UNVALIDATED) scene patterns (§5.1 stage 11)",
    ),
    "score": ("clipforge.score.cmd_score", "re-score a stream without touching extraction"),
    "search": ("clipforge.cmd_search", "find a moment by describing it (§11.6)"),
    "signals": ("clipforge.extract.cmd_signals", "inspect extracted signal series"),
    "status": ("clipforge.pipeline.cmd_status", "show pipeline state for a stream"),
    "synth-markers": (
        "clipforge.extract.cmd_synth_markers",
        "fabricate marker presses for testing (Phase 0's daemon is not built)",
    ),
}


def _peek_command(argv: Sequence[str]) -> str | None:
    """Find the subcommand in argv without parsing, so we import only that one."""
    for token in argv:
        if not token.startswith("-"):
            return token if token in COMMANDS else None
    return None


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    """Build the parser. Arguments are populated only for `command`."""
    parser = argparse.ArgumentParser(
        prog="clipforge",
        description="Stream -> clip pipeline (Phase 1).",
    )
    parser.add_argument("--version", action="version", version=f"clipforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    for name, (module_path, help_text) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        if name == command:
            importlib.import_module(module_path).add_arguments(sub)
            sub.set_defaults(_handler=f"{module_path}:main")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(_peek_command(argv))
    args = parser.parse_args(argv)

    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2

    module_path, func_name = handler.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)(args)


if __name__ == "__main__":
    sys.exit(main())
