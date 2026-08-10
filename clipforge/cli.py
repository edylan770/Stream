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
    "config": ("clipforge.config", "show the merged configuration and its version hash"),
    "db": ("clipforge.db.commands", "init, inspect, or relink the database"),
    "doctor": ("clipforge.doctor", "check external tools and the Python environment"),
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
