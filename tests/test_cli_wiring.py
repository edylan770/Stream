"""CLI argument wiring.

Nested subcommands and shared flags interact badly in argparse, and the failure
is always at the point of use rather than at import — so it gets a test rather
than a careful reading.
"""

from __future__ import annotations

import pytest

from clipforge import cli, config


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_every_registered_command_builds_its_parser(command):
    parser = cli.build_parser(command)
    assert parser is not None


@pytest.mark.parametrize(
    "argv",
    [
        ["db", "init", "--set", "review.port=1"],
        ["db", "info", "--set", "review.port=1"],
        ["db", "schema", "--set", "review.port=1"],
        ["db", "relink", "--from", "a", "--to", "b", "--set", "review.port=1"],
    ],
)
def test_config_flags_work_after_a_nested_subcommand(argv):
    """`clipforge db init --set x=y` is the form anyone would type.

    argparse hands everything after `init` to the sub-parser, so a --set
    declared only on the `db` parser makes this an unrecognized argument.
    """
    parser = cli.build_parser("db")
    args = parser.parse_args(argv)
    assert args.overrides == ["review.port=1"]


def test_nested_subcommand_config_does_not_shadow_the_parent():
    """The other half of the trap: declaring --set on both parent and child
    lets the child's `None` default overwrite a value parsed by the parent."""
    parser = cli.build_parser("db")
    args = parser.parse_args(["db", "init"])
    assert getattr(args, "overrides", None) is None
    assert config.from_args(args).get("review.port") == 8765


@pytest.mark.parametrize("command", ["run", "status"])
def test_pipeline_commands_accept_config_flags(command):
    parser = cli.build_parser(command)
    args = parser.parse_args([command, "some_stream", "--set", "review.port=1"])
    assert args.stream_id == "some_stream"
    assert args.overrides == ["review.port=1"]


def test_status_stream_id_is_optional():
    parser = cli.build_parser("status")
    assert parser.parse_args(["status"]).stream_id is None


def test_run_selection_flags_parse():
    parser = cli.build_parser("run")
    args = parser.parse_args(
        ["run", "s1", "--only", "probe", "--force", "proxy", "--dry-run", "--no-cascade"]
    )
    assert (args.only, args.force, args.dry_run, args.no_cascade) == (
        "probe", ["proxy"], True, True
    )


def test_config_schema_mismatch_is_rejected(tmp_path):
    """The key existed but nothing read it until now — a stale local.yaml would
    merge into something subtly wrong instead of failing."""
    import yaml

    stale = tmp_path / "stale.yaml"
    stale.write_text(yaml.safe_dump({"config_schema": 99}), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="config_schema"):
        config.load(config_file=stale)


def test_current_config_schema_loads():
    assert config.load().get("config_schema") == config.CONFIG_SCHEMA
