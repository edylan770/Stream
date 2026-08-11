"""Layered configuration.

§17 requires every tunable to live in a config file and never in code. This
module is what makes that enforceable: there is one merged view, every value
knows which layer produced it, and the scoring-relevant subset hashes to a
`config_version` that gets stamped onto every candidate row.

Layers, lowest priority first:

  1. ``clipforge/config/clipforge.yaml``  — packaged defaults, committed
  2. ``clipforge/config/local.yaml``      — machine-specific, gitignored
  3. ``--config FILE`` / ``$CLIPFORGE_CONFIG``
  4. ``--set key.path=value`` flags

Relative paths resolve against the *project root*, not the working directory,
so `clipforge run` means the same thing from any subdirectory.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent / "config"
DEFAULTS_FILE = CONFIG_DIR / "clipforge.yaml"
LOCAL_FILE = CONFIG_DIR / "local.yaml"
PROFILES_DIR = CONFIG_DIR / "profiles"
FEATURE_SCHEMA_FILE = CONFIG_DIR / "feature_schema.yaml"

ENV_CONFIG = "CLIPFORGE_CONFIG"

#: Layout version of the config files themselves — not the tuning values, the
#: shape. Bumped when a key is renamed, moved, or removed in a way that would
#: make an older `local.yaml` merge into something wrong rather than something
#: that fails. Checked on load so a stale override is an error rather than a
#: silently ignored file.
CONFIG_SCHEMA = 1

#: Subtrees that determine what a scoring run produces. Changing any of these
#: must produce a new `config_version`; changing `review.port` must not.
VERSIONED_SUBTREES = ("extract", "score")


class ConfigError(RuntimeError):
    """Configuration is missing, malformed, or internally inconsistent."""


# --------------------------------------------------------------------------
# project root
# --------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk upward looking for this project's ``pyproject.toml``.

    Anchoring relative paths to the repo rather than to `os.getcwd()` is what
    keeps `clipforge run` from silently addressing a different database
    depending on which directory it was invoked from.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8")
            except OSError:
                continue
            if 'name = "clipforge"' in text:
                return candidate
    return None


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------


def _flatten(data: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _flatten(value, f"{dotted}.")
        else:
            yield dotted, value


def _deep_merge(
    base: dict[str, Any],
    overlay: Mapping[str, Any],
    provenance: dict[str, str],
    layer: str,
    prefix: str = "",
) -> dict[str, Any]:
    """Merge `overlay` into `base` in place, recording provenance per leaf key."""
    for key, value in overlay.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_merge(base[key], value, provenance, layer, f"{dotted}.")
        else:
            base[key] = value
            if isinstance(value, Mapping):
                for sub_key, _ in _flatten(value, f"{dotted}."):
                    provenance[sub_key] = layer
            else:
                provenance[dotted] = layer
    return base


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value`` from a ``--set`` flag.

    Values go through the YAML scalar parser, so `true`, `12`, `1.5`, `null`,
    and `[1, 2]` all arrive with the type you would get from the config file.
    """
    if "=" not in text:
        raise ConfigError(f"--set expects key.path=value, got {text!r}")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"--set expects a key before '=', got {text!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return key, value


def _nest(dotted: str, value: Any) -> dict[str, Any]:
    parts = dotted.split(".")
    result: dict[str, Any] = {}
    cursor = result
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return result


# --------------------------------------------------------------------------
# canonicalization and hashing
# --------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Normalize for hashing.

    Numbers collapse to a fixed-precision string so that YAML `1`, `1.0`, and
    `1.00` are the same version, while a genuine weight change is not. `bool` is
    checked before `int` because in Python it is one.
    """
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return format(float(value), ".12g")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# profiles and feature schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """A weight profile (§6.5)."""

    name: str
    weights: dict[str, float]
    description: str = ""
    source: Path | None = None

    def weight(self, signal: str) -> float:
        """Weight for `signal`; zero when the profile does not mention it."""
        return float(self.weights.get(signal, 0.0))


def load_profile(name: str, profiles_dir: Path | None = None) -> Profile:
    directory = profiles_dir or PROFILES_DIR
    path = directory / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.yaml"))
        raise ConfigError(
            f"unknown weight profile {name!r}. Available: {', '.join(available) or 'none'}"
        )
    data = _read_yaml(path)
    weights = data.get("weights") or {}
    if not isinstance(weights, Mapping) or not weights:
        raise ConfigError(f"profile {path} has no weights")
    bad = {k: v for k, v in weights.items() if not isinstance(v, (int, float))}
    if bad:
        raise ConfigError(f"profile {path} has non-numeric weights: {sorted(bad)}")
    return Profile(
        name=str(data.get("name", name)),
        weights={str(k): float(v) for k, v in weights.items()},
        description=str(data.get("description", "")).strip(),
        source=path,
    )


@dataclass(frozen=True)
class FeatureSchema:
    """The declared full set of signals for A9 feature vectors."""

    version: int
    signals: dict[str, dict[str, Any]]

    @property
    def keys(self) -> list[str]:
        return list(self.signals)

    def empty_vector(self) -> dict[str, float | None]:
        """A vector with every declared signal present and null.

        A9 wants full vectors from run one. Phase 1 fills three of these; the
        rest stay null so a Phase 1 vector and a Phase 3 vector have the same
        shape and can be compared directly.
        """
        return dict.fromkeys(self.signals, None)


def load_feature_schema(path: Path | None = None) -> FeatureSchema:
    data = _read_yaml(path or FEATURE_SCHEMA_FILE)
    signals: dict[str, dict[str, Any]] = {}
    for group in ("continuous", "events", "composite"):
        for name, meta in (data.get(group) or {}).items():
            if name in signals:
                raise ConfigError(f"feature schema declares {name!r} more than once")
            entry = dict(meta or {})
            entry["group"] = group
            signals[str(name)] = entry
    if not signals:
        raise ConfigError("feature schema declares no signals")
    return FeatureSchema(version=int(data.get("version", 0)), signals=signals)


# --------------------------------------------------------------------------
# the merged config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    provenance: dict[str, str]
    layers: list[tuple[str, Path | None]]
    project_root: Path
    profile: Profile
    feature_schema: FeatureSchema

    # -- access ------------------------------------------------------------

    _MISSING = object()

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        cursor: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                if default is Config._MISSING:
                    raise ConfigError(f"missing config key: {dotted}")
                return default
            cursor = cursor[part]
        return cursor

    def path(self, dotted: str, default: Any = _MISSING) -> Path | None:
        """Resolve a config value to an absolute path (None stays None)."""
        value = self.get(dotted, default)
        if value is None:
            return None
        return self.resolve(value)

    def resolve(self, value: str | os.PathLike[str]) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (self.project_root / candidate).resolve()

    # -- well-known paths --------------------------------------------------

    @property
    def data_root(self) -> Path:
        resolved = self.path("paths.data_root")
        assert resolved is not None  # schema guarantees a value
        return resolved

    @property
    def db_path(self) -> Path:
        explicit = self.path("paths.db", None)
        return explicit if explicit is not None else self.data_root / "clipforge.db"

    @property
    def ffmpeg_override(self) -> str | None:
        value = self.get("paths.ffmpeg", None)
        return str(self.resolve(value)) if value else None

    @property
    def ffprobe_override(self) -> str | None:
        value = self.get("paths.ffprobe", None)
        return str(self.resolve(value)) if value else None

    # -- versioning --------------------------------------------------------

    def version_payload(self) -> dict[str, Any]:
        """Exactly the inputs that determine a scoring run's output."""
        return {
            "subtrees": {name: self.get(name, {}) for name in VERSIONED_SUBTREES},
            "profile": {"name": self.profile.name, "weights": self.profile.weights},
            "feature_schema_version": self.feature_schema.version,
        }

    @property
    def version(self) -> str:
        """``<profile>@<sha256[:8]>`` — stamped onto every candidate row.

        §3.2 describes `config_version` as "which weight profile produced this",
        which cannot distinguish two different weight sets that share a name.
        Hashing the effective config can.
        """
        digest = hashlib.sha256(canonical_json(self.version_payload()).encode()).hexdigest()
        return f"{self.profile.name}@{digest[:8]}"

    # -- display -----------------------------------------------------------

    def flattened(self) -> list[tuple[str, Any, str]]:
        return [
            (key, value, self.provenance.get(key, "default"))
            for key, value in sorted(_flatten(self.data))
        ]


def load(
    config_file: str | os.PathLike[str] | None = None,
    overrides: list[str] | None = None,
    profile: str | None = None,
    start_dir: Path | None = None,
) -> Config:
    """Build the merged configuration."""
    if not DEFAULTS_FILE.is_file():
        raise ConfigError(f"packaged defaults missing at {DEFAULTS_FILE}")

    provenance: dict[str, str] = {}
    layers: list[tuple[str, Path | None]] = []

    merged: dict[str, Any] = {}
    _deep_merge(merged, _read_yaml(DEFAULTS_FILE), provenance, "default")
    layers.append(("default", DEFAULTS_FILE))

    if LOCAL_FILE.is_file():
        _deep_merge(merged, _read_yaml(LOCAL_FILE), provenance, "local")
        layers.append(("local", LOCAL_FILE))

    explicit = config_file or os.environ.get(ENV_CONFIG)
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_file():
            raise ConfigError(f"config file not found: {explicit_path}")
        _deep_merge(merged, _read_yaml(explicit_path), provenance, "file")
        layers.append(("file", explicit_path))

    if overrides:
        flags: dict[str, Any] = {}
        for item in overrides:
            key, value = parse_override(item)
            _deep_merge(flags, _nest(key, value), {}, "flag")
        _deep_merge(merged, flags, provenance, "flag")
        layers.append(("flag", None))

    root = find_project_root(start_dir)
    if root is None:
        root = Path.cwd().resolve()
        print(
            f"warning: no clipforge project root found; resolving relative paths "
            f"against the working directory ({root}). Set an absolute "
            f"paths.data_root in local.yaml to make this unambiguous."
        )

    found_schema = merged.get("config_schema")
    if found_schema != CONFIG_SCHEMA:
        origin = provenance.get("config_schema", "default")
        raise ConfigError(
            f"config_schema is {found_schema!r} (from the {origin} layer), but this "
            f"build expects {CONFIG_SCHEMA}. A config written for a different layout "
            f"would merge into something subtly wrong rather than failing, so it is "
            f"rejected. Compare your local.yaml against "
            f"clipforge/config/local.yaml.example."
        )

    profile_name = profile or merged.get("score", {}).get("profile")
    if not profile_name:
        raise ConfigError("score.profile is not set")

    return Config(
        data=merged,
        provenance=provenance,
        layers=layers,
        project_root=root,
        profile=load_profile(str(profile_name)),
        feature_schema=load_feature_schema(),
    )


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------


def add_config_arguments(parser) -> None:
    """Flags every command that reads config should accept."""
    parser.add_argument(
        "--config",
        dest="config_file",
        metavar="FILE",
        help=f"extra config file layered above local.yaml (or ${ENV_CONFIG})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set score.window.min_window_s=6",
    )
    parser.add_argument("--profile", help="weight profile name (default: score.profile)")


def from_args(args) -> Config:
    return load(
        config_file=getattr(args, "config_file", None),
        overrides=getattr(args, "overrides", None),
        profile=getattr(args, "profile", None),
    )


def add_arguments(parser) -> None:
    parser.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=("show", "version"),
        help="show the merged config (default), or print just the config_version",
    )
    add_config_arguments(parser)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def main(args) -> int:
    cfg = from_args(args)

    if args.action == "version":
        print(cfg.version)
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    "project_root": str(cfg.project_root),
                    "layers": [
                        {"name": name, "path": str(path) if path else None}
                        for name, path in cfg.layers
                    ],
                    "config_version": cfg.version,
                    "profile": {
                        "name": cfg.profile.name,
                        "weights": cfg.profile.weights,
                        "source": str(cfg.profile.source),
                    },
                    "feature_schema_version": cfg.feature_schema.version,
                    "resolved": {
                        "data_root": str(cfg.data_root),
                        "db": str(cfg.db_path),
                    },
                    "values": {key: value for key, value, _ in cfg.flattened()},
                    "provenance": cfg.provenance,
                },
                indent=2,
            )
        )
        return 0

    print(f"project root     {cfg.project_root}")
    for name, path in cfg.layers:
        print(f"layer            {name:<8} {path if path else '(command line)'}")
    print(f"profile          {cfg.profile.name}  ({cfg.profile.source})")
    print(f"feature schema   v{cfg.feature_schema.version}  ({len(cfg.feature_schema.keys)} signals)")
    print(f"config_version   {cfg.version}")
    print()
    print("resolved paths")
    print(f"  data_root      {cfg.data_root}")
    print(f"  db             {cfg.db_path}")
    for name in ("ffmpeg", "ffprobe"):
        override = getattr(cfg, f"{name}_override")
        print(f"  {name:<14} {override or '(env or PATH)'}")
    print()

    rows = cfg.flattened()
    width = max(len(key) for key, _, _ in rows)
    print(f"{'key'.ljust(width)}  {'value':<24} source")
    for key, value, source in rows:
        if value is None:
            rendered = "null"
        elif isinstance(value, (list, dict)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        # Anything not coming from the packaged defaults is what you actually
        # want to see when debugging a machine.
        flag = "" if source == "default" else "  <--"
        print(f"{key.ljust(width)}  {rendered:<24} {source}{flag}")

    print()
    print("weights")
    for signal, weight in sorted(cfg.profile.weights.items(), key=lambda kv: -kv[1]):
        print(f"  {signal:<20} {weight}")
    return 0
