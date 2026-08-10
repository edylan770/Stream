"""Preflight checks.

The pipeline shells out to ffmpeg in three of Phase 1's stages. Discovering
that it is missing halfway through a proxy encode is a bad way to find out, and
this project is explicitly built on one machine and run on another — so the
environment check has to be a first-class command, not a comment in a README.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass

from clipforge import __version__, ffmpeg

MIN_PYTHON = (3, 11)

#: Import name -> what it is needed for. Distribution names differ for some of
#: these (PyYAML imports as `yaml`), which is exactly why we probe imports.
REQUIRED_IMPORTS = {
    "numpy": "signal arrays, scoring grid",
    "scipy": "peak finding (§6.2)",
    "soundfile": "WAV reading for RMS extraction (§5.3)",
    "yaml": "config and weight profiles (§17)",
    "fastapi": "review UI backend (§7)",
    "uvicorn": "review UI server",
}


@dataclass
class Check:
    label: str
    ok: bool
    detail: str
    required: bool = True


def _check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return Check(
        label="python",
        ok=ok,
        detail=(
            f"{platform.python_version()} at {sys.executable}"
            + ("" if ok else f"  (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
        ),
    )


def _check_binary(name: str, override: str | None) -> Check:
    info = ffmpeg.find_binary(name, override)
    if info.ok:
        return Check(label=name, ok=True, detail=f"{info.version}  [{info.source}: {info.path}]")
    return Check(
        label=name,
        ok=False,
        detail="not found on PATH, in CLIPFORGE_%s, or in config" % name.upper(),
    )


def _check_import(module: str, purpose: str) -> Check:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        return Check(label=module, ok=False, detail=f"{purpose} — import failed: {exc}")
    version = getattr(mod, "__version__", "?")
    return Check(label=module, ok=True, detail=f"{version}  ({purpose})")


def _check_data_root(root) -> Check:
    """The data root must exist or be creatable, and be writable.

    On the streaming PC this points at a large external drive. Finding out it is
    unplugged at the start of a run beats finding out four minutes into a proxy
    encode.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".clipforge-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(label="data_root", ok=False, detail=f"{root} — not writable: {exc}")
    return Check(label="data_root", ok=True, detail=str(root))


def run_checks(
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    cfg=None,
) -> list[Check]:
    """Run every preflight check and return the results in display order.

    Explicit arguments win over config, so `doctor --ffmpeg <path>` can be used
    to find a working value before committing it to local.yaml.
    """
    checks = [_check_python()]

    if cfg is not None:
        ffmpeg_path = ffmpeg_path or cfg.ffmpeg_override
        ffprobe_path = ffprobe_path or cfg.ffprobe_override

    checks.append(_check_binary("ffmpeg", ffmpeg_path))
    checks.append(_check_binary("ffprobe", ffprobe_path))
    checks.extend(_check_import(mod, why) for mod, why in REQUIRED_IMPORTS.items())

    if cfg is not None:
        checks.append(_check_data_root(cfg.data_root))
    return checks


def report(checks: list[Check]) -> int:
    """Print the checks. Return a process exit code."""
    width = max(len(c.label) for c in checks)
    failed_required = 0
    print(f"clipforge {__version__}\n")
    for check in checks:
        if check.ok:
            mark = "ok  "
        elif check.required:
            mark = "FAIL"
            failed_required += 1
        else:
            mark = "warn"
        print(f"  [{mark}] {check.label.ljust(width)}  {check.detail}")

    if failed_required:
        print(
            f"\n{failed_required} required check(s) failed. "
            "Missing ffmpeg/ffprobe blocks the probe, proxy, and audio_split stages."
        )
        return 1
    print("\nAll required checks passed.")
    return 0


def add_arguments(parser) -> None:
    from clipforge import config

    config.add_config_arguments(parser)
    parser.add_argument("--ffmpeg", help="explicit path to the ffmpeg binary or its directory")
    parser.add_argument("--ffprobe", help="explicit path to the ffprobe binary or its directory")


def main(args) -> int:
    from clipforge import config

    # A broken config is itself a preflight failure, and the most likely one on
    # a freshly cloned machine. Report it as such rather than tracebacking.
    try:
        cfg = config.from_args(args)
    except config.ConfigError as exc:
        checks = run_checks(args.ffmpeg, args.ffprobe)
        checks.append(Check(label="config", ok=False, detail=str(exc)))
        return report(checks)

    checks = run_checks(args.ffmpeg, args.ffprobe, cfg)
    checks.insert(1, Check(label="config", ok=True, detail=f"version {cfg.version}"))
    return report(checks)
