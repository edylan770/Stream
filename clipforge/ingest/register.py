"""`clipforge register` — stage 1 of §5.1, as a command.

§5.1 lists `register_stream` first among the pipeline stages, but it cannot be
a runnable one: every other stage reads the master path out of the `streams`
row, and this is the step that creates it. So it runs as a command and calls
`Runner.mark_external_done`, which puts a real `pipeline_stages` row in place
and lets the rest of the machinery — staleness cascade included — treat it like
any other completed stage.

What it does *not* do is copy the master. Masters are 40-55 GB (§13.1); the
database stores a path to it. The capture-side files are kilobytes and get
copied into `raw/`, because the directory they arrived in is often a scratch
directory that gets cleaned, and losing the marker log loses the operator's
live judgment calls permanently.

**Two entry points, one core.** `main(args)` is the CLI; the app shell calls
`preflight` and `register_stream` directly. Splitting them is what makes the
shell able to answer "what would this do?" *before* doing it — which matters
most for the anchor. Without `anchor.json` a stream registers with
`marker_time_base='vod'` and every marker time is then read as VOD seconds
rather than converted through §4.1's anchor. Discovering that from a warning
printed after the row is written is discovering it too late.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from clipforge import config, db, paths
from clipforge.pipeline import runner

#: Contract version of anchor.json. Documented in clipforge/capture/README.md
#: and produced by tests/fixtures/make_fixture.py. Phase 0's obs_anchor.py must
#: match; refusing an unknown version beats silently reading a field that has
#: moved.
ANCHOR_SCHEMA = 1

#: Earliest epoch-ms we will believe as a record-start anchor (2020-01-01).
#: Catches the classic mistake of writing seconds where milliseconds belong,
#: which would put every marker ~55 years before the recording started.
MIN_PLAUSIBLE_EPOCH_MS = 1_577_836_800_000


#: §4.1. Markers carry epoch milliseconds and the anchor is the only thing that
#: converts them; without it they are read as VOD seconds and every marker in
#: the stream lands somewhere else entirely.
MARKERS_WITHOUT_ANCHOR = (
    "markers.jsonl is present but anchor.json is not. §4.3 markers carry epoch "
    "milliseconds, and without the anchor there is nothing to convert them "
    "against (§4.1). Register again with --anchor once you have it."
)


class RegisterError(RuntimeError):
    pass


class AlreadyRegistered(RegisterError):
    """The id exists and `force` was not set.

    Its own type because the app shell shows this as a fact about the stream
    rather than as a failure — re-registration is a real operation, it just
    invalidates artifacts and so has to be asked for.
    """


# --------------------------------------------------------------------------
# capture-side files
# --------------------------------------------------------------------------


def find_capture_file(master: Path, explicit: str | None, *names: str) -> Path | None:
    """Explicit path wins; otherwise look beside the master."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RegisterError(f"file not found: {path}")
        return path
    for name in names:
        candidate = master.parent / name
        if candidate.is_file():
            return candidate
    return None


def read_anchor(path: Path) -> int:
    """Extract `record_start_epoch_ms` — §4.1's entire sync solution.

    One number per stream. Everything live-captured is converted through it, so
    a wrong value silently shifts every marker in the stream and nothing later
    can detect that.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegisterError(f"cannot read anchor {path}: {exc}") from exc

    schema = data.get("schema")
    if schema != ANCHOR_SCHEMA:
        raise RegisterError(
            f"{path} declares schema {schema!r}, expected {ANCHOR_SCHEMA}. "
            f"See clipforge/capture/README.md for the contract."
        )

    value = data.get("record_start_epoch_ms")
    if not isinstance(value, int) or isinstance(value, bool):
        raise RegisterError(
            f"{path}: record_start_epoch_ms must be an integer, got {value!r}"
        )
    if value < MIN_PLAUSIBLE_EPOCH_MS:
        raise RegisterError(
            f"{path}: record_start_epoch_ms is {value}, which is before 2020. "
            f"A8 requires epoch MILLISECONDS — this looks like seconds."
        )
    return value


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def default_date(master: Path) -> str:
    """The recording's date, from the master's mtime rather than today's.

    A stream recorded at 23:00 and processed at 09:00 the next morning belongs
    to the night it happened. `streams.date` is what §10.4's cross-stream
    queries group by, so getting this wrong quietly mis-sorts the archive.
    """
    return datetime.fromtimestamp(master.stat().st_mtime).date().isoformat()


def existing_ids(conn) -> set[str]:
    return {row["id"] for row in conn.execute("SELECT id FROM streams")}


def identity_hash(master: Path, anchor_ms: int | None) -> str:
    payload = json.dumps(
        {"name": master.name, "size": master.stat().st_size, "anchor": anchor_ms},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# the core: what to register, what that would do, and doing it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterSpec:
    """A registration request, independent of where it came from."""

    master: Path
    stream_id: str | None = None
    date: str | None = None
    title: str | None = None
    #: Comma-separated, as both the flag and the form field supply it.
    games: str | None = None
    notes: str | None = None
    anchor: str | None = None
    markers: str | None = None
    force: bool = False

    @classmethod
    def from_args(cls, args) -> RegisterSpec:
        return cls(
            master=Path(args.master).expanduser().resolve(),
            stream_id=args.id,
            date=args.date,
            title=args.title,
            games=args.games,
            notes=args.notes,
            anchor=args.anchor,
            markers=args.markers,
            force=bool(args.force),
        )

    def games_json(self) -> str | None:
        if not self.games:
            return None
        return json.dumps([g.strip() for g in self.games.split(",") if g.strip()])


@dataclass(frozen=True)
class Preflight:
    """Everything registration decided, before anything is written."""

    master: Path
    stream_id: str
    date: str
    already_registered: bool
    anchor_path: Path | None
    anchor_ms: int | None
    markers_path: Path | None
    marker_time_base: str
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "master": str(self.master),
            "stream_id": self.stream_id,
            "date": self.date,
            "already_registered": self.already_registered,
            "anchor": str(self.anchor_path) if self.anchor_path else None,
            "anchor_ms": self.anchor_ms,
            "markers": str(self.markers_path) if self.markers_path else None,
            "marker_time_base": self.marker_time_base,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RegisterResult:
    preflight: Preflight
    #: A `streams` row was created rather than updated.
    created: bool
    #: Something downstream depends on changed, so `completed_seq` was bumped.
    material: bool
    copied: list[str] = field(default_factory=list)

    @property
    def stream_id(self) -> str:
        return self.preflight.stream_id

    def to_json(self) -> dict:
        return {
            **self.preflight.to_json(),
            "created": self.created,
            "material": self.material,
            "copied": list(self.copied),
        }


def preflight(cfg, conn, spec: RegisterSpec) -> Preflight:
    """Resolve the id, date and capture files without writing anything.

    Raises for anything that would stop registration outright — a missing
    master, an anchor that cannot be believed — so the caller can report the
    same message whether it is a terminal or a form.
    """
    if not spec.master.is_file():
        raise RegisterError(f"master not found: {spec.master}")

    date = spec.date or default_date(spec.master)
    label = spec.title or spec.master.stem

    known = existing_ids(conn)
    if spec.stream_id:
        stream_id, already = spec.stream_id, spec.stream_id in known
    else:
        # make_stream_id already handles slugging and same-day collisions, so an
        # auto-generated id never collides.
        stream_id = paths.make_stream_id(
            date, label, known, cfg.get("ingest.stream_id_slug_max_len")
        )
        already = False

    anchor_path = find_capture_file(spec.master, spec.anchor, "anchor.json")
    markers_path = find_capture_file(spec.master, spec.markers, "markers.jsonl")
    anchor_ms = read_anchor(anchor_path) if anchor_path else None
    time_base = "epoch" if anchor_ms is not None else "vod"

    warnings = []
    if markers_path is not None and time_base == "vod":
        warnings.append(MARKERS_WITHOUT_ANCHOR)

    return Preflight(
        master=spec.master,
        stream_id=stream_id,
        date=date,
        already_registered=already,
        anchor_path=anchor_path,
        anchor_ms=anchor_ms,
        markers_path=markers_path,
        marker_time_base=time_base,
        warnings=warnings,
    )


def register_stream(cfg, conn, spec: RegisterSpec, pre: Preflight | None = None) -> RegisterResult:
    """Create or update the `streams` row and complete the external stage."""
    pre = pre or preflight(cfg, conn, spec)

    if pre.already_registered and not spec.force:
        raise AlreadyRegistered(
            f"stream {pre.stream_id!r} is already registered. Re-registering can point it "
            f"at a different master, which invalidates every artifact built from the old "
            f"one. Pass --force to update it and re-run whatever depends on what changed."
        )

    stream_paths = paths.StreamPaths(cfg.data_root, pre.stream_id).ensure()
    previous = conn.execute(
        "SELECT * FROM streams WHERE id = ?", (pre.stream_id,)
    ).fetchone()

    copied = _copy_capture_files(stream_paths, pre.anchor_path, pre.markers_path)
    _write_sources(stream_paths, pre.master, pre.anchor_path, pre.markers_path)

    with db.transaction(conn):
        conn.execute(
            """
            INSERT INTO streams
                (id, date, title, games, master_path, record_start_epoch_ms,
                 marker_time_base, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                date = excluded.date, title = excluded.title, games = excluded.games,
                master_path = excluded.master_path,
                record_start_epoch_ms = excluded.record_start_epoch_ms,
                marker_time_base = excluded.marker_time_base, notes = excluded.notes
            """,
            (pre.stream_id, pre.date, spec.title, spec.games_json(), str(pre.master),
             pre.anchor_ms, pre.marker_time_base, spec.notes),
        )

    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=pre.stream_id)

    # Only bump completed_seq when something downstream actually depends on
    # what changed. Fixing a typo in the title should not trigger a four-hour
    # re-encode, but repointing the master must.
    material = _material_change(previous, pre.master, pre.anchor_ms, pre.marker_time_base)
    if material:
        engine.mark_external_done(
            "register_stream", identity_hash(pre.master, pre.anchor_ms)
        )

    return RegisterResult(
        preflight=pre, created=previous is None, material=material, copied=copied
    )


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def add_arguments(parser) -> None:
    parser.add_argument("--master", required=True, help="the OBS recording")
    parser.add_argument("--id", help="stream id (default: <date>_<slug of title or filename>)")
    parser.add_argument("--date", help="ISO date (default: the master's modification date)")
    parser.add_argument("--title")
    parser.add_argument("--games", help="comma-separated")
    parser.add_argument("--notes")
    parser.add_argument("--anchor", help="anchor.json (default: beside the master)")
    parser.add_argument("--markers", help="markers.jsonl (default: beside the master)")
    parser.add_argument(
        "--force", action="store_true",
        help="re-register an existing stream id, re-running everything built from it",
    )
    config.add_config_arguments(parser)


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        result = register_stream(cfg, conn, RegisterSpec.from_args(args))
    except RegisterError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        conn.close()

    _report(result)
    return 0


def _material_change(previous, master: Path, anchor_ms: int | None, time_base: str) -> bool:
    if previous is None:
        return True
    return (
        previous["master_path"] != str(master)
        or previous["record_start_epoch_ms"] != anchor_ms
        or previous["marker_time_base"] != time_base
    )


def _copy_capture_files(
    stream_paths: paths.StreamPaths, anchor: Path | None, markers: Path | None
) -> list[str]:
    """Bring the small capture artifacts into the stream directory.

    The master stays where it is. These do not: they are kilobytes, they cannot
    be regenerated, and the directory they arrived in is frequently temporary.
    """
    copied = []
    if anchor is not None:
        shutil.copy2(anchor, stream_paths.raw_dir / "anchor.json")
        copied.append("anchor.json")
    if markers is not None:
        shutil.copy2(markers, stream_paths.markers_jsonl)
        copied.append("markers.jsonl")
    return copied


def _write_sources(
    stream_paths: paths.StreamPaths, master: Path, anchor: Path | None, markers: Path | None
) -> None:
    """Provenance sidecar, in place of §2.3's symlinks.

    Symlinks need Developer Mode or an elevated prompt on Windows and buy
    nothing this project needs.
    """
    stat = master.stat()
    payload = {
        "master": {
            "path": str(master),
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        },
        "anchor_source": str(anchor) if anchor else None,
        "markers_source": str(markers) if markers else None,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    stream_paths.sources_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _report(result: RegisterResult) -> None:
    pre = result.preflight
    if not result.material:
        print("  metadata only: leaving completed stages alone")

    print(f"{'registered' if result.created else 'updated'}  {pre.stream_id}")
    print(f"  master    {pre.master}")
    print(f"  date      {pre.date}")
    if pre.anchor_ms is not None:
        started = datetime.fromtimestamp(pre.anchor_ms / 1000).isoformat(timespec="seconds")
        print(f"  anchor    {pre.anchor_ms}  ({started})")
    else:
        print("  anchor    none — marker times will be read as VOD seconds")
    print(f"  markers   {'markers.jsonl' if 'markers.jsonl' in result.copied else 'none found'}")

    for warning in pre.warnings:
        print()
        print(textwrap.fill(
            warning, width=88, initial_indent="  WARNING  ", subsequent_indent=" " * 11
        ))

    if not result.created and result.material:
        print("\n  the master or anchor changed: dependent stages will re-run on the next `run`")

    print("\nnext: clipforge run " + pre.stream_id)
