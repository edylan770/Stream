"""Layer A (§4) — the capture scripts, tested without OBS or a keyboard.

Each daemon is a source, a clock and a sink. The sources are the only parts
that need real hardware, so the tests drive everything else directly and assert
the exact bytes against `capture/contract.py`.

The most valuable test in this file is the boring one: that the daemon, the
fixture generator and `register` all agree about the contract. Until now that
agreement was maintained by a note in a README.
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import pytest

from clipforge.capture import contract, input_logger, marker_daemon, obs_anchor
from clipforge.ingest import register

CAPTURE_DIR = Path(contract.__file__).parent


class FakeClock:
    """Epoch ms under the test's control."""

    def __init__(self, start: int = 1_800_000_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += int(seconds * 1000)


# --------------------------------------------------------------------------
# the contract is one definition, not three
# --------------------------------------------------------------------------


def test_register_and_the_contract_agree(fixture_dir):
    """The three-way agreement, asserted rather than documented."""
    assert register.ANCHOR_SCHEMA == contract.ANCHOR_SCHEMA
    assert register.MIN_PLAUSIBLE_EPOCH_MS == contract.MIN_PLAUSIBLE_EPOCH_MS


def test_the_fixture_writes_what_register_reads(fixture_dir, manifest):
    """make_fixture goes through contract.py now, so this is the round trip the
    real capture side will take."""
    anchor = fixture_dir / manifest["anchor"]
    payload = json.loads(anchor.read_text())
    assert payload["schema"] == contract.ANCHOR_SCHEMA
    assert payload["source"] in contract.ANCHOR_SOURCES
    assert register.read_anchor(anchor) == payload["record_start_epoch_ms"]


def test_a_daemon_written_anchor_is_readable_by_register(tmp_path):
    """The end of the chain that has never been closed: what obs_anchor writes,
    register accepts."""
    writer = obs_anchor.AnchorWriter(clock=FakeClock(), log=lambda *_: None)
    recording = tmp_path / "rec" / "stream.mkv"
    recording.parent.mkdir(parents=True)

    written = writer.record_started(str(recording))
    assert register.read_anchor(written) == FakeClock().now


def test_a_daemon_written_marker_log_parses_as_events(tmp_path):
    """And what marker_daemon writes, the marker_events stage can read — the
    other end of the chain that has never been closed."""
    from clipforge.extract import markers as marker_stage

    clock = FakeClock()
    start = clock.now
    path = tmp_path / "markers.jsonl"
    with contract.JsonlSink(path) as sink:
        daemon = marker_daemon.MarkerDaemon(sink=sink, clock=clock, log=lambda *_: None)
        daemon.press(contract.MARKER_DEFINITE)
        clock.advance(10)
        daemon.press(contract.MARKER_MAYBE)

    parsed = marker_stage.parse(
        path, record_start_epoch_ms=start, duration_s=60.0, retro_offset_s=0.0
    )
    assert not parsed.malformed
    assert [event["kind"] for event in parsed.events] == [
        contract.MARKER_DEFINITE, contract.MARKER_MAYBE
    ]
    assert [event["t"] for event in parsed.events] == [0.0, 10.0]


def test_the_marker_stage_and_the_daemon_share_one_kind_list():
    from clipforge.extract import markers as marker_stage

    assert marker_stage.MARKER_KINDS is contract.MARKER_KINDS


# --------------------------------------------------------------------------
# §2.1: capture must run without the application
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", sorted(p.name for p in CAPTURE_DIR.glob("*.py") if p.name != "__init__.py")
)
def test_capture_imports_nothing_from_the_rest_of_clipforge(module):
    """§2.1: "Layer A must be independently runnable and must not depend on the
    application being installed or functional."

    So clipforge/capture/ can be copied to the streaming PC on its own. Only
    `clipforge.capture.*` and the standard library (plus pynput/obsws, imported
    lazily inside the functions that need them) may be reached."""
    tree = ast.parse((CAPTURE_DIR / module).read_text(encoding="utf-8"))
    reached = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.append(node.module)
        elif isinstance(node, ast.Import):
            reached.extend(alias.name for alias in node.names)

    for name in reached:
        if name.startswith("clipforge"):
            assert name.startswith("clipforge.capture"), (
                f"{module} imports {name}, which breaks §2.1's independence"
            )


def test_contract_needs_only_the_standard_library():
    tree = ast.parse((CAPTURE_DIR / "contract.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module in {"__future__", "pathlib"}, node.module


# --------------------------------------------------------------------------
# anchor.json (§4.1)
# --------------------------------------------------------------------------


def test_the_anchor_lands_beside_the_recording(tmp_path):
    """Not in the spec, and the highest-value decision in Phase 0:
    register.find_capture_file already looks beside the master, so this makes
    `clipforge register --master <rec>` work with no flags."""
    recording = tmp_path / "2026-08-14 Marvel.mkv"
    recording.write_bytes(b"x")

    writer = obs_anchor.AnchorWriter(clock=FakeClock(), log=lambda *_: None)
    written = writer.record_started(str(recording))

    assert written == recording.parent / contract.ANCHOR_FILENAME
    assert register.find_capture_file(recording, None, "anchor.json") == written


def test_two_recordings_get_two_anchors(tmp_path):
    """OBS stopped and restarted mid-session. Writing beside each recording
    means the second does not overwrite the first."""
    clock = FakeClock()
    writer = obs_anchor.AnchorWriter(clock=clock, log=lambda *_: None)

    first = tmp_path / "a" / "one.mkv"
    second = tmp_path / "b" / "two.mkv"
    for path in (first, second):
        path.parent.mkdir(parents=True)

    writer.record_started(str(first))
    clock.advance(600)
    writer.record_started(str(second))

    assert register.read_anchor(first.parent / contract.ANCHOR_FILENAME) != \
        register.read_anchor(second.parent / contract.ANCHOR_FILENAME)


def test_the_record_started_event_is_what_triggers_a_write(tmp_path):
    class Event:
        output_state = obs_anchor.STATE_STARTED
        output_active = True
        output_path = str(tmp_path / "stream.mkv")

    writer = obs_anchor.AnchorWriter(clock=FakeClock(), log=lambda *_: None)
    assert obs_anchor.handle_event(writer, Event()) is not None


def test_a_stop_event_writes_nothing(tmp_path):
    class Event:
        output_state = "OBS_WEBSOCKET_OUTPUT_STOPPED"
        output_active = False
        output_path = str(tmp_path / "stream.mkv")

    writer = obs_anchor.AnchorWriter(clock=FakeClock(), log=lambda *_: None)
    assert obs_anchor.handle_event(writer, Event()) is None
    assert writer.written == []


def test_a_start_with_no_path_and_no_fallback_warns_rather_than_crashing(tmp_path):
    """Capture must never fail in a way that could disturb OBS (§4.5)."""
    said = []
    writer = obs_anchor.AnchorWriter(clock=FakeClock(), log=said.append)
    assert writer.record_started(None) is None
    assert any("WARNING" in line for line in said)


def test_seconds_where_milliseconds_belong_is_refused():
    """A8, enforced at the point of writing as well as at ingest."""
    with pytest.raises(ValueError, match="MILLISECONDS"):
        contract.anchor_payload(1_755_123_456, contract.SOURCE_OBS_WEBSOCKET)


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="source"):
        contract.anchor_payload(1_800_000_000_000, "vibes")


def test_the_hotkey_fallback_records_its_own_error_bar(tmp_path):
    """§4.1's fallback writes when the script ran, not when OBS started, so the
    gap between the two fields is the only evidence of how good the number is."""
    payload = contract.anchor_payload(
        1_800_000_000_000, contract.SOURCE_HOTKEY_SCRIPT, written_at_epoch_ms=1_800_000_000_450
    )
    assert payload["written_at_epoch_ms"] - payload["record_start_epoch_ms"] == 450


def test_the_anchor_is_written_atomically(tmp_path):
    """A half-written anchor that still parses is the worst possible artifact."""
    target = tmp_path / contract.ANCHOR_FILENAME
    contract.write_anchor(target, 1_800_000_000_000, contract.SOURCE_SYNTHETIC)
    assert list(tmp_path.iterdir()) == [target]      # no temp left behind


# --------------------------------------------------------------------------
# markers.jsonl (§4.3)
# --------------------------------------------------------------------------


def test_a_press_writes_exactly_the_two_contract_keys(tmp_path):
    """§4.3: no VOD times, no local timestamps, no metadata — the daemon stays
    dumb enough that it cannot fail in an interesting way."""
    clock = FakeClock()
    path = tmp_path / "m.jsonl"
    with contract.JsonlSink(path) as sink:
        marker_daemon.MarkerDaemon(sink=sink, clock=clock, log=lambda *_: None).press(
            contract.MARKER_DEFINITE
        )

    record = json.loads(path.read_text().strip())
    assert record == {"epoch_ms": clock.now, "kind": contract.MARKER_DEFINITE}


def test_the_retro_offset_is_not_applied_here(tmp_path):
    """§4.3 says default the window anchor to t-20, but that is a §17 tunable —
    so the file records the observation and scoring decides what it implies."""
    clock = FakeClock()
    with contract.JsonlSink(tmp_path / "m.jsonl") as sink:
        marker_daemon.MarkerDaemon(sink=sink, clock=clock, log=lambda *_: None).press(
            contract.MARKER_MAYBE
        )
    record = json.loads((tmp_path / "m.jsonl").read_text().strip())
    assert record["epoch_ms"] == clock.now      # raw press time, not t-20


def test_lines_are_flushed_as_they_are_written(tmp_path):
    """A crash mid-stream should cost the presses that never happened, not the
    ones already made."""
    path = tmp_path / "m.jsonl"
    sink = contract.JsonlSink(path)
    daemon = marker_daemon.MarkerDaemon(sink=sink, clock=FakeClock(), log=lambda *_: None)
    daemon.press(contract.MARKER_MAYBE)
    assert path.read_text().count("\n") == 1     # readable before close
    sink.close()


def test_a_restarted_daemon_appends_rather_than_truncating(tmp_path):
    path = tmp_path / "m.jsonl"
    clock = FakeClock()
    for _ in range(2):
        with contract.JsonlSink(path) as sink:
            marker_daemon.MarkerDaemon(sink=sink, clock=clock, log=lambda *_: None).press(
                contract.MARKER_MAYBE
            )
        clock.advance(60)
    assert len(path.read_text().strip().splitlines()) == 2


def test_key_repeat_is_debounced(tmp_path):
    clock = FakeClock()
    with contract.JsonlSink(tmp_path / "m.jsonl") as sink:
        daemon = marker_daemon.MarkerDaemon(
            sink=sink, clock=clock, debounce_s=0.35, log=lambda *_: None
        )
        assert daemon.press(contract.MARKER_MAYBE) is True
        clock.advance(0.1)
        assert daemon.press(contract.MARKER_MAYBE) is False
        clock.advance(1.0)
        assert daemon.press(contract.MARKER_MAYBE) is True

    assert len((tmp_path / "m.jsonl").read_text().strip().splitlines()) == 2


def test_the_two_kinds_debounce_independently(tmp_path):
    """F1 then F2 in quick succession is a deliberate escalation, not a repeat."""
    clock = FakeClock()
    with contract.JsonlSink(tmp_path / "m.jsonl") as sink:
        daemon = marker_daemon.MarkerDaemon(sink=sink, clock=clock, log=lambda *_: None)
        assert daemon.press(contract.MARKER_MAYBE) is True
        assert daemon.press(contract.MARKER_DEFINITE) is True


def test_an_unknown_marker_kind_is_refused(tmp_path):
    with pytest.raises(ValueError):
        contract.marker_record(1_800_000_000_000, "marker_probably")


def test_hotkeys_can_be_rebound():
    """F1 is help or ping in most games, including Marvel Rivals."""
    assert marker_daemon.parse_bindings(None) == marker_daemon.DEFAULT_BINDINGS
    rebound = marker_daemon.parse_bindings(["f9=marker_definite"])
    assert rebound["f9"] == contract.MARKER_DEFINITE
    assert "f2" not in rebound              # the old binding for that kind is gone
    assert rebound["f1"] == contract.MARKER_MAYBE


def test_a_bad_binding_is_refused():
    with pytest.raises(SystemExit):
        marker_daemon.parse_bindings(["f9=marker_probably"])


def test_the_marker_log_is_one_file_per_day(tmp_path):
    """The daemon cannot know where OBS is writing, and §4.5 forbids making it
    depend on OBS to find out."""
    path = marker_daemon.markers_path(tmp_path, date(2026, 8, 14))
    assert path.name == "markers-2026-08-14.jsonl"


# --------------------------------------------------------------------------
# input.jsonl (§4.4) — rates, never keystrokes
# --------------------------------------------------------------------------


def test_the_aggregator_cannot_be_told_which_key_was_pressed():
    """THE privacy guarantee, and it is structural: a global hook sees every
    password typed into every window, so `key()` takes no arguments and no
    function in the module can be handed a key identity."""
    import inspect

    signature = inspect.signature(input_logger.Aggregator.key)
    assert list(signature.parameters) == ["self"]


def test_no_key_identity_appears_in_a_record(tmp_path):
    clock = FakeClock()
    elapsed = iter([0.0, 1.0])
    with contract.JsonlSink(tmp_path / "i.jsonl") as sink:
        agg = input_logger.Aggregator(
            sink=sink, hz=1.0, epoch=clock, elapsed=lambda: next(elapsed)
        )
        agg.tick()
        agg.key()
        agg.key()
        agg.tick()

    record = json.loads((tmp_path / "i.jsonl").read_text().strip())
    assert set(record) == {"epoch_ms", "keys_per_s", "mouse_vel_px_s", "clicks_per_s"}


def test_rates_are_per_second_over_the_window(tmp_path):
    clock = FakeClock()
    elapsed = iter([0.0, 2.0])
    with contract.JsonlSink(tmp_path / "i.jsonl") as sink:
        agg = input_logger.Aggregator(
            sink=sink, hz=0.5, epoch=clock, elapsed=lambda: next(elapsed)
        )
        agg.tick()
        for _ in range(8):
            agg.key()
        for _ in range(2):
            agg.click()
        agg.tick()

    record = json.loads((tmp_path / "i.jsonl").read_text().strip())
    assert record["keys_per_s"] == 4.0          # 8 keys over 2 s
    assert record["clicks_per_s"] == 1.0


def test_mouse_travel_is_distance_not_position(tmp_path):
    """§5.4.1 wants px/s; sampling position at 10 Hz would miss the fast part of
    a flick entirely, which is the whole point of the signal."""
    elapsed = iter([0.0, 1.0])
    with contract.JsonlSink(tmp_path / "i.jsonl") as sink:
        agg = input_logger.Aggregator(
            sink=sink, hz=1.0, epoch=FakeClock(), elapsed=lambda: next(elapsed)
        )
        agg.tick()
        agg.move(0, 0)
        agg.move(3, 4)          # 5 px
        agg.move(3, 4)          # 0 px
        agg.move(0, 0)          # 5 px back — travel, not displacement
        agg.tick()

    assert json.loads((tmp_path / "i.jsonl").read_text().strip())["mouse_vel_px_s"] == 10.0


def test_no_record_is_emitted_before_a_window_elapses(tmp_path):
    elapsed = iter([0.0, 0.01, 0.02])
    with contract.JsonlSink(tmp_path / "i.jsonl") as sink:
        agg = input_logger.Aggregator(
            sink=sink, hz=10.0, epoch=FakeClock(), elapsed=lambda: next(elapsed)
        )
        assert agg.tick() is False
        assert agg.tick() is False
        assert agg.tick() is False
    assert (tmp_path / "i.jsonl").read_text() == ""


def test_counters_reset_between_windows(tmp_path):
    elapsed = iter([0.0, 1.0, 2.0])
    with contract.JsonlSink(tmp_path / "i.jsonl") as sink:
        agg = input_logger.Aggregator(
            sink=sink, hz=1.0, epoch=FakeClock(), elapsed=lambda: next(elapsed)
        )
        agg.tick()
        agg.key()
        agg.tick()
        agg.tick()

    rates = [json.loads(line)["keys_per_s"]
             for line in (tmp_path / "i.jsonl").read_text().strip().splitlines()]
    assert rates == [1.0, 0.0]


def test_the_window_clock_is_monotonic_not_wall_clock():
    """An NTP step mid-stream would otherwise stretch or collapse a bucket and
    put a spurious spike in the one signal meant to detect spikes."""
    import time as time_module

    assert input_logger.Aggregator.__dataclass_fields__["elapsed"].default \
        is time_module.monotonic
    assert input_logger.Aggregator.__dataclass_fields__["epoch"].default \
        is contract.now_epoch_ms


def test_the_default_rate_is_the_spec_rate():
    """§4.4: "Writes JSONL at 10 Hz aggregate"."""
    assert contract.INPUT_HZ == 10.0
