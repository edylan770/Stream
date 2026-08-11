"""The §5.1 stage graph."""

from __future__ import annotations

import pytest

from clipforge.pipeline import stages

#: §5.1's numbered list, verbatim and in order. If the registry drifts from the
#: spec, this is what says so.
SPEC_5_1_ORDER = [
    "register_stream", "probe", "proxy", "audio_split", "audio_features",
    "whisperx", "speaker_assign", "phrase_detect", "input_signals",
    "marker_events", "scene_events", "vision", "embeddings", "score",
    "previews", "digest",
]


def test_every_spec_stage_is_registered():
    assert [s.name for s in stages.iter_specs()] == SPEC_5_1_ORDER


def test_graph_is_valid():
    stages.validate_graph()


def test_phase_1_stages_match_the_build_order():
    """§15 Phase 1: register, probe, proxy, audio split, RMS, markers, score.
    Export and the review UI are commands, not extraction stages."""
    phase_1 = {s.name for s in stages.iter_specs() if s.phase == 1}
    assert phase_1 == {
        "register_stream", "probe", "proxy", "audio_split",
        "audio_features", "marker_events", "score",
    }


def test_dependency_order_is_topological():
    order = stages.default_plan()
    position = {name: index for index, name in enumerate(order)}
    for spec in stages.iter_specs():
        for dep in spec.requires:
            assert position[dep] < position[spec.name], f"{spec.name} before {dep}"


def test_order_is_deterministic():
    assert stages.default_plan() == stages.default_plan()


def test_marker_events_does_not_wait_on_the_transcript():
    """§5.1 lists marker_events tenth, but it parses a JSONL file and needs
    nothing but a registered stream. Treating the list as a chain rather than a
    DAG would make markers wait on WhisperX."""
    assert stages.ancestors("marker_events") == {"register_stream"}


def test_score_depends_on_both_phase_1_signals():
    assert {"audio_features", "marker_events"} <= stages.ancestors("score")


def test_descendants_of_probe_include_everything_derived_from_it():
    downstream = stages.descendants("probe")
    assert {"proxy", "audio_split", "audio_features", "score"} <= downstream
    assert "register_stream" not in downstream


def test_ancestors_are_transitive():
    assert "probe" in stages.ancestors("audio_features")
    assert "register_stream" in stages.ancestors("audio_features")


def test_unknown_stage_lists_the_known_ones():
    with pytest.raises(stages.UnknownStage, match="probe"):
        stages.get("nonsense")


def test_resolve_order_detects_a_cycle(monkeypatch):
    looped = dict(stages.STAGES)
    looped["probe"] = stages.StageSpec(
        name="probe", summary="", phase=1, requires=("proxy",)
    )
    monkeypatch.setattr(stages, "STAGES", looped)
    with pytest.raises(ValueError, match="cycle"):
        stages.resolve_order(["probe", "proxy"])


def test_register_stream_is_external_not_runnable():
    """It needs the master path before a streams row exists to read it from."""
    spec = stages.get("register_stream")
    assert spec.external
    assert "clipforge register" in spec.external_hint


def test_unbuilt_stages_declare_their_phase():
    for spec in stages.iter_specs():
        if not spec.implemented:
            assert spec.phase >= 1
            assert spec.summary
