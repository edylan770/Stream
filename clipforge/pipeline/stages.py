"""The §5.1 stage registry.

All sixteen stages are declared, including the ones no phase has built yet.
Declaring them costs nothing and makes `clipforge status` a readable map of the
whole pipeline rather than a list of the four things that happen to exist.

**Order versus dependencies.** §5.1 presents the stages as a numbered list, but
they are a DAG: `marker_events` is tenth in the listing and depends only on
registration, so it does not need the nine stages above it. The dependencies
here are the real ones. Display order follows the spec's numbering so the two
documents can be read side by side.

**Behaviour lives in the stage module, not here.** Each implemented stage names
a module that provides `run(ctx)` and optionally `outputs(ctx)`, `verify(ctx)`,
and `params(ctx)`. Resolving those lazily keeps this registry importable
without pulling in numpy, and avoids a cycle between the runner and the stages
that use it.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class StageModule(Protocol):
    def run(self, ctx: Any) -> None: ...


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage."""

    name: str
    summary: str
    #: Build phase from §15 that implements it. Phase 1 stages with no module
    #: yet are simply not written; anything above 1 is out of scope by design.
    phase: int
    requires: tuple[str, ...] = ()
    #: Dotted module path providing run/outputs/verify/params. None = not built.
    module: str | None = None
    #: Completed by a command rather than by `run`. See register_stream.
    external: bool = False
    external_hint: str = ""

    @property
    def implemented(self) -> bool:
        return self.module is not None or self.external

    def load(self) -> StageModule:
        if self.module is None:
            raise RuntimeError(f"stage {self.name} has no module")
        return importlib.import_module(self.module)  # type: ignore[return-value]

    # -- optional module hooks, with defaults ------------------------------

    def outputs(self, ctx: Any) -> list[Path]:
        hook = getattr(self.load(), "outputs", None)
        return list(hook(ctx)) if hook else []

    def params(self, ctx: Any) -> dict:
        hook = getattr(self.load(), "params", None)
        return dict(hook(ctx)) if hook else {}

    def verify(self, ctx: Any) -> tuple[bool, str]:
        """True when the stage's work is genuinely present.

        Defaults to "every declared output exists". A stage whose product is
        database rows rather than files overrides this — otherwise deleting the
        row leaves the stage looking permanently done.
        """
        hook = getattr(self.load(), "verify", None)
        if hook is not None:
            return hook(ctx)
        for path in self.outputs(ctx):
            if not path.exists():
                return False, f"missing output {path.name}"
        return True, ""


#: §5.1, in the spec's order. `phase` is the §15 build phase.
STAGE_LIST: list[StageSpec] = [
    StageSpec(
        name="register_stream",
        summary="streams row, anchor, file paths",
        phase=1,
        external=True,
        external_hint="clipforge register --master <file>",
    ),
    # module= is filled in by the commit that implements each stage. Declaring
    # them here first means `status` shows the whole §5.1 map from day one.
    StageSpec(
        name="probe",
        summary="ffprobe: duration, fps, resolution, track map",
        phase=1,
        requires=("register_stream",),
        module="clipforge.ingest.probe",
    ),
    StageSpec(
        name="proxy",
        summary="generate proxy",
        phase=1,
        requires=("probe",),
    ),
    StageSpec(
        name="audio_split",
        summary="extract per-track WAV",
        phase=1,
        requires=("probe",),
    ),
    StageSpec(
        name="audio_features",
        summary="RMS, F0, silence, laughter (per track)",
        phase=1,
        requires=("audio_split",),
    ),
    StageSpec(
        name="whisperx",
        summary="transcript + word timestamps",
        phase=2,
        requires=("audio_split",),
    ),
    StageSpec(
        name="speaker_assign",
        summary="deterministic, via track energy",
        phase=2,
        requires=("whisperx", "audio_features"),
    ),
    StageSpec(
        name="phrase_detect",
        summary="passive voice triggers, repeated phrases",
        phase=2,
        requires=("whisperx",),
    ),
    StageSpec(
        name="input_signals",
        summary="parse input JSONL -> signal_series",
        phase=3,
        requires=("register_stream",),
    ),
    StageSpec(
        name="marker_events",
        summary="parse marker JSONL -> events",
        phase=1,
        requires=("register_stream",),
    ),
    StageSpec(
        name="scene_events",
        summary="parse OBS log -> events",
        phase=3,
        requires=("register_stream",),
    ),
    StageSpec(
        name="vision",
        summary="OCR/template match (deferred, cuttable)",
        phase=7,
        requires=("proxy",),
    ),
    StageSpec(
        name="embeddings",
        summary="per-segment vectors",
        phase=2,
        requires=("whisperx",),
    ),
    StageSpec(
        name="score",
        summary="candidates (cheap, rerunnable)",
        phase=1,
        requires=("audio_features", "marker_events"),
    ),
    StageSpec(
        name="previews",
        summary="2s webm, thumbstrip, waveform per candidate",
        phase=3,
        requires=("score", "proxy"),
    ),
    StageSpec(
        name="digest",
        summary="LLM digest generation",
        phase=5,
        requires=("score", "whisperx"),
    ),
]

STAGES: dict[str, StageSpec] = {spec.name: spec for spec in STAGE_LIST}

#: Position in §5.1's numbered list, used to break ties in the topological sort
#: so output order matches the document.
SPEC_ORDER: dict[str, int] = {spec.name: index for index, spec in enumerate(STAGE_LIST)}


class UnknownStage(KeyError):
    pass


def get(name: str) -> StageSpec:
    try:
        return STAGES[name]
    except KeyError:
        raise UnknownStage(
            f"unknown stage {name!r}. Known: {', '.join(STAGES)}"
        ) from None


def validate_graph() -> None:
    """Every dependency exists and the graph is acyclic."""
    for spec in STAGE_LIST:
        for dep in spec.requires:
            if dep not in STAGES:
                raise ValueError(f"stage {spec.name} requires unknown stage {dep!r}")
    resolve_order(list(STAGES))  # raises on a cycle


def resolve_order(names: list[str]) -> list[str]:
    """Topological order over `names`, ties broken by §5.1 position.

    Kahn's algorithm with a deterministic tie-break, so `run` and `status`
    always list stages in the same order and that order matches the spec.
    """
    wanted = set(names)
    indegree = {
        name: sum(1 for dep in STAGES[name].requires if dep in wanted) for name in wanted
    }
    dependents: dict[str, list[str]] = {name: [] for name in wanted}
    for name in wanted:
        for dep in STAGES[name].requires:
            if dep in wanted:
                dependents[dep].append(name)

    ready = sorted((n for n in wanted if indegree[n] == 0), key=SPEC_ORDER.__getitem__)
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for child in dependents[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=SPEC_ORDER.__getitem__)

    if len(ordered) != len(wanted):
        remaining = sorted(wanted - set(ordered))
        raise ValueError(f"cycle in stage dependencies involving: {remaining}")
    return ordered


def ancestors(name: str) -> set[str]:
    """Everything `name` transitively depends on."""
    seen: set[str] = set()
    stack = list(get(name).requires)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(STAGES[current].requires)
    return seen


def descendants(name: str) -> set[str]:
    """Everything that transitively depends on `name`."""
    seen: set[str] = set()
    stack = [name]
    while stack:
        current = stack.pop()
        for spec in STAGE_LIST:
            if current in spec.requires and spec.name not in seen:
                seen.add(spec.name)
                stack.append(spec.name)
    return seen


def default_plan() -> list[str]:
    """Every stage, in dependency order. Unimplemented ones are reported, not run."""
    return resolve_order(list(STAGES))


def iter_specs() -> Iterator[StageSpec]:
    return iter(STAGE_LIST)
